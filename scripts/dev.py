"""LSSの開発検証入口。実GPU・配備版は明示したsuiteだけで実行する。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
import tempfile
import time
from uuid import uuid4
import warnings
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
# 既存試験の明示実行スイッチを所有する。通常検証へ環境変数を持ち越さない。
ACCEPTANCE = {
    "gpu": ("test_separation_gpu.py", "LSS_RUN_GPU_TEST"),
    "backend-acceptance": ("test_backend_acceptance_gpu.py", "LSS_RUN_BACKEND_ACCEPTANCE"),
    "acquisition-gpu": ("test_acquisition_gpu.py", "LSS_RUN_ACQUISITION_GPU"),
    "bgm-episode": ("test_bgm_episode_gpu.py", "LSS_RUN_BGM_EPISODE_ACCEPTANCE"),
    "bgm-parity": ("test_bgm_cue_episode_parity.py", "LSS_RUN_BGM_CUE_EPISODE_PARITY"),
    "packaged": ("test_packaged_worker.py", "LSS_RUN_PACKAGED_TEST"),
    "deployed": ("test_deployed_backend_acceptance.py", "LSS_RUN_DEPLOYED_ACCEPTANCE"),
    "ui-acceptance": ("test_ui_deployed_acceptance.py", "LSS_RUN_UI_ACCEPTANCE"),
}
NATIVE = {
    "ui-runtime": "tests/test_ui_lifecycle.py",
    "login-runtime": (
        "tests/test_acquisition_login_profile.py::"
        "test_cookies_from_the_lss_webview2_profile_authenticate_the_real_acquisition_tool"
    ),
}
SUITES = ("local", "backend", "bgm", *NATIVE, *ACCEPTANCE)


def selected_tests(suite: str, paths: list[str], root: Path = ROOT) -> list[str]:
    if suite in NATIVE:
        if paths and paths != [NATIVE[suite]]:
            raise ValueError(f"suite={suite}の対象外です: {paths}")
        return [NATIVE[suite]]
    if suite in ACCEPTANCE:
        names = [ACCEPTANCE[suite][0]]
    else:
        excluded = {item[0] for item in ACCEPTANCE.values()} | {"test_ui_lifecycle.py"}
        names = sorted(p.name for p in (root / "tests").glob("test_*.py") if p.name not in excluded)
        if suite == "backend":
            names = [n for n in names if not n.startswith("test_ui_")]
        elif suite == "bgm":
            # 固定ケースの測定・保存・再開境界もBGMの変更範囲に含める。
            names = [n for n in names if any(word in n for word in (
                "bgm", "episode", "activity", "stereo", "alignment", "occurrence",
                "controlled_mix", "bandit", "provider_padding", "song_component",
            ))]
    allowed = {f"tests/{name}" for name in names}
    if paths:
        for value in paths:
            file = value.split("::", 1)[0].replace("\\", "/").removeprefix("./")
            if file not in allowed or value.replace("\\", "/").removeprefix("./") in NATIVE.values():
                raise ValueError(f"suite={suite}の対象外です: {value}")
        return paths
    if not allowed:
        raise ValueError("検証対象がありません")
    return sorted(allowed)


# --- related-test selection (--changed) -------------------------------------------------
# A change that cannot be traced through imports or path strings selects the whole suite.
WHOLE_SUITE_FILES = {"pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "scripts/dev.py"}
SOURCE_ROOTS = ("lss", "scripts", "tests")
MERGE_BASE_REFS = ("main", "origin/main")
NOT_RUN = 3  # exit code: --changed selected no test file, so nothing was verified


def is_whole_suite_trigger(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (path in WHOLE_SUITE_FILES or name == "conftest.py"
            or name == "constraints.txt" or (name.startswith("requirements") and name.endswith(".txt")))


def module_name(path: str) -> str:
    """lss/a/b.py -> lss.a.b, lss/a/__init__.py -> lss.a."""
    parts = path.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def package_of(path: str) -> str:
    """The package that relative imports of this file start from."""
    parts = path.removesuffix(".py").split("/")
    return ".".join(parts[:-1])


def _docstring_nodes(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


SCAN_CALLS = {"rglob", "glob", "iglob", "walk", "iterdir"}


def source_facts(path: str, text: str) -> tuple[set[str], list[str], set[str]]:
    """Import candidates (dotted names with every parent package), string constants and scanned directories.

    A test file that walks a directory tree (rglob/glob/os.walk) depends on every file below the top-level
    directories its string constants name (a structural test over `lss/`).

    `from lss.x import y` yields lss, lss.x and lss.x.y because y may be a module or an attribute.
    A bare `import gpu_run` also yields the scripts./tests./sibling spellings (scripts put on sys.path).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # old escape sequences in other files are not this tool's business
            tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set(), [], set()
    package = package_of(path)
    names: set[str] = set()

    def add(name: str):
        parts = name.split(".") if name else []
        for end in range(1, len(parts) + 1):
            prefix = ".".join(parts[:end])
            names.add(prefix)
            if parts[0] not in SOURCE_ROOTS:
                names.update((f"scripts.{prefix}", f"tests.{prefix}", f"{package}.{prefix}" if package else prefix))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[:len(package.split(".")) - (node.level - 1)] if package else []
                module = ".".join([*base, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            add(module)
            for alias in node.names:
                if alias.name != "*":
                    add(f"{module}.{alias.name}" if module else alias.name)
    # Importing a module runs its parent packages' __init__ first.
    own = module_name(path).split(".")
    names.update(".".join(own[:end]) for end in range(1, len(own)))
    skip = _docstring_nodes(tree)
    strings = [node.value.replace("\\", "/") for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip]
    scans = {c.split("/", 1)[0] for c in strings if c} if path.startswith("tests/") and any(
        isinstance(node, ast.Attribute) and node.attr in SCAN_CALLS for node in ast.walk(tree)) else set()
    return names, strings, scans


def select_related(changed: list[str], sources: dict[str, str], tests: list[str]) -> dict[str, str]:
    """Test files affected by the changed paths: {test file: reason}. Deterministic, no execution.

    Affected = the changed test file itself, a file that transitively imports a changed module, or a file
    with a string constant naming a changed path (basename) or module (dotted name), or one that walks a
    directory tree it names. Whole suite for conftest.py, pytest.ini, requirements, constraints and this script.
    """
    changed = sorted({c.replace("\\", "/").removeprefix("./") for c in changed})
    trigger = next((c for c in changed if is_whole_suite_trigger(c)), None)
    if trigger:
        return {test: f"whole suite ({trigger} changed)" for test in tests}
    facts = {path: source_facts(path, text) for path, text in sources.items()}
    by_name = {module_name(path): path for path in facts}
    changed_modules = {module_name(c): c for c in changed if c.endswith(".py")}
    patterns = [(re.compile(r"(?<![\w.\-])" + re.escape(c.rsplit("/", 1)[-1]) + r"(?!\w)"), c)
                for c in changed if not c.endswith("/__init__.py")]
    reason = {c: "changed" for c in changed if c in facts}
    dependents: dict[str, set[str]] = {}
    for path, (names, strings, scans) in facts.items():
        for name in names:
            if name in by_name and by_name[name] != path:
                dependents.setdefault(by_name[name], set()).add(path)
            if name in changed_modules and path not in reason:
                reason[path] = f"imports {changed_modules[name]}"
        if path in reason:
            continue
        folder = next((c for c in changed if c.split("/", 1)[0] in scans and "/" in c), None)
        if folder:
            reason[path] = f"scans {folder.split('/', 1)[0]}/"
            continue
        for const in strings:
            hit = next((c for pattern, c in patterns if pattern.search(const)), None)
            if hit is None:
                hit = next((c for m, c in changed_modules.items() if const == m or const.startswith(m + ".")), None)
            if hit:
                reason[path] = f"references {hit}"
                break
    queue = sorted(reason)
    while queue:
        current = queue.pop()
        origin = reason[current].split(" ", 1)[1] if reason[current] != "changed" else current
        for dependent in sorted(dependents.get(current, ())):
            if dependent not in reason:
                reason[dependent] = f"imports {origin}"
                queue.append(dependent)
    return {test: reason[test] for test in tests if test in reason}


def read_sources(root: Path = ROOT) -> dict[str, str]:
    listed = git_bytes("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *SOURCE_ROOTS)
    sources = {}
    for raw in sorted(set(listed.split(b"\0"))):
        name = os.fsdecode(raw)
        if name.endswith(".py") and (root / name).is_file():
            sources[name] = (root / name).read_text(encoding="utf-8", errors="replace")
    return sources


def default_base() -> str:
    for ref in MERGE_BASE_REFS:
        result = subprocess.run(["git", "merge-base", "HEAD", ref], cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return "HEAD"


def changed_paths(base: str) -> list[str]:
    """Files changed since `base` (committed, uncommitted and untracked), repo-relative with '/'."""
    tracked = git_bytes("diff", "--name-only", "--no-renames", "-z", base, "--")
    untracked = git_bytes("ls-files", "--others", "--exclude-standard", "-z")
    return sorted({os.fsdecode(name) for name in (*tracked.split(b"\0"), *untracked.split(b"\0")) if name})


def select_changed(suite: str, base: str, root: Path = ROOT) -> dict:
    fallback = not base
    base = base or default_base()
    fallback = fallback and base == "HEAD"
    changed = changed_paths(base)
    suite_files = selected_tests(suite, [], root)
    # No graph can affect an empty diff or an explicit whole-suite trigger.
    # Document/path-string dependencies still use the full existing graph.
    sources = {} if not changed or any(is_whole_suite_trigger(path) for path in changed) else read_sources(root)
    related = select_related(changed, sources, suite_files)
    return {"base": base, "base_fallback": fallback, "changed": changed, "suite_files": len(suite_files),
            "selected": sorted(related), "reasons": related}


def describe_selection(selection: dict, top: int = 5) -> list[str]:
    causes = Counter(re.sub(r"^(imports|references|scans) ", "", why) if why != "changed" else "(test file itself changed)"
                     for why in selection["reasons"].values())
    lines = ["WARNING: neither main nor origin/main resolves; base fell back to HEAD, so only uncommitted and untracked changes count"] if selection.get("base_fallback") else []
    lines += [f"changed since {selection['base'][:10]}: {len(selection['changed'])} files; "
             f"selected {len(selection['selected'])} of {selection['suite_files']} test files"]
    lines += [f"  {count:4d} via {cause}" for cause, count in causes.most_common(top)]
    if len(causes) > top:
        lines.append(f"  ... {len(causes) - top} more causes")
    return lines


# --- run capture and failure digest ------------------------------------------------------
DIGEST_LINES = 30
DIGEST_WIDTH = 200
PROGRESS = re.compile(r"^[.FEsxX]+\s+\[\s*(\d+)%\]\s*$")


def junit_nodeid(classname: str, name: str, root: Path = ROOT) -> str:
    parts = [p for p in classname.split(".") if p]
    for end in range(len(parts), 0, -1):
        module = "/".join(parts[:end]) + ".py"
        if (root / module).is_file():
            return "::".join([module, *parts[end:], name])
    for end, part in enumerate(parts, 1):
        if part.startswith("test_"):
            return "::".join(["/".join(parts[:end]) + ".py", *parts[end:], name])
    return "::".join([*parts, name])


def first_error_line(element: ET.Element, width: int = DIGEST_WIDTH) -> str:
    """The first `E   ...` line of a short traceback, else the exception message."""
    errors = [line[1:].strip() for line in (element.text or "").splitlines() if line.startswith("E ")]
    errors = [line for line in errors if line]
    text = errors[0] if errors else ""
    if text and re.fullmatch(r"[\w.]*(Error|Exception|Failed|Exit)", text) and len(errors) > 1:
        text = f"{text}: {errors[1]}"
    if not text:
        lines = [line.strip() for line in ((element.get("message") or "") or (element.text or "")).splitlines() if line.strip()]
        text = lines[0] if lines else (element.get("type") or "failed without a message")
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 3] + "..."


def failure_digest(path: Path, root: Path = ROOT, limit: int = DIGEST_LINES) -> dict:
    """Failed and errored tests of a junit XML as `KIND nodeid :: first error line`, capped at `limit`."""
    entries = []
    for case in ET.parse(path).getroot().iter("testcase"):
        for kind in ("failure", "error"):
            element = case.find(kind)
            if element is not None:
                nodeid = junit_nodeid(case.get("classname", ""), case.get("name", ""), root)
                entries.append(f"{kind.upper()} {nodeid} :: {first_error_line(element)}")
                break
    return {"total": len(entries), "lines": entries[:limit], "omitted": max(0, len(entries) - limit)}


def console(text: str):
    """Print without failing on characters the console encoding cannot show."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding), flush=True)


def run_pytest(command: list[str], environment: dict[str, str], log_path: Path, stream: bool) -> int:
    """Run pytest and keep its whole console output in `log_path`; the console shows it only with `stream`.

    Without `stream` the console gets one heartbeat per ten percent instead of the progress dots.
    """
    shown, bad = -1, 0
    with log_path.open("w", encoding="utf-8", newline="\n") as log, subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace") as process:
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if stream:
                    console(line.rstrip("\n"))
                    continue
                match = PROGRESS.match(line.rstrip())
                if match:
                    bad += sum(line.count(c) for c in "FE")
                    if int(match.group(1)) // 10 > shown:
                        shown = int(match.group(1)) // 10
                        console(f"  {match.group(1)}%" + (f" ({bad} failed/error so far)" if bad else ""))
            return process.wait()
        except BaseException:
            process.kill()
            raise


def digest_report(status: str, counts: dict, seconds: float, xml: Path, log_path: Path, report_path: Path) -> dict:
    digest = failure_digest(xml) if xml.is_file() else {"total": 0, "lines": [], "omitted": 0}
    if not digest["lines"] and status == "FAILED" and log_path.is_file():
        # No failed test case to name (usage error, crash, collection problem): show where the log ends.
        digest["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
    digest.update(verdict=status, counts=counts, wall_seconds=round(seconds, 1),
                  log=str(log_path), report=str(report_path))
    return digest


def format_digest(digest: dict) -> str:
    counts = digest["counts"]
    lines = [f"verdict: {digest['verdict']} / tests {counts.get('tests', 0)} failed {counts.get('failures', 0)} "
             f"errors {counts.get('errors', 0)} skipped {counts.get('skipped', 0)} / {digest['wall_seconds']}s",
             *digest["lines"]]
    if digest["omitted"]:
        lines.append(f"... {digest['omitted']} more failures in the log")
    lines += [f"  {line[:DIGEST_WIDTH]}" for line in digest.get("log_tail", [])]
    lines += [f"log: {digest['log']}", f"report: {digest['report']}"]
    return "\n".join(lines)


def test_environment(suite: str, state: Path) -> dict[str, str]:
    environment = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith("LSS_RUN_")
                   and k.upper() not in {"LSS_UI_ACCEPTANCE_DOWNLOAD", "PYTEST_ADDOPTS", "PSMODULEPATH"}}
    # pwshのPSModulePathをPython経由でWindows PowerShellへ渡すと、標準の
    # Security moduleが二重ロードになり署名照合まで失敗する。子側の標準値で起動する。
    # 参照録音の置き場所も試験ごとの空の場所にする（開発ツリーの `References` を拾わない）。
    environment.update(LSS_STATE_DIR=str(state), LSS_REFERENCES_DIR=str(state / "References"), PYTHONUTF8="1")
    if suite in ACCEPTANCE:
        environment[ACCEPTANCE[suite][1]] = "1"
    if suite in ACCEPTANCE or suite in NATIVE:
        environment["LSS_TEST_EXTERNAL"] = "1"  # real models and network; ordinary suites stay isolated (conftest)
    else:
        environment.pop("LSS_TEST_EXTERNAL", None)
    return environment


def git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def identity() -> dict:
    # 未追跡の実装も含める。秘密値やdiff本文を証拠へ出力しない。
    digest = hashlib.sha256(git_bytes("diff", "HEAD", "--binary"))
    for name in sorted(git_bytes("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")):
        if name:
            path = ROOT / os.fsdecode(name)
            digest.update(name)
            if path.is_file():
                with path.open("rb") as stream:
                    digest.update(hashlib.file_digest(stream, "sha256").digest())
    return {"head": git_bytes("rev-parse", "HEAD").decode().strip(),
            "changes_sha256": digest.hexdigest(),
            "status": git_bytes("status", "--porcelain").decode("utf-8", errors="replace")}


def junit_counts(path: Path) -> dict[str, int]:
    suites = ET.parse(path).getroot()
    cases = list(suites.iter("testcase"))
    return {"tests": len(cases),
            "skipped": sum(case.find("skipped") is not None for case in cases),
            "failures": sum(case.find("failure") is not None for case in cases),
            "errors": sum(1 for _ in suites.iter("error"))}


def verdict(returncode: int, counts: dict, changed: bool, collect_only: bool) -> str:
    if returncode != 0 or counts.get("errors", 0) or counts.get("failures", 0):
        return "FAILED"
    if collect_only:
        return "COLLECTED_ONLY"
    if counts.get("tests", 0) <= counts.get("skipped", 0):
        return "NOT_VERIFIED"
    if changed:
        return "WORKTREE_CHANGED"
    return "PASSED_WITH_SKIPS" if counts["skipped"] else "PASSED"


def storage() -> dict:
    """検証成果物は放置すると積み上がる。空き容量を毎回見えるところへ出す。"""
    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / 2 ** 30
    return {"free_bytes": usage.free, "free_gb": round(free_gb, 1),
            "total_gb": round(usage.total / 2 ** 30, 1),
            "low_free_space": free_gb < 50,
            "cleanup_entry": ".venv/Scripts/python.exe scripts/disk_cleanup.py report"}


def repository_python(root: Path) -> Path | None:
    """Use a lane's environment when present, otherwise its Git common foundation."""
    local = root / ".venv/Scripts/python.exe"
    if local.is_file():
        return local
    result = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=root, capture_output=True, text=True, encoding="utf-8")
    if result.returncode or not result.stdout.strip():
        return None
    shared = Path(result.stdout.strip()).parent / ".venv/Scripts/python.exe"
    return shared if shared.is_file() else None


def doctor() -> int:
    packages = {}
    for name in ("pytest", "numpy", "soundfile", "torch", "pywebview", "yt-dlp", "pyinstaller"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    expected = repository_python(ROOT)
    result = {"python": sys.executable, "python_version": sys.version.split()[0],
              "repository_python": expected is not None and Path(sys.executable).samefile(expected),
              "expected_python": str(expected) if expected else None,
              "packages": packages, "git": identity(),
              "paths": {str(p): (ROOT / p).exists() for p in (
                  ".venv/Scripts/python.exe", ".providers/bandit/.venv/Scripts/python.exe",
                  "bin/ffmpeg.exe", "ledger/models.tsv", "dist/LSS/LSS.exe")},
              "suites": list(SUITES), "storage": storage(),
              "limitations": "GPU推論・モデル同一性・音楽品質・配備版の受入は未確認"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["repository_python"] and packages["pytest"] else 1


def worker_count(suite: str, paths: list[str], requested: str, collect_only: bool) -> int:
    """直列だと全体が30分超になる。実GPU・ネイティブ画面・配備版の試験は常に直列。"""
    if suite in NATIVE or suite in ACCEPTANCE or collect_only:
        return 1
    if requested != "auto":
        return max(1, int(requested))
    if len({p.split("::", 1)[0] for p in paths}) < 2 or importlib.util.find_spec("xdist") is None:
        return 1
    # GPU実験や他の作業者とCPU・メモリを分け合う。半分の論理コア、上限8、空きコミット3GBにつき1。
    # 試験単位の配分（worksteal）は重い試験が重なりページングファイル不足で失敗したため使わない。
    return max(1, min(8, (os.cpu_count() or 2) // 2, available_commit_bytes() // (3 * 2 ** 30)))


def available_commit_bytes() -> int:
    """Windowsの空きコミット量（物理メモリ＋ページングファイル）。取れなければ制限しない。"""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *((name, ctypes.c_ulonglong) for name in (
                            "total_phys", "avail_phys", "total_page", "avail_page",
                            "total_virtual", "avail_virtual", "avail_extended"))]

        status = Status(length=ctypes.sizeof(Status))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_page)
    except (AttributeError, OSError):
        pass
    return 2 ** 63


def dev_state_root(checkout_root):
    """Imported on use: `dev.py doctor` runs as a lone copy before the rest of `scripts/` exists."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dev_state import dev_state_root as resolve
    finally:
        sys.path.pop(0)
    return resolve(checkout_root)


def run_tests(args, *, selection=None) -> int:
    """Exit 0 passed, 1 failed, 3 (NOT_RUN) when --changed selected no test file (not a success)."""
    if args.changed is not None or args.changed_list:
        if args.paths:
            raise ValueError("--changed selects the files itself; do not pass paths")
        if args.suite in NATIVE or args.suite in ACCEPTANCE:
            raise ValueError(f"--changed does not apply to suite={args.suite}")
        selection = selection if selection is not None else select_changed(args.suite, args.changed or "")
        for line in describe_selection(selection):
            console(line)
        if args.changed_list:
            for name in selection["selected"]:
                console(name)
            return 0
        if not selection["selected"]:
            console(f"NOT_RUN: no test file is affected by the changes; nothing was run and there is no verdict (exit {NOT_RUN})")
            return NOT_RUN
        paths = selection["selected"]
    else:
        paths = selected_tests(args.suite, args.paths, ROOT)
    directory = dev_state_root(ROOT) / "validation/dev" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    import storage_budget
    with storage_budget.reserve(getattr(args, "storage_gb", None), paths=[str(ROOT), str(directory)],
                                archive_paths=[str(directory)], purpose=f"tests:{args.suite}") as reservation:
        return _run_tests_reserved(args, paths, selection, directory, reservation)


def collectable_paths(paths, markers, root: Path = ROOT):
    """Drop the files that cannot be imported without unpublished material when `local` is deselected.

    pytest does not ask `pytest_ignore_collect` about a path named on its command line, and this entry names every file.
    """
    if "notlocal" not in (markers or "").replace(" ", ""):
        return list(paths)
    sys.path.insert(0, str(root / "tests"))
    try:
        from local_tests import LOCAL_FILES
    finally:
        sys.path.pop(0)
    return [path for path in paths if path.replace("\\", "/").split("::", 1)[0] not in LOCAL_FILES]


def _run_tests_reserved(args, paths, selection, directory, reservation):
    paths = collectable_paths(paths, getattr(args, "markers", None))
    directory.mkdir(parents=True)
    xml = directory / "pytest.xml"
    log_path = directory / "pytest.log"
    command = [sys.executable, "-m", "pytest", "-q", "-ra", "--tb=short", *paths, f"--junitxml={xml}"]
    if args.suite not in NATIVE:
        command.extend(f"--deselect={node}" for node in NATIVE.values() if "::" in node)
    if args.keyword:
        command.extend(["-k", args.keyword])
    if getattr(args, "markers", None):
        command.extend(["-m", args.markers])
    if args.collect_only:
        command.append("--collect-only")
    workers = worker_count(args.suite, paths, args.workers, args.collect_only)
    if workers > 1:
        # ファイル単位で配る（module fixtureと同じファイル内の順序を保つ）。
        command.extend(["-n", str(workers), "--dist", "loadfile"])
    before = identity()
    report = {"schema_version": 1, "status": "RUNNING", "suite": args.suite, "workers": workers,
              "command": command, "python": sys.version, "before": before, "log": str(log_path),
              "claim": "選択したsuiteと対象の試験結果。音楽品質の昇格や別の実行物の受入は証明しない"}
    if selection:
        report["selection"] = {key: selection[key] for key in ("base", "base_fallback", "changed", "suite_files", "selected", "reasons")}
    report_path = directory / "report.json"

    def save():
        temporary = directory / "report.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)

    save()
    print(f"検証: {args.suite} / 記録: {report_path}", flush=True)
    started = time.monotonic()
    try:
        # A collect-only run is the listing itself, so it always streams.
        environment = test_environment(args.suite, directory / "state")
        environment.update({key: value for key, value in reservation.child_env().items()
                            if key in {"LSS_STORAGE_RESERVATION", "LSS_STORAGE_POLICY_JSON"}})
        returncode = run_pytest(command, environment, log_path,
                                args.stream or args.collect_only)
        seconds = time.monotonic() - started
        counts = junit_counts(xml) if xml.is_file() else {}
        after = identity()
        status = verdict(returncode, counts, before != after, args.collect_only)
        digest = digest_report(status, counts, seconds, xml, log_path, report_path)
        report.update(status=status, returncode=returncode, counts=counts, after=after, digest=digest)
    except BaseException as error:
        report.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR",
                      error_type=type(error).__name__)
        save()
        raise
    save()
    console(format_digest(digest))
    return 0 if status in {"PASSED", "PASSED_WITH_SKIPS", "COLLECTED_ONLY"} else 1


def disk_report(limit: int) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location("disk_cleanup", ROOT / "scripts/disk_cleanup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(storage(), ensure_ascii=False), flush=True)
    return module.report(limit)


def developer_memory_config(index_dir=None):
    """Use an explicit index or a validated workspace binding before profile defaults.

    App containers redirect LOCALAPPDATA; a workspace-owned derived cache keeps
    normal and managed queries consistent without touching the original index.
    """
    import devmem
    if index_dir is not None:
        return devmem.Config(db_dir=index_dir)
    binding = dev_state_root(ROOT) / 'devmem-index.json'
    if not binding.exists():
        return devmem.Config()
    if binding.stat().st_size > 16384:
        raise ValueError('Developer memory binding is too large')
    value = json.loads(binding.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or type(value.get('schema')) is not int
            or value['schema'] != 1 or not isinstance(value.get('repo'), str)
            or not isinstance(value.get('index_dir'), str)):
        raise ValueError('Invalid developer memory binding')
    expected = (ROOT.resolve() / '.cache/local-ai/devmem').resolve()
    directory = Path(value['index_dir']).resolve()
    if Path(value['repo']).resolve() != ROOT.resolve() or directory != expected:
        raise ValueError('Developer memory binding leaves its owned workspace cache')
    database = directory / 'devmem.sqlite'
    if database.resolve() != database or not database.is_file():
        raise ValueError('Bound developer memory is absent or redirected; reinitialize the workspace binding')
    from dev_hot_paths import require_hot
    require_hot(database, 'workspace developer memory')
    return devmem.Config(db_dir=directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="GPUを起動せず、環境と検証入口を表示")
    test = commands.add_parser("test", help="選択した既存pytestを実行し、対象と結果を記録")
    test.add_argument("paths", nargs="*", help="suite内のtests/test_x.pyやnode idへ限定")
    test.add_argument("--suite", choices=SUITES, default="local")
    test.add_argument("-k", dest="keyword")
    test.add_argument("-m", dest="markers", help="pytest marker expression")
    test.add_argument("--ci", action="store_true",
                      help="public-runner mode: no machine storage policy; not-local tests only")
    test.add_argument("--collect-only", action="store_true")
    test.add_argument("-j", "--workers", default="auto", type=lambda v: v if v == "auto" else str(int(v)),
                      help="並列プロセス数（auto=半分の論理コア・上限8、1=直列）。GPU/ネイティブ/配備版suiteは常に直列")
    test.add_argument("--stream", action="store_true",
                      help="print the full pytest console output (default: progress and a failure digest; the full log is pytest.log in the record)")
    test.add_argument("--storage-gb", type=float, help="estimated peak additional SSD space for this test run")
    test.add_argument("--changed", nargs="?", const="", default=None, metavar="BASE",
                      help="run only the test files affected by files changed since BASE (default: merge-base with main, plus uncommitted and untracked); "
                           f"exit {NOT_RUN} (NOT_RUN) when no test file is affected")
    test.add_argument("--changed-list", action="store_true", help="only print the --changed selection (exit 0)")
    disk = commands.add_parser("disk", help="作業物の大きさと参照を表示（削除はscripts/disk_cleanup.py）")
    disk.add_argument("--limit", type=int, default=25)
    draft = commands.add_parser("draft", help="run resumable local drafts; generated code is never applied")
    draft.add_argument("requests", help="JSONL queue of text messages")
    draft.add_argument("--model", default="ornith-1.0-9b")
    draft.add_argument("--out", required=True, help="subdirectory of .cache/local-ai or release-evidence")
    draft.add_argument("--profile", choices=["default", "test-vectors"], default="default")
    draft.add_argument("--cpu", action="store_true", help="run drafts on CPU without a GPU lease")
    memory = commands.add_parser("memory", help="search existing SSD developer memory; model-free by default")
    memory.add_argument("text")
    memory.add_argument("-k", type=int, default=8)
    memory.add_argument("--index-dir", type=Path, help="directory containing devmem.sqlite")
    memory.add_argument("--refresh", action="store_true", help="refresh sources within an 8-second budget")
    memory.add_argument("--json", action="store_true")
    memory.add_argument("--rerank", action="store_true", help="explicit Qwen reranking at10% GPU duty; existing SSD index only")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "disk":
            return disk_report(args.limit)
        if args.command == "test" and args.ci:
            # Hosted runners have no private machine volume policy or storage supervisor.
            # Keep selection, execution, state isolation and receipts in this entry point.
            if args.suite != "local" or args.markers != "not local" or args.changed is not None:
                parser.error("--ci requires --suite local -m 'not local' without --changed")
            from types import SimpleNamespace
            paths = selected_tests(args.suite, args.paths, ROOT)
            if not args.paths:
                paths += sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                                for p in (ROOT / "tests/gates").glob("test_*.py"))
            directory = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())) / ("lss-ci-" + uuid4().hex[:8])
            return _run_tests_reserved(args, paths, None, directory,
                                       SimpleNamespace(child_env=lambda: {}))
        from dev_hot_paths import require_hot, _git_common_dir
        for label, path in (("development checkout", ROOT), ("Git common directory", _git_common_dir(ROOT)),
                            ("Python environment", sys.prefix), ("Python base", sys.base_prefix),
                            ("test temporary files", tempfile.gettempdir()),
                            ("validation output", dev_state_root(ROOT) / "validation/dev")):
            require_hot(path, label)
        if args.command == "draft":
            if args.model in {"strata", "strata-coder"}:
                import strata_queue
                return strata_queue.main([args.requests, "--out", args.out, "--profile", args.profile]
                                         + (["--cpu"] if args.cpu else []))
            import local_ai_queue
            return local_ai_queue.main([args.requests, "--model", args.model, "--out", args.out,
                                        "--profile", args.profile] + (["--cpu"] if args.cpu else []))
        if args.command == "memory":
            import devmem
            cfg = developer_memory_config(args.index_dir)
            if args.rerank:
                import devmem_rerank
                return devmem_rerank.run(args, cfg)
            args.no_refresh, args.budget, args.mode = not args.refresh, 8.0, "bm25"
            return devmem.cmd_query(args, cfg)
        import storage_budget
        # A --changed run that selects nothing reports NOT_RUN without reserving or waiting for SSD space.
        selection = None
        supervised = bool(os.environ.get(storage_budget.TOKEN_ENV))
        if not supervised and args.changed is not None and not args.paths and args.suite not in NATIVE and args.suite not in ACCEPTANCE:
            selection = select_changed(args.suite, args.changed or "")
        nothing_selected = selection is not None and not selection['selected']
        if not os.environ.get(storage_budget.TOKEN_ENV) and not args.changed_list and not nothing_selected:
            managed = storage_budget.command([sys.executable, str(Path(__file__).resolve()),
                                              *(sys.argv[1:] if argv is None else argv)],
                                             # The run's record lives in the development state, which can be
                                             # outside this checkout; the nested reservation must be contained.
                                             paths=[str(ROOT), str(dev_state_root(ROOT) / "validation/dev")],
                                             required_gb=args.storage_gb, purpose=f"tests:{args.suite}")
            return subprocess.run(managed, **storage_budget.streams(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
        return run_tests(args, selection=selection)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
