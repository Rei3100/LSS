"""作業物の棚卸しと、参照されていない再生成可能な作業物だけの削除。

Development比較は巨大な音声と推論cacheを残す。記録（JSON）はGitにあるが、音声は
無いまま積み上がる。ここでは「何がどれだけ場所を取り、誰から参照されているか」を
先に示し、参照が一つも無く再生成できるものだけを、明示的な `--apply` で消す。

参照元として次を見る。
- Gitが追跡している `.state` の記録（固定plan・report・summary・現在状態）
- 完了台帳 `.state/completed_recipes.tsv` と移動記録 `relocations.json`
- 取得台帳と保存済みのBGM保留結果 `application-outcome.json`
- 進行中作業の覚書 `.ai/active.json`

モデルの重み・取得済みの配布物・Gitが追跡するファイルは、参照の有無に関わらず消さない。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root, require_owned_for_deletion
    from scripts.machine_config import StorageMutationRefused
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root, require_owned_for_deletion
    from machine_config import StorageMutationRefused

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _prunable_roots() -> tuple[Path, ...]:
    """Keep development evidence separate from legacy Product work paths."""
    return (dev_state_root(ROOT) / "validation", ROOT / ".providers", ROOT / ".state/work")


def _local_cleanup_path(path: Path) -> bool:
    from storage_tiers import is_link

    return not is_link(path) and str(path.resolve()) == str(path)


def _local_deletion_path(path: Path) -> bool:
    """Recheck the checkout and every component down to the deletion target."""
    from storage_tiers import is_link

    root_parts = ROOT.parts
    relative_parts = path.parts[len(root_parts):]
    if path.parts[:len(root_parts)] != root_parts or not relative_parts:
        return False
    parent = ROOT
    for part in relative_parts:
        if part == ".." or is_link(parent):
            return False
        parent /= part
    return _local_cleanup_path(path)


def _cleanup_receipt(prefix: str) -> Path:
    record = dev_state_root(ROOT) / "validation" / f"{prefix}-{int(time.time())}.json"
    if not _local_cleanup_path(record):
        raise StorageMutationRefused(f"Storage mutation refused: linked cleanup receipt path: {record}")
    return record


def _reported_path(path: Path) -> str:
    """Preserve legacy relative rows; use absolute rows for external state."""
    return (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix()


# 再取得が高価な配布物と重み。参照が無くても消さない。
PROTECTED_NAMES = frozenset({"downloads", "core", "configs", "runtime", "query-bandit", "banda"})
PROTECTED_SUFFIXES = frozenset({".ckpt", ".pth", ".pt", ".safetensors", ".bin", ".onnx",
                                # 外部から取得した配布物の書庫。作り直せない。
                                ".zip", ".7z", ".tar", ".gz", ".rar"})
# 作り直しに元素材が要るもの、実行時の凍結写し、画面の受入証跡は参照が無くても残す。
PROTECTED_PATTERNS = ("*-inputs", "*frozen*", "ui-*acceptance*")
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/][^\"'<>|?*\r\n]+")


def tracked_files(prefix: str = ".state") -> list[Path]:
    output = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", prefix], text=True)
    paths = [ROOT / line for line in output.splitlines() if line]
    state = dev_state_root(ROOT)
    parts = Path(prefix).parts
    if state != ROOT.resolve() / ".state" and (prefix == "." or (parts and parts[0] == ".state")):
        # Preserve Git protection in the state owner's checkout as well. If it
        # cannot be inspected, the caller must refuse deletion or report UNKNOWN.
        state_prefix = str(Path(*parts[1:])) if prefix != "." else "."
        output = subprocess.check_output(["git", "-C", str(state), "ls-files", state_prefix], text=True)
        paths.extend(state / line for line in output.splitlines() if line)
    return paths


def _normalize(text: str) -> Path | None:
    cleaned = text.replace("\\\\", "\\").rstrip(".,;:)\"' ")
    try:
        return Path(cleaned).resolve()
    except (OSError, ValueError):
        return None


def _paths_in(text: str) -> set[Path]:
    found = set()
    for match in _WINDOWS_PATH.finditer(text):
        resolved = _normalize(match.group(0))
        if resolved is not None:
            found.add(resolved)
    return found


def reference_sources() -> list[Path]:
    """参照元として読むファイル。読めないものは黙って飛ばさず、呼出側へ返す。

    Git管理下の記録だけでは、まだコミットしていない現在の固定planを見落とす。
    各比較の直下にあるJSON（plan/report/summary/受入記録）も参照元として読む。
    """
    sources = [path for path in tracked_files() if path.suffix in {".json", ".tsv", ".xml", ".py"}]
    for relative in (".state/completed_recipes.tsv", ".state/relocations.json",
                     ".state/queue.json", ".ai/active.json"):
        candidate = ROOT / relative
        if candidate.is_file():
            sources.append(candidate)
    validation = dev_state_root(ROOT) / "validation"
    if validation.is_dir():
        sources.extend(_json_under(validation))
    acquisitions = ROOT / ".state/acquisitions"
    if acquisitions.is_dir():
        sources.extend(_json_under(acquisitions))
    providers = ROOT / ".providers"
    if providers.is_dir():
        sources.extend(_json_under(providers))
    return sorted(set(sources))


def _json_under(root: Path) -> list[Path]:
    """JSON files under root. A junction whose target is gone (an earlier relocation) is skipped, not fatal."""
    found = []
    found.extend(path for path in local_files(root) if path.suffix == '.json')
    return sorted(found)


def local_files(root: Path):
    """Walk local real directories only: never follow a relocation junction,
    including broken ones. Other I/O errors abort collection, never authorize GC.
    """
    from storage_tiers import is_link
    if is_link(root):
        return
    def failed(error):
        raise error
    for directory, names, files in os.walk(root, onerror=failed):
        names[:] = [n for n in names if n not in {'.venv', 'venv', 'site-packages', '_internal',
                                                 'node_modules', '.git', '__pycache__'}
                    and not is_link(Path(directory) / n)]
        for name in files:
            path = Path(directory) / name
            if not is_link(path) and path.is_file():
                yield path


def referenced_paths() -> tuple[dict[Path, set[Path]], list[str]]:
    """参照されている所在 → それを指している記録の集合。自分自身の記録は後で除く。"""
    referenced: dict[Path, set[Path]] = {}
    unreadable = []
    for source in reference_sources():
        try:
            text = source.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            unreadable.append(f"{source}: {exc}")
            continue
        for path in _paths_in(text):
            referenced.setdefault(path, set()).add(source.resolve())
    return referenced, unreadable


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _decisions(directory: Path) -> list[str]:
    """比較の直下に置かれた記録から、判断を表す文字列だけを集める。"""
    found = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        for key in _DECISION_KEYS:
            value = record.get(key)
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, dict):
                found.extend(item for item in value.values() if isinstance(item, str))
    return found



# 生き残り・採用が記録された比較は、参照が一時的に無くなっても残す。
CONCLUSION_RECORDS = ("report.json", "summary.json", "receipt.json", "alternatives.json",
                      "residuals.json", "run.json")
_DECISION_KEYS = ("status", "decision", "disposition", "outcome", "state")


def _protected(path: Path) -> bool:
    if path.name in PROTECTED_NAMES or any(path.match(pattern) for pattern in PROTECTED_PATTERNS):
        return True
    # 結論の記録がある比較は、参照が一時的に無くなっても記録ごと消さない。
    if any((path / name).is_file() for name in CONCLUSION_RECORDS) or any(
            path.glob("*disposition*.json")):
        return True
    return any(item.suffix.lower() in PROTECTED_SUFFIXES
               for item in path.rglob("*") if item.is_file())


def survey() -> list[dict]:
    """削除候補になり得る単位（各rootの直下）を、大きさと参照数つきで並べる。"""
    referenced, unreadable = referenced_paths()
    tracked = {path.resolve() for path in tracked_files(".")}
    rows = []
    for base in _prunable_roots():
        if not _local_cleanup_path(base) or not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            if not _local_cleanup_path(entry) or not entry.is_dir() or entry.name == "dev":
                continue
            resolved = entry.resolve()
            # 自分の中の記録が自分を指すのは参照とみなさない。外から必要とされているかを見る。
            inside = [path for path, sources in referenced.items()
                      if (path == resolved or resolved in path.parents)
                      and any(source != resolved and resolved not in source.parents
                              for source in sources)]
            tracked_here = [path for path in tracked if resolved in path.parents]
            rows.append({
                "path": _reported_path(entry),
                "bytes": directory_size(entry),
                "references": len(inside),
                "tracked_files": len(tracked_here),
                "protected": _protected(entry),
                "modified": int(entry.stat().st_mtime),
            })
    rows.sort(key=lambda row: -row["bytes"])
    return rows, unreadable


def prunable(rows: list[dict]) -> list[dict]:
    return [row for row in rows
            if row["references"] == 0 and row["tracked_files"] == 0 and not row["protected"]]


def _human(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def report(limit: int) -> int:
    rows, unreadable = survey()
    candidates = prunable(rows)
    print(f"{'size':>9}  {'refs':>4} {'trk':>4} {'prot':>4}  path", flush=True)
    for row in rows[:limit]:
        print(f"{_human(row['bytes']):>9}  {row['references']:>4} {row['tracked_files']:>4} "
              f"{'yes' if row['protected'] else '-':>4}  {row['path']}", flush=True)
    print(f"\n合計 {_human(sum(row['bytes'] for row in rows))} / "
          f"参照なし・未追跡・保護なし {_human(sum(row['bytes'] for row in candidates))}"
          f"（{len(candidates)}件）", flush=True)
    if unreadable:
        print("読めなかった参照元:", *unreadable, sep="\n  ", flush=True)
    return 0


def prune(apply_changes: bool, limit: int, keep_reports: int = 30) -> int:
    if apply_changes:
        from storage_tiers import load_policy, require_machine_overlay
        require_machine_overlay(load_policy())
        require_owned_for_deletion(ROOT)
        record = _cleanup_receipt("disk-cleanup")
    rows, unreadable = survey()
    if unreadable:
        raise SystemExit("参照元を読めないため削除しません:\n  " + "\n  ".join(unreadable))
    candidates = prunable(rows)[:limit] if limit else prunable(rows)
    freed = 0
    reports = stale_dev_reports(keep_reports)
    if reports:
        size = sum(directory_size(path) for path in reports)
        print(f"{'削除' if apply_changes else '削除候補'}: {_human(size):>9}  "
              f"試験の実行記録 {len(reports)}件（新しい{keep_reports}件と参照済みは残す）", flush=True)
        if apply_changes:
            for path in reports:
                if not _local_deletion_path(path):
                    raise SystemExit(f"削除対象の所在が不正です: {path}")
                shutil.rmtree(path)
        freed += size
    roots = _prunable_roots()
    for row in candidates:
        target = ROOT / row["path"]
        if target.parent not in roots or not _local_cleanup_path(target) or not target.is_dir():
            raise SystemExit(f"削除対象の所在が不正です: {target}")
        print(f"{'削除' if apply_changes else '削除候補'}: {_human(row['bytes']):>9}  {row['path']}",
              flush=True)
        if apply_changes:
            if not _local_deletion_path(target):
                raise SystemExit(f"削除対象の所在が不正です: {target}")
            shutil.rmtree(target)
        freed += row["bytes"]
    summary = {"schema_version": 1, "status": "PRUNED_UNREFERENCED_WORK" if apply_changes
               else "LISTED_WITHOUT_DELETING", "applied": apply_changes,
               "directories": len(candidates), "dev_reports": len(reports),
               "freed_bytes": freed, "freed": _human(freed),
               "rule": "外部からの参照なし・Git未追跡・保護対象なしの作業物と、古い試験の実行記録のみ",
               "removed": candidates}
    if apply_changes:
        record.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["record"] = str(record)
    print(json.dumps({key: value for key, value in summary.items() if key != "removed"},
                     ensure_ascii=False), flush=True)
    return 0


# 記録された判断が「棄却」であることを示す語。要約・報告のstatus/decision/dispositionだけを見る。
REJECTION_MARKERS = ("reject", "rejected", "retired", "closed_insufficient")
MEDIA_SUFFIXES = frozenset({".wav", ".flac", ".mkv", ".mp4", ".m4a", ".mp3", ".aac", ".ogg",
                            ".opus", ".wma", ".aif", ".aiff", ".webm", ".ts", ".mka"})


def rejected_directories(rows: list[dict]) -> list[dict]:
    base = dev_state_root(ROOT) / "validation"
    selected = []
    for row in rows:
        directory = ROOT / row["path"]
        if directory.parent != base or not _local_cleanup_path(directory):
            continue
        decisions = _decisions(directory)
        matched = sorted({value for value in decisions
                          if any(marker in value.lower() for marker in REJECTION_MARKERS)})
        if matched:
            selected.append({**row, "decisions": matched})
    return selected


def retire(apply_changes: bool) -> int:
    """棄却が記録された比較の音声・映像だけを退役させる。記録と根拠のJSONは残す。"""
    if apply_changes:
        from storage_tiers import load_policy, require_machine_overlay
        require_machine_overlay(load_policy())
        require_owned_for_deletion(ROOT)
        record = _cleanup_receipt("media-retirement")
    rows, unreadable = survey()
    if unreadable:
        raise SystemExit("参照元を読めないため退役しません: " + ", ".join(unreadable))
    referenced, _ = referenced_paths()
    tracked = {path.resolve() for path in tracked_files(".")}
    classes, freed = [], 0
    for row in rejected_directories(rows):
        directory = ROOT / row["path"]
        victims = []
        for item in sorted(local_files(directory)):
            if item.suffix.lower() not in MEDIA_SUFFIXES or not _local_cleanup_path(item):
                continue
            if item in tracked:
                continue
            outside = [source for source in referenced.get(item, set())
                       if directory not in source.parents and source != directory]
            if outside:
                continue
            victims.append(item)
        if not victims:
            continue
        size = sum(path.stat().st_size for path in victims)
        classes.append({"path": row["path"], "decisions": row["decisions"],
                        "files": len(victims), "bytes": size})
        print(f"{'退役' if apply_changes else '退役候補'}: {_human(size):>9}  {len(victims):>3}件  "
              f"{row['path']}  ({', '.join(row['decisions'])[:60]})", flush=True)
        if apply_changes:
            for path in victims:
                if not _local_deletion_path(path):
                    raise SystemExit(f"退役対象の所在が不正です: {path}")
                path.unlink()
        freed += size
    summary = {"schema_version": 1, "applied": apply_changes,
               "status": "RETIRED_REJECTED_COMPARISON_MEDIA" if apply_changes else "LISTED_WITHOUT_DELETING",
               "rule": "棄却が記録された比較の、外部参照のない音声・映像のみ。JSON記録は残す。",
               "directories": len(classes), "freed_bytes": freed, "freed": _human(freed),
               "classes": classes}
    if apply_changes:
        record.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["record"] = str(record)
    print(json.dumps({key: value for key, value in summary.items() if key != "classes"},
                     ensure_ascii=False), flush=True)
    return 0

# Large generated media has a shorter lifetime than the JSON evidence that records it.
TRISTEM_NAMES = frozenset({"music.wav", "speech.wav", "effects.wav"})
DEPENDENCY_WORDS = re.compile(
    r"(?:^|[._\[\]-])(input|source|reference|baseline|current|original|dependency|parent|upstream)(?:$|[._\[\]-])",
    re.IGNORECASE,
)
DEPENDENCY_FILES = re.compile(r"(?:plan|resume|active)", re.IGNORECASE)
OUTPUT_WORDS = re.compile(
    r"(?:^|[._\[\]-])(output|outputs|artifact|artifacts|stem|stems|result|results|candidate|candidates|generated|prediction|predictions|separation|separated|render|renders|views|instrumentals)(?:$|[._\[\]-])",
    re.IGNORECASE,
)
PROVIDER_PROTECTED_PARTS = frozenset({
    "runtime", ".venv", "venv", "models", "model", "checkpoints", "checkpoint",
    "downloads", "configs", "assets", "examples", "samples", "source", "src",
    "tests", "test", "weights", "weight", "core",
})
PROVIDER_RUN_RECORDS = frozenset({
    "receipt.json", "report.json", "summary.json", "run.json", "plan.json",
    "alternatives.json", "residuals.json",
})


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except (OSError, RuntimeError, ValueError):
        return str(path).casefold()


def _resolve_candidate(value: str, source: Path, candidates: dict[str, Path], names=None) -> set[str]:
    text = value.replace("\\\\", "\\").strip().strip('"').strip("'").rstrip(".,;:) ]}")
    raw = Path(text)
    if names is None:
        names = {p.name.casefold() for p in candidates.values()}
    if raw.name.casefold() not in names:
        return set()
    choices = [raw] if re.match(r"^[A-Za-z]:[\\/]", text) else [source.parent / raw, ROOT / raw]
    if raw.parts and raw.parts[0] == ".state":
        choices.append(dev_state_root(ROOT).joinpath(*raw.parts[1:]))
    found = set()
    for choice in choices:
        key = _path_key(choice)
        if key in candidates:
            found.add(key)
    return found


def _json_candidate_refs(source: Path, candidates: dict[str, Path],
                         names: set[str]) -> list[tuple[str, str]]:
    data = json.loads(source.read_text(encoding="utf-8"))
    hits: list[tuple[str, str]] = []

    def walk(value, pointer: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{pointer}.{key}")
        elif isinstance(value, list):
            for child in value:
                walk(child, f"{pointer}[]")
        elif isinstance(value, str):
            for key in _resolve_candidate(value, source, candidates, names):
                hits.append((key, pointer))

    walk(data)
    return hits


def _raw_candidate_refs(source: Path, paths: list[tuple[str, Path]]) -> list[tuple[str, str]]:
    """Fallback for intentionally malformed/partial JSON.

    Parse failure is not treated as a global I/O failure. We scan the raw text for
    exact candidate paths and mark only those candidates as UNKNOWN.
    """
    text = source.read_text(encoding="utf-8", errors="ignore")
    haystack = text.replace("\\\\", "\\").replace("/", "\\").casefold()
    hits: set[tuple[str, str]] = set()
    for key, path in paths:
        variants: list[str] = []
        try:
            variants.append(str(path.resolve()))
        except (OSError, RuntimeError, ValueError):
            variants.append(str(path))
        try:
            variants.append(str(path.relative_to(ROOT)))
        except ValueError:
            pass
        state = dev_state_root(ROOT)
        if path.is_relative_to(state):
            variants.append(str(Path(".state") / path.relative_to(state)))
        try:
            variants.append(str(path.relative_to(source.parent)))
        except ValueError:
            pass
        if any(v.replace("/", "\\").casefold() in haystack for v in variants if v):
            hits.add((key, "$[UNPARSEABLE_JSON]"))
    return sorted(hits)


def _collect_json_refs(
    candidates: dict[str, Path], linked_paths: Iterable[Path] = (),
) -> tuple[dict[str, list[tuple[Path, str]]], list[str]]:
    paths = list(candidates.items())
    # A reference through a linked filename can still protect its local target.
    for path in linked_paths:
        key = _path_key(path)
        if key in candidates:
            paths.append((key, path))
    names = {path.name.casefold() for _, path in paths}
    refs: dict[str, list[tuple[Path, str]]] = {key: [] for key in candidates}
    unreadable: list[str] = []
    for source in reference_sources():
        if source.suffix.lower() != ".json":
            continue
        try:
            hits = _json_candidate_refs(source, candidates, names)
        except (UnicodeError, ValueError):
            try:
                hits = _raw_candidate_refs(source, paths)
            except OSError as exc:
                unreadable.append(f"{source}: {exc}")
                continue
        except OSError as exc:
            unreadable.append(f"{source}: {exc}")
            continue
        for key, pointer in hits:
            refs[key].append((source.resolve(), pointer))
    return refs, unreadable

def _protected_validation_context(path: Path, base: Path) -> bool:
    parts = [part.casefold() for part in path.relative_to(base).parts[:-1]]
    return any(
        "frozen" in part or part == "inputs" or part.endswith("-inputs") or
        (part.startswith("ui-") and "acceptance" in part)
        for part in parts
    )


def validation_media_candidates() -> tuple[list[dict], list[str]]:
    base = dev_state_root(ROOT) / "validation"
    if not base.is_dir():
        return [], []
    paths = sorted(
        (path for path in local_files(base) if path.name.lower() in TRISTEM_NAMES),
        key=lambda path: str(path).casefold(),
    )
    candidates = {_path_key(path): path for path in paths}
    refs, unreadable = _collect_json_refs(candidates)
    try:
        tracked = {_path_key(path) for path in tracked_files(".state/validation")}
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"path": _reported_path(path), "bytes": path.stat().st_size,
                 "class": "UNKNOWN", "reason": f"git ls-files failed: {exc}"} for path in paths], unreadable

    rows = []
    for key, path in candidates.items():
        if key in tracked:
            cls, reason = "KEEP", "git-tracked media"
        elif _protected_validation_context(path, base):
            cls, reason = "KEEP", "protected validation context"
        elif not refs[key]:
            cls, reason = "RETIRE", "unreferenced generated tri-stem"
        else:
            dependency = [(src, ptr) for src, ptr in refs[key]
                          if DEPENDENCY_FILES.search(src.name) or DEPENDENCY_WORDS.search(ptr)]
            ambiguous = [(src, ptr) for src, ptr in refs[key]
                         if not (DEPENDENCY_FILES.search(src.name) or DEPENDENCY_WORDS.search(ptr)
                                 or OUTPUT_WORDS.search(ptr))]
            if dependency:
                cls, reason = "KEEP", f"dependency reference ({len(dependency)})"
            elif ambiguous:
                cls, reason = "UNKNOWN", f"ambiguous reference ({len(ambiguous)})"
            else:
                cls, reason = "RETIRE", "output/provenance-only reference"
        rows.append({"path": _reported_path(path), "bytes": path.stat().st_size,
                     "class": cls, "reason": reason})
    rows.sort(key=lambda row: (-row["bytes"], row["path"]))
    return rows, unreadable


def _provider_protected(path: Path, base: Path) -> bool:
    parts = [part.casefold() for part in path.relative_to(base).parts[:-1]]
    if any(part in PROVIDER_PROTECTED_PARTS for part in parts):
        return True
    if any("upstream" in part or "source" in part for part in parts):
        return True
    return any("input" in part or "reference" in part or "baseline" in part or
               "original" in part or "frozen" in part for part in parts)


def _provider_run_root(path: Path, base: Path) -> Path | None:
    current = path.parent
    while current != base and base in current.parents:
        if any((current / name).is_file() for name in PROVIDER_RUN_RECORDS):
            return current
        current = current.parent
    return None


def provider_media_candidates() -> tuple[list[dict], list[str]]:
    from storage_tiers import is_link

    base = ROOT / ".providers"
    if not base.is_dir():
        return [], []
    run_roots: dict[str, Path] = {}
    paths = []
    linked_paths = []
    rows = []
    for path in base.rglob("*"):
        if path.suffix.lower() not in MEDIA_SUFFIXES or _provider_protected(path, base):
            continue
        linked = is_link(path)
        if not linked and not path.is_file():
            continue
        run_root = _provider_run_root(path, base)
        if run_root is None:
            continue
        if linked:
            linked_paths.append(path)
            rows.append({"path": _reported_path(path), "bytes": 0,
                         "class": "KEEP", "reason": "linked"})
            continue
        paths.append(path)
        run_roots[_path_key(path)] = run_root.resolve()
    candidates = {_path_key(path): path for path in paths}
    refs, unreadable = _collect_json_refs(candidates, linked_paths)
    try:
        tracked = {_path_key(path) for path in tracked_files(".providers")}
    except (OSError, subprocess.SubprocessError) as exc:
        rows.extend({"path": _reported_path(path), "bytes": path.stat().st_size,
                     "class": "UNKNOWN", "reason": f"git ls-files failed: {exc}"} for path in paths)
        return rows, unreadable

    for key, path in candidates.items():
        if key in tracked:
            cls, reason = "KEEP", "git-tracked media"
        else:
            dependency = [(src, ptr) for src, ptr in refs[key]
                          if DEPENDENCY_FILES.search(src.name) or DEPENDENCY_WORDS.search(ptr)]
            ambiguous = []
            for src, ptr in refs[key]:
                inside_run = src == run_roots[key] or run_roots[key] in src.parents
                if DEPENDENCY_FILES.search(src.name) or DEPENDENCY_WORDS.search(ptr):
                    continue
                if not OUTPUT_WORDS.search(ptr):
                    ambiguous.append((src, ptr))
                elif not inside_run and not OUTPUT_WORDS.search(ptr):
                    ambiguous.append((src, ptr))
            if dependency:
                cls, reason = "KEEP", f"dependency reference ({len(dependency)})"
            elif ambiguous:
                cls, reason = "UNKNOWN", f"ambiguous reference ({len(ambiguous)})"
            else:
                cls, reason = "RETIRE", "provider experiment output/provenance only"
        rows.append({"path": str(path.relative_to(ROOT).as_posix()), "bytes": path.stat().st_size,
                     "class": cls, "reason": reason})
    rows.sort(key=lambda row: (-row["bytes"], row["path"]))
    return rows, unreadable


def _media_gc(rows: list[dict], unreadable: list[str], apply_changes: bool,
              root: Path, allowed_names: frozenset[str] | None, receipt_prefix: str,
              quiet: bool = False) -> int:
    if apply_changes:
        from storage_tiers import load_policy, require_machine_overlay
        require_machine_overlay(load_policy())
    if unreadable:
        raise SystemExit("参照元を読めないため退役しません: " + ", ".join(unreadable))
    victims = [row for row in rows if row["class"] == "RETIRE"]
    unknown = [row for row in rows if row["class"] == "UNKNOWN"]
    keep = [row for row in rows if row["class"] == "KEEP"]
    freed = sum(row["bytes"] for row in victims)
    if not quiet:
        for row in victims:
            print(f"{'退役' if apply_changes else '退役候補'}: {_human(row['bytes']):>9}  {row['path']}  ({row['reason']})")
    if apply_changes:
        record = _cleanup_receipt(receipt_prefix)
        targets = []
        for row in victims:
            path = ROOT / row["path"]
            if not path.is_relative_to(root) or not _local_cleanup_path(path) or not path.is_file():
                raise SystemExit(f"退役対象の所在が不正です: {path}")
            if allowed_names is not None and path.name.lower() not in allowed_names:
                raise SystemExit(f"退役対象名が不正です: {path}")
            if path.stat().st_size != row["bytes"]:
                raise SystemExit(f"監査後にサイズが変化しました: {path}")
            targets.append(path)
        for path in targets:
            if not _local_deletion_path(path):
                raise SystemExit(f"退役対象の所在が不正です: {path}")
            path.unlink()
    applied_status = ("RETIRED_REBUILDABLE_MEDIA_WITH_UNKNOWN_PRESERVED"
                      if apply_changes and unknown else
                      "RETIRED_REBUILDABLE_MEDIA" if apply_changes else
                      "LISTED_WITHOUT_DELETING")
    summary = {"schema_version": 1, "applied": apply_changes,
               "status": applied_status,
               "retire_files": len(victims), "keep_files": len(keep), "unknown_files": len(unknown),
               "freed_bytes": freed, "freed": _human(freed)}
    if apply_changes:
        record.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["record"] = str(record)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def validation_media_gc(apply_changes: bool, quiet: bool = False) -> int:
    if apply_changes:
        from storage_tiers import load_policy, require_machine_overlay
        require_machine_overlay(load_policy())
        state = require_owned_for_deletion(ROOT)
    else:
        state = dev_state_root(ROOT)
    rows, unreadable = validation_media_candidates()
    return _media_gc(rows, unreadable, apply_changes, state / "validation",
                     TRISTEM_NAMES, "media-gc", quiet)


def provider_media_gc(apply_changes: bool, quiet: bool = False) -> int:
    if apply_changes:
        from storage_tiers import load_policy, require_machine_overlay
        require_machine_overlay(load_policy())
    rows, unreadable = provider_media_candidates()
    return _media_gc(rows, unreadable, apply_changes, ROOT / ".providers",
                     None, "provider-media-gc", quiet)


DEV_REPORT_ROOT = Path("validation/dev")


def stale_dev_reports(keep: int) -> list[Path]:
    """試験のたびに増える実行記録のうち、新しい方からkeep件と参照済みを残す。"""
    base = dev_state_root(ROOT) / DEV_REPORT_ROOT
    if not _local_cleanup_path(base) or not base.is_dir():
        return []
    referenced, _ = referenced_paths()
    tracked = {path.resolve() for path in tracked_files(".")}
    entries = sorted((path for path in base.iterdir() if _local_cleanup_path(path) and path.is_dir()),
                     key=lambda path: path.stat().st_mtime, reverse=True)
    stale = []
    for path in entries[keep:]:
        resolved = path.resolve()
        # Gitが追跡している実行記録は、古くても作業樹から消さない。
        if any(resolved in item.parents for item in tracked):
            continue
        # 記録が自分の所在を書いているのは参照ではない。外から必要とされているかを見る。
        outside = any((item == resolved or resolved in item.parents)
                      and any(source != resolved and resolved not in source.parents
                              for source in sources)
                      for item, sources in referenced.items())
        if outside:
            continue
        stale.append(path)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("report", help="大きさと参照数を並べる（読むだけ）")
    listing.add_argument("--limit", type=int, default=40)
    pruning = commands.add_parser("prune", help="参照のない再生成可能な作業物を消す")
    pruning.add_argument("--apply", action="store_true", help="実際に削除する（既定は候補表示のみ）")
    pruning.add_argument("--limit", type=int, default=0, help="先頭N件だけを対象にする")
    pruning.add_argument("--keep-reports", type=int, default=30,
                         help="残す試験の実行記録の件数（既定30）")
    retiring = commands.add_parser("retire", help="棄却が記録された比較の音声・映像を退役させる")
    retiring.add_argument("--apply", action="store_true", help="実際に削除する（既定は候補表示のみ）")
    media = commands.add_parser("media-gc", help="再生成可能なvalidation tri-stemをfile単位で退役させる")
    media.add_argument("--apply", action="store_true", help="実際に削除する（既定は候補表示のみ）")
    media.add_argument("--quiet", action="store_true", help="候補一覧を省略してsummaryだけ表示する")
    provider_media = commands.add_parser("provider-media-gc", help="provider実験mediaをfile単位で退役させる")
    provider_media.add_argument("--apply", action="store_true", help="実際に削除する（既定は候補表示のみ）")
    provider_media.add_argument("--quiet", action="store_true", help="候補一覧を省略してsummaryだけ表示する")
    args = parser.parse_args()
    if args.command == "report":
        return report(args.limit)
    if args.command == "retire":
        return retire(args.apply)
    if args.command == "media-gc":
        return validation_media_gc(args.apply, args.quiet)
    if args.command == "provider-media-gc":
        return provider_media_gc(args.apply, args.quiet)
    return prune(args.apply, args.limit, args.keep_reports)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StorageMutationRefused as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
