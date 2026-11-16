"""SSD/HDD tiers for LSS development: keep the SSD for work, move idle results to the HDD before space runs out.

    python scripts/storage_tiers.py status                 # free space, watermarks, what is hot, what could move
    python scripts/storage_tiers.py demote [--apply]       # move idle candidates (oldest first) until target free
    python scripts/storage_tiers.py scheduled              # OS-owned logged maintenance, no agent required
    python scripts/storage_tiers.py auto [--detach]        # legacy hook; scheduler_only policy defers to OS task
    python scripts/storage_tiers.py promote <path> [--apply]
                                                           # a demoted junction: back to the SSD; an HDD worktree: to
                                                           # C:/LSS-Work/wt (dry run unless --apply)
    python scripts/storage_tiers.py prune [--apply]        # delete regenerable copies, hard-link duplicates (HDD)
    python scripts/storage_tiers.py worktrees [--retire [--apply] [--hours 72]]
                                                           # idle clean worktrees (--retire: every volume, dry run unless --apply)
    python scripts/storage_tiers.py worktrees --report     # where each worktree physically is, last commit, who uses it

Order of tier maintenance (`scheduled` -> `maintain`) when C: is short: 1. the read-only stage cache (`scripts/stage.py`,
`stage` in the policy) is pruned least recently used first until C: has `evict_first_below_free_gb` free, because it is
regenerable by construction and never demoted; 2. superseded downloads and temp orphans are deleted; 3. idle results
are demoted to the HDD; then (weekly) the HDD roots are pruned and idle worktrees retired. Active lanes belong on the
SSD: `worktrees --report` flags those still on the HDD and `promote <worktree> --apply` moves one (`worktree_promote`).

Policy: public `scripts/storage_policy.json` plus `config/storage_policy.local.json` below
`LSS_PRIVATE_ROOT`. Deletion and relocation require the machine overlay and nonempty
protection lists; reports and dry runs can use public defaults. A candidate moves only when nothing under it was written for
`min_idle_hours`, no process runs from or names it, and (for a worktree's own state/env) the worktree itself
is idle. Every file is copied with its SHA-256 computed while reading the source, the copy is read back and
compared, and only then is the original removed and a junction left at its path, so recorded paths, Evidence
and frozen candidates keep working. The mover runs at background I/O priority, so a running build or test is
not made to wait, and it starts at a watermark (not at the moment the disk is full). Relocation receipts go to `receipts_root`; moved results require a verified copy.
Policy-owned regenerable leftovers are deleted with local receipts.

An enabled or invalid development sleep policy defers `auto` before policy/root
discovery, daily cleanups, archive scans or detached launch. Explicit commands and
admission floors remain available; deferral never claims to have freed capacity.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
    from scripts.machine_config import StorageMutationRefused, load_config, require_machine_overlay
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root
    from machine_config import StorageMutationRefused, load_config, require_machine_overlay


POLICY_PATH = Path(__file__).with_name("storage_policy.json")
CHUNK = 8 * 1024 * 1024
MANIFEST = ".lss-relocation-manifest.json"
PATCH = "UNCOMMITTED.patch"
PARTIAL = ".lss-tier-partial"          # marker written first into a copy in progress (proves it is ours)
PARTIAL_SUFFIX = ".lss-partial"

if os.name == "nt":
    # The OS task runs under pythonw (no console), so every git/powershell/tasklist child would open its own console
    # window on the user's desktop. Give every child a hidden console instead unless a call asks otherwise.
    _CREATE_NO_WINDOW = 0x08000000

    class _HiddenPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = _HiddenPopen


def main_root() -> Path:
    """The main checkout (worktrees live under it), whichever checkout runs this script."""
    common = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (Path(__file__).parent / common_path).resolve()
    return common_path.parent


def load_policy(path: Path = POLICY_PATH) -> dict:
    policy = load_config(path)
    if policy.get("schema_version") != 1 or policy["target_free_gb"] < policy["start_free_gb"]:
        raise ValueError("storage policy is invalid")
    return policy


def is_link(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


@dataclass
class Tree:
    bytes: int = 0
    files: int = 0
    newest: float = 0.0
    links: list[str] = field(default_factory=list)
    directory_names: set[str] = field(default_factory=set)


def survey(root: Path) -> Tree:
    """Size and newest write under root, never following junctions or symlinks (counted, not entered)."""
    tree = Tree()
    stack = [str(root)]
    while stack:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    tree.links.append(entry.path)
                    continue
                tree.newest = max(tree.newest, st.st_mtime)
                if stat.S_ISDIR(st.st_mode):
                    tree.directory_names.add(entry.name.lower())
                    stack.append(entry.path)
                else:
                    tree.bytes += st.st_size
                    tree.files += 1
    return tree


def free_gb(volume: str) -> float:
    return shutil.disk_usage(volume).free / 1e9


@contextlib.contextmanager
def storage_mutation(policy: dict, paths):
    """Keep admission atomic with a mutation; existing cleanup callers retain on OSError."""
    require_machine_overlay(policy)
    import storage_budget
    try:
        with storage_budget.mutation_guard(paths, policy):
            yield
    except storage_budget.BudgetError as exc:
        raise OSError(str(exc)) from exc


def process_texts() -> list[str]:
    """Executable paths and command lines of running processes, lower-cased (Windows; empty elsewhere)."""
    if os.name != "nt":
        return []
    command = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ExecutablePath)`t$($_.CommandLine)\" }"]
    environment = {key: value for key, value in os.environ.items() if key != "PSModulePath"}
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=60, env=environment,
                             encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.TimeoutExpired):
        return ["<process list unavailable>"]
    return [line.lower() for line in out.splitlines() if line.strip()]


def worktree_idle_hours(worktree: Path) -> float | None:
    """Hours since the worktree's HEAD moved (a commit, checkout, merge or reset); None if not a worktree.

    The index is not a signal: read-only commands (`git status`, the session-start lane listing) refresh its stat
    cache and rewrite it, which made every worktree look active and kept 50+ GB of idle results on the SSD.
    Ongoing writes are still caught by the candidate's own newest-file age."""
    marker = worktree / ".git"
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text.split(":", 1)[1].strip())
    stamps = [p.stat().st_mtime for p in (gitdir / "HEAD", gitdir / "logs" / "HEAD") if p.exists()]
    return (time.time() - max(stamps)) / 3600 if stamps else None


_TRACKED: dict[str, set[str]] = {}


def tracked_dirs(repository: Path) -> set[str]:
    """Every directory (lower-case, absolute) that holds a file git tracks in this checkout. Such a directory is part
    of the repository, not a result: it never moves (git would see its files through a junction as changed)."""
    key = str(repository).lower()
    if key not in _TRACKED:
        out = subprocess.run(["git", "-C", str(repository), "ls-files", "-z"], capture_output=True, check=False).stdout
        found = set()
        for name in out.decode("utf-8", errors="replace").split("\0"):
            parent = Path(name).parent
            while str(parent) not in (".", ""):
                text = str(repository / parent).lower()
                if text in found:
                    break
                found.add(text)
                parent = parent.parent
        _TRACKED[key] = found
    return _TRACKED[key]


def _checkout_of(path: Path, root: Path) -> Path | None:
    """The checkout whose index governs path: a worktree under root/.worktrees, else root itself."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) >= 2 and parts[0] == ".worktrees":
        return root / ".worktrees" / parts[1]
    return root


@dataclass
class Candidate:
    path: str
    cls: str
    gb: float
    files: int
    idle_hours: float
    eligible: bool
    reason: str
    minimum_gb: float = 0.0


def _never(path: Path, root: Path, policy: dict) -> bool:
    if any(part.lower() in policy.get("hot_directory_names", []) for part in path.parts):
        return True
    for item in policy["never"]:
        blocked = Path(item) if Path(item).is_absolute() or ":" in item else root / item
        try:
            path.relative_to(blocked)
            return True
        except ValueError:
            pass
    return False


def idle_threshold_hours(policy: dict, free_gb: float | None = None) -> float:
    """`min_idle_hours`, shortened to `pressure_min_idle_hours` while C: free space is below `pressure_free_gb`.

    Fast producers (an 8.4 GB package build every hour or two) outrun a fixed 6 h wait and can fill C:; under
    pressure only the age rule relaxes, the in-use, link and git-tracked protections stay."""
    if "pressure_free_gb" not in policy:
        return policy["min_idle_hours"]
    if free_gb is None:
        free_gb = shutil.disk_usage(policy["hot_volume"]).free / 1e9
    return policy["pressure_min_idle_hours"] if free_gb < policy["pressure_free_gb"] else policy["min_idle_hours"]


def _glob_parts(base: Path, pattern: str) -> list[Path]:
    """Path.glob, one pattern part at a time, so a directory that cannot be listed (a junction whose target was
    reclaimed, or a path removed during the walk) is skipped instead of aborting every other match in the scope."""
    paths = [base]
    for part in Path(pattern).parts:
        found = []
        for path in paths:
            try:
                found.extend(path.glob(part))
            except OSError:
                continue
        paths = found
    return paths


def _named_in(record: Path, name: str) -> bool:
    """Whether a record (the installed-product release record) names this directory; an unreadable record names
    every directory, so nothing it guards (the deployed release, its rollback target) moves on a read failure."""
    try:
        return name.lower() in record.read_text(encoding="utf-8").lower()
    except OSError:
        return True


def candidates(policy: dict, root: Path, *, now: float | None = None, processes: list[str] | None = None) -> list[Candidate]:
    import storage_budget
    protected = storage_budget.protected_paths(policy)
    completed = storage_budget.completed_paths(policy)
    now = time.time() if now is None else now
    processes = process_texts() if processes is None else processes
    processes = [text.replace("/", "\\") for text in processes]
    found: dict[str, Candidate] = {}
    min_idle = idle_threshold_hours(policy)

    def evaluate(path: Path, rule: dict, worktree: Path | None, depth: int) -> None:
        key = str(path)
        if key in found or is_link(path) or not path.is_dir() or _never(path, root, policy):   # is_link first: is_dir follows a junction to the HDD
            return
        tree = survey(path)
        checkout = _checkout_of(path, root)
        holds_tracked = checkout is not None and key.lower() in tracked_dirs(checkout)
        hot_descendant = bool(tree.directory_names.intersection(policy.get("hot_directory_names", [])))
        if (tree.links or holds_tracked or hot_descendant) and depth < 4:
            # A directory holding junctions (earlier moves) or files git tracks is not moved whole: its other parts are
            # judged one by one (a tracked file itself never moves; an untracked result folder beside it does).
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    evaluate(child, rule, worktree, depth + 1)
            return
        gb = tree.bytes / 1e9
        idle = (now - tree.newest) / 3600 if tree.newest else float("inf")
        finished = any(_inside(path, Path(p)) and tree.newest <= stamp for p, stamp in completed)
        reason, eligible = "idle", True
        if gb < rule.get("min_candidate_gb", policy["min_candidate_gb"]):
            reason, eligible = "small", False
        elif tree.links:
            reason, eligible = "contains links", False
        elif hot_descendant:
            reason, eligible = "contains hot development environment", False
        elif holds_tracked:
            reason, eligible = "holds files tracked by git", False
        elif any(_inside(path, Path(p)) or _inside(Path(p), path) for p in protected):
            reason, eligible = "protected by an active storage reservation", False
        elif idle < max(min_idle, rule.get("min_idle_hours", 0)) and not finished:
            reason, eligible = f"written {idle:.1f} h ago", False
        elif "<process list unavailable>" in processes:
            reason, eligible = "process list unavailable", False
        elif any(key.lower() in text for text in processes):
            reason, eligible = "named by a running process", False
        elif rule.get("keep_named_in") and _named_in(root / rule["keep_named_in"], rule["_top"].name):
            reason, eligible = f"named by {rule['keep_named_in']}", False
        elif worktree is not None:
            wt_idle = worktree_idle_hours(worktree)
            if wt_idle is not None and wt_idle < min_idle and not finished:
                reason, eligible = f"worktree active {wt_idle:.1f} h ago", False
        if eligible and finished:
            reason = "explicit output completed; unchanged since completion"
        found[key] = Candidate(key, rule["class"], round(gb, 3), tree.files, round(idle, 1), eligible, reason,
                               rule.get("min_candidate_gb", policy["min_candidate_gb"]))

    for rule in policy["candidates"]:
        pattern = rule["glob"]
        if rule.get("absolute"):
            anchor = Path(Path(pattern).anchor)
            matches = sorted(_glob_parts(anchor, str(Path(pattern).relative_to(anchor))))
        else:
            matches = sorted(_glob_parts(root, pattern))
        for path in matches:
            # keep_named_in matches the glob's own directory name, also when its children are judged one by one
            evaluate(path, {**rule, "_top": path}, path.parent if rule.get("requires_idle_worktree") else None, 0)
            for owned, _ in completed:
                output = Path(owned)
                if output != path and _inside(output, path):
                    evaluate(output, {**rule, "_top": path},
                             path.parent if rule.get("requires_idle_worktree") else None, 0)
    return sorted(found.values(), key=lambda c: (-c.idle_hours, -c.gb))


def background_priority() -> None:
    """Background I/O and CPU priority for this process (Windows PROCESS_MODE_BACKGROUND_BEGIN)."""
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00100000)


def _long(path: Path | str) -> str:
    text = str(path)
    return text if os.name != "nt" or text.startswith("\\\\?\\") else "\\\\?\\" + os.path.abspath(text)


def _copy_file(source: Path, target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(_long(source), "rb") as reader, open(_long(target), "wb") as writer:
        while chunk := reader.read(CHUNK):
            digest.update(chunk)
            writer.write(chunk)
            size += len(chunk)
    shutil.copystat(_long(source), _long(target))
    return size, digest.hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_long(path), "rb") as reader:
        while chunk := reader.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_tree(root: Path, *, links: list[Path] | None = None, skip=None) -> tuple[list[Path], list[Path]]:
    """Files and directories under root (sorted files). A link inside raises unless `links` collects it (the link is
    then neither entered nor copied); `skip(path, is_dir)` drops a directory or file with everything under it."""
    out, dirs, stack = [], [], [root]
    while stack:
        directory = stack.pop()
        with os.scandir(_long(directory)) as entries:
            for entry in entries:
                path = directory / entry.name
                st = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    if links is None:
                        raise RuntimeError(f"link inside a candidate is not moved: {path}")
                    links.append(path)
                elif skip is not None and skip(path, stat.S_ISDIR(st.st_mode)):
                    continue
                elif stat.S_ISDIR(st.st_mode):
                    dirs.append(path)
                    stack.append(path)
                else:
                    out.append(path)
    return sorted(out), dirs


def _files(root: Path) -> list[Path]:
    return _scan_tree(root)[0]


def copy_verified(source: Path, target: Path, *, allow_existing: bool = False, links: list[Path] | None = None,
                  skip=None) -> dict:
    """Copy every file, hashing the source as it is read; read the copy back and compare. Raises on mismatch.
    With `links` a junction or symlink inside is collected there and left out (the caller recreates it); `skip` drops
    regenerable paths."""
    if target.exists() and not allow_existing:
        raise FileExistsError(target)
    files, dirs = _scan_tree(source, links=links, skip=skip)
    rows = {}
    for path in files:
        relative = str(path.relative_to(source))
        size, sha = _copy_file(path, target / relative)
        rows[relative] = {"bytes": size, "sha256": sha}
    target.mkdir(parents=True, exist_ok=True)
    for directory in dirs:
        (target / directory.relative_to(source)).mkdir(parents=True, exist_ok=True)
    mismatched = [relative for relative, row in rows.items()
                  if os.path.getsize(_long(target / relative)) != row["bytes"] or _hash(target / relative) != row["sha256"]]
    if mismatched:
        raise RuntimeError(f"verification failed for {len(mismatched)} files, original kept: {mismatched[:5]}")
    return rows


def _replace_with_retry(source: Path, target: Path, attempts: int = 12) -> None:
    """Rename a just-written directory. A scanner holding a fresh file briefly denies the rename (WinError 5/32)."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(30.0, 0.5 * 2 ** attempt))


def _remove_tree(path: Path) -> None:
    def writable(function, name, _):
        os.chmod(name, stat.S_IWRITE)
        function(name)
    shutil.rmtree(_long(path), onerror=writable)


def make_junction(link: Path, target: Path) -> None:
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def archive_path(policy: dict, path: Path) -> Path:
    stem = str(path).replace(":", "").replace("\\", "/").strip("/")
    target = Path(policy["archive_root"]) / stem
    return target if not target.exists() else target.with_name(f"{target.name}-{time.strftime('%Y%m%dT%H%M%S')}")


def _receipt(policy: dict, record: dict) -> Path:
    receipts = Path(policy["receipts_root"])
    receipts.mkdir(parents=True, exist_ok=True)
    name = Path(record["source"]).name.strip(".") or "dir"
    path = receipts / f"tier-{record['action']}-{time.strftime('%Y%m%dT%H%M%S')}-{name}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(Path(policy["archive_root"]) / "index.jsonl", "a", encoding="utf-8") as index:
        index.write(json.dumps({k: v for k, v in record.items() if k != "files"}, ensure_ascii=False) + "\n")
    return path


def demote_one(policy: dict, source: Path) -> dict:
    require_machine_overlay(policy)
    with storage_mutation(policy, [source]):
        result = _demote_one(policy, source)
    import storage_budget
    storage_budget.acknowledge_archived(source, policy)
    return result


def _demote_one(policy: dict, source: Path) -> dict:
    """Verified move of one directory to the archive, junction at the original path."""
    require_machine_overlay(policy)
    if is_link(source):
        raise RuntimeError(f"already a link: {source}")
    stage_root = Path(os.path.expandvars((policy.get("stage") or {}).get("root", "C:/LSS-Work/stage")))
    if source == stage_root or source.is_relative_to(stage_root):
        raise RuntimeError(f"the stage cache is pruned, never demoted: {source}")
    target = archive_path(policy, source)
    started, free_before = time.time(), free_gb(policy["hot_volume"])
    staging = target.with_name(target.name + PARTIAL_SUFFIX)
    staging.mkdir(parents=True)
    (staging / PARTIAL).write_text(str(source), encoding="utf-8")
    rows = copy_verified(source, staging, allow_existing=True)
    rows.pop(PARTIAL, None)
    _replace_with_retry(staging, target)
    (target / PARTIAL).unlink()             # the marker leaves only once the verified copy has its final name
    record = {"type": "lss_storage_tier", "action": "demote", "source": str(source), "archive": str(target),
              "files": rows, "file_count": len(rows), "bytes": sum(r["bytes"] for r in rows.values()),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (target / MANIFEST).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    _remove_tree(source)
    make_junction(source, target)
    seen = sum(1 for _ in _files_via_link(source))
    if seen != len(rows) + 1:     # + the manifest
        raise RuntimeError(f"junction probe saw {seen} files, expected {len(rows) + 1}")
    record.update(status="DEMOTED_VERIFIED_JUNCTION", seconds=round(time.time() - started, 1),
                  free_gb_before=round(free_before, 2), free_gb_after=round(free_gb(policy["hot_volume"]), 2))
    record["receipt"] = str(_receipt(policy, record))
    return record


def _files_via_link(link: Path):
    for directory, _, files in os.walk(link):
        for name in files:
            yield os.path.join(directory, name)


def promote(policy: dict, link: Path) -> dict:
    require_machine_overlay(policy)
    with storage_mutation(policy, [link]):
        return _promote(policy, link)


def _promote(policy: dict, link: Path) -> dict:
    """Bring a demoted directory back to the SSD (verified); the archive copy stays as a backup."""
    require_machine_overlay(policy)
    if not is_link(link):
        raise RuntimeError(f"not a junction: {link}")
    target = Path(os.path.realpath(link))
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    staging = link.with_name(link.name + ".promote-tmp")
    if staging.exists():
        _remove_tree(staging)
    rows = copy_verified(target, staging)
    expected = {k: v for k, v in manifest["files"].items()}
    rows.pop(MANIFEST, None)
    if rows != expected:
        _remove_tree(staging)
        raise RuntimeError("archive copy differs from its demotion manifest; nothing changed")
    (staging / MANIFEST).unlink()
    os.rmdir(link)                 # removes the junction only, never the archive behind it
    os.replace(staging, link)
    record = {"type": "lss_storage_tier", "action": "promote", "source": str(link), "archive": str(target),
              "file_count": len(rows), "bytes": sum(r["bytes"] for r in rows.values()),
              "status": "PROMOTED_VERIFIED", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    record["receipt"] = str(_receipt(policy, record))
    return record


class Lock:
    """One mover at a time: an OS byte-range lock on an open file. The OS releases it when the holder exits or is
    killed, so a stale lock cannot exist and no process id is ever guessed or signalled."""

    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("another mover is running")
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import msvcrt

            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.stream.close()

def recover(policy: dict, log=print) -> None:
    """After an interrupted mover: drop our own unfinished copies, record finished moves that lost their receipt."""
    require_machine_overlay(policy)
    archive = Path(policy["archive_root"])
    if not archive.is_dir():
        return
    recorded = set()
    index = archive / "index.jsonl"
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            try:
                recorded.add(json.loads(line)["archive"])
            except (ValueError, KeyError):
                pass
    for directory, names, files in os.walk(archive):
        here = Path(directory)
        if here.name.endswith(PARTIAL_SUFFIX) and PARTIAL in files:
            _remove_tree(here)               # our own unverified copy; its source was never touched
            log(f"removed unfinished copy {here}")
            names[:] = []
            continue
        if MANIFEST in files:
            names[:] = []
            if str(here) in recorded:
                continue
            record = json.loads((here / MANIFEST).read_text(encoding="utf-8"))
            source = Path(record["source"])
            moved = is_link(source) and Path(os.path.realpath(source)) == here.resolve()
            record.update(status="DEMOTED_VERIFIED_JUNCTION_RECOVERED" if moved else "VERIFIED_COPY_SOURCE_KEPT",
                          recovered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            record["receipt"] = str(_receipt(policy, record))
            log(f"recorded {record['status']}: {source}")


def local_state(policy: dict) -> Path:
    """Lock, journal, log and deletion receipts live on the SSD, so a run with nothing to do never wakes the HDD."""
    path = Path(os.path.expandvars(policy.get("local_state_root", "%LOCALAPPDATA%\\LSS-dev\\storage-tiers")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _disposable_name(name: str, policy: dict) -> bool:
    rule = policy.get("disposable") or {}
    return name in rule.get("names", ()) or any(name.startswith(prefix) for prefix in rule.get("prefixes", ()))


def disposables(policy: dict, root: Path, candidate: Path) -> list[Path]:
    """Regenerable directories inside an idle candidate (pytest temp roots, Python caches), outermost only.
    Never a link, never anything holding a git-tracked file."""
    if not policy.get("disposable"):
        return []
    checkout = _checkout_of(candidate, root)
    tracked = tracked_dirs(checkout) if checkout is not None else set()
    found, stack = [], [candidate]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            path = Path(entry.path)
            if is_link(path) or not entry.is_dir(follow_symlinks=False):
                continue
            if _disposable_name(entry.name, policy) or is_package_copy(path, policy, root):
                if str(path).lower() not in tracked:
                    found.append(path)
                continue
            stack.append(path)
    return sorted(found)


# --- What is regenerable, what is kept --------------------------------------------------------------------------------
# Deleted (never carried to the HDD): Python/pytest caches, WebView2 profiles of test cases, and copies of an LSS
# package (LSS.exe + _internal), which its commit rebuilds; the deployed package, every recorded rollback payload and
# the build of a frozen candidate still waiting for its verdict are protected. Model weights outside the canonical
# stores are re-downloadable copies and are deleted. Large files that exist twice with the same bytes on one volume
# become hard links (the path stays, the bytes are stored once). Everything else (receipts, Evidence, result audio,
# fixtures) is kept: moved to the HDD when idle.

_PROTECTED: dict[str, set[str]] = {}
MODEL_SUFFIXES = (".ckpt", ".pth", ".pt", ".safetensors", ".onnx")


def protected_packages(policy: dict, root: Path) -> set[str]:
    """Package directories that must stay: every package path named by a release record (deployed, rollback,
    previous), configured extras, and builds of frozen candidates without a closing VERDICT/DEPLOYED/HELD/SUPERSEDED."""
    state = dev_state_root(root)
    key = (str(root) + "|" + str(state)).lower()
    if key in _PROTECTED:
        return _PROTECTED[key]
    found = {str(Path(p)).lower().rstrip("\\") for p in policy.get("protected_packages", [])}

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            text = value.lower().rstrip("\\")
            if text.endswith("\\dist\\lss") or text.endswith("\\freesoft\\lss") or "\\rollback\\" in text:
                found.add(text)

    for record in (state / "releases").rglob("*.json"):
        if "product" in record.name:
            try:
                walk(json.loads(record.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
    for release_root in policy.get("frozen_roots", []):
        base = Path(release_root)
        if not base.is_dir():
            continue
        for frozen in base.glob("FROZEN-*"):
            names = [q.name.upper() for q in frozen.iterdir()] if frozen.is_dir() else []
            if not any(n.startswith(("VERDICT", "DEPLOYED", "HELD", "SUPERSEDED")) for n in names):
                found.add(str(base / f"build-{frozen.name[7:]}" / "dist" / "lss").lower())
    found |= {str(Path(p).resolve()).lower().rstrip('\\') for p in list(found)}
    _PROTECTED[key] = found
    return found


def is_package_copy(path: Path, policy: dict, root: Path) -> bool:
    """A copy of an LSS package (LSS.exe beside _internal) that no release or pending candidate needs."""
    if not policy.get("delete_package_copies"):
        return False
    try:
        if not (path / "LSS.exe").is_file() or not (path / "_internal").is_dir():
            return False
    except OSError:
        return False
    text = str(path).lower().rstrip("\\")
    if "\\rollback\\" in text + "\\":
        return False                                   # a rollback payload, wherever a junction put it
    return not any(text == q or text.startswith(q + "\\") or q.startswith(text + "\\")
                   for q in protected_packages(policy, root))


def _regenerable_roots(policy: dict, root: Path, top: Path) -> list[Path]:
    """Outermost regenerable directories under (and including) top."""
    if is_package_copy(top, policy, root) or _disposable_name(top.name, policy):
        return [top]
    return disposables(policy, root, top)


def _canonical_model(path: str, policy: dict) -> bool:
    text = path.lower()
    return any(text.startswith(str(Path(c)).lower().rstrip("\\") + "\\") for c in policy.get("model_stores", []))


def stored_model_index(policy: dict) -> dict:
    stored = {}
    for store in policy.get("model_stores", []):
        for directory, names, files in os.walk(store):
            names[:] = [n for n in names if not is_link(Path(directory) / n)]
            for name in files:
                path = Path(directory) / name
                if name.lower().endswith(MODEL_SUFFIXES) and not is_link(path):
                    try:
                        stored.setdefault((name.lower(), path.stat().st_size), []).append(path)
                    except OSError:
                        pass
    return stored


def verified_model_copy(full: str, size: int, stored: dict) -> bool:
    """A matching filename and byte count only select candidates; SHA256 proves identity.
    Mutable or unreadable files stay. Canonical originals are never deleted here.
    """
    path = Path(full)
    originals = stored.get((path.name.lower(), size), [])
    if not originals:
        return False
    def stamp(p):
        st = p.stat()
        return st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino
    try:
        before = stamp(path)
        digest = _hash(path)
        if before != stamp(path):
            return False
        for original in originals:
            first = stamp(original)
            if digest == _hash(original) and first == stamp(original):
                return True
    except OSError:
        pass
    return False


def model_copies(policy: dict, roots: list[Path], idle: float) -> list[tuple[str, int]]:
    """Model weight files outside the canonical model stores (re-downloadable copies), idle and not read-only."""
    if not policy.get("model_stores"):
        return []
    found = []
    for top in roots:
        if not top.is_dir():
            continue
        for directory, names, files in os.walk(top):
            names[:] = [n for n in names if not is_link(Path(directory) / n)]
            for name in files:
                if not name.lower().endswith(MODEL_SUFFIXES):
                    continue
                full = os.path.join(directory, name)
                if _canonical_model(full, policy):
                    continue
                try:
                    st = os.stat(full, follow_symlinks=False)
                except OSError:
                    continue
                if st.st_mtime < idle and st.st_size >= 1024 * 1024:
                    found.append((full, st.st_size))
    # Name and size narrow the candidates; SHA256 proves a duplicate of a canonical
    # weight. Unknown checkpoints and already shared files stay.
    stored = stored_model_index(policy)
    copies = []
    for full, size in found:
        if (os.path.basename(full).lower(), size) in stored:
            try:
                if os.stat(full).st_nlink > 1:
                    continue
            except OSError:
                continue
            if verified_model_copy(full, size, stored):
                copies.append((full, size))
    return copies


_REVISION = re.compile(r"^(?P<name>.+)-r(?P<n>\d+)$")


def _staging_leftover(name: str) -> bool:
    return name.startswith(".staging") or name.endswith(".part")


def _holds_weights(path: Path) -> bool:
    for _directory, names, files in os.walk(path):
        if any(_staging_leftover(n) for n in names + files):
            return False
        if any(f.lower().endswith(MODEL_SUFFIXES) for f in files):
            return True
    return False


def superseded_models(policy: dict, processes: list[str], idle: float) -> list[Path]:
    """Superseded acquisition revisions and abandoned downloads in the canonical model stores on the SSD
    (`superseded_models_about` in the policy). Idle and named by no running process; never through a link."""
    if not policy.get("superseded_models") or "<process list unavailable>" in processes:
        return []
    hot = Path(policy["hot_volume"]).anchor.lower()
    found: list[Path] = []
    for store in map(Path, policy.get("model_stores", [])):
        if not on_hot_volume(store, policy) or not store.is_dir():
            continue
        children = [c for c in store.iterdir() if c.is_dir() and not is_link(c)]
        revisions: dict[str, list[tuple[int, Path]]] = {}
        for child in children:
            match = _REVISION.match(child.name)
            if match:
                revisions.setdefault(match["name"], []).append((int(match["n"]), child))
            for inner in child.iterdir():
                if _staging_leftover(inner.name) and not is_link(inner):
                    found.append(inner)
        for rows in revisions.values():
            complete = [n for n, path in rows if _holds_weights(path)]
            if complete:
                found += [path for n, path in rows if n < max(complete)]
    gone = []
    for path in found:
        if _model_managed_path(policy, path):
            continue
        named = {str(path).lower()} | ({str(path.parent).lower()} if _staging_leftover(path.name) else set())
        if any(text in proc for proc in processes for text in named):
            continue
        if path.is_dir() and (_recent(path, idle) or survey(path).links):
            continue
        if path.is_file() and path.stat().st_mtime >= idle:
            continue
        if path.parent in gone:
            continue
        gone.append(path)
    return [g for g in gone if not any(g != o and g.is_relative_to(o) for o in gone)]


def temp_orphans(policy: dict, now: float | None = None) -> list[Path]:
    """Idle entries in %TEMP% (`temp_orphans` in the policy: one rule or a list). A rule with prefixes takes the
    directories carrying one; a rule with no prefixes takes every top-level file and directory. A directory's idle
    time is the newest write in its tree, a file's its mtime; a link or junction is never listed or entered."""
    rules = policy.get("temp_orphans") or []
    rules = [rules] if isinstance(rules, dict) else rules
    now = time.time() if now is None else now
    found: dict[Path, None] = {}
    for rule in rules:
        base = Path(os.path.expandvars(rule["root"]))
        if not base.is_dir():
            continue
        cutoff = now - rule["min_idle_hours"] * 3600
        prefixes = tuple(rule.get("prefixes") or ())
        for p in base.iterdir():
            if (prefixes and not p.name.startswith(prefixes)) or is_link(p) or p in found:
                continue
            try:
                if p.is_dir():
                    stale = not _recent(p, cutoff) and survey(p).newest <= cutoff
                elif prefixes:
                    continue
                else:
                    stale = p.stat().st_mtime <= cutoff
            except OSError:
                continue
            if stale:
                found[p] = None
    return sorted(found)


def unused_local_models(policy: dict, processes: list[str], idle: float) -> list[Path]:
    """Only positively identified, idle trial/retired models; absence of a reference proves nothing."""
    return [Path(row["path"]) for row in local_model_decisions(policy, processes, idle) if row["eligible"]]


MODEL_PROVENANCE = "lss-model.json"
_REFERENCE_SUFFIXES = {".py", ".ps1", ".psm1", ".psd1", ".bat", ".cmd", ".sh", ".js", ".ts", ".json",
                       ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".txt", ".xml", ".md"}
_PRIVATE_REFERENCE = re.compile(r"(^|[./_-])(secrets?|cookies?|tokens?|credentials?|auth|authentication)([./_-]|$)")


def _model_managed_path(policy: dict, path: Path) -> bool:
    """Other hot-cleanup rules cannot bypass the local-model deletion contract, even through a parent."""
    return any(path.is_relative_to(root) or root.is_relative_to(path)
               for root in (Path(os.path.expandvars(p)) for p in policy.get("unused_local_models", {}).get("roots", [])))


def _model_reference_texts(rule: dict, repository: Path) -> list[str]:
    """Bounded name search in saved profiles and tracked scripts/configs. Read failures fail closed.

    Never open private/auth files or follow links. No repository/state/archive walk is needed.
    """
    paths = []
    for folder in rule.get("profile_dirs", []):
        directory = Path(os.path.expandvars(folder))
        if is_link(directory):
            raise OSError("linked profile directory")
        if directory.exists():
            files, _directories = _scan_tree(directory)
            paths.extend(p for p in files if p.suffix.lower() == ".json")
    result = subprocess.run(["git", "-C", str(repository), "ls-files", "-z"],
                            capture_output=True, check=True, timeout=30)
    for name in result.stdout.decode("utf-8", errors="strict").split("\0"):
        rel = Path(name)
        if name and (rel.parts[0] in {"scripts", "config", "configs"}
                     or (len(rel.parts) == 1 and rel.suffix.lower() in _REFERENCE_SUFFIXES - {".py"})):
            if rel.is_absolute() or ".." in rel.parts:
                raise OSError("invalid tracked reference path")
            paths.append(repository / rel)
    texts = []
    for path in paths:
        relative = path.as_posix().lower()
        if _PRIVATE_REFERENCE.search(relative) or any(p.lower().startswith(".env") for p in path.parts):
            continue
        texts.append(relative)
        if path.suffix.lower() not in _REFERENCE_SUFFIXES:
            continue
        if any(is_link(p) for p in (path, *path.parents)) or path.stat().st_size > 2 * 1024 * 1024:
            raise OSError("reference is linked or exceeds scan limit")
        texts.append(path.read_text(encoding="utf-8-sig").lower())
    return texts


def _model_provenance(path: Path) -> dict:
    """A complete per-file SHA256 inventory, excluding only the provenance record itself."""
    record = path / MODEL_PROVENANCE
    if is_link(record) or record.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("invalid provenance record")
    data = json.loads(record.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid provenance record")
    source, revision = data.get("source"), data.get("revision")
    if (not isinstance(source, str) or not re.fullmatch(r"https?://[^\s]+|[\w.-]+/[\w.-]+", source)
            or not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision)):
        raise ValueError("missing source or exact revision")
    files, total = data.get("files"), data.get("total_bytes")
    if not isinstance(files, dict) or not files or type(total) is not int or total < 0:
        raise ValueError("missing file inventory or total bytes")
    for name, item in files.items():
        rel = Path(name)
        if (not name or rel.is_absolute() or rel.drive or ".." in rel.parts or "\\" in name
                or rel.as_posix() != name or name == MODEL_PROVENANCE
                or not isinstance(item, dict) or type(item.get("bytes")) is not int or item["bytes"] < 0
                or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", item["sha256"])):
            raise ValueError("missing or invalid per-file size/hash")
    if sum(item["bytes"] for item in files.values()) != total:
        raise ValueError("total bytes mismatch")
    return {"source": source, "revision": revision, "status": data.get("status"), "total_bytes": total,
            "files": {name: {"bytes": item["bytes"], "sha256": item["sha256"].lower()} for name, item in files.items()}}


def local_model_decisions(policy: dict, processes: list[str], idle: float, *, repository: Path | None = None,
                          cwds=None, verify_hashes: bool = True, only: Path | None = None) -> list[dict]:
    """Report every folder, including UNKNOWN; deletion requires every guard to succeed.

    `.cache` is ordinary model content, never intrinsically regenerable. Its bytes need the same inventory.
    """
    rule = policy.get("unused_local_models")
    if not rule:
        return []
    if "min_idle_hours" in rule:
        idle = min(idle, time.time() - float(rule["min_idle_hours"]) * 3600)
    keep = {str(k).lower() for k in rule.get("keep", [])}
    refs = None
    reference_error = False
    decisions = []
    for store in (Path(os.path.expandvars(p)) for p in rule.get("roots", [])):
        if not store.is_dir() or is_link(store):
            continue
        for child in sorted(store.iterdir()):
            if not child.is_dir() or (only is not None and child != only):
                continue
            row = {"path": str(child), "eligible": False, "reason": "unknown provenance"}
            decisions.append(row)
            if is_link(child):
                row["reason"] = "linked model folder"
                continue
            try:
                row["provenance"] = provenance = _model_provenance(child)
            except (OSError, ValueError, TypeError):
                continue
            if child.name.lower() in keep:
                row["reason"] = "keep list"
            elif provenance["status"] != "retired":
                # A trial still in use or awaiting a decision is not evidence of being unneeded.
                row["reason"] = "status is not retired"
            elif "<process list unavailable>" in processes:
                row["reason"] = "process inventory unavailable"
            else:
                if cwds is None:
                    cwds = process_cwds()
                if cwds[1] or worktree_users(child, [p.lower() for p in processes], cwds[0]) or any(
                        child.name.lower() in p.lower() for p in processes):
                    row["reason"] = "running process or unavailable process directories"
                    continue
                if refs is None and not reference_error:
                    try:
                        refs = _model_reference_texts(rule, repository if repository is not None else main_root())
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                        reference_error = True
                if reference_error:
                    row["reason"] = "reference scan unavailable"
                elif any(child.name.lower() in text for text in refs):
                    row["reason"] = "saved profile/preset or tracked script/config reference"
                else:
                    try:
                        files, directories = _scan_tree(child)  # raises on links or unreadable entries
                        if any(p.stat().st_mtime >= idle for p in [child, *directories, *files]):
                            row["reason"] = "not idle beyond policy threshold"
                            continue
                        actual = {p.relative_to(child).as_posix(): p for p in files if p != child / MODEL_PROVENANCE}
                        if actual.keys() != provenance["files"].keys():
                            raise ValueError("inventory differs")
                        for name, path in actual.items():
                            before = path.stat()
                            expected = provenance["files"][name]
                            if (before.st_size != expected["bytes"]
                                    or (verify_hashes and _hash(path) != expected["sha256"])):
                                raise ValueError("content differs")
                            after = path.stat()
                            if ((after.st_size, after.st_mtime_ns, after.st_ino)
                                    != (before.st_size, before.st_mtime_ns, before.st_ino)):
                                raise ValueError("content changes during verification")
                    except (OSError, ValueError, RuntimeError):
                        row["reason"] = "unknown provenance: inventory/content cannot be verified"
                        continue
                    row.update(eligible=True, reason="verified idle trial/retired model")
    return decisions


def clean_hot(policy: dict, *, apply: bool, log=print) -> dict:
    """Delete verified retired/trial models, superseded downloads and temp orphans; receipt UNKNOWN keeps too."""
    if apply:
        require_machine_overlay(policy)
    processes = [q.replace("/", "\\") for q in process_texts()]
    if "<process list unavailable>" in processes:
        raise OSError("process inventory unavailable: cannot safely determine active reservations")
    cwds, unreadable_count = process_cwds()
    idle = time.time() - idle_threshold_hours(policy) * 3600
    models = local_model_decisions(policy, processes, idle, cwds=(cwds, unreadable_count), verify_hashes=not apply)
    kept = [row for row in models if not row["eligible"]]
    for row in kept:
        log(f"{row['path']}: kept: {row['reason']}")
    targets = [("superseded_model", p) for p in superseded_models(policy, processes, idle)]
    targets += [("unused_local_model", Path(row["path"])) for row in models if row["eligible"]]
    targets += [("temp_orphan", p) for p in temp_orphans(policy) if not _model_managed_path(policy, p)]
    rows = []
    for rule, path in targets:
        if worktree_users(path, processes, cwds):
            continue
        try:
            size = survey(path).bytes if path.is_dir() else path.stat().st_size
        except OSError:
            continue
        if not apply:
            log(f"would delete {rule} {path} ({size / 1e9:.2f} GB)")
            rows.append({"path": str(path), "bytes": size, "rule": rule})
            continue
        try:
            with storage_mutation(policy, [path]):
                model_receipt = None
                if rule == "unused_local_model":
                    # Refresh dependencies and verify every payload hash under the mutation guard.
                    fresh = local_model_decisions(policy, process_texts(), idle, cwds=process_cwds(), only=path)
                    if len(fresh) != 1 or not fresh[0]["eligible"]:
                        reason = fresh[0]["reason"] if fresh else "model folder changed"
                        kept.append({"path": str(path), "eligible": False, "reason": reason})
                        log(f"{path}: kept: {reason}")
                        continue
                    receipts = local_state(policy) / "receipts"
                    receipts.mkdir(exist_ok=True)
                    model_receipt = receipts / f"tier-local-model-{time.time_ns()}.json"
                    # Persist re-acquisition information before removing its in-folder original.
                    model_receipt.write_text(json.dumps({"action": "authorize_model_deletion", **fresh[0]}, indent=1),
                                             encoding="utf-8")
                if path.is_dir():
                    if rule == "temp_orphan":
                        moved = path.with_name(path.name + ".lss-delete")
                        os.rename(path, moved)
                        path = moved
                    _remove_tree(path)
                else:
                    os.unlink(_long(path))
            rows.append({"path": str(path), "bytes": size, "rule": rule,
                         **({"provenance_receipt": str(model_receipt)} if model_receipt else {})})
            log(f"deleted {rule} {path} ({size / 1e9:.2f} GB)")
        except OSError as exc:
            log(f"kept {path}: {exc}")
            kept.append({"path": str(path), "eligible": False, "reason": "mutation refused or failed"})
    if apply and (rows or kept):
        receipts = local_state(policy) / "receipts"
        receipts.mkdir(exist_ok=True)
        (receipts / f"tier-clean-hot-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
            {"type": "lss_storage_tier", "action": "clean_hot", "deleted": rows, "kept": kept,
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"deleted": rows, "kept": kept, "dry_run": not apply, "bytes": sum(r["bytes"] for r in rows)}


def _hash_cache(policy: dict) -> tuple[Path, dict]:
    path = local_state(policy) / "hashes.json"
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path, {}


def dedupe(policy: dict, roots: list[Path], *, apply: bool, log=print) -> dict:
    """Large files with identical bytes on one volume become hard links to one copy (oldest kept).
    Paths keep working, nothing is lost; read-only and recently written files are left alone."""
    if apply:
        require_machine_overlay(policy)
    minimum = int(policy.get("dedupe_min_mb", 64)) * 1024 * 1024
    idle = time.time() - policy["min_idle_hours"] * 3600
    by_size: dict[tuple[str, int], list[tuple[str, os.stat_result]]] = {}
    for top in roots:
        if not top.is_dir():
            continue
        for directory, names, files in os.walk(top):
            names[:] = [n for n in names if not is_link(Path(directory) / n)]
            for name in files:
                full = os.path.join(directory, name)
                try:
                    st = os.stat(full, follow_symlinks=False)
                except OSError:
                    continue
                if (st.st_size < minimum or st.st_mtime >= idle
                        or getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_READONLY):
                    continue
                by_size.setdefault((os.path.splitdrive(full)[0].lower(), st.st_size), []).append((full, st))
    cache_path, cache = _hash_cache(policy)
    saved, linked = 0, []
    for (_, size), rows in by_size.items():
        if len(rows) < 2:
            continue
        groups: dict[str, list[tuple[str, os.stat_result]]] = {}
        for full, st in rows:
            key = f"{full}|{st.st_size}|{st.st_mtime_ns}"
            digest = cache.get(key)
            if digest is None:
                try:
                    digest = cache[key] = _hash(Path(full))
                except OSError:
                    continue
            groups.setdefault(digest, []).append((full, st))
        for digest, same in groups.items():
            same.sort(key=lambda r: r[1].st_mtime)
            keep_path, keep_st = same[0]
            for full, st in same[1:]:
                if st.st_ino and st.st_ino == keep_st.st_ino and st.st_dev == keep_st.st_dev:
                    continue                                   # already the same file
                if apply:
                    temporary = full + ".lss-link-tmp"
                    try:
                        with storage_mutation(policy, [keep_path, full]):
                            os.link(keep_path, temporary)
                            os.replace(temporary, full)
                    except OSError as exc:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                        log(f"kept duplicate {full}: {exc}")
                        continue
                saved += size
                linked.append({"path": full, "same_as": keep_path, "bytes": size, "sha256": digest})
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {"linked": linked, "bytes": saved}


SCAN_SKIP = {"_internal", "site-packages", ".git", "node_modules"}


def scan_regenerable(policy: dict, root: Path, top: Path, *, max_depth: int = 8) -> tuple[list[Path], list[tuple[str, int]]]:
    """One shallow pass over directory entries only (no file is opened): package copies, disposable caches and model
    weight files. It never descends into a package, a cache, a junction or a model store, and stops at max_depth."""
    regenerable, weights = [], []
    stack = [(top, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        names = {e.name for e in entries}
        if "LSS.exe" in names and "_internal" in names and is_package_copy(Path(directory), policy, root):
            regenerable.append(Path(directory))
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    path = Path(entry.path)
                    if is_link(path):
                        continue
                    if _disposable_name(entry.name, policy):
                        regenerable.append(path)
                    elif (entry.name not in SCAN_SKIP and depth < max_depth
                          and not _canonical_model(entry.path + "\\x", policy)):
                        stack.append((path, depth + 1))
                elif entry.name.lower().endswith(MODEL_SUFFIXES):
                    weights.append((entry.path, entry.stat(follow_symlinks=False).st_size))
            except OSError:
                continue
    return regenerable, weights


def _recent(path: Path, idle: float) -> bool:
    """Cheap activity check: the directory itself or its direct entries written after `idle`."""
    try:
        if os.stat(path).st_mtime > idle:
            return True
        with os.scandir(path) as entries:
            return any(e.stat(follow_symlinks=False).st_mtime > idle for e in entries)
    except OSError:
        return True


def prune(policy: dict, root: Path, *, apply: bool, log=print) -> dict:
    """Delete regenerable directories and stray model copies under the HDD archive and development roots.
    Model duplicate candidates require SHA256 verification. Only idle content, never one a running process names, never a protected
    package. The space returned is measured on the volume, not by summing files."""
    if apply:
        require_machine_overlay(policy)
    roots = ([Path(policy["archive_root"])] if policy.get("archive_root") else [])
    roots += [Path(q) for q in policy.get("prune_roots", [])]
    roots = [r for r in roots if not any(r != o and r.is_relative_to(o) for o in roots)]
    processes = [q.replace("/", "\\") for q in process_texts()]
    if "<process list unavailable>" in processes:
        return {"dirs": 0, "models": 0, "freed_gb": 0.0}
    idle = time.time() - idle_threshold_hours(policy) * 3600
    gone, weights = [], []
    for top in roots:
        if top.is_dir():
            found, files = scan_regenerable(policy, root, top)
            gone += found
            weights += files
    named = lambda text: any(text.lower() in proc for proc in processes)   # noqa: E731
    gone = [g for g in gone if not _never(g, root, policy) and not named(str(g)) and not _recent(g, idle)]
    stored = stored_model_index(policy)
    models = []
    for full, size in weights:
        if (os.path.basename(full).lower(), size) not in stored or _canonical_model(full, policy) or named(full):
            continue
        try:
            st = os.stat(full)
        except OSError:
            continue
        if st.st_nlink == 1 and st.st_mtime < idle and verified_model_copy(full, size, stored):
            models.append((full, size))
    for path in gone:
        log(f"{'delete' if apply else 'would delete'} regenerable {path}")
    for full, size in models:
        log(f"{'delete' if apply else 'would delete'} model copy {full} ({size / 1e9:.2f} GB)")
    before = {r.anchor: free_gb(r.anchor) for r in roots if r.exists()}
    if apply:
        rows = []
        for path in gone:
            try:
                with storage_mutation(policy, [path]):
                    _remove_tree(path)
                rows.append({"path": str(path), "rule": "regenerable"})
            except OSError as exc:
                log(f"kept {path}: {exc}")
        for full, size in models:
            try:
                with storage_mutation(policy, [full]):
                    os.unlink(_long(Path(full)))
                rows.append({"path": full, "bytes": size, "rule": "model_copy_of_a_stored_model"})
            except OSError as exc:
                log(f"kept model copy {full}: {exc}")
        if rows:
            receipts = local_state(policy) / "receipts"
            receipts.mkdir(exist_ok=True)
            (receipts / f"tier-prune-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
                {"type": "lss_storage_tier", "action": "prune", "deleted": rows,
                 "protected_packages": sorted(protected_packages(policy, root)),
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ensure_ascii=False, indent=1),
                encoding="utf-8")
    shared = dedupe(policy, roots, apply=apply, log=log) if policy.get("dedupe_on_prune") else {"linked": [], "bytes": 0}
    freed = sum(free_gb(a) - b for a, b in before.items()) if apply else 0.0
    log(f"regenerable {len(gone)} dirs, model copies {sum(n for _, n in models) / 1e9:.2f} GB in {len(models)} files; "
        f"{'freed %.1f GB' % freed if apply else 'dry run'}")
    return {"dirs": len(gone), "models": len(models), "freed_gb": round(freed, 1), "linked": len(shared["linked"])}


def delete_disposables(policy: dict, paths: list[Path]) -> dict:
    """Delete regenerable directories on the SSD and write a receipt next to the lock (no HDD access)."""
    require_machine_overlay(policy)
    rows = []
    for path in paths:
        tree = survey(path)
        if tree.links:
            continue                                   # never remove through a link
        try:
            with storage_mutation(policy, [path]):
                _remove_tree(path)
            rows.append({"path": str(path), "bytes": tree.bytes, "files": tree.files})
        except OSError:
            continue
    record = {"type": "lss_storage_tier", "action": "delete_disposable", "rule": policy.get("disposable"),
              "deleted": rows, "bytes": sum(r["bytes"] for r in rows),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if rows:
        receipts = local_state(policy) / "receipts"
        receipts.mkdir(exist_ok=True)
        (receipts / f"tier-delete-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def clean_scratch(policy: dict, *, apply: bool = True, log=print) -> dict:
    """Delete, on the SSD, the bulk of finished acceptance and post-deploy runs (copied packages, models, media,
    audio, databases). Their small text Evidence (JSON, Markdown, logs) stays where it is; links are never entered,
    only left in place; nothing goes to the HDD. A run is finished once nothing in it was written for min_idle_hours."""
    if apply:
        require_machine_overlay(policy)
    import glob

    rule = policy.get("scratch_runs") or {}
    keep = tuple(s.lower() for s in rule.get("keep_suffixes", []))
    keep_bytes = float(rule.get("keep_max_mb", 20)) * 1e6
    idle_seconds = float(rule.get("min_idle_hours", 12)) * 3600
    now = time.time()
    rows = []
    for pattern in rule.get("globs", []):
        for run in sorted(Path(p) for p in glob.glob(pattern)):
            if not run.is_dir() or is_link(run):
                continue
            files, dirs, newest, stack = [], [], 0.0, [run]
            while stack:
                folder = stack.pop()
                dirs.append(folder)
                try:
                    entries = list(os.scandir(folder))
                except OSError:
                    continue
                for entry in entries:
                    path = Path(entry.path)
                    if is_link(path):
                        continue                       # a mirrored install link: never followed, never removed
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(path)
                        continue
                    newest = max(newest, info.st_mtime)        # file writes only: folder times move on cleanup
                    if not (path.suffix.lower() in keep and info.st_size <= keep_bytes):
                        files.append((path, info.st_size))
            if now - newest < idle_seconds or not files:
                continue
            size = sum(n for _, n in files)
            if not apply:
                log(f"would delete {size / 1e9:.2f} GB ({len(files)} files) in {run}")
                rows.append({"path": str(run), "bytes": size, "files": len(files)})
                continue
            gone = 0
            try:
                with storage_mutation(policy, [run]):
                    for path, n in files:
                        try:
                            os.chmod(path, stat.S_IWRITE)
                            path.unlink()
                            gone += n
                        except OSError:
                            pass
                    for folder in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
                        if folder != run:
                            with contextlib.suppress(OSError):
                                folder.rmdir()         # only empty folders; kept Evidence keeps its folder
            except OSError as exc:
                log(f"kept scratch {run}: {exc}")
                continue
            rows.append({"path": str(run), "bytes": gone, "files": len(files)})
            log(f"deleted {gone / 1e9:.2f} GB scratch ({len(files)} files) in {run}; text Evidence kept")
    record = {"type": "lss_storage_tier", "action": "delete_scratch", "rule": rule.get("globs"), "deleted": rows,
              "bytes": sum(r["bytes"] for r in rows), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if rows and apply:
        receipts = local_state(policy) / "receipts"
        receipts.mkdir(exist_ok=True)
        (receipts / f"tier-delete-scratch-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def demote(policy: dict, root: Path, *, apply: bool, until_free_gb: float | None = None,
           completed_only: bool = False, log=print) -> list[dict]:
    if apply:
        require_machine_overlay(policy)
    goal = policy["target_free_gb"] if until_free_gb is None else until_free_gb
    chosen = [c for c in candidates(policy, root) if c.eligible
              and (not completed_only or c.reason.startswith("explicit output completed"))
              and (not policy.get('fast_volumes') or on_volume(Path(c.path), policy['hot_volume']))]
    done = []
    if not apply:
        need = max(0.0, goal - free_gb(policy["hot_volume"]))
        total = 0.0
        for c in chosen:
            if total >= need:
                break
            total += c.gb
            log(f"would move {c.gb:7.2f} GB idle {c.idle_hours:7.1f} h  {c.path}")
        log(f"free {free_gb(policy['hot_volume']):.1f} GB, goal {goal} GB, would move {total:.1f} GB")
        return done
    state = local_state(policy)
    journal = state / "journal.json"
    if not chosen and not journal.exists():
        return done                                   # nothing to do: the HDD is not touched
    background_priority()
    with Lock(state / ".mover.lock"):
        if journal.exists():                          # the previous mover was interrupted mid-move
            recover(policy, log)
            journal.unlink()
        for c in chosen:
            if free_gb(policy["hot_volume"]) >= goal:
                break
            source = Path(c.path)
            try:
                gone = _regenerable_roots(policy, root, source)
                if gone:
                    deleted = delete_disposables(policy, gone)
                    log(f"deleted {deleted['bytes'] / 1e9:.2f} GB regenerable ({len(deleted['deleted'])} dirs) in {c.path}")
                    if free_gb(policy["hot_volume"]) >= goal:
                        break
                if not policy.get("move_to_hdd", True):
                    continue                          # deletion only: nothing is carried to the HDD
                if not source.exists() or survey(source).bytes / 1e9 < c.minimum_gb:
                    continue                          # what is left is not worth an HDD trip
                journal.write_text(json.dumps({"source": c.path}), encoding="utf-8")
                record = demote_one(policy, source)
                journal.unlink()
                done.append(record)
                log(f"moved {record['bytes'] / 1e9:.2f} GB in {record['seconds']} s: {c.path}")
            except Exception as exc:  # noqa: BLE001 - one failure keeps the original and moves on
                log(f"kept {c.path}: {exc}")          # the journal stays: the next run removes the marked partial copy
    return done


@dataclass
class IdleWorktree:
    path: str
    branch: str
    head: str
    gb: float
    idle_hours: float
    eligible: bool
    reason: str


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def worktree_entries(root: Path) -> list[dict]:
    """Registered worktrees from `git worktree list --porcelain`: worktree, HEAD, branch (without refs/heads/),
    locked, prunable, bare (absent keys are simply missing)."""
    entries, entry = [], {}
    for line in _git("-C", str(root), "worktree", "list", "--porcelain").splitlines() + [""]:
        if line:
            key, _, value = line.partition(" ")
            entry[key] = value or True
            continue
        if entry:
            entry["branch"] = str(entry.get("branch", "")).replace("refs/heads/", "")
            entries.append(entry)
            entry = {}
    return entries


_SHA = re.compile(r"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])")
RELEASE_KEYS = ("candidate_code_commit", "release_id", "frozen_package", "frozen_identity", "runtime_manifest",
                "deployed_executable", "rollback")


def protected_build_ids(policy: dict, root: Path) -> set[str]:
    """Commit ids (7-40 hex) whose build checkout must stay: the deployed code and its rollback target named by
    .state/releases/current-product.json, and every FROZEN-<id> candidate under the frozen roots that has no closing
    VERDICT/DEPLOYED/HELD/SUPERSEDED file yet."""
    found: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            found.update(_SHA.findall(value.lower()))

    release_record = dev_state_root(root) / "releases" / "current-product.json"
    try:
        record = json.loads(release_record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {}
    for key in RELEASE_KEYS:
        walk(record.get(key))
    for release_root in policy.get("frozen_roots", []):
        base = Path(release_root)
        if not base.is_dir():
            continue
        for frozen in base.glob("FROZEN-*"):
            names = [q.name.upper() for q in frozen.iterdir()] if frozen.is_dir() else []
            if not any(n.startswith(("VERDICT", "DEPLOYED", "HELD", "SUPERSEDED")) for n in names):
                found.update(_SHA.findall(frozen.name[7:].lower()))
    return found


def last_commit_hours(path: Path) -> float | None:
    """Hours since the newest commit reachable from the worktree's HEAD was made; None if unreadable."""
    out = _git("-C", str(path), "log", "-1", "--format=%ct", "HEAD").strip()
    return (time.time() - int(out)) / 3600 if out.isdigit() else None


def _inside(path: Path, other: Path) -> bool:
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except (ValueError, OSError):
        return False


def protection_reason(path: Path, head: str, protected: set[str], packages: set[str]) -> str | None:
    """Why a worktree must stay where and as it is (frozen candidate, deployed/rollback/pending build, protected
    package), or None."""
    text = str(path).lower().rstrip("\\")
    if any(part.startswith("frozen-") for part in map(str.lower, path.parts)):
        return "inside a frozen candidate"
    if any(head.lower().startswith(p) for p in protected) or any(
            f"-{p}" in path.name.lower() for p in protected if path.name.lower().startswith(("build-", "frozen-"))):
        return "deployed, rollback or pending frozen candidate"
    if any(q == text or q.startswith(text + "\\") or text.startswith(q + "\\") for q in packages):
        return "holds a protected package"
    return None


def idle_worktrees(policy: dict, root: Path, *, processes: list[str] | None = None, all_volumes: bool = False,
                   commit_hours: float | None = None, cwd: Path | None = None) -> list[IdleWorktree]:
    """Registered worktrees that could be retired: idle for `worktree_idle_hours` (a worktree with uncommitted or
    untracked work needs `dirty_worktree_idle_hours`, and `retire_worktree` archives that work first), not
    named by a running process, not locked, not the current session's directory, not a protected build, and (when
    detached) their HEAD kept by a branch. The branch keeps every commit and the archive keeps everything else. By default only worktrees on the hot volume are considered (the scheduled retirement); `all_volumes`
    adds the HDD ones and `commit_hours` requires the HEAD commit itself to be older than that too (`--retire`)."""
    import storage_budget
    reserved_paths = storage_budget.protected_paths(policy)
    processes = [p.replace("/", "\\") for p in (process_texts() if processes is None else processes)]
    limit = policy.get("worktree_idle_hours", 72) if commit_hours is None else commit_hours
    hot = Path(policy["hot_volume"]).anchor.lower()
    here = Path.cwd() if cwd is None else cwd
    protected = protected_build_ids(policy, root)
    packages = {q for q in protected_packages(policy, root)}

    def judge(entry: dict) -> IdleWorktree | None:
        path, head, branch = Path(entry["worktree"]), str(entry.get("HEAD", "")), entry["branch"]
        if (path.resolve() == root.resolve() or entry.get("bare") or not path.is_dir()
                or (not all_volumes and not str(path).lower().startswith(hot))):
            return None                               # the main checkout; worktrees already on the HDD (default)
        reason = "idle"
        idle = worktree_idle_hours(path)
        text = str(path).lower().rstrip("\\")
        if entry.get("locked"):
            reason = "locked"
        elif any(_inside(path, Path(p)) or _inside(Path(p), path) for p in reserved_paths):
            reason = "protected by an active storage reservation"
        elif _inside(here, path) or _inside(Path(__file__), path):
            reason = "current session directory"
        elif protection_reason(path, head, protected, packages):
            reason = protection_reason(path, head, protected, packages)
        elif "<process list unavailable>" in processes or any(text in p for p in processes):
            reason = "named by a running process"
        else:
            if commit_hours is not None:
                commit_age = last_commit_hours(path)
                if commit_age is None or commit_age < commit_hours:
                    return IdleWorktree(str(path), branch, head, 0.0, commit_age or 0.0, False,
                                        f"commit {commit_age:.0f} h ago" if commit_age is not None else "HEAD unreadable")
            tree = survey(path)
            newest = (time.time() - tree.newest) / 3600 if tree.newest else float("inf")
            idle = min(idle if idle is not None else float("inf"), newest)
            gb = round(tree.bytes / 1e9, 3)
            if idle < limit:
                reason = f"active {idle:.0f} h ago"
            elif _git("-C", str(path), "status", "--porcelain", "--untracked-files=all").strip():
                dirty_limit = policy.get("dirty_worktree_idle_hours")
                if dirty_limit is not None and idle >= dirty_limit:
                    reason = "idle; uncommitted work archived"
                else:
                    reason = "uncommitted or untracked work"
            if reason in ("idle", "idle; uncommitted work archived") and not branch and not any(
                    b for b in _git("-C", str(root), "branch", "--contains", head, "--format=%(refname:short)").split()):
                reason = "detached HEAD not kept by any branch"
            return IdleWorktree(str(path), branch, head, gb, round(idle, 1), reason in ("idle", "idle; uncommitted work archived"), reason)
        return IdleWorktree(str(path), branch, head, 0.0, round(idle or 0.0, 1), False, reason)

    entries = worktree_entries(root)
    with ThreadPoolExecutor(max_workers=8 if all_volumes else 1) as pool:
        found = [w for w in pool.map(judge, entries) if w is not None]
    return sorted(found, key=lambda w: -w.idle_hours)


def retire_worktree(policy: dict, root: Path, worktree: IdleWorktree) -> dict:
    require_machine_overlay(policy)
    with storage_mutation(policy, [worktree.path]):
        return _retire_worktree(policy, root, worktree)


def _retire_worktree(policy: dict, root: Path, worktree: IdleWorktree) -> dict:
    """Archive what git does not keep (ignored results, and for a dirty worktree its untracked files, its modified
    tracked files and `UNCOMMITTED.patch`; verified copy; regenerable junk is dropped), unhook links
    (the link only, never its target), then remove the checkout and its git registration. The branch stays."""
    require_machine_overlay(policy)
    path = Path(worktree.path)
    links = [Path(p) for p in survey(path).links]
    link_targets = {str(p): os.path.realpath(p) for p in links}
    listed = _git("-C", str(path), "ls-files", "--others", "--ignored", "--exclude-standard", "-z").split("\0")
    under_link = lambda rel: any((path / rel).is_relative_to(link) for link in links)   # noqa: E731
    keep, dropped = [], 0
    for rel in filter(None, listed):
        if under_link(rel) or not (path / rel).is_file():
            continue
        parts = Path(rel).parts
        if (any(_disposable_name(part, policy) for part in parts[:-1]) or _disposable_name(parts[-1], policy)
                or parts[0] in policy.get("regenerable_worktree_tops", [])
                or parts[-1].lower().endswith(MODEL_SUFFIXES)
                or any(is_package_copy(path.joinpath(*parts[:i]), policy, root) for i in range(1, len(parts)))):
            dropped += 1
            continue
        keep.append(rel)
    uncommitted, patch = [], None
    if _git("-C", str(path), "status", "--porcelain", "--untracked-files=all").strip():
        # work git does not keep: untracked files and tracked files that differ from HEAD, whole, plus the diff
        changed = (_git("-C", str(path), "ls-files", "--others", "--exclude-standard", "-z").split("\0")
                   + _git("-C", str(path), "diff", "--name-only", "-z", "HEAD").split("\0"))
        for rel in dict.fromkeys(filter(None, changed)):
            parts = Path(rel).parts
            if (under_link(rel) or not (path / rel).is_file() or rel in keep
                    or any(_disposable_name(part, policy) for part in parts)):
                continue
            uncommitted.append(rel)
        patch = subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--binary"], capture_output=True).stdout
    archive = Path(policy["archive_root"]) / "worktrees" / f"{path.name}-{time.strftime('%Y%m%dT%H%M%S')}"
    rows = {}
    for rel in keep + uncommitted:
        size, sha = _copy_file(path / rel, archive / rel)
        rows[rel] = {"bytes": size, "sha256": sha}
    if patch:
        archive.mkdir(parents=True, exist_ok=True)
        (archive / PATCH).write_bytes(patch)
        rows[PATCH] = {"bytes": len(patch), "sha256": hashlib.sha256(patch).hexdigest()}
    bad = [rel for rel, row in rows.items()
           if os.path.getsize(_long(archive / rel)) != row["bytes"] or _hash(archive / rel) != row["sha256"]]
    if bad:
        raise RuntimeError(f"archive verification failed for {len(bad)} files; worktree kept: {bad[:3]}")
    record = {"type": "lss_storage_tier", "action": "retire_worktree", "source": str(path), "branch": worktree.branch,
              "head": worktree.head, "archive": str(archive) if rows else None, "files": rows,
              "file_count": len(rows), "bytes": sum(r["bytes"] for r in rows.values()),
              "regenerable_dropped": dropped, "links_unhooked": link_targets,
              "uncommitted": uncommitted, "patch": PATCH if patch else None,
              "restore": f"git worktree add {path} {worktree.branch or worktree.head}; copy the archive back"
                         + ("; `git apply` UNCOMMITTED.patch (or copy the listed uncommitted files over the checkout)"
                            if patch else ""),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if rows:
        (archive / MANIFEST).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    for link in sorted(links, key=lambda p: -len(p.parts)):
        try:
            os.rmdir(link)                    # a junction/dir symlink: removes the link only, never its target
        except NotADirectoryError:
            os.unlink(link)
    if survey(path).links:
        raise RuntimeError("links remain inside the worktree; not removed")
    result = subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        # git cannot delete paths longer than MAX_PATH; everything it does not keep is archived and verified and
        # no link remains, so finish with a long-path delete and drop the registration.
        if path.exists():
            _remove_tree(path)
        prune = subprocess.run(["git", "-C", str(root), "worktree", "prune"], capture_output=True, text=True)
        if path.exists() or prune.returncode != 0:
            raise RuntimeError(f"git worktree remove failed: {result.stderr.strip()}")
        record["removal"] = "long-path delete + git worktree prune"
    record["status"] = "WORKTREE_RETIRED_BRANCH_KEPT"
    record["receipt"] = str(_receipt(policy, record)) if rows else None
    if not rows:
        receipts = local_state(policy) / "receipts"
        receipts.mkdir(exist_ok=True)
        (receipts / f"tier-retire-{time.strftime('%Y%m%dT%H%M%S')}-{path.name}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        try:                                             # the storage journal also records a retirement that archived nothing
            journal = Path(policy["archive_root"]) / "index.jsonl"
            journal.parent.mkdir(parents=True, exist_ok=True)
            with open(journal, "a", encoding="utf-8") as index:
                index.write(json.dumps({k: v for k, v in record.items() if k != "files"}, ensure_ascii=False) + "\n")
        except OSError:
            pass
    return record


# --- Where worktrees live: work on the SSD, keep on the HDD -------------------------------------------------------------
# A lane's tests, imports, git and pytest temp roots all hit the volume its worktree is on; on the USB HDD that makes
# every run wait. `worktrees --report` shows where each registered worktree physically is and who uses it; `promote`
# moves an HDD worktree to the SSD with a verified copy and leaves its old path working.

def worktree_promote_settings(policy: dict) -> dict:
    rule = policy.get("worktree_promote") or {}
    return {"root": os.path.expandvars(rule.get("root", "%LOCALAPPDATA%/LSS-dev/worktrees")),
            "min_free_after_gb": float(rule.get("min_free_after_gb", policy.get("pressure_free_gb", 80))),
            "active_commit_hours": float(rule.get("active_commit_hours", 48)),
            "regenerable_dirs": list(rule.get("regenerable_dirs", ["__pycache__", ".pytest_cache", "EBWebView"])),
            "regenerable_dir_prefixes": list(rule.get("regenerable_dir_prefixes", ["pytest-of-"]))}


def on_hot_volume(path: Path, policy: dict) -> bool:
    """Whether the bytes behind path (junctions resolved) are on the hot volume."""
    try:
        real = os.path.realpath(path)
    except OSError:
        real = str(path)
    return any(on_volume(Path(real), volume) for volume in policy.get('fast_volumes', [policy['hot_volume']]))


def on_volume(path: Path, volume: str) -> bool:
    return Path(os.path.realpath(path)).anchor.casefold() == Path(volume).anchor.casefold()


def process_cwds() -> tuple[list[tuple[int, str, str]], int]:
    """(pid, image name, current directory) of the running processes, and how many could not be read.

    psutil is not in the environment, so the directory is read the way psutil does it: the process's PEB ->
    RTL_USER_PROCESS_PARAMETERS.CurrentDirectory through NtQueryInformationProcess + ReadProcessMemory (64-bit
    processes of this user; protected, elevated and 32-bit ones are counted as unreadable). Open file handles are not
    enumerated here; `promote --apply` proves nobody holds files by renaming the directory (Windows refuses)."""
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        return [], 0
    from ctypes import wintypes

    k32, ntdll = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("ntdll")
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_size_t)]
    k32.IsWow64Process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    ntdll.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                                                ctypes.POINTER(ctypes.c_ulong)]

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", ctypes.c_long), ("flags", wintypes.DWORD),
                    ("name", ctypes.c_wchar * 260)]

    class Basic(ctypes.Structure):
        _fields_ = [("exit", ctypes.c_void_p), ("peb", ctypes.c_void_p), ("affinity", ctypes.c_void_p),
                    ("priority", ctypes.c_void_p), ("pid", ctypes.c_void_p), ("parent", ctypes.c_void_p)]

    def read(handle, address: int, size: int) -> bytes | None:
        buffer, got = ctypes.create_string_buffer(size), ctypes.c_size_t()
        ok = k32.ReadProcessMemory(handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(got))
        return buffer.raw[:size] if ok and got.value == size else None

    found, unreadable = [], 0
    snapshot = k32.CreateToolhelp32Snapshot(0x2, 0)
    entry = Entry()
    entry.dwSize = ctypes.sizeof(Entry)
    more = k32.Process32FirstW(snapshot, ctypes.byref(entry))
    while more:
        pid, name = int(entry.pid), entry.name
        more = k32.Process32NextW(snapshot, ctypes.byref(entry))
        if pid in (0, 4):
            continue
        handle = k32.OpenProcess(0x0400 | 0x0010, False, pid)     # query information + vm read
        if not handle:
            unreadable += 1
            continue
        try:
            wow = wintypes.BOOL()
            info = Basic()
            raw = None
            if (k32.IsWow64Process(handle, ctypes.byref(wow)) and not wow.value
                    and ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(info), ctypes.sizeof(info), None) == 0
                    and info.peb):
                params = read(handle, info.peb + 0x20, 8)                      # PEB.ProcessParameters
                if params:
                    unicode_string = read(handle, int.from_bytes(params, "little") + 0x38, 16)   # CurrentDirectory.DosPath
                    if unicode_string:
                        length = int.from_bytes(unicode_string[0:2], "little")
                        buffer = int.from_bytes(unicode_string[8:16], "little")
                        raw = read(handle, buffer, length) if length and buffer else b""
            if raw is None:
                unreadable += 1
            elif raw:
                text = raw.decode("utf-16-le", errors="replace")
                found.append((pid, name, text.rstrip("\\") or text))
        finally:
            k32.CloseHandle(handle)
    k32.CloseHandle(snapshot)
    return found, unreadable


def _contains(directory: str, other: str) -> bool:
    directory, other = directory.lower().replace("/", "\\").rstrip("\\"), other.lower().replace("/", "\\").rstrip("\\")
    return other == directory or other.startswith(directory + "\\")


def worktree_users(path: Path, processes: list[str], cwds: list[tuple[int, str, str]]) -> list[str]:
    """Processes using a worktree: a current directory inside it, or its path in their executable or command line.
    Movers (this script) are not users, so a `promote <path>` call does not block itself."""
    users = [f"cwd {name} pid {pid}" for pid, name, cwd in cwds if _contains(str(path), cwd)]
    text = str(path).lower().replace("/", "\\").rstrip("\\")
    for line in processes:
        line = line.replace("/", "\\")
        if text in line and "storage_tiers.py" not in line and "<process list unavailable>" not in line:
            users.append("names it: " + line.split("\t")[0].rsplit("\\", 1)[-1])
    return users


def worktree_placement(policy: dict, root: Path, *, processes: list[str] | None = None,
                       cwds: tuple[list[tuple[int, str, str]], int] | None = None) -> list[dict]:
    """Every registered worktree with its physical volume (junctions resolved), hot or not, age of its last commit
    and the processes using it."""
    processes = [p.replace("/", "\\") for p in (process_texts() if processes is None else processes)]
    cwd_rows, unreadable = process_cwds() if cwds is None else cwds
    main = root.resolve()

    def row(entry: dict) -> dict:
        path = Path(entry["worktree"])
        physical = Path(os.path.realpath(path))
        is_main = path.resolve() == main
        age = last_commit_hours(path) if path.is_dir() else None
        users = [] if is_main or not path.is_dir() else worktree_users(path, processes, cwd_rows)
        if physical != path and not is_main and path.is_dir():
            users += worktree_users(physical, processes, cwd_rows)
        return {"path": str(path), "branch": entry["branch"], "head": str(entry.get("HEAD", "")), "main": is_main,
                "physical": str(physical), "drive": os.path.splitdrive(str(physical))[0].upper() or "?",
                "hot": on_hot_volume(path, policy), "junction": is_link(path), "missing": not path.is_dir(),
                "commit_hours": None if age is None else round(age, 1), "users": users,
                "unreadable_processes": unreadable}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(row, worktree_entries(root)))


def promoted_name(path: Path) -> str:
    return f"{path.parent.name}-{path.name}" if path.name.lower() == "src" else path.name


def promote_worktree(policy: dict, root: Path, path: Path, *, apply: bool = False, log=print,
                     processes: list[str] | None = None,
                     cwds: tuple[list[tuple[int, str, str]], int] | None = None) -> dict:
    """Move a worktree from the HDD to the SSD (`worktree_promote` in the policy), dry run unless apply.

    The copy is verified (SHA-256 while reading the source, read back and compared; pytest/Python caches and WebView
    profiles are regenerable and left behind; junctions inside, such as the shared .venv, are recreated pointing at the
    same targets). Then `git worktree repair` points git's registration at the new path, `git status` must show the
    same HEAD and the same changes there as before, and only then is the old directory renamed away (Windows
    refuses while any process holds a file in it or has it as its directory), a junction left at the OLD path so
    everything that still names it keeps working, and the renamed original deleted. `git worktree list` therefore
    shows the real SSD path and the HDD path stays valid as a junction. Refused (nothing touched) when the path is
    not a registered worktree, is the main checkout, is already on the SSD, is a protected/frozen/build checkout, a
    process uses it, or C: free after the move would fall below `min_free_after_gb`."""
    if apply:
        require_machine_overlay(policy)
    settings = worktree_promote_settings(policy)
    path = Path(path)
    entries = {str(Path(e["worktree"])).lower().rstrip("\\"): e for e in worktree_entries(root)}
    entry = entries.get(str(path).lower().rstrip("\\")) or entries.get(os.path.realpath(path).lower().rstrip("\\"))
    refusals: list[str] = []
    record: dict = {"type": "lss_storage_tier", "action": "promote_worktree", "source": str(path),
                    "status": "REFUSED", "refusals": refusals}
    if entry is None:
        refusals.append("not a registered worktree")
        return record
    path = Path(entry["worktree"])
    record["source"] = str(path)
    head = str(entry.get("HEAD", ""))
    if path.resolve() == root.resolve() or entry.get("bare"):
        refusals.append("the main checkout never moves")
    if not path.is_dir():
        refusals.append("worktree directory is missing")
    elif on_hot_volume(path, policy):
        refusals.append("already on the SSD")
    if entry.get("locked"):
        refusals.append("locked")
    if _is_build_checkout(entry):
        refusals.append("detached build checkout (disposable, not a lane)")
    protection = protection_reason(path, head, protected_build_ids(policy, root), protected_packages(policy, root))
    if protection:
        refusals.append(protection)
    destination = Path(settings["root"]) / promoted_name(path)
    if destination.exists():
        refusals.append(f"destination exists: {destination}")
    if refusals:
        return record
    processes = [p.replace("/", "\\") for p in (process_texts() if processes is None else processes)]
    cwd_rows, unreadable = process_cwds() if cwds is None else cwds
    users = worktree_users(path, processes, cwd_rows)
    if "<process list unavailable>" in processes:
        users.append("process list unavailable")
    if users:
        refusals.append("in use: " + "; ".join(users[:5]))
    scan = _promotion_scan(policy, path)
    files, links = scan["files"], scan["links"]
    size = sum(os.path.getsize(_long(f)) for f in files)
    free_before = free_gb(policy["hot_volume"])
    free_after = free_before - size / 1e9
    record.update(destination=str(destination), bytes=size, file_count=len(files),
                  links={str(link): os.path.realpath(link) for link in links},
                  free_gb_before=round(free_before, 2), free_gb_after=round(free_after, 2),
                  min_free_after_gb=settings["min_free_after_gb"], process_cwds_unreadable=unreadable)
    if free_after < settings["min_free_after_gb"]:
        refusals.append(f"C: free would fall to {free_after:.1f} GB, below the floor {settings['min_free_after_gb']} GB")
    if refusals:
        return record
    if not apply:
        record["status"] = "DRY_RUN_WOULD_PROMOTE"
        return record
    background_priority()
    with Lock(local_state(policy) / ".mover.lock"):
        with storage_mutation(policy, [path, destination]):
            return _promote_worktree_apply(policy, root, path, entry, destination, links, record, log)


def _fresh_tracked_dirs(path: Path) -> set[str]:
    _TRACKED.pop(str(path).lower(), None)
    return tracked_dirs(path)


def _promotion_scan(policy: dict, path: Path) -> dict:
    """What a promotion copies: files, directories and the links inside (recreated, not copied). Only DIRECTORIES in
    the exact regenerable set (`regenerable_dirs`, `regenerable_dir_prefixes`) are left behind, and only when git
    ignores them and tracks nothing inside; a file is never skipped by its name."""
    cfg = worktree_promote_settings(policy)
    tracked = _fresh_tracked_dirs(path)
    found: list[Path] = []

    def regenerable(p: Path, is_dir: bool) -> bool:
        return (is_dir and (p.name in cfg["regenerable_dirs"] or p.name.startswith(tuple(cfg["regenerable_dir_prefixes"])))
                and str(p).lower() not in tracked)

    def first(p: Path, is_dir: bool) -> bool:
        if regenerable(p, is_dir):
            found.append(p)
            return True
        return False

    links: list[Path] = []
    files, dirs = _scan_tree(path, links=links, skip=first)
    skipped = {str(p.relative_to(path)) for p in found}
    if found:
        listing = "\0".join(p.relative_to(path).as_posix() for p in found) + "\0"
        out = subprocess.run(["git", "-C", str(path), "check-ignore", "-z", "--stdin"], input=listing.encode("utf-8"),
                             capture_output=True)
        ignored = ({str(Path(n)) for n in out.stdout.decode("utf-8", errors="replace").split("\0") if n}
                   if out.returncode == 0 else set())
        if ignored != skipped:               # an untracked, not-ignored cache would change `git status`: copy it
            skipped = ignored
            links = []
            files, dirs = _scan_tree(path, links=links, skip=lambda p, d: d and str(p.relative_to(path)) in ignored)
    return {"files": files, "dirs": dirs, "links": links, "skipped": skipped}


def _snapshot(root: Path, scan: dict) -> dict:
    files = {}
    for f in scan["files"]:
        st = os.stat(_long(f))
        files[str(f.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return {"files": files, "dirs": {str(d.relative_to(root)) for d in scan["dirs"]},
            "links": {str(link.relative_to(root)) for link in scan["links"]}}


def _restore_registration(root: Path, path: Path) -> None:
    """Point git's registration of the worktree back at `path` and prove it; raises when it cannot be proven."""
    repaired = subprocess.run(["git", "-C", str(root), "worktree", "repair", str(path)], capture_output=True, text=True)
    registered = any(Path(e["worktree"]).resolve() == path.resolve() for e in worktree_entries(root))
    if repaired.returncode != 0 or not registered:
        raise RuntimeError(f"git registration was not restored to {path} ({repaired.stderr.strip()[:200]}); "
                           "the copy is kept for a manual `git worktree repair`")


def _discard_copy(root: Path, path: Path, destination: Path, staging: Path, links: list[Path]) -> None:
    """Undo a promotion that did not complete: registration back at the original (proven first), then the copy goes."""
    _restore_registration(root, path)
    for leftover in (destination, staging):
        if leftover.exists():
            for link in (leftover / item.relative_to(path) for item in links):
                if is_link(link):
                    os.rmdir(link)
            _remove_tree(leftover)


def _promote_worktree_apply(policy: dict, root: Path, path: Path, entry: dict, destination: Path, links: list[Path],
                            record: dict, log) -> dict:
    require_machine_overlay(policy)
    started = time.time()
    probe = path.with_name(path.name + ".lss-probe")
    try:                                   # Windows refuses to rename a directory a process holds files in or runs from
        os.rename(path, probe)
        os.rename(probe, path)
    except OSError as exc:
        if probe.exists() and not path.exists():
            os.rename(probe, path)
        record["refusals"].append(f"a process holds files under it ({exc})")
        return record
    scan = _promotion_scan(policy, path)           # the state that gets copied, fixed here
    before = _snapshot(path, scan)
    links = scan["links"]
    skipped = scan["skipped"]
    status_before = _git("-C", str(path), "status", "--porcelain=v1", "--untracked-files=all")
    head_before = _git("-C", str(path), "rev-parse", "HEAD").strip()
    staging = destination.with_name(destination.name + PARTIAL_SUFFIX)
    staging.mkdir(parents=True)
    (staging / PARTIAL).write_text(str(path), encoding="utf-8")
    try:
        rows = copy_verified(path, staging, allow_existing=True, links=[],
                             skip=lambda p, d: d and str(p.relative_to(path)) in skipped)
        rows.pop(PARTIAL, None)
        _replace_with_retry(staging, destination)
        (destination / PARTIAL).unlink()
        for link in links:
            make_junction(destination / link.relative_to(path), Path(os.path.realpath(link)))
        repaired = subprocess.run(["git", "-C", str(root), "worktree", "repair", str(destination)],
                                  capture_output=True, text=True)
        status_after = _git("-C", str(destination), "status", "--porcelain=v1", "--untracked-files=all")
        head_after = _git("-C", str(destination), "rev-parse", "HEAD").strip()
        registered = any(Path(e["worktree"]).resolve() == destination.resolve() for e in worktree_entries(root))
        if repaired.returncode != 0 or not registered or head_after != head_before or status_after != status_before:
            raise RuntimeError("the copy does not behave like the original under git (HEAD or status differs, or the "
                               f"registration was not repaired: {repaired.stderr.strip()[:200]}); original kept")
    except BaseException:
        _discard_copy(root, path, destination, staging, links)
        raise
    gone = path.with_name(path.name + ".lss-promoted-" + time.strftime("%Y%m%dT%H%M%S"))
    try:
        os.rename(path, gone)              # atomic proof that nothing holds it; the old path names nothing for a moment
    except BaseException:
        _discard_copy(root, path, destination, staging, links)
        raise
    try:
        make_junction(path, destination)
    except BaseException:
        os.rename(gone, path)
        _discard_copy(root, path, destination, staging, links)
        raise
    after_links: list[Path] = []
    after_files, after_dirs = _scan_tree(gone, links=after_links,
                                         skip=lambda p, d: d and str(p.relative_to(gone)) in skipped)
    after = _snapshot(gone, {"files": after_files, "dirs": after_dirs, "links": after_links})
    changed = sorted({k for k in before["files"].keys() | after["files"].keys() if before["files"].get(k) != after["files"].get(k)}
                     | (before["dirs"] ^ after["dirs"]) | (before["links"] ^ after["links"]))
    leftover = None
    if changed:                             # something wrote to the original while it was copied: never delete it
        leftover = f"{gone}: changed during the copy ({len(changed)} paths, e.g. {changed[:3]}); original kept"
        log(f"KEPT original {leftover}")
    else:
        for link in links:                  # unhook inner links first: only the link is removed, never its target
            inner = gone / link.relative_to(path)
            if is_link(inner):
                os.rmdir(inner)
        try:
            _remove_tree(gone)
        except OSError as exc:
            leftover = f"{gone}: {exc}"
    record.update(status="WORKTREE_PROMOTED_ORIGINAL_CHANGED_KEPT" if changed else "WORKTREE_PROMOTED_VERIFIED_LEGACY_JUNCTION",
                  files=rows, changed_during_copy=changed[:50],
                  bytes=sum(r["bytes"] for r in rows.values()), file_count=len(rows),
                  legacy_junction=str(path), registered_path=str(destination), leftover_original=leftover,
                  seconds=round(time.time() - started, 1), free_gb_after=round(free_gb(policy["hot_volume"]), 2),
                  head=head_before, branch=entry["branch"],
                  restore=f"remove the junction {path}, copy {destination} back to {path}, "
                          f"`git worktree repair {path}`, remove {destination}",
                  utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    record["receipt"] = str(_receipt(policy, record))
    log(f"promoted {path} -> {destination}: {record['bytes'] / 1e9:.2f} GB in {record['seconds']} s")
    return record


def running_process_names() -> set[str]:
    if os.name != "nt":
        return set()
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=30,
                             encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {line.split('","')[0].strip('"').lower() for line in out.splitlines() if line.startswith('"')}


def recent_git_hours(root: Path) -> float:
    """Hours since git last changed anything in the repo or any of its worktrees (index, HEAD, reflog)."""
    git = root / ".git"
    stamps = [p for p in (git / "index", git / "HEAD", git / "logs" / "HEAD")]
    worktrees = git / "worktrees"
    if worktrees.is_dir():
        for entry in os.scandir(worktrees):
            stamps += [Path(entry.path) / "index", Path(entry.path) / "HEAD", Path(entry.path) / "logs" / "HEAD"]
    times = [p.stat().st_mtime for p in stamps if p.exists()]
    return (time.time() - max(times)) / 3600 if times else float("inf")


def developing(policy: dict, root: Path, *, names: set[str] | None = None) -> bool:
    rule = policy.get("run_only_while_developing")
    if not rule:
        return True
    names = running_process_names() if names is None else names
    agent = any(p.lower() in names for p in rule.get("processes", []))
    return agent and recent_git_hours(root) <= rule.get("recent_git_hours", 3)


def _stamp(policy: dict, name: str) -> Path:
    return local_state(policy) / f"last-cleanup-{name}.txt"


def cleanups_due(policy: dict) -> list[dict]:
    """`user_cleanups` rules whose tool is installed (`exists`) and whose last run is older than `every_hours`."""
    due = []
    for rule in policy.get("user_cleanups", []):
        if not Path(os.path.expandvars(rule["exists"])).exists():
            continue
        try:
            last = float(_stamp(policy, rule["name"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            last = 0.0
        interval = rule["every_hours"]
        if "pressure_every_hours" in rule and free_gb(policy["hot_volume"]) < policy["start_free_gb"]:
            interval = rule["pressure_every_hours"]
        if time.time() - last > interval * 3600:
            due.append(rule)
    return due


def _aged_entries(rule: dict, now: float) -> list[Path]:
    base = Path(os.path.expandvars(rule["exists"]))
    if is_link(base) or any(is_link(p) for p in base.parents):
        return []
    cutoff = now - rule["older_than_days"] * 86400
    found = []
    for entry in base.glob(rule.get("glob", "*")):
        if is_link(entry):
            continue
        tree = survey(entry) if entry.is_dir() else None
        if tree and tree.links:
            continue
        newest = tree.newest if tree else entry.stat().st_mtime
        if newest and newest < cutoff:
            found.append(entry)
    return found


def run_cleanup(policy: dict, rule: dict, *, log=print) -> float:
    """One `user_cleanups` rule: the tool's own cleanup command (old versions, download caches; the tool keeps what
    is current) or, for `older_than_days`, deletion of entries not written for that long (crash dumps, error
    reports). Hidden; only success advances the interval stamp; failure is receipted and retried next pass.
    Returns the GB the volume gained."""
    require_machine_overlay(policy)
    before = free_gb(policy["hot_volume"])
    error = None
    deleted = []
    if "older_than_days" in rule:
        kept = 0
        entries = _aged_entries(rule, time.time())
        for entry in entries:
            try:
                with storage_mutation(policy, [entry]):
                    _remove_tree(entry) if entry.is_dir() else os.unlink(_long(entry))
                deleted.append(str(entry))
            except OSError:
                kept += 1
        outcome = f"{len(entries) - kept} deleted, {kept} kept"
    else:
        flags = 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW
        environment = {key: value for key, value in os.environ.items() if key != "PSModulePath"}
        try:
            commands = rule.get("pressure_commands", rule["commands"]) if before < policy["start_free_gb"] else rule["commands"]
            # Check each command: a later successful command cannot hide an earlier failure.
            script = "; ".join("$global:LASTEXITCODE = 0; & { " + c + " }; if (-not $? -or $LASTEXITCODE -ne 0) { exit 1 }" for c in commands)
            result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
                                    creationflags=flags, env=environment)
            outcome = f"rc {result.returncode}"
            if result.returncode:
                error = f"{outcome}: {getattr(result, 'stdout', '')[-1500:]} {getattr(result, 'stderr', '')[-1500:]}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            outcome = f"failed: {exc}"
            error = outcome
    if error is None:
        _stamp(policy, rule["name"]).write_text(str(time.time()), encoding="utf-8")
    gained = free_gb(policy["hot_volume"]) - before
    log(f"cleanup {rule['name']} ({outcome}): freed {gained:.1f} GB")
    receipts = local_state(policy) / "receipts"
    receipts.mkdir(exist_ok=True)
    (receipts / f"cleanup-{rule['name']}-{time.time_ns()}.json").write_text(json.dumps(
        {"name": rule["name"], "outcome": outcome, "error": error, "deleted": deleted,
         "free_gb_before": before, "free_gb_after": free_gb(policy["hot_volume"])}), encoding="utf-8")
    if error:
        raise RuntimeError(error)
    return gained


def uncommitted_storage_code(repo: Path = Path(__file__).resolve().parents[1]) -> list[str]:
    """Paths of storage scripts/policy with uncommitted changes in the checkout the task runs."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "scripts/storage_*.py",
                              "scripts/storage_policy.json", "scripts/retire_worktree.ps1", "scripts/disk_cleanup.py"],
                             capture_output=True, text=True, timeout=60,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line[3:] for line in out.splitlines() if line.strip()]


TICK_SLACK_SECONDS = 600


def scheduled(policy: dict, root: Path) -> int:
    """OS-owned pass, including PC cleanup without an agent. One lock covers the
    whole pass; Task Scheduler tracks its lifetime and exit code. Receipts record
    failures and observed free-space deltas (concurrent writes can make them negative).
    An hourly tick checks SSD capacity. Separate success stamps admit SSD cleanup
    and HDD-capable maintenance; healthy not-due ticks never discover targets.
    Quiet policy is checked by main before discovering any storage roots.
    Refused while any storage script has uncommitted edits: the task runs the main
    checkout, so an agent's half-written code would otherwise act unattended.
    """
    dirty = uncommitted_storage_code()
    if dirty:
        print("unattended run refused: uncommitted storage code: " + ", ".join(dirty[:5]), flush=True)
        return 0
    require_machine_overlay(policy)
    state = local_state(policy)
    try:
        with Lock(state / ".scheduled.lock"):
            background_priority()
            record = {"schema": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "free_gb_before": free_gb(policy["hot_volume"]), "steps": [], "errors": []}
            record['volumes_before'] = {v: free_gb(v) for v in policy.get('fast_volumes', [])}
            with (state / "mover.log").open("a", encoding="utf-8", buffering=1) as stream:
                def log(line):
                    stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

                def step(name, action):
                    try:
                        action()
                        record["steps"].append(name)
                    except Exception as exc:
                        record["errors"].append({"step": name, "error": str(exc)})
                        log(f"FAILED {name}: {exc}")

                def read_epoch(name):
                    try:
                        value = float((state / name).read_text(encoding="utf-8"))
                        if not 0 <= value <= time.time():
                            raise ValueError("Invalid success timestamp")
                        return value
                    except FileNotFoundError:
                        return None
                    except (OSError, ValueError) as exc:
                        record["errors"].append({"step": f"cadence:{name}", "error": str(exc)})
                        return None

                def write_epoch(name):
                    pending = state / (name + ".tmp")
                    pending.write_text(str(time.time()), encoding="utf-8")
                    os.replace(pending, state / name)

                def pressure():
                    limits = policy.get("volume_limits", {})
                    return any(free_gb(volume) < limits.get(volume, {}).get(
                        "start_free_gb", policy["start_free_gb"])
                        for volume in policy.get("fast_volumes", [policy["hot_volume"]]))

                def user_cleanups():
                    for rule in cleanups_due(policy):
                        step(rule["name"], lambda rule=rule: run_cleanup(policy, rule, log=log))

                def ssd_work():
                    step("pc_cleanup_discovery", user_cleanups)
                    # Canonical SSD copies are listed independently. Do not even
                    # resolve cold model roots or legacy junctions on this pass.
                    fast = {Path(v).anchor.casefold() for v in policy.get("fast_volumes", [policy["hot_volume"]])}
                    hot_policy = dict(policy, model_stores=[p for p in policy.get("model_stores", [])
                        if Path(p).anchor.casefold() in fast and not is_link(Path(p))])
                    step("clean_hot", lambda: clean_hot(hot_policy, apply=True, log=log))
                    step("scratch_runs", lambda: clean_scratch(policy, apply=True, log=log))

                def hdd_work():
                    from storage_layout import maintain as arrange
                    step("storage_layout", lambda: arrange(policy, root, log=log))
                    from storage_retention import maintain as retain
                    step("owned_media_retention", lambda: retain(policy, root, log=log))
                    step("development_tiers", lambda: maintain(policy, root, log=log, hot_cleaned=True))
                    from storage_inventory import refresh
                    step("dependency_inventory", lambda: refresh(policy))

                log("scheduled storage pass started")
                cadence = policy.get("scheduled_cadence")
                record["skipped"] = []
                if cadence is None:
                    # Preserve the entry point for policies without tier cadence.
                    from storage_layout import maintain as arrange
                    step("storage_layout", lambda: arrange(policy, root, log=log))
                    user_cleanups()
                    step("clean_hot", lambda: clean_hot(policy, apply=True, log=log))
                    from storage_retention import maintain as retain
                    step("owned_media_retention", lambda: retain(policy, root, log=log))
                    step("development_tiers", lambda: maintain(policy, root, log=log, hot_cleaned=True))
                    from storage_inventory import refresh
                    step("dependency_inventory", lambda: refresh(policy))
                else:
                    periods = {group: float(cadence[f"{group}_every_hours"]) for group in ("ssd", "hdd")}
                    if any(not 0 < hours < float("inf") for hours in periods.values()):
                        raise ValueError("Cleanup intervals must be positive finite hours")
                    now = time.time()
                    stamps = {group: read_epoch(f"{group}_success_epoch") for group in periods}
                    # The hourly tick starts about an hour after the previous pass ended:
                    # a group due every hour must not skip every other tick.
                    due = {group: stamp is None or now - stamp >= periods[group] * 3600 - TICK_SLACK_SECONDS
                           for group, stamp in stamps.items()}
                    if due["ssd"] or pressure():
                        errors_before = len(record["errors"])
                        ssd_work()
                        if len(record["errors"]) == errors_before:
                            step("ssd_cadence", lambda: write_epoch("ssd_success_epoch"))
                    else:
                        record["skipped"].append({"group": "ssd", "reason": "not_due_and_healthy"})
                    # SSD deletion may relieve pressure: do not wake the HDD then.
                    if due["hdd"] or pressure() or (state / "journal.json").exists():
                        errors_before = len(record["errors"])
                        hdd_work()
                        if len(record["errors"]) == errors_before:
                            step("hdd_cadence", lambda: write_epoch("hdd_success_epoch"))
                    else:
                        record["skipped"].append({"group": "hdd", "reason": "not_due_and_healthy"})
                record["free_gb_after"] = free_gb(policy["hot_volume"])
                record['volumes_after'] = {v: free_gb(v) for v in policy.get('fast_volumes', [])}
                record["status"] = "failed" if record["errors"] else "completed"
                record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                pending = state / "last-scheduled.json.tmp"
                pending.write_text(json.dumps(record, indent=2), encoding="utf-8")
                os.replace(pending, state / "last-scheduled.json")
                log(f"scheduled storage pass {record['status']}: free {record['free_gb_before']:.2f} -> "
                    f"{record['free_gb_after']:.2f} GB")
                return 1 if record["errors"] else 0
    except (OSError, RuntimeError):
        # An overlapping run keeps its lock; this pass performs no work.
        return 2


def stage_pressure(policy: dict) -> bool:
    try:
        import stage
    except ImportError:
        return False
    return stage.pressure(policy)


def stage_relief(policy: dict, *, log=print) -> dict:
    """First relief when C: is short: evict the read-only stage cache (scripts/stage.py) least recently used first
    until C: has `stage.evict_first_below_free_gb` free. Regenerable by construction, so it goes before any demotion."""
    require_machine_overlay(policy)
    try:
        import stage
    except ImportError:
        return {}
    return stage.prune_for_free(policy, log=log)


def maintain(policy: dict, root: Path, *, log=print, hot_cleaned=False) -> None:
    """One scheduled pass: give back the stage cache, tier idle results when C: is short of space, then retire idle worktrees."""
    require_machine_overlay(policy)
    try:
        stage_relief(policy, log=log)
    except Exception as exc:  # noqa: BLE001 - the cache is optional relief; demotion must still run
        log(f"stage cache relief skipped: {exc}")
    scopes = [dict(policy, hot_volume=volume, **{k: v for k, v in limits.items()
              if k in ('start_free_gb', 'target_free_gb', 'pressure_free_gb', 'pressure_min_idle_hours')})
              for volume, limits in policy.get('volume_limits', {}).items()] or [policy]
    for scope in scopes:
        if free_gb(scope['hot_volume']) < scope['start_free_gb']:
            if not hot_cleaned:
                clean_hot(scope, apply=True, log=log)
            demote(scope, root, apply=True, log=log)
        else:
            demote(scope, root, apply=True, until_free_gb=float('inf'), completed_only=True, log=log)
    stamp = local_state(policy) / "last-prune.txt"
    try:
        last = float(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        last = 0.0
    if policy.get("prune_roots") is not None and time.time() - last > policy.get("prune_every_hours", 24) * 3600:
        background_priority()
        with Lock(local_state(policy) / ".mover.lock"):
            prune(policy, root, apply=True, log=log)
        stamp.write_text(str(time.time()), encoding="utf-8")
        # Once per prune cycle also survey every volume (E: build and lane worktrees, ~1-2 min): clean, idle
        # worktrees whose last commit is also older than `retire_commit_hours` are retired; protections as in `worktrees --retire`.
        wanted = [w for w in idle_worktrees(policy, root, all_volumes=True, commit_hours=policy.get("retire_commit_hours", 72.0)) if w.eligible]
    else:
        wanted = [w for w in idle_worktrees(policy, root) if w.eligible]
    if not wanted or not policy.get("move_to_hdd", True):
        return                                         # retiring a worktree archives it to the HDD
    background_priority()
    with Lock(local_state(policy) / ".mover.lock"):
        for worktree in wanted:
            try:
                record = retire_worktree(policy, root, worktree)
                log(f"retired worktree {worktree.path} [{worktree.branch or worktree.head[:12]}]: archived "
                    f"{record['bytes'] / 1e9:.2f} GB, dropped {record['regenerable_dropped']} regenerable files")
            except Exception as exc:  # noqa: BLE001 - a failure keeps that worktree
                log(f"kept worktree {worktree.path}: {exc}")


def audit(policy: dict, *, full_hash: bool = False) -> dict:
    """Check the archive against its manifests and the hot paths: every moved file present with its size (and
    SHA-256 with full_hash), the original path a junction to it (or promoted back), nothing unaccounted for."""
    if not policy.get("archive_root"):
        return {"moved": [], "problems": [], "unaccounted": [], "status": "archive_not_configured"}
    archive = Path(policy["archive_root"])
    promoted = set()
    index = archive / "index.jsonl"
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("action") == "promote":
                promoted.add(row["archive"])
    report = {"moved": [], "problems": [], "unaccounted": []}
    for directory, names, files in os.walk(archive):
        here = Path(directory)
        if here == archive:
            continue
        if MANIFEST in files:
            names[:] = []
            record = json.loads((here / MANIFEST).read_text(encoding="utf-8"))
            missing = [rel for rel, row in record["files"].items()
                       if not (here / rel).is_file() or os.path.getsize(_long(here / rel)) != row["bytes"]
                       or (full_hash and _hash(here / rel) != row["sha256"])]
            source = Path(record["source"])
            linked = is_link(source) and Path(os.path.realpath(source)).resolve() == here.resolve()
            state = "linked" if linked else "promoted" if str(here) in promoted else "unlinked"
            report["moved"].append({"archive": str(here), "source": str(source), "files": len(record["files"]),
                                    "state": state, "missing_or_different": len(missing)})
            if missing or state == "unlinked":
                report["problems"].append({"archive": str(here), "state": state, "missing_or_different": missing[:5]})
        elif here.name.endswith(PARTIAL_SUFFIX):
            names[:] = []
            report["unaccounted" if PARTIAL not in files else "problems"].append(
                {"archive": str(here), "state": "partial copy" + (" (ours, removed by the next run)" if PARTIAL in files else " without marker")})
        elif files:
            report["unaccounted"].append({"archive": str(here), "state": "files outside any manifest"})
            names[:] = []
    return report


def _is_build_checkout(entry: dict) -> bool:
    return not entry.get("branch") and re.fullmatch(r"build-[0-9a-f]+", Path(entry["worktree"]).name.lower()) is not None


def hygiene_scan(policy: dict, root: Path, *, include_builds: bool = False, ttl: float = 3600.0) -> list[dict]:
    """Per-worktree state for dev_hygiene.ps1, scanned in parallel: path, branch, head, state (missing | skipped |
    scanned), dirty path count and a sample, and the branches that contain a clean worktree's HEAD. Detached
    `build-<sha>` checkouts are skipped unless include_builds. A clean result is cached by HEAD + index mtime + size
    for `ttl` seconds (an untracked file does not touch the index, so a clean verdict must expire); a dirty one is
    always checked again."""
    cache_file = local_state(policy) / "hygiene-scan.json"
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    now = time.time()
    main = root.resolve()

    def index_stamp(path: Path) -> str:
        try:
            text = (path / ".git").read_text(encoding="utf-8").strip()
            gitdir = Path(text.split(":", 1)[1].strip()) if text.startswith("gitdir:") else path / ".git"
            st = (gitdir / "index").stat()
            return f"{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            return "?"

    def scan(entry: dict) -> dict | None:
        path = Path(entry["worktree"])
        if path.resolve() == main or entry.get("bare"):
            return None
        row = {"path": entry["worktree"], "branch": entry["branch"], "head": str(entry.get("HEAD", "")),
               "state": "scanned", "dirty": 0, "sample": [], "contained": [], "cached": False}
        if not path.is_dir():
            return {**row, "state": "missing"}
        if not include_builds and _is_build_checkout(entry):
            return {**row, "state": "skipped"}
        key = f"{row['head']}|{index_stamp(path)}"
        hit = cache.get(row["path"])
        if hit and hit.get("key") == key and now - hit.get("time", 0) < ttl and not hit.get("dirty"):
            return {**row, "dirty": 0, "contained": hit["contained"], "cached": True}
        dirty = _git("-C", str(path), "status", "--porcelain", "--untracked-files=normal").splitlines()
        row["dirty"], row["sample"] = len(dirty), dirty[:3]
        if not dirty:
            row["contained"] = [b for b in _git("-C", str(root), "branch", "--contains", row["head"],
                                                "--format=%(refname:short)").split() if b != row["branch"]]
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(scan, worktree_entries(root)) if r is not None]
    fresh = {r["path"]: {"key": f"{r['head']}|{index_stamp(Path(r['path']))}", "time": now, "dirty": r["dirty"],
                         "contained": r["contained"]}
             for r in rows if r["state"] == "scanned" and not r["cached"]}
    kept = {p: v for p, v in cache.items() if p not in fresh and Path(p).is_dir()}
    try:
        cache_file.write_text(json.dumps({**kept, **fresh}), encoding="utf-8")
    except OSError:
        pass
    return rows


def status(policy: dict, root: Path) -> dict:
    rows = candidates(policy, root)
    available = free_gb(policy["hot_volume"])
    movable = sum(c.gb for c in rows if c.eligible)
    return {"free_gb": round(available, 1), "start_free_gb": policy["start_free_gb"],
            'volumes': {volume: {'free_gb': round(free_gb(volume), 1), **limits}
                        for volume, limits in policy.get('volume_limits', {}).items()},
            "target_free_gb": policy["target_free_gb"],
            "eligible_gb": round(movable, 1),
            "gap_after_demote_gb": round(max(0.0, policy["target_free_gb"] - available - movable), 1),
            "held_gb": round(sum(c.gb for c in rows if not c.eligible), 1),
            "candidates": [asdict(c) for c in rows],
            "local_models": local_model_decisions(policy, process_texts(),
                            time.time() - policy.get("min_idle_hours", 2) * 3600,
                            repository=root, verify_hashes=False) if policy.get("unused_local_models") else []}


def print_placement(policy: dict, root: Path) -> int:
    rows = worktree_placement(policy, root)
    active_hours = worktree_promote_settings(policy)["active_commit_hours"]

    def active(r: dict) -> bool:
        return (not r["hot"] and not r["main"] and bool(r["branch"]) and r["commit_hours"] is not None
                and r["commit_hours"] < active_hours)

    print("process use: current directory read through the process PEB (ctypes; psutil is not installed) plus executable/"
          "command-line naming; open file handles are not enumerated (promote --apply proves them by renaming the directory)")
    for r in sorted(rows, key=lambda r: (r["hot"], r["commit_hours"] if r["commit_hours"] is not None else 1e9)):
        age = "   n/a" if r["commit_hours"] is None else f"{r['commit_hours']:5.1f}h"
        flag = "HDD-ACTIVE" if active(r) else "hdd" if not r["hot"] else "ssd"
        where = r["drive"] + (" via junction" if r["junction"] else "")
        use = ("; in use: " + ", ".join(r["users"][:3])) if r["users"] else ""
        print(f"  {flag:10s} {where:16s} commit {age}  {r['path']}  [{r['branch'] or r['head'][:9]}]"
              + (" (missing)" if r["missing"] else "") + use)
    unreadable = rows[0]["unreadable_processes"] if rows else 0
    print(f"{len(rows)} worktrees: {sum(1 for r in rows if r['hot'])} on the SSD, {sum(1 for r in rows if not r['hot'])} on the HDD; "
          f"{sum(1 for r in rows if active(r))} active lane(s) (commit < {active_hours:.0f} h) on the HDD; "
          f"{unreadable} processes unreadable (services/elevated/32-bit)")
    if any(active(r) for r in rows):
        print("promote with: python scripts/storage_tiers.py promote <path> --apply")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("scheduled", help="OS task owns a logged, serialized maintenance pass; no detached child")
    d = sub.add_parser("demote")
    d.add_argument("--apply", action="store_true")
    d.add_argument("--until-free-gb", type=float)
    m = sub.add_parser("maintain")
    m.add_argument("--apply", action="store_true")
    done = sub.add_parser("complete", help="archive only explicit finished output, within existing ownership rules")
    done.add_argument("--apply", action="store_true")
    r = sub.add_parser("worktrees")
    r.add_argument("--retire", action="store_true",
                   help="list (and with --apply remove) clean worktrees on every volume idle > --hours; branches stay")
    r.add_argument("--apply", action="store_true", help="with --retire: archive ignored results and remove the checkouts")
    r.add_argument("--hours", type=float, default=72.0, help="with --retire: minimum hours since the last commit and last write")
    r.add_argument("--report", action="store_true",
                   help="every registered worktree: physical drive (junctions resolved), last commit age, processes using it")
    h = sub.add_parser("hygiene-scan", help="dev_hygiene.ps1 helper: parallel, cached dirty scan of worktrees (JSON)")
    h.add_argument("--all", action="store_true", help="also scan detached build-<sha> worktrees")
    h.add_argument("--ttl-minutes", type=float, default=60.0)
    a = sub.add_parser("auto")
    a.add_argument("--detach", action="store_true")
    p = sub.add_parser("promote", help="a demoted junction back to the SSD, or an HDD worktree to the SSD (dry run unless --apply)")
    p.add_argument("path")
    p.add_argument("--apply", action="store_true", help="with a worktree: perform the move")
    q = sub.add_parser("prune")
    q.add_argument("--apply", action="store_true")
    c = sub.add_parser("clean-hot", help="superseded model downloads and orphaned temp extractions on the SSD")
    c.add_argument("--apply", action="store_true")
    v = sub.add_parser("audit")
    v.add_argument("--hash", action="store_true", help="re-read every archived file and compare its SHA-256")
    args = parser.parse_args(argv)
    if args.command in ("auto", "scheduled"):
        from dev_quiet_mode import quiet_requested
        if quiet_requested():
            print(json.dumps({"status": "deferred", "reason": "sleep_quiet_mode"}))
            return 0
    policy = load_policy()
    if args.command in ("auto", "scheduled") or getattr(args, "apply", False):
        require_machine_overlay(policy)
    root = main_root()
    if args.command == "scheduled":
        return scheduled(policy, root)
    if args.command == "auto" and policy.get("scheduler_only"):
        print(json.dumps({"status": "scheduler_owned", "task": "LSS Storage Tiers"}))
        return 0
    if args.command == "status":
        report = status(policy, root)
        for volume, limits in report.get('volumes', {}).items():
            print(f"{volume} free {limits['free_gb']} GB; demote below {limits['start_free_gb']}, until {limits['target_free_gb']}")
        print(f"C: free {report['free_gb']} GB (move starts below {report['start_free_gb']}, until {report['target_free_gb']}); "
              f"movable now {report['eligible_gb']} GB, held {report['held_gb']} GB")
        if report["gap_after_demote_gb"]:
            print(f"demotion alone is insufficient: {report['gap_after_demote_gb']} GB short of target even if all "
                  "currently eligible candidates move (cache relief and concurrent writes not included)")
        try:
            import stage
            print(f"stage cache {stage.total_bytes(policy) / 1e9:.2f} GB of {stage.settings(policy)['max_gb']} GB "
                  f"(pruned first when C: < {stage.settings(policy)['evict_first_below_free_gb']} GB free)")
        except ImportError:
            pass
        for c in report["candidates"]:
            if c["gb"] < policy["min_candidate_gb"]:
                continue
            print(f"  {'MOVE' if c['eligible'] else 'keep'} {c['gb']:7.2f} GB  idle {c['idle_hours']:8.1f} h  {c['cls']:14s} {c['path']}  ({c['reason']})")
        for model in report["local_models"]:
            action = "candidate (payload hashes checked by clean-hot)" if model["eligible"] else "kept"
            print(f"  {model['path']}: {action}: {model['reason']}")
        return 0
    if args.command == "demote":
        demote(policy, root, apply=args.apply, until_free_gb=args.until_free_gb)
        return 0
    if args.command == "worktrees" and args.report:
        return print_placement(policy, root)
    if args.command == "worktrees" and args.retire:
        found = idle_worktrees(policy, root, all_volumes=True, commit_hours=args.hours)
        wanted = [w for w in found if w.eligible]
        for wt in found:
            print(f"  {'RETIRE' if wt.eligible else 'keep'} {wt.gb:7.2f} GB idle {wt.idle_hours:7.1f} h  {wt.path}  [{wt.branch or wt.head[:12]}]  ({wt.reason})")
        print(f"{len(wanted)} of {len(found)} worktrees are retirement candidates, {sum(w.gb for w in wanted):.2f} GB"
              f"{'' if args.apply else ' (dry run; --apply retires them: ignored results archived, branches kept)'}")
        if args.apply and wanted:
            background_priority()
            with Lock(local_state(policy) / ".mover.lock"):
                for wt in wanted:
                    try:
                        record = retire_worktree(policy, root, wt)
                        print(f"retired {wt.path} [{wt.branch or wt.head[:12]}]: archived {record['bytes'] / 1e9:.2f} GB at {record['archive']}")
                    except Exception as exc:  # noqa: BLE001 - a failure keeps that worktree
                        print(f"kept {wt.path}: {exc}")
        return 0
    if args.command == "hygiene-scan":
        print(json.dumps(hygiene_scan(policy, root, include_builds=args.all, ttl=args.ttl_minutes * 60), ensure_ascii=True))
        return 0
    if args.command == "worktrees":
        for wt in idle_worktrees(policy, root):
            print(f"  {'RETIRE' if wt.eligible else 'keep'} {wt.gb:6.2f} GB idle {wt.idle_hours:7.1f} h  {wt.path}  [{wt.branch or wt.head[:12]}]  ({wt.reason})")
        return 0
    if args.command == "maintain":
        if args.apply:
            maintain(policy, root)
        return 0
    if args.command == "complete":
        demote(policy, root, apply=args.apply, until_free_gb=float("inf"), completed_only=True)
        return 0
    if args.command == "auto":
        due = cleanups_due(policy)                    # light and at most daily, so not gated on development
        if due:
            with open(local_state(policy) / "mover.log", "a", encoding="utf-8") as stream:
                for rule in due:
                    run_cleanup(policy, rule, log=lambda line: stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n"))
        import storage_budget
        development = developing(policy, root)
        if not development and not storage_budget.completed_paths(policy):
            return 0
        short = free_gb(policy["hot_volume"]) < policy["start_free_gb"]
        rows = candidates(policy, root)
        completed_ready = any(c.eligible and c.reason.startswith("explicit output completed") for c in rows)
        if not development and not completed_ready:
            return 0
        movable = completed_ready or short and (any(c.eligible for c in rows)
                             or bool(temp_orphans(policy))
                             or bool(superseded_models(policy, [q.replace("/", "\\") for q in process_texts()],
                                                       time.time() - idle_threshold_hours(policy) * 3600)))
        if not (movable or stage_pressure(policy) or (local_state(policy) / "journal.json").exists()
                or any(w.eligible for w in idle_worktrees(policy, root))):
            return 0                                  # nothing to do: decided on the SSD, the HDD is not touched
        if args.detach:
            log = local_state(policy) / "mover.log"
            # NEW_GROUP | NO_WINDOW (a hidden console its git/powershell children inherit, so none opens a window;
            # DETACHED_PROCESS would override NO_WINDOW and give each child a visible console), and BREAKAWAY_FROM_JOB
            # so closing the session hook or tool call that started it does not kill it mid-copy (a job that forbids
            # breakaway gets the plain start).
            flags = 0x00000200 | 0x08000000 if os.name == "nt" else 0
            command = [sys.executable, "-u", str(Path(__file__)),
                       "maintain" if development else "complete", "--apply"]
            with open(log, "a", encoding="utf-8") as stream:
                try:
                    subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     creationflags=flags | 0x01000000, close_fds=True)
                except OSError:
                    subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
            print(f"storage maintenance started in the background (log {log})")
            return 0
        if development:
            maintain(policy, root)
        else:
            demote(policy, root, apply=True, until_free_gb=float("inf"), completed_only=True)
        return 0
    if args.command == "prune":
        if args.apply:
            background_priority()
            with Lock(local_state(policy) / ".mover.lock"):
                prune(policy, root, apply=True)
        else:
            prune(policy, root, apply=False)
        return 0
    if args.command == "clean-hot":
        if args.apply:
            with Lock(local_state(policy) / ".mover.lock"):
                report = clean_hot(policy, apply=True)
        else:
            report = clean_hot(policy, apply=False)
        print(f"{len(report['deleted'])} items, {report['bytes'] / 1e9:.2f} GB{'' if args.apply else ' (dry run)'}")
        return 0
    if args.command == "promote":
        target = Path(args.path)
        if is_link(target) and (Path(os.path.realpath(target)) / MANIFEST).exists():
            if not args.apply:
                print(json.dumps({"status": "DRY_RUN_WOULD_PROMOTE", "path": str(target)}))
                return 0
            print(json.dumps(promote(policy, target), ensure_ascii=False))
            return 0
        record = promote_worktree(policy, root, target, apply=args.apply)
        print(json.dumps({k: v for k, v in record.items() if k != "files"}, ensure_ascii=False, indent=1))
        if record["status"] == "DRY_RUN_WOULD_PROMOTE":
            print("dry run; --apply moves it")
        return 0 if record["status"] in ("DRY_RUN_WOULD_PROMOTE", "WORKTREE_PROMOTED_VERIFIED_LEGACY_JUNCTION") else 1
    if args.command == "audit":
        report = audit(policy, full_hash=args.hash)
        print(f"{len(report['moved'])} moved directories; problems {len(report['problems'])}; unaccounted {len(report['unaccounted'])}")
        for row in report["moved"]:
            print(f"  {row['state']:9s} files {row['files']:5d} bad {row['missing_or_different']}  {row['source']}")
        for row in report["problems"] + report["unaccounted"]:
            print(f"  ! {row['state']}: {row['archive']}")
        return 1 if report["problems"] or report["unaccounted"] else 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StorageMutationRefused as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
    except RuntimeError as exc:          # another mover holds the lock: nothing to do, not a failure
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {exc}")
        raise SystemExit(0)
