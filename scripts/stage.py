"""Read-only SSD stage cache: work on the SSD, keep on the HDD.

    python scripts/stage.py get <source-path> [--purpose X] [--force] [--strict]   # prints the verified SSD path
    python scripts/stage.py list [--json]                                          # entries, size, last access
    python scripts/stage.py prune [--to-gb N] [--grace-minutes M]                  # LRU eviction down to N GB (default: the cap)

    from stage import stage_path               # scripts/ and tests/ (never lss/ product code)
    local = stage_path(media_root / "TestCorpus" / "..." / "episode.mkv", purpose="probe")

Repeated reads of reference and test media on the USB HDD are what make runs wait. `get` copies a file or directory
once to the SSD (`stage.root` in `scripts/storage_policy.json`) and returns the SSD path; the HDD original is never
touched, and the copy is a regenerable cache by construction: it is evicted first when C: runs short
(`storage_tiers.py` calls `prune_for_free` before any demotion and never demotes the stage root).

- Layout: `<root>/<sha1(source path)[:16]>-<bytes>-<mtime_ns>/<name>` plus `.stage.json` (source, per-file SHA-256,
  purpose). The size and mtime are part of the name, so a changed source is a different entry and the old one is
  removed once the new one is complete. A hit costs two `stat` calls, no hashing (`--verify` re-hashes).
- Verification: SHA-256 is computed while the source is read, the copy is read back from the SSD and compared; a hash
  already recorded in a `.lss-relocation-manifest.json` above the source must agree too. The source is re-statted
  after the copy: a file that changed meanwhile is not staged. Files are marked read-only.
- Concurrency: a per-entry OS byte-range lock (the one `storage_tiers.Lock` uses), the copy goes to `<entry>.partial`
  and is renamed into place atomically, so two processes staging the same file copy it once and nobody sees half a copy.
  Cache misses also hold a shared capacity lock from admission through publication: different files cannot spend
  the same free space concurrently. Cache hits do not wait for that copy lock.
- Eviction: least recently used first (last access is kept in `index.json`), never an entry accessed within
  `grace_minutes` and never one a process has open: the entry is renamed first and Windows refuses to rename a
  directory holding open files. The cap (`max_gb`) is enforced on every `get`.
- A source already on the hot volume (junctions resolved) is returned unchanged. `stage_path` falls back to the
  original path with a warning when staging is refused (larger than the cap, C: too full); `--strict` makes it an error.
- Cache creation, replacement and eviction require the machine overlay and
  nonempty protection lists. Listing and an already-hot source need no overlay.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import storage_tiers as tiers  # noqa: E402

DEFAULTS = {"root": "%LOCALAPPDATA%/LSS-dev/stage", "max_gb": 40.0, "evict_first_below_free_gb": 100.0,
            "min_free_after_gb": 30.0, "grace_minutes": 10.0}
META = ".stage.json"
PARTIAL_SUFFIX = ".partial"
EVICTING_SUFFIX = ".evicting"


class StageRefused(RuntimeError):
    """Staging is not possible (too large for the cap, C: too full, source unreadable); use the original path."""


def settings(policy: dict) -> dict:
    return {**DEFAULTS, **(policy.get("stage") or {})}


def cache_volume(policy: dict) -> str:
    """Capacity follows the resolved cache drive, including legacy junctions."""
    return str(stage_root(policy).resolve().anchor)


def stage_root(policy: dict) -> Path:
    return Path(os.path.expandvars(settings(policy)["root"]))


def _now() -> float:
    return time.time()


# --- identity of a source --------------------------------------------------------------------------------------------

def _identify(real: Path) -> tuple[str, int, int, list[Path] | None]:
    """(kind, bytes, newest mtime_ns, files) of a file or directory; a directory's files come back sorted. The
    entry name of a directory also carries the file count, so removing a file is a change too."""
    st = os.stat(tiers._long(real))
    if stat.S_ISDIR(st.st_mode):
        files = tiers._files(real)
        stats = [os.stat(tiers._long(f)) for f in files]
        return "dir", sum(s.st_size for s in stats), max([s.st_mtime_ns for s in stats] + [st.st_mtime_ns]), files
    return "file", st.st_size, st.st_mtime_ns, None


def entry_name(real: Path, size: int, mtime_ns: int, count: int | None = None) -> str:
    digest = hashlib.sha1(str(real).lower().encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{size}-{mtime_ns}" + ("" if count is None else f"-{count}f")


def receipt_hash(real: Path, size: int) -> str | None:
    """SHA-256 of a file from a `.lss-relocation-manifest.json` in one of its parent directories, when the size agrees."""
    for parent in list(real.parents)[:6]:
        manifest = parent / tiers.MANIFEST
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows = record.get("files") or record.get("files_sha256") or {}
        row = rows.get(str(real.relative_to(parent))) or rows.get(str(real.relative_to(parent)).replace("\\", "/"))
        if row and row.get("bytes") == size and row.get("sha256"):
            return str(row["sha256"]).lower()
    return None


# --- index and locks -------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _locked(root: Path, name: str, timeout: float = 3600.0):
    """Wait for the lock `name` under root/.locks (an OS byte-range lock, released when the holder exits)."""
    deadline = _now() + timeout
    while True:
        lock = tiers.Lock(root / ".locks" / f"{name}.lock")
        try:
            lock.__enter__()
            break
        except RuntimeError:
            if _now() > deadline:
                raise StageRefused(f"timed out waiting for the stage lock {name}")
            time.sleep(0.1)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _read_index(root: Path) -> dict:
    try:
        return json.loads((root / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"entries": {}}


def _update_index(root: Path, change) -> None:
    with _locked(root, "index", timeout=60.0):
        index = _read_index(root)
        change(index["entries"])
        temporary = root / "index.json.tmp"
        temporary.write_text(json.dumps(index, indent=1), encoding="utf-8")
        os.replace(temporary, root / "index.json")


def _touch(root: Path, name: str, **fields) -> None:
    def change(entries: dict) -> None:
        entries.setdefault(name, {}).update(last_access=_now(), **fields)
    _update_index(root, change)


def entries(policy: dict) -> list[dict]:
    """Complete entries with size, source, purpose and last access, least recently used first."""
    root = stage_root(policy)
    index = _read_index(root)["entries"]
    found = []
    if not root.is_dir():
        return found
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith(".") or path.name.endswith((PARTIAL_SUFFIX, EVICTING_SUFFIX)):
            continue
        try:
            meta = json.loads((path / META).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        row = index.get(path.name, {})
        found.append({"name": path.name, "path": str(path), "source": meta.get("source"), "bytes": meta.get("bytes", 0),
                      "purpose": row.get("purpose") or meta.get("purpose"),
                      "last_access": row.get("last_access") or meta.get("staged_at", path.stat().st_mtime)})
    return sorted(found, key=lambda e: e["last_access"])


def total_bytes(policy: dict) -> int:
    return sum(e["bytes"] for e in entries(policy))


# --- eviction --------------------------------------------------------------------------------------------------------

def _evict(policy: dict, path: Path) -> bool:
    """Remove an entry unless a process has it open: the rename is the exclusive-open test (Windows refuses to rename
    a directory holding open files)."""
    tiers.require_machine_overlay(policy)
    tomb = path.with_name(path.name + EVICTING_SUFFIX)
    if tomb.exists():
        tiers._remove_tree(tomb)
    try:
        os.rename(path, tomb)
    except OSError:
        return False
    try:
        tiers._remove_tree(tomb)
    except OSError:
        pass                                              # a leftover .evicting directory is removed by the next prune
    return True


def _sweep(policy: dict) -> None:
    """Drop unfinished copies and half-removed entries nobody holds (their entry lock proves no stager is running)."""
    tiers.require_machine_overlay(policy)
    root = stage_root(policy)
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.is_dir() and path.name.endswith((PARTIAL_SUFFIX, EVICTING_SUFFIX)):
            key = path.name.rsplit(".", 1)[0]
            try:
                with _locked(root, key, timeout=0.2):
                    tiers._remove_tree(path)
            except (StageRefused, OSError):
                pass


def prune(policy: dict, *, to_gb: float | None = None, until_free_gb: float | None = None,
          grace_minutes: float | None = None, keep: set[str] | None = None, log=print) -> dict:
    """Evict least recently used entries until the cache is at most `to_gb` (default: the cap) and, with
    `until_free_gb`, C: has that much free. Entries inside the grace period, `keep`, or open elsewhere stay."""
    tiers.require_machine_overlay(policy)
    cfg = settings(policy)
    root = stage_root(policy)
    to_gb = cfg["max_gb"] if to_gb is None and until_free_gb is None else to_gb
    grace = cfg["grace_minutes"] if grace_minutes is None else grace_minutes
    removed, kept_open, freed = [], [], 0
    _sweep(policy)
    for item in entries(policy):
        total = total_bytes(policy)
        enough_size = to_gb is None or total <= to_gb * 1e9
        enough_free = until_free_gb is None or tiers.free_gb(cache_volume(policy)) >= until_free_gb
        if enough_size and enough_free:
            break
        if (keep and item["name"] in keep) or _now() - item["last_access"] < grace * 60:
            continue
        try:
            with _locked(root, item["name"], timeout=5.0):    # not while another process is staging or verifying it
                if _evict(policy, Path(item["path"])):
                    removed.append(item)
                    freed += item["bytes"]
                    _update_index(root, lambda e, n=item["name"]: e.pop(n, None))
                    log(f"evicted {item['bytes'] / 1e9:.2f} GB {item['source']}")
                else:
                    kept_open.append(item["name"])
        except StageRefused:                              # its lock is held: it is being staged right now
            kept_open.append(item["name"])
    result = {"evicted": len(removed), "bytes": freed, "in_use": kept_open, "remaining_bytes": total_bytes(policy)}
    if removed:
        receipts = tiers.local_state(policy) / "receipts"
        receipts.mkdir(exist_ok=True)
        (receipts / f"tier-stage-prune-{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(
            {"type": "lss_storage_tier", "action": "stage_prune", "deleted": [
                {"path": e["path"], "source": e["source"], "bytes": e["bytes"], "rule": "stage_cache_lru"} for e in removed],
             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def pressure(policy: dict) -> bool:
    """C: is below `evict_first_below_free_gb` and the cache holds something to give back."""
    cfg = settings(policy)
    return (stage_root(policy).is_dir() and tiers.free_gb(cache_volume(policy)) < cfg["evict_first_below_free_gb"]
            and bool(entries(policy)))


def prune_for_free(policy: dict, *, log=print) -> dict:
    """The first thing storage_tiers gives back when C: is short, before any demotion: evict LRU entries until C: has
    `evict_first_below_free_gb` free or the cache is empty."""
    tiers.require_machine_overlay(policy)
    if not pressure(policy):
        return {"evicted": 0, "bytes": 0, "in_use": [], "remaining_bytes": total_bytes(policy)}
    return prune(policy, until_free_gb=settings(policy)["evict_first_below_free_gb"], log=log)


# --- staging ---------------------------------------------------------------------------------------------------------

def _complete(entry: Path, kind: str, source_name: str, size: int) -> bool:
    try:
        meta = json.loads((entry / META).read_text(encoding="utf-8"))
        return meta.get("kind") == kind and meta.get("bytes") == size and (entry / source_name).exists()
    except (OSError, ValueError):
        return False


def _make_room(policy: dict, size: int, keep: set[str]) -> None:
    cfg = settings(policy)
    if size > cfg["max_gb"] * 1e9:
        raise StageRefused(f"{size / 1e9:.1f} GB is larger than the stage cap {cfg['max_gb']} GB")
    if total_bytes(policy) + size > cfg["max_gb"] * 1e9:
        prune(policy, to_gb=cfg["max_gb"] - size / 1e9, keep=keep, log=lambda *_: None)
    if total_bytes(policy) + size > cfg["max_gb"] * 1e9:
        raise StageRefused("the stage cap is held by entries in use or accessed a moment ago")
    wanted_free = cfg["min_free_after_gb"] + size / 1e9
    if tiers.free_gb(cache_volume(policy)) < wanted_free:
        prune(policy, until_free_gb=wanted_free, keep=keep, log=lambda *_: None)
    if tiers.free_gb(cache_volume(policy)) < wanted_free:
        volume = cache_volume(policy).rstrip('\\/')
        raise StageRefused(f"{volume} free would fall below {cfg['min_free_after_gb']} GB")


def get(policy: dict, source, *, purpose: str | None = None, force: bool = False, verify: bool = False) -> dict:
    """Stage a file or directory; returns {"path", "staged", "hit", "bytes", ...}. `path` is the SSD copy, or the
    source itself when it already is on the hot volume (and not `force`). Raises StageRefused when staging is not
    possible, FileNotFoundError for a missing source."""
    source = Path(source)
    real = Path(os.path.realpath(source))
    if not real.exists():
        raise FileNotFoundError(source)
    if not force and tiers.on_hot_volume(real, policy):
        return {"path": str(source), "staged": False, "hit": False, "reason": "already on the hot volume"}
    tiers.require_machine_overlay(policy)
    root = stage_root(policy)
    root.mkdir(parents=True, exist_ok=True)
    kind, size, mtime_ns, files = _identify(real)
    count = len(files) if files is not None else None
    name = entry_name(real, size, mtime_ns, count)
    entry = root / name
    if _complete(entry, kind, real.name, size):
        if verify:
            _verify_entry(entry, real.name)
        _touch(root, name, source=str(real), purpose=purpose)
        if _complete(entry, kind, real.name, size):         # a prune may have evicted it between the check and the touch
            return {"path": str(entry / real.name), "staged": True, "hit": True, "bytes": size, "entry": name}
        _update_index(root, lambda e: e.pop(name, None))
    started = _now()
    with _locked(root, "capacity"), _locked(root, name):
        if _complete(entry, kind, real.name, size):         # another process staged it while we waited (we hold its lock)
            _touch(root, name, source=str(real), purpose=purpose)
            return {"path": str(entry / real.name), "staged": True, "hit": True, "bytes": size, "entry": name}
        _make_room(policy, size, keep={name})
        partial = entry.with_name(name + PARTIAL_SUFFIX)
        if partial.exists():
            tiers._remove_tree(partial)
        if entry.exists():                                   # incomplete leftover (no .stage.json)
            tiers._remove_tree(entry)
        partial.mkdir(parents=True)
        try:
            rows, hash_source = _copy_in(real, partial / real.name, kind, size, files)
            again = _identify(real)
            if again[1:3] != (size, mtime_ns) or (files is not None and len(again[3]) != len(files)):
                raise StageRefused(f"{real} changed while it was staged")
            meta = {"type": "lss_stage_entry", "source": str(real), "kind": kind, "bytes": size, "source_mtime_ns": mtime_ns,
                    "files": rows, "hash_source": hash_source, "purpose": purpose, "staged_at": _now(),
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (partial / META).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            for path in (partial / real.name).rglob("*") if kind == "dir" else [partial / real.name]:
                if path.is_file():
                    os.chmod(path, stat.S_IREAD)
            tiers._replace_with_retry(partial, entry)
        except BaseException:
            if partial.exists():
                tiers._remove_tree(partial)
            raise
    _touch(root, name, source=str(real), purpose=purpose)
    seconds = _now() - started
    for other in root.iterdir():                             # same source, older identity: stale
        if other.is_dir() and other.name != name and other.name.startswith(name.split("-")[0] + "-") \
                and not other.name.endswith((PARTIAL_SUFFIX, EVICTING_SUFFIX)):
            if _evict(policy, other):
                _update_index(root, lambda e, n=other.name: e.pop(n, None))
    receipts = tiers.local_state(policy) / "receipts"
    receipts.mkdir(exist_ok=True)
    (receipts / f"tier-stage-{time.strftime('%Y%m%dT%H%M%S')}-{real.name[:40]}.json").write_text(json.dumps(
        {"type": "lss_storage_tier", "action": "stage", "source": str(real), "entry": str(entry), "bytes": size,
         "files": len(rows), "hash_source": hash_source, "seconds": round(seconds, 2), "purpose": purpose,
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"path": str(entry / real.name), "staged": True, "hit": False, "bytes": size, "entry": name,
            "seconds": round(seconds, 2), "hash_source": hash_source}


def _copy_in(real: Path, target: Path, kind: str, size: int, files: list[Path] | None) -> tuple[dict, str]:
    """Verified copy into the partial entry: hash while reading the source, read the copy back, compare."""
    if kind == "dir":
        return tiers.copy_verified(real, target), "read"
    copied, sha = tiers._copy_file(real, target)
    expected = receipt_hash(real, size)
    if copied != size or os.path.getsize(tiers._long(target)) != size or tiers._hash(target) != sha:
        raise RuntimeError(f"verification failed for {real}; nothing staged")
    if expected and expected != sha:
        raise RuntimeError(f"{real} differs from the SHA-256 in its receipt; nothing staged")
    return {real.name: {"bytes": size, "sha256": sha}}, "receipt+read" if expected else "read"


def _verify_entry(entry: Path, name: str) -> None:
    meta = json.loads((entry / META).read_text(encoding="utf-8"))
    base = entry / name
    for relative, row in meta["files"].items():
        path = base if meta["kind"] == "file" else base / relative
        if os.path.getsize(tiers._long(path)) != row["bytes"] or tiers._hash(path) != row["sha256"]:
            raise RuntimeError(f"staged copy {path} no longer matches its SHA-256; evict it with prune --to-gb 0")


def stage_path(path, *, purpose: str | None = None, policy: dict | None = None, force: bool = False,
               strict: bool = False) -> Path:
    """The SSD path to read instead of `path`: a verified read-only copy (made on first use), or `path` itself when
    it is already on the hot volume or staging is refused (a warning goes to stderr; `strict` raises instead)."""
    policy = tiers.load_policy() if policy is None else policy
    try:
        return Path(get(policy, path, purpose=purpose, force=force)["path"])
    except (RuntimeError, OSError) as exc:
        if strict:
            raise
        print(f"stage: using the original {path}: {exc}", file=sys.stderr)
        return Path(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("get")
    g.add_argument("path")
    g.add_argument("--purpose")
    g.add_argument("--force", action="store_true", help="stage even when the source is on the hot volume")
    g.add_argument("--strict", action="store_true", help="exit 2 instead of printing the original path when staging is refused")
    g.add_argument("--verify", action="store_true", help="re-hash an existing staged copy before returning it")
    ls = sub.add_parser("list")
    ls.add_argument("--json", action="store_true")
    p = sub.add_parser("prune")
    p.add_argument("--to-gb", type=float, help="evict LRU entries until the cache is at most this big (default: the cap)")
    p.add_argument("--grace-minutes", type=float, help="keep entries accessed more recently than this (default: policy)")
    args = parser.parse_args(argv)
    policy = tiers.load_policy()
    if args.command == "get":
        try:
            result = get(policy, args.path, purpose=args.purpose, force=args.force, verify=args.verify)
        except (RuntimeError, OSError) as exc:
            if args.strict:
                print(f"stage: refused: {exc}", file=sys.stderr)
                return 2
            print(f"stage: using the original {args.path}: {exc}", file=sys.stderr)
            print(args.path)
            return 0
        note = "hit" if result.get("hit") else "staged" if result["staged"] else result.get("reason", "")
        print(f"stage: {note} {result.get('bytes', 0) / 1e9:.2f} GB"
              + (f" in {result['seconds']} s" if "seconds" in result else ""), file=sys.stderr)
        print(result["path"])
        return 0
    if args.command == "list":
        rows = entries(policy)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False))
            return 0
        cap = settings(policy)["max_gb"]
        print(f"{len(rows)} entries, {sum(r['bytes'] for r in rows) / 1e9:.2f} of {cap} GB at {stage_root(policy)}; "
              f"Stage volume free {tiers.free_gb(policy['hot_volume']):.1f} GB")
        for r in rows:
            age = (_now() - r["last_access"]) / 3600
            print(f"  {r['bytes'] / 1e9:7.2f} GB  used {age:6.1f} h ago  {r['purpose'] or '-':12s} {r['source']}")
        return 0
    result = prune(policy, to_gb=args.to_gb, grace_minutes=args.grace_minutes)
    print(f"evicted {result['evicted']} entries, {result['bytes'] / 1e9:.2f} GB; {result['remaining_bytes'] / 1e9:.2f} GB remain"
          + (f"; in use, kept: {len(result['in_use'])}" if result["in_use"] else ""))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except tiers.StorageMutationRefused as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
