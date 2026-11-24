"""Bounded revisit planning from verified, constant-offset BGM cue memory links.

Plans only nominate an original source to revisit. They grant no audio or publication authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from contextlib import contextmanager

POLICY = "bgm-memory-refresh-v2"
DIRECTORY = "bgm-memory-refresh"
MIN_OVERLAP_SECONDS = 6.0
MIN_EXTENSION_SECONDS = 2.0
MIN_DONOR_SURPLUS_SECONDS = 5.0


def _attempt_path(state_root: Path, token: str) -> Path:
    return Path(state_root) / DIRECTORY / f"{token}.json"


@contextmanager
def _locked(path: Path):
    """Small cross-process lock without importing the audio-heavy application session module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
                                encoding="utf-8", newline="\n", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _present(member: dict) -> bool:
    from lss.bgm.cue_memory import _still_present
    try:
        return _still_present(member)
    except Exception:  # malformed memory entries are simply ineligible
        return False


def _union_length(ranges: list[tuple[float, float]]) -> float:
    total = 0.0
    end = None
    for low, high in sorted(ranges):
        if high <= low:
            continue
        if end is None or low > end:
            total += high - low
            end = high
        elif high > end:
            total += high - end
            end = high
    return total


def _link_geometry(link: dict, target_id: str, target: dict, witness: dict) -> tuple[float, float] | None:
    """Return verified shared length and witness-only source coverage, in seconds."""
    if not isinstance(link, dict):
        return None
    from lss.bgm import cue_memory
    try:
        if link.get("rate") is not None or link.get("paired") is not True:
            return None
        if not math.isfinite(float(link.get("waveform_explained", 0))) or float(link.get("waveform_explained", 0)) < cue_memory.MIN_EXPLAINED:
            return None
        if {link.get("a"), link.get("b")} != {target_id, witness.get("asset_id")}:
            return None
        target_side, witness_side = ("a", "b") if link["a"] == target_id else ("b", "a")
        target_duration, witness_duration = float(target["duration_seconds"]), float(witness["duration_seconds"])
        if not all(math.isfinite(v) and v > 0 for v in (target_duration, witness_duration)):
            return None
        tr = link[f"{target_side}_ranges"]
        wr = link[f"{witness_side}_ranges"]
        if not isinstance(tr, list) or len(tr) != len(wr) or not tr:
            return None
        paired_target: list[tuple[float, float]] = []
        paired_witness: list[tuple[float, float]] = []
        mapped_witness: list[tuple[float, float]] = []
        clock_delta = None
        for target_range, witness_range in zip(tr, wr):
            t0, t1 = map(float, target_range)
            w0, w1 = map(float, witness_range)
            # A constant-offset match has equal extent on both clocks. Reject malformed or stretched rows.
            if (not all(math.isfinite(v) for v in (t0, t1, w0, w1)) or t1 <= t0 or w1 <= w0
                    or abs((t1 - t0) - (w1 - w0)) > 0.15):
                return None
            delta = t0 - w0
            if clock_delta is not None and abs(delta - clock_delta) > .15:
                return None
            if clock_delta is None:
                clock_delta = delta
            # The matcher's fixed windows can extend slightly beyond a short reference's final sample.
            # Count only the paired intersection physically present on BOTH clocks, without granting the
            # padded window any extra shared seconds or requiring an arbitrary endpoint tolerance.
            low, high = max(t0, 0.0, delta), min(t1, w1 + delta, target_duration, delta + witness_duration)
            if high > low:
                paired_target.append((low, high))
                paired_witness.append((low - delta, high - delta))
            mapped_witness.append((delta, witness_duration + delta))
        overlap = min(_union_length(paired_target), _union_length(paired_witness))
        # Map the complete witness duration into target time using the verified constant offset.
        extension = _union_length([(max(0.0, low - target_duration), max(0.0, high - target_duration))
                                   for low, high in mapped_witness])
        extension += _union_length([(max(0.0, -high), max(0.0, -low)) for low, high in mapped_witness])
        return overlap, extension
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _current_sources(state_root: Path) -> list[tuple[str, dict, list[dict]]]:
    from lss.bgm import current_best
    found = []
    folder = Path(state_root) / current_best.DIRECTORY
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        if path.name.endswith(".switch.json"):
            continue
        try:
            record = current_best.read(state_root, path.stem)
            current = record["current"]
            if current.get("pinned"):
                continue
            rows = current["files"]
            if not isinstance(rows, list) or not rows:
                continue
            # CurrentBest is authoritative. Keep only rows whose accepted media still passes the same check
            # used by cue_memory when it compares members.
            if all(isinstance(row, dict) for row in rows):
                found.append((path.stem, current, rows))
        except Exception:  # malformed records are not evidence
            continue
    return found


def _source_path(rows: list[dict], source_sha: str) -> str | None:
    for row in rows:
        run_path = row.get("run_path")
        if not run_path:
            continue
        run = _read(Path(run_path))
        source = (run or {}).get("source") or {}
        candidate = source.get("path")
        if source.get("sha256") == source_sha and candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def _live(receipt: dict) -> bool:
    return receipt.get("status") == "running" and _pid_alive(receipt.get("pid"))


def _evaluated(state_root: Path, recipe_id, memory_recipe) -> dict[str, set[str]]:
    """External PCM already queried by a terminal full-source attempt, per source, for these recipes."""
    evaluated: dict[str, set[str]] = {}
    for path in (Path(state_root) / DIRECTORY).glob("*.json"):
        receipt = _read(path) or {}
        prior = receipt.get("plan") or {}
        if (receipt.get("type") == "bgm_memory_refresh_attempt" and receipt.get("policy") == POLICY
                and receipt.get("status") in ("complete", "failed")
                and prior.get("processing_recipe_id") == recipe_id
                and prior.get("memory_recipe_id") == memory_recipe):
            evaluated.setdefault(prior.get("source_sha256"), set()).update(prior.get("external_pcm_sha256") or [])
    return evaluated


def plans(state_root: Path, recipe_id: str | None = None) -> list[dict]:
    """Return eligible deterministic plans; malformed or unavailable state yields no plan."""
    try:
        from lss.bgm import cue_memory
        memory = cue_memory.read(Path(state_root))
        members = memory.get("members")
        links = memory.get("links")
        if not isinstance(members, dict) or not isinstance(links, list):
            return []
        current = {sha: (record, rows) for sha, record, rows in _current_sources(Path(state_root))}
        accepted = {(row.get("cue_key"), row.get("pcm_sha256"), row.get("public_path"))
                    for _, rows in current.values() for row in rows}
        eligible = []
        present = {key: member for key, member in members.items()
                   if isinstance(member, dict) and _present(member)}
        memory_recipe = cue_memory.recipe_id()
        evaluated = _evaluated(Path(state_root), recipe_id, memory_recipe)
        for target_id, target in present.items():
            if not target.get("unresolved_extent"):
                continue
            source_sha = target.get("source_sha256")
            if source_sha not in current:
                continue
            record, rows = current[source_sha]
            if not any(row.get("cue_key") == target.get("cue_key")
                       and row.get("pcm_sha256") == target.get("pcm_sha256")
                       and row.get("public_path") == target.get("path") for row in rows):
                continue
            source_path = _source_path(rows, source_sha)
            if not source_path:
                continue
            external_pcm = sorted({m["pcm_sha256"] for m in present.values()
                                   if m.get("source_sha256") != source_sha
                                   and (m.get("cue_key"), m.get("pcm_sha256"), m.get("path")) in accepted})
            for witness_id, witness in present.items():
                if (witness_id == target_id or witness.get("source_sha256") == source_sha
                        or witness.get("unresolved_extent")
                        or (target.get("source_name") and witness.get("source_name") == target.get("source_name"))):
                    continue
                # A full-source attempt has already queried this accepted external audio. Another cue/pair
                # nomination, own-source publication, or unrelated library growth is not new support.
                if witness.get("pcm_sha256") in evaluated.get(source_sha, set()):
                    continue
                if float(witness.get("duration_seconds", 0)) < float(target.get("duration_seconds", 0)) + MIN_DONOR_SURPLUS_SECONDS:
                    continue
                if (witness.get("cue_key"), witness.get("pcm_sha256"), witness.get("path")) not in accepted:
                    continue
                for link in links:
                    geometry = _link_geometry(link, target_id, target, witness)
                    if geometry is None:
                        continue
                    overlap, extension = geometry
                    if overlap < MIN_OVERLAP_SECONDS or extension < MIN_EXTENSION_SECONDS:
                        continue
                    # Token contains no labels, names, timestamps, or mutable presentation metadata.
                    token_body = {"policy": POLICY, "recipe_id": recipe_id,
                                  "memory_recipe_id": memory_recipe, "target_source_sha256": source_sha,
                                  "external_pcm_sha256": external_pcm}
                    token = hashlib.sha256(json.dumps(token_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    plan = {"source_path": source_path, "source_sha256": source_sha, "token": token,
                            "target_member": target_id, "witness_member": witness_id,
                            "witness_pcm_sha256": witness.get("pcm_sha256"),
                            "processing_recipe_id": recipe_id, "memory_recipe_id": memory_recipe,
                            "external_pcm_sha256": external_pcm}
                    existing = _read(_attempt_path(Path(state_root), token))
                    if existing and existing.get("status") in ("complete", "failed"):
                        continue
                    if existing and _live(existing):
                        continue
                    eligible.append((token, plan))
        # The caller excludes its foreground source, then executes only one plan.
        unique = {token: plan for token, plan in reversed(sorted(eligible, key=lambda row: row[0]))}
        return [unique[token] for token in sorted(unique)]
    except Exception:  # absent/malformed memory falls back to ordinary BGM processing
        return []


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        if os.name != "nt":
            os.kill(pid, 0)
            return True
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied is not permission to retry work that may still be running.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    except (OSError, TypeError, ValueError):
        return False


def pending(state_root: Path, *, recipe_id: str | None = None) -> list[dict]:
    return plans(Path(state_root), recipe_id=recipe_id)


def _claims_lock(state_root: Path) -> Path:
    return Path(state_root) / DIRECTORY / "claims.lock"


def claim(state_root: Path, plan: dict) -> bool:
    """Atomically mark a plan running before work begins; terminal tokens cannot be claimed again.

    A live attempt for the same source blocks another token for it: unrelated library growth changes the token,
    not the evidence, and the later request must not analyse the source a second time.
    """
    state_root = Path(state_root)
    token = str(plan.get("token") or "")
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        return False
    path = _attempt_path(state_root, token)
    with _locked(_claims_lock(state_root)):
        old = _read(path)
        if old and old.get("status") in ("complete", "failed"):
            return False
        if old and _live(old):
            return False
        for other in path.parent.glob("*.json"):
            receipt = _read(other) or {}
            if (other != path and _live(receipt)
                    and (receipt.get("plan") or {}).get("source_sha256") == plan.get("source_sha256")):
                return False
        # A plan computed before another attempt finished may name a witness that attempt already queried.
        done = _evaluated(state_root, plan.get("processing_recipe_id"), plan.get("memory_recipe_id"))
        if plan.get("witness_pcm_sha256") in done.get(plan.get("source_sha256"), set()):
            return False
        receipt = {"type": "bgm_memory_refresh_attempt", "policy": POLICY, "token": token,
                   "status": "running", "pid": os.getpid(), "plan": plan}
        _write_atomic(path, receipt)
    return True


def release(state_root: Path, plan: dict) -> bool:
    """Return this process's unfinished claim (a user quit), so the next request naming the source resumes it."""
    path = _attempt_path(Path(state_root), str(plan.get("token") or ""))
    with _locked(_claims_lock(Path(state_root))):
        receipt = _read(path)
        if not receipt or receipt.get("status") != "running" or receipt.get("pid") != os.getpid():
            return False
        path.unlink()
    return True


def finish(state_root: Path, plan: dict, outcome: dict) -> Path | None:
    """Record a terminal result for a previously claimed plan."""
    path = _attempt_path(Path(state_root), str(plan.get("token") or ""))
    if not path.is_file():
        return None
    with _locked(_claims_lock(Path(state_root))):
        receipt = _read(path)
        if not receipt or receipt.get("status") != "running":
            return None
        success = outcome.get("ok")
        if success is None:
            success = outcome.get("status") in {"complete", "completed", "published", "success", "ok"}
        status = "complete" if success else "failed"
        _write_atomic(path, {**receipt, "status": status, "outcome": outcome})
    return path
