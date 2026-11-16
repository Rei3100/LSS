"""Cross-process SSD admission for development jobs, never Product runtime.

Reservations are estimates in decimal GB, not measured usage. The full estimate
remains outstanding until exit (conservatively double-counting bytes already
written). OS byte-lock leases, rather than PID guesses, establish ownership.
State errors fail closed. Receipts request archival only for explicit owned paths;
this module never moves or deletes output. Callers keep the context alive until
all writers exit and pass child_env() to nested wrappers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

TOKEN_ENV = "LSS_STORAGE_RESERVATION"


class BudgetError(RuntimeError):
    """Capacity unavailable or reservation state cannot safely establish ownership."""


def _tiers():
    # Lazy: the mover imports protected_paths while it is itself being loaded.
    import storage_tiers
    return storage_tiers


def _root(policy):
    return _tiers().local_state(policy) / "budget"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise BudgetError(f"storage budget {name} must be a finite nonnegative number")
    return float(value)


def _paths(paths):
    return sorted({str(p) for item in paths for p in
                   (Path(os.path.abspath(os.fspath(item))), Path(item).resolve())})


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _mutex(root):
    lock = _tiers().Lock(root / ".budget.lock")
    while True:
        try:
            lock.__enter__()
            break
        except RuntimeError:
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _validate(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        token = record["token"]
        if record["schema_version"] != 1 or not isinstance(token, str) or uuid.UUID(hex=token).hex != path.stem:
            raise ValueError("invalid identity")
        _number(record["required_gb"], "record required_gb")
        if 'volumes' in record and (not isinstance(record['volumes'], list) or not record['volumes']
                or any(not isinstance(v, str) or not Path(v).is_absolute() for v in record['volumes'])):
            raise ValueError('invalid reservation volumes')
        if not isinstance(record["paths"], list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in record["paths"]):
            raise ValueError("invalid paths")
        if not isinstance(record["archive_paths"], list) or not _contained(record["archive_paths"], record["paths"]):
            raise ValueError("invalid archive paths")
        if not isinstance(record["purpose"], str):
            raise ValueError("invalid purpose")
        return record
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BudgetError(f"storage budget state unreadable or malformed: {path}: {exc}") from exc


def _finish(root, record, status, active=True):
    completed = dict(record, status=status, completed_at=time.time())
    _write(root / "receipts" / f"{record['token']}.json", completed)
    if record["archive_paths"]:
        _write(root / "pending" / f"{record['token']}.json", completed)
    if active:
        (root / "active" / f"{record['token']}.json").unlink()


def _active(root):
    records = []
    for path in sorted((root / "active").glob("*.json")):
        record = _validate(path)
        lease_path = root / "leases" / f"{record['token']}.lock"
        if not lease_path.is_file():
            raise BudgetError(f"storage budget lease missing: {lease_path}")
        lease = _tiers().Lock(lease_path)
        try:
            lease.__enter__()
        except RuntimeError:
            records.append(record)
        else:
            try:
                _finish(root, record, "owner_exited")
            finally:
                lease.__exit__(None, None, None)
    return records


def protected_paths(policy=None):
    """Return lexical and resolved paths of live reservations; errors block movers."""
    policy = _policy() if policy is None else policy
    try:
        root = _root(policy)
        with _mutex(root):
            return sorted({p for row in _active(root) for p in row["paths"]})
    except OSError as exc:
        raise BudgetError(f"storage budget state unavailable: {exc}") from exc


def completed_paths(policy=None):
    """Explicit owned output completion hints; missing/corrupt receipts never expedite moves."""
    policy = _policy() if policy is None else policy
    root = _root(policy)
    found = []
    with _mutex(root):
        for path in (root / "pending").glob("*.json"):
            try:
                record = _validate(path)
                stamp = _number(record["completed_at"], "completed_at")
                found.extend((p, stamp) for p in record["archive_paths"])
            except (BudgetError, KeyError):
                continue
    return found


def acknowledge_archived(source, policy=None):
    """Drop completion hints the mover has archived; the durable receipt stays."""
    policy = _policy() if policy is None else policy
    root = _root(policy)
    moved = _paths([source])
    with _mutex(root):
        for path in (root / "pending").glob("*.json"):
            try:
                record = _validate(path)
            except BudgetError:
                continue
            remaining = [p for p in record["archive_paths"] if not _contained([p], moved)]
            if not remaining:
                path.unlink()
            elif remaining != record["archive_paths"]:
                _write(path, dict(record, archive_paths=remaining))


@contextmanager
def mutation_guard(paths, policy=None):
    """Hold admission gate across mover mutations; refuse overlapping active jobs."""
    policy = _policy() if policy is None else policy
    try:
        root = _root(policy)
        targets = _paths(paths)
        with _mutex(root):
            protected = [p for row in _active(root) for p in row["paths"]]
            if any(_contained([a], [b]) or _contained([b], [a]) for a in targets for b in protected):
                raise BudgetError("storage mutation overlaps an active reservation")
            yield
    except OSError as exc:
        raise BudgetError(f"storage budget state unavailable: {exc}") from exc


@dataclass
class Reservation:
    token: str
    required_gb: float
    paths: list[str]
    inherited: bool = False
    policy: dict | None = None

    def child_env(self):
        """Full environment for nested wrappers; token is validated against live state."""
        environment = dict(os.environ, **{TOKEN_ENV: self.token})
        if self.policy is not None:
            environment["LSS_STORAGE_POLICY_JSON"] = json.dumps(self.policy)
            temporary = self.policy.get('temp_root')
            if temporary:
                from dev_hot_paths import require_hot
                require_hot(temporary, 'development temporary files',
                            self.policy.get('fast_volumes', self.policy['hot_volume']))
                Path(temporary).mkdir(parents=True, exist_ok=True)
                environment.update(TEMP=temporary, TMP=temporary)
        return environment


def reservation_volumes(paths, policy):
    """Charge each actual SSD that a protected job can grow on. A C: junction
    resolving to D: is charged to D:. Legacy unknown reservations charge all
    requested volumes conservatively; HDD input protection consumes no SSD budget.
    """
    if not policy.get('fast_volumes'):
        return [policy['hot_volume']]
    volumes = policy['fast_volumes']
    found = [v for v in volumes if any(_tiers().on_volume(Path(p), v) for p in paths)]
    return found or [policy.get('work_volume', policy['hot_volume'])]


def _contained(paths, parents):
    return all(any(Path(p).is_relative_to(Path(parent)) for parent in parents) for p in paths)


def _live_policy(policy):
    """Custom policy routing never wakes the live machine's maintenance task."""
    if os.name != "nt" or os.environ.get("LSS_STORAGE_POLICY") or os.environ.get("LSS_STORAGE_POLICY_JSON"):
        return False
    try:
        shipped = _tiers().load_policy()
    except FileNotFoundError:
        return False  # No installed policy can authorize the live maintenance task.
    for key in ("local_state_root", "hot_volume", "archive_root"):
        if key not in policy or key not in shipped:
            return False
        if Path(os.path.expandvars(policy[key])).resolve() != Path(os.path.expandvars(shipped[key])).resolve():
            return False
    return True


def nudge(policy, *, reason, log=print):
    """Request the existing OS task, at most once per shared 30-second interval.

    This is a best-effort hint, never a mover subprocess or an admission override.
    nudge.json is bounded diagnostic status; durable job receipts are retained.
    Sleep mode returns quiet_deferred before any task, lock or storage discovery.
    """
    path = None
    result = None
    from dev_quiet_mode import quiet_requested
    if quiet_requested():
        return {"status": "quiet_deferred"}
    try:
        if not _live_policy(policy):
            return {"status": "custom_policy_skipped"}
        root = _root(policy)
        path = root / "nudge.json"
        with _mutex(root):
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                last = _number(previous["requested_at"], "nudge timestamp")
                if time.time() - last < 30:
                    return {"status": "throttled"}
            result = dict(status="requested", requested_at=time.time(), reason=reason)
            _write(path, result)
        # Never launch under the registry mutex: the task may immediately scan it.
        completed = subprocess.run(["schtasks.exe", "/Run", "/TN", "LSS Storage Tiers"],
                                   capture_output=True, text=True, timeout=10,
                                   encoding="oem" if os.name == "nt" else "utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        result.update(status="requested" if completed.returncode == 0 else "failed",
                      returncode=completed.returncode,
                      detail=(completed.stderr or completed.stdout or "").strip()[:500])
        with _mutex(root):
            _write(path, result)
        if completed.returncode:
            log(f"storage maintenance nudge failed: {result['detail']}")
        return result
    except (OSError, ValueError, KeyError, TypeError, BudgetError, subprocess.TimeoutExpired) as exc:
        log(f"storage maintenance nudge unavailable: {exc}")
        failure = dict(result or {}, status="error", detail=str(exc)[:500])
        if path is not None and result is not None:
            try:
                with _mutex(root):
                    _write(path, failure)
            except (OSError, BudgetError):
                pass
        return failure


@contextmanager
def reserve(required_gb=None, paths=(), purpose="", policy=None, wait=True, poll=15, log=print, archive_paths=(),
            _allow_nudge=True):
    """Wait for free minus outstanding >= floor + estimate, or refuse immediately.

    archive_paths explicitly designate owned output eligible for later archival.
    Nested wrappers reuse a live token only within its estimate and protected paths;
    expansion is refused immediately to prevent waiting on their own parent.
    """
    policy = _policy() if policy is None else policy
    settings = policy.get("storage_budget", {})
    required = _number(settings.get("default_job_gb", 5) if required_gb is None else required_gb, "required_gb")
    floor = _number(settings.get("floor_gb", 30), "floor_gb")
    poll = _number(poll, "poll")
    if poll == 0:
        raise BudgetError("storage budget poll must be positive")
    explicit = _paths(paths)
    volumes = reservation_volumes(explicit, policy)
    archives = _paths(archive_paths)
    if not _contained(archives, explicit):
        raise BudgetError("storage budget archive_paths must be contained in protected paths")
    lease = None
    admitted = None
    root = None
    last_wait = None
    wait_nudged = False
    try:
        root = _root(policy)
        while admitted is None:
            with _mutex(root):
                records = _active(root)
                inherited = os.environ.get(TOKEN_ENV)
                if inherited:
                    parent = next((r for r in records if r["token"] == inherited), None)
                    if parent is not None and required_gb is None:
                        required = parent["required_gb"]
                    if parent is None or required > parent["required_gb"] or not _contained(explicit, parent["paths"]):
                        raise BudgetError("storage budget inherited reservation is stale or insufficient; nested expansion refused")
                    admitted = Reservation(inherited, required, explicit, True, policy)
                    record = dict(schema_version=1, token=uuid.uuid4().hex, parent_token=inherited,
                                  required_gb=0, paths=explicit, archive_paths=archives,
                                  purpose=str(purpose), started_at=time.time(), pid=os.getpid())
                else:
                    checks = []
                    for volume in volumes:
                        free = _number(_tiers().free_gb(volume), 'free_gb')
                        outstanding = sum(r['required_gb'] for r in records
                                          if 'volumes' not in r or volume in r['volumes'])
                        volume_floor = _number(policy.get('volume_limits', {}).get(volume, {}).get('budget_floor_gb', floor), 'volume_floor_gb')
                        checks.append((volume, free, outstanding, volume_floor))
                    if all(free - outstanding >= limit + required for _, free, outstanding, limit in checks):
                        token = uuid.uuid4().hex
                        acquired = _tiers().Lock(root / "leases" / f"{token}.lock")
                        acquired.__enter__()
                        lease = acquired
                        record = dict(schema_version=1, token=token, required_gb=required, paths=explicit,
                                      archive_paths=archives, purpose=str(purpose), started_at=time.time(), pid=os.getpid(),
                                      volumes=volumes)
                        _write(root / "active" / f"{token}.json", record)
                        admitted = Reservation(token, required, explicit, False, policy)
                    else:
                        volume, free, outstanding, limit = next(c for c in checks if c[1] - c[2] < c[3] + required)
                        why = (f"SSD admission waiting on {volume}: free {free:.2f} GB - reserved {outstanding:.2f} GB "
                               f"< floor {limit:g} + estimate {required:g} GB ({purpose})")
                        if not wait:
                            raise BudgetError(why)
            if admitted is None:
                if why != last_wait:
                    log(why)
                    last_wait = why
                if not wait_nudged and _allow_nudge:
                    nudge(policy, reason="capacity_wait", log=log)
                    wait_nudged = True
                time.sleep(poll)
        status = "completed"
        try:
            yield admitted
        except OSError as exc:
            status = "failed"
            raise _JobOSError(exc) from exc  # the job's own error, not the budget state's
        except BaseException:
            status = "failed"
            raise
        finally:
            if not admitted.inherited or archives:
                with _mutex(root):
                    _finish(root, record, status, active=not admitted.inherited)
                if archives and _allow_nudge:
                    nudge(policy, reason="explicit_output_completed", log=log)
    except _JobOSError as wrapped:
        raise wrapped.__cause__
    except OSError as exc:
        raise BudgetError(f"storage budget state unavailable: {exc}") from exc
    finally:
        if lease is not None:
            lease.__exit__(None, None, None)


class _JobOSError(Exception):
    """Carries an OSError raised by the reserved job past the budget's own OSError translation."""


class _Job:
    """Minimal kill-on-close Windows job; no GPU/runtime imports."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        class Basic(ctypes.Structure):
            _fields_ = [("user", ctypes.c_int64), ("job", ctypes.c_int64), ("flags", wintypes.DWORD),
                        ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("active", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k.CreateJobObjectW.restype = wintypes.HANDLE
        self.k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.k.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(f"CreateJobObject failed: {ctypes.get_last_error()}")
        info = Extended()
        info.basic.flags = 0x2000 | 0x800  # KILL_ON_JOB_CLOSE | BREAKAWAY_OK: only a launcher that asks for
        # CREATE_BREAKAWAY_FROM_JOB (hidden_run, a detached maintain) outlives the job; ordinary children stay owned
        if not self.k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise OSError(f"SetInformationJobObject failed: {ctypes.get_last_error()}")

    def assign(self, child):
        if not self.k.AssignProcessToJobObject(self.handle, int(child._handle)):
            raise OSError("AssignProcessToJobObject failed")

    def close(self):
        if self.handle:
            self.k.TerminateJobObject(self.handle, 1)
            self.k.CloseHandle(self.handle)
            self.handle = None


def streams():
    """stdout/stderr to hand a CREATE_NO_WINDOW child, which otherwise gets no standard handles."""
    result = {}
    for name, stream in (("stdout", sys.stdout), ("stderr", sys.stderr)):
        try:
            stream.fileno()
            result[name] = stream
        except (AttributeError, OSError, ValueError):
            pass
    return result


@contextmanager
def owned_child(command, **options):
    """Own a spawned tree through exit, exceptions and Windows controller death."""
    job = None
    child = None
    try:
        flags = options.pop('creationflags', 0)
        if os.name == "nt":
            job = _Job()
            flags |= subprocess.CREATE_NO_WINDOW | 0x4  # suspended until Job assignment
        child = subprocess.Popen(command, creationflags=flags, **options)
        if job is not None:
            job.assign(child)
            import ctypes
            from ctypes import wintypes
            resume = ctypes.WinDLL("ntdll").NtResumeProcess
            resume.argtypes = [wintypes.HANDLE]
            resume.restype = ctypes.c_long
            if resume(int(child._handle)) != 0:
                raise BudgetError("could not resume storage-owned child")
        yield child
    finally:
        if job is not None:
            job.close()
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()


def run(command, reservation):
    """Own the child until exit; Windows job ownership survives runner termination."""
    with owned_child(command, env=reservation.child_env(), **streams()) as child:
        return child.wait()


def _policy():
    if os.environ.get("LSS_STORAGE_POLICY_JSON"):
        try:
            result = json.loads(os.environ["LSS_STORAGE_POLICY_JSON"])
            if not isinstance(result, dict):
                raise ValueError("expected object")
            return result
        except ValueError as exc:
            raise BudgetError(f"storage budget inherited policy malformed: {exc}") from exc
    path = os.environ.get("LSS_STORAGE_POLICY")
    return _tiers().load_policy(Path(path)) if path else _tiers().load_policy()


def command(command, *, paths=(), archive_paths=(), purpose="", required_gb=None):
    """Build the persistent supervisor command; paths protect, archives are explicit ownership."""
    result = [sys.executable, str(Path(__file__).resolve()), "run", "--purpose", purpose]
    if required_gb is not None:
        result += ["--required-gb", str(required_gb)]
    for path in paths:
        result += ["--protect", str(path)]
    for path in archive_paths:
        result += ["--archive", str(path)]
    return result + ["--", *map(str, command)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--required-gb", type=float)
    runner.add_argument("--protect", action="append", default=[])
    runner.add_argument("--archive", action="append", default=[])
    runner.add_argument("--purpose", default="development job")
    runner.add_argument("--policy", type=Path)
    runner.add_argument("--no-wait", action="store_true")
    runner.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    try:
        policy = _tiers().load_policy(args.policy) if args.policy else _policy()
        with reserve(args.required_gb, [str(Path.cwd()), *args.protect], args.purpose, policy,
                     wait=not args.no_wait, archive_paths=args.archive, _allow_nudge=args.policy is None) as reservation:
            code = run(command, reservation)
            if code:
                raise SystemExit(code)
        return 0
    except BudgetError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
