"""Run a development command while holding the same GPU lease as the LSS worker.

    python scripts/gpu_run.py --purpose "ASR probe 137" -- python probe.py arg ...

Waits for the lease (another agent's probe or a real LSS run), shows who holds it, runs the command, and
releases the lease when the command ends. The lease is a named OS mutex (`lss.core.gpu_lease`): if the holder
crashes the OS releases it, so there is no lock file to forget or clean up.

The command runs at idle CPU priority and without a console window, so development jobs never compete with the
user's desktop: lower load never lowers output quality. Outputs are unchanged; `--normal-priority` opts out.

The child must not take the lease itself: a command that calls `gpu_lease.acquire()` (e.g. the validation
`infer_views.py` helpers) waits forever on the mutex this process holds. Run such a command
without gpu_run, or make its acquire a no-op in the child.

Load cap: a long GPU job must not pin the GPU at full load (overnight validation jobs otherwise run it at 99 % and
84 C). The command's whole process tree is suspended and resumed in cycles, `--duty` of each `--period` running
(default 0.25 of 1 s: short bursts, flat load). This throttles any inference code without changing its outputs;
`--duty 1` opts out (only with a stated reason).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOLDER = Path(os.path.expandvars(r"%LOCALAPPDATA%\LSS-dev\gpu-holder.json"))


def _holder() -> str:
    try:
        row = json.loads(HOLDER.read_text(encoding="utf-8"))
        return f"{row['purpose']} (pid {row['pid']}, since {row['since']})"
    except (OSError, ValueError, KeyError):
        return "an LSS worker or an unnamed holder"


def _validate_in_process_command(command: list[str]) -> Path:
    """Allow only this Python executable followed by an existing Python script."""
    if len(command) < 2:
        raise ValueError("--in-process requires: current-python existing-script.py [args...]")
    try:
        requested_python = Path(command[0]).resolve(strict=True)
        current_python = Path(sys.executable).resolve(strict=True)
        script = Path(command[1]).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"--in-process command path is invalid: {exc}") from exc
    if requested_python != current_python:
        raise ValueError("--in-process only accepts the current Python executable")
    if not script.is_file() or script.suffix.lower() != ".py":
        raise ValueError("--in-process only accepts an existing .py script")
    return script


def _run_in_process(command: list[str], script: Path) -> int:
    import runpy

    old_argv = sys.argv
    sys.argv = [str(script), *command[2:]]
    try:
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            if isinstance(code, int):
                return code
            if code:
                print(code, file=sys.stderr)
                return 1
            return 0
        return 0
    finally:
        sys.argv = old_argv


def _tree(root: int) -> list[int]:
    """root and all its descendants (Toolhelp snapshot)."""
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]
    k = ctypes.windll.kernel32
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    parent = {}
    e = Entry()
    e.dwSize = ctypes.sizeof(Entry)
    ok = k.Process32First(snap, ctypes.byref(e))
    while ok:
        parent[e.th32ProcessID] = e.th32ParentProcessID
        ok = k.Process32Next(snap, ctypes.byref(e))
    k.CloseHandle(snap)

    def created(pid: int) -> int | None:
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        ok = k.GetProcessTimes(h, *[ctypes.byref(t) for t in times])
        k.CloseHandle(h)
        return (times[0].dwHighDateTime << 32 | times[0].dwLowDateTime) if ok else None

    # Windows reuses process ids: a process whose recorded parent id now belongs to one of our descendants is not
    # that descendant's child unless it was created after it. Never include this controller or its ancestors
    # (a suspended controller can never resume anything).
    protected, p = set(), os.getpid()
    while p and p not in protected:
        protected.add(p)
        p = parent.get(p, 0)
    out, todo = [root], [root]
    while todo:
        p = todo.pop()
        born = created(p)
        for c, q in parent.items():
            if q != p or c in out or c in protected:
                continue
            child_born = created(c)
            if born is not None and child_born is not None and child_born < born:
                continue
            out.append(c)
            todo.append(c)
    return out


def _quiet_io(pids: list[int], done: set[int]) -> None:
    """Very low I/O and memory priority for every process of the tree (once per process): disk and page traffic
    yield to the desktop; what the process computes and writes is unchanged."""
    if os.name != "nt":
        return
    import ctypes
    k, nt = ctypes.windll.kernel32, ctypes.windll.ntdll
    for pid in pids:
        if pid in done:
            continue
        h = k.OpenProcess(0x0200, False, pid)  # PROCESS_SET_INFORMATION
        if h:
            very_low = ctypes.c_ulong(0)
            nt.NtSetInformationProcess(h, 33, ctypes.byref(very_low), 4)      # ProcessIoPriority: very low
            memory = ctypes.c_ulong(1)
            nt.NtSetInformationProcess(h, 39, ctypes.byref(memory), 4)        # ProcessPagePriority: very low
            k.CloseHandle(h)
        done.add(pid)


def _normal_io(pids: list[int], done: set[int]) -> None:
    """Restore normal CPU/I/O/page priority, including inherited background state."""
    if os.name != 'nt':
        return
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    nt = ctypes.WinDLL('ntdll', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    nt.NtSetInformationProcess.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG]
    nt.NtSetInformationProcess.restype = wintypes.LONG
    for pid in pids:
        # Native runtimes can lower their own startup priority after enumeration.
        # Reassert this explicit performance policy while the owned Job runs.
        handle = k.OpenProcess(0x0200, False, pid)
        if handle:
            try:
                k.SetPriorityClass(handle, 0x20)
                io, memory = wintypes.ULONG(2), wintypes.ULONG(5)
                nt.NtSetInformationProcess(handle, 33, ctypes.byref(io), 4)
                nt.NtSetInformationProcess(handle, 39, ctypes.byref(memory), 4)
            finally:
                k.CloseHandle(handle)
        done.add(pid)


def _signal(pids: list[int], resume: bool) -> None:
    import ctypes
    k, nt = ctypes.windll.kernel32, ctypes.windll.ntdll
    for pid in pids:
        h = k.OpenProcess(0x0800, False, pid)  # PROCESS_SUSPEND_RESUME
        if h:
            (nt.NtResumeProcess if resume else nt.NtSuspendProcess)(h)
            k.CloseHandle(h)


MIN_COMMIT_GB = float(os.environ.get("LSS_MIN_COMMIT_HEADROOM_GB", "4"))
MIN_RAM_GB = float(os.environ.get("LSS_MIN_FREE_RAM_GB", "0"))


def memory_headroom() -> tuple[float, float]:
    """(commit headroom, available RAM) in GB (Windows GlobalMemoryStatusEx; infinite elsewhere)."""
    if os.name != "nt":
        return float("inf"), float("inf")
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                    ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong)]
    status = Status()
    status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail_page / 1e9, status.avail_phys / 1e9


def wait_for_memory(name: str, *, poll: float = 30.0, log=print) -> None:
    """A job starts only while the PC has commit headroom and free RAM (LSS_MIN_COMMIT_HEADROOM_GB, default 4;
    LSS_MIN_FREE_RAM_GB, default 0). Starting a heavy job into a nearly full commit limit pages the desktop and lets
    native allocations fail mid-run; waiting changes when the job runs, never what it computes."""
    shown = False
    while True:
        commit, ram = memory_headroom()
        if commit >= MIN_COMMIT_GB and ram >= MIN_RAM_GB:
            return
        if not shown:
            log(f"[memory] {name} waits: commit headroom {commit:.1f} GB (need {MIN_COMMIT_GB:g}), "
                f"free RAM {ram:.1f} GB (need {MIN_RAM_GB:g})")
            shown = True
        time.sleep(poll)


def _run_capped(child: subprocess.Popen, duty: float, period: float, *, members=None, signal=None, observe=None, background_io=True) -> int:
    quieted: set[int] = set()
    members = members or (lambda: _tree(child.pid))
    send = signal or _signal
    if os.name != "nt":
        return child.wait()
    if duty >= 1:
        while True:
            pids = members()
            if observe is not None:
                observe(pids)
            if background_io:
                _quiet_io(pids, quieted)
            else:
                _normal_io(pids, quieted)
            try:
                return child.wait(timeout=period * 5)
            except subprocess.TimeoutExpired:
                pass
    on, off = period * duty, period * (1 - duty)
    try:
        while True:
            try:
                return child.wait(timeout=on)
            except subprocess.TimeoutExpired:
                pass
            pids = members()
            if observe is not None:
                observe(pids)
            if background_io:
                _quiet_io(pids, quieted)
            else:
                _normal_io(pids, quieted)
            send(pids, resume=False)
            time.sleep(off)
            send(pids, resume=True)
    finally:
        if child.poll() is None:
            send(members(), resume=True)


class _Attached:
    """A running process (not our child) seen through the Popen calls _run_capped uses."""

    def __init__(self, pid: int):
        import ctypes
        self.pid, self._k = pid, ctypes.windll.kernel32
        self._h = self._k.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
        if not self._h:
            raise SystemExit(f"process {pid} not found")

    def wait(self, timeout: float | None = None) -> int:
        if self._k.WaitForSingleObject(self._h, 0xFFFFFFFF if timeout is None else int(timeout * 1000)) != 0:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return 0

    def poll(self) -> int | None:
        try:
            return self.wait(0)
        except subprocess.TimeoutExpired:
            return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--purpose", default="development GPU job")
    parser.add_argument("--in-process", action="store_true",
                        help="run the current Python .py command under this process's GPU lease")
    parser.add_argument("--normal-priority", action="store_true", help="do not lower the command's CPU priority")
    parser.add_argument("--duty", type=float, default=0.25, help="fraction of each period the job runs (1 = no cap)")
    parser.add_argument("--period", type=float, default=1.0, help="suspend/resume cycle in seconds")
    parser.add_argument("--attach", type=int, help="cap an already running process tree (no lease) until it exits")
    parser.add_argument("--owned-job", action="store_true", help="contain and cap asynchronous descendants through a Windows Job")
    parser.add_argument("--cap-receipt", type=Path, help="atomic live membership receipt for the owned Job")
    parser.add_argument("--storage-gb", type=float, help="estimated peak additional SSD space")
    parser.add_argument("--protect", action="append", default=[])
    parser.add_argument("--archive", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.attach:
        return _run_capped(_Attached(args.attach), args.duty, args.period)
    from dev_quiet_mode import GPUDeferred, require_gpu_allowed
    try:
        require_gpu_allowed()
    except GPUDeferred as exc:
        print(str(exc), file=sys.stderr)
        return 3
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    if args.in_process:
        args.script = _validate_in_process_command(command)
    if args.in_process:  # same process, same interpreter: no storage-budget relaunch
        return _run_reserved(args, command, os.environ.copy())
    # Admission happens before taking the GPU lease, so a disk wait cannot monopolize the GPU.
    import storage_budget
    if not os.environ.get(storage_budget.TOKEN_ENV):
        managed = storage_budget.command([sys.executable, str(Path(__file__).resolve()),
                                          *(sys.argv[1:] if argv is None else argv)],
                                         required_gb=args.storage_gb, purpose=args.purpose,
                                         paths=[str(Path.cwd()), *args.protect, *args.archive])
        return subprocess.run(managed, **storage_budget.streams(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
    with storage_budget.reserve(args.storage_gb, paths=[str(Path.cwd()), *args.protect, *args.archive],
                                archive_paths=args.archive, purpose=args.purpose) as storage:
        return _run_reserved(args, command, storage.child_env())


def _run_reserved(args, command, environment) -> int:
    from lss.core import gpu_lease

    waited = []

    def on_wait():
        if not waited:
            print(f"[gpu_run] waiting for the GPU: held by {_holder()}", flush=True)
        waited.append(1)
    with gpu_lease.acquire(on_wait=on_wait):
        wait_for_memory(args.purpose, log=lambda line: print(line, flush=True))
        HOLDER.parent.mkdir(parents=True, exist_ok=True)
        holder = {"purpose": args.purpose, "pid": os.getpid(), "command": command,
                  "since": time.strftime("%Y-%m-%d %H:%M:%S")}
        if args.in_process:
            holder["execution_mode"] = "in_process"
        HOLDER.write_text(json.dumps(holder, ensure_ascii=False), encoding="utf-8")
        owned = None
        try:
            if args.in_process:
                return _run_in_process(command, args.script)
            flags = 0 if os.name != "nt" else subprocess.CREATE_NO_WINDOW | (
                0 if args.normal_priority else subprocess.IDLE_PRIORITY_CLASS)
            exe = str(Path(shutil.which(command[0]) or command[0]).resolve())  # CreateProcess needs a real path
            if getattr(args, 'owned_job', False):
                from strata_process_job import OwnedProcessJob
                owned = OwnedProcessJob()
                child = owned.start([exe, *command[1:]], stdout=sys.stdout, stderr=sys.stderr, env=environment,
                                    normal_priority=args.normal_priority)
                last_receipt = [0.0]

                def observe(pids):
                    target = getattr(args, 'cap_receipt', None)
                    if target is None or time.monotonic() - last_receipt[0] < 5:
                        return
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_suffix(target.suffix + '.tmp')
                    temp.write_text(json.dumps({'controller_pid': os.getpid(), 'launcher_pid': child.pid,
                                     'members': pids, 'ownership': 'windows-job', 'duty': args.duty,
                                     'period': args.period, 'normal_priority': args.normal_priority,
                                     'captured_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                     'metric': 'process runnable duty, not instantaneous GPU utilization'}), encoding='utf-8')
                    temp.replace(target)
                    last_receipt[0] = time.monotonic()

                return _run_capped(child, args.duty, args.period, members=owned.members, signal=owned.signal,
                                   observe=observe, background_io=not args.normal_priority)
            child = subprocess.Popen([exe, *command[1:]], creationflags=flags, stdin=subprocess.DEVNULL,
                                     stdout=sys.stdout, stderr=sys.stderr, env=environment)  # no console: hand the streams over
            return _run_capped(child, args.duty, args.period)
        finally:
            if owned is not None:
                owned.close()  # includes orphaned engines, before releasing the GPU lease
            HOLDER.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
