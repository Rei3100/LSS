"""Run one bounded CPU follow-up after the owned full-source SFC task succeeds."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root, dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root, dev_evidence_root

import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUCCESS = "DEV_ONLY_FULL_SOURCE_SFC_REGEN_COMPLETE"
POLL_SECONDS = 60
MAX_ERROR = 1000


def decide(run: dict[str, Any] | None, launcher: dict[str, Any] | None,
           process_alive: bool) -> str:
    """Pure completion gate: wait, refuse, or invoke the sole CPU follow-up."""
    if not isinstance(run, dict) or not isinstance(launcher, dict):
        return "retry_receipts"
    if launcher.get("status") == "FAILED" or launcher.get("exit_code") not in (None, 0):
        return "refuse"
    if run.get("status") == "DEV_ONLY_FULL_SOURCE_SFC_REGEN_FAILED":
        return "refuse"
    if launcher.get("status") == "COMPLETED":
        if launcher.get("exit_code") != 0 or run.get("status") != SUCCESS or run.get("gpu_executed") is not True:
            return "refuse"
        return "invoke"
    if launcher.get("status") != "RUNNING":
        return "refuse"
    if run.get("status") == SUCCESS and run.get("gpu_executed") is True:
        return "wait_for_launcher_finalization" if not process_alive else "wait"
    if not process_alive:
        return "refuse"
    if run.get("status") not in ("RUNNING", None):
        return "refuse"
    return "wait"


def bounded_decision(run: dict[str, Any] | None, launcher: dict[str, Any] | None,
                     process_alive: bool, transient_polls: int,
                     finalization_waits: int) -> tuple[str, int, int]:
    action = decide(run, launcher, process_alive)
    if action == "retry_receipts":
        if transient_polls >= 2:
            return "refuse", transient_polls, finalization_waits
        return action, transient_polls + 1, finalization_waits
    transient_polls = 0
    if action == "wait_for_launcher_finalization":
        if finalization_waits >= 2:
            return "refuse", transient_polls, finalization_waits
        return action, transient_polls, finalization_waits + 1
    return action, transient_polls, finalization_waits


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def process_handle(pid: int):
    """Open and retain a Windows process handle so a reused PID cannot fool polling."""
    if os.name != "nt":
        raise RuntimeError("This coordinator requires Windows process handles")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess failed for owned launcher child")
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel, handle


def process_created_utc(kernel, handle) -> datetime:
    class FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, ctypes.POINTER(FileTime),
                                       ctypes.POINTER(FileTime), ctypes.POINTER(FileTime),
                                       ctypes.POINTER(FileTime))
    kernel.GetProcessTimes.restype = wintypes.BOOL
    created = FileTime()
    exited = FileTime()
    kernel_time = FileTime()
    user_time = FileTime()
    if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                  ctypes.byref(kernel_time), ctypes.byref(user_time)):
        raise OSError(ctypes.get_last_error(), "GetProcessTimes failed for owned launcher child")
    ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
    return datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks / 10)


def parse_expected_creation(value: str) -> datetime:
    """Accept ISO-8601 UTC or CIM DMTF datetime as supplied from the process inventory."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("expected process creation timestamp must include a timezone")
        return parsed.astimezone(timezone.utc)
    except ValueError:
        if len(value) < 25 or value[14] != ".":
            raise ValueError("expected process creation time must be timezone-qualified ISO or CIM DMTF")
        base = datetime.strptime(value[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        micros = int((value[15:21] + "000000")[:6])
        sign = value[21]
        offset_minutes = int(value[22:25])
        if sign not in ("+", "-"):
            raise ValueError("invalid CIM DMTF timezone sign")
        offset = timedelta(minutes=offset_minutes * (1 if sign == "+" else -1))
        return base.replace(microsecond=micros) - offset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def handle_alive(kernel, handle) -> bool:
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    result = kernel.WaitForSingleObject(handle, 0)
    if result == 0:
        return False
    if result == 0x102:
        return True
    raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")


def set_below_normal(process: subprocess.Popen) -> None:
    if os.name != "nt":
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.SetPriorityClass.restype = wintypes.BOOL
    if not kernel.SetPriorityClass(int(process._handle), 0x00004000):
        raise OSError(ctypes.get_last_error(), "Could not set follow-up BelowNormal priority")


def append_event(log: Path, receipt: Path, row: dict[str, Any]) -> None:
    row["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(receipt)


def follow(args) -> int:
    run_path = args.run_dir / "run-receipt.json"
    launcher_path = args.launcher_receipt
    receipt = args.coordinator_dir / "receipt.json"
    log = args.coordinator_dir / "events.jsonl"
    if args.run_dir.resolve().parent != dev_state_root(ROOT).resolve():
        raise ValueError("run directory must be a direct child of the development state root")
    if args.coordinator_dir.resolve().parent != dev_state_root(ROOT).resolve():
        raise ValueError("coordinator directory must be a direct child of the development state root")
    if receipt.exists():
        raise ValueError(f"coordinator receipt already exists: {receipt}")

    launcher = read_json(launcher_path)
    if not launcher or launcher.get("status") != "RUNNING":
        raise ValueError("launcher must be live at coordinator startup")
    pid = launcher.get("child_pid")
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("live launcher receipt has no valid child PID")
    if pid != args.expected_pid:
        raise ValueError("launcher child PID differs from authoritative expected PID")
    kernel, handle = process_handle(pid)
    try:
        actual_created = process_created_utc(kernel, handle)
        expected_created = parse_expected_creation(args.expected_process_created)
        if abs((actual_created - expected_created).total_seconds()) > 0.01:
            raise ValueError("launcher child creation time differs from authoritative process identity")
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    helper_script = dev_evidence_root(ROOT) / "scene-web-evidence" / "peer_full_source_memory_probe.py"
    helper_hash = file_sha256(helper_script)
    row = {"schema_version": 1, "status": "POLLING", "child_pid": pid,
           "run_receipt": str(run_path), "launcher_receipt": str(launcher_path),
           "process_created_utc": actual_created.isoformat(),
           "helper_sha256_at_start": helper_hash, "poll_seconds": POLL_SECONDS}
    try:
        transient_receipt_polls = 0
        finalization_waits = 0
        while True:
            alive = handle_alive(kernel, handle)
            run = read_json(run_path)
            launcher = read_json(launcher_path)
            action, transient_receipt_polls, finalization_waits = bounded_decision(
                run, launcher, alive, transient_receipt_polls, finalization_waits)
            row["gate"] = action
            append_event(log, receipt, row)
            if action == "refuse":
                row["status"] = "REFUSED"
                row["error"] = "Native process or completion receipts failed the success gate"
                append_event(log, receipt, row)
                return 2
            if action == "invoke":
                break
            time.sleep(POLL_SECONDS)

        args.coordinator_dir.mkdir(parents=True, exist_ok=True)
        if file_sha256(helper_script) != helper_hash:
            raise ValueError("CPU helper script changed while coordinator was waiting")
        command = [sys.executable, str(helper_script),
                   "--run-dir", str(args.run_dir), "--source-commit", args.source_commit,
                   "--peer-current-best", str(args.peer_current_best),
                   "--peer-state-root", str(args.peer_state_root),
                   "--private-state", str(args.private_state), "--output", str(args.output),
                   "--quiet-seconds", "60"]
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
        child_log = args.coordinator_dir / "followup.stdout.log"
        with child_log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                set_below_normal(child)
                child.wait()
            except BaseException:
                child.kill()
                child.wait()
                raise
        with child_log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_ERROR * 4))
            output_tail = stream.read().decode("utf-8", errors="replace")[-MAX_ERROR:]
        row.update({"status": "COMPLETED" if child.returncode == 0 else "FAILED",
                    "followup_exit_code": child.returncode,
                    "followup_output_tail": output_tail, "command": command})
        append_event(log, receipt, row)
        return int(child.returncode or 0)
    except BaseException as exc:
        row.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR]})
        append_event(log, receipt, row)
        return 2
    finally:
        kernel.CloseHandle(handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launcher-receipt", type=Path, required=True)
    parser.add_argument("--expected-pid", type=int, required=True)
    parser.add_argument("--expected-process-created", required=True,
                        help="authoritative CIM DMTF or timezone-qualified ISO process creation timestamp")
    parser.add_argument("--coordinator-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--peer-current-best", type=Path, required=True)
    parser.add_argument("--peer-state-root", type=Path, required=True)
    parser.add_argument("--private-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return follow(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {str(exc)[:MAX_ERROR]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
