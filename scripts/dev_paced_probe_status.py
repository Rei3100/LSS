"""Compact status reader for DEV paced worker runs; never initializes torch or CUDA."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def process_state(pid):
    if os.name != "nt":
        return "unknown"
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = api.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return "unknown_inaccessible"
    try:
        code = wintypes.DWORD()
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
            return "unknown"
        return "running" if code.value == 259 else f"exited_{code.value}"
    finally:
        api.CloseHandle(handle)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


def progress_rows(path: Path, tail_bytes: int = 2_000_000) -> list[dict]:
    """Read a bounded tail, ignoring a possibly partial first or last JSONL row."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - tail_bytes))
            data = stream.read()
    except OSError:
        return []
    lines = data.splitlines()
    if size > tail_bytes and lines:
        lines = lines[1:]
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def snapshot(run_dir: Path, allowance_path: Path, *, now: float | None = None,
             process_probe=process_state) -> dict:
    now = time.time() if now is None else now
    receipt_path = run_dir / "run-receipt.json"
    progress_path = run_dir / "progress.jsonl"
    receipt = read_json(receipt_path) or {}
    rows = progress_rows(progress_path)
    events = [row for row in rows if row.get("phase") == "forward_rest"]
    elapsed = [row.get("elapsed_seconds") for row in rows
               if isinstance(row.get("elapsed_seconds"), (int, float))
               and not isinstance(row.get("elapsed_seconds"), bool)]
    completed_at = [row.get("completed_at") for row in events
                    if isinstance(row.get("completed_at"), (int, float))
                    and not isinstance(row.get("completed_at"), bool)]
    deltas = [b - a for a, b in zip(completed_at, completed_at[1:])]
    memory_rows = [row.get("cuda_memory_observation") for row in events
                   if isinstance(row.get("cuda_memory_observation"), dict)]
    latest_memory = memory_rows[-1] if memory_rows else None
    last_row = rows[-1] if rows else None
    last_wall = last_row.get("wall_time") if last_row else None
    pid = (receipt.get("lease") or {}).get("pid")
    process = process_probe(pid) if isinstance(pid, int) else "unknown"
    allowance = read_json(allowance_path) or {}
    windows = allowance.get("windows", {})
    quota = {key: {name: row.get(name) for name in (
        "used_percent", "resets_at", "seconds_to_reset", "permitted_percent_per_minute",
        "relative_wallclock_rate_multiplier", "unknown_rate_probe_fraction",
        "rate_status", "recent_percent_per_minute", "planning_rate_upper_bound_percent_per_minute")}
              for key, row in windows.items() if isinstance(row, dict)}
    gpu_observed = bool(events)
    return {
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "receipt_status": receipt.get("status"),
        "receipt_gpu_executed_snapshot": receipt.get("gpu_executed"),
        "gpu_execution_observed_from_progress": gpu_observed,
        "gpu_execution_observation": "forward_rest progress events observed" if gpu_observed else "unknown_no_forward_event",
        "pid": pid,
        "process_visibility": process,
        "progress": {
            "path": str(progress_path.resolve()),
            "event_count": len(events),
            "observed_inter_forward_intervals": max(0, len(completed_at) - 1),
            "latest_elapsed_seconds": elapsed[-1] if elapsed else None,
            "inter_forward_wallclock_gaps_seconds": deltas,
            "minimum_observed_inter_forward_wallclock_gap_seconds": min(deltas) if deltas else None,
            "last_phase": last_row.get("phase") if last_row else None,
            "last_progress_age_seconds": max(0., now - last_wall) if isinstance(last_wall, (int, float)) else None,
            "latest_cuda_memory_observation": latest_memory,
        },
        "allowance": {"status": allowance.get("status"), "windows": quota,
                      "relative_wallclock_rate_multiplier": allowance.get("relative_wallclock_rate_multiplier"),
                      "configured_unknown_rate_probe_fraction": allowance.get("configured_unknown_rate_probe_fraction")},
        "scope": "local progress and quota metadata only; no torch/CUDA/audio/model access; progress events are emitted at rest start, so inter-forward wall-clock gaps include inference and scheduling time and do not certify monotonic rest duration",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", type=int, default=0, help="watch duration in seconds; 0 writes one snapshot")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    state_root = dev_state_root(ROOT).resolve()
    output = args.output.resolve()
    if state_root not in run_dir.parents:
        parser.error("--run-dir must be beneath the development state root")
    if state_root not in output.parents:
        parser.error("--output must be beneath the development state root")
    if args.watch < 0 or args.interval < 1:
        parser.error("watch must be nonnegative and interval must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    stop_file = output.parent / "stop"
    allowance_path = state_root / "development-allowance" / "latest.json"
    deadline = time.monotonic() + args.watch
    previous = None
    while True:
        value = snapshot(run_dir, allowance_path)
        encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if encoded != previous:
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(encoded, encoding="utf8")
            temporary.replace(output)
            previous = encoded
        if (not args.watch or stop_file.exists() or value.get("receipt_status") not in {"RUNNING", None}
                or time.monotonic() >= deadline):
            break
        time.sleep(min(args.interval, max(0., deadline - time.monotonic())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
