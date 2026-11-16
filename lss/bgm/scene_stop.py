"""Cooperative stop request for the hidden scene-naming worker (deploy / rollback replace its files).

A deploy or rollback writes ``<native root>/stop-request`` (``deploy-to-freesoft.ps1``: ``$deployRoot\\scene-native``),
waits a bounded time for ``LSS.exe --scene-naming*`` processes to exit, and only then force-stops what is left and
moves runtime files. A worker honours a request only if it was written at or after the moment the worker enabled
stop checks (``enable``), so a request left behind by an interrupted deploy never stops a later worker. Checks run
between cues, between download blocks, inside long waits (retry sleep, install lock, ASR process, LLM start-up)
and between hashing blocks. A stop is not a failure: no retry delay or attempt is recorded for it, a resumable
``.part`` download prefix is kept, and the partially-named source resumes on the next eligible run.

``request_and_wait`` is the tested mirror of the PowerShell sequence (request, poll, bounded timeout).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

STOP_REQUEST = "stop-request"
DEPLOY_WAIT_SECONDS = 60.0

_enabled_at: float | None = None


class StopRequested(Exception):
    """Deliberately not a RuntimeError/OSError: the naming paths that turn those into recorded attempt failures
    (``ASR_UNAVAILABLE``, transient errors, bootstrap retry delays) must let a stop pass through unrecorded."""


def request_path(root: Path | None = None) -> Path:
    if root is None:
        from lss.bgm import scene_native

        root = scene_native.native_root()
    return Path(root) / STOP_REQUEST


def enable() -> None:
    """Called by worker entry points; the first call fixes the time from which requests apply to this process."""
    global _enabled_at
    if _enabled_at is None:
        _enabled_at = time.time()


def stop_requested(root: Path | None = None) -> bool:
    if _enabled_at is None:
        return False
    try:
        written = request_path(root).stat().st_mtime
    except OSError:
        return False
    # One second of slack for file-system timestamp granularity; a stale request is minutes or more older.
    return written >= _enabled_at - 1.0


def check(root: Path | None = None) -> None:
    if stop_requested(root):
        raise StopRequested("scene_worker_stop_requested")


def wait(seconds: float, *, root: Path | None = None, step: float = 1.0, sleep=None) -> None:
    """``time.sleep`` in ``step`` slices that ends early (raising ``StopRequested``) when a stop is requested."""
    sleep = sleep or time.sleep
    remaining = max(0.0, float(seconds))
    while True:
        check(root)
        if remaining <= 0:
            return
        slice_ = min(step, remaining)
        sleep(slice_)
        remaining -= slice_


def request(root: Path) -> Path:
    path = request_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"requested_at={time.time():.3f} pid={os.getpid()}\n", encoding="ascii")
    return path


def clear(root: Path) -> None:
    request_path(root).unlink(missing_ok=True)


def request_and_wait(root: Path, workers_running, *, timeout: float = DEPLOY_WAIT_SECONDS, poll: float = 0.5,
                     sleep=time.sleep, clock=time.monotonic) -> dict:
    """Request a stop, then wait until ``workers_running()`` is false or ``timeout`` passes.

    The caller force-stops whatever remains after a timeout and clears the request when its file
    replacement is finished (or failed).
    """
    request(root)
    started = clock()
    while True:
        if not workers_running():
            return {"state": "stopped", "waited_seconds": round(clock() - started, 3)}
        if clock() - started >= timeout:
            return {"state": "timeout", "waited_seconds": round(clock() - started, 3)}
        sleep(poll)
