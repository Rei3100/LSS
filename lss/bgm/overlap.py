"""GPU/CPU overlap inside one BGM job: CPU-only work whose inputs are already decided runs on a small thread pool
(the CPU lane) while the job's GPU stages continue. Design and proof: `efficiency-evidence/OVERLAP-DESIGN.md`,
`cloud-reports/gpu-cpu-overlap.md`.

Rules the callers keep (each one is a reason the lane cannot change an output byte):

* Lane tasks never touch torch/CUDA. They read, write, verify and hash files with libsndfile, mutagen and hashlib.
* A task only does work whose inputs no later stage can change; it writes the same bytes the serial path writes,
  and every result is joined in plan order by the one thread that writes records (receipts, ``episode-cues.json``,
  ``run.json``, CurrentBest).
* Free-space checks inside a task reserve their bytes in the shared ledger (`lss.output.writer.output_reservations`)
  until the task ends, so two tasks cannot pass the same check.
* The lane is closed inside the job's existing cleanup scope. ``close(failed=True)`` cancels tasks that have not
  started, waits for running ones and removes every file a task created, so a stopped or failed job leaves no new
  file behind.

This module is not a Recipe input (`lss.application._BGM_IMPLEMENTATION`) and not a runtime adapter: it decides
nothing about the audio. ``LSS_BGM_OVERLAP=0`` turns the lane off; the callers then take the unchanged serial path.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

ENV = "LSS_BGM_OVERLAP"
WORKERS_ENV = "LSS_BGM_OVERLAP_WORKERS"
DEFAULT_WORKERS = 2
MAXIMUM_WORKERS = 8

# Test seam: called at the start of every task with its label (proof tests inject random delays here).
_TASK_HOOK: Callable[[str], None] | None = None


def enabled() -> bool:
    """Kill switch. Anything but ``0`` (or unset) keeps the overlap on."""
    return (os.environ.get(ENV) or "1").strip() != "0"


def workers() -> int:
    try:
        value = int((os.environ.get(WORKERS_ENV) or str(DEFAULT_WORKERS)).strip())
    except ValueError:
        value = DEFAULT_WORKERS
    return max(1, min(MAXIMUM_WORKERS, value))


class Lane:
    """A bounded CPU thread pool for one job stage. Use ``open_lane()``; ``None`` means run serially."""

    def __init__(self, max_workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lss-bgm-cpu")
        self._lock = threading.Lock()
        self._futures: list[Future] = []
        self._created: list[Path] = []
        self._closed = False
        self.busy_seconds = 0.0
        self.task_count = 0

    def submit(self, label: str, fn: Callable, *args, **kwargs) -> Future:
        from lss.output import writer

        def run():
            started = time.perf_counter()
            try:
                if _TASK_HOOK is not None:
                    _TASK_HOOK(label)
                with writer.output_reservations():
                    return fn(*args, **kwargs)
            finally:
                with self._lock:
                    self.busy_seconds += time.perf_counter() - started
                    self.task_count += 1

        with self._lock:
            if self._closed:
                raise RuntimeError("the CPU lane is closed")
            future = self._executor.submit(run)
            self._futures.append(future)
        return future

    def created(self, path: Path) -> None:
        """A task wrote ``path`` (it did not exist before). Removed again if the job fails."""
        with self._lock:
            self._created.append(Path(path))

    def created_files(self) -> list[Path]:
        with self._lock:
            return list(self._created)

    def close(self, *, failed: bool) -> None:
        """Join every task. On failure, cancel what has not started and remove what the tasks created."""
        with self._lock:
            self._closed = True
            futures = list(self._futures)
        if failed:
            for future in futures:
                future.cancel()
        self._executor.shutdown(wait=True)
        if failed:
            for path in self.created_files():
                path.unlink(missing_ok=True)

    def __enter__(self) -> "Lane":
        return self

    def __exit__(self, kind, value, traceback) -> None:
        self.close(failed=kind is not None)


def result_or_none(future: Future | None):
    """A task's result, or ``None`` when there is no task or it failed. The lane is advisory: the caller then does
    the same work itself, serially, so a lane failure can neither fail a job nor change what it writes."""
    if future is None:
        return None
    try:
        return future.result()
    except Exception:  # noqa: BLE001 - the serial path redoes the work and raises its own error if it is real
        return None


def open_lane() -> Lane | None:
    return Lane(workers()) if enabled() else None
