"""GPUを使わず、Windowsの実プロセス・実mutexで排他所有と異常終了回復を確認する。"""
import os
from pathlib import Path
import subprocess
import sys
import threading
from uuid import uuid4

import pytest

from lss.core import gpu_lease


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windowsの実mutexを確認")


def child(device, body):
    return subprocess.Popen(
        [sys._base_executable, "-u", "-c", "from lss.core.gpu_lease import acquire\n" + body, str(device)],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def device_id():
    # 実際のCUDAは触らず、同時テストとも衝突しない排他名を使う。
    return 100000 + (uuid4().int % 100000000)


def test_other_process_waits_and_acquires_after_owner_releases():
    device = device_id()
    lease = gpu_lease.acquire(device)
    process = child(device, "import sys\nprint('ready')\nwith acquire(int(sys.argv[1]), on_wait=lambda: print('waiting')):\n print('acquired')\n")
    try:
        assert process.stdout.readline().strip() == "ready"
        assert process.stdout.readline().strip() == "waiting"
        assert process.poll() is None
        lease.close()
        output, error = process.communicate(timeout=10)
        assert process.returncode == 0, error
        assert output.strip() == "acquired"
        lease.close()  # 二重終了は別の所有権を解放しない。
    finally:
        lease.close()
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_named_mutex_is_reentrant_on_the_same_thread_without_cuda():
    device = device_id()
    outer = gpu_lease.acquire(device)
    try:
        inner = gpu_lease.acquire(device)
        try:
            assert not outer.abandoned and not inner.abandoned
        finally:
            inner.close()
    finally:
        outer.close()


def test_killed_owner_does_not_leave_a_permanent_gpu_lock():
    device = device_id()
    process = child(device, "import sys\nwith acquire(int(sys.argv[1])):\n print('held')\n sys.stdin.readline()\n")
    waiting, completed = threading.Event(), threading.Event()
    outcomes, failures = [], []

    def contender():
        try:
            with gpu_lease.acquire(device, on_wait=waiting.set) as lease:
                outcomes.append(lease.abandoned)
        except BaseException as exc:
            failures.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=contender, daemon=True)
    try:
        assert process.stdout.readline().strip() == "held"
        thread.start()
        assert waiting.wait(5)
        assert not completed.is_set()
        process.kill()
        process.communicate(timeout=10)
        assert completed.wait(5)
        assert not failures
        assert outcomes == [True]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        if thread.ident is not None:
            thread.join(timeout=5)


def test_failure_in_owner_body_releases_to_a_different_thread():
    device = device_id()
    with pytest.raises(ValueError):
        with gpu_lease.acquire(device):
            raise ValueError("推論失敗")
    completed = threading.Event()

    def contender():
        with gpu_lease.acquire(device):
            completed.set()

    thread = threading.Thread(target=contender, daemon=True)
    thread.start()
    assert completed.wait(5)
    thread.join(timeout=5)


def test_wait_failure_closes_native_handle_without_unlocked_fallback(monkeypatch):
    closed = []

    class API:
        def CreateMutexW(self, *args):
            return 7

        def WaitForSingleObject(self, *args):
            return 0xFFFFFFFF

        def CloseHandle(self, handle):
            closed.append(handle)
            return True

    monkeypatch.setattr(gpu_lease, "_kernel32", API)
    with pytest.raises(gpu_lease.GpuLeaseUnavailableError, match="WaitForSingleObject"):
        gpu_lease.acquire()
    assert closed == [7]


def test_wrong_thread_cannot_discard_the_owner_handle():
    lease = gpu_lease.acquire(device_id())
    errors = []

    def wrong_thread():
        try:
            lease.close()
        except gpu_lease.GpuLeaseUnavailableError as exc:
            errors.append(exc)

    thread = threading.Thread(target=wrong_thread)
    try:
        thread.start()
        thread.join(timeout=5)
        assert len(errors) == 1
        assert lease._handle is not None
    finally:
        lease.close()
