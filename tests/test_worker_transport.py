"""Windowsの実名前付きパイプで接続継続・直列処理・UIなしの起動を確認する。"""
from multiprocessing import connection
from multiprocessing.context import AuthenticationError
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest

from lss.core import client, worker
from lss.core.jobs import SeparationQueue


def _connect(address, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return connection.Client(address, authkey=worker.AUTHKEY)
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("試験ワーカーへ接続できません")


def test_probe_and_disconnect_do_not_interrupt_sequential_processing(monkeypatch):
    first_started = threading.Event()
    second_started = threading.Event()
    release = threading.Event()
    probed = threading.Event()
    calls = []

    def handle(job):
        calls.append(job["id"])
        if job["id"] == 1:
            first_started.set()
            assert release.wait(5)
        else:
            second_started.set()
        return {"status": "ok", "id": job["id"]}

    monkeypatch.setattr(worker, "handle_job", handle)
    address = fr"\\.\pipe\lss-transport-{uuid4().hex}"
    threading.Thread(target=worker.serve, args=(address,), daemon=True).start()
    first = _connect(address)
    second = None
    try:
        first.send({"id": 1})
        assert first_started.wait(2)

        def probe():
            if client.is_worker_running(address):
                probed.set()

        threading.Thread(target=probe, daemon=True).start()
        assert probed.wait(2), "分離中も接続確認に応答する"
        second = _connect(address)
        second.send({"id": 2})
        assert not second_started.wait(0.1), "分離は並列に実行しない"
        first.close()  # 結果の受信前に呼び出し元が切断する。
        release.set()
        assert second.poll(5)
        assert second.recv() == {"status": "ok", "id": 2}
        assert calls == [1, 2]
        assert client.is_worker_running(address)
    finally:
        release.set()
        first.close()
        if second is not None:
            second.close()


def test_authentication_failure_does_not_stop_worker(monkeypatch):
    monkeypatch.setattr(worker, "handle_job", lambda job: {"status": "ok"})
    address = fr"\\.\pipe\lss-transport-{uuid4().hex}"
    threading.Thread(target=worker.serve, args=(address,), daemon=True).start()
    _connect(address).close()
    with pytest.raises(AuthenticationError):
        connection.Client(address, authkey=b"wrong")
    conn = _connect(address)
    try:
        conn.send({})
        assert conn.poll(5)
        assert conn.recv()["status"] == "ok"
    finally:
        conn.close()


def test_protocol_ping_responds_while_processing_lock_is_busy(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def handle(job):
        started.set()
        assert release.wait(5)
        return {"status": "ok"}

    monkeypatch.setattr(worker, "handle_job", handle)
    address = fr"\\.\pipe\lss-ping-{uuid4().hex}"
    threading.Thread(target=worker.serve, args=(address,), daemon=True).start()
    busy = _connect(address)
    try:
        busy.send({"id": 1})
        assert started.wait(2)
        started_at = time.monotonic()
        assert client.is_worker_running(address)
        assert time.monotonic() - started_at < 1.0
    finally:
        release.set()
        busy.close()


def test_real_worker_entrypoint_and_queue_without_ui(tmp_path):
    address = fr"\\.\pipe\lss-process-{uuid4().hex}"
    proc = subprocess.Popen(
        [sys.executable, str(client.REPO_ROOT / "lss_gui.py"), "--worker", "--address", address],
        cwd=client.REPO_ROOT, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        _connect(address, timeout=40).close()
        queue = SeparationQueue(
            lambda job: client.dispatch_job(job, address=address), state_path=tmp_path / "queue.json",
        )
        queue.add([tmp_path / "missing-a.wav", tmp_path / "missing-b.wav"], "unused", "Vocals")
        queue.run()
        assert [item.status for item in queue.snapshot()] == ["failed", "failed"]
        assert proc.poll() is None
        assert client.is_worker_running(address)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_opt_in_progress_stream_preserves_final_response(monkeypatch):
    def handle(job, progress=None):
        assert job["id"] == 7
        assert progress is not None
        progress({"phase": "separating", "completed_bytes": 10, "total_bytes": 20})
        return {"status": "ok", "id": 7}

    monkeypatch.setattr(worker, "handle_job", handle)
    address = fr"\\.\pipe\lss-progress-{uuid4().hex}"
    threading.Thread(target=worker.serve, args=(address,), daemon=True).start()
    events = []

    result = client.dispatch_job(
        {"id": 7}, address=address, connect_timeout=5, on_progress=events.append,
    )

    assert events == [{"phase": "separating", "completed_bytes": 10, "total_bytes": 20}]
    assert result == {"status": "ok", "id": 7}


def _serve_in_thread(address):
    thread = threading.Thread(target=worker.serve, args=(address,), daemon=True)
    thread.start()
    return thread


def test_shutdown_request_stops_an_idle_worker(monkeypatch):
    """常駐ワーカーを残すとGPUとパイプを掴んだまま生き残る。終了要求で必ず落ちる。"""
    monkeypatch.setattr(worker, "handle_job", lambda job: {"status": "ok"})
    address = fr"\\.\pipe\lss-shutdown-{uuid4().hex}"
    thread = _serve_in_thread(address)
    _connect(address).close()

    response = client.request_worker_shutdown(address)

    assert response is not None and response["status"] == "ok"
    assert response["busy"] is False
    thread.join(5)
    assert not thread.is_alive(), "acceptで待ち続けず終了する"
    assert not client.is_worker_running(address)


def test_shutdown_waits_for_the_running_job_before_exiting(monkeypatch):
    """実行中の1件は公開まで仕上げてから落ちる。途中で殺さない。"""
    started = threading.Event()
    release = threading.Event()
    finished = []

    def handle(job):
        started.set()
        assert release.wait(5)
        finished.append(job)
        return {"status": "ok"}

    monkeypatch.setattr(worker, "handle_job", handle)
    address = fr"\\.\pipe\lss-shutdown-busy-{uuid4().hex}"
    thread = _serve_in_thread(address)
    running = _connect(address)
    try:
        running.send({"cmd": "separate"})
        assert started.wait(5)

        response = client.request_worker_shutdown(address)
        assert response is not None and response["busy"] is True
        thread.join(0.5)
        assert thread.is_alive(), "処理中の1件を打ち切らない"

        release.set()
        assert running.recv() == {"status": "ok"}
        thread.join(5)
        assert not thread.is_alive()
        assert finished, "実行中のジョブは最後まで走る"
    finally:
        running.close()


def test_shutdown_request_is_harmless_when_no_worker_is_running():
    address = fr"\\.\pipe\lss-shutdown-absent-{uuid4().hex}"
    assert client.request_worker_shutdown(address) is None
