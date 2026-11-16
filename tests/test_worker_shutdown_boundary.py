"""終了要求後は実行中の1件を保存し、待機中の分離を新たに開始しない。"""
import time
from threading import Event, Lock, Thread

import pytest

from lss.core import worker


class Connection:
    def __init__(self, job):
        self.job = job
        self.received = Event()
        self.responses = []
        self.closed = False

    def recv(self):
        self.received.set()
        return self.job

    def send(self, response):
        self.responses.append(response)

    def close(self):
        self.closed = True


def test_shutdown_preserves_current_job_and_rejects_waiting_job(monkeypatch):
    started, release, stop = Event(), Event(), Event()
    lock = Lock()
    calls = []

    def handle(job):
        calls.append(job["id"])
        started.set()
        assert release.wait(5)
        return {"status": "ok", "outputs": ["saved.flac"]}

    monkeypatch.setattr(worker, "handle_job", handle)
    monkeypatch.setattr(worker, "_wake_listener", lambda *args: True)
    first = Connection({"id": 1})
    waiting = Connection({"id": 2})
    threads = [Thread(target=worker._serve_connection, args=(conn, lock, stop)) for conn in (first, waiting)]
    try:
        threads[0].start()
        assert started.wait(5)
        threads[1].start()
        assert waiting.received.wait(5)
        shutdown = Connection({"cmd": "shutdown"})
        worker._serve_connection(shutdown, lock, stop)
        assert shutdown.responses[0]["busy"] is True
        assert stop.is_set()
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(5)
                assert not thread.is_alive()
    assert calls == [1]
    assert first.responses == [{"status": "ok", "outputs": ["saved.flac"]}]
    assert waiting.responses == [{
        "status": "error", "error_code": "worker_shutting_down",
        "message": "分離ワーカーは終了処理中です。未開始の処理は再開後に実行してください。",
    }]
    assert first.closed and waiting.closed


def test_shutdown_rejects_progress_job_without_loading_model(monkeypatch):
    stop = Event()
    stop.set()
    monkeypatch.setattr(worker, "handle_job", lambda *a, **k: (_ for _ in ()).throw(AssertionError("終了後に分離を開始しない")))
    conn = Connection({"id": 1, "_stream_progress": True})
    worker._serve_connection(conn, Lock(), stop)
    assert len(conn.responses) == 1
    assert conn.responses[0]["error_code"] == "worker_shutting_down"
    assert conn.closed


def test_quit_while_processing_abandons_the_current_job_before_publication(monkeypatch):
    """凍結Design 40a: 処理中に終了したら、その1件は中断され保存されない。"""
    abandon = Event()
    monkeypatch.setattr(worker, "_ABANDON", abandon)
    monkeypatch.setattr(worker, "_wake_listener", lambda *args: True)
    worker._abandon_checkpoint()  # 終了要求の前は止めない

    plain = Connection({"cmd": "shutdown"})
    worker._serve_connection(plain, Lock(), Event())
    assert not abandon.is_set(), "通常の終了要求で処理中の1件を捨ててはいけない"

    quitting = Connection({"cmd": "shutdown", "abandon": True})
    worker._serve_connection(quitting, Lock(), Event())
    assert quitting.responses[0]["status"] == "ok"
    assert abandon.is_set()
    try:
        worker._abandon_checkpoint()
    except worker.JobAbandonedError as exc:
        assert worker._error_code(exc) == "job_abandoned"
    else:
        raise AssertionError("終了後の公開前の確認で止まらなかった")


def test_quit_while_processing_frees_the_pipe_name_for_a_relaunched_lss(monkeypatch):
    """実機の反証: 処理中に終了してすぐ起動し直すと、前のworkerの推論中の接続が同じ名前のパイプを掴んだままで、
    新しいworkerが待ち受けを作れず「workerが25.0秒以内に起動しなかった」で行が失敗した
    （FROZEN-9fac7db evidence-builder bgm-quit-busy-relaunch-now）。終了要求で接続を閉じて名前を空ける。"""
    from multiprocessing import connection
    from uuid import uuid4

    from lss.core import client

    address = fr"\\.\pipe\lss-relaunch-{uuid4().hex}"
    started, release, finished = Event(), Event(), Event()

    def long_inference(job, progress=None, **_kwargs):
        started.set()
        assert release.wait(30)
        progress({"phase": "separating"})  # 閉じた接続への進捗は黙って捨てる
        finished.set()
        return {"status": "ok"}

    monkeypatch.setattr(worker, "_handle", long_inference)
    monkeypatch.setattr(worker, "_ABANDON", Event())
    for name in ("cleanup_stale_work", "reconcile_publications"):
        monkeypatch.setattr(worker.session, name, lambda: None)
    for name in ("_reconcile_bgm_current_best", "_keep_state_backup", "_ensure_reference_library",
                 "_retire_idle_reference_media"):
        monkeypatch.setattr(worker, name, lambda: None)
    server = Thread(target=worker.serve, kwargs={"address": address}, daemon=True)
    server.start()
    deadline = time.monotonic() + 20
    while not client.is_worker_running(address):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    job = connection.Client(address, authkey=worker.AUTHKEY)
    job.send({"mix_path": "x.wav", "_stream_progress": True})
    assert started.wait(10)
    try:
        assert client.request_worker_shutdown(address, abandon=True)["busy"] is True
        # 前のworkerは推論を続けているが、名前は空いている
        deadline = time.monotonic() + 5
        while True:
            try:
                relaunched = connection.Listener(address, family="AF_PIPE", authkey=worker.AUTHKEY)
                break
            except PermissionError:
                assert time.monotonic() < deadline, "前のworkerがパイプの名前を掴んだまま"
                time.sleep(0.05)
        relaunched.close()
        assert not finished.is_set()
    finally:
        release.set()
        job.close()
    assert finished.wait(10), "推論は最後まで続ける"
    server.join(10)
    assert not server.is_alive()


_BUSY_WORKER = '''
import sys, time
sys.path.insert(0, {tree!r})
from lss.core import worker

def long_job(job, progress=None, **_kwargs):
    end = time.monotonic() + 25
    while time.monotonic() < end:
        if progress:
            progress({{"phase": "separating"}})
        sum(i * i for i in range(20000))  # Python-level work that keeps the GIL busy, like a real job
    return {{"status": "ok"}}

worker._handle = long_job
for name in ("_reconcile_bgm_current_best", "_keep_state_backup", "_ensure_reference_library",
             "_retire_idle_reference_media"):
    setattr(worker, name, lambda: None)
worker.session.cleanup_stale_work = lambda: None
worker.session.reconcile_publications = lambda: None
worker.main(["--address", {address!r}])
'''


def test_a_busy_worker_process_frees_the_pipe_name_within_seconds_of_an_abandon_shutdown(tmp_path):
    """実機の反証: 処理中に終了した後も前のworkerがパイプの名前を約100秒掴み、すぐ起動し直した LSS の新しいworkerは
    待ち受けを作れず終了コード1で落ちた（FROZEN-7fc5b01、probe_pipe_after_quit で first_free 97.6 s）。終了要求の
    接続自身とジョブの接続が名前を残し、待ち受けを起こす自分への接続が「使用中」で1回20秒ずつ待たされていた。"""
    import os
    import subprocess
    import sys
    from multiprocessing import connection
    from pathlib import Path
    from uuid import uuid4

    from lss.core import client

    address = fr"\\.\pipe\lss-busy-{uuid4().hex}"
    script = tmp_path / "busy_worker.py"
    script.write_text(_BUSY_WORKER.format(tree=str(Path(__file__).resolve().parents[1]), address=address),
                      encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(script)], env={**os.environ, "LSS_STATE_DIR": str(tmp_path)})
    try:
        deadline = time.monotonic() + 60
        while not client.is_worker_running(address):
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.1)
        job = connection.Client(address, authkey=worker.AUTHKEY)
        job.send({"mix_path": "x.wav", "_stream_progress": True})
        assert job.poll(10)
        job.close()  # 終了した LSS の画面
        assert client.request_worker_shutdown(address, abandon=True)["busy"] is True
        asked = time.monotonic()
        while True:
            try:
                connection.Listener(address, family="AF_PIPE", authkey=b"relaunch").close()
                break
            except PermissionError:
                assert time.monotonic() - asked < 5, "前のworkerがパイプの名前を掴んだまま"
                time.sleep(0.1)
        assert process.poll() is None, "推論は前のworkerで続く"
    finally:
        process.kill()
        process.wait(30)


def test_listening_waits_briefly_for_the_previous_worker_to_free_the_name(monkeypatch):
    attempts = []

    def listener(address, family, authkey):
        attempts.append(address)
        if len(attempts) < 3:
            raise PermissionError(5, "アクセスが拒否されました")
        return "listener"

    monkeypatch.setattr(worker.connection, "Listener", listener)
    monkeypatch.setattr(worker.time, "sleep", lambda _s: None)
    assert worker._listen("pipe", b"k", patience=5) == "listener"
    assert len(attempts) == 3


def test_a_worker_that_cannot_listen_exits_with_a_code_instead_of_an_error_window(monkeypatch):
    def refused(*_a, **_k):
        raise PermissionError(5, "アクセスが拒否されました")

    monkeypatch.setattr(worker, "serve", refused)
    with pytest.raises(SystemExit) as exited:
        worker.main(["--address", "pipe"])
    assert exited.value.code == 1


def test_abandoned_job_publishes_nothing_and_leaves_no_staged_file(tmp_path, monkeypatch):
    """公開前の確認で止めた1件は、LSSフォルダにも公開の記録にも何も残さない。"""
    from tests.test_writer import _call_write, _setup_recipe
    from lss.output import writer

    abandon = Event()
    abandon.set()
    monkeypatch.setattr(worker, "_ABANDON", abandon)
    mix_path, store_dir = _setup_recipe(tmp_path)
    state_root = tmp_path / "state"
    try:
        _call_write(mix_path, store_dir, validate_before_publish=worker._abandon_checkpoint,
                    publication_state_root=state_root)
    except worker.JobAbandonedError:
        pass
    else:
        raise AssertionError("中断されなかった")
    assert list(writer.output_dir_for(mix_path).iterdir()) == []
    assert not (state_root / "publications").exists() or not any((state_root / "publications").iterdir())
