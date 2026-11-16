"""凍結Design 40a: 処理中に終了したら、その1件は中断され保存されない。終了後に常駐も作りかけも残さない。

画面のprocess側の取り決め（終了の要求・利用者のフォルダへの書き込みの待ち・workerの起動の止め方）を見る。
BGM候補の工程の区切りは `test_application_bgm_candidates`、workerの区切りは `test_worker_shutdown_boundary`。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.core import client, quitting
from lss.ui.web.session import WebSession


@pytest.fixture(autouse=True)
def fresh_process_flags(monkeypatch):
    # 終了の要求・workerの起動の止めはprocess全体の状態。試験ごとに新しくする。
    monkeypatch.setattr(quitting, "_REQUESTED", threading.Event())
    monkeypatch.setattr(quitting, "_sealed", False)
    monkeypatch.setattr(quitting, "_writes", 0)
    monkeypatch.setattr(client, "_closing", threading.Event())
    monkeypatch.setattr(client, "_spawned", None)


@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def _audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros((4410, 2), dtype="float32"), 44100, subtype="PCM_16")
    return path


def _wait(predicate, timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("条件が成立しなかった")


def test_quit_while_processing_abandons_the_source_and_stops_spawning_workers(state, monkeypatch):
    from lss.ui.web.host import Host

    entered, release = threading.Event(), threading.Event()

    def worker(job):
        entered.set()
        assert release.wait(10)
        return {"status": "error", "error_code": "job_abandoned", "message": "中断"}

    requests = []
    monkeypatch.setattr(client, "request_worker_shutdown",
                        lambda *a, **k: requests.append(k) or {"status": "ok", "busy": True})
    session = WebSession(dispatch=worker, state_path=state / "queue.json")
    [item] = session.add([_audio(state / "one.wav")])
    host = Host(session=session)
    session.start()
    assert entered.wait(10)

    host.quit()

    assert requests == [{"abandon": True}]
    assert quitting.requested() and client._closing.is_set()
    release.set()
    _wait(lambda: not session.running())
    saved = json.loads((state / "queue.json").read_text(encoding="utf-8"))
    assert [row["status"] for row in saved["items"]] == ["running"], "中断した1件を失敗として残さない"
    restored = WebSession(dispatch=worker, state_path=state / "queue.json")
    assert restored.snapshot()["st"][item] == "queued"


def test_quit_when_idle_does_not_abandon(state, monkeypatch):
    from lss.ui.web.host import Host

    requests = []
    monkeypatch.setattr(client, "request_worker_shutdown", lambda *a, **k: requests.append(k) or None)
    host = Host(session=WebSession(dispatch=lambda job: {"status": "ok", "outputs": []},
                                   state_path=state / "queue.json"))
    host.quit()
    assert requests == [{"abandon": False}]
    assert not quitting.requested()


class _Session:
    def __init__(self) -> None:
        self.busy = True

    def running(self) -> bool:
        return self.busy


def _quitting_host(monkeypatch, session) -> object:
    from lss.ui.web import host as host_module

    monkeypatch.setattr(host_module, "QUIT_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(host_module, "QUIT_WRITE_LIMIT_SECONDS", 5.0)
    host = host_module.Host.__new__(host_module.Host)
    host.session = session
    host._quitting, host._abandon, host._worker_reached = True, True, True
    return host


def test_exit_waits_while_lss_is_writing_into_the_users_folder(monkeypatch):
    session = _Session()
    host = _quitting_host(monkeypatch, session)
    inside = threading.Event()

    def publishing():
        with quitting.writing_user_folder():
            inside.set()
            time.sleep(0.6)
        session.busy = False

    thread = threading.Thread(target=publishing)
    thread.start()
    assert inside.wait(5)
    started = time.monotonic()
    host._finish_quit()
    assert time.monotonic() - started >= 0.4
    assert not session.running()
    thread.join(5)


def test_exit_does_not_wait_for_inference_or_internal_work(monkeypatch):
    session = _Session()  # 推論の結果待ちのまま終わらない
    host = _quitting_host(monkeypatch, session)
    started = time.monotonic()
    host._finish_quit()
    assert time.monotonic() - started < 2.0
    assert session.running()


def test_the_seal_refuses_a_write_that_had_not_begun_but_waits_for_one_in_progress():
    with quitting.writing_user_folder():
        assert not quitting.seal_if_idle(), "書いている途中で封をしない"
    quitting.hold_publication()
    quitting.hold_publication()  # 同じスレッドの2回目は1回分
    assert quitting.writing() and not quitting.seal_if_idle()
    quitting.release_publication()
    assert not quitting.writing()
    assert quitting.seal_if_idle()
    with pytest.raises(quitting.JobAbandonedError):
        with quitting.writing_user_folder():
            pytest.fail("封の後に利用者のフォルダへ書き始めた")
    with pytest.raises(quitting.JobAbandonedError):
        quitting.hold_publication()
    assert not quitting.writing()


def test_exit_seals_once_nothing_is_being_written(monkeypatch):
    session = _Session()
    host = _quitting_host(monkeypatch, session)
    host._finish_quit()
    with pytest.raises(quitting.JobAbandonedError):
        with quitting.writing_user_folder():
            pass


def test_a_job_refused_by_the_previous_lss_worker_continues_on_a_new_worker(monkeypatch):
    """終了直後に起動し直すと、前の LSS が終了させた常駐workerに当たることがある。行を失敗にしない。"""
    replies = iter([{"status": "error", "error_code": "worker_shutting_down", "message": "終了中"},
                    {"status": "ok", "outputs": ["x.flac"]}])
    monkeypatch.setattr(client, "_dispatch_once", lambda *a, **k: next(replies))
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)
    assert client.dispatch_job({"mix_path": "x.wav"}) == {"status": "ok", "outputs": ["x.flac"]}


def test_our_own_quit_still_surfaces_the_refusal(monkeypatch):
    refusal = {"status": "error", "error_code": "worker_shutting_down", "message": "終了中"}
    monkeypatch.setattr(client, "_dispatch_once", lambda *a, **k: refusal)
    client.close()
    assert client.dispatch_job({"mix_path": "x.wav"}) == refusal


def test_no_worker_is_started_or_sent_a_job_after_quitting(monkeypatch):
    monkeypatch.setattr(client, "is_worker_running", lambda address=None: False)
    monkeypatch.setattr(client, "spawn_worker", lambda address=None: pytest.fail("終了後にworkerを起動した"))
    client.close()
    with pytest.raises(client.ClosingError):
        client.ensure_worker_running()
    with pytest.raises(OSError):
        client.dispatch_job({"mix_path": "x.wav"})


def test_a_worker_still_starting_at_quit_receives_the_shutdown(monkeypatch):
    """起動の途中で終了すると最初の終了要求は届かない。届くまで送り直し、画面の無い常駐を残さない。"""
    class Starting:
        def poll(self):
            return None

    monkeypatch.setattr(client, "_spawned", Starting())
    replies = iter([None, None, {"status": "ok", "busy": False}])
    sent = []
    monkeypatch.setattr(client, "request_worker_shutdown", lambda **k: sent.append(k) or next(replies))
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)

    assert client.finish_worker_shutdown(abandon=True, timeout=5.0) is True
    assert sent == [{"abandon": True}] * 3


def test_no_shutdown_is_resent_when_this_lss_started_no_worker(monkeypatch):
    monkeypatch.setattr(client, "request_worker_shutdown", lambda **k: pytest.fail("送り直さない"))
    assert client.finish_worker_shutdown(abandon=True) is True
