from uuid import uuid4
from types import SimpleNamespace
import inspect
import os
import subprocess
import sys
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor

from lss.core import client


def test_is_worker_running_false_when_no_server():
    address = fr"\\.\pipe\lss-worker-test-nonexistent-{uuid4().hex[:8]}"
    assert client.is_worker_running(address=address) is False


def test_is_worker_running_false_on_authentication_failure(monkeypatch):
    monkeypatch.setattr(
        client.connection, "Client",
        lambda *a, **k: (_ for _ in ()).throw(client.AuthenticationError("wrong worker")),
    )

    assert client.is_worker_running("occupied") is False


@pytest.mark.parametrize("response", [
    {"status": "ok", "protocol_version": client.protocol.VERSION - 1},
    {"status": "error"},
    None,
])
def test_worker_probe_requires_matching_protocol_response(monkeypatch, response):
    sent = []
    conn = SimpleNamespace(
        send=sent.append,
        poll=lambda timeout: response is not None,
        recv=lambda: response,
        close=lambda: None,
    )
    monkeypatch.setattr(client.connection, "Client", lambda *a, **k: conn)

    assert client.is_worker_running("custom") is False
    assert sent == [{"cmd": "ping", "protocol_version": client.protocol.VERSION}]


def test_worker_probe_accepts_matching_protocol_response(monkeypatch):
    response = {"status": "ok", "protocol_version": client.protocol.VERSION}
    conn = SimpleNamespace(
        send=lambda job: None, poll=lambda timeout: True,
        recv=lambda: response, close=lambda: None,
    )
    monkeypatch.setattr(client.connection, "Client", lambda *a, **k: conn)

    assert client.is_worker_running("custom") is True


def test_ensure_worker_running_raises_timeout_when_never_comes_up(monkeypatch):
    monkeypatch.setattr(client, "is_worker_running", lambda address: False)
    stopped = []
    proc = SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append(True), wait=lambda timeout: 0)
    monkeypatch.setattr(client, "spawn_worker", lambda address: proc)

    try:
        client.ensure_worker_running(startup_timeout=0.3)
        assert False, "TimeoutErrorが送出されるはず"
    except TimeoutError:
        pass
    assert stopped == [True]


def test_startup_timeout_kills_worker_that_ignores_terminate(monkeypatch):
    monkeypatch.setattr(client, "is_worker_running", lambda address: False)
    actions = []

    class StubbornProcess:
        def poll(self):
            return None

        def terminate(self):
            actions.append("terminate")

        def kill(self):
            actions.append("kill")

        def wait(self, timeout):
            actions.append(("wait", timeout))
            if actions.count(("wait", timeout)) == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 1

    monkeypatch.setattr(client, "spawn_worker", lambda address: StubbornProcess())

    with pytest.raises(TimeoutError, match="起動しなかった"):
        client.ensure_worker_running(startup_timeout=0)

    assert actions == ["terminate", ("wait", 5), "kill", ("wait", 5)]


def test_cold_worker_start_beyond_the_old_25_seconds_is_waited_for(monkeypatch):
    """入れ替え直後の最初の起動は冷えたディスクから読む（2026-10-15、凍結3b3a456で25秒を2回超えて行が失敗）。

    生きている起動途中のworkerは止めずに待ち、待ち受けた時点で使う。"""
    clock = [0.0]
    monkeypatch.setattr(client, "time", SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    monkeypatch.setattr(client, "is_worker_running", lambda address: clock[0] >= 40.0)
    stopped = []
    proc = SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append(True), wait=lambda timeout: 0)
    monkeypatch.setattr(client, "spawn_worker", lambda address: proc)
    monkeypatch.setattr(client, "_closing", threading.Event())

    client.ensure_worker_running(startup_timeout=client.WORKER_STARTUP_SECONDS, address="custom")

    assert stopped == []
    assert 40.0 <= clock[0] < 41.0


def test_dispatch_and_quit_wait_as_long_as_a_cold_worker_start():
    assert client.WORKER_STARTUP_SECONDS >= 120.0
    assert inspect.signature(client.dispatch_job).parameters["connect_timeout"].default == client.WORKER_STARTUP_SECONDS
    # 起動途中で終了したとき、終了要求を届け直す上限は起動の上限と同じ。
    assert inspect.signature(client.finish_worker_shutdown).parameters["timeout"].default == \
        client.WORKER_STARTUP_SECONDS


def test_quit_while_a_worker_starts_stops_waiting_and_leaves_it_to_the_shutdown(monkeypatch):
    closing = threading.Event()
    monkeypatch.setattr(client, "_closing", closing)
    monkeypatch.setattr(client, "is_worker_running", lambda address: False)
    stopped = []

    def poll():
        closing.set()  # 起動を待つ間に利用者が終了した
        return None

    proc = SimpleNamespace(poll=poll, terminate=lambda: stopped.append(True), wait=lambda timeout: 0)
    monkeypatch.setattr(client, "spawn_worker", lambda address: proc)

    with pytest.raises(client.ClosingError):
        client.ensure_worker_running(startup_timeout=client.WORKER_STARTUP_SECONDS, address="custom")
    # 止めない。終了処理（finish_worker_shutdown）が待ち受け次第、終了要求を届ける。
    assert stopped == []


def test_client_import_does_not_load_inference_or_ui():
    result = subprocess.run([sys.executable, "-c", "import sys; import lss.core.client; assert 'torch' not in sys.modules; assert 'PySide6' not in sys.modules"],
                            cwd=client.REPO_ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("frozen", [False, True])
def test_spawn_passes_address_and_hides_console(monkeypatch, frozen):
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    calls = []
    monkeypatch.setattr(client.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs)))
    client.spawn_worker("custom-pipe")
    args, options = calls[0]
    assert args[-2:] == ["--address", "custom-pipe"]
    assert ("--worker" in args) == frozen
    assert options["creationflags"] == subprocess.CREATE_NO_WINDOW


def test_spawn_strict_profile_configures_only_the_child_environment(monkeypatch):
    monkeypatch.setenv("LSS_DEV_SFC_EXECUTION_PROFILE", "strict_deterministic_paced60")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = []
    monkeypatch.setattr(client.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs)))

    client.spawn_worker("custom-pipe")

    _, options = calls[0]
    assert options["env"]["LSS_DEV_SFC_EXECUTION_PROFILE"] == "strict_deterministic_paced60"
    assert options["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_spawn_default_profile_preserves_inherited_environment(monkeypatch):
    monkeypatch.delenv("LSS_DEV_SFC_EXECUTION_PROFILE", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = []
    monkeypatch.setattr(client.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs)))

    client.spawn_worker("custom-pipe")

    assert "env" not in calls[0][1]


def test_spawn_rejects_invalid_profile_or_conflicting_workspace_before_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(client.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setenv("LSS_DEV_SFC_EXECUTION_PROFILE", "unknown")
    with pytest.raises(ValueError, match="Unsupported internal SFC execution profile"):
        client.spawn_worker("custom-pipe")
    assert calls == []

    monkeypatch.setenv("LSS_DEV_SFC_EXECUTION_PROFILE", "strict_deterministic")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="requires CUBLAS_WORKSPACE_CONFIG"):
        client.spawn_worker("custom-pipe")
    assert calls == []


def test_startup_reports_process_exit_without_waiting_for_timeout(monkeypatch):
    monkeypatch.setattr(client, "is_worker_running", lambda address: False)
    monkeypatch.setattr(client, "spawn_worker", lambda address: SimpleNamespace(poll=lambda: 7, returncode=7))
    with pytest.raises(RuntimeError, match="終了コード: 7"):
        client.ensure_worker_running()


def test_dispatch_uses_same_custom_address_for_startup_and_connection(monkeypatch):
    addresses = []
    monkeypatch.setattr(client, "ensure_worker_running", lambda **kwargs: addresses.append(kwargs["address"]))
    conn = SimpleNamespace(send=lambda job: None, recv=lambda: {"status": "ok"}, close=lambda: None)
    monkeypatch.setattr(client.connection, "Client", lambda address, **kwargs: addresses.append(address) or conn)
    assert client.dispatch_job({}, address="custom")["status"] == "ok"
    assert addresses == ["custom", "custom"]


def test_dispatch_streams_progress_before_returning_result(monkeypatch):
    monkeypatch.setattr(client, "ensure_worker_running", lambda **kwargs: None)
    responses = iter([
        {"type": "progress", "phase": "separating", "completed_bytes": 2},
        {"status": "ok", "outputs": ["result.flac"]},
    ])
    sent = []
    conn = SimpleNamespace(
        send=lambda job: sent.append(job), recv=lambda: next(responses), close=lambda: None,
    )
    monkeypatch.setattr(client.connection, "Client", lambda *a, **k: conn)
    events = []

    result = client.dispatch_job({"mix_path": "input.wav"}, on_progress=events.append)

    assert sent == [{"mix_path": "input.wav", "_stream_progress": True}]
    assert events == [{"phase": "separating", "completed_bytes": 2}]
    assert result == {"status": "ok", "outputs": ["result.flac"]}


def test_progress_callback_failure_does_not_lose_final_response(monkeypatch):
    monkeypatch.setattr(client, "ensure_worker_running", lambda **kwargs: None)
    responses = iter([
        {"type": "progress", "phase": "separating"},
        {"status": "ok"},
    ])
    conn = SimpleNamespace(send=lambda job: None, recv=lambda: next(responses), close=lambda: None)
    monkeypatch.setattr(client.connection, "Client", lambda *a, **k: conn)

    result = client.dispatch_job(
        {}, on_progress=lambda event: (_ for _ in ()).throw(RuntimeError("表示失敗")),
    )

    assert result == {"status": "ok"}


def test_transport_disconnect_reconnects_and_resends_once(monkeypatch):
    starts = []
    monkeypatch.setattr(client, "ensure_worker_running", lambda **kwargs: starts.append(kwargs["address"]))
    sent = []
    first = SimpleNamespace(
        send=lambda job: sent.append((1, job)),
        recv=lambda: (_ for _ in ()).throw(EOFError("worker stopped")),
        close=lambda: None,
    )
    second = SimpleNamespace(
        send=lambda job: sent.append((2, job)),
        recv=lambda: {"status": "ok"},
        close=lambda: None,
    )
    connections = iter([first, second])
    monkeypatch.setattr(client.connection, "Client", lambda *a, **k: next(connections))

    result = client.dispatch_job({"mix_path": "input.wav"}, address="custom")

    assert result == {"status": "ok"}
    assert starts == ["custom", "custom"]
    assert sent == [
        (1, {"mix_path": "input.wav"}),
        (2, {"mix_path": "input.wav"}),
    ]


def test_transport_disconnect_is_reported_after_retry_limit(monkeypatch):
    monkeypatch.setattr(client, "ensure_worker_running", lambda **kwargs: None)
    conn = SimpleNamespace(
        send=lambda job: None,
        recv=lambda: (_ for _ in ()).throw(EOFError("worker stopped")),
        close=lambda: None,
    )
    monkeypatch.setattr(client.connection, "Client", lambda *a, **k: conn)

    with pytest.raises(EOFError, match="worker stopped"):
        client.dispatch_job({}, transport_retries=0)


def test_negative_transport_retry_count_is_rejected():
    with pytest.raises(ValueError, match="0以上"):
        client.dispatch_job({}, transport_retries=-1)


def test_concurrent_startup_requests_spawn_only_one_worker(monkeypatch):
    starts = []
    monkeypatch.setattr(client, "is_worker_running", lambda address: bool(starts))
    monkeypatch.setattr(client, "spawn_worker", lambda address: starts.append(address))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.ensure_worker_running, address="custom") for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert starts == ["custom"]
