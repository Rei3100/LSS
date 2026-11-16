import json
from pathlib import Path

import pytest

from lss.core import jobs
from lss.core.jobs import SeparationQueue
from lss.ui import contract
from lss.ui import state as ui_state


def test_queue_continues_after_failure_and_retries_only_failed_item(tmp_path):
    calls = []

    def dispatch(job):
        calls.append(job["mix_path"])
        if len(calls) == 1:
            raise OSError("通信切断")
        return {"status": "ok", "outputs": ["result.flac"]}

    queue = SeparationQueue(dispatch, state_path=tmp_path / "queue.json")
    ids = queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Vocals")
    queue.run()
    assert [item.status for item in queue.snapshot()] == ["failed", "completed"]
    assert queue.snapshot()[0].result["error_code"] == "worker_unavailable"
    queue.retry(ids[0])
    queue.run()
    assert len(calls) == 3
    assert calls[0] == calls[2]
    assert all(item.status == "completed" for item in queue.snapshot())
    with pytest.raises(ValueError):
        queue.retry(ids[1])


def test_queue_accepts_new_input_during_processing_and_pauses_between_files(tmp_path):
    calls = []

    def dispatch(job):
        calls.append(job)
        if len(calls) == 1:
            assert queue.snapshot()[0].status == "running"
            queue.add([tmp_path / "b.wav"], "model", "Vocals")
            queue.pause()
            with pytest.raises(RuntimeError, match="処理中"):
                queue.run()
        return {"status": "skipped", "outputs": ["existing.flac"]}

    queue = SeparationQueue(dispatch, state_path=tmp_path / "queue.json")
    queue.add([tmp_path / "a.wav"], "model", "Vocals")
    assert queue.add([tmp_path / "a.wav"], "model", "Vocals") == []
    queue.run()
    assert [item.status for item in queue.snapshot()] == ["skipped", "queued"]
    queue.run()
    assert len(calls) == 2
    assert all(item.status == "skipped" for item in queue.snapshot())
    copied = queue.snapshot()
    copied[0].result["outputs"].clear()
    assert queue.snapshot()[0].result["outputs"] == ["existing.flac"]


@pytest.mark.parametrize("response", [
    None,
    {"status": "unknown"},
    {"status": "error", "message": "分離失敗"},
    {"status": "error", "error_code": "processing_failed"},
    {"status": "ok"},
    {"status": "ok", "outputs": []},
    {"status": "skipped", "outputs": [""]},
    {"status": "ok", "outputs": ["result.flac"], "invalid": object()},
])
def test_invalid_or_failed_response_is_recorded_as_failure(tmp_path, response):
    queue = SeparationQueue(lambda job: response, state_path=tmp_path / "queue.json")
    queue.add([tmp_path / "a.wav"], "model", "Vocals")
    queue.run()
    assert queue.snapshot()[0].status == "failed"
    assert queue.snapshot()[0].result["error_code"] == "invalid_worker_response"


def test_validated_zero_result_is_a_successful_queue_outcome(tmp_path):
    zero_result = {
        "reason": "music_not_wanted",
        "validation_result": {
            "validator_id": "bgm-semantic-gate", "validator_revision": "r1",
            "hard_gates_passed": True, "evidence": {"music_frames": 100},
        },
    }
    queue = SeparationQueue(
        lambda job: {"status": "ok", "outputs": [], "zero_result": zero_result},
        state_path=tmp_path / "queue.json",
    )
    queue.add([tmp_path / "source.wav"], "recipe", "bgm")

    queue.run()

    item = queue.snapshot()[0]
    assert item.status == "completed"
    assert item.result["zero_result"] == zero_result


def test_zero_result_without_hard_gate_evidence_is_rejected(tmp_path):
    queue = SeparationQueue(
        lambda job: {
            "status": "ok", "outputs": [],
            "zero_result": {
                "reason": "no_wanted_score",
                "validation_result": {
                    "validator_id": "gate", "validator_revision": "r1",
                    "hard_gates_passed": False,
                },
            },
        },
        state_path=tmp_path / "queue.json",
    )
    queue.add([tmp_path / "source.wav"], "recipe", "bgm")

    queue.run()

    assert queue.snapshot()[0].status == "failed"
    assert queue.snapshot()[0].result["error_code"] == "invalid_worker_response"


def test_interrupted_runner_can_resume_pending_job(tmp_path):
    calls = []

    def dispatch(job):
        calls.append(job)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return {"status": "ok", "outputs": ["result.flac"]}

    queue = SeparationQueue(dispatch, state_path=tmp_path / "queue.json")
    queue.add([tmp_path / "a.wav"], "model", "Vocals")
    with pytest.raises(KeyboardInterrupt):
        queue.run()
    assert queue.snapshot()[0].status == "queued"
    queue.run()
    assert queue.snapshot()[0].status == "completed"


def test_queue_restores_items_and_requeues_interrupted_work(tmp_path):
    state_path = tmp_path / "queue.json"
    first = SeparationQueue(
        lambda job: {"status": "ok", "outputs": ["result.flac"]}, state_path=state_path,
    )
    ids = first.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Vocals")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["items"][0]["status"] = "running"  # プロセスが処理中に終了した状態を再現する。
    payload["items"][0]["result"] = {"status": "partial"}
    state_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    calls = []
    restored = SeparationQueue(
        lambda job: calls.append(job["mix_path"]) or {"status": "ok", "outputs": ["result.flac"]},
        state_path=state_path,
    )

    assert [item.id for item in restored.snapshot()] == ids
    assert [item.status for item in restored.snapshot()] == ["queued", "queued"]
    assert restored.snapshot()[0].result is None
    restored.run()
    assert calls == [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]
    assert all(item.status == "completed" for item in restored.snapshot())

    loaded_again = SeparationQueue(state_path=state_path)
    assert [item.status for item in loaded_again.snapshot()] == ["completed", "completed"]


def test_corrupt_queue_state_is_reported_without_overwrite(tmp_path):
    state_path = tmp_path / "queue.json"
    state_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="キュー状態を読み込めません"):
        SeparationQueue(state_path=state_path)

    assert state_path.read_text(encoding="utf-8") == "{broken"


@pytest.mark.parametrize("mutation", ["duplicate_id", "invalid_result", "invalid_progress"])
def test_structurally_invalid_queue_state_is_rejected_without_overwrite(tmp_path, mutation):
    state_path = tmp_path / "queue.json"
    item = {
        "id": "item-1", "path": "input.wav", "model_id": "model",
        "stem_target": "Vocals", "status": "queued", "result": None, "progress": None,
    }
    items = [dict(item)]
    if mutation == "duplicate_id":
        items.append(dict(item))
    elif mutation == "invalid_result":
        items[0]["result"] = []
    else:
        items[0]["progress"] = "separating"
    original = json.dumps({"version": 1, "items": items}, ensure_ascii=False)
    state_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="キュー状態を読み込めません"):
        SeparationQueue(state_path=state_path)

    assert state_path.read_text(encoding="utf-8") == original


def test_non_transport_dispatch_exception_has_stable_error_code(tmp_path):
    def dispatch(job):
        raise ValueError("呼び出し失敗")

    queue = SeparationQueue(dispatch, state_path=tmp_path / "queue.json")
    queue.add([tmp_path / "a.wav"], "model", "Vocals")
    queue.run()

    assert queue.snapshot()[0].result["error_code"] == "dispatch_failed"


def test_completed_or_skipped_input_can_be_added_again(tmp_path):
    responses = iter([
        {"status": "ok", "outputs": ["first.flac"]},
        {"status": "skipped", "outputs": ["first.flac"]},
    ])
    queue = SeparationQueue(lambda job: next(responses), state_path=tmp_path / "queue.json")
    source = tmp_path / "edited-later.wav"

    first_id = queue.add([source], "model", "Vocals")[0]
    queue.run()
    second_id = queue.add([source], "model", "Vocals")[0]

    assert second_id != first_id
    assert [item.status for item in queue.snapshot()] == ["completed", "queued"]
    queue.run()
    assert [item.status for item in queue.snapshot()] == ["completed", "skipped"]


def test_failed_input_can_be_readded_without_losing_failure_history(tmp_path):
    queue = SeparationQueue(
        lambda job: {
            "status": "error", "error_code": "processing_failed", "message": "failed",
        },
        state_path=tmp_path / "queue.json",
    )
    source = tmp_path / "retry.wav"
    first_id = queue.add([source], "model", "Vocals")[0]
    queue.run()

    second_id = queue.add([source], "model", "Vocals")[0]

    assert second_id != first_id
    assert [item.status for item in queue.snapshot()] == ["failed", "queued"]


def test_default_dispatch_exposes_progress_in_snapshot_and_callback(tmp_path, monkeypatch):
    observed = []

    def dispatch(job, on_progress=None):
        on_progress({"phase": "checking_model", "model_id": job["model_id"]})
        on_progress({"phase": "separating"})
        assert queue.snapshot()[0].progress == {"phase": "separating"}
        return {"status": "ok", "outputs": ["result.flac"]}

    monkeypatch.setattr(jobs, "dispatch_job", dispatch)
    queue = SeparationQueue(
        state_path=tmp_path / "queue.json",
        on_progress=lambda item_id, event: observed.append((item_id, event)),
    )
    item_id = queue.add([tmp_path / "input.wav"], "model", "Vocals")[0]

    queue.run()

    item = queue.snapshot()[0]
    assert item.status == "completed"
    assert item.progress == {"phase": "completed"}
    assert observed == [
        (item_id, {"phase": "checking_model", "model_id": "model"}),
        (item_id, {"phase": "separating"}),
    ]


def test_progress_callback_failure_does_not_stop_queue(tmp_path, monkeypatch):
    def dispatch(job, on_progress=None):
        on_progress({"phase": "separating"})
        return {"status": "ok", "outputs": ["result.flac"]}

    monkeypatch.setattr(jobs, "dispatch_job", dispatch)
    queue = SeparationQueue(
        state_path=tmp_path / "queue.json",
        on_progress=lambda *args: (_ for _ in ()).throw(RuntimeError("表示失敗")),
    )
    queue.add([tmp_path / "input.wav"], "model", "Vocals")

    queue.run()

    assert queue.snapshot()[0].status == "completed"


def test_nonserializable_progress_is_ignored_without_breaking_state_save(tmp_path, monkeypatch):
    observed = []

    def dispatch(job, on_progress=None):
        on_progress({"phase": "separating", "invalid": object()})
        on_progress({"phase": "separating", "completed": 1})
        return {"status": "ok", "outputs": ["result.flac"]}

    monkeypatch.setattr(jobs, "dispatch_job", dispatch)
    state_path = tmp_path / "queue.json"
    queue = SeparationQueue(
        state_path=state_path,
        on_progress=lambda item_id, event: observed.append(event),
    )
    queue.add([tmp_path / "input.wav"], "model", "Vocals")

    queue.run()

    assert queue.snapshot()[0].status == "completed"
    assert observed == [{"phase": "separating", "completed": 1}]
    assert json.loads(state_path.read_text(encoding="utf-8"))["items"][0]["status"] == "completed"


def _queue(tmp_path, dispatch, **kwargs):
    return SeparationQueue(dispatch, state_path=tmp_path / "queue.json", **kwargs)


ZERO_RESULT = {
    "reason": "no_wanted_score",
    "validation_result": {
        "validator_id": "v", "validator_revision": "1", "hard_gates_passed": True,
    },
}


def test_every_terminal_outcome_arrives_through_on_result(tmp_path):
    """終端phaseが流れない終わり方でも、結果は必ず同期で届く。"""
    responses = [
        {"status": "ok", "outputs": ["a.flac"]},
        {"status": "skipped", "outputs": ["b.flac"]},
        {"status": "skipped", "outputs": [], "zero_result": ZERO_RESULT},
        {"status": "error", "error_code": "processing_failed", "message": "x"},
        None,  # workerへ届かず invalid_worker_response になる
    ]
    seen = []
    progress_events = []
    queue = _queue(
        tmp_path, lambda job: responses.pop(0),
        on_result=lambda item_id, result: seen.append((item_id, result["status"])),
        on_progress=lambda item_id, event: progress_events.append(event),
    )
    ids = queue.add([tmp_path / f"{n}.wav" for n in range(5)], "model", "Instrumental")
    queue.run()

    assert [item_id for item_id, _ in seen] == ids
    assert [status for _, status in seen] == ["ok", "skipped", "skipped", "error", "error"]
    # 進捗は1件も流れていないのに、5件すべての終端が届いている。
    assert progress_events == []
    assert [item.status for item in queue.snapshot()] == [
        "completed", "skipped", "skipped", "failed", "failed",
    ]


def test_session_blocking_result_stops_before_the_next_source_starts(tmp_path):
    """セッションを止める失敗は、次のSOURCEが開始する前に停止できる。"""
    started = []

    def dispatch(job):
        started.append(job["mix_path"])
        return {"status": "error", "error_code": "gpu_unavailable", "message": "x"}

    def on_result(item_id, result):
        if contract.classify_result(result).failure.scope is contract.FailureScope.SESSION:
            queue.pause()

    queue = _queue(tmp_path, dispatch, on_result=on_result)
    queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Instrumental")
    queue.run()

    assert len(started) == 1
    assert [item.status for item in queue.snapshot()] == ["failed", "queued"]


def test_retry_and_skip_are_deterministic_after_a_halt(tmp_path):
    """RETRYは同じSOURCEをやり直し、SKIPは失敗のまま次へ進む。"""
    attempts = []
    fail_first = {"a.wav"}

    def dispatch(job):
        name = Path(job["mix_path"]).name
        attempts.append(name)
        if name in fail_first:
            return {"status": "error", "error_code": "gpu_unavailable", "message": "x"}
        return {"status": "ok", "outputs": ["out.flac"]}

    halted = []

    def on_result(item_id, result):
        failure = contract.classify_result(result).failure
        if failure is not None and failure.scope is contract.FailureScope.SESSION:
            halted.append(item_id)
            queue.pause()

    queue = _queue(tmp_path, dispatch, on_result=on_result)
    ids = queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Instrumental")
    queue.run()
    assert attempts == ["a.wav"] and halted == [ids[0]]

    # RETRY: 同じSOURCEをやり直す。既存の結果は壊さない。
    fail_first.clear()
    queue.retry(ids[0])
    queue.run()
    assert attempts == ["a.wav", "a.wav", "b.wav"]
    assert [item.status for item in queue.snapshot()] == ["completed", "completed"]


def test_skip_leaves_the_failed_source_and_continues(tmp_path):
    def dispatch(job):
        if Path(job["mix_path"]).name == "a.wav":
            return {"status": "error", "error_code": "gpu_unavailable", "message": "x"}
        return {"status": "ok", "outputs": ["out.flac"]}

    queue = _queue(tmp_path, dispatch)
    queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Instrumental")
    queue.run()  # SKIP = 失敗のまま継続
    statuses = [item.status for item in queue.snapshot()]
    assert statuses == ["failed", "completed"]
    assert queue.snapshot()[0].result["error_code"] == "gpu_unavailable"


def test_safe_stop_does_not_start_another_source(tmp_path):
    started = []

    def dispatch(job):
        started.append(job["mix_path"])
        return {"status": "ok", "outputs": ["out.flac"]}

    queue = _queue(tmp_path, dispatch, on_result=lambda item_id, result: queue.pause())
    queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Instrumental")
    queue.run()

    assert len(started) == 1
    assert [item.status for item in queue.snapshot()] == ["completed", "queued"]
    queue.run()  # RESUME
    assert len(started) == 2


def test_halt_while_stop_is_pending_keeps_the_stop_intent(tmp_path):
    """停止要求中にHALTしても、回復後に停止意思が消えない。"""
    model = ui_state.UiModel(
        phase=ui_state.Phase.PROCESSING, sources=2, pending=2,
    )
    model = ui_state.request_stop(model)  # 安全停止を要求
    assert ui_state.display_state(model) is ui_state.UiState.STOPPING

    started = []

    def dispatch(job):
        started.append(Path(job["mix_path"]).name)
        if len(started) == 1:
            return {"status": "error", "error_code": "gpu_unavailable", "message": "x"}
        return {"status": "ok", "outputs": ["out.flac"]}

    def on_result(item_id, result):
        nonlocal model
        failure = contract.classify_result(result).failure
        if failure is not None and failure.scope is contract.FailureScope.SESSION:
            model = ui_state.fault_raised(model)
        queue.pause()

    queue = _queue(tmp_path, dispatch, on_result=on_result)
    ids = queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Instrumental")
    queue.run()

    assert ui_state.display_state(model) is ui_state.UiState.HALTED
    # HALTEDからは通常のSTOPを出せない。
    assert ui_state.can_request_stop(model) is False

    # RETRYで回復すると、消えていなかった停止要求がSTOPPINGとして戻る。
    model = ui_state.fault_resolved(model)
    assert ui_state.display_state(model) is ui_state.UiState.STOPPING
    assert model.stop_requested is True

    queue.retry(ids[0])
    queue.run()
    # 停止意思が生きているので、やり直した1件で止まり次のSOURCEは始まらない。
    assert started == ["a.wav", "a.wav"]
    assert [item.status for item in queue.snapshot()] == ["completed", "queued"]


def test_a_failing_result_listener_stops_instead_of_advancing(tmp_path):
    """継続可否を判断できないまま次のSOURCEを始めない。"""
    started = []

    def dispatch(job):
        started.append(job["mix_path"])
        return {"status": "ok", "outputs": ["out.flac"]}

    def on_result(item_id, result):
        raise RuntimeError("表示側の不具合")

    queue = _queue(tmp_path, dispatch, on_result=on_result)
    queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Instrumental")
    with pytest.raises(RuntimeError):
        queue.run()

    assert len(started) == 1
    # 結果は保存済みで、次のSOURCEは開始していない。
    assert [item.status for item in queue.snapshot()] == ["completed", "queued"]


def test_result_listener_can_read_the_queue_without_deadlock(tmp_path):
    """通知はロックの外で行う。表示側は通知の中でキューを読める。"""
    observed = []
    queue = _queue(
        tmp_path, lambda job: {"status": "ok", "outputs": ["out.flac"]},
        on_result=lambda item_id, result: observed.append(
            [item.status for item in queue.snapshot()]
        ),
    )
    queue.add([tmp_path / "a.wav", tmp_path / "b.wav"], "model", "Instrumental")
    queue.run()
    # 通知時点で当該SOURCEは既に終端として保存されている。
    assert observed == [["completed", "queued"], ["completed", "completed"]]


def test_default_add_does_not_need_the_network_when_the_model_config_is_not_cached(tmp_path, monkeypatch):
    from lss.core import catalog
    from lss.harness import fetch

    monkeypatch.setattr(catalog, "default_model", lambda: {"model_id": "model", "config_url": "https://example.invalid/c.yaml"})

    def offline(_row, target=catalog.INSTRUMENTAL):
        raise fetch.ModelDownloadError("offline")

    monkeypatch.setattr(catalog, "requested_stem", offline)
    queue = SeparationQueue(lambda job: {"status": "error", "error_code": "model_download_failed"},
                            state_path=tmp_path / "queue.json")
    ids = queue.add([tmp_path / "a.wav"])
    assert len(ids) == 1
    item = queue.snapshot()[0]
    assert (item.model_id, item.stem_target) == ("model", catalog.INSTRUMENTAL)


def test_default_add_still_rejects_a_model_that_does_not_output_the_requested_result(tmp_path, monkeypatch):
    from lss.core import catalog

    monkeypatch.setattr(catalog, "default_model", lambda: {"model_id": "model"})

    def vocals_only(_row, target=catalog.INSTRUMENTAL):
        raise ValueError("このモデルはInstrumentalを出力しません")

    monkeypatch.setattr(catalog, "requested_stem", vocals_only)
    queue = SeparationQueue(lambda job: {}, state_path=tmp_path / "queue.json")
    with pytest.raises(ValueError):
        queue.add([tmp_path / "a.wav"])


def test_saved_results_do_not_repeat_the_runtime_description(tmp_path):
    revision = {"runtime_id": "runtime-1", "dependency_versions": [{"name": "x", "version": "1"}] * 200}

    def dispatch(job):
        return {"status": "ok", "outputs": [job["mix_path"] + ".flac"], "runtime_id": "runtime-1",
                "runtime_revision": revision}

    state = tmp_path / "queue.json"
    queue = SeparationQueue(dispatch, state_path=state)
    queue.add([tmp_path / "a.flac", tmp_path / "b.flac"])
    queue.run()
    # 実行中のキューは完全な結果を持ち続ける。
    assert all(item.result["runtime_revision"] == revision for item in queue.snapshot())
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert all("runtime_revision" not in item["result"] for item in saved["items"])
    assert all(item["result"]["runtime_id"] == "runtime-1" for item in saved["items"])
    assert "dependency_versions" not in state.read_text(encoding="utf-8")
    restored = SeparationQueue(dispatch, state_path=state)
    assert [item.status for item in restored.snapshot()] == ["completed", "completed"]
    assert [item.result["outputs"] for item in restored.snapshot()] == [
        [str(tmp_path / "a.flac") + ".flac"], [str(tmp_path / "b.flac") + ".flac"]]


# --- 欲しい結果（伴奏・BGM）------------------------------------------------


def test_frozen_targets_run_in_order_and_survive_a_restart(tmp_path) -> None:
    from unittest.mock import patch

    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    calls = []

    def dispatch(job):
        calls.append(("inst", Path(job["mix_path"]).name))
        return {"status": "ok", "outputs": [job["mix_path"] + ".flac"]}

    def bgm(path, run, on_progress):
        calls.append(("bgm", Path(path).name))
        return {"status": "ok", "bgm": {"k": "held", "n": 2}}

    path = tmp_path / "queue.json"
    queue = SeparationQueue(dispatch, state_path=path, bgm=bgm)
    with patch("lss.core.jobs.iter_input_files", return_value=[source]):
        (item_id,) = queue.add([source], "m", "Instrumental", targets=["bgm", "inst"])
    queue.run()
    assert calls == [("inst", "a.wav"), ("bgm", "a.wav")]
    restored = SeparationQueue(dispatch, state_path=path)
    (item,) = restored.snapshot()
    assert item.id == item_id and item.targets == ["inst", "bgm"] and item.status == "completed"
    assert item.outcomes == {"inst": {"k": "done", "n": 1}, "bgm": {"k": "held", "n": 2}}


def test_an_interrupted_item_keeps_only_accepted_outcomes(tmp_path) -> None:
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"version": 1, "items": [{
        "id": "a", "path": str(tmp_path / "a.wav"), "model_id": "m", "stem_target": "Instrumental",
        "status": "running", "result": {"status": "ok", "outputs": ["x.flac"]}, "progress": None,
        "targets": ["inst", "bgm"],
        "outcomes": {"inst": {"k": "done", "n": 1}, "bgm": {"k": "held", "n": 1}},
    }]}), encoding="utf-8")
    (item,) = SeparationQueue(state_path=path).snapshot()
    assert item.status == "queued"
    assert item.outcomes == {"inst": {"k": "done", "n": 1}}
    assert item.unresolved() == ["bgm"]
    assert item.result == {"status": "ok", "outputs": ["x.flac"]}


@pytest.mark.parametrize("reply", [
    {"status": "error", "error_code": "job_abandoned", "message": "中断"},
    {"status": "error", "error_code": "worker_shutting_down", "message": "終了中"},
])
def test_quitting_while_processing_releases_the_item_without_a_failure(tmp_path, reply) -> None:
    """凍結Design 40a: 処理中に終了した1件は失敗にせず、次の起動で処理待ちへ戻る。受理済みは持ち越す。"""
    from unittest.mock import patch

    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    path = tmp_path / "queue.json"
    queue = None

    def dispatch(job):
        return {"status": "ok", "outputs": [job["mix_path"] + ".flac"]}

    def bgm(path_, run, on_progress):
        queue.abandon()  # 確認カードの「終了する」が工程の途中で届く
        return reply

    queue = SeparationQueue(dispatch, state_path=path, bgm=bgm)
    with patch("lss.core.jobs.iter_input_files", return_value=[source]):
        queue.add([source], "m", "Instrumental", targets=["inst", "bgm"])
    queue.run()

    saved = json.loads(path.read_text(encoding="utf-8"))["items"][0]
    assert saved["status"] == "running", "終了で中断した1件を失敗・完了として保存してはいけない"
    assert saved["outcomes"] == {"inst": {"k": "done", "n": 1}}
    (item,) = SeparationQueue(state_path=path).snapshot()
    assert item.status == "queued" and item.unresolved() == ["bgm"]


def test_quitting_starts_neither_the_second_target_nor_a_queued_follower(tmp_path) -> None:
    from unittest.mock import patch

    first, follower = tmp_path / "a.wav", tmp_path / "b.wav"
    for path in (first, follower):
        path.write_bytes(b"x")
    started = []
    queue = None

    def dispatch(job):
        started.append(("inst", Path(job["mix_path"]).name))
        queue.abandon()  # 伴奏の分離中に終了した
        return {"status": "error", "error_code": "job_abandoned"}

    def bgm(path, run, on_progress):
        started.append(("bgm", Path(path).name))
        return {"status": "ok", "bgm": {"k": "held", "n": 1}}

    path = tmp_path / "queue.json"
    queue = SeparationQueue(dispatch, state_path=path, bgm=bgm)
    with patch("lss.core.jobs.iter_input_files", return_value=[first, follower]):
        queue.add([first, follower], "m", "Instrumental", targets=["inst", "bgm"])
    queue.run()

    assert started == [("inst", "a.wav")]
    restored = SeparationQueue(state_path=path).snapshot()
    assert [(item.path.name, item.status, item.outcomes) for item in restored] == [
        ("a.wav", "queued", None), ("b.wav", "queued", None)]


def test_quitting_stops_writing_the_queue_a_relaunched_lss_owns(tmp_path) -> None:
    """終了した後の古いprocessは、すぐ起動し直した次の LSS のキューの記録を上書きしない。"""
    first = tmp_path / "a.wav"
    first.write_bytes(b"x")
    path = tmp_path / "queue.json"
    queue = None

    def dispatch(job):
        queue.abandon()
        # 次の LSS が同じ記録へ新しい音源を足した
        relaunched = SeparationQueue(state_path=path)
        relaunched.add([tmp_path / "b.wav"], "m", "Instrumental")
        return {"status": "error", "error_code": "job_abandoned"}

    queue = SeparationQueue(dispatch, state_path=path)
    queue.add([first], "m", "Instrumental")
    queue.run()

    restored = SeparationQueue(state_path=path).snapshot()
    assert [item.path.name for item in restored] == ["a.wav", "b.wav"]
    assert [item.status for item in restored] == ["queued", "queued"]


def test_unknown_targets_are_rejected() -> None:
    from lss.core.jobs import normalized_targets

    assert normalized_targets(["bgm", "inst"]) == ["inst", "bgm"]
    for bad in ([], ["vocals"], "inst"):
        with pytest.raises(ValueError):
            normalized_targets(bad)
