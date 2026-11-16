"""製品要求はmodelを選ばず、処理結果の意味を明示して返す。"""
import json
from pathlib import Path

import pytest

from lss import application
from lss.core import catalog, jobs, models_store
from lss.harness import fetch, tsv_io
from lss.harness.fetch import file_sha256


def _default_catalog(tmp_path, monkeypatch):
    ledger = tmp_path / "models.tsv"
    row = {
        "model_id": "technical-model", "状態": "既定",
        "config_url": "https://example.test/config.yaml",
    }
    tsv_io.save_tsv(ledger, list(row), [row])
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    monkeypatch.setenv("LSS_MODEL_DIR", str(tmp_path / "models"))
    config = fetch.model_cache_dir() / row["model_id"] / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("training:\n  instruments: [Vocals, Instrumental]\n", encoding="utf-8")


def test_accompaniment_request_selects_the_default_internally(tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    calls = []
    queue = jobs.SeparationQueue(
        lambda job: calls.append(job) or {
            "status": "ok", "outputs": [str(tmp_path / "result.flac")],
            "recipe_id": "separation-v1-exact",
        },
        state_path=tmp_path / "queue.json",
    )
    request = application.ProcessingRequest.from_paths(
        [source], application.UserIntent.EXTRACT_ACCOMPANIMENT,
    )

    submission = application.submit(request, queue)
    assert submission.outcome is None and len(submission.queue_item_ids) == 1
    queue.run()

    assert calls == [{
        "mix_path": str(source.resolve()),
        "model_id": "technical-model",
        "stem_target": catalog.INSTRUMENTAL,
    }]
    outcome = application.outcome_from_item(queue.snapshot()[0])
    assert outcome == application.ProcessingOutcome(
        disposition=application.ResultDisposition.PUBLISHED,
        outputs=(tmp_path / "result.flac",),
        recipe_id="separation-v1-exact",
    )


def test_bgm_request_uses_the_product_candidate_route_without_entering_the_queue(
    tmp_path, monkeypatch,
):
    queue = jobs.SeparationQueue(
        lambda job: pytest.fail("BGM要求を伴奏用queueへ送らない"),
        state_path=tmp_path / "queue.json",
    )
    request = application.ProcessingRequest.from_paths(
        [tmp_path / "episode.wav"], application.UserIntent.EXTRACT_BGM_CUES,
    )
    held = tmp_path / "candidate.flac"
    evidence = tmp_path / "run.json"
    expected = application.ProcessingOutcome(
        disposition=application.ResultDisposition.HELD,
        held_artifacts=(held,), evidence_path=evidence,
        message="意味を未確定の候補を保留しました",
    )
    calls = []
    monkeypatch.setattr(
        application, "run_bgm_candidate_request",
        lambda actual: calls.append(actual) or expected,
    )

    submission = application.submit(request, queue)

    assert submission.queue_item_ids == ()
    assert submission.outcome == expected
    assert calls == [request]
    assert queue.snapshot() == []


def test_bgm_product_route_converts_runner_failures_to_a_terminal_failure(tmp_path):
    queue = jobs.SeparationQueue(
        lambda job: pytest.fail("BGM要求を伴奏用queueへ送らない"),
        state_path=tmp_path / "queue.json",
    )
    request = application.ProcessingRequest.from_paths(
        [tmp_path / "episode.wav"], application.UserIntent.EXTRACT_BGM_CUES,
    )

    def fail(_request):
        raise OSError("候補保存に失敗しました token=SECRET")

    submission = application.submit(request, queue, bgm_runner=fail)

    assert submission.outcome.disposition == application.ResultDisposition.FAILED
    assert submission.outcome.error_code == "bgm_candidate_processing_failed"
    assert "SECRET" not in submission.outcome.message
    assert queue.snapshot() == []


@pytest.mark.parametrize(
    ("result", "disposition", "reused"),
    [
        ({"status": "ok", "outputs": ["result.flac"], "recipe_id": "r1"}, "PUBLISHED", False),
        ({"status": "skipped", "outputs": ["result.flac"], "recipe_id": "r1"}, "PUBLISHED", True),
        ({
            "status": "ok", "outputs": [], "recipe_id": "r1",
            "zero_result": {
                "reason": "no_wanted_target",
                "validation_result": {
                    "validator_id": "exact_digital_silence",
                    "validator_revision": "1",
                    "hard_gates_passed": True,
                    "evidence": {"all_samples_exactly_zero": True},
                },
            },
        }, "ZERO", False),
        ({"status": "error", "error_code": "unsupported_channels", "message": "6ch"}, "UNSUPPORTED", False),
        ({"status": "error", "error_code": "gpu_unavailable", "message": "CUDA"}, "FAILED", False),
    ],
)
def test_worker_results_have_distinct_product_dispositions(result, disposition, reused):
    outcome = application.outcome_from_worker_result(result)
    assert outcome.disposition == disposition
    assert outcome.reused is reused


def test_invalid_worker_success_is_a_failure_not_a_false_publication():
    outcome = application.outcome_from_worker_result({"status": "ok", "outputs": []})
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert outcome.error_code == "invalid_worker_response"


def test_active_queue_item_has_no_terminal_outcome(tmp_path):
    item = jobs.QueueItem("id", tmp_path / "source.wav", "internal", "Instrumental")
    assert application.outcome_from_item(item) is None


def test_product_outcome_is_recovered_from_the_durable_queue(tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    state_path = tmp_path / "queue.json"
    queue = jobs.SeparationQueue(
        lambda job: {"status": "skipped", "outputs": [str(tmp_path / "existing.flac")],
                     "recipe_id": "separation-v1-existing"},
        state_path=state_path,
    )
    submission = application.submit(application.ProcessingRequest(
        sources=(source,), intent=application.UserIntent.EXTRACT_ACCOMPANIMENT,
    ), queue)
    queue.run()

    restored = jobs.SeparationQueue(
        lambda job: pytest.fail("終端結果の復元ではworkerを再実行しない"), state_path=state_path,
    )
    item = restored.snapshot()[0]
    assert item.id == submission.queue_item_ids[0]
    assert application.outcome_from_item(item) == application.ProcessingOutcome(
        disposition=application.ResultDisposition.PUBLISHED,
        outputs=(tmp_path / "existing.flac",),
        recipe_id="separation-v1-existing",
        reused=True,
    )


def test_empty_request_is_rejected_before_resolution():
    with pytest.raises(ValueError, match="入力"):
        application.ProcessingRequest((), application.UserIntent.EXTRACT_ACCOMPANIMENT)


def _bgm_development_plan(tmp_path):
    source = tmp_path / "episode.wav"
    source.write_bytes(b"immutable episode")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "status": "FROZEN_BEFORE_ROUTED_CANDIDATES",
        "promotion_allowed": False,
        "holdout_consumed": False,
        "source": {
            "path": str(source.resolve()),
            "sha256": file_sha256(source),
        },
        "renderer_id": "bgm-development-cue-renderer-v1-test",
    }), encoding="utf-8")
    request = application.ProcessingRequest(
        (source,), application.UserIntent.EXTRACT_BGM_CUES,
    )
    return request, plan, source


def test_development_bgm_candidate_is_held_and_never_reported_as_published(tmp_path):
    request, plan, source = _bgm_development_plan(tmp_path)
    candidate = tmp_path / "candidates" / "candidate.flac"
    candidate.parent.mkdir()
    candidate.write_bytes(b"unpublished candidate")
    report = tmp_path / "run.json"

    def run(plan_path, *, resume):
        assert plan_path == plan.resolve() and resume is True
        report.write_text(json.dumps({
            "status": "completed_development_routed_episode_candidates",
            "promotion_allowed": False,
            "holdout_consumed": False,
            "publication_state": "not_published",
            "source_hashes_after_verified": True,
            "plan_sha256": file_sha256(plan),
            "outcome": "unresolved_candidates",
            "source": {"path": str(source.resolve()), "sha256": file_sha256(source)},
            "renderer_id": "bgm-development-cue-renderer-v1-test",
            "candidates": [{"asset": {
                "path": str(candidate.resolve()),
                "sha256": file_sha256(candidate),
                "representation": "noncanonical_staged_candidate",
                "publication_state": "not_published",
            }}],
        }), encoding="utf-8")
        return report

    outcome = application.run_bgm_development(request, plan, resume=True, runner=run)

    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.outputs == ()
    assert outcome.held_artifacts == (candidate.resolve(),)
    assert outcome.evidence_path == report.resolve()
    assert outcome.record_path == (tmp_path / "application-outcome.json").resolve()
    assert outcome.recipe_id == "bgm-development-cue-renderer-v1-test"
    recorded = json.loads(outcome.record_path.read_text(encoding="utf-8"))
    assert recorded == {
        "schema_version": 1,
        "type": "processing_outcome",
        "intent": "extract_bgm_cues",
        "source": {"path": str(source.resolve()), "sha256": file_sha256(source)},
        "disposition": "HELD",
        "outputs": [],
        "held_artifacts": [{"path": str(candidate.resolve()), "sha256": file_sha256(candidate)}],
        "evidence": {"path": str(report.resolve()), "sha256": file_sha256(report)},
        "plan": {"path": str(plan.resolve()), "sha256": file_sha256(plan)},
        "recipe_id": "bgm-development-cue-renderer-v1-test",
        "publication_allowed": False,
        "reason": "unresolved_candidates",
    }


def test_development_bgm_no_activity_is_held_not_a_false_zero(tmp_path):
    request, plan, source = _bgm_development_plan(tmp_path)
    report = tmp_path / "run.json"

    def run(_plan_path, *, resume):
        report.write_text(json.dumps({
            "status": "completed_development_routed_episode_candidates",
            "promotion_allowed": False,
            "holdout_consumed": False,
            "publication_state": "not_published",
            "source_hashes_after_verified": True,
            "plan_sha256": file_sha256(plan),
            "outcome": "unresolved_no_activity_candidates",
            "source": {"path": str(source.resolve()), "sha256": file_sha256(source)},
            "renderer_id": "bgm-development-cue-renderer-v1-test",
            "candidates": [],
        }), encoding="utf-8")
        return report

    outcome = application.run_bgm_development(request, plan, runner=run)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.held_artifacts == () and outcome.zero_result is None
    assert json.loads(outcome.record_path.read_text(encoding="utf-8"))["reason"] == (
        "unresolved_no_activity_candidates"
    )


@pytest.mark.parametrize("field", ["promotion_allowed", "holdout_consumed", "publication_state"])
def test_development_bgm_rejects_a_result_that_crosses_the_product_boundary(tmp_path, field):
    request, plan, source = _bgm_development_plan(tmp_path)
    report = tmp_path / "run.json"

    def run(_plan_path, *, resume):
        payload = {
            "status": "completed_development_routed_episode_candidates",
            "promotion_allowed": False,
            "holdout_consumed": False,
            "publication_state": "not_published",
            "source_hashes_after_verified": True,
            "plan_sha256": file_sha256(plan),
            "outcome": "unresolved_no_activity_candidates",
            "source": {"path": str(source.resolve()), "sha256": file_sha256(source)},
            "renderer_id": "bgm-development-cue-renderer-v1-test",
            "candidates": [],
        }
        payload[field] = True if field != "publication_state" else "published"
        report.write_text(json.dumps(payload), encoding="utf-8")
        return report

    outcome = application.run_bgm_development(request, plan, runner=run)
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert outcome.error_code == "invalid_bgm_development_result"
    assert not (tmp_path / "application-outcome.json").exists()


def test_development_bgm_rejects_a_request_for_another_source(tmp_path):
    request, plan, _ = _bgm_development_plan(tmp_path)
    other = tmp_path / "other.wav"
    other.write_bytes(b"other")
    mismatched = application.ProcessingRequest((other,), request.intent)

    outcome = application.run_bgm_development(
        mismatched, plan, runner=lambda *a, **k: pytest.fail("原本照合が先"),
    )
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert "原本" in outcome.message


def test_candidate_change_before_outcome_publication_preserves_previous_record(
        tmp_path, monkeypatch):
    request, plan, source = _bgm_development_plan(tmp_path)
    candidate = tmp_path / "candidates" / "candidate.flac"
    candidate.parent.mkdir()
    candidate.write_bytes(b"candidate before change")
    report = tmp_path / "run.json"
    previous = tmp_path / "application-outcome.json"
    previous.write_bytes(b"previous validated outcome")

    def run(_plan_path, *, resume):
        report.write_text(json.dumps({
            "status": "completed_development_routed_episode_candidates",
            "promotion_allowed": False,
            "holdout_consumed": False,
            "publication_state": "not_published",
            "source_hashes_after_verified": True,
            "plan_sha256": file_sha256(plan),
            "outcome": "unresolved_candidates",
            "source": {"path": str(source.resolve()), "sha256": file_sha256(source)},
            "renderer_id": "bgm-development-cue-renderer-v1-test",
            "candidates": [{"asset": {
                "path": str(candidate.resolve()),
                "sha256": file_sha256(candidate),
                "representation": "noncanonical_staged_candidate",
                "publication_state": "not_published",
            }}],
        }), encoding="utf-8")
        return report

    original_write = application._write_outcome_atomic

    def change_before_replace(path, payload, validate):
        def changed_validation():
            candidate.write_bytes(b"candidate changed during outcome save")
            validate()
        return original_write(path, payload, changed_validation)

    monkeypatch.setattr(application, "_write_outcome_atomic", change_before_replace)
    outcome = application.run_bgm_development(request, plan, runner=run)

    assert outcome.disposition == application.ResultDisposition.FAILED
    assert previous.read_bytes() == b"previous validated outcome"
    assert not list(tmp_path.glob(".application-outcome.json-*.tmp"))


def test_bgm_step_reports_an_unreachable_worker_as_an_engine_stop(tmp_path, monkeypatch) -> None:
    """画面の順次処理から呼ぶBGM候補工程。workerへ届かないのは行の失敗ではなくHALTの理由。"""
    import numpy as np
    import soundfile as sf

    from lss import application
    from lss.ui import contract

    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    source = tmp_path / "episode.wav"
    # 完全な無音は解析せずに検証済みの無しになるので、workerへ届く音を渡す。
    tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(44100) / 44100)
    sf.write(str(source), np.stack([tone, tone], axis=1).astype("float32"), 44100, subtype="PCM_16")

    def unreachable(job, on_progress=None):
        raise ConnectionRefusedError("pipe is closed")

    result = application.bgm_candidate_step(source, unreachable)
    assert result["status"] == "error" and result["error_code"] == "worker_unavailable"
    failure = contract.classify_failure(result["error_code"])
    assert failure.scope is contract.FailureScope.SESSION
