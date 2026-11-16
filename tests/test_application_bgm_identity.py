"""保留候補へ既知録音の根拠を接続し、未一致・短い残り・以前の結果を守る。"""
import json

import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.bgm import identity, identity_candidates, identity_run
from lss.harness.fetch import file_sha256
from tests.test_application import _bgm_development_plan


@pytest.fixture
def routed_result(tmp_path, monkeypatch):
    request, plan, source = _bgm_development_plan(tmp_path)
    signal = tmp_path / "routed.wav"
    sf.write(signal, np.zeros((40 * identity.SAMPLE_RATE, 2)), identity.SAMPLE_RATE)
    reference = tmp_path / "known.flac"
    reference.write_bytes(b"pinned reference")
    candidates = []
    for name, start, end in (("cue-a", 0, 20), ("cue-b", 25, 40)):
        path = tmp_path / "candidates" / f"{name}.flac"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        candidates.append({
            "asset": {"candidate_asset_id": name, "path": str(path), "sha256": file_sha256(path),
                      "representation": "noncanonical_staged_candidate", "publication_state": "not_published"},
            "origin": {"separation_recipe_id": "renderer", "activity_recipe_id": "activity",
                       "sample_rate": identity.SAMPLE_RATE, "start_frame": start * identity.SAMPLE_RATE,
                       "end_frame": end * identity.SAMPLE_RATE},
        })
    report = tmp_path / "run.json"
    report.write_text(json.dumps({
        "status": identity_run.ROUTED_STATUS, "promotion_allowed": False, "holdout_consumed": False,
        "publication_state": "not_published", "source_hashes_after_verified": True,
        "plan_sha256": file_sha256(plan), "outcome": "unresolved_candidates",
        "source": {"path": str(source), "sha256": file_sha256(source)},
        "renderer_id": "bgm-development-cue-renderer-v1-test",
        "rendering": {"signal": {"path": str(signal), "sha256": file_sha256(signal)}},
        "candidates": candidates,
    }), encoding="utf-8")
    match = identity.RecordingMatch(
        "known", str(reference), file_sha256(reference), 1.0, 15.0, 2.0, 1.0,
        200, 0.99, 0.0, 0.01, 99.0, 10, 0.0, 0.99, 0.0, 100, 0, 1, "known_exact",
    )
    monkeypatch.setattr(identity.ReferenceIndex, "build", lambda *a, **k: None)
    monkeypatch.setattr(identity, "match", lambda *a, **k: [match])
    return request, plan, report, reference, signal, candidates


def _execute(case, **kwargs):
    request, plan, report, reference, *_ = case
    return application.run_bgm_development(
        request, plan, runner=lambda *a, **k: report, reference_paths=[reference], **kwargs,
    )


def test_recording_spans_are_attached_without_publishing_or_removing_any_parent(routed_result):
    *_, report, reference, signal, candidates = routed_result
    before = report.read_bytes()
    outcome = _execute(routed_result)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.outputs == () and outcome.zero_result is None
    assert len(outcome.held_artifacts) == 2
    assert report.read_bytes() == before
    assert outcome.identity_evidence_path == report.parent / identity_run.RECORD_NAME
    recorded = json.loads(outcome.record_path.read_bytes())
    view = recorded["recording_identity"]
    assert view["sha256"] == file_sha256(outcome.identity_evidence_path)
    assert view["semantic_role"] == "unresolved"
    [activity] = view["activity_runs"]
    pieces = activity["candidates"]
    assert [(c["start_seconds"], c["end_seconds"]) for c in pieces] == [(0, 1), (1, 15), (15, 20), (25, 40)]
    assert pieces[0]["short_unresolved_remainder"] is True
    assert pieces[1]["identity_state"] == "known_exact"
    assert pieces[1]["identity_evidence"][0]["reference_sha256"] == file_sha256(reference)
    assert pieces[-1]["candidate_id"] == "cue-b" and pieces[-1]["identity_state"] == "unresolved"
    assert {d["decision"] for d in activity["novelty_decisions"]} == {"pending", "known_recording_exists"}
    stamp = outcome.identity_evidence_path.stat().st_mtime_ns
    _execute(routed_result)
    assert outcome.identity_evidence_path.stat().st_mtime_ns == stamp


@pytest.mark.parametrize("change", ["reference", "signal", "annotation", "settings", "missing_reference"])
def test_late_identity_change_preserves_previous_outcome(routed_result, monkeypatch, change):
    request, plan, report, reference, signal, _ = routed_result
    previous = _execute(routed_result).record_path
    before = previous.read_bytes()
    original_write = application._write_outcome_atomic

    def changed_write(path, payload, validate):
        def changed_validation():
            if change == "settings":
                monkeypatch.setattr(identity, "MIN_VOTES", identity.MIN_VOTES + 1)
            elif change == "missing_reference":
                reference.unlink()
            else:
                target = {"reference": reference, "signal": signal,
                          "annotation": report.parent / identity_run.RECORD_NAME}[change]
                target.write_bytes(target.read_bytes() + b"\n")
            validate()
        original_write(path, payload, changed_validation)

    monkeypatch.setattr(application, "_write_outcome_atomic", changed_write)
    outcome = _execute(routed_result)
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert previous.read_bytes() == before
    assert not list(plan.parent.glob(".application-outcome.json-*.tmp"))


def test_annotation_with_missing_candidate_span_cannot_be_joined(routed_result):
    outcome = _execute(routed_result)
    record = json.loads(outcome.identity_evidence_path.read_bytes())
    record["activity_runs"][0]["candidates"].pop(0)
    outcome.identity_evidence_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(identity_candidates.IdentityCandidatesError, match="欠落"):
        identity_candidates.annotation_view(routed_result[2])


def test_failed_identity_decode_is_a_failure_and_preserves_the_previous_outcome(routed_result, monkeypatch):
    previous = _execute(routed_result).record_path
    before = previous.read_bytes()
    (routed_result[2].parent / identity_run.RECORD_NAME).unlink()

    def failed_match(*a, **k):
        raise identity.DecodeError("照合用の音声を復号できません")

    monkeypatch.setattr(identity, "match", failed_match)
    outcome = _execute(routed_result)
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert "復号" in outcome.message
    assert previous.read_bytes() == before


def test_resume_without_reference_arguments_keeps_current_identity_evidence(routed_result, monkeypatch):
    initial = _execute(routed_result)
    recorded = json.loads(initial.record_path.read_bytes())
    request, plan, report, *_ = routed_result
    monkeypatch.setattr(identity_run, "annotate_run", lambda *a, **k: pytest.fail("保存済み根拠を使う"))
    outcome = application.run_bgm_development(request, plan, runner=lambda *a, **k: report, resume=True)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.identity_evidence_path == initial.identity_evidence_path
    assert json.loads(outcome.record_path.read_bytes())["recording_identity"] == recorded["recording_identity"]


def test_resume_retains_stale_identity_as_unavailable_and_recovers_when_settings_return(routed_result, monkeypatch):
    initial = _execute(routed_result)
    recorded = json.loads(initial.record_path.read_bytes())["recording_identity"]
    request, plan, report, *_ = routed_result
    with monkeypatch.context() as changed:
        changed.setattr(identity, "MIN_VOTES", identity.MIN_VOTES + 1)
        outcome = application.run_bgm_development(request, plan, runner=lambda *a, **k: report, resume=True)
        assert outcome.disposition == application.ResultDisposition.HELD
        assert len(outcome.held_artifacts) == 2 and outcome.identity_evidence_path is None
        assert outcome.unavailable_evidence == ("recording_identity",)
        assert outcome.outputs == () and outcome.zero_result is None
        assert json.loads(outcome.record_path.read_bytes())["recording_identity"] == recorded
    monkeypatch.setattr(identity_run, "annotate_run", lambda *a, **k: pytest.fail("保存済み根拠を再照合しない"))
    recovered = application.read_bgm_development_outcome(initial.record_path)
    assert recovered.disposition == application.ResultDisposition.HELD
    assert recovered.held_artifacts == initial.held_artifacts
    assert recovered.identity_evidence_path == initial.identity_evidence_path
    assert recovered.unavailable_evidence == ()


def test_plan_change_during_outcome_save_preserves_the_previous_record(routed_result, monkeypatch):
    previous = _execute(routed_result).record_path
    before = previous.read_bytes()
    original_write = application._write_outcome_atomic

    def changed_write(path, payload, validate):
        def changed_validation():
            plan = routed_result[1]
            plan.write_bytes(plan.read_bytes() + b"\n")
            validate()
        original_write(path, payload, changed_validation)

    monkeypatch.setattr(application, "_write_outcome_atomic", changed_write)
    outcome = _execute(routed_result)
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert previous.read_bytes() == before
