"""一時的に使えない保留根拠の復元と、同時保存で新しい良い結果を消さない境界。"""
import json
from pathlib import Path

import pytest

from lss import application
from lss.bgm import held_results
from lss.validation import reference_residual_episode as stage
from tests.test_application_bgm_reference_residual import bundle, case, execute


@pytest.mark.parametrize("kind", ["music_alternative", "reference_residual", "recording_identity"])
def test_unavailable_evidence_is_retained_on_resume_and_recovers_without_stage_execution(bundle, kind, monkeypatch):
    previous = execute(bundle)
    assert previous.disposition == application.ResultDisposition.HELD, previous.message
    snapshot = previous.record_path.read_bytes()
    target = {"music_alternative": previous.alternative_evidence_path,
              "reference_residual": previous.reference_residual_evidence_path,
              "recording_identity": previous.identity_evidence_path}[kind]
    target_bytes = target.read_bytes(); target.unlink()
    checked = application.read_bgm_development_outcome(previous.record_path)
    assert checked.disposition == application.ResultDisposition.HELD and checked.held_artifacts == previous.held_artifacts
    assert kind in checked.unavailable_evidence and checked.outputs == () and checked.zero_result is None
    assert previous.record_path.read_bytes() == snapshot
    resumed = execute(bundle, attach=False)
    assert resumed.disposition == application.ResultDisposition.HELD, resumed.message
    assert kind in resumed.unavailable_evidence
    before, after = json.loads(snapshot), json.loads(previous.record_path.read_bytes())
    for key in ("recording_identity", "held_alternatives", "held_reference_residuals"):
        assert after[key] == before[key]
    target.write_bytes(target_bytes)
    monkeypatch.setattr(stage, "_render", lambda *a: pytest.fail("根拠の復元後にStageを再実行しない"))
    recovered = application.read_bgm_development_outcome(previous.record_path)
    assert recovered.disposition == application.ResultDisposition.HELD
    assert recovered.unavailable_evidence == ()
    assert recovered.held_alternative_artifacts == previous.held_alternative_artifacts
    assert recovered.held_reference_residual_artifacts == previous.held_reference_residual_artifacts
    assert recovered.identity_evidence_path == previous.identity_evidence_path


def test_newer_concurrent_outcome_with_added_branches_cannot_be_replaced_by_an_older_save(bundle, monkeypatch):
    previous = execute(bundle, attach=False)
    assert previous.disposition == application.ResultDisposition.HELD
    newer = json.loads(previous.record_path.read_bytes())
    newer["held_alternatives"] = held_results.alternative_view(bundle[2], bundle[4])
    newer["held_reference_residuals"] = stage.read_result(bundle[2], bundle[3])
    newer_bytes = (json.dumps(newer, ensure_ascii=False, indent=2) + "\n").encode()
    writer = application._write_outcome_atomic
    def concurrent(path, payload, validate):
        path.write_bytes(newer_bytes)
        writer(path, payload, validate)
    monkeypatch.setattr(application, "_write_outcome_atomic", concurrent)
    older = execute(bundle, attach=False)
    assert older.disposition == application.ResultDisposition.FAILED
    assert "別の処理で更新" in older.message
    assert previous.record_path.read_bytes() == newer_bytes
    latest = application.read_bgm_development_outcome(previous.record_path)
    assert latest.disposition == application.ResultDisposition.HELD and latest.unavailable_evidence == ()
    assert len(latest.held_alternative_artifacts) == len(latest.held_reference_residual_artifacts) == 1


def test_concurrent_first_public_outcome_record_is_preserved(bundle, monkeypatch):
    path = bundle[1].parent / "application-outcome.json"
    assert not path.exists()
    writer = application._write_outcome_atomic
    created = []
    def concurrent(path, payload, validate):
        newer = {**payload, "held_alternatives": held_results.alternative_view(bundle[2], bundle[4])}
        data = (json.dumps(newer, ensure_ascii=False, indent=2) + "\n").encode()
        path.write_bytes(data); created.append(data)
        writer(path, payload, validate)
    monkeypatch.setattr(application, "_write_outcome_atomic", concurrent)
    result = execute(bundle, attach=False)
    assert result.disposition == application.ResultDisposition.FAILED
    assert path.read_bytes() == created[0]
    latest = application.read_bgm_development_outcome(path)
    assert latest.disposition == application.ResultDisposition.HELD and len(latest.held_alternative_artifacts) == 1


def test_evidence_from_another_upstream_record_is_not_inherited(bundle):
    previous = execute(bundle)
    record = json.loads(previous.record_path.read_bytes()); record["evidence"]["sha256"] = "a" * 64
    previous.record_path.write_text(json.dumps(record), encoding="utf-8")
    current = execute(bundle, attach=False)
    assert current.disposition == application.ResultDisposition.HELD
    assert current.held_alternative_artifacts == current.held_reference_residual_artifacts == ()
    assert current.unavailable_evidence == ()
