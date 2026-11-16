"""保存済みのBGM保留結果を、Stage実行や記録更新なしで再利用する。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lss import application
from lss.bgm import identity, identity_candidates, identity_run
from tests.test_application_bgm_identity import routed_result, _execute


def _snapshot(case, outcome):
    paths = [case[1], case[2], outcome.record_path, outcome.identity_evidence_path, *outcome.held_artifacts]
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths if path is not None}


def test_saved_held_cues_and_current_identity_are_read_without_running_or_writing(routed_result, monkeypatch):
    original = _execute(routed_result)
    before = _snapshot(routed_result, original)
    monkeypatch.setattr(application, "run_bgm_development", lambda *a, **k: pytest.fail("Stage実行は禁止"))
    monkeypatch.setattr(application, "_write_outcome_atomic", lambda *a, **k: pytest.fail("更新は禁止"))
    monkeypatch.setattr(identity_run, "annotate_run", lambda *a, **k: pytest.fail("照合の再計算は禁止"))
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.HELD and outcome.reused is True
    assert outcome.outputs == () and outcome.zero_result is None
    assert outcome.held_artifacts == original.held_artifacts
    assert outcome.identity_evidence_path == original.identity_evidence_path
    assert outcome.recipe_id == original.recipe_id
    assert _snapshot(routed_result, original) == before


@pytest.mark.parametrize("change", ["source", "plan", "run", "candidate", "missing_candidate", "record"])
def test_changed_or_missing_required_input_is_not_reused(routed_result, change):
    original = _execute(routed_result)
    paths = {"source": routed_result[0].sources[0], "plan": routed_result[1], "run": routed_result[2],
             "candidate": original.held_artifacts[0], "missing_candidate": original.held_artifacts[0],
             "record": original.record_path}
    if change == "missing_candidate":
        paths[change].unlink()
    else:
        # run・plan・製品結果の構造が壊れる場合も、例外を漏らさずFAILEDを返す。
        paths[change].write_bytes(b"[]" if change in {"plan", "run", "record"} else b"changed")
    before = original.record_path.read_bytes()
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.FAILED and outcome.reused is False
    assert outcome.held_artifacts == () and outcome.outputs == ()
    assert original.record_path.read_bytes() == before


@pytest.mark.parametrize("change", ["reference", "signal", "annotation", "settings", "missing_annotation"])
def test_stale_optional_identity_keeps_all_held_parent_cues(routed_result, monkeypatch, change):
    original = _execute(routed_result)
    before = original.record_path.read_bytes()
    if change == "settings":
        monkeypatch.setattr(identity, "MIN_VOTES", identity.MIN_VOTES + 1)
    elif change == "missing_annotation":
        original.identity_evidence_path.unlink()
    else:
        paths = {"reference": routed_result[3], "signal": routed_result[4], "annotation": original.identity_evidence_path}
        paths[change].write_bytes(b"changed")
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.HELD and outcome.reused is True
    assert outcome.held_artifacts == original.held_artifacts and outcome.identity_evidence_path is None
    assert original.record_path.read_bytes() == before


def test_initial_records_without_a_plan_field_are_read_using_the_run_plan(routed_result):
    original = _execute(routed_result)
    record = json.loads(original.record_path.read_bytes())
    record.pop("plan")
    original.record_path.write_text(json.dumps(record), encoding="utf-8")
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.held_artifacts == original.held_artifacts


def test_nonstandard_plan_filename_is_retained_and_reused(routed_result):
    request, plan, report, reference, *rest = routed_result
    alternate = plan.with_name("frozen-plan.json")
    plan.rename(alternate)
    case = (request, alternate, report, reference, *rest)
    original = _execute(case)
    assert original.disposition == application.ResultDisposition.HELD
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.HELD and outcome.reused is True


@pytest.mark.parametrize("field,value", [("disposition", "PUBLISHED"), ("disposition", "ZERO"),
                                        ("publication_allowed", 0), ("schema_version", True),
                                        ("recipe_id", "another-recipe"), ("held_artifacts", []),
                                        ("evidence", {"path": "other", "sha256": "a" * 64})])
def test_result_contract_cannot_be_changed_during_readback(routed_result, field, value):
    original = _execute(routed_result)
    record = json.loads(original.record_path.read_bytes())
    record[field] = value
    original.record_path.write_text(json.dumps(record), encoding="utf-8")
    assert application.read_bgm_development_outcome(original.record_path).disposition == application.ResultDisposition.FAILED


def test_product_record_change_during_validation_is_rejected(routed_result, monkeypatch):
    original = _execute(routed_result)
    verify = application._verify_bgm_inputs

    def changed(*a, **k):
        original.record_path.write_bytes(original.record_path.read_bytes() + b"\n")
        verify(*a, **k)

    monkeypatch.setattr(application, "_verify_bgm_inputs", changed)
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.FAILED


def test_late_optional_identity_change_cannot_be_returned_as_current(routed_result, monkeypatch):
    original = _execute(routed_result)
    verify = application._verify_bgm_inputs

    def changed(*a, **k):
        routed_result[3].write_bytes(b"reference changed after first check")
        verify(*a, **k)

    monkeypatch.setattr(application, "_verify_bgm_inputs", changed)
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.held_artifacts == original.held_artifacts and outcome.identity_evidence_path is None


def test_no_activity_remains_held_and_is_not_a_false_target_absence(routed_result):
    request, plan, report, *_ = routed_result
    payload = json.loads(report.read_bytes())
    payload["candidates"] = []
    payload["outcome"] = "unresolved_no_activity_candidates"
    report.write_text(json.dumps(payload), encoding="utf-8")
    original = application.run_bgm_development(request, plan, runner=lambda *a, **k: report)
    outcome = application.read_bgm_development_outcome(original.record_path)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.held_artifacts == () and outcome.zero_result is None
