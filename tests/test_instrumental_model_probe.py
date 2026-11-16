import json

import pytest

from lss.harness.fetch import file_sha256
from scripts.probe_instrumental_model import baseline_inputs, physical_target, summarize


def test_other_is_only_a_development_hypothesis_and_arbitrary_stems_are_refused():
    assert physical_target(["Vocals", "Instrumental"])[0] == "Instrumental"
    stem, evidence = physical_target(["other"])
    assert stem == "other" and "hypothesis_only" in evidence
    for stems in (["drums"], ["other", "vocals"], [], ["Instrumental", "instrumental"]):
        with pytest.raises(ValueError):
            physical_target(stems)


def test_mean_or_quiet_improvement_does_not_overrule_regression():
    cases = [{"case_id": "quiet", "variant": "quiet-accompaniment",
              "comparison": {"regression": False, "uncorrected_snr_delta_db": 4.0}},
             {"case_id": "absence", "variant": "vocals-only", "comparison": {"regression": True}}]
    decision = summarize(cases)
    assert decision["quiet_improvement_cases"] == ["quiet"]
    assert decision["disposition"] == "reject_as_general_replacement"
    assert decision["regression_cases"] == ["absence"]
    cases[1]["comparison"]["regression"] = False
    assert summarize(cases)["disposition"] == "development_candidate_survivor"


def test_no_quiet_improvement_does_not_claim_task_solved():
    assert summarize([{"case_id": "mixed", "variant": "singing-mix", "comparison": {"regression": False}}])[
        "disposition"] == "reject_as_general_replacement"


def test_baseline_rejects_incomplete_duplicate_or_tampered_cached_evidence(tmp_path):
    signal = tmp_path / "signal.wav"
    signal.write_bytes(b"fixed signal bytes")
    artifact = {"path": str(signal), "sha256": file_sha256(signal)}
    plan = {"role": "development_diagnostic_only", "promotion_allowed": False,
            "cases": [{"case_id": "one", "artifacts": {"input": artifact, "target": artifact, "vocals": artifact}}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    baseline = {"role": "development_diagnostic_only", "promotion_allowed": False,
                "plan_sha256": file_sha256(plan_path), "cases": [{"case_id": "one", "output_sha256": artifact["sha256"],
                "result": {"status": "ok", "outputs": [str(signal)]}}]}
    baseline_path = tmp_path / "report.json"
    baseline_path.write_text(json.dumps(baseline))
    assert baseline_inputs(baseline_path)[0] == baseline
    baseline["cases"].append(baseline["cases"][0])
    baseline_path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="一意"):
        baseline_inputs(baseline_path)
    baseline["cases"] = []
    baseline_path.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="一意"):
        baseline_inputs(baseline_path)
    baseline["cases"] = [{"case_id": "one", "output_sha256": artifact["sha256"],
                          "result": {"status": "ok", "outputs": [str(signal)]}}]
    baseline_path.write_text(json.dumps(baseline))
    signal.write_bytes(b"changed signal bytes")
    with pytest.raises(ValueError, match="内容"):
        baseline_inputs(baseline_path)
