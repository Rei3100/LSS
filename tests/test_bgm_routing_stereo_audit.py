"""全体改善で隠れる片側損傷と、未完了・改変済みcacheの拒否。"""
from pathlib import Path
import pytest
import soundfile as sf

from lss.harness.fetch import file_sha256
from lss.validation.activity_quality import activity_quality
from scripts import audit_bgm_routing_stereo as audit
from scripts import probe_bgm_disagreement_routing as probe
from tests.test_bgm_disagreement_probe import fixed_inputs, write


def completed(tmp_path):
    paths = fixed_inputs(tmp_path)
    plan = probe.freeze(*paths, tmp_path / "original")
    return probe.evaluate(plan)


def test_global_improvement_with_right_channel_damage_rejects_a_prior_survivor(tmp_path):
    report_path = completed(tmp_path)
    report = audit.read(report_path)
    reference = sf.read(tmp_path / "source.wav", always_2d=True)[0].T
    candidate = reference.copy()
    candidate[1] *= 0.01
    case = report["cases"][0]
    sf.write(case["artifact"]["path"], candidate.T, 44100, subtype="DOUBLE")
    case["artifact"]["sha256"] = file_sha256(Path(case["artifact"]["path"]))
    case["singing"] = False
    case["quality"] = activity_quality(candidate, reference, [(0, reference.shape[1])])
    original_path = report_path.parent / "plan.json"
    original = audit.read(original_path)
    baseline = activity_quality(reference * 0.1, reference, [(0, reference.shape[1])])
    original["cases"][0]["baseline_quality"] = baseline
    write(original_path, original)
    report["plan_sha256"] = file_sha256(original_path)
    assert (case["quality"]["separation_and_activity_fidelity"]["scale_dependent_snr_db"]
            > baseline["separation_and_activity_fidelity"]["scale_dependent_snr_db"])
    case["failures"] = []
    write(report_path, report)
    plan = audit.freeze([report_path], tmp_path / "audit")
    result = audit.read(audit.evaluate(plan))["groups"][0]
    assert result["prior_disposition"] == "development_survivor"
    assert result["disposition"] == "reject_general_routing"
    assert result["newly_detected_spatial_regression_cases"] == [case["case_id"]]
    assert any(row["component"] == "right" and row["metric"] == "projection_gain_db"
               for row in result["cases"][0]["spatial_regressions"])
    assert audit.read(report_path) == report


@pytest.mark.parametrize("mutation", ["duplicate", "incomplete", "holdout", "unfinished", "plan_hash"])
def test_invalid_original_comparison_cannot_be_frozen(tmp_path, mutation):
    path = completed(tmp_path)
    report = audit.read(path)
    if mutation == "duplicate":
        report["cases"] *= 2
    elif mutation == "incomplete":
        report["cases"] = []
    elif mutation == "holdout":
        report["holdout_consumed"] = True
    elif mutation == "unfinished":
        report["source_hashes_after_verified"] = False
    else:
        report["plan_sha256"] = "changed"
    write(path, report)
    with pytest.raises(ValueError):
        audit.freeze([path], tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_changed_cache_is_rejected_before_comparison_and_report_is_not_overwritten(tmp_path):
    path = completed(tmp_path)
    plan = audit.freeze([path], tmp_path / "audit")
    output = audit.evaluate(plan)
    digest = file_sha256(output)
    with pytest.raises(FileExistsError):
        audit.evaluate(plan)
    assert file_sha256(output) == digest
    with (tmp_path / "music.wav").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="変更"):
        audit.evaluate(plan)
    assert file_sha256(output) == digest


def test_singing_is_compared_to_music_and_absence_does_not_create_spatial_scores(tmp_path):
    path = completed(tmp_path)
    plan = audit.freeze([path], tmp_path / "audit")
    result = audit.read(audit.evaluate(plan))
    assert result["groups"][0]["disposition"] == "development_survivor"
    assert result["groups"][0]["cases"][0]["spatial_regressions"] == []
    assert result["promotion_allowed"] is False and result["holdout_consumed"] is False
    assert result["source_hashes_after_verified"] is True
