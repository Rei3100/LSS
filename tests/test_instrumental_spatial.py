import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.harness.fetch import file_sha256
from scripts import probe_instrumental_spatial as probe


def fixed_baseline(tmp_path):
    rng = np.random.default_rng(51)
    target = rng.normal(0, 0.03, (2, 257))
    vocals = rng.normal(0, 0.01, target.shape)
    artifacts = {}
    for role, values in {"input": target + vocals, "target": target, "vocals": vocals}.items():
        path = tmp_path / f"{role}.wav"
        sf.write(path, values.T, 44100, subtype="DOUBLE")
        artifacts[role] = {"path": str(path), "sha256": file_sha256(path)}
    saved = tmp_path / "first.flac"
    sf.write(saved, (target * 0.8).T, 44100, subtype="PCM_24")
    tags = FLAC(saved)
    tags["LSS_OUTPUT_HEADROOM_SHIFT"] = "0"
    tags.save()
    estimate = sf.read(saved, always_2d=True)[0].T
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": [{"case_id": "quiet", "sample_rate": 44100,
        "frames": 257, "artifacts": artifacts}]}), encoding="utf-8")
    baseline = tmp_path / "report.json"
    baseline.write_text(json.dumps({"role": "development_diagnostic_only", "promotion_allowed": False,
        "plan_sha256": file_sha256(plan), "cases": [{"case_id": "quiet", "variant": "quiet-accompaniment",
            "result": {"outputs": [str(saved)]}, "output_sha256": file_sha256(saved),
            "quality": probe.instrumental_quality(estimate, target, vocals)}]}), encoding="utf-8")
    return baseline, artifacts


def test_changed_audio_stops_before_creating_comparison(tmp_path, monkeypatch):
    baseline, artifacts = fixed_baseline(tmp_path)
    with open(artifacts["target"]["path"], "ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(probe, "refine_stereo", lambda *a: pytest.fail("入力検証が先"))
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="変更"):
        probe.run(baseline, output)
    assert not output.exists()


def test_missing_case_cannot_be_reported_as_complete(tmp_path):
    baseline, _ = fixed_baseline(tmp_path)
    payload = json.loads(baseline.read_text())
    payload["cases"] = []
    baseline.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="完全"):
        probe.run(baseline, tmp_path / "comparison")


def test_post_run_mutation_has_no_final_report(tmp_path, monkeypatch):
    baseline, artifacts = fixed_baseline(tmp_path)
    original = probe.refine_stereo

    def mutate(*args):
        result = original(*args)
        with open(artifacts["input"]["path"], "ab") as stream:
            stream.write(b"changed during run")
        return result

    monkeypatch.setattr(probe, "refine_stereo", mutate)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="比較中"):
        probe.run(baseline, output)
    assert (output / "comparison-plan.json").exists()
    assert not (output / "report.json").exists()


def test_complete_matrix_keeps_fixed_decisions_and_original_audio(tmp_path):
    baseline, artifacts = fixed_baseline(tmp_path)
    output = tmp_path / "comparison"
    report = json.loads(probe.run(baseline, output).read_text())
    assert report["status"] == "complete_development_comparison"
    assert report["promotion_allowed"] is False and report["default_changed"] is False
    assert report["source_hashes_after_verified"] is True
    assert [v["weight"] for v in report["cases"][0]["variants"]] == list(probe.BLENDS)
    assert len(report["decisions"]) == len(probe.BLENDS)
    for variant in report["cases"][0]["variants"]:
        data, rate = sf.read(variant["artifact"]["path"], always_2d=True)
        assert data.shape == (257, 2) and rate == 44100 and np.isfinite(data).all()
        assert file_sha256(Path(variant["artifact"]["path"])) == variant["artifact"]["sha256"]
    assert all(file_sha256(Path(record["path"])) == record["sha256"] for record in artifacts.values())
