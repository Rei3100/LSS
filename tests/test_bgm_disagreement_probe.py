import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.harness.fetch import file_sha256
from scripts import probe_bgm_disagreement_routing as probe


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def artifact(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def fixed_inputs(tmp_path):
    frames = 513
    music = np.random.default_rng(63).normal(0, 0.02, (frames, 2))
    source = tmp_path / "source.wav"
    sf.write(source, music, 44100, subtype="DOUBLE")
    first = tmp_path / "first.flac"
    sf.write(first, music * 0.1, 44100, subtype="PCM_24")
    tags = FLAC(first)
    tags["LSS_OUTPUT_HEADROOM_SHIFT"] = "0"
    tags.save()
    stems = {}
    for role, values in {"music": music, "speech": np.zeros_like(music), "effects": np.zeros_like(music)}.items():
        path = tmp_path / f"{role}.wav"
        sf.write(path, values, 44100, subtype="DOUBLE")
        stems[role] = artifact(path)
    source_artifact = {**artifact(source), "frames": frames}
    case = {"case_id": "example-vocals-only", "source": source_artifact, "reference": artifact(source),
            "wanted_role_policy": "entire_constructed_song_including_singing"}
    run_path = tmp_path / "run.json"
    write(run_path, {"source": source_artifact, "separation": {
        "artifacts": [{**artifact(first), "stem": "Instrumental"}]}})
    original = tmp_path / "plan.json"
    write(original, {"cases": [case]})
    baseline = tmp_path / "report.json"
    write(baseline, {"status": "complete_development_diagnostic", "promotion_allowed": False,
        "runtime_id": "original-runtime", "plan_sha256": file_sha256(original), "cases": [{
            "case_id": case["case_id"], "run_path": str(run_path), "run_sha256": file_sha256(run_path),
            "quality": probe.activity_quality(music.T * 0.1, music.T, [])}]})
    song_plan = tmp_path / "song-plan.json"
    write(song_plan, {"promotion_allowed": False, "holdout_consumed": False, "cases": [case]})
    receipt = tmp_path / "receipt.json"
    write(receipt, {"status": "COMPLETED_CUDA_WITHOUT_FALLBACK", "plan_sha256": file_sha256(song_plan),
        "runtime": {"device": "cuda"}, "cases": [{"case_id": case["case_id"],
            "source_sha256": source_artifact["sha256"], "stems": stems}]})
    activity = tmp_path / "activity.json"
    write(activity, {"schema_version": 1, "method": "stem_energy_share_hysteresis", "onset_share": 0.2,
        "offset_share": 0.1, "onset_music_rms": 0.001, "offset_music_rms": 0.0005,
        "window_frames": 32, "hop_frames": 16, "bridge_gap_frames": 16,
        "minimum_active_frames": 16, "padding_frames": 0})
    return baseline, song_plan, receipt, activity


@pytest.mark.parametrize("mutation", ["duplicate", "incomplete", "fallback"])
def test_invalid_receipt_does_not_start_comparison(tmp_path, mutation):
    paths = fixed_inputs(tmp_path)
    receipt = json.loads(paths[2].read_text())
    if mutation == "duplicate":
        receipt["cases"] *= 2
    elif mutation == "incomplete":
        receipt["cases"] = []
    else:
        receipt["status"] = "cpu_fallback"
    write(paths[2], receipt)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="完了"):
        probe.freeze(*paths, output)
    assert not output.exists()


def test_cached_stem_change_is_rejected_before_evaluation(tmp_path, monkeypatch):
    paths = fixed_inputs(tmp_path)
    plan = probe.freeze(*paths, tmp_path / "comparison")
    with open(tmp_path / "music.wav", "ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(probe, "route_stereo", lambda *a: pytest.fail("推定hash検証が先"))
    with pytest.raises(ValueError, match="変更"):
        probe.evaluate(plan)
    assert not (plan.parent / "cases").exists()


@pytest.mark.parametrize("cue_persistent", [False, True])
def test_all_completed_cases_keep_singing_and_no_promotion(tmp_path, cue_persistent):
    paths = fixed_inputs(tmp_path)
    plan = probe.freeze(*paths, tmp_path / "comparison", cue_persistent=cue_persistent)
    report = json.loads(probe.evaluate(plan).read_text())
    assert report["disposition"] == "development_survivor"
    assert report["promotion_allowed"] is False and report["default_changed"] is False
    assert report["source_hashes_after_verified"]
    assert report["singing_improved_cases"] == ["example-vocals-only"]
    assert report["cases"][0]["music_weight_mean"] == 1
    assert file_sha256(Path(report["cases"][0]["artifact"]["path"])) == report["cases"][0]["artifact"]["sha256"]


def test_a_better_joint_score_cannot_hide_an_individual_channel_regression(tmp_path, monkeypatch):
    baseline, song_plan, receipt, activity = fixed_inputs(tmp_path)
    plan_data = json.loads(song_plan.read_text(encoding="utf-8"))
    plan_data["cases"][0]["wanted_role_policy"] = "instrumental_score_only"
    write(song_plan, plan_data)
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["plan_sha256"] = file_sha256(song_plan)
    write(receipt, receipt_data)
    plan = probe.freeze(baseline, song_plan, receipt, activity, tmp_path / "comparison")

    def damage_right_channel(instrumental, stems, recipe):
        audio = stems["Music"].copy()
        audio[1] *= 0.05
        return SimpleNamespace(
            audio=audio,
            music_weights=np.ones(audio.shape[1]),
            windows=(),
        )

    monkeypatch.setattr(probe, "route_stereo", damage_right_channel)
    report = json.loads(probe.evaluate(plan).read_text(encoding="utf-8"))
    case = report["cases"][0]

    assert case["comparison"]["snr_against_instrumental_db"] > 0
    assert "wanted_signal_regression" not in case["failures"]
    assert "individual_stereo_component_regression" in case["failures"]
    assert any(row["component"] == "right" for row in case["spatial_regressions"])
    assert report["disposition"] == "reject_general_routing"
