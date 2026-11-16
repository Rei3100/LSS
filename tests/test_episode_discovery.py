import json
from pathlib import Path

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.validation.episode_discovery import run_episode_discovery


def test_episode_only_run_preserves_unknowns_and_source(tmp_path):
    source_root = tmp_path / "user-corpus"
    source_root.mkdir()
    source = source_root / "episode.wav"
    samples = np.zeros((1000, 2), dtype=np.float32)
    samples[200:600] = 0.1
    sf.write(source, samples, 1000, subtype="FLOAT")
    source_before = source.read_bytes()
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "method": "windowed_rms_hysteresis",
        "window_frames": 100,
        "hop_frames": 50,
        "onset_rms": 0.08,
        "offset_rms": 0.04,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }), encoding="utf-8")

    def separate(job, *, destination_dir):
        destination_dir.mkdir(parents=True, exist_ok=True)
        output = destination_dir / "episode - Instrumental.flac"
        sf.write(output, samples, 1000, subtype="PCM_24")
        tags = FLAC(output)
        tags["LSS_STEM"] = "Instrumental"
        tags.save()
        return {
            "status": "ok", "outputs": [str(output)],
            "input_sha256": __import__("hashlib").sha256(source_before).hexdigest(),
            "recipe_id": "separation-test", "recipe_revision": {"recipe_id": "separation-test"},
            "runtime_id": "runtime-test", "runtime_revision": {"runtime_id": "runtime-test"},
            "config_sha256": "c" * 64, "elapsed_seconds": 1.0, "rtf": 1.0,
            "peak_vram_mb": 1.0,
        }

    output_dir = tmp_path / "run"
    run_path = run_episode_discovery(
        source, [recipe], output_dir,
        source_id="work/episode-1", model_id="model", separate=separate,
    )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    detection_path = Path(run["activity_runs"][0]["record_path"])
    detection = json.loads(detection_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed_unannotated_episode_only_observation"
    assert run["semantic_ground_truth"] is None
    assert run["external_evidence_used"] == []
    assert detection["metric_status"] == "not_scored_without_ground_truth"
    assert detection["candidates"][0]["semantic_role"] == "unresolved_music_like"
    assert detection["candidates"][0]["identity_state"] == "unresolved"
    assert run["outcome"]["kind"] == "unresolved_candidates"
    candidate_asset = detection["candidates"][0]["asset"]
    candidate_path = Path(candidate_asset["path"])
    candidate_info = sf.info(candidate_path)
    assert candidate_asset["representation"] == "noncanonical_staged_candidate"
    assert candidate_asset["publication_state"] == "not_published"
    assert candidate_asset["source_start_frame"] == detection["candidates"][0]["start_frame"]
    assert candidate_asset["source_end_frame"] == detection["candidates"][0]["end_frame"]
    assert candidate_info.frames == (
        detection["candidates"][0]["end_frame"]
        - detection["candidates"][0]["start_frame"]
    )
    assert candidate_info.samplerate == 1000
    assert candidate_info.channels == 2
    assert candidate_path.is_relative_to(output_dir / "candidates")
    assert source.read_bytes() == source_before
    assert not (source_root / "LSS").exists()


def test_episode_only_run_accepts_validated_zero_without_fake_artifacts(tmp_path):
    source = tmp_path / "episode.wav"
    sf.write(source, np.zeros((1000, 1), dtype=np.float32), 1000, subtype="FLOAT")
    source_before = source.read_bytes()
    input_sha256 = __import__("hashlib").sha256(source_before).hexdigest()
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "method": "windowed_rms_hysteresis",
        "window_frames": 100,
        "hop_frames": 50,
        "onset_rms": 0.08,
        "offset_rms": 0.04,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }), encoding="utf-8")
    zero_result = {
        "reason": "no_wanted_score",
        "validation_result": {
            "validator_id": "bgm-zero-test",
            "validator_revision": "r1",
            "hard_gates_passed": True,
        },
    }

    def separate(job, *, destination_dir):
        return {
            "status": "skipped",
            "outputs": [],
            "input_sha256": input_sha256,
            "recipe_id": "separation-test",
            "recipe_revision": {"recipe_id": "separation-test"},
            "runtime_id": "runtime-test",
            "runtime_revision": {"runtime_id": "runtime-test"},
            "zero_result": zero_result,
        }

    output_dir = tmp_path / "run"
    run_path = run_episode_discovery(
        source, [recipe], output_dir,
        source_id="work/episode-zero", model_id="model", separate=separate,
    )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    assert run["status"] == "completed_validated_zero_result"
    assert run["outcome"] == {"kind": "zero", **zero_result}
    assert run["separation"]["artifacts"] == []
    assert run["activity_runs"] == []
    assert not (output_dir / "candidates").exists()
    assert source.read_bytes() == source_before


def test_episode_only_run_materializes_stem_energy_share_candidates(tmp_path):
    source = tmp_path / "episode.wav"
    mixture = np.zeros((1000, 2), dtype=np.float32)
    mixture[200:600] = 0.1
    sf.write(source, mixture, 1000, subtype="FLOAT")
    source_before = source.read_bytes()
    recipe = tmp_path / "stem-energy-share.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "method": "stem_energy_share_hysteresis",
        "onset_share": 0.2,
        "offset_share": 0.1,
        "onset_music_rms": 0.05,
        "offset_music_rms": 0.025,
        "window_frames": 100,
        "hop_frames": 50,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }), encoding="utf-8")

    def separate(job, *, destination_dir):
        destination_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for stem, samples in {
            "Music": mixture,
            "Speech": np.full_like(mixture, 0.03),
            "Effects": np.zeros_like(mixture),
        }.items():
            output = destination_dir / f"episode - {stem}.flac"
            sf.write(output, samples, 1000, subtype="PCM_24")
            tags = FLAC(output)
            tags["LSS_STEM"] = stem
            tags.save()
            outputs.append(str(output))
        return {
            "status": "ok", "outputs": outputs,
            "input_sha256": __import__("hashlib").sha256(source_before).hexdigest(),
            "recipe_id": "separation-test", "recipe_revision": {"recipe_id": "separation-test"},
            "runtime_id": "runtime-test", "runtime_revision": {"runtime_id": "runtime-test"},
            "config_sha256": "c" * 64, "elapsed_seconds": 1.0, "rtf": 1.0,
            "peak_vram_mb": 1.0,
        }

    output_dir = tmp_path / "run"
    run_path = run_episode_discovery(
        source, [recipe], output_dir,
        source_id="work/episode-stem-ratio", model_id="model", separate=separate,
    )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    activity_run = run["activity_runs"][0]
    detection = json.loads(Path(activity_run["record_path"]).read_text(encoding="utf-8"))
    assert activity_run["method"] == "stem_energy_share_hysteresis"
    assert activity_run["analysis_stems"] == ["Music", "Speech", "Effects"]
    assert detection["analysis_signal"]["stem"] == "Music"
    assert [item["stem"] for item in detection["supporting_analysis_signals"]] == [
        "Speech", "Effects",
    ]
    assert detection["candidates"]
    assert Path(detection["candidates"][0]["asset"]["path"]).is_file()
    assert sf.info(detection["candidates"][0]["asset"]["path"]).channels == 2
    assert source.read_bytes() == source_before


def test_episode_only_run_accepts_the_relative_boundary_recipe(tmp_path):
    """区間の終わりを相対量で決めるRecipeも、通常の番組経路から呼べる。"""
    source_root = tmp_path / "user-corpus"
    source_root.mkdir()
    source = source_root / "episode.wav"
    samples = np.zeros((1000, 2), dtype=np.float32)
    samples[200:400] = 0.2
    samples[400:600] = 0.05
    samples[600:800] = 0.2
    sf.write(source, samples, 1000, subtype="FLOAT")
    source_before = source.read_bytes()
    recipe = tmp_path / "relative.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "method": "windowed_rms_relative_hysteresis",
        "window_frames": 100,
        "hop_frames": 50,
        "onset_rms": 0.08,
        "offset_ratio_db": 20.0,
        "offset_floor_rms": 0.001,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }), encoding="utf-8")

    def separate(job, *, destination_dir):
        destination_dir.mkdir(parents=True, exist_ok=True)
        output = destination_dir / "episode - Instrumental.flac"
        sf.write(output, samples, 1000, subtype="PCM_24")
        tags = FLAC(output)
        tags["LSS_STEM"] = "Instrumental"
        tags.save()
        return {
            "status": "ok", "outputs": [str(output)],
            "input_sha256": __import__("hashlib").sha256(source_before).hexdigest(),
            "recipe_id": "separation-test", "recipe_revision": {"recipe_id": "separation-test"},
            "runtime_id": "runtime-test", "runtime_revision": {"runtime_id": "runtime-test"},
            "config_sha256": "c" * 64, "elapsed_seconds": 1.0, "rtf": 1.0,
            "peak_vram_mb": 1.0,
        }

    run_path = run_episode_discovery(
        source, [recipe], tmp_path / "run",
        source_id="work/episode-2", model_id="model", separate=separate,
    )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    detection = json.loads(Path(run["activity_runs"][0]["record_path"]).read_text(encoding="utf-8"))
    assert detection["activity_implementation"]["algorithm"] == "windowed_rms_relative_hysteresis"
    assert detection["activity_recipe"]["recipe_id"].startswith("bgm-activity-relative-rms-v1-")
    # 静かな中間部を挟んでも一つの区間として残る。
    assert len(detection["candidates"]) == 1
    assert detection["candidates"][0]["semantic_role"] == "unresolved_music_like"
    assert source.read_bytes() == source_before
