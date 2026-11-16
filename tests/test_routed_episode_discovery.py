import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.harness.fetch import file_sha256
from lss.harness.tsv_io import save_tsv
from lss.validation import routed_episode_discovery as discovery


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def audio_artifact(path):
    info = sf.info(path)
    return {**discovery.artifact(path), "frames": info.frames, "sample_rate": info.samplerate,
            "channels": info.channels}


def cached_episode(tmp_path):
    raw = np.random.default_rng(84).normal(0, 0.1, (1024, 2))
    source = tmp_path / "source.wav"
    sf.write(source, raw, 44100, subtype="DOUBLE")
    source_record = {**discovery.artifact(source), "corpus_role": "development", "source_id": "episode"}
    instrumental = []
    for name, gain in (("Instrumental", 0.1), ("Vocals", 0.9)):
        path = tmp_path / f"{name}.flac"
        sf.write(path, raw * gain, 44100, subtype="PCM_24")
        tags = FLAC(path)
        tags["LSS_OUTPUT_HEADROOM_SHIFT"] = "0"
        tags["LSS_INPUT_SHA256"] = source_record["sha256"]
        tags["LSS_RECIPE_ID"] = "instrumental-recipe"
        tags["LSS_STEM"] = name
        tags.save()
        instrumental.append({**audio_artifact(path), "stem": name})
    music_plan = tmp_path / "music-plan.json"
    write(music_plan, {"source": source_record, "candidate": {"provider": "fixture"}})
    stems = {}
    for role, gain in (("music", 1), ("speech", 0), ("effects", 0)):
        path = tmp_path / f"{role}.wav"
        sf.write(path, raw * gain, 44100, subtype="DOUBLE")
        stems[role] = audio_artifact(path)
    instrumental_run = tmp_path / "instrumental-run.json"
    write(instrumental_run, {"status": "completed_unannotated_episode_only_observation", "source": source_record,
        "separation": {"artifacts": instrumental, "runtime_revision": {"device": "cuda:0"},
                       "runtime_id": "instrumental-runtime", "recipe_id": "instrumental-recipe",
                       "recipe_revision": {"recipe_id": "instrumental-recipe"}}})
    music_run = tmp_path / "music-run.json"
    write(music_run, {"status": "PASSED_REPRESENTATIVE_HARD_SCREEN", "hard_failure_count": 0,
        "source": source_record, "runtime": {"model_device": "cuda:0"}, "outputs": stems,
        "prepared_input": audio_artifact(source), "plan": discovery.artifact(music_plan),
        "runner": {"sha256": "a" * 64}})
    activity = tmp_path / "activity.json"
    write(activity, {"schema_version": 1, "method": "stem_energy_share_hysteresis", "onset_share": 0.2,
        "offset_share": 0.1, "onset_music_rms": 0.001, "offset_music_rms": 0.0005,
        "window_frames": 32, "hop_frames": 16, "bridge_gap_frames": 0,
        "minimum_active_frames": 16, "padding_frames": 0})
    return instrumental_run, music_run, activity


def test_cache_identity_mismatch_is_rejected_before_creating_run(tmp_path):
    inputs = cached_episode(tmp_path)
    with open(tmp_path / "music.wav", "ab") as stream:
        stream.write(b"changed cache")
    output = tmp_path / "discovery"
    with pytest.raises(ValueError, match="内容"):
        discovery.freeze(*inputs, output, source_id="episode")
    assert not output.exists()


def test_equal_frames_do_not_hide_sample_offset(tmp_path):
    inputs = cached_episode(tmp_path)
    raw, _ = sf.read(tmp_path / "source.wav", always_2d=True)
    record = json.loads(inputs[0].read_text())
    for item, gain in zip(record["separation"]["artifacts"], [0.1, 0.9]):
        path = Path(item["path"])
        sf.write(path, np.roll(raw, 12, axis=0) * gain, 44100, subtype="PCM_24")
        tags = FLAC(path)
        tags["LSS_OUTPUT_HEADROOM_SHIFT"] = "0"
        tags.save()
        item["sha256"] = file_sha256(path)
    write(inputs[0], record)
    with pytest.raises(ValueError, match="時間ずれ"):
        discovery.freeze(*inputs, tmp_path / "discovery", source_id="episode")


def test_interrupted_candidate_save_resumes_without_repeating_renderer(tmp_path, monkeypatch):
    inputs = cached_episode(tmp_path)
    plan = discovery.freeze(*inputs, tmp_path / "discovery", source_id="episode")

    def interrupt(saved):
        raise RuntimeError("候補保存後に中断")

    with pytest.raises(RuntimeError, match="中断"):
        discovery.run(plan, on_candidate=interrupt)
    assert (plan.parent / "rendering.json").exists()
    assert not (plan.parent / "run.json").exists()
    monkeypatch.setattr(discovery, "route_cues_stereo", lambda *a: pytest.fail("保存済みrendererを再利用"))
    report = json.loads(discovery.run(plan, resume=True).read_text())
    assert report["status"] == "completed_development_routed_episode_candidates"
    assert report["publication_state"] == "not_published" and report["promotion_allowed"] is False
    assert report["source_hashes_after_verified"]
    candidate = report["candidates"][0]
    assert candidate["asset"]["candidate_asset_id"].startswith("bgm-candidate-v3-")
    tags = FLAC(candidate["asset"]["path"])
    metadata = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    assert metadata["origin"]["source_sha256"] == json.loads(plan.read_text())["source"]["sha256"]
    assert metadata["semantic_role"] == "unresolved_music_like"
    before = Path(candidate["asset"]["path"]).read_bytes()
    discovery.run(plan, resume=True)
    assert Path(candidate["asset"]["path"]).read_bytes() == before
    Path(candidate["asset"]["path"]).unlink()
    discovery.run(plan, resume=True)
    assert Path(candidate["asset"]["path"]).read_bytes() == before


def test_source_change_stops_resume_and_protects_saved_candidate(tmp_path):
    inputs = cached_episode(tmp_path)
    plan = discovery.freeze(*inputs, tmp_path / "discovery", source_id="episode")
    report = json.loads(discovery.run(plan).read_text())
    candidate = Path(report["candidates"][0]["asset"]["path"])
    before = candidate.read_bytes()
    with open(tmp_path / "source.wav", "ab") as stream:
        stream.write(b"changed source")
    with pytest.raises(ValueError, match="内容"):
        discovery.run(plan, resume=True)
    assert candidate.read_bytes() == before


def current_queue(tmp_path, instrumental_run):
    observed = json.loads(instrumental_run.read_text())
    result = {**observed["separation"], "input_sha256": observed["source"]["sha256"],
              "outputs": [observed["separation"]["artifacts"][0]["path"]]}
    result.pop("artifacts")
    path = tmp_path / "queue.json"
    write(path, {"version": 1, "items": [{"id": "current-item", "status": "completed", "result": result}]})
    state = tmp_path / "_state"
    state.mkdir()
    outputs = [{"path": result["outputs"][0], "sha256": file_sha256(Path(result["outputs"][0]))}]
    save_tsv(state / "completed_recipes.tsv", ["input_sha256", "recipe_id", "outputs"], [{
        "input_sha256": observed["source"]["sha256"], "recipe_id": "instrumental-recipe",
        "outputs": json.dumps(outputs),
    }])
    return path


def test_current_queue_cache_is_checked_against_completion_hash(tmp_path):
    instrumental_run, music_run, activity = cached_episode(tmp_path)
    queue = current_queue(tmp_path, instrumental_run)
    plan = discovery.freeze(queue, music_run, activity, tmp_path / "discovery", source_id="episode")
    payload = json.loads(plan.read_text())
    assert payload["cache_origin"] == {"kind": "completed_queue_item", "item_id": "current-item"}
    assert payload["alignment_status"] == "unverified_without_complementary_cache"
    assert any(item["path"].endswith("completed_recipes.tsv") for item in payload["records"])
    assert discovery.run(plan).exists()


def test_finite_tampering_of_queue_output_is_not_accepted_as_new_hash(tmp_path):
    instrumental_run, music_run, activity = cached_episode(tmp_path)
    queue = current_queue(tmp_path, instrumental_run)
    path = tmp_path / "Instrumental.flac"
    tags = dict(FLAC(path))
    sf.write(path, np.zeros((1024, 2)), 44100, subtype="PCM_24")
    saved = FLAC(path)
    saved.update(tags)
    saved.save()
    with pytest.raises(ValueError, match="内容"):
        discovery.freeze(queue, music_run, activity, tmp_path / "discovery", source_id="episode")
    assert not (tmp_path / "discovery").exists()


@pytest.mark.parametrize("mutation", ["interval", "decision", "missing_receipt", "outside_signal"])
def test_changed_rendering_metadata_stops_resume_without_altering_saved_candidate(tmp_path, mutation):
    inputs = cached_episode(tmp_path)
    plan = discovery.freeze(*inputs, tmp_path / "discovery", source_id="episode")

    def interrupt(saved):
        raise RuntimeError("候補保存後に中断")

    with pytest.raises(RuntimeError):
        discovery.run(plan, on_candidate=interrupt)
    candidates = list((plan.parent / "candidates").glob("*.flac"))
    before = {path: path.read_bytes() for path in candidates}
    rendering_path = plan.parent / "rendering.json"
    rendered = json.loads(rendering_path.read_text())
    if mutation == "interval":
        rendered["cues"][0]["start_frame"] += 1
    elif mutation == "decision":
        rendered["cues"][0]["use_music"] = not rendered["cues"][0]["use_music"]
    elif mutation == "missing_receipt":
        rendered.pop("payload_sha256")
    else:
        rendered["signal"]["path"] = str(tmp_path / "music.wav")
        rendered["signal"]["sha256"] = file_sha256(tmp_path / "music.wav")
        rendered["payload_sha256"] = discovery.rendering_digest(rendered)
    write(rendering_path, rendered)
    with pytest.raises(ValueError, match="rendering"):
        discovery.run(plan, resume=True)
    assert {path: path.read_bytes() for path in (plan.parent / "candidates").glob("*.flac")} == before
    assert not (plan.parent / "run.json").exists()


def test_rendering_change_during_candidate_save_cannot_create_completion(tmp_path):
    plan = discovery.freeze(*cached_episode(tmp_path), tmp_path / "discovery", source_id="episode")

    def modify(saved):
        with (plan.parent / "rendering.json").open("a") as stream:
            stream.write(" ")

    with pytest.raises(ValueError, match="保存中に区間"):
        discovery.run(plan, on_candidate=modify)
    assert not (plan.parent / "run.json").exists()


def test_completed_rendering_cannot_be_replaced_by_a_different_self_consistent_record(tmp_path):
    plan = discovery.freeze(*cached_episode(tmp_path), tmp_path / "discovery", source_id="episode")
    report = discovery.run(plan)
    before = report.read_bytes()
    rendering_path = plan.parent / "rendering.json"
    rendered = json.loads(rendering_path.read_text())
    rendered["cues"][0]["use_music"] = not rendered["cues"][0]["use_music"]
    rendered["payload_sha256"] = discovery.rendering_digest(rendered)
    write(rendering_path, rendered)
    with pytest.raises(ValueError, match="完了済み"):
        discovery.run(plan, resume=True)
    assert report.read_bytes() == before
