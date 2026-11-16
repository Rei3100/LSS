import json
from pathlib import Path

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss.harness.fetch import file_sha256
from lss.validation import held_bgm_alternatives as held
from lss.validation import routed_episode_discovery as routed
from tests.test_routed_episode_discovery import cached_episode, write


def upstream(tmp_path, *, ambiguous=True):
    instrumental_run, music_run, activity = cached_episode(tmp_path)
    if ambiguous:
        raw, rate = sf.read(tmp_path / "source.wav", always_2d=True)
        music_path = tmp_path / "music.wav"
        sf.write(music_path, raw * 0.05, rate, subtype="DOUBLE")
        music = json.loads(music_run.read_text())
        music["outputs"]["music"]["sha256"] = file_sha256(music_path)
        write(music_run, music)
    plan = routed.freeze(instrumental_run, music_run, activity, tmp_path / "upstream", source_id="episode")
    return routed.run(plan)


def test_held_music_keeps_shared_occurrence_and_primary_bytes(tmp_path):
    run_path = upstream(tmp_path)
    prior = json.loads(run_path.read_text())
    primary_path = Path(prior["candidates"][0]["asset"]["path"])
    before = run_path.read_bytes(), primary_path.read_bytes()
    plan = held.freeze(run_path, tmp_path / "held")
    report = json.loads(held.run(plan).read_text())
    assert report["outcome"] == "held_alternatives" and report["original_candidates_unchanged"]
    assert report["publication_state"] == "not_published" and report["promotion_allowed"] is False
    candidate = report["candidates"][0]
    assert candidate["candidate_role"] == "held_alternative" and candidate["selection_state"] == "unresolved"
    assert candidate["primary_candidate_asset_id"] == prior["candidates"][0]["asset"]["candidate_asset_id"]
    assert candidate["origin"]["start_frame"] == prior["candidates"][0]["origin"]["start_frame"]
    assert candidate["origin"]["end_frame"] == prior["candidates"][0]["origin"]["end_frame"]
    tags = FLAC(candidate["asset"]["path"])
    metadata = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    assert metadata["evidence"]["hold_reason"] == held.RULE
    assert metadata["evidence"]["primary_candidate_asset_id"] == candidate["primary_candidate_asset_id"]
    assert metadata["semantic_role"] == "unresolved_music_like"
    source, _ = sf.read(tmp_path / "music.wav", always_2d=True)
    saved, _ = sf.read(candidate["asset"]["path"], always_2d=True)
    np.testing.assert_allclose(source, saved, atol=2 ** -23, rtol=0)
    assert (run_path.read_bytes(), primary_path.read_bytes()) == before
    alternate_before = Path(candidate["asset"]["path"]).read_bytes()
    held.run(plan)
    assert Path(candidate["asset"]["path"]).read_bytes() == alternate_before
    assert (run_path.read_bytes(), primary_path.read_bytes()) == before


def test_interrupted_held_save_can_resume_and_has_no_early_final_report(tmp_path):
    plan = held.freeze(upstream(tmp_path), tmp_path / "held")

    def stop(saved):
        raise InterruptedError("保留候補保存後に中断")

    with pytest.raises(InterruptedError):
        held.run(plan, on_candidate=stop)
    assert not (plan.parent / "alternatives.json").exists()
    assert len(list((plan.parent / "candidates").glob("*.flac"))) == 1
    assert held.run(plan).exists()


def test_no_eligible_alternatives_is_not_semantic_target_absence(tmp_path):
    plan = held.freeze(upstream(tmp_path, ambiguous=False), tmp_path / "held")
    report = json.loads(held.run(plan).read_text())
    assert report["outcome"] == "no_eligible_ambiguous_candidates"
    assert report["candidates"] == [] and report["selection_state"] == "unresolved"
    assert report["semantic_ground_truth"] is None


def test_tampered_primary_candidate_cannot_become_a_new_reference(tmp_path):
    run_path = upstream(tmp_path)
    prior = json.loads(run_path.read_text())
    candidate = Path(prior["candidates"][0]["asset"]["path"])
    tags = dict(FLAC(candidate))
    sf.write(candidate, np.zeros((1024, 2)), 44100, subtype="PCM_24")
    saved = FLAC(candidate)
    saved.update(tags)
    saved.save()
    with pytest.raises(ValueError, match="内容"):
        held.freeze(run_path, tmp_path / "held")
    assert not (tmp_path / "held").exists()


def test_music_cache_change_stops_held_save_before_creating_candidates(tmp_path):
    plan = held.freeze(upstream(tmp_path), tmp_path / "held")
    with open(tmp_path / "music.wav", "ab") as stream:
        stream.write(b"changed cache")
    with pytest.raises(ValueError, match="内容"):
        held.run(plan)
    assert not (plan.parent / "candidates").exists()
