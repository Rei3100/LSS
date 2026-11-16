"""シリーズの再演からBGM候補を保留するDevelopment入口。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.application import (ProcessingRequest, ResultDisposition, UserIntent,
                            run_bgm_series_development)
from lss.bgm import content_guard, series_cue_run, series_cues
from tests.test_bgm_identity import RATE, _dialogue_and_noise, _music


def _episode(path, placements, *, seed, seconds=200.0):
    mix = _dialogue_and_noise(seconds, seed=seed) * 0.3
    for start, recording in placements:
        mix[int(start * RATE):int(start * RATE) + recording.size] += recording
    sf.write(path, np.stack([mix, mix], axis=1), RATE, subtype="FLOAT")
    return path


@pytest.fixture
def episodes(tmp_path):
    cue = _music(41, 20.0)
    return [_episode(tmp_path / "ep1.wav", [(30.0, cue)], seed=1),
            _episode(tmp_path / "ep2.wav", [(140.0, cue)], seed=2)]


def _request(paths):
    return ProcessingRequest.from_paths(paths, UserIntent.EXTRACT_BGM_CUES)


def test_a_recurring_recording_is_held_without_publishing(episodes, tmp_path):
    outcome = run_bgm_series_development(
        _request(episodes), tmp_path / "run", rank_by_mutual_explained=False)

    assert outcome.disposition is ResultDisposition.HELD
    assert len(outcome.held_artifacts) == 1
    assert outcome.held_artifacts[0].is_file()
    assert outcome.recipe_id == series_cues.DEFAULT_RECIPE.recipe_id
    assert outcome.source_locations == tuple(Path(path).resolve() for path in episodes)
    record = series_cue_run.read_result(outcome.record_path)
    assert record["publication_state"] == series_cue_run.PUBLICATION_STATE
    assert record["promotion_allowed"] is False
    assert record["holdout_consumed"] is False


def test_the_default_recipe_is_used_when_none_is_given(episodes, tmp_path):
    seen = {}

    def runner(episodes_map, recipe, out, **kwargs):
        seen["recipe"] = recipe
        seen["labels"] = sorted(episodes_map)
        return series_cue_run.run_series_cues(episodes_map, recipe, out, **kwargs)

    run_bgm_series_development(_request(episodes), tmp_path / "run", runner=runner,
                               rank_by_mutual_explained=False)
    assert seen["recipe"] is series_cues.DEFAULT_RECIPE
    assert seen["labels"] == ["ep1", "ep2"]


def test_labels_can_be_given_for_the_record(episodes, tmp_path):
    outcome = run_bgm_series_development(
        _request(episodes), tmp_path / "run", episode_labels=["第1話", "第8話"],
        rank_by_mutual_explained=False)
    record = series_cue_run.read_result(outcome.record_path)
    assert set(record["episodes"]) == {"第1話", "第8話"}


def test_a_guard_rejection_leaves_zero_but_keeps_the_provenance(episodes, tmp_path):
    first = run_bgm_series_development(
        _request(episodes), tmp_path / "plan", rank_by_mutual_explained=False)
    key = series_cue_run.read_result(first.record_path)["candidates"][0]["recording_key"]

    outcome = run_bgm_series_development(
        _request(episodes), tmp_path / "guarded", rank_by_mutual_explained=False,
        measurements=[{"recording_key": key, "speech_db": -2.0, "music_db": -5.2,
                       "effects_db": -16.9}])

    assert outcome.disposition is ResultDisposition.ZERO
    assert outcome.held_artifacts == ()
    assert outcome.zero_result["guard_rejected_count"] == 1
    assert outcome.zero_result["guard_state"] == series_cues.GUARD_APPLIED
    record = series_cue_run.read_result(outcome.record_path)
    assert record["candidates"][0]["guard_reason"] == content_guard.REJECT_SPEECH
    assert Path(record["candidates"][0]["artifact"]["path"]).is_file()


def test_no_recurrence_is_a_zero_result_not_a_failure(tmp_path):
    episodes = [_episode(tmp_path / "ep1.wav", [(30.0, _music(41, 20.0))], seed=1),
                _episode(tmp_path / "ep2.wav", [(140.0, _music(77, 20.0))], seed=2)]
    outcome = run_bgm_series_development(
        _request(episodes), tmp_path / "run", rank_by_mutual_explained=False)
    assert outcome.disposition is ResultDisposition.ZERO
    assert outcome.zero_result["span_count"] == 0
    assert any("欲しい劇伴かどうかは判定しない" in line
               for line in outcome.zero_result["claim_boundary"])


def _cross_episode_only():
    base = series_cues.DEFAULT_RECIPE
    return series_cues.SeriesCueRecipe(
        reuse=base.reuse, quality=base.quality, selection=base.selection, guard=base.guard,
        guard_target=base.guard_target, include_within_episode=False)


def test_one_episode_is_refused_when_only_cross_episode_recurrence_is_declared(tmp_path):
    episode = _episode(tmp_path / "ep1.wav", [(30.0, _music(41, 20.0))], seed=1)
    outcome = run_bgm_series_development(
        _request([episode]), tmp_path / "run", recipe=_cross_episode_only())
    assert outcome.disposition is ResultDisposition.FAILED
    assert outcome.error_code == "series_bgm_requires_multiple_sources"
    assert not (tmp_path / "run").exists()


def test_one_episode_is_accepted_by_default_because_repeats_inside_it_count(tmp_path):
    cue = _music(41, 20.0)
    episode = _episode(tmp_path / "ep1.wav", [(30.0, cue), (150.0, cue)],
                       seed=1, seconds=260.0)
    outcome = run_bgm_series_development(
        _request([episode]), tmp_path / "run", rank_by_mutual_explained=False)
    assert outcome.disposition is ResultDisposition.HELD
    record = series_cue_run.read_result(outcome.record_path)
    assert record["include_within_episode"] is True
    assert record["candidates"][0]["episode_count"] == 1


def test_another_intent_is_refused(episodes, tmp_path):
    outcome = run_bgm_series_development(
        ProcessingRequest.from_paths(episodes, UserIntent.EXTRACT_ACCOMPANIMENT),
        tmp_path / "run")
    assert outcome.disposition is ResultDisposition.FAILED
    assert outcome.error_code == "intent_mismatch"


def test_duplicate_labels_are_refused(episodes, tmp_path):
    outcome = run_bgm_series_development(
        _request(episodes), tmp_path / "run", episode_labels=["ep", "ep"])
    assert outcome.disposition is ResultDisposition.FAILED
    assert outcome.error_code == "invalid_bgm_series_development_result"


def test_a_broken_record_is_reported_as_a_failure(episodes, tmp_path):
    def runner(episodes_map, recipe, out, **kwargs):
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        record = path / series_cue_run.RECORD_NAME
        record.write_text(json.dumps({"type": "other"}), encoding="utf-8")
        return record

    outcome = run_bgm_series_development(_request(episodes), tmp_path / "run", runner=runner)
    assert outcome.disposition is ResultDisposition.FAILED
    assert outcome.error_code == "invalid_bgm_series_development_result"
