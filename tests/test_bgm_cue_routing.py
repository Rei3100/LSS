from dataclasses import replace

import numpy as np
import pytest

from lss.bgm.cue_routing import CueRoutingRecipe, route_cues_stereo
from lss.bgm.routing import DisagreementRoutingRecipe


def recipe():
    return CueRoutingRecipe(window_recipe=DisagreementRoutingRecipe(window_frames=16, hop_frames=8))


def stems(music):
    return {"Music": music, "Speech": np.zeros_like(music), "Effects": np.zeros_like(music)}


def test_interrupted_singing_uses_music_for_entire_candidate():
    music = np.random.default_rng(27).normal(0, 0.1, (2, 257))
    parent = music.copy()
    parent[:, 100:130] *= 0.01
    result = route_cues_stereo(parent, stems(music), [(0, 257)], recipe())
    np.testing.assert_array_equal(result.audio, music)
    assert result.cues[0]["use_music"]
    assert 0 < result.cues[0]["positive_window_count"] < result.cues[0]["window_count"]


def test_separate_candidates_do_not_inherit_neighbor_disagreement():
    music = np.ones((2, 257)) * 0.1
    parent = music.copy()
    parent[:, 100:130] *= 0.01
    result = route_cues_stereo(parent, stems(music), [(0, 60), (90, 150), (180, 257)], recipe())
    assert [cue["use_music"] for cue in result.cues] == [False, True, False]
    expected = np.zeros_like(music)
    for start, end in [(0, 60), (90, 150), (180, 257)]:
        expected[:, start:end] = music[:, start:end]
    np.testing.assert_array_equal(result.audio, expected)


def test_zero_candidates_never_reintroduce_parent_residual():
    parent = np.ones((2, 257)) * 0.1
    result = route_cues_stereo(parent, stems(np.zeros_like(parent)), [], recipe())
    assert result.cues == ()
    assert np.count_nonzero(result.audio) == 0


@pytest.mark.parametrize("intervals", [[(0, 20), (10, 30)], [(10, 20), (0, 10)],
                                        [(0, 258)], [(0, 0)], [(-1, 1)], [(True, 20)]])
def test_invalid_candidate_timeline_is_rejected(intervals):
    music = np.ones((2, 257)) * 0.1
    with pytest.raises(ValueError, match="区間"):
        route_cues_stereo(music, stems(music), intervals, recipe())


def test_recipe_and_window_rules_have_stable_identity():
    base = recipe()
    assert base.recipe_id == recipe().recipe_id
    changed = replace(base, window_recipe=replace(base.window_recipe, minimum_disagreement_share=0.2))
    assert changed.recipe_id != base.recipe_id
    with pytest.raises(ValueError):
        replace(base, outside_activity="parent-residual").validate()


def test_context_preserves_tail_and_merges_overlapping_coverage_with_origins():
    music = np.ones((2, 257)) * 0.1
    parent = music * 0.1
    selected = [(10, 50), (65, 100), (240, 250)]
    result = route_cues_stereo(parent, stems(music), selected, replace(recipe(), context_frames=16))
    assert [(c["start_frame"], c["end_frame"]) for c in result.cues] == [(0, 116), (224, 257)]
    assert result.cues[0]["activity_intervals"] == selected[:2]
    np.testing.assert_array_equal(result.audio[:, 0:116], music[:, 0:116])
    assert np.count_nonzero(result.audio[:, 116:224]) == 0
    assert replace(recipe(), context_frames=16).recipe_id != recipe().recipe_id


def test_context_cannot_create_activity_from_empty_candidates():
    music = np.ones((2, 257)) * 0.1
    result = route_cues_stereo(music * 0.1, stems(music), [], replace(recipe(), context_frames=16))
    assert not result.cues and np.count_nonzero(result.audio) == 0


def test_directional_rule_does_not_treat_missing_instruments_as_singing_addition():
    parent = np.random.default_rng(95).normal(0, 0.1, (2, 257))
    music = parent * 0.1
    result = route_cues_stereo(parent, stems(music), [(0, 257)],
                               replace(recipe(), require_energy_increase=True))
    assert result.cues[0]["positive_window_count"] > 0
    assert not result.cues[0]["music_energy_increased"] and not result.cues[0]["use_music"]
    np.testing.assert_array_equal(result.audio, parent)


def test_directional_rule_preserves_additional_component_and_has_distinct_identity():
    music = np.random.default_rng(82).normal(0, 0.1, (2, 257))
    guarded = replace(recipe(), require_energy_increase=True)
    result = route_cues_stereo(music * 0.1, stems(music), [(0, 257)], guarded)
    assert result.cues[0]["music_energy_increased"] and result.cues[0]["use_music"]
    np.testing.assert_array_equal(result.audio, music)
    assert guarded.recipe_id != recipe().recipe_id


def test_cue_analysis_does_not_allocate_the_unused_interpolated_renderer(monkeypatch):
    """長い番組のcue経路は全sampleの係数と混合音声を必要としない。"""
    import lss.bgm.routing as routing

    def unused_renderer(*args, **kwargs):
        pytest.fail("区間分岐では窓間の全sampleの補間を行わない")

    monkeypatch.setattr(routing.np, "interp", unused_renderer)
    music = np.random.default_rng(82).normal(0, 0.1, (2, 257))
    guarded = replace(recipe(), require_energy_increase=True)
    result = route_cues_stereo(music * 0.1, stems(music), [(0, 257)], guarded)
    np.testing.assert_array_equal(result.audio, music)
    assert result.cues[0]["positive_window_count"] == result.cues[0]["window_count"]
