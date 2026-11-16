"""保留候補を曲単位へ分け直し、公開してよい区間だけを決める判定（`lss.bgm.cue_semantics`）。"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lss.bgm import cue_semantics as cs

RECIPE = cs.default_recipe()


def _evidence(seconds: int, *, music=(), singing=(), embedding=None, level=-20.0):
    """窓ごとの観測。``music``/``singing`` は [開始, 終了) の秒の並び。埋め込みは ``embedding(t)``。"""
    probabilities = np.zeros((seconds, 527), dtype=np.float32)
    for start, end in music:
        probabilities[start:end, cs.MUSIC_CLASS] = 0.9
    for start, end in singing:
        probabilities[start:end, cs.SINGING_CLASSES[0]] = 0.6
    vectors = np.array([embedding(t) if embedding else np.ones(8) for t in range(seconds)], dtype=np.float32)
    return cs.WindowEvidence.from_arrays(np.full(seconds, level), probabilities, vectors)


def _decide(candidates, evidence, **kwargs):
    recipe = dataclasses.replace(RECIPE, **kwargs) if kwargs else RECIPE
    return cs.decide(cs.segment(candidates, evidence, recipe), evidence, recipe,
                     source_seconds=float(evidence.count))


def test_music_bounded_by_music_gaps_is_published():
    evidence = _evidence(200, music=[(40, 100)])
    [cue] = _decide([(30.0, 110.0)], evidence)
    assert cue["semantic_class"] == cs.WANTED and cue["reasons"] == []
    assert (cue["start_seconds"], cue["end_seconds"]) == (40.0, 100.0)
    assert cue["start_edge"]["kind"] == cue["end_edge"]["kind"] == cs.EDGE_SILENCE


def test_non_music_windows_split_a_candidate_and_are_not_published():
    # 候補の中で音楽に聞こえない所が続けば切る。聞こえない窓は区間へ入れない。
    evidence = _evidence(300, music=[(20, 60), (70, 120)])
    cues = _decide([(10.0, 130.0)], evidence)
    assert [(c["start_seconds"], c["end_seconds"]) for c in cues] == [(20.0, 60.0), (70.0, 120.0)]
    assert all(c["semantic_class"] == cs.WANTED for c in cues)


def test_contiguous_candidates_are_joined_before_deciding():
    # 候補の境目は曲の切れ目とは限らない。隙間なく続く候補は一続きとして判定する。
    evidence = _evidence(200, music=[(20, 90)])
    [cue] = _decide([(10.0, 50.0), (50.0, 100.0)], evidence)
    assert (cue["start_seconds"], cue["end_seconds"]) == (20.0, 90.0)
    assert cue["candidates"] == [[10.0, 50.0], [50.0, 100.0]]


def test_strong_change_splits_into_two_cues_and_both_edges_are_trusted():
    def embedding(t):
        return np.array([1, 0, 0, 0, 0, 0, 0, 0.0]) if t < 60 else np.array([0, 1, 0, 0, 0, 0, 0, 0.0])

    cues = _decide([(10.0, 110.0)], _evidence(200, music=[(20, 100)], embedding=embedding))
    assert [(c["start_seconds"], c["end_seconds"]) for c in cues] == [(20.0, 60.0), (60.0, 100.0)]
    assert cues[0]["end_edge"]["kind"] == cs.EDGE_CHANGE and cues[1]["start_edge"]["kind"] == cs.EDGE_CHANGE
    assert [c["semantic_class"] for c in cues] == [cs.WANTED, cs.WANTED]


def test_moderate_change_inside_is_held_as_possible_merge():
    # 分けるほど強くないが一曲とも言えない変化は、公開せずHELDに残す。
    def embedding(t):
        angle = 0.0 if t < 60 else 0.58  # 1-cos = 0.164（分ける0.19未満、疑い0.15以上）
        return np.array([np.cos(angle), np.sin(angle), 0, 0, 0, 0, 0, 0.0])

    [cue] = _decide([(10.0, 110.0)], _evidence(200, music=[(20, 100)], embedding=embedding))
    assert cue["semantic_class"] == cs.UNCERTAIN and "possible_change_inside" in cue["reasons"]


def test_short_wanted_singing_is_published_and_recorded_as_a_fact():
    [cue] = _decide([(10.0, 110.0)], _evidence(200, music=[(20, 100)], singing=[(50, 55)]))
    assert cue["semantic_class"] == cs.WANTED and "singing_present" not in cue["reasons"]
    assert cue["features"]["singing_present"] is True


def test_wanted_choir_observation_does_not_become_an_absence_or_hold():
    evidence = _evidence(200, music=[(20, 100)])
    probabilities = np.zeros((200, 527), dtype=np.float32)
    probabilities[:, cs.MUSIC_CLASS] = evidence.music_probability
    probabilities[50:55, 28] = 0.7  # Choir
    evidence = cs.WindowEvidence.from_arrays(evidence.music_db, probabilities, evidence.embeddings)
    [cue] = _decide([(10.0, 110.0)], evidence)
    assert cue["semantic_class"] == cs.WANTED and cue["features"]["singing_present"] is True


def test_episode_edge_theme_shaped_singing_is_wanted_music():
    evidence = _evidence(1500, music=[(100, 190)], singing=[(100, 190)])
    [cue] = _decide([(90.0, 200.0)], evidence)
    assert cue["semantic_class"] == cs.WANTED
    assert cue["features"]["theme_shape"] is True


def test_legacy_recipe_keeps_historical_vocal_exclusion_and_identity():
    legacy = dataclasses.replace(RECIPE, vocal_policy="legacy_exclusion_v1")
    assert "vocal_policy" not in legacy.to_dict()
    assert legacy.to_dict() == {key: value for key, value in dataclasses.asdict(legacy).items()
                                if key not in {"edge_extension_seconds", "edge_extension_floor_db", "vocal_policy"}}
    evidence = _evidence(200, music=[(20, 100)], singing=[(50, 55)])
    segments = cs.segment([(10.0, 110.0)], evidence, legacy)
    [cue] = cs.decide(segments, evidence, legacy, source_seconds=200.0)
    assert cue["semantic_class"] == cs.UNCERTAIN and "singing_present" in cue["reasons"]


def test_partial_known_music_and_split_doubt_do_not_invent_singing_absence():
    evidence = _evidence(200, music=[(20, 100)])
    legacy = dataclasses.replace(RECIPE, vocal_policy="legacy_exclusion_v1")
    segments = cs.segment([(10.0, 110.0)], evidence, legacy)
    [cue] = cs.decide(segments, evidence, legacy, source_seconds=200.0, known_spans=[(20.0, 25.0)])
    assert cue["features"]["singing_present"] is False
    assert "singing_present" not in cue["reasons"]
    assert "partly_registered_reference" in cue["reasons"]


def test_long_run_and_short_run_are_not_published():
    evidence = _evidence(600, music=[(20, 25), (100, 300)])
    cues = _decide([(10.0, 30.0), (90.0, 310.0)], evidence)
    assert [c["semantic_class"] for c in cues] == [cs.UNCERTAIN, cs.UNCERTAIN]
    assert "shorter_than_wanted_minimum" in cues[0]["reasons"]
    assert "longer_than_one_cue" in cues[1]["reasons"]


def test_music_continuing_outside_the_candidate_is_not_a_trusted_edge():
    # 曲の頭が候補の外で鳴っているなら、その候補の端は曲の端ではない。
    evidence = _evidence(200, music=[(20, 100)])
    [cue] = _decide([(40.0, 100.0)], evidence)
    assert cue["semantic_class"] == cs.UNCERTAIN
    assert "start_edge_music_continues_outside_candidate" in cue["reasons"]


def test_non_music_candidate_is_non_bgm():
    evidence = _evidence(200, music=[(20, 22)])
    cues = _decide([(10.0, 60.0)], evidence)
    assert all(c["semantic_class"] != cs.WANTED for c in cues)


def test_registered_reference_match_is_known_released_and_absence_of_match_is_not_evidence():
    evidence = _evidence(200, music=[(20, 100)])
    recipe = RECIPE
    segments = cs.segment([(10.0, 110.0)], evidence, recipe)
    [known] = cs.decide(segments, evidence, recipe, source_seconds=200.0, known_spans=[(20.0, 95.0)])
    assert known["semantic_class"] == cs.KNOWN_RELEASED
    [unknown] = cs.decide(segments, evidence, recipe, source_seconds=200.0, known_spans=[])
    assert unknown["semantic_class"] == cs.WANTED


def test_recipe_identity_changes_with_every_declared_value():
    changed = dataclasses.replace(RECIPE, split_distance=0.2)
    assert changed.recipe_id != RECIPE.recipe_id
    with pytest.raises(cs.CueSemanticsError):
        dataclasses.replace(RECIPE, doubt_distance=0.5).validate()


def test_window_evidence_rejects_non_finite_values():
    probabilities = np.zeros((10, 527), dtype=np.float32)
    probabilities[3, cs.MUSIC_CLASS] = np.nan
    with pytest.raises(cs.CueSemanticsError):
        cs.WindowEvidence.from_arrays(np.zeros(10), probabilities, np.ones((10, 4)))


def test_edge_extension_keeps_a_quiet_fade_that_the_tagger_no_longer_calls_music():
    # 90秒から先はタグが音楽と言わないが、SFCの音楽の出力は下限以上で4秒続く（フェードの尻尾）。
    evidence = _evidence(200, music=[(40, 90)])
    levels = np.full(200, -80.0)
    levels[38:94] = -30.0
    evidence = cs.WindowEvidence(music_db=levels, music_probability=evidence.music_probability,
                                 singing_probability=evidence.singing_probability, embeddings=evidence.embeddings)
    [plain] = _decide([(30.0, 110.0)], evidence)
    [extended] = _decide([(30.0, 110.0)], evidence, edge_extension_seconds=6)
    assert (plain["start_seconds"], plain["end_seconds"]) == (40.0, 90.0)
    assert (extended["start_seconds"], extended["end_seconds"]) == (38.0, 94.0)
    assert extended["end_edge"]["extended_seconds"] == 4 and extended["semantic_class"] == cs.WANTED
    assert dataclasses.replace(RECIPE, edge_extension_seconds=6).recipe_id != RECIPE.recipe_id
