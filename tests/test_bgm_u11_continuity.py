"""Gap-aware music continuity and the chime-only Music window discount, which only nominates and rechecks rendered PCM (ep1 u11 blockers).

u11 (seg13, 934-974 s) is quiet glockenspiel music. The first-pass tagger dips below Music 0.5 for
1-2 s five times (35/40 raw windows) and the source tagger calls the glockenspiel itself
``Jingle, tinkle`` in three windows (9/40 = 0.225 event windows). The synthetic per-window
sequences below reproduce those observations; no real media is used.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lss.bgm import cue_scope, cue_semantics as cs, rendered_content
from lss.sound import audioset

RECIPE = cs.default_recipe()
LEGACY_CONTINUITY = dataclasses.replace(RECIPE, continuity_raw_floor=1.0)
SOURCE = {"path": "episode.mkv", "sha256": "a" * 64}
EPISODE = 1400
I = audioset.INDEX


def _stem(music: dict[int, float], *, level=-45.0, embedding=None, seconds=EPISODE):
    """First-pass (music stem) tags: ``music`` maps window -> Music probability, others are 0."""
    p = np.zeros((seconds, 527), np.float32)
    for window, value in music.items():
        p[window, cs.MUSIC_CLASS] = value
    vectors = np.array([embedding(t) if embedding else np.ones(8) for t in range(seconds)], np.float32)
    return p, cs.WindowEvidence.from_arrays(np.full(seconds, level), p, vectors)


def _body(first, last, misses=None, value=0.9):
    music = {t: value for t in range(first, last)}
    music.update(misses or {})
    return music


U11_MISSES = {938: .43, 946: .43, 947: .11, 951: .47, 952: .43}


def _u11_stem():
    return _stem(_body(934, 974, U11_MISSES))


def _decide(evidence, candidate, recipe=RECIPE):
    segments = cs.segment([candidate], evidence, recipe)
    return cs.decide(segments, evidence, recipe, source_seconds=float(evidence.count))


def _u11_mixture():
    mixture = np.zeros((EPISODE, 527), np.float32)
    mixture[:, cs.MUSIC_CLASS] = 0.8
    mixture[940:944, I["Meow"]] = 0.6     # real animal calls
    mixture[945, I["Spray"]] = 0.4
    mixture[955, I["Animal"]] = 0.3
    mixture[950:953, I["Jingle, tinkle"]] = 0.5  # the glockenspiel music itself
    return mixture


# --- (a) gap-aware continuity ---------------------------------------------------------------

def test_u11_short_interior_tagger_dips_are_continuous_music():
    _, evidence = _u11_stem()
    [legacy] = _decide(evidence, (930.0, 980.0), LEGACY_CONTINUITY)
    assert (legacy["start_seconds"], legacy["end_seconds"]) == (934.0, 974.0)
    assert legacy["features"]["music_fraction"] == 0.875
    assert "music_not_continuous" in legacy["reasons"]
    [cue] = _decide(evidence, (930.0, 980.0))
    assert cue["features"]["music_fraction"] == 1.0 and cue["features"]["raw_music_fraction"] == 0.875
    assert cue["semantic_class"] == cs.WANTED and cue["reasons"] == []


def test_interior_gap_of_gap_seconds_still_splits_and_a_body_across_it_is_not_continuous():
    # A 3-s gap (== gap_seconds) is a music stop: segment() splits there.
    _, evidence = _stem(_body(100, 140, {t: 0.1 for t in range(118, 121)}))
    cues = _decide(evidence, (95.0, 145.0))
    assert [(c["start_seconds"], c["end_seconds"]) for c in cues] == [(100.0, 118.0), (121.0, 140.0)]
    # A body that still spans it (e.g. joined by another route) does not get the gap counted as music.
    _, evidence = _stem(_body(100, 120, {t: 0.1 for t in range(108, 111)}))
    whole = {"start_seconds": 100.0, "end_seconds": 120.0, "core_seconds": [100.0, 120.0],
             "start_edge": {"kind": cs.EDGE_SILENCE}, "end_edge": {"kind": cs.EDGE_SILENCE}}
    [cue] = cs.decide([whole], evidence, RECIPE, source_seconds=float(EPISODE))
    assert cue["features"]["music_fraction"] == cue["features"]["raw_music_fraction"] == 0.85
    assert "music_not_continuous" in cue["reasons"]


@pytest.mark.parametrize("pattern, raw", [((1, 1), 0.5), ((3, 1), 0.75)])
def test_a_body_with_much_of_its_seconds_in_short_gaps_does_not_pass(pattern, raw):
    on, off = pattern
    misses = {t: 0.1 for t in range(100, 140) if (t - 100) % (on + off) >= on}
    _, evidence = _stem(_body(100, 140, misses))
    [cue] = _decide(evidence, (95.0, 145.0))
    # segment() keeps one body (every gap is short), but its raw fraction is below the floor.
    assert (cue["start_seconds"], cue["end_seconds"]) == (100.0, 139.0)   # window 139 is a miss
    assert cue["features"]["raw_music_fraction"] == pytest.approx(raw, abs=0.03)
    assert cue["features"]["music_fraction"] == cue["features"]["raw_music_fraction"]
    assert "music_not_continuous" in cue["reasons"] and cue["semantic_class"] == cs.UNCERTAIN


def test_gaps_at_the_body_edge_are_not_filled():
    mask = np.array([False, True, True, False, True, False])
    assert cs.fill_short_gaps(mask, 3).tolist() == [False, True, True, True, True, False]
    weights = np.ones(6)
    no_floor = dataclasses.replace(RECIPE, continuity_raw_floor=0.0)
    assert cs.continuity_fraction(mask, weights, no_floor) == (4 / 6, 3 / 6)
    assert cs.continuity_fraction(mask, weights, RECIPE) == (3 / 6, 3 / 6)   # below the raw floor


def test_razor_thin_doubt_distance_is_unchanged_by_gap_aware_continuity():
    angle = np.arccos(1 - 0.1315)   # 1-cos just above doubt_distance 0.13

    def embedding(t):
        a = 0.0 if t < 120 else angle
        return np.array([np.cos(a), np.sin(a), 0, 0, 0, 0, 0, 0.0])

    _, evidence = _stem(_body(100, 140, {105: .3, 130: .3}), embedding=embedding)
    [legacy] = _decide(evidence, (95.0, 145.0), LEGACY_CONTINUITY)
    [cue] = _decide(evidence, (95.0, 145.0))
    assert cue["features"]["inner_split_distance"] == legacy["features"]["inner_split_distance"]
    assert cue["features"]["inner_split_distance"] >= RECIPE.doubt_distance
    assert "possible_change_inside" in cue["reasons"] and "possible_change_inside" in legacy["reasons"]


def test_continuity_floor_is_declared_and_legacy_identity_is_kept():
    assert RECIPE.continuity_raw_floor == 0.8
    assert "continuity_raw_floor" not in LEGACY_CONTINUITY.to_dict()
    assert LEGACY_CONTINUITY.recipe_id != RECIPE.recipe_id
    with pytest.raises(cs.CueSemanticsError):
        dataclasses.replace(RECIPE, continuity_raw_floor=1.5).validate()


# --- (b) chime-only Music window discount: nomination and rendered recheck only -------------

def _scoped(mixture, start=934.0, end=974.0):
    item = {"start_seconds": start, "end_seconds": end, "core_seconds": [start, end],
            "start_edge": {"kind": "music_gap"}, "end_edge": {"kind": "music_gap"},
            "semantic_class": cs.WANTED, "reasons": [], "features": {}}
    cue_scope.decide([item], SOURCE, mixture=mixture)
    return item


def _first_pass_with_events():
    stem, _ = _u11_stem()
    first = stem.copy()
    first[940:944, I["Meow"]] = 0.6
    first[945, I["Spray"]] = 0.4
    first[955, I["Animal"]] = 0.3
    first[950:953, I["Jingle, tinkle"]] = 0.5   # 9/40 first-pass event windows, three chime-only
    return first


def test_chime_classes_are_event_classes_named_in_the_vocabulary():
    names = {audioset.LABELS[index] for index in cue_scope.CHIME_CLASSES}
    assert names == {"Doorbell", "Ding-dong", "Ding", "Jingle, tinkle"}
    assert set(cue_scope.CHIME_CLASSES) <= set(cue_scope.EVENT_CLASSES)


def test_decision_hold_counts_every_chime_window_but_safety_can_report_the_discount():
    stem, _ = _u11_stem()
    mixture = _u11_mixture()
    item = _scoped(mixture)
    assert item["safety"]["event_window_fraction"] == 0.225
    assert "chime_music_windows_not_counted" not in item["safety"]
    assert item["scope"]["disposition"] == "HELD" and item["scope"]["reason"] == "safety_hold"
    music = (mixture[:, cs.MUSIC_CLASS], stem[:, cs.MUSIC_CLASS])
    observed = cue_scope.safety(item, mixture, music=music)
    assert observed["event_window_fraction"] == 0.225 and observed["heavy_nonmusic_event_scene"]
    assert observed["chime_discounted_event_window_fraction"] == 0.15
    assert observed["chime_discounted_heavy_nonmusic_event_scene"] is False
    assert observed["chime_music_windows_not_counted"] == 3


def test_u11_is_held_at_decision_level_nominated_and_published_only_after_the_rendered_check():
    item = _scoped(_u11_mixture())
    assert item["scope"]["disposition"] == "HELD" and item["scope"]["reason"] == "safety_hold"
    rendered_content.nominate([item], SOURCE, _first_pass_with_events(), _u11_mixture())
    assert rendered_content.nominated(item)
    assert item["scope"]["disposition"] == "HELD"            # still held until the rendered PCM passes
    check = item["rendered_content_check"]
    assert check["decision_event_window_fraction"] == 0.225
    assert check["nomination_event_window_fraction"] == 0.15
    assert check["chime_music_windows_not_counted"] == 3
    assert check["first_pass_safety"]["event_window_fraction"] == 0.225


def test_u11_end_to_end_semantics_and_scope_hold_then_nominate():
    stem, evidence = _u11_stem()
    [cue] = _decide(evidence, (930.0, 980.0))
    cue_scope.decide([cue], SOURCE, mixture=_u11_mixture())
    assert cue["scope"]["disposition"] == "HELD" and cue["scope"]["reason"] == "safety_hold"
    rendered_content.nominate([cue], SOURCE, _first_pass_with_events(), _u11_mixture())
    assert rendered_content.nominated(cue)


def test_falsifier_undiscounted_0379_discounted_0182_is_held_and_nominated_not_published():
    # ep149 seg13-like: 55 windows, 21 event windows, 13 of them chime-only Music windows.
    n = 55
    mixture = np.zeros((EPISODE, 527), np.float32)
    mixture[:, cs.MUSIC_CLASS] = 0.8
    mixture[0:8, I["Smash, crash"]] = 0.5
    mixture[8:21, I["Jingle, tinkle"]] = 0.5
    item = _scoped(mixture, 0.0, float(n))
    assert item["safety"]["event_window_fraction"] == round(21 / n, 4) == 0.3818
    assert item["scope"]["disposition"] == "HELD" and item["scope"]["reason"] == "safety_hold"
    first = np.zeros((EPISODE, 527), np.float32)
    first[:, cs.MUSIC_CLASS] = 0.9
    first[0:8, I["Smash, crash"]] = 0.5
    first[8:21, I["Jingle, tinkle"]] = 0.5
    rendered_content.nominate([item], SOURCE, first, mixture)
    assert item["scope"]["disposition"] == "HELD"
    assert rendered_content.nominated(item)
    check = item["rendered_content_check"]
    assert check["decision_event_window_fraction"] == 0.3818
    assert check["nomination_event_window_fraction"] == round(8 / n, 4)
    assert check["chime_music_windows_not_counted"] == 13
    # Without the source tags no discount applies: the first-pass fraction is the undiscounted one.
    plain = _scoped(mixture, 0.0, float(n))
    rendered_content.nominate([plain], SOURCE, first)
    assert not rendered_content.nominated(plain)


def test_a_doorbell_where_the_source_is_not_music_still_counts():
    stem, _ = _u11_stem()
    mixture = _u11_mixture()
    mixture[950:953, cs.MUSIC_CLASS] = 0.3   # the source does not call these windows music
    music = (mixture[:, cs.MUSIC_CLASS], stem[:, cs.MUSIC_CLASS])
    item = _scoped(mixture)
    observed = cue_scope.safety(item, mixture, music=music)
    assert observed["chime_discounted_event_window_fraction"] == 0.225
    assert observed["chime_music_windows_not_counted"] == 0
    mixture = _u11_mixture()
    mixture[950:953, I["Jingle, tinkle"]] = 0.0
    mixture[950:953, I["Doorbell"]] = 0.6
    mixture[950:953, cs.MUSIC_CLASS] = 0.2
    music = (mixture[:, cs.MUSIC_CLASS], stem[:, cs.MUSIC_CLASS])
    assert cue_scope.safety(item, mixture, music=music)["chime_discounted_event_window_fraction"] == 0.225


def test_a_chime_in_a_window_where_the_stem_is_not_music_still_counts():
    stem, _ = _stem(_body(934, 974, {950: .3, 951: .3, 952: .3}))
    mixture = _u11_mixture()
    item = _scoped(mixture)
    music = (mixture[:, cs.MUSIC_CLASS], stem[:, cs.MUSIC_CLASS])
    assert cue_scope.safety(item, mixture, music=music)["chime_discounted_event_window_fraction"] == 0.225


def test_a_chime_co_labelled_with_another_event_class_still_counts():
    stem, _ = _u11_stem()
    mixture = _u11_mixture()
    mixture[950:953, I["Smash, crash"]] = 0.3
    item = _scoped(mixture)
    music = (mixture[:, cs.MUSIC_CLASS], stem[:, cs.MUSIC_CLASS])
    observed = cue_scope.safety(item, mixture, music=music)
    assert observed["chime_discounted_event_window_fraction"] == 0.225
    assert observed["chime_music_windows_not_counted"] == 0


def test_u15_like_battle_effects_stay_held_and_are_not_nominated():
    mixture = np.zeros((EPISODE, 527), np.float32)
    mixture[:, cs.MUSIC_CLASS] = 0.8
    for window in (503, 507, 511, 515, 519):
        mixture[window, I["Smash, crash"]] = 0.5
    for window in (525, 529, 533, 537):
        mixture[window, I["Whoosh, swoosh, swish"]] = 0.4
    for window in (541, 545, 549, 553):
        mixture[window, I["Animal"]] = 0.3
    item = _scoped(mixture, 500.0, 555.0)
    assert item["safety"]["event_window_fraction"] == round(13 / 55, 4)
    assert item["scope"]["disposition"] == "HELD" and item["scope"]["reason"] == "safety_hold"
    first = mixture.copy()
    first[:, cs.MUSIC_CLASS] = 0.9
    rendered_content.nominate([item], SOURCE, first, mixture)
    assert not rendered_content.nominated(item)
