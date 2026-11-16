"""同じ録音の使用箇所を、重なりの少なさで選ぶ。音声は使わない。"""
from __future__ import annotations

import pytest

from lss.bgm import occurrence_quality as quality


def _recipe(minimum=0.5):
    return quality.OccurrenceQualityRecipe(minimum_explained=minimum)


def _occurrence(occurrence_id, recording_key, explained, start=0.0, end=10.0, source_id="ep1"):
    return {"occurrence_id": occurrence_id, "recording_key": recording_key,
            "source_id": source_id, "start_seconds": start, "end_seconds": end,
            "waveform_explained": explained}


def test_recipe_id_is_stable_and_depends_on_declaration():
    first = _recipe(0.5).recipe_id
    assert first == _recipe(0.5).recipe_id
    assert first != _recipe(0.6).recipe_id
    assert first.startswith("bgm-occurrence-quality-v1-")


@pytest.mark.parametrize("minimum", [-0.1, 1.1, "0.5", True, float("nan")])
def test_recipe_rejects_invalid_minimum(minimum):
    with pytest.raises(quality.OccurrenceQualityError):
        quality.OccurrenceQualityRecipe(minimum_explained=minimum).validate()


def test_selects_the_least_overlaid_occurrence():
    result = quality.select_best_occurrences([
        _occurrence("a", "cue-1", 0.261, 0.0, 14.0),
        _occurrence("b", "cue-1", 0.697, 100.0, 106.0),
        _occurrence("c", "cue-1", 0.360, 200.0, 218.0),
    ], _recipe())
    group = result["groups"][0]
    assert group["selected_occurrence_id"] == "b"
    assert group["selected_explained"] == pytest.approx(0.697)
    assert [item["occurrence_id"] for item in group["occurrences"]] == ["b", "c", "a"]
    assert group["explained_spread"] == pytest.approx(0.436)
    assert quality.SELECTED in group["occurrences"][0]["reasons"]


def test_longer_occurrence_wins_only_when_overlap_is_equal():
    result = quality.select_best_occurrences([
        _occurrence("short", "cue-1", 0.8, 0.0, 5.0),
        _occurrence("long", "cue-1", 0.8, 50.0, 90.0),
    ], _recipe())
    assert result["groups"][0]["selected_occurrence_id"] == "long"


def test_theme_recording_is_excluded_and_selects_nothing():
    result = quality.select_best_occurrences([
        _occurrence("op-1", "opening", 0.95),
        _occurrence("cue-a", "cue-1", 0.40),
    ], _recipe(), theme_recording_keys=["opening"])
    themes = [g for g in result["groups"] if g["theme_excluded"]]
    assert len(themes) == 1
    assert themes[0]["selected_occurrence_id"] is None
    assert quality.EXCLUDED_THEME in themes[0]["occurrences"][0]["reasons"]
    assert result["theme_excluded_count"] == 1
    assert result["selected_count"] == 1


def test_heavily_overlaid_is_marked_but_kept():
    result = quality.select_best_occurrences([
        _occurrence("a", "cue-1", 0.30),
        _occurrence("b", "cue-1", 0.20),
    ], _recipe(0.5))
    group = result["groups"][0]
    assert group["occurrence_count"] == 2
    assert group["selected_occurrence_id"] == "a"
    assert quality.HEAVILY_OVERLAID in group["occurrences"][0]["reasons"]
    assert result["selected_above_minimum"] == 0


def test_single_occurrence_is_selected_and_named():
    result = quality.select_best_occurrences([_occurrence("only", "cue-1", 0.9)], _recipe())
    reasons = result["groups"][0]["occurrences"][0]["reasons"]
    assert quality.SELECTED in reasons and quality.SINGLE_OCCURRENCE in reasons
    assert result["groups"][0]["explained_spread"] == pytest.approx(0.0)


def test_groups_are_separated_per_recording():
    result = quality.select_best_occurrences([
        _occurrence("a", "cue-1", 0.4), _occurrence("b", "cue-2", 0.9),
    ], _recipe())
    assert [g["recording_key"] for g in result["groups"]] == ["cue-1", "cue-2"]
    assert result["recording_count"] == 2


def test_duplicate_occurrence_id_is_rejected():
    with pytest.raises(quality.OccurrenceQualityError):
        quality.select_best_occurrences([
            _occurrence("same", "cue-1", 0.4), _occurrence("same", "cue-2", 0.5),
        ], _recipe())


@pytest.mark.parametrize("broken", [
    {"occurrence_id": "", "recording_key": "c", "source_id": "s",
     "start_seconds": 0.0, "end_seconds": 1.0, "waveform_explained": 0.5},
    {"occurrence_id": "a", "recording_key": "c", "source_id": "s",
     "start_seconds": 5.0, "end_seconds": 5.0, "waveform_explained": 0.5},
    {"occurrence_id": "a", "recording_key": "c", "source_id": "s",
     "start_seconds": 0.0, "end_seconds": 1.0, "waveform_explained": 1.4},
    {"occurrence_id": "a", "recording_key": "c", "source_id": "s",
     "start_seconds": 0.0, "end_seconds": 1.0, "waveform_explained": None},
])
def test_broken_occurrence_is_rejected(broken):
    with pytest.raises(quality.OccurrenceQualityError):
        quality.select_best_occurrences([broken], _recipe())


def test_result_carries_recipe_identity_and_boundary():
    result = quality.select_best_occurrences([_occurrence("a", "cue-1", 0.4)], _recipe())
    assert result["recipe_id"] == _recipe().recipe_id
    assert result["recipe"]["method"] == quality.METHOD
    assert any("判断ではない" in line for line in result["claim_boundary"])


def test_mutual_scores_average_explanations_from_other_occurrences():
    pairwise = {"a|b": 0.7, "a|c": 0.5, "b|a": 0.3, "b|c": 0.1, "c|a": 0.2, "c|b": 0.4}
    scores = quality.mutual_explained_scores(pairwise, ["a", "b", "c"])
    assert scores == {"a": pytest.approx(0.6), "b": pytest.approx(0.2), "c": pytest.approx(0.3)}
    assert max(scores, key=scores.get) == "a"


def test_mutual_scores_require_every_ordered_pair():
    with pytest.raises(quality.OccurrenceQualityError):
        quality.mutual_explained_scores({"a|b": 0.5}, ["a", "b"])


@pytest.mark.parametrize("ids", [["only"], ["dup", "dup"]])
def test_mutual_scores_reject_bad_identifier_sets(ids):
    with pytest.raises(quality.OccurrenceQualityError):
        quality.mutual_explained_scores({}, ids)


def test_mutual_scores_reject_out_of_range_values():
    with pytest.raises(quality.OccurrenceQualityError):
        quality.mutual_explained_scores({"a|b": 1.5, "b|a": 0.2}, ["a", "b"])


def test_evidence_source_is_recorded_and_validated():
    released = _occurrence("a", "cue-1", 0.6)
    mutual = {**_occurrence("b", "cue-1", 0.4), "evidence_source": quality.EVIDENCE_MUTUAL}
    result = quality.select_best_occurrences([released, mutual], _recipe())
    group = result["groups"][0]
    assert group["evidence_sources"] == sorted([quality.EVIDENCE_RELEASED, quality.EVIDENCE_MUTUAL])
    assert group["occurrences"][0]["evidence_source"] == quality.EVIDENCE_RELEASED
    with pytest.raises(quality.OccurrenceQualityError):
        quality.select_best_occurrences(
            [{**_occurrence("c", "cue-1", 0.4), "evidence_source": "guess"}], _recipe())
