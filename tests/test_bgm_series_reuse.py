"""シリーズ内の再演を録音単位のまとまりへ直す。音声は使わない。"""
from __future__ import annotations

import pytest

from lss.bgm import series_reuse as reuse


def _recipe(identical=0.95, released=0.5, occurrence=0.5):
    return reuse.SeriesReuseRecipe(identical_explained=identical,
                                   released_overlap_share=released,
                                   occurrence_overlap_share=occurrence)


def _pair(first, first_start, first_end, second, second_start, second_end, explained=0.4):
    return {"first": first, "first_start_seconds": first_start, "first_end_seconds": first_end,
            "second": second, "second_start_seconds": second_start, "second_end_seconds": second_end,
            "waveform_explained": explained}


def test_recipe_id_depends_on_the_declaration():
    assert _recipe().recipe_id == _recipe().recipe_id
    assert _recipe().recipe_id != _recipe(identical=0.9).recipe_id
    assert _recipe().recipe_id.startswith("bgm-series-reuse-v1-")


@pytest.mark.parametrize("kwargs", [{"identical": 0.0}, {"identical": 1.5},
                                    {"released": -0.1}, {"occurrence": 0}])
def test_recipe_rejects_invalid_shares(kwargs):
    with pytest.raises(reuse.SeriesReuseError):
        _recipe(**kwargs).validate()


def test_same_recording_across_three_episodes_becomes_one_group():
    result = reuse.group_recurrence([
        _pair("ep1", 100.0, 130.0, "ep8", 700.0, 730.0),
        _pair("ep1", 100.0, 130.0, "ep9", 950.0, 980.0),
        _pair("ep8", 700.0, 730.0, "ep9", 950.0, 980.0),
    ], _recipe())
    assert result["group_count"] == 1
    group = result["groups"][0]
    assert group["occurrence_count"] == 3 and group["episode_count"] == 3
    assert [item["source_id"] for item in group["occurrences"]] == ["ep1", "ep8", "ep9"]


def test_overlapping_spans_in_one_episode_are_the_same_occurrence():
    result = reuse.group_recurrence([
        _pair("ep1", 100.0, 130.0, "ep8", 700.0, 730.0),
        _pair("ep1", 102.0, 132.0, "ep9", 950.0, 980.0),
    ], _recipe())
    assert result["group_count"] == 1
    assert result["groups"][0]["episode_count"] == 3


def test_distant_spans_in_one_episode_stay_separate_groups():
    result = reuse.group_recurrence([
        _pair("ep1", 100.0, 130.0, "ep8", 700.0, 730.0),
        _pair("ep1", 900.0, 930.0, "ep9", 400.0, 430.0),
    ], _recipe())
    assert result["group_count"] == 2


def test_identical_reuse_is_separated_from_other_context():
    result = reuse.group_recurrence([
        _pair("ep1", 0.2, 14.2, "ep8", 0.2, 14.2, explained=0.999),
        _pair("ep1", 700.0, 722.0, "ep8", 870.0, 892.0, explained=0.392),
    ], _recipe())
    classes = {g["reuse_class"] for g in result["groups"]}
    assert classes == {reuse.REUSE_IDENTICAL, reuse.REUSE_OTHER_CONTEXT}
    assert result["unreleased_identical_reuse"] == 1
    assert result["unreleased_other_context"] == 1


def test_released_occurrences_are_marked_and_counted():
    result = reuse.group_recurrence(
        [_pair("ep1", 100.0, 130.0, "ep8", 700.0, 730.0)], _recipe(),
        released_spans={"ep1": [(95.0, 140.0)]})
    group = result["groups"][0]
    assert group["released"] is True
    assert group["known_state"] == reuse.KNOWN_RELEASED
    assert result["released_count"] == 1 and result["unreleased_count"] == 0


def test_partial_released_overlap_below_share_stays_unreleased():
    result = reuse.group_recurrence(
        [_pair("ep1", 100.0, 130.0, "ep8", 700.0, 730.0)], _recipe(),
        released_spans={"ep1": [(100.0, 105.0)]})
    assert result["groups"][0]["released"] is False


def test_explained_values_are_kept_for_the_group():
    result = reuse.group_recurrence([
        _pair("ep1", 100.0, 130.0, "ep8", 700.0, 730.0, explained=0.30),
        _pair("ep1", 100.0, 130.0, "ep9", 950.0, 980.0, explained=0.50),
    ], _recipe())
    group = result["groups"][0]
    assert group["explained_values"] == [0.3, 0.5]
    assert group["explained_median"] == pytest.approx(0.5)


@pytest.mark.parametrize("broken", [
    {"first": "", "first_start_seconds": 0.0, "first_end_seconds": 1.0,
     "second": "ep8", "second_start_seconds": 0.0, "second_end_seconds": 1.0},
    {"first": "ep1", "first_start_seconds": 5.0, "first_end_seconds": 5.0,
     "second": "ep8", "second_start_seconds": 0.0, "second_end_seconds": 1.0},
    {"first": "ep1", "first_start_seconds": 0.0, "first_end_seconds": 1.0,
     "second": "ep8", "second_start_seconds": 0.0, "second_end_seconds": 1.0,
     "waveform_explained": 1.4},
])
def test_broken_pairs_are_rejected(broken):
    with pytest.raises(reuse.SeriesReuseError):
        reuse.group_recurrence([broken], _recipe())


def test_overlap_share_and_coverage_helpers():
    assert reuse.overlap_share((0.0, 10.0), (5.0, 15.0)) == pytest.approx(0.5)
    assert reuse.overlap_share((0.0, 10.0), (20.0, 30.0)) == 0.0
    assert reuse.covered_by((0.0, 10.0), [(0.0, 6.0)], 0.5) is True
    assert reuse.covered_by((0.0, 10.0), [(0.0, 4.0)], 0.5) is False


def test_overlapping_spans_in_one_episode_become_one_occurrence():
    """同じ話の中で大きく重なる区間を、二つの使用箇所として数えない。"""
    result = reuse.group_recurrence([
        _pair("ep1", 900.0, 920.0, "ep6", 100.0, 120.0),
        _pair("ep1", 900.0, 935.0, "ep7", 300.0, 335.0),
    ], _recipe())
    assert result["group_count"] == 1
    group = result["groups"][0]
    ep1 = [item for item in group["occurrences"] if item["source_id"] == "ep1"]
    assert len(ep1) == 1
    assert ep1[0]["start_seconds"] == pytest.approx(900.0)
    assert ep1[0]["end_seconds"] == pytest.approx(935.0)
    assert group["occurrence_count"] == 3
