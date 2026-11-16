"""候補が一曲だと言えるのはどんな一致のときかを、理由の名前ごと固定する。"""
from __future__ import annotations

import pytest

from lss.bgm.cue_extent import (
    CueExtentError,
    CueExtentRecipe,
    combine_readings,
    extent_for_span,
    read_cue_extent,
)


def recipe(**changes) -> CueExtentRecipe:
    base = {"minimum_similarity": 0.7, "minimum_single_recording_coverage": 0.8,
            "maximum_offset_spread_seconds": 0.5}
    return CueExtentRecipe(**{**base, **changes})


def match(reference_id="cue-a", start=0.0, end=80.0, offset=-10.0, similarity=0.9) -> dict:
    return {"reference_id": reference_id, "query_start_seconds": start,
            "query_end_seconds": end, "offset_seconds": offset, "similarity": similarity}


def test_one_continuous_released_recording_covering_the_candidate_is_one_cue():
    read = read_cue_extent(0.0, 100.0, [match(end=85.0), match(start=85.0, end=100.0)], recipe())

    assert read["state"] == "single_cue"
    assert read["evidence"]["reason"] == "one_continuous_released_recording"
    assert read["evidence"]["single_recording_coverage"] == pytest.approx(1.0)
    assert read["recipe_id"].startswith("bgm-cue-extent-v1-")


def test_two_released_recordings_inside_one_candidate_are_more_than_one_cue():
    read = read_cue_extent(
        0.0, 100.0, [match(end=40.0), match(reference_id="cue-b", start=40.0, end=100.0)], recipe())

    assert read["state"] == "multiple_cues"
    assert read["evidence"]["reason"] == "more_than_one_released_recording"
    assert read["evidence"]["matched_reference_ids"] == ["cue-a", "cue-b"]


@pytest.mark.parametrize("matches,reason", [
    ([], "no_usable_match"),
    ([match(similarity=0.5)], "no_usable_match"),
    ([match(start=200.0, end=260.0)], "no_usable_match"),
    ([match(end=40.0)], "single_recording_coverage_below_minimum"),
    ([match(end=50.0), match(start=50.0, end=100.0, offset=-40.0)],
     "matched_offsets_not_one_continuous_playback"),
])
def test_every_reason_for_not_calling_it_one_cue_has_a_name(matches, reason):
    read = read_cue_extent(0.0, 100.0, matches, recipe())

    assert read["state"] == "unresolved"
    assert read["evidence"]["reason"] == reason


def test_an_edited_reuse_of_the_same_recording_is_not_called_one_cue():
    """同じ録音でも、参照側の位置が飛べば一続きの再生とは言わない。"""
    read = read_cue_extent(0.0, 100.0, [
        match(end=30.0, offset=-1.0), match(start=30.0, end=70.0, offset=-25.0),
        match(start=70.0, end=100.0, offset=10.0),
    ], recipe())

    assert read["state"] == "unresolved"
    assert read["evidence"]["offset_spread_seconds"] == pytest.approx(35.0)


def test_the_declared_thresholds_change_the_reading():
    matches = [match(end=70.0)]

    assert read_cue_extent(0.0, 100.0, matches, recipe())["state"] == "unresolved"
    assert read_cue_extent(
        0.0, 100.0, matches, recipe(minimum_single_recording_coverage=0.7))["state"] == "single_cue"


@pytest.mark.parametrize("changes", [
    {"minimum_similarity": 0.0},
    {"minimum_similarity": 1.5},
    {"minimum_single_recording_coverage": 0.0},
    {"maximum_offset_spread_seconds": -1.0},
    {"maximum_offset_spread_seconds": float("inf")},
    {"method": "released-reference-cue-extent"},
])
def test_rejects_unusable_declarations(changes):
    with pytest.raises(CueExtentError):
        recipe(**changes).validate()


@pytest.mark.parametrize("bad", [
    {"reference_id": "", "query_start_seconds": 0.0, "query_end_seconds": 1.0,
     "offset_seconds": 0.0, "similarity": 0.9},
    {"reference_id": "a", "query_start_seconds": 5.0, "query_end_seconds": 5.0,
     "offset_seconds": 0.0, "similarity": 0.9},
    {"reference_id": "a", "query_start_seconds": 0.0, "query_end_seconds": 1.0,
     "offset_seconds": None, "similarity": 0.9},
    "not-a-match",
])
def test_rejects_matches_that_cannot_be_read(bad):
    with pytest.raises(CueExtentError):
        read_cue_extent(0.0, 100.0, [bad], recipe())


def test_rejects_an_empty_candidate_span():
    with pytest.raises(CueExtentError, match="候補の区間"):
        read_cue_extent(10.0, 10.0, [match()], recipe())


def test_one_reading_saying_more_than_one_cue_decides_the_whole():
    combined = combine_readings([
        {"state": "single_cue", "evidence": {"from": "identity"}},
        {"state": "multiple_cues", "evidence": {"from": "music_gap"}},
    ])

    assert combined["state"] == "multiple_cues"
    assert combined["evidence"]["reason"] == "at_least_one_reading_says_more_than_one_cue"
    assert [row["state"] for row in combined["evidence"]["readings"]] == [
        "single_cue", "multiple_cues"]


def test_one_cue_passes_only_when_nothing_disagrees():
    assert combine_readings([
        {"state": "single_cue", "evidence": {}}, {"state": "unresolved", "evidence": {}},
    ])["state"] == "single_cue"
    assert combine_readings([
        {"state": "unresolved", "evidence": {}}, {"state": "unresolved", "evidence": {}},
    ])["state"] == "unresolved"


def test_combining_keeps_every_reason():
    combined = combine_readings([{"state": "unresolved", "evidence": {"reason": "no_usable_match"}},
                                 {"state": "unresolved", "evidence": {"reason": "no_interior_music_gap"}}])

    reasons = [row["evidence"]["reason"] for row in combined["evidence"]["readings"]]
    assert reasons == ["no_usable_match", "no_interior_music_gap"]


@pytest.mark.parametrize("readings", [[], [{"state": "maybe"}], ["single_cue"]])
def test_rejects_readings_that_cannot_be_combined(readings):
    with pytest.raises(CueExtentError):
        combine_readings(readings)


def child(start=0.0, end=100.0, state="single_cue", reason="one_continuous_released_recording"):
    return {"start_seconds": start, "end_seconds": end,
            "cue_extent": {"state": state, "evidence": {"reason": reason}}}


def test_one_child_covering_the_whole_span_carries_its_reading_up():
    read = extent_for_span(0.0, 100.0, [child()])

    assert read["state"] == "single_cue"
    assert read["evidence"]["reason"] == "one_child_is_one_cue_and_covers_the_span"


def test_two_decided_children_mean_the_span_holds_more_than_one_cue():
    read = extent_for_span(0.0, 100.0, [child(end=40.0), child(start=40.0)])

    assert read["state"] == "multiple_cues"
    assert read["evidence"]["reason"] == "more_than_one_child_is_its_own_cue"


def test_a_child_that_holds_more_than_one_cue_decides_the_span():
    read = extent_for_span(0.0, 100.0, [child(state="multiple_cues", reason="x")])

    assert read["state"] == "multiple_cues"
    assert read["evidence"]["reason"] == "a_child_holds_more_than_one_cue"


@pytest.mark.parametrize("children", [
    [],
    [child(end=60.0)],
    [{"start_seconds": 0.0, "end_seconds": 100.0}],
    [child(state="unresolved", reason="no_usable_match")],
    [child(start=-50.0, end=100.0)],
])
def test_a_span_without_a_covering_decided_child_gets_no_reading(children):
    assert extent_for_span(0.0, 100.0, children) is None


def test_an_empty_parent_span_is_refused():
    with pytest.raises(CueExtentError):
        extent_for_span(10.0, 10.0, [])
