"""別の回に同じ録音が出ることを、候補の範囲の根拠として読めることを固定する。"""
from __future__ import annotations

import pytest

from lss.bgm.cue_extent import CueExtentRecipe, read_cue_extent
from lss.bgm.recurring_extent import (
    EVIDENCE_KIND,
    RecurringExtentError,
    read_recurring_extent,
    recurring_matches,
    spans_for_episode,
)


def span(start=0.0, end=60.0, other="ep8", other_start=500.0, other_end=560.0,
         explained=0.8) -> dict:
    return {"other_episode": other, "start_seconds": start, "end_seconds": end,
            "other_start_seconds": other_start, "other_end_seconds": other_end,
            "waveform_explained": explained}


def recipe(**changes) -> CueExtentRecipe:
    base = {"minimum_similarity": 0.3, "minimum_single_recording_coverage": 0.8,
            "maximum_offset_spread_seconds": 0.5}
    return CueExtentRecipe(**{**base, **changes})


def test_one_recurring_recording_covering_the_candidate_is_one_cue():
    matches = recurring_matches([span(end=50.0, other_end=550.0),
                                 span(start=50.0, end=100.0, other_start=550.0,
                                      other_end=600.0)])
    read = read_cue_extent(0.0, 100.0, matches, recipe(), evidence_kind=EVIDENCE_KIND)

    assert read["state"] == "single_cue"
    assert read["evidence"]["reason"] == "one_continuous_recurring_recording"
    assert read["evidence"]["evidence_kind"] == "recurring_recording"
    assert read["evidence"]["single_recording_coverage"] == pytest.approx(1.0)


def test_two_recurring_recordings_inside_one_candidate_are_more_than_one_cue():
    matches = recurring_matches([span(end=40.0, other_end=540.0),
                                 span(start=40.0, end=100.0, other_start=900.0,
                                      other_end=960.0)])
    read = read_cue_extent(0.0, 100.0, matches, recipe(), evidence_kind=EVIDENCE_KIND)

    assert read["state"] == "multiple_cues"
    assert read["evidence"]["reason"] == "more_than_one_recurring_recording"
    assert len(read["evidence"]["matched_reference_ids"]) == 2


def test_the_same_recording_played_twice_in_the_other_episode_keeps_one_name():
    matches = recurring_matches([span(end=60.0, other_start=500.0, other_end=560.0),
                                 span(end=60.0, other_start=900.0, other_end=960.0)])

    assert len({row["reference_id"] for row in matches}) == 1


def test_the_same_recording_played_twice_in_this_episode_keeps_one_name():
    matches = recurring_matches([span(start=0.0, end=60.0),
                                 span(start=300.0, end=360.0)])

    assert len({row["reference_id"] for row in matches}) == 1


def test_recordings_shared_with_different_episodes_are_named_apart():
    matches = recurring_matches([span(other="ep8"), span(other="ep12")])

    assert len({row["reference_id"] for row in matches}) == 2


def test_a_weak_recurring_span_is_left_out_by_the_declared_strength():
    matches = recurring_matches([span(explained=0.2)])
    read = read_cue_extent(0.0, 60.0, matches, recipe(minimum_similarity=0.5),
                           evidence_kind=EVIDENCE_KIND)

    assert read["state"] == "unresolved"
    assert read["evidence"]["reason"] == "no_usable_match"


def test_occurrences_are_read_from_both_sides_of_the_pair():
    spans = spans_for_episode([{
        "first_start_seconds": 10.0, "first_end_seconds": 40.0,
        "second_start_seconds": 200.0, "second_end_seconds": 230.0,
        "waveform_explained": 0.7}], first="ep1", second="ep8")

    assert spans["ep1"][0]["start_seconds"] == pytest.approx(10.0)
    assert spans["ep1"][0]["other_start_seconds"] == pytest.approx(200.0)
    assert spans["ep8"][0]["start_seconds"] == pytest.approx(200.0)
    assert spans["ep8"][0]["other_episode"] == "ep1"


@pytest.mark.parametrize("rows", [
    "spans",
    [{**span(), "other_episode": " "}],
    [{**span(), "end_seconds": 0.0}],
    [{**span(), "other_end_seconds": 0.0}],
    [{**span(), "waveform_explained": None}],
    [{**span(), "waveform_explained": 1.5}],
])
def test_a_malformed_recurring_span_is_refused(rows):
    with pytest.raises(RecurringExtentError):
        recurring_matches(rows)


def test_a_pair_needs_two_different_episodes():
    with pytest.raises(RecurringExtentError):
        spans_for_episode([], first="ep1", second="ep1")


def test_two_recurring_names_do_not_decide_that_a_candidate_holds_two_cues():
    matches = recurring_matches([span(end=40.0, other_end=540.0),
                                 span(start=40.0, end=100.0, other_start=900.0,
                                      other_end=960.0)])
    read = read_recurring_extent(0.0, 100.0, matches, recipe())

    assert read["state"] == "unresolved"
    assert read["evidence"]["reason"] == "recurring_names_may_be_one_recording"
    assert len(read["evidence"]["matched_reference_ids"]) == 2


def test_one_recurring_recording_still_decides_the_extent():
    matches = recurring_matches([span(end=100.0, other_end=600.0)])
    read = read_recurring_extent(0.0, 100.0, matches, recipe())

    assert read["state"] == "single_cue"
    assert read["evidence"]["reason"] == "one_continuous_recurring_recording"


def test_an_unknown_evidence_kind_is_refused():
    from lss.bgm.cue_extent import CueExtentError

    with pytest.raises(CueExtentError):
        read_cue_extent(0.0, 60.0, [], recipe(), evidence_kind="guessed_recording")


def test_a_repeat_inside_one_source_links_both_appearances():
    matches = recurring_matches([
        {**span(start=30.0, end=55.0, other="ep1", other_start=200.0, other_end=225.0),
         "same_timeline": True},
        {**span(start=200.0, end=225.0, other="ep1", other_start=30.0, other_end=55.0),
         "same_timeline": True}])

    assert len({row["reference_id"] for row in matches}) == 1


def test_spans_from_different_sources_are_not_linked_across_timelines():
    matches = recurring_matches([
        span(start=30.0, end=55.0, other_start=200.0, other_end=225.0),
        span(start=200.0, end=225.0, other_start=30.0, other_end=55.0)])

    assert len({row["reference_id"] for row in matches}) == 2


def test_a_malformed_same_timeline_flag_is_refused():
    with pytest.raises(RecurringExtentError):
        recurring_matches([{**span(), "same_timeline": "yes"}])
