"""別々の録音が続けて鳴る位置だけを境目として数えることを固定する。"""
from __future__ import annotations

import pytest

from lss.bgm.identity_joins import (
    IdentityJoinError,
    IdentityJoinRecipe,
    read_joins,
)


def recipe(**changes) -> IdentityJoinRecipe:
    base = {"minimum_similarity": 0.7, "minimum_match_seconds": 6.0,
            "maximum_join_seconds": 2.0}
    return IdentityJoinRecipe(**{**base, **changes})


def match(reference_id="cue-a", start=0.0, end=30.0, similarity=0.9) -> dict:
    return {"reference_id": reference_id, "query_start_seconds": start,
            "query_end_seconds": end, "similarity": similarity}


def test_two_different_recordings_without_room_between_them_are_a_boundary():
    read = read_joins(
        [match(end=30.0), match(reference_id="cue-b", start=31.0, end=60.0)], recipe())

    assert len(read["joins"]) == 1
    join = read["joins"][0]
    assert join["before_reference_id"] == "cue-a"
    assert join["after_reference_id"] == "cue-b"
    assert join["earliest_seconds"] == pytest.approx(30.0)
    assert join["latest_seconds"] == pytest.approx(31.0)
    assert join["seconds"] == pytest.approx(30.5)
    assert read["separated_changes"] == []
    assert read["recipe_id"].startswith("bgm-identity-join-v1-")


def test_a_wide_gap_between_two_recordings_is_not_a_located_boundary():
    read = read_joins(
        [match(end=30.0), match(reference_id="cue-b", start=90.0, end=120.0)], recipe())

    assert read["joins"] == []
    assert len(read["separated_changes"]) == 1
    assert read["separated_changes"][0]["gap_seconds"] == pytest.approx(60.0)


def test_the_same_recording_matched_again_is_one_playback_not_a_boundary():
    read = read_joins([match(end=30.0), match(start=31.0, end=60.0)], recipe())

    assert read["joins"] == []
    assert read["separated_changes"] == []
    assert len(read["spans"]) == 1
    assert read["spans"][0]["match_count"] == 2
    assert read["spans"][0]["end_seconds"] == pytest.approx(60.0)


def test_the_same_recording_after_a_wide_gap_stays_two_playbacks_without_a_boundary():
    read = read_joins([match(end=30.0), match(start=200.0, end=230.0)], recipe())

    assert [span["start_seconds"] for span in read["spans"]] == [0.0, 200.0]
    assert read["joins"] == []
    assert read["separated_changes"] == []


def test_overlapping_matches_keep_the_stronger_one():
    read = read_joins(
        [match(end=30.0, similarity=0.75),
         match(reference_id="cue-b", start=20.0, end=50.0, similarity=0.95)], recipe())

    assert [span["reference_id"] for span in read["spans"]] == ["cue-b"]
    assert [row["reason"] for row in read["dropped"]] == ["overlaps_a_stronger_match"]
    assert read["dropped"][0]["reference_id"] == "cue-a"
    assert read["accepted_count"] == 1


@pytest.mark.parametrize("rows,reason", [
    ([match(similarity=0.5)], "similarity_below_minimum"),
    ([match(end=3.0)], "shorter_than_minimum"),
])
def test_weak_or_short_matches_are_dropped_with_a_named_reason(rows, reason):
    read = read_joins(rows, recipe())

    assert read["usable_count"] == 0
    assert [row["reason"] for row in read["dropped"]] == [reason]
    assert read["spans"] == []


def test_a_dropped_overlap_does_not_create_a_boundary_with_its_neighbour():
    read = read_joins(
        [match(end=30.0, similarity=0.95),
         match(reference_id="cue-b", start=29.0, end=40.0, similarity=0.8),
         match(reference_id="cue-c", start=41.0, end=70.0, similarity=0.8)], recipe())

    assert [span["reference_id"] for span in read["spans"]] == ["cue-a", "cue-c"]
    assert len(read["joins"]) == 0
    assert len(read["separated_changes"]) == 1


@pytest.mark.parametrize("changes", [
    {"minimum_similarity": 0.0}, {"minimum_similarity": 1.5},
    {"minimum_match_seconds": 0.0}, {"minimum_match_seconds": float("inf")},
    {"maximum_join_seconds": -1.0}, {"schema_version": 2}, {"method": "other"},
])
def test_an_unsupported_declaration_is_refused(changes):
    with pytest.raises(IdentityJoinError):
        recipe(**changes).validate()


@pytest.mark.parametrize("rows", [
    "matches",
    [{"query_start_seconds": 0.0, "query_end_seconds": 10.0, "similarity": 0.9}],
    [match(start=10.0, end=10.0)],
    [match(similarity=1.5)],
    [{**match(), "query_end_seconds": None}],
])
def test_a_malformed_match_is_refused(rows):
    with pytest.raises(IdentityJoinError):
        read_joins(rows, recipe())
