"""根拠のある止まりで候補を切り、断片の出どころと短さの理由を残す。"""
from __future__ import annotations

import pytest

from lss.bgm.cue_split import CueSplitError, CueSplitRecipe, split_at_gaps


def recipe(**changes) -> CueSplitRecipe:
    return CueSplitRecipe(**{"minimum_piece_seconds": 20.0, **changes})


def gap(start, end) -> dict:
    return {"start_seconds": start, "end_seconds": end}


def test_the_stop_itself_belongs_to_neither_piece():
    split = split_at_gaps(100.0, 240.0, [gap(190.0, 192.0)], recipe())

    assert [(row["start_seconds"], row["end_seconds"]) for row in split["pieces"]] == [
        (100.0, 190.0), (192.0, 240.0)]
    assert split["usable_piece_count"] == 2
    assert split["parent"] == {"start_seconds": 100.0, "end_seconds": 240.0, "seconds": 140.0}
    assert split["recipe_id"].startswith("bgm-cue-split-v1-")


def test_several_stops_give_several_pieces_in_order():
    split = split_at_gaps(0.0, 300.0, [gap(200.0, 202.0), gap(100.0, 101.0)], recipe())

    assert [row["seconds"] for row in split["pieces"]] == [100.0, 99.0, 98.0]
    assert split["gaps_used"] == [{"start_seconds": 100.0, "end_seconds": 101.0},
                                  {"start_seconds": 200.0, "end_seconds": 202.0}]


def test_a_short_piece_is_kept_with_its_reason():
    split = split_at_gaps(0.0, 120.0, [gap(10.0, 12.0)], recipe())

    short, long = split["pieces"]
    assert short["usable"] is False and short["reason"] == "piece_below_minimum_length"
    assert long["usable"] is True
    assert split["usable_piece_count"] == 1


def test_a_stop_touching_the_edge_leaves_one_piece():
    split = split_at_gaps(0.0, 100.0, [gap(0.0, 5.0)], recipe())

    assert len(split["pieces"]) == 1
    assert split["pieces"][0]["start_seconds"] == 5.0


def test_no_stop_returns_the_candidate_unchanged():
    split = split_at_gaps(10.0, 60.0, [], recipe())

    assert len(split["pieces"]) == 1
    assert (split["pieces"][0]["start_seconds"], split["pieces"][0]["end_seconds"]) == (10.0, 60.0)


@pytest.mark.parametrize("start,end,gaps,message", [
    (10.0, 10.0, [], "候補の区間"),
    (0.0, 100.0, [gap(50.0, 50.0)], "止まりの区間"),
    (0.0, 100.0, [gap(90.0, 110.0)], "候補の外"),
    (0.0, 100.0, [gap(-5.0, 10.0)], "候補の外"),
    (0.0, 100.0, [gap(10.0, 30.0), gap(20.0, 40.0)], "重なって"),
    (0.0, 100.0, ["not-a-gap"], "止まりはオブジェクト"),
])
def test_rejects_spans_that_cannot_be_split(start, end, gaps, message):
    with pytest.raises(CueSplitError, match=message):
        split_at_gaps(start, end, gaps, recipe())


@pytest.mark.parametrize("changes", [
    {"minimum_piece_seconds": 0.0},
    {"minimum_piece_seconds": float("inf")},
    {"method": "split-at-stops"},
])
def test_rejects_unusable_declarations(changes):
    with pytest.raises(CueSplitError):
        recipe(**changes).validate()
