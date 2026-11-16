"""再演のまとまりから切り出す区間を選ぶ。音声は使わない。"""
from __future__ import annotations

import pytest

from lss.bgm import cue_selection, series_reuse


def _recipe(start=1.0, end=6.5, minimum=4.0):
    return cue_selection.CueSelectionRecipe(
        start_pad_seconds=start, end_pad_seconds=end, minimum_seconds=minimum)


def _group(key, occurrences, released=False, reuse=series_reuse.REUSE_OTHER_CONTEXT, median=0.4):
    return {"recording_key": key, "released": released, "reuse_class": reuse,
            "occurrence_count": len(occurrences),
            "episode_count": len({item[0] for item in occurrences}),
            "explained_median": median,
            "occurrences": [{"source_id": s, "start_seconds": a, "end_seconds": b,
                             "seconds": round(b - a, 3)} for s, a, b in occurrences]}


def _grouped(*groups):
    return {"groups": list(groups)}


def test_recipe_id_depends_on_the_declaration():
    assert _recipe().recipe_id == _recipe().recipe_id
    assert _recipe().recipe_id != _recipe(end=7.0).recipe_id
    assert _recipe().recipe_id.startswith("bgm-cue-selection-v1-")


@pytest.mark.parametrize("kwargs", [{"start": -1.0}, {"end": float("nan")}, {"minimum": 0.0}])
def test_recipe_rejects_invalid_values(kwargs):
    with pytest.raises(cue_selection.CueSelectionError):
        _recipe(**kwargs).validate()


def test_padding_is_asymmetric_because_ends_are_looser():
    result = cue_selection.select_extraction_spans(
        _grouped(_group("r0", [("ep1", 100.0, 130.0)])), _recipe())
    span = result["selected"][0]
    assert span["start_seconds"] == pytest.approx(99.0)
    assert span["end_seconds"] == pytest.approx(136.5)
    assert span["occurrence_start_seconds"] == pytest.approx(100.0)


def test_start_padding_does_not_go_below_zero():
    result = cue_selection.select_extraction_spans(
        _grouped(_group("r0", [("ep1", 0.2, 14.2)])), _recipe())
    assert result["selected"][0]["start_seconds"] == pytest.approx(0.0)


def test_end_padding_is_clipped_to_the_source_length():
    result = cue_selection.select_extraction_spans(
        _grouped(_group("r0", [("ep1", 100.0, 130.0)])), _recipe(),
        source_seconds={"ep1": 132.0})
    assert result["selected"][0]["end_seconds"] == pytest.approx(132.0)


def test_released_and_identical_reuse_are_skipped_with_reasons():
    result = cue_selection.select_extraction_spans(_grouped(
        _group("r0", [("ep1", 10.0, 40.0)], released=True),
        _group("r1", [("ep1", 50.0, 80.0)], reuse=series_reuse.REUSE_IDENTICAL),
        _group("r2", [("ep1", 90.0, 120.0)]),
    ), _recipe())
    assert result["selected_count"] == 1 and result["selected"][0]["recording_key"] == "r2"
    reasons = {item["reason"] for item in result["skipped"]}
    assert reasons == {cue_selection.SKIP_RELEASED, cue_selection.SKIP_IDENTICAL}


def test_short_occurrence_is_skipped_but_named():
    result = cue_selection.select_extraction_spans(
        _grouped(_group("r0", [("ep1", 10.0, 12.0)])), _recipe(minimum=4.0))
    assert result["selected_count"] == 0
    assert result["skipped"][0]["reason"] == cue_selection.SKIP_TOO_SHORT
    assert result["skipped"][0]["seconds"] == pytest.approx(2.0)


def test_quality_selection_wins_over_the_longest_occurrence():
    grouped = _grouped(_group("r0", [("ep1", 10.0, 60.0), ("ep8", 500.0, 520.0)]))
    longest = cue_selection.select_extraction_spans(grouped, _recipe())
    assert longest["selected"][0]["source_id"] == "ep1"
    chosen = cue_selection.select_extraction_spans(
        grouped, _recipe(), selected_occurrence_ids=["ep8@500.000"])
    assert chosen["selected"][0]["source_id"] == "ep8"
    assert chosen["selected"][0]["selected_by"] == "occurrence_quality"


def test_group_without_occurrences_is_skipped():
    result = cue_selection.select_extraction_spans(_grouped(_group("r0", [])), _recipe())
    assert result["skipped"][0]["reason"] == cue_selection.SKIP_NO_OCCURRENCE


def test_rejects_input_that_is_not_a_grouping():
    with pytest.raises(cue_selection.CueSelectionError):
        cue_selection.select_extraction_spans({"nope": []}, _recipe())


def test_result_states_that_candidates_are_not_published():
    result = cue_selection.select_extraction_spans(
        _grouped(_group("r0", [("ep1", 10.0, 40.0)])), _recipe())
    assert any("正式なBGM成果物ではない" in line for line in result["claim_boundary"])


def test_an_excluded_recording_does_not_fall_back_to_its_longest_occurrence():
    # 前段（主題歌の除外など）が外した録音を、ここで拾い直さない。
    grouped = _grouped(_group("r0", [("ep1", 10.0, 30.0), ("ep2", 50.0, 70.0)]))
    kept = cue_selection.select_extraction_spans(grouped, _recipe())
    assert kept["selected_count"] == 1
    result = cue_selection.select_extraction_spans(
        grouped, _recipe(), excluded_recording_keys=["r0"])
    assert result["selected_count"] == 0
    assert result["skipped"] == [{"recording_key": "r0",
                                  "reason": cue_selection.SKIP_EXCLUDED}]


def test_exclusion_ignores_keys_that_are_not_present():
    grouped = _grouped(_group("r0", [("ep1", 10.0, 30.0), ("ep2", 50.0, 70.0)]))
    result = cue_selection.select_extraction_spans(
        grouped, _recipe(), excluded_recording_keys=["other", None, ""])
    assert result["selected_count"] == 1


def test_the_other_occurrences_of_the_recording_are_carried_with_the_span():
    # 同じ録音が他のどこで鳴っているかは来歴であり、複数の回を使う処理の入口でもある。
    grouped = _grouped(_group("r0", [("ep1", 10.0, 30.0), ("ep2", 50.0, 70.0),
                                     ("ep3", 90.0, 115.0)]))
    result = cue_selection.select_extraction_spans(
        grouped, _recipe(), selected_occurrence_ids=["ep1@10.000"])
    [span] = result["selected"]
    assert span["source_id"] == "ep1"
    assert span["other_occurrences"] == [
        {"source_id": "ep2", "start_seconds": 50.0, "end_seconds": 70.0},
        {"source_id": "ep3", "start_seconds": 90.0, "end_seconds": 115.0}]
