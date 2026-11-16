"""シリーズの未発売cue経路を一つの宣言として実行する。音声は使わない。"""
from __future__ import annotations

import pytest

from lss.bgm import content_guard, cue_selection, occurrence_quality, series_cues, series_reuse


def _recipe(*, identical=0.95, minimum_explained=0.5, minimum_seconds=4.0,
            speech=0.0, effects=0.0):
    return series_cues.SeriesCueRecipe(
        reuse=series_reuse.SeriesReuseRecipe(
            identical_explained=identical, released_overlap_share=0.5,
            occurrence_overlap_share=0.5),
        quality=occurrence_quality.OccurrenceQualityRecipe(minimum_explained=minimum_explained),
        selection=cue_selection.CueSelectionRecipe(
            start_pad_seconds=1.0, end_pad_seconds=6.5, minimum_seconds=minimum_seconds),
        guard=content_guard.ContentGuardRecipe(
            music_over_speech_db=speech, music_over_effects_db=effects))


def _pair(first, first_start, first_end, second, second_start, second_end, explained=0.4):
    return {"first": first, "first_start_seconds": first_start, "first_end_seconds": first_end,
            "second": second, "second_start_seconds": second_start,
            "second_end_seconds": second_end, "waveform_explained": explained}


def test_recipe_id_follows_every_stage():
    base = _recipe()
    assert base.recipe_id == _recipe().recipe_id
    assert base.recipe_id.startswith("bgm-series-cues-v1-")
    assert base.recipe_id != _recipe(minimum_seconds=5.0).recipe_id
    assert base.recipe_id != _recipe(speech=3.0).recipe_id
    assert set(base.stage_ids) == {"reuse", "quality", "selection", "guard"}


def test_recipe_rejects_a_stage_that_is_not_a_recipe():
    recipe = series_cues.SeriesCueRecipe(
        reuse=series_reuse.SeriesReuseRecipe(
            identical_explained=0.95, released_overlap_share=0.5, occurrence_overlap_share=0.5),
        quality=occurrence_quality.OccurrenceQualityRecipe(minimum_explained=0.5),
        selection=cue_selection.CueSelectionRecipe(
            start_pad_seconds=1.0, end_pad_seconds=6.5, minimum_seconds=4.0),
        guard="guard")
    with pytest.raises(series_cues.SeriesCueError):
        recipe.validate()


def test_recipe_rejects_an_unknown_schema():
    with pytest.raises(series_cues.SeriesCueError):
        series_cues.SeriesCueRecipe(
            reuse=series_reuse.SeriesReuseRecipe(
                identical_explained=0.95, released_overlap_share=0.5,
                occurrence_overlap_share=0.5),
            quality=occurrence_quality.OccurrenceQualityRecipe(minimum_explained=0.5),
            selection=cue_selection.CueSelectionRecipe(
                start_pad_seconds=1.0, end_pad_seconds=6.5, minimum_seconds=4.0),
            guard=content_guard.ContentGuardRecipe(
                music_over_speech_db=0.0, music_over_effects_db=0.0),
            method="other").validate()


def test_an_unreleased_recurrence_becomes_one_span():
    plan = series_cues.plan_cues(
        [_pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0)], _recipe(),
        source_seconds={"ep1": 1400.0, "ep2": 1400.0})
    assert plan["span_count"] == 1
    span = plan["spans"][0]
    assert span["source_id"] in {"ep1", "ep2"}
    assert span["start_seconds"] == pytest.approx(span["occurrence_start_seconds"] - 1.0)
    assert span["end_seconds"] == pytest.approx(span["occurrence_end_seconds"] + 6.5)
    assert plan["occurrence_ranking_basis"] == "group_explained_median"


def test_a_released_recurrence_is_not_offered():
    plan = series_cues.plan_cues(
        [_pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0)], _recipe(),
        released_spans={"ep1": [(90.0, 130.0)]})
    assert plan["span_count"] == 0
    assert plan["grouped"]["released_count"] == 1


def test_identical_segment_reuse_is_not_offered():
    plan = series_cues.plan_cues(
        [_pair("ep1", 0.2, 14.2, "ep2", 0.2, 14.2, explained=0.99)], _recipe())
    assert plan["span_count"] == 0
    assert plan["grouped"]["unreleased_identical_reuse"] == 1


def test_a_theme_recording_is_excluded_from_the_plan():
    plan = series_cues.plan_cues([_pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0)], _recipe())
    key = plan["grouped"]["groups"][0]["recording_key"]
    excluded = series_cues.plan_cues(
        [_pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0)], _recipe(),
        theme_recording_keys=[key])
    assert excluded["occurrence_quality"]["theme_excluded_count"] == 1
    assert excluded["span_count"] == 0


def test_mutual_scores_choose_the_least_overlaid_occurrence():
    pair = _pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0)
    plan = series_cues.plan_cues(
        [pair], _recipe(),
        mutual_explained={series_cues.occurrence_id("ep2", 300.0): 0.9,
                          series_cues.occurrence_id("ep1", 100.0): 0.2})
    assert plan["occurrence_ranking_basis"] == "mutual_occurrence"
    assert plan["spans"][0]["source_id"] == "ep2"


def test_padding_is_clipped_to_the_source_length():
    plan = series_cues.plan_cues(
        [_pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0)], _recipe(),
        source_seconds={"ep1": 121.0, "ep2": 121.0})
    assert plan["spans"][0]["end_seconds"] == pytest.approx(121.0)


def test_a_span_shorter_than_the_minimum_is_skipped():
    plan = series_cues.plan_cues(
        [_pair("ep1", 100.0, 102.0, "ep2", 300.0, 302.0)], _recipe(minimum_seconds=4.0))
    assert plan["span_count"] == 0
    assert plan["selection"]["skipped"][0]["reason"] == cue_selection.SKIP_TOO_SHORT


def _plan_with_two_groups():
    return series_cues.plan_cues(
        [_pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0),
         _pair("ep1", 500.0, 530.0, "ep2", 700.0, 730.0)], _recipe())


def test_without_measurements_nothing_is_rejected_but_nothing_is_confirmed():
    plan = _plan_with_two_groups()
    result = series_cues.apply_content_guard(plan, _recipe())
    assert result["guard_state"] == series_cues.GUARD_NOT_MEASURED
    assert result["held_count"] == plan["span_count"]
    assert result["guard"] is None
    assert any("確かめていない" in line for line in result["claim_boundary"])


def test_measurements_reject_recurring_dialogue():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    measurements = [
        {"recording_key": keys[0], "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0},
        {"recording_key": keys[1], "speech_db": -2.0, "music_db": -5.2, "effects_db": -16.9},
    ]
    result = series_cues.apply_content_guard(plan, _recipe(), measurements)
    assert result["guard_state"] == series_cues.GUARD_APPLIED
    assert [span["recording_key"] for span in result["held"]] == [keys[0]]
    assert result["rejected"][0]["reason"] == content_guard.REJECT_SPEECH
    assert result["rejected"][0]["recording_key"] == keys[1]


def test_a_span_without_a_measurement_is_refused():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    with pytest.raises(series_cues.SeriesCueError):
        series_cues.apply_content_guard(
            plan, _recipe(),
            [{"recording_key": keys[0], "speech_db": -20.0, "music_db": -1.0,
              "effects_db": -30.0}])


def test_the_guard_needs_a_plan():
    with pytest.raises(series_cues.SeriesCueError):
        series_cues.apply_content_guard({"spans": None}, _recipe(), [])


def test_the_plan_states_what_it_does_not_claim():
    plan = series_cues.plan_cues([_pair("ep1", 100.0, 120.0, "ep2", 300.0, 320.0)], _recipe())
    assert any("欲しい劇伴かどうかは判定しない" in line for line in plan["claim_boundary"])
    assert any("一度しか鳴らない曲" in line for line in plan["claim_boundary"])


def test_measurements_from_more_than_one_series_are_refused():
    # recording_keyはシリーズの中でだけ一意。混ぜて渡すと区間と測定の対応が決まらない。
    plan = _plan_with_two_groups()
    key = plan["spans"][0]["recording_key"]
    rows = [{"recording_key": key, "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0},
            {"recording_key": key, "speech_db": -2.0, "music_db": -5.2, "effects_db": -16.9}]
    with pytest.raises(series_cues.SeriesCueError):
        series_cues.apply_content_guard(plan, _recipe(), rows)


def _recipe_for(target):
    base = _recipe()
    return series_cues.SeriesCueRecipe(
        reuse=base.reuse, quality=base.quality, selection=base.selection, guard=base.guard,
        guard_target=target)


def test_the_guard_target_is_part_of_the_declaration():
    occurrence = _recipe_for(series_cues.GUARD_TARGET_OCCURRENCE)
    estimate = _recipe_for(series_cues.GUARD_TARGET_ESTIMATE)
    assert occurrence.recipe_id != estimate.recipe_id
    assert occurrence.stage_ids == estimate.stage_ids  # 段は同じで、かけどころだけ違う


def test_an_unknown_guard_target_is_refused():
    with pytest.raises(series_cues.SeriesCueError):
        _recipe_for("anything").validate()


def test_the_guard_target_is_recorded_even_without_measurements():
    plan = _plan_with_two_groups()
    result = series_cues.apply_content_guard(plan, _recipe_for(series_cues.GUARD_TARGET_ESTIMATE))
    assert result["guard_target"] == series_cues.GUARD_TARGET_ESTIMATE
    assert result["guard_state"] == series_cues.GUARD_NOT_MEASURED


def test_measurements_of_only_the_wrong_target_are_refused():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    rows = [{"recording_key": key, "target": series_cues.GUARD_TARGET_OCCURRENCE,
             "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0} for key in keys]
    with pytest.raises(series_cues.SeriesCueError):
        series_cues.apply_content_guard(
            plan, _recipe_for(series_cues.GUARD_TARGET_ESTIMATE), rows)


def _both_targets(keys, *, estimate, occurrence):
    rows = []
    for key in keys:
        rows.append({"recording_key": key, "target": series_cues.GUARD_TARGET_ESTIMATE,
                     **estimate})
        rows.append({"recording_key": key, "target": series_cues.GUARD_TARGET_OCCURRENCE,
                     **occurrence})
    return rows


def test_an_untrusted_estimate_is_judged_on_the_occurrence_span():
    # 実番組の既発売cueで、残した割合が下限に届かない推定は原本の区間より録音から遠かった。
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    rows = _both_targets(
        keys,
        estimate={"speech_db": -1.0, "music_db": -8.0, "effects_db": -30.0},   # 落ちる
        occurrence={"speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0})  # 通る
    recipe = _recipe_for(series_cues.GUARD_TARGET_ESTIMATE)

    trusted = series_cues.apply_content_guard(plan, recipe, rows)
    assert trusted["held_count"] == 0
    assert set(trusted["guard_targets_used"].values()) == {series_cues.GUARD_TARGET_ESTIMATE}

    fallen_back = series_cues.apply_content_guard(
        plan, recipe, rows, untrusted_recording_keys=keys)
    assert fallen_back["held_count"] == len(keys)
    assert set(fallen_back["guard_targets_used"].values()) == {
        series_cues.GUARD_TARGET_OCCURRENCE}
    assert fallen_back["untrusted_recording_keys"] == sorted(keys)


def test_only_the_untrusted_recordings_fall_back():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    rows = _both_targets(
        keys,
        estimate={"speech_db": -1.0, "music_db": -8.0, "effects_db": -30.0},
        occurrence={"speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0})
    result = series_cues.apply_content_guard(
        plan, _recipe_for(series_cues.GUARD_TARGET_ESTIMATE), rows,
        untrusted_recording_keys=[keys[0]])
    assert result["guard_targets_used"][keys[0]] == series_cues.GUARD_TARGET_OCCURRENCE
    assert result["guard_targets_used"][keys[1]] == series_cues.GUARD_TARGET_ESTIMATE
    assert [span["recording_key"] for span in result["held"]] == [keys[0]]


def test_a_measurement_with_an_unknown_target_is_refused():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    rows = [{"recording_key": key, "target": "something", "speech_db": -20.0,
             "music_db": -1.0, "effects_db": -30.0} for key in keys]
    with pytest.raises(series_cues.SeriesCueError):
        series_cues.apply_content_guard(plan, _recipe_for(
            series_cues.GUARD_TARGET_ESTIMATE), rows)


def test_the_same_recording_and_target_cannot_be_measured_twice():
    plan = _plan_with_two_groups()
    key = plan["spans"][0]["recording_key"]
    rows = [{"recording_key": key, "target": series_cues.GUARD_TARGET_ESTIMATE,
             "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0}] * 2
    with pytest.raises(series_cues.SeriesCueError):
        series_cues.apply_content_guard(plan, _recipe_for(
            series_cues.GUARD_TARGET_ESTIMATE), rows)


def test_measurements_of_the_declared_target_are_accepted():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    rows = [{"recording_key": key, "target": series_cues.GUARD_TARGET_ESTIMATE,
             "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0} for key in keys]
    result = series_cues.apply_content_guard(
        plan, _recipe_for(series_cues.GUARD_TARGET_ESTIMATE), rows)
    assert result["guard_state"] == series_cues.GUARD_APPLIED
    assert result["guard_target"] == series_cues.GUARD_TARGET_ESTIMATE
    assert result["held_count"] == 2


def test_the_default_declares_the_estimate_as_the_guard_target():
    # 実番組18件で、原本の区間にかけると台詞の下の劇伴まで落ちる。
    assert series_cues.DEFAULT_RECIPE.guard_target == series_cues.GUARD_TARGET_ESTIMATE


def _marking_recipe(*, explained=0.53, share=0.33):
    base = _recipe()
    return series_cues.SeriesCueRecipe(
        reuse=base.reuse, quality=base.quality, selection=base.selection, guard=base.guard,
        stock_effect_explained=explained, stock_effect_energy_share=share)


def test_a_recording_that_repeats_identically_is_marked():
    # 既製の効果音は毎回同じ波形なので、説明率も残した割合も高く出る。
    recipe = _marking_recipe()
    assert series_cues.marks_for(0.60, 0.40, recipe) == [series_cues.MARK_STOCK_EFFECT]


@pytest.mark.parametrize("explained,share", [
    (0.608, 0.271),  # 説明率は高いが残した割合が低い（実データの最近傍）
    (0.470, 0.414),  # 残した割合は高いが説明率が低い（実データの最近傍）
    (0.20, 0.10),
])
def test_a_score_under_different_dialogue_is_not_marked(explained, share):
    assert series_cues.marks_for(explained, share, _marking_recipe()) == []


def test_a_candidate_without_an_estimate_gets_no_mark():
    recipe = _marking_recipe()
    assert series_cues.marks_for(0.9, None, recipe) == []
    assert series_cues.marks_for(None, 0.9, recipe) == []


@pytest.mark.parametrize("kwargs", [{"explained": 0.0}, {"share": 1.5},
                                    {"explained": float("nan")}])
def test_impossible_marking_thresholds_are_refused(kwargs):
    with pytest.raises(series_cues.SeriesCueError):
        _marking_recipe(**kwargs).validate()


def test_the_marking_thresholds_are_part_of_the_declaration():
    assert _marking_recipe().recipe_id != _marking_recipe(explained=0.6).recipe_id


def test_the_default_marks_but_does_not_reject():
    # 印であって、落とす判断ではない。関門は別に走る。
    assert series_cues.DEFAULT_RECIPE.stock_effect_explained == 0.53
    assert series_cues.DEFAULT_RECIPE.stock_effect_energy_share == 0.33


def _no_measurement_recipe(*, enabled=True, explained=0.53, share=0.33):
    base = _recipe()
    return series_cues.SeriesCueRecipe(
        reuse=base.reuse, quality=base.quality, selection=base.selection, guard=base.guard,
        stock_effect_explained=explained, stock_effect_energy_share=share,
        select_without_measurement=enabled)


def test_without_measurements_the_marks_alone_can_select():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    result = series_cues.apply_content_guard(
        plan, _no_measurement_recipe(), marked_recording_keys=[keys[0]],
        untrusted_recording_keys=[keys[1]])
    assert result["guard_state"] == series_cues.GUARD_MARKS_ONLY
    assert result["held_count"] == 0
    reasons = {item["recording_key"]: item["reason"] for item in result["rejected"]}
    assert reasons[keys[0]] == series_cues.MARK_STOCK_EFFECT
    assert reasons[keys[1]] == series_cues.REJECT_UNTRUSTED_ESTIMATE
    assert any("分離器の測定を使っていない" in line for line in result["claim_boundary"])


def test_marks_only_selection_keeps_what_is_neither_marked_nor_untrusted():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    result = series_cues.apply_content_guard(
        plan, _no_measurement_recipe(), marked_recording_keys=[keys[0]])
    assert [span["recording_key"] for span in result["held"]] == [keys[1]]


def test_without_the_declaration_no_measurements_means_no_selection():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    result = series_cues.apply_content_guard(
        plan, _no_measurement_recipe(enabled=False), marked_recording_keys=keys)
    assert result["guard_state"] == series_cues.GUARD_NOT_MEASURED
    assert result["held_count"] == len(keys)


def test_measurements_still_win_over_the_marks():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    rows = [{"recording_key": key, "target": series_cues.GUARD_TARGET_OCCURRENCE,
             "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0} for key in keys]
    result = series_cues.apply_content_guard(
        plan, _no_measurement_recipe(), rows, marked_recording_keys=keys)
    assert result["guard_state"] == series_cues.GUARD_APPLIED
    assert result["held_count"] == len(keys)


def test_the_declaration_changes_the_route_identifier_too():
    assert _no_measurement_recipe().recipe_id != _no_measurement_recipe(enabled=False).recipe_id


def test_the_mark_names_identical_repetition_not_effects():
    # 当てはめていない材料では、主題歌のような「毎回同じ音源の音楽」にも鳴った。
    assert series_cues.MARK_IDENTICAL_REPEAT == "likely_identical_repeat"
    assert series_cues.MARK_STOCK_EFFECT == series_cues.MARK_IDENTICAL_REPEAT


def test_the_default_does_not_select_on_the_mark_alone():
    # 差し引ける参照が無い作品では、本物の音楽を落としかねない。
    assert series_cues.DEFAULT_RECIPE.select_without_measurement is False


def test_the_default_still_marks_so_the_record_keeps_the_signal():
    assert series_cues.marks_for(0.9, 0.9, series_cues.DEFAULT_RECIPE) == [
        series_cues.MARK_IDENTICAL_REPEAT]


def test_without_measurements_the_default_holds_everything():
    plan = _plan_with_two_groups()
    keys = [span["recording_key"] for span in plan["spans"]]
    result = series_cues.apply_content_guard(
        plan, series_cues.DEFAULT_RECIPE, marked_recording_keys=keys)
    assert result["guard_state"] == series_cues.GUARD_NOT_MEASURED
    assert result["held_count"] == len(keys)
