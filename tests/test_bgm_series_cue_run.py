"""シリーズの未発売cue経路を、実際の音声に対して実行する。分離器は使わない。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import content_guard, cue_selection, occurrence_quality, series_cue_run, series_cues, series_reuse
from tests.test_bgm_identity import RATE, _dialogue_and_noise, _music


def _recipe(*, minimum_seconds=4.0, speech=0.0, effects=0.0):
    return series_cues.SeriesCueRecipe(
        reuse=series_reuse.SeriesReuseRecipe(
            identical_explained=0.95, released_overlap_share=0.5, occurrence_overlap_share=0.5),
        quality=occurrence_quality.OccurrenceQualityRecipe(minimum_explained=0.5),
        selection=cue_selection.CueSelectionRecipe(
            start_pad_seconds=1.0, end_pad_seconds=6.5, minimum_seconds=minimum_seconds),
        guard=content_guard.ContentGuardRecipe(
            music_over_speech_db=speech, music_over_effects_db=effects))


def _episode(path, seconds, placements, *, seed):
    """台詞と雑音の上に、指定した時刻へ録音を重ねた1話ぶんの原本mix。

    実番組と同じく、話ごとに重なりが違うようにする。重なりが無いと説明率が0.95を超え、
    「区間ごと同一の使い回し」として別クラスへ分かれてしまう。
    """
    mix = _dialogue_and_noise(seconds, seed=seed) * 0.3
    for start, recording in placements:
        length = recording.size
        mix[int(start * RATE):int(start * RATE) + length] += recording
    sf.write(path, np.stack([mix, mix], axis=1), RATE, subtype="FLOAT")
    return path


@pytest.fixture
def series(tmp_path):
    """未発売の録音が2話に出る。既発売の録音は1話目だけに出る。"""
    cue = _music(41, 20.0)
    released = _music(42, 20.0)
    reference = tmp_path / "released.flac"
    sf.write(reference, np.stack([released, released], axis=1), RATE, subtype="PCM_24")
    first = _episode(tmp_path / "ep1.wav", 200.0, [(30.0, cue), (120.0, released)], seed=1)
    second = _episode(tmp_path / "ep2.wav", 200.0, [(140.0, cue)], seed=2)
    return {"ep1": first, "ep2": second}, reference


def test_a_recurring_unreleased_recording_becomes_one_held_candidate(series, tmp_path):
    episodes, _ = series
    before = {path: path.read_bytes() for path in episodes.values()}

    record_path = series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False)

    assert {path: path.read_bytes() for path in before} == before  # 原本は不変
    record = series_cue_run.read_result(record_path)
    assert record["span_count"] == 1
    [candidate] = record["candidates"]
    assert candidate["source_id"] in episodes
    assert candidate["artifact"]["publication_state"] == series_cue_run.PUBLICATION_STATE
    assert record["promotion_allowed"] is False
    assert record["holdout_consumed"] is False
    saved = tmp_path / "run" / series_cue_run.CANDIDATE_DIRNAME / candidate["file"]
    assert saved.is_file()
    assert sf.info(str(saved)).channels == 2


def test_the_found_span_covers_where_the_recording_actually_plays(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False))
    [candidate] = record["candidates"]
    played = 30.0 if candidate["source_id"] == "ep1" else 140.0
    assert candidate["start_seconds"] <= played + 1.0
    assert candidate["end_seconds"] >= played + 10.0


def test_a_released_recording_is_subtracted_and_not_offered(series, tmp_path):
    episodes, reference = series
    third = _episode(tmp_path / "ep3.wav", 200.0, [(60.0, _music(42, 20.0))], seed=3)
    both = {**episodes, "ep3": third}

    without = series_cue_run.read_result(series_cue_run.run_series_cues(
        both, _recipe(), tmp_path / "without", rank_by_mutual_explained=False))
    with_reference = series_cue_run.read_result(series_cue_run.run_series_cues(
        both, _recipe(), tmp_path / "with", reference_paths=[reference],
        cache_dir=tmp_path / "cache", rank_by_mutual_explained=False))

    assert without["span_count"] > with_reference["span_count"]
    assert with_reference["released_group_count"] >= 1
    assert with_reference["reference_count"] == 1


def test_without_measurements_the_guard_is_recorded_as_not_applied(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False))
    assert record["guard_state"] == series_cues.GUARD_NOT_MEASURED
    assert record["held_count"] == record["span_count"]
    assert record["candidates"][0]["guard_reason"] is None


def test_measurements_reject_a_candidate_without_deleting_its_audio(series, tmp_path):
    episodes, _ = series
    plan_path = series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "plan", rank_by_mutual_explained=False)
    key = series_cue_run.read_result(plan_path)["candidates"][0]["recording_key"]

    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "guarded", rank_by_mutual_explained=False,
        measurements=[{"recording_key": key, "speech_db": -2.0, "music_db": -5.2,
                       "effects_db": -16.9}]))

    assert record["guard_state"] == series_cues.GUARD_APPLIED
    assert record["held_count"] == 0
    assert record["guard_rejected_count"] == 1
    [candidate] = record["candidates"]
    assert candidate["guard_reason"] == content_guard.REJECT_SPEECH
    # 落とした候補の音声も来歴のために残す。削除は別の判断。
    assert (tmp_path / "guarded" / series_cue_run.CANDIDATE_DIRNAME / candidate["file"]).is_file()


def test_mutual_ranking_chooses_an_occurrence_and_is_recorded(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=True))
    assert record["occurrence_ranking_basis"] == "mutual_occurrence"
    assert record["span_count"] == 1


def test_one_episode_is_not_a_series(tmp_path):
    episode = _episode(tmp_path / "ep1.wav", 60.0, [(10.0, _music(41, 20.0))], seed=1)
    with pytest.raises(series_cue_run.SeriesCueRunError):
        series_cue_run.run_series_cues({"ep1": episode}, _recipe(), tmp_path / "run")


def test_a_missing_input_is_refused(tmp_path):
    with pytest.raises(series_cue_run.SeriesCueRunError):
        series_cue_run.run_series_cues(
            {"ep1": tmp_path / "missing.wav", "ep2": tmp_path / "missing2.wav"},
            _recipe(), tmp_path / "run")


def test_the_record_carries_the_recipe_of_every_stage(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False))
    assert record["recipe_id"].startswith("bgm-series-cues-v1-")
    assert set(record["stage_ids"]) == {"reuse", "quality", "selection", "guard"}
    assert record["matching_settings_id"]
    assert set(record["episodes"]) == set(episodes)
    assert all(item["sha256"] for item in record["episodes"].values())


def test_a_changed_candidate_file_is_refused_on_read(series, tmp_path):
    episodes, _ = series
    record_path = series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    target = tmp_path / "run" / series_cue_run.CANDIDATE_DIRNAME / record["candidates"][0]["file"]
    target.write_bytes(b"not audio")
    with pytest.raises(series_cue_run.SeriesCueRunError):
        series_cue_run.read_result(record_path)


def test_a_record_that_allows_promotion_is_refused(tmp_path):
    path = tmp_path / "series-cues.json"
    path.write_text(json.dumps({"type": "bgm_series_cue_run", "publication_state": "not_published",
                                "promotion_allowed": True, "candidates": []}), encoding="utf-8")
    with pytest.raises(series_cue_run.SeriesCueRunError):
        series_cue_run.read_result(path)


def test_a_container_the_audio_library_cannot_open_is_decoded_like_any_input(series, tmp_path):
    """実際の素材はmkv等で届く。製品と同じ入力準備を通して切り出せる。"""
    from tests.test_audio_input import transcode

    episodes = {}
    for label, source in series[0].items():
        target = tmp_path / f"{label}.mkv"
        transcode(source, target, codec="libopus")
        with pytest.raises((RuntimeError, OSError, sf.LibsndfileError)):
            sf.info(str(target))  # 音声ライブラリでは直接開けない
        episodes[label] = target

    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False))

    assert record["span_count"] == 1
    saved = Path(record["candidates"][0]["artifact"]["path"])
    assert saved.is_file() and sf.info(str(saved)).channels == 2
    # 記録が指す原本は渡された素材そのもの。復号した一時ファイルではない。
    assert Path(record["candidates"][0]["source"]["path"]).suffix == ".mkv"


def _stack_recipe():
    from lss.bgm import occurrence_stack
    return occurrence_stack.OccurrenceStackRecipe(
        fir_taps=17, block_seconds=0.25, combine=occurrence_stack.COMBINE_JOINT)


def test_the_music_of_a_candidate_is_estimated_from_its_other_occurrences(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False,
        stack_recipe=_stack_recipe()))

    assert record["music_estimate_count"] == 1
    assert record["music_estimate_recipe_id"] == _stack_recipe().recipe_id
    [candidate] = record["candidates"]
    estimate = candidate["music_estimate"]
    assert estimate is not None
    saved = Path(estimate["path"])
    assert saved.is_file() and saved.parent.name == series_cue_run.MUSIC_DIRNAME
    assert estimate["publication_state"] == series_cue_run.PUBLICATION_STATE
    assert estimate["occurrence_count"] == 2
    assert sf.info(str(saved)).frames == sf.info(
        str(Path(candidate["artifact"]["path"]))).frames


def test_the_estimate_removes_more_of_the_overlay_than_the_raw_span(series, tmp_path):
    """推定は、原本の区間そのままより台詞・雑音が減っていること。"""
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False,
        stack_recipe=_stack_recipe()))
    [candidate] = record["candidates"]
    raw, _ = sf.read(str(Path(candidate["artifact"]["path"])), always_2d=True, dtype="float64")
    music, _ = sf.read(str(Path(candidate["music_estimate"]["path"])),
                       always_2d=True, dtype="float64")
    # 上に乗っていたものは残差へ落ちるので、推定のエネルギーは元より小さい
    assert np.sum(music ** 2) < np.sum(raw ** 2)
    assert np.sum(music ** 2) > 0.0


def test_without_a_stack_recipe_no_music_is_estimated(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False))
    assert record["music_estimate_count"] == 0
    assert record["music_estimate_recipe_id"] is None
    assert record["candidates"][0]["music_estimate"] is None
    assert not (tmp_path / "run" / series_cue_run.MUSIC_DIRNAME).exists()


def test_the_record_says_whether_the_estimate_can_be_trusted(series, tmp_path):
    from lss.bgm import occurrence_stack

    episodes, _ = series
    recipe = occurrence_stack.OccurrenceStackRecipe(
        fir_taps=17, block_seconds=0.25, combine=occurrence_stack.COMBINE_JOINT,
        minimum_energy_share=0.99)
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False,
        stack_recipe=recipe))
    estimate = record["candidates"][0]["music_estimate"]
    assert 0.0 < estimate["energy_share"] < 0.99
    assert estimate["meets_minimum_energy_share"] is False
    assert record["music_estimate_count"] == 1
    assert record["music_estimate_trusted_count"] == 0


def test_an_untrusted_estimate_is_judged_on_the_occurrence_span(series, tmp_path):
    from lss.bgm import occurrence_stack, series_cues as cues

    episodes, _ = series
    plan_path = series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "plan", rank_by_mutual_explained=False)
    key = series_cue_run.read_result(plan_path)["candidates"][0]["recording_key"]
    # 推定は必ず下限に届かない宣言にして、原本の区間で判定させる。
    strict = occurrence_stack.OccurrenceStackRecipe(
        fir_taps=17, block_seconds=0.25, combine=occurrence_stack.COMBINE_JOINT,
        minimum_energy_share=0.99)
    recipe = cues.SeriesCueRecipe(
        reuse=_recipe().reuse, quality=_recipe().quality, selection=_recipe().selection,
        guard=_recipe().guard, guard_target=cues.GUARD_TARGET_ESTIMATE)
    measurements = [
        {"recording_key": key, "target": cues.GUARD_TARGET_ESTIMATE,
         "speech_db": -1.0, "music_db": -8.0, "effects_db": -30.0},
        {"recording_key": key, "target": cues.GUARD_TARGET_OCCURRENCE,
         "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0},
    ]
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, recipe, tmp_path / "run", rank_by_mutual_explained=False,
        stack_recipe=strict, measurements=measurements))

    assert record["untrusted_estimate_recording_keys"] == [key]
    assert record["guard_targets_used"][key] == cues.GUARD_TARGET_OCCURRENCE
    assert record["held_count"] == 1  # 推定で判定していたら落ちていた
    assert record["music_estimate_trusted_count"] == 0


def _within_recipe(**kwargs):
    from lss.bgm import series_cues as cues
    base = _recipe(**kwargs)
    return cues.SeriesCueRecipe(
        reuse=base.reuse, quality=base.quality, selection=base.selection, guard=base.guard,
        guard_target=base.guard_target, include_within_episode=True)


def test_a_recording_repeated_inside_one_episode_is_found(tmp_path):
    """1話しか無くても、同じ録音が2回鳴っていれば候補になる。"""
    cue = _music(41, 20.0)
    single = _episode(tmp_path / "ep1.wav", 260.0,
                      [(30.0, cue), (150.0, cue)], seed=1)

    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        {"ep1": single}, _within_recipe(), tmp_path / "run",
        rank_by_mutual_explained=False))

    assert record["include_within_episode"] is True
    assert record["within_episode_pairs"] >= 1
    assert record["span_count"] == 1
    [candidate] = record["candidates"]
    assert candidate["source_id"] == "ep1"
    assert candidate["occurrence_count"] == 2
    assert candidate["episode_count"] == 1


def test_one_episode_without_the_declaration_is_refused(tmp_path):
    single = _episode(tmp_path / "ep1.wav", 260.0,
                      [(30.0, _music(41, 20.0)), (150.0, _music(41, 20.0))], seed=1)
    with pytest.raises(series_cue_run.SeriesCueRunError):
        series_cue_run.run_series_cues({"ep1": single}, _recipe(), tmp_path / "run")


def test_within_episode_recurrence_adds_to_the_cross_episode_route(series, tmp_path):
    episodes, _ = series
    without = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "without", rank_by_mutual_explained=False))
    with_within = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _within_recipe(), tmp_path / "with", rank_by_mutual_explained=False))
    assert without["include_within_episode"] is False
    assert without["within_episode_pairs"] == 0
    assert with_within["span_count"] >= without["span_count"]


def test_the_declaration_changes_the_route_identifier():
    assert _within_recipe().recipe_id != _recipe().recipe_id


def test_a_candidate_carries_its_marks(series, tmp_path):
    from lss.bgm import occurrence_stack, series_cues as cues

    episodes, _ = series
    base = _recipe()
    # 必ず印が付く宣言と、決して付かない宣言で、同じ候補の印だけが変わることを見る。
    marking = cues.SeriesCueRecipe(
        reuse=base.reuse, quality=base.quality, selection=base.selection, guard=base.guard,
        stock_effect_explained=0.01, stock_effect_energy_share=0.01)
    never = cues.SeriesCueRecipe(
        reuse=base.reuse, quality=base.quality, selection=base.selection, guard=base.guard,
        stock_effect_explained=1.0, stock_effect_energy_share=1.0)
    stack_recipe = occurrence_stack.OccurrenceStackRecipe(
        fir_taps=17, block_seconds=0.25, combine=occurrence_stack.COMBINE_JOINT)

    marked = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, marking, tmp_path / "marked", rank_by_mutual_explained=False,
        stack_recipe=stack_recipe))
    plain = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, never, tmp_path / "plain", rank_by_mutual_explained=False,
        stack_recipe=stack_recipe))

    assert marked["marked_stock_effect_count"] == marked["span_count"]
    assert marked["candidates"][0]["marks"] == [cues.MARK_STOCK_EFFECT]
    assert plain["marked_stock_effect_count"] == 0
    assert plain["candidates"][0]["marks"] == []
    # 印は落とす判断ではない。どちらも保留されている。
    assert marked["held_count"] == plain["held_count"] == marked["span_count"]


def test_without_an_estimate_there_is_no_mark(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False))
    assert record["marked_stock_effect_count"] == 0
    assert record["candidates"][0]["marks"] == []


def _comparable(record):
    """時刻と出力先を除いた、経路の結論。"""
    return [{key: value for key, value in item.items() if key not in ("artifact", "file")}
            for item in record["candidates"]]


def test_a_rerun_reuses_the_matches_and_reaches_the_same_candidates(series, tmp_path, monkeypatch):
    episodes, _ = series
    cache = tmp_path / "cache"
    first = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "first", cache_dir=cache, rank_by_mutual_explained=False))
    assert first["match_cache"] == {"used": True, "reused": 0, "computed": 1}

    def refuse(*args, **kwargs):
        raise AssertionError("保存した照合があるのに照合し直した")

    monkeypatch.setattr(series_cue_run.identity, "shared_recordings", refuse)
    second = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "second", cache_dir=cache, rank_by_mutual_explained=False))
    assert second["match_cache"] == {"used": True, "reused": 1, "computed": 0}
    assert _comparable(second) == _comparable(first)
    assert second["recipe_id"] == first["recipe_id"]


def test_adding_an_episode_only_matches_the_new_pairs(series, tmp_path):
    episodes, _ = series
    cache = tmp_path / "cache"
    series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "two", cache_dir=cache, rank_by_mutual_explained=False)
    third = _episode(tmp_path / "ep3.wav", 200.0, [(60.0, _music(41, 20.0))], seed=3)
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        {**episodes, "ep3": third}, _recipe(), tmp_path / "three", cache_dir=cache,
        rank_by_mutual_explained=False))
    assert record["match_cache"] == {"used": True, "reused": 1, "computed": 2}


def test_without_a_cache_directory_nothing_is_reused(series, tmp_path):
    episodes, _ = series
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "run", rank_by_mutual_explained=False))
    assert record["match_cache"] == {"used": False, "reused": 0, "computed": 1}


def test_a_broken_saved_match_is_not_used(series, tmp_path):
    episodes, _ = series
    cache = tmp_path / "cache"
    first = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "first", cache_dir=cache, rank_by_mutual_explained=False))
    [saved] = list(cache.glob("series-matches-*.json"))
    saved.write_text("{", encoding="utf-8")
    second = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "second", cache_dir=cache, rank_by_mutual_explained=False))
    assert second["match_cache"]["computed"] == 1
    assert _comparable(second) == _comparable(first)


def test_a_changed_matching_setting_does_not_reuse_old_matches(series, tmp_path, monkeypatch):
    episodes, _ = series
    cache = tmp_path / "cache"
    series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "first", cache_dir=cache, rank_by_mutual_explained=False)
    monkeypatch.setattr(series_cue_run.identity, "matching_settings_id", lambda: "changed")
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "second", cache_dir=cache, rank_by_mutual_explained=False))
    assert record["match_cache"] == {"used": True, "reused": 0, "computed": 1}


def test_a_changed_input_signal_does_not_reuse_old_matches(series, tmp_path):
    episodes, _ = series
    cache = tmp_path / "cache"
    series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "first", cache_dir=cache, rank_by_mutual_explained=False)
    _episode(episodes["ep2"], 200.0, [(100.0, _music(41, 20.0))], seed=5)
    record = series_cue_run.read_result(series_cue_run.run_series_cues(
        episodes, _recipe(), tmp_path / "second", cache_dir=cache, rank_by_mutual_explained=False))
    assert record["match_cache"] == {"used": True, "reused": 0, "computed": 1}
