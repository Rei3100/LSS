"""候補の中身から、繰り返す台詞・効果音を落とす関門。音声は使わない。"""
from __future__ import annotations

import pytest

from lss.bgm import content_guard


def _recipe(speech=0.0, effects=0.0):
    return content_guard.ContentGuardRecipe(
        music_over_speech_db=speech, music_over_effects_db=effects)


def _row(name, speech, music, effects):
    return {"name": name, "speech_db": speech, "music_db": music, "effects_db": effects}


def test_recipe_id_depends_on_the_declaration():
    assert _recipe().recipe_id == _recipe().recipe_id
    assert _recipe().recipe_id != _recipe(speech=3.0).recipe_id
    assert _recipe().recipe_id.startswith("bgm-content-guard-v1-")


@pytest.mark.parametrize("kwargs", [{"speech": float("nan")}, {"effects": 500.0}])
def test_recipe_rejects_values_outside_the_measurement(kwargs):
    with pytest.raises(content_guard.ContentGuardError):
        _recipe(**kwargs).validate()


def test_recipe_rejects_an_unknown_schema():
    recipe = content_guard.ContentGuardRecipe(
        music_over_speech_db=0.0, music_over_effects_db=0.0, method="other")
    with pytest.raises(content_guard.ContentGuardError):
        recipe.validate()


def test_music_over_both_is_kept():
    result = content_guard.screen_candidates([_row("a", -20.0, -1.0, -30.0)], _recipe())
    assert result["kept_count"] == 1
    assert result["rejected_count"] == 0
    assert result["kept"][0]["state"] == content_guard.KEEP
    assert result["kept"][0]["music_minus_speech_db"] == 19.0
    assert result["kept"][0]["music_minus_effects_db"] == 29.0


def test_recurring_dialogue_is_rejected_as_speech():
    # 実測: 決め台詞の再演は music−speech が負になる（−3.1 / −3.8 / −8.9dB）
    result = content_guard.screen_candidates([_row("a", -2.0, -5.2, -16.9)], _recipe())
    assert result["rejected"] == [{"name": "a", "music_minus_speech_db": -3.2,
                                   "music_minus_effects_db": 11.7,
                                   "reason": content_guard.REJECT_SPEECH}]


def test_recurring_stock_effects_are_rejected_as_effects():
    # 実測: 既製の効果音の再演は music−effects が負になる（−7.0 / −1.4 / −0.5dB）
    result = content_guard.screen_candidates([_row("a", -47.8, -8.3, -1.3)], _recipe())
    assert result["rejected"][0]["reason"] == content_guard.REJECT_EFFECTS


def test_speech_is_reported_first_when_both_fail():
    result = content_guard.screen_candidates([_row("a", -1.0, -6.0, -2.0)], _recipe())
    assert result["rejected"][0]["reason"] == content_guard.REJECT_SPEECH


def test_released_controls_all_pass_at_zero_margin():
    # 既発売録音に一致する対照8件の実測値。確実な音楽を1件も落とさない
    controls = [(-46.8, 0.0, -32.3), (-39.0, 0.0, -25.2), (-39.1, 0.0, -14.1),
                (-7.2, 0.0, -19.7), (-28.6, 0.0, -6.8), (-42.2, 0.0, -1.6),
                (-15.3, 0.0, -14.4), (-35.7, 0.0, -2.7)]
    rows = [_row(f"control-{index}", *values) for index, values in enumerate(controls)]
    result = content_guard.screen_candidates(rows, _recipe())
    assert result["kept_count"] == len(controls)


def test_a_margin_can_be_required_above_zero():
    rows = [_row("a", -2.0, 0.0, -30.0)]
    assert content_guard.screen_candidates(rows, _recipe())["kept_count"] == 1
    assert content_guard.screen_candidates(rows, _recipe(speech=3.0))["rejected_count"] == 1


def test_order_is_preserved_and_candidates_are_not_rewritten():
    rows = [_row("a", -20.0, -1.0, -30.0), _row("b", -1.0, -6.0, -30.0),
            _row("c", -20.0, -2.0, -30.0)]
    result = content_guard.screen_candidates(rows, _recipe())
    assert [item["name"] for item in result["kept"]] == ["a", "c"]
    assert [item["name"] for item in result["rejected"]] == ["b"]
    assert rows[0] == {"name": "a", "speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0}


def test_a_recording_key_can_stand_in_for_the_name():
    rows = [{"recording_key": "series-recurrence-001", "speech_db": -20.0,
             "music_db": -1.0, "effects_db": -30.0}]
    result = content_guard.screen_candidates(rows, _recipe())
    assert result["kept"][0]["name"] == "series-recurrence-001"


@pytest.mark.parametrize("row", [
    {"speech_db": -20.0, "music_db": -1.0, "effects_db": -30.0},
    {"name": "a", "music_db": -1.0, "effects_db": -30.0},
    {"name": "a", "speech_db": -20.0, "music_db": None, "effects_db": -30.0},
    {"name": "a", "speech_db": -20.0, "music_db": True, "effects_db": -30.0},
])
def test_incomplete_measurements_are_refused(row):
    with pytest.raises(content_guard.ContentGuardError):
        content_guard.screen_candidates([row], _recipe())


def test_a_non_object_measurement_is_refused():
    with pytest.raises(content_guard.ContentGuardError):
        content_guard.screen_candidates(["a"], _recipe())


def test_the_result_carries_the_analyzer_role_and_its_boundary():
    result = content_guard.screen_candidates([], _recipe())
    assert result["analyzer_role"] == content_guard.ANALYZER_ROLE
    assert result["recipe_id"].startswith("bgm-content-guard-v1-")
    assert any("欲しい劇伴かどうかは判定しない" in line for line in result["claim_boundary"])
