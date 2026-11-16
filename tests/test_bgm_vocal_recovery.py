"""効果音の出力へ誤配分された歌唱を足すかは、正解を使わず候補の音楽・効果音・4分類の歌唱だけで決める。"""
from __future__ import annotations

import numpy as np

from lss.bgm import vocal_recovery as vr

RATE = 8000
SECONDS = 6


def _tone(level=0.1, hz=330, seconds=SECONDS):
    t = np.arange(seconds * RATE) / RATE
    return level * np.sin(2 * np.pi * hz * t)


def _stereo(left, right=None):
    return np.stack([left, left if right is None else right], axis=1)


def _decide(music, effects_vocals, music_vocals, mixture_vocals=None):
    """原本の歌唱は、既定では候補の歌唱と効果音の出力の歌唱の和（SFCが分けて入れた同じ歌唱）。"""
    if mixture_vocals is None:
        mixture_vocals = music_vocals + effects_vocals
    return vr.recover(music, effects_vocals, music_vocals, mixture_vocals, RATE, vr.default_recipe())


def test_singing_split_across_channels_is_added_back():
    """SFCが左右を別々に推論し、同じ歌唱の左だけを効果音へ入れた場合。足すと元の歌唱に戻る。"""
    singing = _tone()
    music = _stereo(np.zeros_like(singing), singing)
    effects_vocals = _stereo(singing, np.zeros_like(singing))

    challenger, decision = _decide(music, effects_vocals, music.copy())

    assert decision["adopt"] is True and decision["reasons"] == []
    np.testing.assert_allclose(challenger, _stereo(singing), atol=1e-9)
    assert decision["features"]["singing_level_difference_db"] == 0.0


def test_a_voice_where_the_candidate_has_no_singing_is_not_added():
    silent = np.zeros(SECONDS * RATE)
    music = _stereo(_tone(0.1, 110))
    challenger, decision = _decide(music, _stereo(_tone(0.1, 440)), _stereo(silent))

    assert decision["adopt"] is False
    assert "nothing_meaningful_to_add" in decision["reasons"]
    np.testing.assert_allclose(challenger, music)


def test_only_the_time_with_existing_singing_is_added():
    """候補の歌唱が前半だけにある時、後半の声（台詞など）は足さない。"""
    half = SECONDS * RATE // 2
    singing = _tone()
    music_vocals = singing.copy()
    music_vocals[half:] = 0
    voice = _tone(0.1, 220)
    effects_vocals = np.concatenate([singing[:half], voice[half:]])

    challenger, decision = _decide(_stereo(music_vocals), _stereo(effects_vocals), _stereo(music_vocals))

    added = challenger - _stereo(music_vocals)
    assert np.max(np.abs(added[half + RATE // 5:])) < 1e-9
    np.testing.assert_allclose(added[:half - RATE // 5], _stereo(singing[:half - RATE // 5]), atol=1e-9)
    assert decision["features"]["kept_fraction"] < 0.6


def test_a_sound_that_the_finished_mix_does_not_sing_is_not_added():
    """効果音の出力の「歌唱」が、原本（完成mix）の歌唱に無い時は足さない（器楽の一部を歌唱と誤判定した
    候補の音楽と、同時に鳴る別の音）。"""
    singing = _tone()
    music = _stereo(singing)
    foreign = _stereo(_tone(0.05, 523))

    challenger, decision = _decide(music, foreign, music.copy(), mixture_vocals=music.copy())

    assert decision["adopt"] is False
    assert "nothing_meaningful_to_add" in decision["reasons"] or "disagrees_with_mixture_singing" in decision["reasons"]
    assert decision["features"]["agreeing_fraction"] == 0.0
    np.testing.assert_allclose(challenger, music)


def test_only_the_frames_that_the_finished_mix_sings_are_added():
    half = SECONDS * RATE // 2
    singing = _tone()
    music = _stereo(np.zeros_like(singing), singing)
    effects_vocals = _stereo(singing, np.zeros_like(singing))
    mixture = _stereo(singing)
    mixture[half:, 0] = 0  # 後半は原本の左に歌唱が無い（効果音の出力の左は別の音）

    challenger, decision = _decide(music, effects_vocals, music.copy(), mixture_vocals=mixture)

    added = challenger - music
    assert np.max(np.abs(added[half + RATE // 5:])) < 1e-9
    np.testing.assert_allclose(added[:half - RATE // 5], effects_vocals[:half - RATE // 5], atol=1e-9)
    assert 0.3 < decision["features"]["agreeing_fraction"] < 0.7


def test_a_different_voice_at_the_same_time_is_not_added():
    """候補の「歌唱」と同時に鳴るが、倍音の位置が違う声（叫び等）は、原本の4分類が歌唱と見なしても足さない。"""
    singing = _tone(0.1, 330)
    shout = _tone(0.05, 523)
    music = _stereo(np.zeros_like(singing), singing)
    effects_vocals = _stereo(shout, np.zeros_like(shout))

    _, decision = _decide(music, effects_vocals, music.copy())

    assert decision["adopt"] is False and "different_voice" in decision["reasons"]
    assert decision["features"]["same_voice_similarity"] < 0.2


def test_mostly_outside_the_existing_singing_is_ambiguous():
    tenth = SECONDS * RATE // 10
    music_vocals = np.zeros(SECONDS * RATE)
    music_vocals[:tenth] = _tone()[:tenth]
    effects_vocals = _tone(0.1, 220)
    effects_vocals[:tenth] = music_vocals[:tenth]

    _, decision = _decide(_stereo(music_vocals), _stereo(effects_vocals), _stereo(music_vocals))

    assert decision["adopt"] is False and "mostly_outside_existing_singing" in decision["reasons"]


def test_much_louder_than_the_existing_singing_is_not_added():
    singing = _tone(0.01)
    music = _stereo(singing)
    # gate（候補の歌唱+10dBまで）は通るが、候補の歌唱を+6dB以上超える量
    _, decision = _decide(music, _stereo(3 * singing), music.copy())

    assert decision["adopt"] is False and "exceeds_existing_singing" in decision["reasons"]


def test_negligible_against_the_existing_singing_is_not_added():
    singing = _tone()
    _, decision = _decide(_stereo(singing), _stereo(0.05 * singing), _stereo(singing))

    assert decision["adopt"] is False and "negligible_against_existing_singing" in decision["reasons"]


def test_a_sound_placed_elsewhere_or_delayed_is_not_added():
    singing = _tone()
    music = _stereo(singing)
    # 既にある歌唱は中央、足す成分は左右で逆相（歌唱の左右の相関を大きく下げる）
    opposite = _stereo(0.8 * singing, -0.8 * singing)
    _, anti = _decide(music, opposite, music.copy())
    assert anti["adopt"] is False and "stereo_changed" in anti["reasons"]
    delayed = _stereo(np.roll(0.5 * singing, 40))
    _, late = _decide(music, delayed, music.copy())
    assert late["adopt"] is False and "timing_changed" in late["reasons"]


def test_effects_far_below_the_music_are_not_worth_the_four_stem_analysis():
    music = _stereo(_tone())
    quiet = vr.eligibility(music, 1e-3 * music, vr.default_recipe())
    competing = vr.eligibility(music, 0.3 * music, vr.default_recipe())

    assert quiet["eligible"] is False and quiet["reasons"] == ["effects_negligible"]
    assert competing["eligible"] is True


def test_the_gate_does_not_fade_at_the_cue_edges():
    singing = _stereo(_tone())
    weight, on = vr.gate(singing, singing, RATE, vr.default_recipe())

    assert on.all() and np.all(weight == 1.0)


def test_the_recipe_identifier_follows_every_setting():
    assert vr.default_recipe().recipe_id != vr.VocalRecoveryRecipe(gate_margin_db=12.0).recipe_id
