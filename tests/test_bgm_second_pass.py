"""2回目を試す価値と採否は、正解を使わず1回目・2回目の出力だけで決める。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from lss.bgm import second_pass as sp

RATE = 8000
SECONDS = 20


def _write(path: Path, values: np.ndarray) -> Path:
    if values.ndim == 1:
        values = np.stack([values, values], axis=1)
    sf.write(str(path), values.astype(np.float32), RATE, subtype="FLOAT")
    return path


def _tone(level=0.1, hz=330):
    t = np.arange(SECONDS * RATE) / RATE
    return level * np.sin(2 * np.pi * hz * t)


def _clicks(level=0.3, seed=0):
    rng = np.random.default_rng(seed)
    clicks = np.zeros(SECONDS * RATE)
    for second in range(1, SECONDS):
        clicks[second * RATE:second * RATE + RATE // 10] = rng.normal(0, level, RATE // 10)
    return clicks


def _decide(tmp_path, first, second):
    return sp.adoption(_write(tmp_path / "first.wav", first), _write(tmp_path / "second.wav", second),
                       0.0, float(SECONDS), sp.default_recipe())


def test_removing_bursts_of_leaked_effects_is_adopted(tmp_path):
    tone = _tone()
    decision = _decide(tmp_path, tone + _clicks(), tone)
    assert decision["adopt"] is True and decision["reasons"] == []
    assert decision["features"]["removed_concentration"] >= 0.95


def test_weakening_the_music_everywhere_is_not_adopted(tmp_path):
    """歌唱を削る2回目のように、取り除いた成分が時間的に広く、音楽だけの時間も弱まる。"""
    tone = _tone()
    decision = _decide(tmp_path, tone + _clicks(), 0.6 * tone)
    assert decision["adopt"] is False and "music_weakened" in decision["reasons"]
    assert decision["features"]["steady_db"] < -3


def test_removing_part_of_the_spectrum_is_not_adopted(tmp_path):
    low, high = _tone(0.1, 200), _tone(0.1, 2400)
    first = low + high + _clicks()
    second = low + 0.3 * high + (first - low - high - _clicks())  # 高い帯域だけ弱める
    decision = _decide(tmp_path, first, second)
    assert decision["adopt"] is False and "spectral_loss" in decision["reasons"]


def test_changing_stereo_or_timing_is_not_adopted(tmp_path):
    tone = _tone()
    left_only = np.stack([tone, np.zeros_like(tone)], axis=1)
    stereo = _decide(tmp_path, np.stack([tone, tone], axis=1) + _clicks()[:, None], left_only + 0.0)
    assert stereo["adopt"] is False and "stereo_changed" in stereo["reasons"]
    delayed = _decide(tmp_path, tone + _clicks(), np.roll(tone, 40))
    assert delayed["adopt"] is False and "timing_changed" in delayed["reasons"]


def test_removing_nothing_meaningful_keeps_the_first_pass(tmp_path):
    tone = _tone()
    rng = np.random.default_rng(1)
    decision = _decide(tmp_path, tone, tone + rng.normal(0, 1e-5, tone.size))
    assert decision["adopt"] is False and "nothing_meaningful_removed" in decision["reasons"]


def test_music_alone_is_not_worth_a_second_pass(tmp_path):
    silent = np.zeros(SECONDS * RATE)
    stems = {"music": _write(tmp_path / "music.wav", _tone()),
             "speech": _write(tmp_path / "speech.wav", silent),
             "effects": _write(tmp_path / "effects.wav", silent)}
    decision = sp.eligibility(stems, 0.0, float(SECONDS), sp.default_recipe())
    assert decision["eligible"] is False
    assert set(decision["reasons"]) == {"no_competing_non_music", "music_dominant"}


def test_music_competing_with_dialogue_and_effects_is_worth_a_second_pass(tmp_path):
    rng = np.random.default_rng(2)
    stems = {"music": _write(tmp_path / "music.wav", _tone(0.05)),
             "speech": _write(tmp_path / "speech.wav", rng.normal(0, 0.1, SECONDS * RATE)),
             "effects": _write(tmp_path / "effects.wav", _clicks())}
    decision = sp.eligibility(stems, 0.0, float(SECONDS), sp.default_recipe())
    assert decision["eligible"] is True and decision["reasons"] == []


def test_the_recipe_identifier_follows_every_setting():
    assert sp.default_recipe().recipe_id != sp.SecondPassRecipe(minimum_removed_db=-30.0).recipe_id
