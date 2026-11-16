"""原本と推定だけから、取りこぼした大きさを読み取ることを固定する。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.separation.level_recovery import (
    LevelRecoveryError,
    LevelRecoveryRecipe,
    apply_reading,
    read_missing_level,
)

RATE = 44100


def recipe(**changes) -> LevelRecoveryRecipe:
    base = {"minimum_estimate_dbfs": -60.0, "maximum_gain_db": 12.0, "minimum_gain_db": 0.1}
    return LevelRecoveryRecipe(**{**base, **changes})


def _signals(seed: int, seconds: float = 2.0):
    rng = np.random.default_rng(seed)
    size = int(seconds * RATE)
    music = rng.standard_normal((2, size)) * 0.2
    voice = rng.standard_normal((2, size)) * 0.2
    return music, voice


def test_an_attenuated_estimate_is_read_as_missing_level():
    music, voice = _signals(1)
    mixture = music + voice
    estimate = music * 0.5

    read = read_missing_level(mixture, estimate, recipe())

    assert read["state"] == "recovered"
    assert read["reason"] == "residual_explains_a_missing_share_of_the_estimate"
    # 半分になった推定はちょうど6dB戻る。無関係な歌唱の分だけ上振れし得る。
    assert 5.9 <= read["gain_db"] <= 6.5


def test_the_recovered_estimate_is_closer_to_the_true_accompaniment():
    music, voice = _signals(2)
    mixture = music + voice
    estimate = music * 0.5

    read = read_missing_level(mixture, estimate, recipe())
    recovered = apply_reading(estimate, read)

    def error(value):
        return float(np.sum((value - music) ** 2))

    assert error(recovered) < error(estimate)


def test_a_faithful_estimate_is_left_alone():
    music, voice = _signals(3)

    read = read_missing_level(music + voice, music, recipe())

    assert read["gain_db"] <= 1.0


def test_a_silent_estimate_is_never_given_sound():
    music, voice = _signals(4)
    estimate = np.zeros_like(music)

    read = read_missing_level(music + voice, estimate, recipe())
    recovered = apply_reading(estimate, read)

    assert read["state"] == "unchanged"
    assert read["reason"] == "estimate_below_the_declared_floor"
    assert not np.any(recovered)


def test_the_declared_ceiling_limits_how_much_is_given_back():
    music, voice = _signals(5)
    estimate = music * 0.05

    read = read_missing_level(music + voice, estimate, recipe(maximum_gain_db=3.0))

    assert read["reason"] == "gain_reached_the_declared_ceiling"
    assert read["gain_db"] == pytest.approx(3.0)


def test_an_estimate_larger_than_the_mixture_is_not_reduced():
    music, voice = _signals(6)
    estimate = music * 2.0

    read = read_missing_level(music + voice, estimate, recipe())
    recovered = apply_reading(estimate, read)

    assert read["state"] == "unchanged"
    assert read["gain_db"] == 0.0
    assert np.allclose(recovered, estimate)


@pytest.mark.parametrize("changes", [
    {"minimum_estimate_dbfs": 0.0}, {"maximum_gain_db": -1.0},
    {"minimum_gain_db": -0.1}, {"minimum_gain_db": 13.0},
    {"schema_version": 2}, {"method": "other"},
])
def test_an_unsupported_declaration_is_refused(changes):
    with pytest.raises(LevelRecoveryError):
        recipe(**changes).validate()


@pytest.mark.parametrize("mixture,estimate", [
    (np.zeros((2, 10)), np.zeros((2, 5))),
    (np.zeros((0, 10)), np.zeros((0, 10))),
    (np.full((2, 10), np.nan), np.zeros((2, 10))),
])
def test_a_malformed_reading_is_refused(mixture, estimate):
    with pytest.raises(LevelRecoveryError):
        read_missing_level(mixture, estimate, recipe())


def test_a_malformed_reading_cannot_be_applied():
    with pytest.raises(LevelRecoveryError):
        apply_reading(np.zeros((2, 10)), {"state": "recovered", "gain_db": -1.0})
