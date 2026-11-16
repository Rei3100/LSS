"""同じ録音の二つの使用箇所から、余計な音が少ない方を参照なしで読み取る。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm.occurrence_contrast import (
    OccurrenceContrastError,
    OccurrenceContrastRecipe,
    compare,
)
from tests.test_bgm_identity import RATE, _dialogue_and_noise, _music


def recipe(**changes) -> OccurrenceContrastRecipe:
    base = {"window_frames": 2048, "hop_frames": 512, "minimum_seconds": 2.0,
            "minimum_excess_share_margin": 0.02}
    return OccurrenceContrastRecipe(**{**base, **changes})


def _si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    scale = float(np.dot(estimate, target) / np.dot(target, target))
    projection = scale * target
    noise = estimate - projection
    return 10.0 * np.log10(float(np.sum(projection ** 2)) / float(np.sum(noise ** 2)))


def test_the_occurrence_with_less_added_sound_is_named():
    cue = _music(51, 10.0)
    over = _dialogue_and_noise(10.0, seed=2)[: cue.size]
    quiet, loud = cue + over * 0.05, cue + over * 0.6

    read = compare(quiet, loud, RATE, recipe())

    assert read["state"] == "first_has_less_excess"
    assert read["reason"] == "first_adds_less_than_the_common_part"
    assert read["excess_share_difference"] < 0


def test_the_reading_agrees_with_the_true_distance_from_the_recording():
    cue = _music(52, 10.0)
    over = _dialogue_and_noise(10.0, seed=3)[: cue.size]
    quiet, loud = cue + over * 0.05, cue + over * 0.6

    read = compare(loud, quiet, RATE, recipe())

    assert read["state"] == "second_has_less_excess"
    # 正解の録音との距離でも、静かな方が良い。
    assert _si_sdr(quiet, cue) > _si_sdr(loud, cue)


def test_two_equally_contaminated_occurrences_are_not_ranked():
    cue = _music(53, 10.0)
    over = _dialogue_and_noise(10.0, seed=4)[: cue.size]

    read = compare(cue + over * 0.3, cue + over * 0.3, RATE, recipe())

    assert read["state"] == "unresolved"
    assert read["reason"] == "excess_difference_below_margin"


def test_a_declared_margin_can_refuse_a_small_difference():
    cue = _music(54, 10.0)
    over = _dialogue_and_noise(10.0, seed=5)[: cue.size]
    pair = (cue + over * 0.05, cue + over * 0.6)

    assert compare(*pair, RATE, recipe())["state"] != "unresolved"
    assert compare(*pair, RATE, recipe(minimum_excess_share_margin=0.9))["state"] == "unresolved"


def test_stereo_input_is_compared_as_one_signal():
    cue = _music(55, 10.0)
    over = _dialogue_and_noise(10.0, seed=6)[: cue.size]
    quiet = np.stack([cue + over * 0.05, cue + over * 0.05])
    loud = np.stack([cue + over * 0.6, cue + over * 0.6])

    assert compare(quiet, loud, RATE, recipe())["state"] == "first_has_less_excess"


def test_only_the_common_length_is_compared():
    cue = _music(56, 10.0)
    over = _dialogue_and_noise(10.0, seed=7)[: cue.size]
    read = compare(cue + over * 0.05, (cue + over * 0.6)[: 6 * RATE], RATE, recipe())

    assert read["compared_seconds"] == pytest.approx(6.0, abs=0.01)
    assert read["state"] == "first_has_less_excess"


@pytest.mark.parametrize("changes", [
    {"window_frames": 0}, {"hop_frames": 0}, {"hop_frames": 4096},
    {"minimum_seconds": 0.0}, {"minimum_excess_share_margin": 0.0},
    {"schema_version": 2}, {"method": "other"},
])
def test_an_unsupported_declaration_is_refused(changes):
    with pytest.raises(OccurrenceContrastError):
        recipe(**changes).validate()


@pytest.mark.parametrize("first,second,rate", [
    (np.zeros(0), np.zeros(RATE), RATE),
    (np.zeros(RATE), np.zeros(RATE), 0),
    (np.zeros(RATE), np.zeros(RATE // 2), RATE),
    (np.full(3 * RATE, np.nan), np.zeros(3 * RATE), RATE),
    (np.zeros((2, 2, 3)), np.zeros(3 * RATE), RATE),
])
def test_a_malformed_comparison_is_refused(first, second, rate):
    with pytest.raises(OccurrenceContrastError):
        compare(first, second, rate, recipe())
