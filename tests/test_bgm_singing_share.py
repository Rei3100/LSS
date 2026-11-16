"""歌唱stemの取り分は観測であって採用judgementではない。どちらとも言えない帯を残す。"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import singing_share

RATE = 44100


def stems(share: float, frames: int = RATE, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """狙った取り分になる歌唱stemと器楽stemを作る。"""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((2, frames))
    unit = base / np.sqrt(np.sum(base ** 2))
    other = rng.standard_normal((2, frames))
    other = other / np.sqrt(np.sum(other ** 2))
    return unit * np.sqrt(share), other * np.sqrt(1.0 - share)


def recipe(**changes) -> singing_share.SingingShareRecipe:
    base = {"maximum_absent_share": 0.02, "minimum_present_share": 0.04,
            "minimum_music_dbfs": -70.0}
    return singing_share.SingingShareRecipe(**{**base, **changes})


@pytest.mark.parametrize("share,state,reason", [
    (0.0, "absent", "below_absent_band"),
    (0.01, "absent", "below_absent_band"),
    (0.03, "unresolved", "inside_undecided_band"),
    (0.05, "present", "above_present_band"),
    (0.9, "present", "above_present_band"),
])
def test_the_band_leaves_the_middle_unresolved(share, state, reason):
    measured = singing_share.measure(*stems(share))
    assert measured["singing_share"] == pytest.approx(share, abs=1e-6)
    decided = singing_share.classify(measured, recipe())
    assert (decided["singing"], decided["reason"]) == (state, reason)
    # 観測であって役割ではない、と結果自身が言う。
    assert decided["is_observation_not_role"] is True


def test_music_without_any_energy_gets_no_singing_judgement():
    silence = np.zeros((2, RATE))
    measured = singing_share.measure(silence, silence)
    assert measured["singing_share"] is None and measured["music_present"] is False
    decided = singing_share.classify(measured, recipe())
    assert (decided["singing"], decided["reason"]) == ("unresolved", "no_music_energy")


def test_measurements_carry_no_thresholds_and_repeat_exactly():
    first = singing_share.measure(*stems(0.3))
    assert singing_share.measure(*stems(0.3)) == first
    assert "singing" not in first and "recipe_id" not in first
    assert first["frames"] == RATE and first["channels"] == 2


def test_files_are_measured_only_when_both_stems_share_a_timeline(tmp_path):
    vocals, instrumental = stems(0.25, frames=RATE * 2)
    vocal_path, instrumental_path = tmp_path / "music_vocals.wav", tmp_path / "music_instrumental.wav"
    sf.write(vocal_path, vocals.T, RATE, subtype="FLOAT")
    sf.write(instrumental_path, instrumental.T, RATE, subtype="FLOAT")
    measured = singing_share.measure_files(vocal_path, instrumental_path)
    assert measured["singing_share"] == pytest.approx(0.25, abs=1e-6)
    assert measured["samplerate"] == RATE and measured["frames"] == RATE * 2
    short = tmp_path / "short.wav"
    sf.write(short, instrumental[:, : RATE].T, RATE, subtype="FLOAT")
    with pytest.raises(singing_share.SingingShareError, match="時間軸"):
        singing_share.measure_files(vocal_path, short)


def test_non_finite_stems_are_refused(tmp_path):
    vocals, instrumental = stems(0.2)
    vocals[0][5] = np.inf
    with pytest.raises(singing_share.SingingShareError, match="有限"):
        singing_share.measure(vocals, instrumental)
    path = tmp_path / "broken.wav"
    sf.write(path, vocals.T, RATE, subtype="FLOAT")
    other = tmp_path / "other.wav"
    sf.write(other, instrumental.T, RATE, subtype="FLOAT")
    with pytest.raises(singing_share.SingingShareError, match="有限"):
        singing_share.measure_files(path, other)


@pytest.mark.parametrize("changes", [
    {"minimum_music_dbfs": 0.0}, {"minimum_music_dbfs": float("-inf")},
    {"minimum_music_dbfs": True},
    {"maximum_absent_share": -0.1}, {"maximum_absent_share": 1.5},
    {"minimum_present_share": float("nan")}, {"minimum_present_share": True},
    {"maximum_absent_share": 0.5, "minimum_present_share": 0.4},
    {"schema_version": 2}, {"method": "guessed"},
])
def test_an_unusable_band_can_never_decide(changes):
    broken = recipe(**changes)
    with pytest.raises(singing_share.SingingShareError):
        broken.to_dict()
    with pytest.raises(singing_share.SingingShareError):
        singing_share.classify(singing_share.measure(*stems(0.5)), broken)


def test_a_single_threshold_band_is_allowed_but_still_named():
    strict = recipe(maximum_absent_share=0.02, minimum_present_share=0.02)
    assert singing_share.classify(singing_share.measure(*stems(0.02)), strict)["singing"] == "absent"
    assert singing_share.classify(singing_share.measure(*stems(0.03)), strict)["singing"] == "present"


def test_observations_from_somewhere_else_are_refused():
    with pytest.raises(singing_share.SingingShareError):
        singing_share.classify({"singing_share": 0.5}, recipe())
    with pytest.raises(singing_share.SingingShareError):
        singing_share.measure(np.zeros((2, 10)), np.zeros((2, 11)))


def test_a_region_with_almost_no_music_gets_no_singing_judgement():
    """音楽がほぼ無い区間では、雑音どうしの比を歌唱ありと読まない。"""
    vocals, instrumental = stems(0.47)
    faint = vocals * 1e-5, instrumental * 1e-5
    measured = singing_share.measure(*faint)
    assert measured["singing_share"] == pytest.approx(0.47, abs=1e-6)
    assert measured["music_rms_dbfs"] < -70.0
    decided = singing_share.classify(measured, recipe())
    assert (decided["singing"], decided["reason"]) == ("unresolved", "music_level_too_low")
    # 同じ比でも、判断できるだけの音楽があれば歌唱ありと言える。
    loud = singing_share.measure(vocals, instrumental)
    assert loud["music_rms_dbfs"] > -70.0
    assert singing_share.classify(loud, recipe())["singing"] == "present"


def test_the_music_level_is_reported_with_every_decision():
    measured = singing_share.measure(*stems(0.5))
    decided = singing_share.classify(measured, recipe())
    assert decided["music_rms_dbfs"] == measured["music_rms_dbfs"]
