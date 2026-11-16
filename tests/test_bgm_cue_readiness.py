"""公開前のcueを参照なしで検査する。通ることを「良い劇伴」と言い換えない。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm import cue_readiness as readiness

RATE = 44100


def noise(seconds: float = 3.0, amplitude: float = 0.1, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = int(RATE * seconds)
    return np.stack([rng.standard_normal(frames) * amplitude,
                     rng.standard_normal(frames) * amplitude])


def recipe(**changes) -> readiness.CueReadinessRecipe:
    base = {"required_channels": 2, "minimum_loudness_lufs": -45.0,
            "maximum_true_peak_dbtp": -1.0, "maximum_clipped_samples": 0,
            "minimum_spectral_edge_hz": 8000.0, "spectral_edge_drop_db": 60.0,
            "minimum_side_to_mid_db": -30.0}
    return readiness.CueReadinessRecipe(**{**base, **changes})


def test_a_healthy_stereo_cue_passes_and_says_what_it_did_not_check():
    result = readiness.evaluate(readiness.measure(noise(), RATE), recipe())
    assert result["passed"] is True and result["failures"] == []
    assert result["recipe_id"] == recipe().recipe_id
    assert "loudness_lufs" in result["checked"] and "side_to_mid_db" in result["checked"]
    # 通っても意味的な正しさや分離の質は何も言っていない。
    assert any("L4" in item for item in result["not_checked"])
    assert any("L2" in item for item in result["not_checked"])


def test_measurements_carry_no_judgement_and_repeat_exactly():
    first = readiness.measure(noise(), RATE)
    assert readiness.measure(noise(), RATE) == first
    assert "passed" not in first and "failures" not in first
    assert first["channels"] == 2 and first["seconds"] == 3.0
    assert first["finite"] is True and first["silent"] is False
    assert first["bandwidth_hz_p99"] >= first["bandwidth_hz_p95"]
    assert first["spectral_edge_hz_60db"] >= first["spectral_edge_hz_40db"]


def test_non_finite_samples_are_reported_without_scanning_loudness():
    broken = noise()
    broken[0][10] = np.nan
    measured = readiness.measure(broken, RATE)
    assert measured["finite"] is False and measured["loudness_lufs"] is None
    failures = [item["check"] for item in readiness.evaluate(measured, recipe())["failures"]]
    assert "non_finite_samples" in failures and "loudness_lufs" in failures


def test_a_silent_cue_and_a_silent_channel_are_both_named():
    silence = np.zeros((2, RATE * 3))
    failures = [item["check"] for item in
                readiness.evaluate(readiness.measure(silence, RATE), recipe())["failures"]]
    assert "silent_cue" in failures and "silent_channel" in failures
    half = noise()
    half[1] = 0.0
    half_failures = [item["check"] for item in
                     readiness.evaluate(readiness.measure(half, RATE), recipe())["failures"]]
    assert "silent_channel" in half_failures and "silent_cue" not in half_failures


def test_a_cue_collapsed_to_one_channel_is_not_published_as_stereo():
    values = noise()
    collapsed = np.stack([values[0], values[0]])
    measured = readiness.measure(collapsed, RATE)
    assert measured["channel_correlation"] == pytest.approx(1.0)
    failures = [item["check"] for item in readiness.evaluate(measured, recipe())["failures"]]
    assert "side_to_mid_db" in failures
    # 左右の広がりを要求しないRecipeでは、同じcueが通る。
    assert readiness.evaluate(measured, recipe(minimum_side_to_mid_db=None))["passed"] is True


def test_quiet_loud_clipped_and_narrow_band_cues_are_each_named():
    quiet = [item["check"] for item in
             readiness.evaluate(readiness.measure(noise(amplitude=1e-4), RATE), recipe())["failures"]]
    assert quiet == ["loudness_lufs"]
    loud = readiness.measure(np.clip(noise(amplitude=4.0), -1.0, 1.0), RATE)
    loud_failures = [item["check"] for item in readiness.evaluate(loud, recipe())["failures"]]
    assert "true_peak_dbtp" in loud_failures and "clipped_samples" in loud_failures
    frames = RATE * 3
    tone = np.sin(2 * np.pi * 440 * np.arange(frames) / RATE) * 0.2
    narrow = [item["check"] for item in
              readiness.evaluate(readiness.measure(np.stack([tone, tone * 0.999]), RATE),
                                 recipe())["failures"]]
    assert "spectral_edge_hz_60db" in narrow


def test_a_mono_cue_is_measured_without_a_stereo_judgement():
    frames = RATE * 3
    rng = np.random.default_rng(5)
    mono = rng.standard_normal((1, frames)) * 0.1
    measured = readiness.measure(mono, RATE)
    assert measured["side_to_mid_db"] is None and measured["channel_correlation"] is None
    assert readiness.evaluate(measured, recipe(required_channels=1,
                                               minimum_side_to_mid_db=None))["passed"] is True
    failures = [item["check"] for item in readiness.evaluate(measured, recipe())["failures"]]
    assert "channel_count" in failures


@pytest.mark.parametrize("changes", [
    {"required_channels": 0}, {"required_channels": 3}, {"required_channels": 2.0},
    {"maximum_clipped_samples": -1}, {"maximum_clipped_samples": 0.0},
    {"minimum_loudness_lufs": float("nan")}, {"maximum_true_peak_dbtp": float("inf")},
    {"minimum_spectral_edge_hz": 0.0}, {"spectral_edge_drop_db": 50.0},
    {"minimum_side_to_mid_db": float("nan")},
    {"required_channels": 1}, {"schema_version": 2}, {"method": "guessed"},
])
def test_an_incomplete_declaration_can_never_judge_a_cue(changes):
    broken = recipe(**changes)
    with pytest.raises(readiness.CueReadinessError):
        broken.to_dict()
    with pytest.raises(readiness.CueReadinessError):
        readiness.evaluate(readiness.measure(noise(), RATE), broken)


@pytest.mark.parametrize("samples", [np.zeros((0, 10)), np.zeros((2, 0)), np.zeros(10)])
def test_audio_that_is_not_a_channel_timeline_is_refused(samples):
    with pytest.raises(readiness.CueReadinessError):
        readiness.measure(samples, RATE)


def test_measurements_from_somewhere_else_are_refused():
    with pytest.raises(readiness.CueReadinessError):
        readiness.evaluate({"loudness_lufs": -20.0}, recipe())
    with pytest.raises(readiness.CueReadinessError):
        readiness.measure(noise(), 0)


def test_the_edge_finds_band_limiting_where_the_energy_share_only_finds_bass():
    """低音寄りの劇伴と、帯域を切り落とされた出力を取り違えない。"""
    frames = RATE * 4
    rng = np.random.default_rng(11)
    wide = rng.standard_normal((2, frames)) * 0.1
    # 低域へ強く寄せるが帯域は切らない（実際の劇伴に近い形）。
    spectrum = np.fft.rfft(wide, axis=1)
    freqs = np.fft.rfftfreq(frames, 1.0 / RATE)
    tilted = np.fft.irfft(spectrum / (1.0 + freqs / 200.0), n=frames, axis=1)
    tilted = tilted / np.max(np.abs(tilted)) * 0.3
    # 同じ素材から12kHz以上を落とした出力。
    limited = np.fft.irfft(np.fft.rfft(tilted, axis=1) * (freqs < 12000), n=frames, axis=1)

    bass = readiness.measure(tilted, RATE)
    cut = readiness.measure(limited, RATE)
    # 累積比はどちらも低く出る。ここだけを見ると両者を区別できない。
    assert bass["bandwidth_hz_p95"] < 5000 and cut["bandwidth_hz_p95"] < 5000
    assert bass["spectral_edge_hz_60db"] > 15000 and cut["spectral_edge_hz_60db"] < 13000

    gate = recipe(minimum_spectral_edge_hz=15000.0, minimum_loudness_lufs=-60.0,
                  minimum_side_to_mid_db=None)
    assert readiness.evaluate(bass, gate)["passed"] is True
    assert [item["check"] for item in readiness.evaluate(cut, gate)["failures"]] == [
        "spectral_edge_hz_60db"]


def test_side_only_music_retains_its_bandwidth_and_lowpass_is_still_rejected():
    frames = RATE * 3
    time = np.arange(frames) / RATE
    low = 0.12 * np.sin(2 * np.pi * 440 * time)
    high = 0.08 * np.sin(2 * np.pi * 16000 * time)
    gate = recipe(minimum_side_to_mid_db=None)
    for values in (np.stack([high, -high]), np.stack([low + high, low - high])):
        legacy = readiness.measure(values, RATE, method=readiness.LEGACY_METHOD)
        current = readiness.measure(values, RATE)
        assert legacy["spectral_edge_hz_60db"] is None or legacy["spectral_edge_hz_60db"] < 8000
        assert current["bandwidth_hz_p99"] > 15000
        assert current["spectral_edge_hz_60db"] > 15000
        assert readiness.evaluate(current, gate)["passed"]
        assert not readiness.evaluate(legacy, recipe(method=readiness.LEGACY_METHOD,
                                                   minimum_side_to_mid_db=None))["passed"]
    limited = np.stack([low, -low])
    failures = readiness.evaluate(readiness.measure(limited, RATE), gate)["failures"]
    assert [item["check"] for item in failures] == ["spectral_edge_hz_60db"]


def test_spectrum_is_invariant_to_channel_polarity_and_permutation():
    values = noise()
    expected = readiness._bandwidth(values, RATE)
    for transformed in (values[::-1], values * np.array([[-1], [1]]), -values):
        assert readiness._bandwidth(transformed, RATE) == expected


def test_mono_and_duplicated_channels_retain_exact_historical_spectral_values():
    mono = noise()[:1]
    expected = readiness._bandwidth(mono, RATE, method=readiness.LEGACY_METHOD)
    assert readiness._bandwidth(mono, RATE) == expected
    assert readiness._bandwidth(np.repeat(mono, 2, axis=0), RATE) == expected


def test_historical_recipe_keeps_its_identity_and_requires_matching_measurement():
    old = recipe(method=readiness.LEGACY_METHOD)
    assert old.recipe_id == "bgm-cue-readiness-v1-ebff8ddd1555109b"
    assert old.recipe_id != recipe().recipe_id
    assert old.to_dict()["method"] == "cue-publication-readiness-v1"
    measured = readiness.measure(noise(), RATE, method=old.method)
    assert readiness.evaluate(measured, old)["method"] == old.method
    with pytest.raises(readiness.CueReadinessError):
        readiness.evaluate(measured, recipe())
    with pytest.raises(readiness.CueReadinessError):
        readiness.evaluate(readiness.measure(noise(), RATE), old)
    with pytest.raises(readiness.CueReadinessError):
        readiness.measure(noise(), RATE, method="guessed")
