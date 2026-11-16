"""意味を仮定せず、空間処理の音声保持・無音・退化条件を検証する。"""
from dataclasses import replace

import numpy as np
import pytest

from lss.bgm.refinement import SpatialRefinementRecipe, refine_stereo


def recipe():
    return SpatialRefinementRecipe(n_fft=128, hop_frames=32, covariance_frames=9)


@pytest.mark.parametrize("frames", [1, 91, 128, 517])
def test_clean_target_keeps_level_timing_and_stereo(frames):
    raw = np.random.default_rng(42).normal(0, 0.1, (2, frames))
    result = refine_stereo(raw, {"Music": raw.copy(), "Speech": np.zeros_like(raw)}, recipe())
    assert result.stems["Music"].shape == raw.shape
    np.testing.assert_allclose(result.stems["Music"], raw, atol=2e-6, rtol=2e-4)
    assert np.count_nonzero(result.stems["Speech"]) == 0
    np.testing.assert_array_equal(result.unexplained, np.zeros_like(raw))


@pytest.mark.parametrize("iterations", [1, 2])
def test_silent_target_is_not_populated_from_unexplained_mixture(iterations):
    raw = np.random.default_rng(7).normal(0, 0.1, (2, 513))
    result = refine_stereo(raw, {"Music": np.zeros_like(raw)}, replace(recipe(), iterations=iterations))
    np.testing.assert_array_equal(result.stems["Music"], np.zeros_like(raw))
    np.testing.assert_allclose(result.unexplained, raw, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("raw", [np.zeros((2, 257)), np.ones((2, 257)) * 0.05])
def test_silence_and_dual_mono_are_finite(raw):
    result = refine_stereo(raw, {"Music": raw.copy()}, recipe())
    assert np.isfinite(result.stems["Music"]).all()
    np.testing.assert_allclose(result.stems["Music"], raw, atol=1e-7)


def test_separates_coincident_frequency_with_independent_stereo_positions():
    # 同一周波数の左右別sourceを、center化せず空間情報で保持する。
    t = np.arange(2048)
    left = np.sin(t * 0.12) * (0.15 + 0.08 * np.sin(t * 0.007))
    right = np.sin(t * 0.12 + 0.9) * (0.13 + 0.09 * np.sin(t * 0.011))
    target = np.vstack((left, np.zeros_like(left)))
    interferer = np.vstack((np.zeros_like(right), right))
    raw = target + interferer
    result = refine_stereo(raw, {"Music": target, "Effects": interferer}, recipe())
    np.testing.assert_allclose(result.stems["Music"], target, atol=2e-6)
    np.testing.assert_allclose(result.stems["Effects"], interferer, atol=2e-6)


def test_inputs_are_immutable_and_processing_is_gain_equivariant():
    rng = np.random.default_rng(1)
    raw = rng.normal(0, 0.1, (2, 513))
    target = raw * 0.7
    original = raw.copy(), target.copy()
    first = refine_stereo(raw, {"Music": target}, recipe())
    quiet = refine_stereo(raw * 1e-5, {"Music": target * 1e-5}, recipe())
    np.testing.assert_allclose(quiet.stems["Music"] / 1e-5, first.stems["Music"], atol=1e-10)
    np.testing.assert_array_equal(raw, original[0])
    np.testing.assert_array_equal(target, original[1])


@pytest.mark.parametrize("override", [
    {"iterations": True}, {"covariance_frames": 8}, {"hop_frames": 65},
    {"relative_regularization": float("nan")}, {"n_fft": 127},
    {"residual_policy": "return-to-music"},
])
def test_invalid_recipe_is_rejected(override):
    with pytest.raises(ValueError):
        replace(recipe(), **override).validate()


def test_every_output_affecting_parameter_changes_identity():
    parent = recipe()
    for change in ({"iterations": 2}, {"covariance_frames": 11}, {"n_fft": 256},
                   {"hop_frames": 16}, {"relative_regularization": 1e-5}):
        assert replace(parent, **change).recipe_id != parent.recipe_id


@pytest.mark.parametrize("value", [np.zeros((1, 500)), np.zeros((2, 0)),
                                  np.full((2, 500), np.nan), np.zeros((2, 500), dtype=complex)])
def test_invalid_audio_is_rejected(value):
    with pytest.raises(ValueError):
        refine_stereo(value, {"Music": value}, recipe())


def test_mismatched_source_timeline_is_rejected():
    with pytest.raises(ValueError, match="sample数"):
        refine_stereo(np.zeros((2, 500)), {"Music": np.zeros((2, 499))}, recipe())


@pytest.mark.parametrize("iterations", [1, 2])
def test_blocks_match_full_context_without_boundary_joins(iterations):
    rng = np.random.default_rng(4)
    target = rng.normal(0, 0.08, (2, 5003))
    interference = rng.normal(0, 0.05, target.shape)
    raw = target + interference
    stems = {"Music": target + interference * 0.1, "Effects": interference * 0.9}
    full = refine_stereo(raw, stems, replace(recipe(), iterations=iterations))
    blocked = refine_stereo(raw, stems, replace(recipe(), iterations=iterations, block_frames=512))
    for name in stems:
        np.testing.assert_allclose(blocked.stems[name], full.stems[name], atol=1e-8)
    np.testing.assert_allclose(blocked.unexplained, full.unexplained, atol=1e-8)


def test_refinement_remains_finite_after_huge_power_change():
    rng = np.random.default_rng(3)
    target = rng.normal(0, 0.1, (2, 6001))
    target[:, 2000:] *= 1e-10
    interference = rng.normal(0, 0.1, target.shape)
    interference[:, 2000:] *= 1e-11
    stems = {"Music": target, "Effects": interference, "Speech": np.zeros_like(target)}
    raw = target + interference
    result = refine_stereo(raw, stems, recipe())
    for name in stems:
        assert np.isfinite(result.stems[name]).all()
        assert result.stems[name].shape == raw.shape
    np.testing.assert_array_equal(result.stems["Speech"], np.zeros_like(raw))
