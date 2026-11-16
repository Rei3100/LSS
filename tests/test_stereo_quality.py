import numpy as np
import pytest

from lss.validation.stereo_quality import stereo_quality


def test_gain_error_is_visible_even_when_si_sdr_is_high():
    reference = np.random.default_rng(2).normal(0, 0.1, (2, 1024))
    quality = stereo_quality(reference * 0.5, reference)
    assert quality["si_sdr_db"]["joint"] > 100
    assert quality["projection_gain_db"]["joint"] == pytest.approx(-6.020599913)
    assert quality["scale_dependent_snr_db"] == pytest.approx(6.020599913)


def test_channel_gain_damage_is_not_hidden_by_joint_projection():
    reference = np.random.default_rng(3).normal(0, 0.1, (2, 1024))
    estimate = reference.copy()
    estimate[1] *= 0.5
    quality = stereo_quality(estimate, reference)
    assert quality["relative_channel_gain_error_db"] == pytest.approx(6.020599913)


def test_dual_mono_has_no_fabricated_side_quality():
    reference = np.ones((2, 1024)) * 0.1
    quality = stereo_quality(reference, reference)
    assert quality["si_sdr_db"]["side"] is None
    assert quality["side_to_mid_balance_error_db"] is None


def test_absent_reference_has_no_target_quality():
    quality = stereo_quality(np.ones((2, 1024)) * 0.1, np.zeros((2, 1024)))
    assert quality["scale_dependent_snr_db"] is None
    assert all(value is None for value in quality["si_sdr_db"].values())


def test_inverted_target_is_not_a_positive_quality_result():
    reference = np.random.default_rng(4).normal(0, 0.1, (2, 1024))
    quality = stereo_quality(-reference, reference)
    assert quality["si_sdr_db"]["joint"] < 0
    assert quality["projection_gain_db"]["joint"] is None


def test_mismatched_timing_and_nonfinite_reference_are_rejected():
    for reference in (np.zeros((2, 9)), np.full((2, 10), np.nan)):
        with pytest.raises(ValueError):
            stereo_quality(np.zeros((2, 10)), reference)


def test_stereo_to_dual_mono_damage_is_explicit_instead_of_unknown():
    reference = np.random.default_rng(5).normal(0, 0.1, (2, 1024))
    estimate = np.repeat(reference.mean(axis=0, keepdims=True), 2, axis=0)
    quality = stereo_quality(estimate, reference)
    assert quality["spatial_component_lost"]["side"] is True
    assert quality["side_to_mid_balance_error_db"] < -100
    assert quality["si_sdr_db"]["side"] < 0


@pytest.mark.parametrize("component", ["left", "right", "mid"])
def test_lost_channel_or_mid_is_not_hidden_as_unmeasurable(component):
    reference = np.random.default_rng(6).normal(0, 0.1, (2, 1024))
    estimate = reference.copy()
    if component == "mid":
        estimate[1] = -estimate[0]
    else:
        estimate[0 if component == "left" else 1] = 0
    quality = stereo_quality(estimate, reference)
    assert quality["spatial_component_lost"][component] is True
