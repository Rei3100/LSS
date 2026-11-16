import numpy as np
import pytest

from lss.validation.instrumental_quality import instrumental_quality


def test_leakage_and_target_gain_are_separate_from_uncorrected_fidelity():
    target, vocals = np.random.default_rng(11).normal(0, 0.1, (2, 2, 4096))
    quality = instrumental_quality(target * 0.5 + vocals * 0.01, target, vocals)
    fit = quality["component_fit"]
    assert fit["target_gain"] == pytest.approx(0.5)
    assert fit["vocals_gain"] == pytest.approx(0.01)
    assert fit["vocals_transfer_db"] == pytest.approx(-40)
    assert quality["fidelity"]["scale_dependent_snr_db"] < 7


def test_absent_target_measures_energy_without_fabricating_target_score():
    vocals = np.random.default_rng(12).normal(0, 0.1, (2, 4096))
    quality = instrumental_quality(vocals * 0.01, np.zeros_like(vocals), vocals)
    assert quality["target_absent_output_to_vocals_db"] == pytest.approx(-40)
    assert quality["fidelity"]["scale_dependent_snr_db"] is None
    assert quality["component_fit"] is None


def test_correlated_components_are_not_independent_leakage_evidence():
    target = np.random.default_rng(13).normal(0, 0.1, (2, 4096))
    assert instrumental_quality(target, target, target)["component_fit"] is None


@pytest.mark.parametrize("vocals", [np.zeros((2, 9)), np.full((2, 10), np.inf)])
def test_wrong_timeline_or_nonfinite_component_is_rejected(vocals):
    with pytest.raises(ValueError):
        instrumental_quality(np.zeros((2, 10)), np.zeros((2, 10)), vocals)
