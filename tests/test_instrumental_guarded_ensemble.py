import numpy as np
import pytest

from scripts.probe_instrumental_guarded_ensemble import guarded_mean


def test_energy_increase_is_vetoed_without_normalizing_or_changing_parent():
    parent = np.ones((2, 100)) * 0.01
    snapshot = parent.copy()
    result, trace = guarded_mean(parent, parent * 10, 0.5)
    np.testing.assert_array_equal(result, snapshot)
    np.testing.assert_array_equal(parent, snapshot)
    assert trace["energy_increase_veto"] and not trace["enabled"]
    assert trace["semantic_presence"] == "unresolved"


def test_attenuation_is_vetoed_even_when_energy_decreases():
    parent = np.ones((2, 100))
    result, trace = guarded_mean(parent, parent * 0.1, 0.5)
    np.testing.assert_array_equal(result, parent)
    assert trace["parent_projection_veto"] and not trace["energy_increase_veto"]


def test_small_fixed_mean_is_enabled_without_gain_correction():
    parent = np.arange(20).reshape(2, 10) / 100
    result, trace = guarded_mean(parent, parent * 0.995, 0.5)
    np.testing.assert_array_equal(result, parent * 0.5 + parent * 0.995 * 0.5)
    assert trace["enabled"]


def test_exact_zero_parent_is_kept_without_semantic_absence_claim():
    parent = np.zeros((2, 4))
    result, trace = guarded_mean(parent, np.ones_like(parent), 0.5)
    np.testing.assert_array_equal(result, parent)
    assert trace["candidate_projection_to_parent_db"] is None
    assert trace["semantic_presence"] == "unresolved"


@pytest.mark.parametrize("weight", [0, -1, 2, True, float("nan")])
def test_invalid_weight_is_refused(weight):
    with pytest.raises(ValueError):
        guarded_mean(np.ones((2, 4)), np.ones((2, 4)), weight)


@pytest.mark.parametrize("alternate", [np.ones((2, 3)), np.ones((1, 4)), np.full((2, 4), np.nan), np.ones((2, 4), dtype=complex)])
def test_shape_and_finiteness_are_enforced(alternate):
    with pytest.raises(ValueError):
        guarded_mean(np.ones((2, 4)), alternate, 0.5)
