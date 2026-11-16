import json

import numpy as np
import pytest

from lss.validation.stereo_comparison import spatial_regressions
from lss.validation.stereo_quality import stereo_quality


def tones():
    time = np.arange(4096) / 4096
    return np.array([np.sin(2 * np.pi * 41 * time), 0.1 * np.sin(2 * np.pi * 71 * time)]), time


def test_global_improvement_cannot_hide_nonzero_channel_attenuation():
    reference, time = tones()
    before = reference.copy()
    before[0] += 0.6 * np.sin(2 * np.pi * 103 * time)
    after = reference.copy()
    after[1] *= 0.01
    prior, current = stereo_quality(before, reference), stereo_quality(after, reference)
    assert current["scale_dependent_snr_db"] > prior["scale_dependent_snr_db"]
    assert not any(current["spatial_component_lost"].values())
    assert abs(current["projection_gain_db"]["joint"]) < 0.1
    failures = spatial_regressions(prior, current)
    assert any(row["component"] == "right" and row["metric"] == "projection_gain_db" for row in failures)
    json.dumps(failures, allow_nan=False)


def test_recovering_reference_width_is_not_a_failure_even_when_width_changes():
    reference, _ = tones()
    before = reference.copy()
    before[1] *= 0.05
    assert spatial_regressions(stereo_quality(before, reference), stereo_quality(reference, reference)) == []


def test_reference_silent_side_does_not_receive_an_invented_score():
    reference = np.ones((2, 20))
    result = stereo_quality(reference, reference)
    assert result["projection_gain_db"]["side"] is None
    assert spatial_regressions(result, result) == []


def test_component_phase_error_is_not_hidden_by_projection_gain():
    reference, time = tones()
    before = reference.copy()
    before[0] += np.sin(2 * np.pi * 99 * time)
    after = reference.copy()
    after[1] += 0.1 * np.sin(2 * np.pi * 111 * time)
    prior, current = stereo_quality(before, reference), stereo_quality(after, reference)
    assert current["scale_dependent_snr_db"] > prior["scale_dependent_snr_db"]
    assert abs(current["projection_gain_db"]["right"]) < 1e-9
    assert any(row["component"] == "right" and row["metric"] == "si_sdr_db"
               for row in spatial_regressions(prior, current))


def test_mid_side_loss_can_hide_from_joint_and_both_channel_gains():
    time = np.arange(4096) / 4096
    mid = np.sin(2 * np.pi * 41 * time)
    side = 0.01 * np.sin(2 * np.pi * 71 * time)
    reference = np.array([mid + side, mid - side])
    before = reference + 0.6 * np.sin(2 * np.pi * 99 * time)
    after = np.array([mid + 0.01 * side, mid - 0.01 * side])
    prior, current = stereo_quality(before, reference), stereo_quality(after, reference)
    assert current["scale_dependent_snr_db"] > prior["scale_dependent_snr_db"]
    assert all(abs(current["projection_gain_db"][name]) < 0.1 for name in ("joint", "left", "right"))
    assert any(row["component"] == "side" and row["metric"] == "projection_gain_db"
               for row in spatial_regressions(prior, current))


def test_invalid_metric_is_not_hidden_by_reference_silence():
    reference = np.ones((2, 20))
    prior, current = stereo_quality(reference, reference), stereo_quality(reference, reference)
    current["projection_gain_db"]["side"] = float("nan")
    with pytest.raises(ValueError, match="有限値"):
        spatial_regressions(prior, current)


def test_small_declared_gain_tolerance_is_respected():
    reference, _ = tones()
    assert spatial_regressions(stereo_quality(reference, reference), stereo_quality(reference * 0.995, reference)) == []


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_tolerance_is_refused(value):
    reference, _ = tones()
    quality = stereo_quality(reference, reference)
    with pytest.raises(ValueError):
        spatial_regressions(quality, quality, maximum_extra_gain_error_db=value)
