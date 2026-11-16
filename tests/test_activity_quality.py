"""時間的なrecallだけで歌唱削除や無対象の混入を隠さない。"""
import numpy as np
import pytest

from lss.validation.activity_quality import activity_quality


def test_overlapping_candidates_count_union_and_quiet_energy_loss_separately():
    target = np.ones((2, 10))
    target[:, :5] = 0.01
    result = activity_quality(target, target, [(3, 8), (5, 10)])
    assert result["candidate_union_frames"] == 7
    assert result["reference_frame_retained_fraction"] == 0.7
    assert result["reference_energy_retained_fraction"] > 0.99
    assert result["separation_fidelity"]["scale_dependent_snr_db"] > 100
    assert result["separation_and_activity_fidelity"]["scale_dependent_snr_db"] < 50


def test_full_temporal_coverage_does_not_hide_removal_of_wanted_singing():
    target = np.ones((2, 10))
    result = activity_quality(np.zeros_like(target), target, [(0, 10)])
    assert result["reference_frame_retained_fraction"] == 1
    assert result["reference_energy_retained_fraction"] == 1
    assert result["separation_and_activity_fidelity"]["spatial_component_lost"]["left"]
    assert result["separation_and_activity_fidelity"]["scale_dependent_snr_db"] == 0


def test_absent_target_has_contamination_evidence_without_invented_recall():
    result = activity_quality(np.ones((2, 10)), np.zeros((2, 10)), [(2, 7)])
    assert result["target_present"] is False
    assert result["reference_frame_retained_fraction"] is None
    assert result["reference_energy_retained_fraction"] is None
    assert result["candidate_frames_outside_reference"] == 5
    assert result["retained_signal_energy_without_target"] == 10
    assert result["separation_and_activity_fidelity"] is None


@pytest.mark.parametrize("interval", [(-1, 2), (2, 2), (0, 11), (True, 4), (0, 1.5)])
def test_invalid_timeline_is_rejected(interval):
    with pytest.raises(ValueError):
        activity_quality(np.ones((2, 10)), np.ones((2, 10)), [interval])
