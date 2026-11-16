"""強い歌唱で全体の点が改善しても静かな伴奏の削除を隠さない。"""
import json

import numpy as np
import pytest

from lss.validation.song_component_comparison import component_regressions, wanted_components
from lss.validation.stereo_quality import stereo_quality


def signals():
    time = np.arange(4096) / 4096
    accompaniment = np.tile(0.01 * np.sin(2 * np.pi * 41 * time), (2, 1))
    singing = np.tile(np.sin(2 * np.pi * 71 * time), (2, 1))
    noise = np.tile(0.1 * np.sin(2 * np.pi * 103 * time), (2, 1))
    return accompaniment, singing, noise


def test_whole_song_improvement_does_not_hide_loss_of_quiet_accompaniment():
    accompaniment, singing, noise = signals()
    reference = accompaniment + singing
    before, after = reference + noise, 0.01 * accompaniment + singing
    prior, current = stereo_quality(before, reference), stereo_quality(after, reference)
    assert current["scale_dependent_snr_db"] > prior["scale_dependent_snr_db"]
    assert abs(current["projection_gain_db"]["joint"]) < 0.1
    failures = component_regressions(wanted_components(before, accompaniment, singing),
                                     wanted_components(after, accompaniment, singing))
    assert [row["component"] for row in failures] == ["accompaniment"]
    assert failures[0]["extra_gain_error_db"] > 39
    json.dumps(failures, allow_nan=False)


def test_correlated_and_single_present_components_remain_unresolved_without_fake_scores():
    accompaniment, singing, _ = signals()
    correlated = wanted_components(singing * 2, singing, singing)
    assert correlated["state"] == "undetermined" and correlated["gains"] is None
    assert component_regressions(correlated, correlated) == []
    one = wanted_components(singing, np.zeros_like(accompaniment), singing)
    assert one["state"] == "not_jointly_present" and one["gains"] is None


def test_recovery_of_component_gain_is_not_a_regression():
    accompaniment, singing, _ = signals()
    poor = wanted_components(accompaniment * 0.1 + singing, accompaniment, singing)
    recovered = wanted_components(accompaniment + singing, accompaniment, singing)
    assert component_regressions(poor, recovered) == []


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_invalid_extra_gain_tolerance_is_rejected(value):
    accompaniment, singing, _ = signals()
    measured = wanted_components(accompaniment + singing, accompaniment, singing)
    with pytest.raises(ValueError):
        component_regressions(measured, measured, maximum_extra_gain_error_db=value)
