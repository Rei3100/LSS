"""全体のrecallや平均改善で個別の欠損・音楽不在時の混入を隠さない。"""
import numpy as np

from lss.validation.activity_quality import activity_quality
from scripts.probe_bgm_song_preservation import compare

RULE = {"maximum_wanted_snr_regression_db": 0.5,
        "maximum_absent_candidate_frame_increase": 0,
        "singing_required_snr_improvement_db": 1.0}


def test_more_temporal_recall_cannot_hide_signal_damage():
    target = np.ones((2, 100))
    before = activity_quality(target, target, [(1, 100)])
    after = activity_quality(target * 0.5, target, [(0, 100)])
    assert compare(before, after, RULE)["regression"]


def test_absent_target_candidate_growth_is_regression():
    target = np.zeros((2, 100))
    before = activity_quality(np.ones_like(target), target, [])
    after = activity_quality(np.ones_like(target), target, [(0, 10)])
    assert compare(before, after, RULE)["regression"]


def test_restored_wanted_voice_is_measured_without_source_role_inference():
    target = np.ones((2, 100))
    before = activity_quality(np.zeros_like(target), target, [])
    after = activity_quality(target * 0.9, target, [(0, 100)])
    result = compare(before, after, RULE)
    assert result["material_singing_improvement"]
    assert not result["regression"]
