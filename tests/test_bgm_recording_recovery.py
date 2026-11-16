"""参照条件付き再構成は原本の係数・左右・時間軸を保持し、未一致区間を変えない。"""
from dataclasses import replace

import numpy as np
import pytest

from lss.bgm.identity import RecordingMatch
from lss.bgm.recording_recovery import RecordingRecoveryRecipe, reconstruct_recording


def _match(start=0.5, end=2.5, reference_start=0.5):
    return RecordingMatch("reference", "reference.flac", "a" * 64, start, end, reference_start,
                          0, 100, 1, 0, 0.1, 10, 10, 0, 1, 0, 10, 0, 0, "known_exact")


@pytest.fixture
def signals():
    time = np.arange(3000) / 1000
    reference = np.stack([np.sin(time * 2 * np.pi * 43), np.cos(time * 2 * np.pi * 71)])
    target = reference * np.array([0.7, 1.3])[:, None]
    base = target * 0.1
    recipe = RecordingRecoveryRecipe(sample_rate=1000, fir_taps=3, fit_window_frames=256,
                                     fit_hop_frames=128, boundary_fade_frames=20,
                                     alignment_probe_frames=200, maximum_alignment_refinement_frames=2)
    return target, base, reference, recipe


def test_coefficients_are_estimated_from_original_and_channels_are_not_mixed(signals):
    original, base, reference, recipe = signals
    before = original.copy(), base.copy(), reference.copy()
    result, observations = reconstruct_recording(original, base, reference, _match(), recipe)
    assert all(np.array_equal(actual, snapshot) for actual, snapshot in zip((original, base, reference), before))
    assert result.shape == original.shape and np.isfinite(result).all()
    assert np.max(np.abs(result[:, 600:2400] - original[:, 600:2400])) < 0.003
    assert np.array_equal(result[:, :500], base[:, :500]) and np.array_equal(result[:, 2500:], base[:, 2500:])
    assert observations["publication_allowed"] is False and observations["semantic_role"] == "unresolved"
    assert observations["artifact_kind"] == "reference_conditioned_reconstruction"


def test_nonzero_reference_is_not_copied_at_reference_volume(signals):
    original, base, reference, recipe = signals
    result, _ = reconstruct_recording(original * 0.2, base * 0.2, reference, _match(), recipe)
    assert np.max(np.abs(result[:, 600:2400] - original[:, 600:2400] * 0.2)) < 0.001


def test_rounded_end_timestamp_only_uses_the_verified_overlap(signals):
    original, base, reference, recipe = signals
    result, observations = reconstruct_recording(original, base, reference, _match(0, 3.001, 0),
                                                  replace(recipe, timestamp_tolerance_frames=1))
    assert result.shape == original.shape and observations["source_end_frame"] == 3000
    assert np.isfinite(result).all()


def test_silent_alignment_probe_does_not_invent_a_sample_shift(signals):
    original, base, reference, recipe = [value.copy() if isinstance(value, np.ndarray) else value for value in signals]
    for values in (original, base, reference):
        values[:, :1000] = 0
    result, observations = reconstruct_recording(original, base, reference, _match(0, 3, 0), recipe)
    assert observations["alignment_refinement_frames"] == 0
    assert np.max(np.abs(result[:, 1200:2800] - original[:, 1200:2800])) < 0.003


@pytest.mark.parametrize("match", [replace(_match(), identity_state="unresolved"), _match(-1, 1),
                                    _match(0, 4), _match(0, 1, 3), _match(float("nan"), 1)])
def test_unverified_or_invalid_alignment_is_rejected(signals, match):
    with pytest.raises(ValueError):
        reconstruct_recording(*signals[:3], match, signals[3])


def test_mismatched_timeline_and_nonfinite_input_are_rejected(signals):
    source, base, reference, recipe = signals
    with pytest.raises(ValueError, match="時間軸"):
        reconstruct_recording(source[:, :-1], base, reference, _match(), recipe)
    broken = source.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        reconstruct_recording(broken, base, reference, _match(), recipe)


@pytest.mark.parametrize("update", [{"blend": 0}, {"ridge": 0}, {"fir_taps": 2},
                                    {"fit_hop_frames": 300}, {"boundary_fade_frames": -1}])
def test_invalid_recipe_is_rejected(signals, update):
    with pytest.raises(ValueError):
        replace(signals[3], **update).to_dict()
