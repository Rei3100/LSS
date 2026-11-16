"""元Musicに無い成分の回収・既存歌唱の保持・固定整列・非公開境界を確認する。"""
from dataclasses import replace

import numpy as np
import pytest

from lss.bgm.reference_residual import ReferenceResidualRecipe, recover_reference_residual
from lss.bgm.reference_alignment import ReferenceAlignmentRecipe
from tests.test_bgm_recording_recovery import _match, signals


def _recipe(fit):
    return ReferenceResidualRecipe(fit_recipe=replace(fit, boundary_fade_frames=0, blend=1,
                                                     maximum_alignment_refinement_frames=0),
                                    boundary_fade_frames=20)


def test_missing_reference_component_is_recovered_and_other_intervals_are_unchanged(signals):
    source, base, ref, fit = signals
    before = source.copy(), base.copy(), ref.copy()
    result, observations = recover_reference_residual(source, base, ref, _match(), _recipe(fit))
    assert np.max(np.abs(result[:, 600:2400] - source[:, 600:2400])) < 0.003
    assert np.array_equal(result[:, :500], base[:, :500]) and np.array_equal(result[:, 2500:], base[:, 2500:])
    assert all(np.array_equal(actual, saved) for actual, saved in zip((source, base, ref), before))
    assert observations["publication_allowed"] is False and observations["alignment_from_original_only"]
    assert observations["artifact_kind"] == "reference_conditioned_residual_candidate"


def test_singing_already_present_in_the_parent_is_retained(signals):
    source, base, ref, fit = signals
    time = np.arange(source.shape[1]) / fit.sample_rate
    singing = 0.3 * np.stack([np.sin(2 * np.pi * 101 * time), np.cos(2 * np.pi * 137 * time)])
    without, _ = recover_reference_residual(source, base, ref, _match(), _recipe(fit))
    with_singing, _ = recover_reference_residual(source + singing, base + singing, ref, _match(), _recipe(fit))
    np.testing.assert_allclose(with_singing - without, singing, rtol=0, atol=1e-12)


def test_no_source_minus_parent_residual_leaves_the_parent_bit_identical(signals):
    source, _, ref, fit = signals
    result, _ = recover_reference_residual(source, source, ref, _match(), _recipe(fit))
    assert np.array_equal(result, source)


def test_finished_mix_ducking_is_not_restored_to_reference_level(signals):
    source, base, ref, fit = signals
    result, _ = recover_reference_residual(source * 0.05, base * 0.05, ref, _match(), _recipe(fit))
    assert np.max(np.abs(result[:, 600:2400] - source[:, 600:2400] * 0.05)) < 0.0002


@pytest.mark.parametrize("match", [replace(_match(), identity_state="unresolved"), _match(-1, 1),
                                    _match(0, 4), _match(float("nan"), 1)])
def test_unverified_match_and_bad_timeline_are_rejected(signals, match):
    with pytest.raises(ValueError):
        recover_reference_residual(*signals[:3], match, _recipe(signals[3]))


@pytest.mark.parametrize("update", [{"blend": 0}, {"blend": True}, {"boundary_fade_frames": -1},
                                    {"schema_version": True}, {"method": "other"}])
def test_invalid_recipe_is_not_run(signals, update):
    with pytest.raises(ValueError):
        recover_reference_residual(*signals[:3], _match(), replace(_recipe(signals[3]), **update))


def test_nonfinite_parent_and_different_timeline_are_rejected(signals):
    source, base, ref, fit = signals
    broken = base.copy()
    broken[0, 1000] = np.nan
    with pytest.raises(ValueError, match="有限"):
        recover_reference_residual(source, broken, ref, _match(), _recipe(fit))
    with pytest.raises(ValueError, match="時間軸"):
        recover_reference_residual(source[:, :-1], base, ref, _match(), _recipe(fit))


def test_optional_speed_alignment_has_a_distinct_recipe_and_preserves_parent_singing():
    from scipy.signal import resample_poly
    from lss.bgm.recording_recovery import RecordingRecoveryRecipe

    random = np.random.default_rng(794)
    reference = random.normal(size=(2, 10000)) * 0.02
    target = reference * np.array([0.7, 1.2])[:, None]
    singing = random.normal(size=target.shape) * 0.01
    fit = RecordingRecoveryRecipe(sample_rate=1000, fir_taps=3, fit_window_frames=1000,
        fit_hop_frames=500, boundary_fade_frames=0, maximum_alignment_refinement_frames=0)
    recipe = ReferenceResidualRecipe(fit_recipe=fit, boundary_fade_frames=20,
        alignment_recipe=ReferenceAlignmentRecipe(sample_rate=1000, probe_frames=1000,
            minimum_probe_frames=100, maximum_offset_frames=5))
    warped = resample_poly(reference, 1001, 1000, axis=1)
    recovered, observations = recover_reference_residual(target + singing, target * 0.5 + singing,
                                                        warped, _match(0, 10, 0), recipe)
    without, _ = recover_reference_residual(target, target * 0.5, warped, _match(0, 10, 0), recipe)
    assert observations["speed_alignment"]["speed_correction_used"]
    np.testing.assert_allclose(recovered - without, singing, rtol=0, atol=1e-12)
    baseline = target * 0.5 + singing
    assert np.sum((recovered - target - singing) ** 2) < np.sum((baseline - target - singing) ** 2) * 0.1
    assert replace(recipe, alignment_recipe=None).recipe_id != recipe.recipe_id
    assert "alignment_recipe" not in ReferenceResidualRecipe().to_dict()
    assert ReferenceResidualRecipe().recipe_id == "bgm-reference-residual-v1-8cbeefbb8d78c6f7"
