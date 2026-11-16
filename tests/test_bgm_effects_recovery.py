"""Effects回収Stageの保存条件と、短い入力・左右の共通maskを確認する。"""
import numpy as np
import pytest

from lss.bgm.effects_recovery import EffectsRecoveryRecipe, recover_effects


@pytest.mark.parametrize("frames", [1, 127, 8000])
def test_recovery_keeps_timeline_input_bytes_and_the_sum_of_stems(frames):
    time = np.arange(frames) / 8000
    tone = np.sin(2 * np.pi * 500 * time)
    stereo = np.stack([tone, -tone * 0.3])
    stems = {"music": stereo * 0.6, "effects": stereo * 0.4, "speech": np.zeros_like(stereo)}
    before = {role: value.copy() for role, value in stems.items()}
    original = stereo.copy()
    recipe = EffectsRecoveryRecipe(n_fft=256, hop_frames=64, time_median_frames=9, frequency_median_bins=9)
    result, evidence = recover_effects(original, stems, recipe)
    np.testing.assert_allclose(sum(result.values()), sum(stems.values()), rtol=0, atol=2e-16)
    for role in stems:
        assert result[role].shape == stereo.shape and np.isfinite(result[role]).all()
        np.testing.assert_array_equal(stems[role], before[role])
    np.testing.assert_array_equal(original, stereo)
    np.testing.assert_allclose(result["music"][1], result["music"][0] * -0.3, atol=1e-15, rtol=0)
    assert evidence["publication_allowed"] is False
    if frames == 8000:
        assert evidence["recovered_energy"] > 0


def test_silent_music_does_not_acquire_effects_when_music_support_is_required():
    rng = np.random.default_rng(42)
    effects = rng.standard_normal((2, 1000))
    stems = {"music": np.zeros_like(effects), "effects": effects, "speech": np.zeros_like(effects)}
    result, evidence = recover_effects(effects, stems, EffectsRecoveryRecipe())
    np.testing.assert_array_equal(result["music"], stems["music"])
    np.testing.assert_array_equal(result["effects"], stems["effects"])
    assert evidence["recovered_energy"] == 0


def test_mismatched_source_and_stem_timelines_are_rejected():
    source = np.zeros((2, 100))
    stems = {"music": source, "speech": source, "effects": np.zeros((2, 99))}
    with pytest.raises(ValueError, match="時間軸"):
        recover_effects(source, stems, EffectsRecoveryRecipe())


@pytest.mark.parametrize("phase", [0, np.pi / 2, np.pi])
def test_coherence_gate_recovers_matching_phase_and_rejects_other_phase(phase):
    time = np.arange(32000) / 8000
    music = np.stack([np.sin(2 * np.pi * 500 * time)] * 2)
    effects = np.stack([np.sin(2 * np.pi * 500 * time + phase)] * 2)
    stems = {"music": music, "effects": effects, "speech": np.zeros_like(music)}
    recipe = EffectsRecoveryRecipe(n_fft=256, hop_frames=64, time_median_frames=9, frequency_median_bins=9,
                                   minimum_phase_coherence=0.9)
    result, evidence = recover_effects(music + effects, stems, recipe)
    delta = result["music"][:, 256:-256] - music[:, 256:-256]
    if phase == 0:
        assert np.sum(delta ** 2) > 100
    else:
        assert np.sum(delta ** 2) < 1e-8
    assert evidence["publication_allowed"] is False


@pytest.mark.parametrize("minimum", [-0.1, 1.1, float("nan"), True])
def test_invalid_coherence_conditions_cannot_be_used(minimum):
    with pytest.raises(ValueError, match="coherence"):
        EffectsRecoveryRecipe(minimum_phase_coherence=minimum).validate()
