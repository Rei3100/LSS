import numpy as np
import pytest

from lss.bgm.inference_variants import (InferenceVariantRecipe, average_stereo,
                                       ensemble_identity, restore_stereo, transform_stereo)


@pytest.mark.parametrize("gain", [-6.0, 0.0, 6.0])
@pytest.mark.parametrize("padding", [0, 11025])
def test_transform_restores_gain_channels_and_sample_positions(gain, padding):
    source = np.stack([np.arange(101) / 100, -np.arange(101) / 200])
    original = source.copy()
    recipe = InferenceVariantRecipe(gain, padding)
    transformed = transform_stereo(source, recipe)
    assert np.count_nonzero(transformed[:, :padding]) == 0
    assert transformed.shape == (2, 101 + padding)
    np.testing.assert_allclose(restore_stereo(transformed, 101, recipe), source, atol=1e-15)
    np.testing.assert_array_equal(source, original)
    if gain == 6:
        assert transformed.max() > 1  # 実験入力はclipされない。


def test_padding_does_not_wrap_future_samples_into_start():
    source = np.zeros((2, 100))
    source[:, -1] = 0.3
    recipe = InferenceVariantRecipe(0.0, 9)
    transformed = transform_stereo(source, recipe)
    assert np.count_nonzero(transformed[:, :-1]) == 0
    assert np.all(transformed[:, -1] == 0.3)


@pytest.mark.parametrize("definition", [{"input_gain_db": float("nan")}, {"input_gain_db": 25},
                                        {"input_gain_db": True}, {"leading_silence_frames": -1},
                                        {"leading_silence_frames": 0.5}])
def test_invalid_definition_is_rejected(definition):
    with pytest.raises(ValueError):
        InferenceVariantRecipe(**definition).to_dict()


def test_wrong_inference_length_is_rejected():
    with pytest.raises(ValueError, match="timeline"):
        restore_stereo(np.zeros((2, 100)), 100, InferenceVariantRecipe(0.0, 9))


def test_average_uses_common_weight_and_preserves_inputs():
    first = np.stack([np.ones(100), np.ones(100) * 0.5])
    second = first * -0.5
    original = first.copy()
    np.testing.assert_array_equal(average_stereo([first, second]), first * 0.25)
    np.testing.assert_array_equal(first, original)
    with pytest.raises(ValueError, match="timeline"):
        average_stereo([first, second[:, :-1]])


@pytest.mark.parametrize("source", [np.zeros((1, 100)), np.zeros((2, 0)),
                                    np.ones((2, 100), dtype=complex), np.full((2, 100), np.nan)])
def test_invalid_signal_is_rejected(source):
    with pytest.raises(ValueError):
        transform_stereo(source, InferenceVariantRecipe())


def test_recipe_identity_includes_gain_padding_and_ensemble_order():
    base, gain, shift = InferenceVariantRecipe(), InferenceVariantRecipe(6.0), InferenceVariantRecipe(0.0, 11025)
    assert len({recipe.recipe_id for recipe in (base, gain, shift)}) == 3
    assert ensemble_identity([base, shift]) != ensemble_identity([shift, base])
    with pytest.raises(ValueError):
        ensemble_identity([base, base])


@pytest.mark.parametrize("gain,padding", [(0.0, 0), (-6.0, 11), (6.0, 1)])
def test_orthonormal_mid_side_view_restores_spatial_information_and_timeline(gain, padding):
    recipe = InferenceVariantRecipe(gain, padding, schema_version=2,
        method="orthonormal-stereo-basis-gain-and-leading-silence", channel_basis="mid_side")
    source = np.random.default_rng(22).normal(size=(2, 1001))
    original = source.copy()
    view = transform_stereo(source, recipe)
    restored = restore_stereo(view, source.shape[1], recipe)
    np.testing.assert_allclose(restored, source, atol=2e-15, rtol=1e-15)
    np.testing.assert_array_equal(source, original)
    assert view.shape == (2, source.shape[1] + padding) and not np.any(view[:, :padding])
    np.testing.assert_allclose(np.sum(view ** 2), np.sum(source ** 2) * 10 ** (gain / 10), rtol=1e-15)


def test_centered_dialogue_has_no_side_and_side_only_audio_has_no_mid():
    recipe = InferenceVariantRecipe(schema_version=2,
        method="orthonormal-stereo-basis-gain-and-leading-silence", channel_basis="mid_side")
    signal = np.arange(20) / 20
    assert not np.any(transform_stereo(np.stack([signal, signal]), recipe)[1])
    assert not np.any(transform_stereo(np.stack([signal, -signal]), recipe)[0])


def test_old_variant_identity_is_unchanged_and_new_basis_is_explicit():
    legacy = InferenceVariantRecipe()
    assert legacy.to_dict() == {"input_gain_db": 0.0, "leading_silence_frames": 0, "schema_version": 1,
                                "method": "common-stereo-gain-and-leading-silence"}
    assert legacy.recipe_id == "bgm-inference-variant-v1-04219dd78d62d030"
    mid_side = InferenceVariantRecipe(schema_version=2,
        method="orthonormal-stereo-basis-gain-and-leading-silence", channel_basis="mid_side")
    left_right = InferenceVariantRecipe(schema_version=2,
        method="orthonormal-stereo-basis-gain-and-leading-silence", channel_basis="left_right")
    assert len({legacy.recipe_id, mid_side.recipe_id, left_right.recipe_id}) == 3
    assert InferenceVariantRecipe(**mid_side.to_dict()) == mid_side


@pytest.mark.parametrize("definition", [{"channel_basis": "mid_side"}, {"schema_version": True},
    {"schema_version": 2}, {"schema_version": 2,
     "method": "orthonormal-stereo-basis-gain-and-leading-silence", "channel_basis": "mono"}])
def test_basis_cannot_silently_change_an_existing_recipe(definition):
    with pytest.raises(ValueError):
        InferenceVariantRecipe(**definition).validate()
