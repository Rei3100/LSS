import pytest

from lss.validation.facing_music import band_features, renamed_state


def test_band_features_keep_frequency_real_imaginary_and_channel_order():
    import torch
    # 異なる左右・周波数・時刻を用い、shapeだけ合う誤ったflattenを検出する。
    real = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    spectral = torch.complex(real, 100 + real)
    features = band_features(spectral, 1, 3)
    assert features.shape == (1, 4, 2, 4)
    assert features[0, 0, 0].tolist() == [4, 104, 8, 108]
    assert features[0, 0, 1].tolist() == [16, 116, 20, 120]
    assert features[0, 3, 1].tolist() == [19, 119, 23, 123]
    assert spectral.real.equal(real)


def test_release_names_connect_to_same_operations_without_changing_values():
    sentinels = [object() for _ in range(6)]
    names = ["model.band_split.norm_fc_modules.12.combined.0.weight",
             "model.band_split.norm_fc_modules.12.combined.1.bias",
             "model.mask_estim.music_vocals.norm_mlp.3.combined.0.weight",
             "model.mask_estim.music_vocals.norm_mlp.3.combined.1.0.weight",
             "model.mask_estim.music_vocals.norm_mlp.3.combined.2.0.bias",
             "model.tf_model.seqband.0.rnn.weight_ih_l0"]
    mapped = renamed_state(dict(zip(names, sentinels)))
    assert list(mapped) == ["band_split.norm_fc_modules.12.norm.weight",
        "band_split.norm_fc_modules.12.fc.bias", "mask_estim.music_vocals.norm_mlp.3.norm.weight",
        "mask_estim.music_vocals.norm_mlp.3.hidden.0.weight",
        "mask_estim.music_vocals.norm_mlp.3.output.0.bias", "tf_model.seqband.0.rnn.weight_ih_l0"]
    assert all(a is b for a, b in zip(mapped.values(), sentinels))


def test_two_release_keys_cannot_silently_overwrite_one_operation():
    with pytest.raises(ValueError, match="複数の値"):
        renamed_state({"model.band_split.norm_fc_modules.0.combined.0.weight": object(),
                       "model.band_split.norm_fc_modules.0.norm.weight": object()})


def test_serialized_aliases_must_have_identical_tensor_values():
    import torch
    name = "model.mask_estim.music_vocals.norm_mlp.0."
    value = torch.tensor([0.125, -0.5])
    mapped = renamed_state({name + "norm.weight": value, name + "combined.0.weight": value.clone()})
    assert mapped["mask_estim.music_vocals.norm_mlp.0.norm.weight"] is value
    with pytest.raises(ValueError, match="複数の値"):
        renamed_state({name + "norm.weight": value, name + "combined.0.weight": value + 0.125})


def test_parameterless_transpose_positions_are_removed_without_reordering_rnns():
    value = object()
    assert renamed_state({"model.tf_model.seqband.26.rnn.weight_ih_l0": value}) == {
        "tf_model.seqband.13.rnn.weight_ih_l0": value}
    with pytest.raises(ValueError, match="transpose位置"):
        renamed_state({"model.tf_model.seqband.27.rnn.weight_ih_l0": value})
