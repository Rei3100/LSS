"""短い信号の反射境界はsampleを削らず、既存の有効な反射と一致する。"""
import pytest
import torch
import torch.nn.functional as functional

from lss.bgm.provider_padding import reflect_pad_last_axis


@pytest.mark.parametrize("padding", [(0, 0), (3, 2), (4, 4)])
def test_matches_native_reflection_where_native_is_defined(padding):
    audio = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)
    assert torch.equal(reflect_pad_last_axis(audio, padding), functional.pad(audio, padding, mode="reflect"))


def test_reflection_can_exceed_input_length_without_moving_original_samples():
    audio = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
    padded = reflect_pad_last_axis(audio, (7, 6))
    assert padded.shape == (1, 2, 16)
    assert torch.equal(padded[..., 7:10], audio)
    assert padded[0, 0].tolist() == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1]


def test_single_sample_has_defined_constant_context():
    audio = torch.tensor([[[0.125], [-0.25]]])
    padded = reflect_pad_last_axis(audio, (10, 8))
    assert padded.shape == (1, 2, 19)
    assert torch.equal(padded[..., 10:11], audio)
    assert torch.all(padded[:, 0] == 0.125)
    assert torch.all(padded[:, 1] == -0.25)
