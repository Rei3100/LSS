"""SFC解析の区間の重ね合わせ（`sfc_analysis._windowed_css`）は、区間の境目で平均の相手が不連続に替わらない。"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lss.bgm import sfc_analysis  # noqa: E402


class _Separator:
    """12秒区間・6秒ずらしのwrapperの代わり。区間ごとに ``gains`` の値を掛けて3 stemとして返す。"""

    def __init__(self, gains=None):
        self.fs, self.css_segment_size, self.css_shift_size = 100, 12, 6
        self.gains = gains
        self.calls = 0

    def __call__(self, segment, ref=None):
        gain = 1.0 if self.gains is None else self.gains[self.calls]
        self.calls += 1
        return (segment * gain).unsqueeze(1).repeat(1, 3, 1, 1)


def test_identity_separation_reproduces_the_input_exactly():
    wav = torch.from_numpy(np.random.default_rng(0).normal(size=(1, 1, 100 * 50)).astype(np.float32))
    out = sfc_analysis._windowed_css(_Separator(), wav)
    assert out.shape == (1, 3, 1, wav.shape[-1])
    np.testing.assert_allclose(out[0, 1, 0].numpy(), wav[0, 0].numpy(), atol=1e-5)


GAINS = [1.0, 1.4, 0.9, 1.6, 1.1, 0.7, 1.5, 1.2, 0.8, 1.3, 1.0, 1.4, 0.9, 1.6, 1.1, 0.7, 1.5, 1.2, 0.8, 1.3]


def _upstream_flat(frames, rate=100, size=1200, shift=600):
    """upstreamのcssと同じ：各sampleは、それを含む区間の推定（入力×係数）の平坦な平均。"""
    total, count = np.zeros(frames), np.zeros(frames)
    for index in range(-(-(frames - (size - shift)) // shift)):
        start, end = index * shift, min(index * shift + size, frames)
        total[start:end] += GAINS[index]
        count[start:end] += 1
    return total / count


def test_different_estimates_per_segment_are_joined_without_steps():
    wav = torch.ones((1, 1, 100 * 50), dtype=torch.float32)
    out = sfc_analysis._windowed_css(_Separator(GAINS), wav)[0, 1, 0].numpy()
    flat = _upstream_flat(wav.shape[-1])
    # upstreamは6秒ごとに平均の相手が替わり段になる。端を傾けると1 sampleあたりの動きはその1/10未満
    # （この試験は100Hzで傾きが25 sample。実音声の44.1kHzでは11025 sampleに広がる）。
    assert np.max(np.abs(np.diff(flat))) > 0.2
    assert np.max(np.abs(np.diff(out))) < np.max(np.abs(np.diff(flat))) / 10


def test_samples_away_from_segment_edges_equal_the_upstream_flat_average():
    wav = torch.ones((1, 1, 100 * 50), dtype=torch.float32)
    out = sfc_analysis._windowed_css(_Separator(GAINS), wav)[0, 1, 0].numpy()
    flat = _upstream_flat(wav.shape[-1])
    near_edge = np.zeros(wav.shape[-1], dtype=bool)
    for boundary in range(0, wav.shape[-1] + 1, 600):
        near_edge[max(0, boundary - 25):boundary + 25] = True
    np.testing.assert_allclose(out[~near_edge], flat[~near_edge], atol=1e-5)
    assert np.mean(near_edge) < 0.1


def test_silent_segments_are_zero_like_upstream():
    wav = torch.zeros((1, 1, 100 * 30), dtype=torch.float32)
    separator = _Separator()
    out = sfc_analysis._windowed_css(separator, wav)
    assert separator.calls == 0 and float(out.abs().max()) == 0.0 and out.shape == (1, 3, 1, 3000)
