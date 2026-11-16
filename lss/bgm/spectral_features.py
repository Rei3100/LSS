"""音楽の性質を比べるための、共通の特徴表現。

音色（log等間隔の帯域）と音名（12類）を、同じ窓と同じ正規化で作る。ここでは判断を
持たず、比べられる形にするだけである。しきい値も既定値も持たない。

`lss/bgm/cue_segmentation.py` と `lss/bgm/reference_repetition.py` が同じ表現を使う。
別々に作り直すと、同じ「似ている」が二つの意味を持ってしまう。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD = "log-band-and-chroma-v1"


class SpectralFeatureError(ValueError):
    """特徴の宣言、または渡された音声が契約に合わない。"""


@dataclass(frozen=True)
class SpectralFeatureRecipe:
    n_fft: int
    stft_hop_frames: int
    feature_seconds: float
    feature_hop_seconds: float
    band_count: int
    low_hz: float
    high_hz: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise SpectralFeatureError("未対応の特徴schemaです")
        for field in ("n_fft", "stft_hop_frames", "band_count"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SpectralFeatureError(f"{field}は正の整数である必要があります")
        if self.n_fft & (self.n_fft - 1):
            raise SpectralFeatureError("n_fftは2の冪である必要があります")
        if self.stft_hop_frames > self.n_fft:
            raise SpectralFeatureError("stft_hop_framesはn_fft以下である必要があります")
        for field in ("feature_seconds", "feature_hop_seconds", "low_hz", "high_hz"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise SpectralFeatureError(f"{field}は有限な正数である必要があります")
        if self.feature_hop_seconds > self.feature_seconds:
            raise SpectralFeatureError("feature_hop_secondsはfeature_seconds以下である必要があります")
        if self.low_hz >= self.high_hz:
            raise SpectralFeatureError("low_hzはhigh_hz未満である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def block_and_hop(self, sample_rate: int) -> tuple[int, int]:
        """特徴一つがまとめるSTFT frame数と、その進み幅。"""
        block = max(1, int(round(self.feature_seconds * sample_rate / self.stft_hop_frames)))
        hop = max(1, int(round(self.feature_hop_seconds * sample_rate / self.stft_hop_frames)))
        return block, hop


def feature_count(frames: int, sample_rate: int, recipe: SpectralFeatureRecipe) -> int:
    """その長さから何個の特徴が得られるか。区間の端を決める側が先に知るために使う。"""
    recipe.validate()
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise SpectralFeatureError("framesは正の整数である必要があります")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise SpectralFeatureError("sample_rateは正の整数である必要があります")
    if frames < recipe.n_fft:
        return 0
    stft_frames = 1 + (frames - recipe.n_fft) // recipe.stft_hop_frames
    block, hop = recipe.block_and_hop(sample_rate)
    return 1 + max(0, stft_frames - block) // hop


def mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
        raise SpectralFeatureError("(channels, frames)の音声が必要です")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise SpectralFeatureError("音声に有限でない値があります")
    return data.astype(np.float64, copy=False).mean(axis=0)


def magnitudes(values: np.ndarray, recipe: SpectralFeatureRecipe) -> np.ndarray:
    """(stft_frames, bins)の振幅。窓はHann、端は切り捨てる。"""
    size, hop = recipe.n_fft, recipe.stft_hop_frames
    if values.size < size:
        raise SpectralFeatureError("特徴を測るには窓より長い音声が必要です")
    window = np.hanning(size)
    starts = range(0, values.size - size + 1, hop)
    return np.stack([np.abs(np.fft.rfft(values[start:start + size] * window))
                     for start in starts])


def band_matrix(freqs: np.ndarray, recipe: SpectralFeatureRecipe) -> np.ndarray:
    """log等間隔の帯域へまとめる行列。帯域外は捨てる。"""
    edges = np.geomspace(recipe.low_hz, recipe.high_hz, recipe.band_count + 1)
    matrix = np.zeros((recipe.band_count, freqs.size))
    for index in range(recipe.band_count):
        selected = (freqs >= edges[index]) & (freqs < edges[index + 1])
        matrix[index, selected] = 1.0
    if not matrix.any(axis=1).all():
        raise SpectralFeatureError("空の帯域があります。band_countか帯域の宣言を見直してください")
    return matrix


def chroma_matrix(freqs: np.ndarray, recipe: SpectralFeatureRecipe) -> np.ndarray:
    """音名12類へまとめる行列。基準は440Hzの平均律。"""
    matrix = np.zeros((12, freqs.size))
    selected = (freqs >= recipe.low_hz) & (freqs < recipe.high_hz) & (freqs > 0)
    midi = np.zeros(freqs.size)
    midi[selected] = 69.0 + 12.0 * np.log2(freqs[selected] / 440.0)
    classes = np.rint(midi).astype(int) % 12
    for index in range(12):
        matrix[index, selected & (classes == index)] = 1.0
    return matrix


def reduce_to_features(
    values: np.ndarray, matrix: np.ndarray, *, block: int, hop: int,
) -> np.ndarray:
    """帯域へまとめ、時間方向へ平均し、長さ1へ正規化した特徴列を返す。"""
    reduced = values @ matrix.T
    count = 1 + max(0, (reduced.shape[0] - block)) // hop
    rows = np.stack([reduced[index * hop:index * hop + block].mean(axis=0)
                     for index in range(count)])
    rows = np.log1p(rows)
    rows = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


def timbre_and_chroma(
    samples: np.ndarray, sample_rate: int, recipe: SpectralFeatureRecipe,
) -> tuple[np.ndarray, np.ndarray]:
    """音色と音名の特徴列を同じ時間割りで返す。"""
    recipe.validate()
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise SpectralFeatureError("sample_rateは正の整数である必要があります")
    values = magnitudes(mono(samples), recipe)
    freqs = np.fft.rfftfreq(recipe.n_fft, 1.0 / sample_rate)
    if recipe.high_hz > freqs[-1]:
        raise SpectralFeatureError("high_hzがこのsample rateのNyquistを超えています")
    block, hop = recipe.block_and_hop(sample_rate)
    return (reduce_to_features(values, band_matrix(freqs, recipe), block=block, hop=hop),
            reduce_to_features(values, chroma_matrix(freqs, recipe), block=block, hop=hop))
