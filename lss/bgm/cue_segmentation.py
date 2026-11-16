"""音楽の活動区間の中から、曲が変わる位置の候補を信号だけで探すStage。

`docs/BGM.md` の「Cue formation」に対応する。音が有るかではなく、音楽の性質が
変わったかを見る。意味的role、同一性、公開可否は判断しない。閾値を含むRecipeは
呼び出し側が先に宣言する。既定値は持たない。

観測であって正解ではない。一曲の中の展開でも値は動く。ここで返す位置は「曲の切れ目の
候補」であり、確定した境界ではない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np

from lss.bgm.spectral_features import (
    SpectralFeatureError,
    SpectralFeatureRecipe,
    timbre_and_chroma,
)

SCHEMA_VERSION = 1
METHOD = "spectral-chroma-novelty-checkerboard"


class CueSegmentationError(ValueError):
    """区切り候補のRecipeまたは入力が契約に合わない。"""


@dataclass(frozen=True)
class CueSegmentationRecipe:
    n_fft: int
    stft_hop_frames: int
    feature_seconds: float
    feature_hop_seconds: float
    band_count: int
    low_hz: float
    high_hz: float
    kernel_seconds: float
    minimum_boundary_gap_seconds: float
    novelty_threshold: float
    chroma_weight: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise CueSegmentationError("未対応の区切り候補schemaです")
        for field in ("n_fft", "stft_hop_frames", "band_count"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CueSegmentationError(f"{field}は正の整数である必要があります")
        if self.n_fft & (self.n_fft - 1):
            raise CueSegmentationError("n_fftは2の冪である必要があります")
        if self.stft_hop_frames > self.n_fft:
            raise CueSegmentationError("stft_hop_framesはn_fft以下である必要があります")
        for field in ("feature_seconds", "feature_hop_seconds", "kernel_seconds",
                      "minimum_boundary_gap_seconds", "low_hz", "high_hz"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise CueSegmentationError(f"{field}は有限な正数である必要があります")
        if self.feature_hop_seconds > self.feature_seconds:
            raise CueSegmentationError("feature_hop_secondsはfeature_seconds以下である必要があります")
        if self.low_hz >= self.high_hz:
            raise CueSegmentationError("low_hzはhigh_hz未満である必要があります")
        for field in ("novelty_threshold", "chroma_weight"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0.0 <= float(value) <= 1.0):
                raise CueSegmentationError(f"{field}は0以上1以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def feature_recipe(self) -> SpectralFeatureRecipe:
        return SpectralFeatureRecipe(
            n_fft=self.n_fft, stft_hop_frames=self.stft_hop_frames,
            feature_seconds=self.feature_seconds, feature_hop_seconds=self.feature_hop_seconds,
            band_count=self.band_count, low_hz=self.low_hz, high_hz=self.high_hz,
        )

    @property
    def recipe_id(self) -> str:
        self.validate()
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"bgm-cue-segmentation-v{self.schema_version}-{digest}"


@dataclass(frozen=True)
class CueBoundaryCandidate:
    frame: int
    seconds: float
    novelty: float


@dataclass(frozen=True)
class CueSegmentationResult:
    recipe_id: str
    sample_rate: int
    source_frames: int
    feature_count: int
    feature_hop_frames: int
    novelty: tuple[float, ...]
    boundaries: tuple[CueBoundaryCandidate, ...]


def _novelty(features: np.ndarray, half: int) -> np.ndarray:
    """自己相似行列の対角へcheckerboard kernelを当て、変化の強さを返す。

    kernelの重みの総量で割る。信号ごとの最大値で正規化すると、何も変わらない信号でも
    最大の変化に見えてしまうため、しきい値は信号によらない目盛りの上で宣言する。
    """
    similarity = features @ features.T
    offsets = np.arange(-half, half + 1)
    taper = np.exp(-0.5 * (offsets / (half / 2.0)) ** 2)
    sign = np.outer(np.sign(offsets), np.sign(offsets))
    kernel = np.outer(taper, taper) * sign
    scale = float(np.abs(kernel).sum())
    values = np.zeros(features.shape[0])
    for center in range(half, features.shape[0] - half):
        block = similarity[center - half:center + half + 1, center - half:center + half + 1]
        values[center] = float(np.sum(block * kernel)) / scale
    return np.maximum(values, 0.0)


def _peaks(novelty: np.ndarray, *, threshold: float, minimum_gap: int) -> list[int]:
    """しきい値以上の極大を、強い順に最小間隔を守って選ぶ。"""
    candidates = [
        index for index in range(1, novelty.size - 1)
        if novelty[index] >= threshold
        and novelty[index] >= novelty[index - 1]
        and novelty[index] > novelty[index + 1]
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: (-novelty[item], item)):
        if all(abs(index - chosen) >= minimum_gap for chosen in selected):
            selected.append(index)
    return sorted(selected)


def segment_cues(
    samples: np.ndarray, sample_rate: int, recipe: CueSegmentationRecipe,
) -> CueSegmentationResult:
    """区間一つの中から、曲が変わる位置の候補を返す。境界を確定しない。"""
    recipe.validate()
    feature_recipe = recipe.feature_recipe
    try:
        timbre, chroma = timbre_and_chroma(samples, sample_rate, feature_recipe)
    except SpectralFeatureError as exc:
        raise CueSegmentationError(str(exc)) from exc
    block, hop = feature_recipe.block_and_hop(sample_rate)
    half = max(1, int(round(recipe.kernel_seconds / recipe.feature_hop_seconds)))
    if timbre.shape[0] < 2 * half + 3:
        raise CueSegmentationError("kernelに対して区間が短すぎます")

    novelty = ((1.0 - recipe.chroma_weight) * _novelty(timbre, half)
               + recipe.chroma_weight * _novelty(chroma, half))
    feature_hop_frames = hop * recipe.stft_hop_frames
    minimum_gap = max(1, int(round(recipe.minimum_boundary_gap_seconds / recipe.feature_hop_seconds)))
    indices = _peaks(novelty, threshold=recipe.novelty_threshold, minimum_gap=minimum_gap)
    # 特徴一つはblock分の時間をまとめているので、その中央を位置とする。
    offset = block * recipe.stft_hop_frames // 2
    boundaries = tuple(
        CueBoundaryCandidate(
            frame=index * feature_hop_frames + offset,
            seconds=(index * feature_hop_frames + offset) / sample_rate,
            novelty=float(novelty[index]),
        )
        for index in indices
    )
    return CueSegmentationResult(
        recipe_id=recipe.recipe_id,
        sample_rate=sample_rate,
        source_frames=int(np.asarray(samples).shape[-1]),
        feature_count=int(timbre.shape[0]),
        feature_hop_frames=feature_hop_frames,
        novelty=tuple(float(value) for value in novelty),
        boundaries=boundaries,
    )
