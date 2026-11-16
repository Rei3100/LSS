"""参照録音の中で、二つの位置が同じ楽節の別の回かどうかを測る。

同じ録音への一致で参照位置が飛ぶとき、曲が編集されている場合と、繰り返しの多い曲へ
照合が別の回に当たった場合がある。ここでは後者だけを測る。参照録音そのものの中で、
飛ぶ前と飛ぶ後の位置が似ているかを、その録音自身の中の無関係な組と比べる。

観測であって判定ではない。似ていることは「同じ回である」ことを示さないし、似ていない
ことは「編集である」ことを示さない。しきい値は呼び出し側が先に宣言する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np

from lss.bgm.spectral_features import SpectralFeatureRecipe, timbre_and_chroma

SCHEMA_VERSION = 1
METHOD = "reference-self-similarity-null-v1"


class ReferenceRepetitionError(ValueError):
    """繰り返しの読み取りの宣言、または入力が契約に合わない。"""


@dataclass(frozen=True)
class ReferenceRepetitionRecipe:
    feature: SpectralFeatureRecipe
    chroma_weight: float
    comparison_seconds: float
    null_pair_count: int
    null_percentile: float
    random_seed: int
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise ReferenceRepetitionError("未対応の繰り返し読み取りschemaです")
        if not isinstance(self.feature, SpectralFeatureRecipe):
            raise ReferenceRepetitionError("共通の特徴Recipeが必要です")
        self.feature.validate()
        value = self.chroma_weight
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not 0.0 <= float(value) <= 1.0):
            raise ReferenceRepetitionError("chroma_weightは0以上1以下である必要があります")
        value = self.comparison_seconds
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ReferenceRepetitionError("comparison_secondsは有限な正数である必要があります")
        for field in ("null_pair_count", "random_seed"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ReferenceRepetitionError(f"{field}は0以上の整数である必要があります")
        if self.null_pair_count < 10:
            raise ReferenceRepetitionError("null_pair_countは10以上である必要があります")
        value = self.null_percentile
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not 0.0 < float(value) < 100.0):
            raise ReferenceRepetitionError("null_percentileは0より大きく100未満である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "feature": self.feature.to_dict()}

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-reference-repetition-v{SCHEMA_VERSION}-" + hashlib.sha256(
            body.encode()).hexdigest()[:16]


def _similarity(timbre: np.ndarray, chroma: np.ndarray, weight: float,
                first: int, second: int, span: int) -> float:
    def block(values, index):
        rows = values[index:index + span]
        if rows.shape[0] < span:
            raise ReferenceRepetitionError("比べる区間が参照録音の外に出ています")
        mean = rows.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean

    timbre_similarity = float(block(timbre, first) @ block(timbre, second))
    chroma_similarity = float(block(chroma, first) @ block(chroma, second))
    return (1.0 - weight) * timbre_similarity + weight * chroma_similarity


def compare_positions(
    samples: np.ndarray, sample_rate: int, pairs: list, recipe: ReferenceRepetitionRecipe,
) -> dict:
    """参照録音内の位置の組を、その録音自身の無関係な組と比べる。"""
    recipe.validate()
    timbre, chroma = timbre_and_chroma(samples, sample_rate, recipe.feature)
    span = max(1, int(round(recipe.comparison_seconds / recipe.feature.feature_hop_seconds)))
    count = timbre.shape[0]
    if count <= span + 1:
        raise ReferenceRepetitionError("参照録音がこの比較には短すぎます")
    last = count - span

    def index_of(seconds: Any, context: str) -> int:
        if isinstance(seconds, bool) or not isinstance(seconds, Real) or seconds != seconds:
            raise ReferenceRepetitionError(f"{context}が不正です: {seconds!r}")
        index = int(round(float(seconds) / recipe.feature.feature_hop_seconds))
        if index < 0 or index > last:
            raise ReferenceRepetitionError(f"{context}が参照録音の外です: {seconds}")
        return index

    generator = np.random.default_rng(recipe.random_seed)
    null_values = []
    while len(null_values) < recipe.null_pair_count:
        first, second = (int(generator.integers(0, last + 1)) for _ in range(2))
        if abs(first - second) < span:
            continue
        null_values.append(_similarity(timbre, chroma, recipe.chroma_weight, first, second, span))
    threshold = float(np.percentile(null_values, recipe.null_percentile))

    rows = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ReferenceRepetitionError("位置の組はオブジェクトである必要があります")
        first = index_of(pair.get("first_seconds"), "first_seconds")
        second = index_of(pair.get("second_seconds"), "second_seconds")
        value = _similarity(timbre, chroma, recipe.chroma_weight, first, second, span)
        rows.append({
            "label": pair.get("label"),
            "first_seconds": float(pair["first_seconds"]),
            "second_seconds": float(pair["second_seconds"]),
            "similarity": value,
            "above_null_threshold": bool(value > threshold),
        })
    return {
        "recipe_id": recipe.recipe_id,
        "feature_count": int(count),
        "comparison_frames": span,
        "null": {"pair_count": len(null_values), "percentile": recipe.null_percentile,
                 "threshold": threshold, "minimum": float(np.min(null_values)),
                 "median": float(np.median(null_values)), "maximum": float(np.max(null_values))},
        "pairs": rows,
    }
