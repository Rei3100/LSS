"""入力の音量・解析位相・可逆stereo基底を変える、timelineを保つDevelopment部品。"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Sequence

import numpy as np


def _stereo(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0
            or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
        raise ValueError("同じtimelineの有限な実数stereo配列が必要です")
    return values.astype(np.float64, copy=False)


@dataclass(frozen=True)
class InferenceVariantRecipe:
    input_gain_db: float = 0.0
    leading_silence_frames: int = 0
    schema_version: int = 1
    method: str = "common-stereo-gain-and-leading-silence"
    channel_basis: str = "left_right"

    def validate(self) -> None:
        if (isinstance(self.input_gain_db, bool) or not isinstance(self.input_gain_db, Real)
                or not math.isfinite(self.input_gain_db) or abs(self.input_gain_db) > 24):
            raise ValueError("入力gainは有限な-24〜24dBが必要です")
        if (type(self.leading_silence_frames) is not int or self.leading_silence_frames < 0
                or type(self.schema_version) is not int or self.schema_version not in {1, 2}):
            raise ValueError("入力variantの定義が不正です")
        if self.schema_version == 1:
            if self.method != "common-stereo-gain-and-leading-silence" or self.channel_basis != "left_right":
                raise ValueError("従来variantは元の左右channelだけを使用します")
        elif (self.method != "orthonormal-stereo-basis-gain-and-leading-silence"
              or not isinstance(self.channel_basis, str)
              or self.channel_basis not in {"left_right", "mid_side"}):
            raise ValueError("入力variantのstereo基底が不正です")

    def to_dict(self) -> dict:
        self.validate()
        result = asdict(self)
        if self.schema_version == 1:
            result.pop("channel_basis")  # 既存の定義・ID・cacheを変えない。
        return result

    @property
    def recipe_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-inference-variant-v{self.schema_version}-" + hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _basis(values: np.ndarray, recipe: InferenceVariantRecipe) -> np.ndarray:
    """直交M/S変換は自身が逆変換。左右の情報とenergyを保ち、混合の不在も保つ。"""
    if recipe.channel_basis == "left_right":
        return values
    try:
        with np.errstate(over="raise", invalid="raise"):
            left, right = values * (1 / math.sqrt(2))
            return np.stack([left + right, left - right])
    except FloatingPointError as exc:
        raise ValueError("stereo基底の変換を有限精度で表現できません") from exc


def transform_stereo(values: np.ndarray, recipe: InferenceVariantRecipe) -> np.ndarray:
    """宣言した可逆基底へ変換し、両channel共通gainと先頭無音を加える。clipはしない。"""
    recipe.validate()
    source = _basis(_stereo(values), recipe)
    result = np.pad(source, ((0, 0), (recipe.leading_silence_frames, 0)))
    return result * (10 ** (recipe.input_gain_db / 20))


def restore_stereo(values: np.ndarray, source_frames: int, recipe: InferenceVariantRecipe) -> np.ndarray:
    """全stemを元の音量とsample位置へ戻し、壊れた推論長は拒否する。"""
    recipe.validate()
    output = _stereo(values)
    if type(source_frames) is not int or source_frames <= 0 or output.shape[1] != source_frames + recipe.leading_silence_frames:
        raise ValueError("推論outputのtimelineが入力variantと一致しません")
    restored = output[:, recipe.leading_silence_frames:].copy() / (10 ** (recipe.input_gain_db / 20))
    return _basis(restored, recipe)


def ensemble_identity(recipes: Sequence[InferenceVariantRecipe]) -> str:
    if not recipes or len({recipe.recipe_id for recipe in recipes}) != len(recipes):
        raise ValueError("ensembleには重複しないvariantが必要です")
    encoded = json.dumps({"schema_version": 1, "method": "equal-common-stereo-average",
                          "ordered_variants": [recipe.to_dict() for recipe in recipes]},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "bgm-inference-ensemble-v1-" + hashlib.sha256(encoded.encode()).hexdigest()[:16]


def average_stereo(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("ensembleのoutputが空です")
    checked = [_stereo(value) for value in values]
    if any(value.shape != checked[0].shape for value in checked):
        raise ValueError("ensembleのstereo/timelineが一致しません")
    result = np.zeros_like(checked[0])
    for value in checked:
        result += value / len(checked)
    return result
