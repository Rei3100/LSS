"""親stemの局所energyから同時刻の分離branchを混ぜるDevelopment部品。"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Mapping

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class DominanceRoutingRecipe:
    energy_window_frames: int = 44101
    lower_music_share: float = 0.8
    upper_music_share: float = 0.95
    minimum_music_rms: float = 0.0005
    full_music_rms: float = 0.001
    schema_version: int = 1
    method: str = "parent-stem-energy-common-stereo-routing"

    def validate(self) -> None:
        if (type(self.energy_window_frames) is not int or self.energy_window_frames < 1
                or self.energy_window_frames % 2 != 1 or type(self.schema_version) is not int
                or self.schema_version != 1 or self.method != "parent-stem-energy-common-stereo-routing"):
            raise ValueError("局所energyのwindow/Recipe定義が不正です")
        numbers = (self.lower_music_share, self.upper_music_share, self.minimum_music_rms, self.full_music_rms)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in numbers):
            raise ValueError("routingの境界値には有限な実数が必要です")
        if not (0 <= self.lower_music_share < self.upper_music_share <= 1
                and 0 <= self.minimum_music_rms < self.full_music_rms):
            raise ValueError("routingの境界値の順序が不正です")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-dominance-routing-v1-" + hashlib.sha256(encoded.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DominanceRoutingResult:
    stems: Mapping[str, np.ndarray]
    parent_weight: np.ndarray
    dominant_weight: np.ndarray
    mixed_weight: np.ndarray


def route_stereo(parent: Mapping[str, np.ndarray], dominant: Mapping[str, np.ndarray] | None,
                 mixed: Mapping[str, np.ndarray] | None,
                 recipe: DominanceRoutingRecipe) -> DominanceRoutingResult:
    """parentのMusic優勢度で分岐。全stem・左右共通weight。BGM意味判定ではない。"""
    recipe.validate()
    required = {"music", "speech", "effects"}
    if set(parent) != required:
        raise ValueError("routingにはMusic/Speech/Effectsの3stemが必要です")
    shape = np.asarray(parent["music"]).shape

    def checked(stems):
        if set(stems) != required:
            raise ValueError("branchのstem setが親と一致しません")
        result = {}
        for name, values in stems.items():
            values = np.asarray(values)
            if (values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0 or values.shape != shape
                    or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
                raise ValueError("branchの有限stereo/timelineが一致しません")
            result[name] = values.astype(np.float64, copy=False)
        return result

    parent = checked(parent)
    dominant = checked(dominant) if dominant is not None else None
    mixed = checked(mixed) if mixed is not None else None
    try:
        with np.errstate(over="raise", invalid="raise"):
            powers = {name: uniform_filter1d(np.mean(values ** 2, axis=0), recipe.energy_window_frames,
                                            mode="nearest") for name, values in parent.items()}
            total = sum(powers.values())
    except FloatingPointError as exc:
        raise ValueError("局所routingのenergy集計が有限範囲を超えました") from exc
    if not np.isfinite(total).all() or any(not np.isfinite(value).all() for value in powers.values()):
        raise ValueError("局所routingのenergy集計が有限範囲を超えました")
    share = np.divide(powers["music"], total, out=np.zeros_like(total), where=total > 0)
    dominance = np.clip((share - recipe.lower_music_share) / (recipe.upper_music_share - recipe.lower_music_share), 0, 1)
    presence = np.clip((np.sqrt(np.maximum(0, powers["music"])) - recipe.minimum_music_rms)
                       / (recipe.full_music_rms - recipe.minimum_music_rms), 0, 1)
    dominant_weight = presence * dominance if dominant is not None else np.zeros_like(presence)
    mixed_weight = presence * (1 - dominance) if mixed is not None else np.zeros_like(presence)
    parent_weight = 1 - dominant_weight - mixed_weight
    stems = {name: values * parent_weight[None, :] for name, values in parent.items()}
    for child, weight in ((dominant, dominant_weight), (mixed, mixed_weight)):
        if child is not None:
            for name, values in child.items():
                stems[name] += values * weight[None, :]
    if any(not np.isfinite(values).all() for values in stems.values()):
        raise ValueError("局所routingの出力が有限範囲を超えました")
    return DominanceRoutingResult(stems, parent_weight, dominant_weight, mixed_weight)
