"""親の分離結果を残し、refinementの影響範囲を明示する候補stage。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StereoBlendRecipe:
    refinement_strength: float
    schema_version: int = 1
    method: str = "common-stereo-convex-blend"

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("未対応のblend schemaです")
        if self.method != "common-stereo-convex-blend":
            raise ValueError("未対応のblend methodです")
        value = self.refinement_strength
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("refinement_strengthは0以上1以下の有限値が必要です")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "bgm-stereo-blend-v1-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def blend_stereo(parent: np.ndarray, refined: np.ndarray, recipe: StereoBlendRecipe) -> np.ndarray:
    """左右共通の固定係数。音量補正や独立trimを行わず、親入力も変更しない。"""
    recipe.validate()
    arrays = []
    for value in (parent, refined):
        data = np.asarray(value)
        if (
            data.ndim != 2 or data.shape[0] != 2 or data.shape[1] == 0
            or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data)
            or not np.isfinite(data).all()
        ):
            raise ValueError("blendには同じtimelineの有限stereo配列が必要です")
        arrays.append(data.astype(np.float64, copy=False))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("blend入力のtimelineが一致しません")
    return (1 - recipe.refinement_strength) * arrays[0] + recipe.refinement_strength * arrays[1]
