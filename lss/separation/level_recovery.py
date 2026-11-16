"""分離した対象が取りこぼした大きさを、原本と推定だけから読み取る。

`docs/QUALITY_AND_VALIDATION.md` のL2（対象の保持）に対応する。歌唱が強く伴奏が小さい
入力で、推定した伴奏の大きさが下がることが固定診断で測られている。参照音源は製品では
使えないので、原本と推定だけで済む読み取りにする。

考え方は次の一つだけである。原本から推定を引いた残りに、推定と同じ形の成分がどれだけ
混じっているかを一つの係数で測る。混じっていれば、その分だけ推定が取りこぼしている。

作らない・削らないを守る。

- 大きさを上げることしかしない。推定を小さくする方向へは動かさない。
- 推定に十分な音が無いときは何もしない。対象不在の出力へ音を作らない。
- 上限を先に宣言する。係数がいくら大きく出ても、その上限を超えて上げない。
- 歌唱と伴奏は完全には無相関ではないので、この係数は上振れし得る。観測であって証明では
  ない。採否は固定ケースの比較で決める。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD = "mixture-residual-scalar-level-recovery-v1"
STATES = ("recovered", "unchanged")
REASONS = (
    "residual_explains_a_missing_share_of_the_estimate",
    "no_missing_share_in_the_residual",
    "estimate_below_the_declared_floor",
    "gain_reached_the_declared_ceiling",
)
_SILENCE_FLOOR_DBFS = -300.0


class LevelRecoveryError(ValueError):
    """取りこぼしの読み取りの宣言、または渡された音声が契約に合わない。"""


@dataclass(frozen=True)
class LevelRecoveryRecipe:
    """戻してよい上限と、触ってよい最小の音量を先に固定する。"""

    minimum_estimate_dbfs: float
    maximum_gain_db: float
    minimum_gain_db: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise LevelRecoveryError("未対応の取りこぼし読み取りschemaです")
        value = self.minimum_estimate_dbfs
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or float(value) >= 0.0):
            raise LevelRecoveryError("minimum_estimate_dbfsは0未満の有限な値である必要があります")
        for field in ("maximum_gain_db", "minimum_gain_db"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or float(value) < 0.0):
                raise LevelRecoveryError(f"{field}は0以上の有限な値である必要があります")
        if self.minimum_gain_db > self.maximum_gain_db:
            raise LevelRecoveryError("戻す下限が上限を超えています")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (f"bgm-level-recovery-v{SCHEMA_VERSION}-"
                + hashlib.sha256(body.encode()).hexdigest()[:16])


def _samples(value: Any, context: str) -> np.ndarray:
    data = np.asarray(value)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
        raise LevelRecoveryError(f"{context}は(channels, frames)の音声である必要があります")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise LevelRecoveryError(f"{context}に有限でない値があります")
    return data.astype(np.float64, copy=False)


def _dbfs(value: float) -> float:
    return _SILENCE_FLOOR_DBFS if value <= 0 else max(_SILENCE_FLOOR_DBFS,
                                                      20.0 * math.log10(value))


def read_missing_level(mixture: Any, estimate: Any, recipe: LevelRecoveryRecipe) -> dict:
    """原本と推定から、推定が取りこぼした大きさを一つの係数として読む。

    音は作らない。返すのは観測と、宣言に収めた戻し量だけである。
    """
    recipe.validate()
    left, right = _samples(mixture, "mixture"), _samples(estimate, "estimate")
    if left.shape != right.shape:
        raise LevelRecoveryError("原本と推定の形が一致しません")
    energy = float(np.sum(right ** 2))
    frames = right.shape[1]
    level = _dbfs(math.sqrt(energy / max(frames * right.shape[0], 1)))
    common = {"recipe_id": recipe.recipe_id, "estimate_dbfs": level,
              "residual_share": None, "gain_db": 0.0}
    if energy <= 0 or level < recipe.minimum_estimate_dbfs:
        # 対象がほとんど無い出力へ音を作らない。
        return {"state": "unchanged", "reason": "estimate_below_the_declared_floor", **common}

    residual = left - right
    share = float(np.sum(residual * right) / energy)
    common["residual_share"] = share
    if share <= 0:
        return {"state": "unchanged", "reason": "no_missing_share_in_the_residual", **common}

    gain_db = _dbfs(1.0 + share)
    reason = "residual_explains_a_missing_share_of_the_estimate"
    if gain_db > recipe.maximum_gain_db:
        gain_db, reason = recipe.maximum_gain_db, "gain_reached_the_declared_ceiling"
    if gain_db < recipe.minimum_gain_db:
        return {"state": "unchanged", "reason": "no_missing_share_in_the_residual", **common}
    return {"state": "recovered", "reason": reason, **{**common, "gain_db": gain_db}}


def apply_reading(estimate: Any, reading: dict) -> np.ndarray:
    """読み取った戻し量を推定へ掛ける。下げる方向へは動かさない。"""
    if not isinstance(reading, dict) or reading.get("state") not in STATES:
        raise LevelRecoveryError(f"取りこぼしの読み取りが不正です: {reading!r}")
    data = _samples(estimate, "estimate")
    gain_db = reading.get("gain_db")
    if (isinstance(gain_db, bool) or not isinstance(gain_db, Real)
            or not math.isfinite(float(gain_db)) or float(gain_db) < 0.0):
        raise LevelRecoveryError("戻し量は0以上の有限な値である必要があります")
    if reading["state"] == "unchanged" or float(gain_db) == 0.0:
        return data
    return data * (10.0 ** (float(gain_db) / 20.0))
