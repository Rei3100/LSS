"""分離した二つのstemの大きさの釣り合いから、その場面の種類を観測する。

`docs/PRODUCT.md` の「未対応targetは隠す、または`UNSUPPORTED`として明示し」に対応する。
固定診断では、歌唱が強く伴奏が小さい場面で品質が大きく落ちることが測られている。
出力の良し悪しを黙って渡すのではなく、その場面がどの種類かを観測として残す。

参照音源は要らない。取り出した対象のstemと、その相手のstemの大きさだけを見る。

ここで返すのは観測であって品質ではない。釣り合いが悪いことは「失敗した」ではなく、
「現在の対応範囲の弱い側にいる」という意味である。境目は呼び出し側が先に宣言する。
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
METHOD = "target-to-other-stem-balance-v1"
STATES = ("target_dominant", "balanced", "target_quiet", "target_absent", "unresolved")


class SceneBalanceError(ValueError):
    """釣り合いの宣言、または渡された音声が契約に合わない。"""


@dataclass(frozen=True)
class SceneBalanceRecipe:
    """場面の種類を分ける境目を先に固定する。"""

    absent_below_db: float
    quiet_below_db: float
    dominant_above_db: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise SceneBalanceError("未対応の釣り合い読み取りschemaです")
        for field in ("absent_below_db", "quiet_below_db", "dominant_above_db"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value))):
                raise SceneBalanceError(f"{field}は有限な値である必要があります")
        if not self.absent_below_db < self.quiet_below_db < self.dominant_above_db:
            raise SceneBalanceError("境目は不在 < 静か < 優勢の順である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (f"scene-balance-v{SCHEMA_VERSION}-"
                + hashlib.sha256(body.encode()).hexdigest()[:16])


def _samples(value: Any, context: str) -> np.ndarray:
    data = np.asarray(value)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
        raise SceneBalanceError(f"{context}は(channels, frames)の音声である必要があります")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise SceneBalanceError(f"{context}に有限でない値があります")
    return data.astype(np.float64, copy=False)


def balance_state(target_energy: float, other_energy: float,
                  recipe: SceneBalanceRecipe) -> dict:
    """大きさの組だけから帯を決める。音声を持たずに読めるようにする。"""
    recipe.validate()
    for name, value in (("target_energy", target_energy), ("other_energy", other_energy)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or float(value) < 0.0):
            raise SceneBalanceError(f"{name}は0以上の有限な値である必要があります")
    if target_energy <= 0 and other_energy <= 0:
        return {"state": "unresolved", "reason": "both_stems_are_silent",
                "recipe_id": recipe.recipe_id, "balance_db": None}
    if other_energy <= 0:
        return {"state": "target_dominant", "reason": "the_other_stem_is_silent",
                "recipe_id": recipe.recipe_id, "balance_db": None}
    if target_energy <= 0:
        return {"state": "target_absent", "reason": "the_target_stem_is_silent",
                "recipe_id": recipe.recipe_id, "balance_db": None}

    balance = 10.0 * math.log10(float(target_energy) / float(other_energy))
    if balance < recipe.absent_below_db:
        state, reason = "target_absent", "below_the_declared_absent_boundary"
    elif balance < recipe.quiet_below_db:
        state, reason = "target_quiet", "below_the_declared_quiet_boundary"
    elif balance > recipe.dominant_above_db:
        state, reason = "target_dominant", "above_the_declared_dominant_boundary"
    else:
        state, reason = "balanced", "between_the_declared_boundaries"
    return {"state": state, "reason": reason, "recipe_id": recipe.recipe_id,
            "balance_db": balance,
            # 観測であって品質ではない。弱い側にいることだけを示す。
            "is_observation_not_quality": True}


def read_balance(target: Any, other: Any, recipe: SceneBalanceRecipe) -> dict:
    """取り出した対象と相手のstemの大きさの比を、宣言した帯へ当てる。"""
    recipe.validate()
    left, right = _samples(target, "target"), _samples(other, "other")
    if left.shape != right.shape:
        raise SceneBalanceError("二つのstemの形が一致しません")
    return balance_state(float(np.sum(left ** 2)), float(np.sum(right ** 2)), recipe)


TIMED_METHOD = "target-to-other-stem-balance-over-time-v1"


@dataclass(frozen=True)
class TimedSceneBalanceRecipe:
    """長い素材を区間ごとに読むための宣言。

    素材一本の比は、場面がいくつも混ざると境目付近へ寄る。区間ごとに読み、
    どの帯にどれだけの時間がいたかで示す。境目の意味はファイル単位と同じである。
    """

    window_seconds: float
    hop_seconds: float
    absent_below_db: float
    quiet_below_db: float
    dominant_above_db: float
    schema_version: int = SCHEMA_VERSION
    method: str = TIMED_METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != TIMED_METHOD:
            raise SceneBalanceError("未対応の区間ごとの釣り合いschemaです")
        for field in ("window_seconds", "hop_seconds"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or float(value) <= 0.0):
                raise SceneBalanceError(f"{field}は有限な正数である必要があります")
        if self.hop_seconds > self.window_seconds:
            raise SceneBalanceError("hop_secondsはwindow_seconds以下である必要があります")
        self.bands.validate()

    @property
    def bands(self) -> SceneBalanceRecipe:
        return SceneBalanceRecipe(absent_below_db=self.absent_below_db,
                                  quiet_below_db=self.quiet_below_db,
                                  dominant_above_db=self.dominant_above_db)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (f"scene-balance-timed-v{SCHEMA_VERSION}-"
                + hashlib.sha256(body.encode()).hexdigest()[:16])


def read_balance_over_time(target: Any, other: Any, sample_rate: int,
                           recipe: TimedSceneBalanceRecipe) -> dict:
    """区間ごとに釣り合いを読み、どの帯にどれだけの時間がいたかを返す。

    素材一本の値だけでは、場面が混ざったときに境目付近へ寄って意味を失う。
    ここでは帯ごとの時間の割合を返す。判定はしない。
    """
    recipe.validate()
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise SceneBalanceError("sample_rateは正の整数である必要があります")
    left, right = _samples(target, "target"), _samples(other, "other")
    if left.shape != right.shape:
        raise SceneBalanceError("二つのstemの形が一致しません")
    window = max(1, int(round(recipe.window_seconds * sample_rate)))
    hop = max(1, int(round(recipe.hop_seconds * sample_rate)))
    if left.shape[1] < window:
        raise SceneBalanceError("区間ごとに読むには窓より長い音声が必要です")

    bands = recipe.bands
    counts: dict[str, int] = {}
    values = []
    total = 1 + (left.shape[1] - window) // hop
    for index in range(total):
        start = index * hop
        read = read_balance(left[:, start:start + window], right[:, start:start + window], bands)
        counts[read["state"]] = counts.get(read["state"], 0) + 1
        if read["balance_db"] is not None:
            values.append(read["balance_db"])
    return {
        "recipe_id": recipe.recipe_id,
        "window_count": total,
        "state_shares": {name: counts[name] / total for name in sorted(counts)},
        "median_balance_db": float(np.median(values)) if values else None,
        "is_observation_not_quality": True,
    }


def read_balance_from_segments(segments: list, recipe: TimedSceneBalanceRecipe, *,
                               other_energy_of) -> dict:
    """区間ごとの大きさから、ファイル1件の帯と区間ごとの割合をまとめて返す。

    音声そのものは受け取らない。長い素材でも持ち回る量が増えない。
    区間の長さは呼び出し側が宣言し、窓は連続する区間をまとめて作る。
    """
    recipe.validate()
    rows = list(segments)
    if not rows:
        raise SceneBalanceError("区間の大きさがありません")
    span = max(1, int(round(recipe.window_seconds / recipe.hop_seconds)))
    bands = recipe.bands
    target_total = sum(float(row["estimate_energy"]) for row in rows)
    other_total = sum(float(other_energy_of(row)) for row in rows)
    overall = balance_state(target_total, other_total, bands)

    counts: dict[str, int] = {}
    values = []
    windows = 0
    for start in range(0, max(1, len(rows) - span + 1)):
        group = rows[start:start + span]
        if len(group) < span and len(rows) >= span:
            break
        read = balance_state(sum(float(row["estimate_energy"]) for row in group),
                             sum(float(other_energy_of(row)) for row in group), bands)
        counts[read["state"]] = counts.get(read["state"], 0) + 1
        if read["balance_db"] is not None:
            values.append(read["balance_db"])
        windows += 1
    if windows <= 0:
        return overall
    return {**overall, "over_time": {
        "recipe_id": recipe.recipe_id, "window_count": windows,
        "state_shares": {name: counts[name] / windows for name in sorted(counts)},
        "median_balance_db": float(np.median(values)) if values else None,
        "is_observation_not_quality": True}}
