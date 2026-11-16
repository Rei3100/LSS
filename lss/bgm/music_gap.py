"""候補の内側で音楽が一度止まる箇所を、Music推定だけから見つける。

音楽が止まってまた始まるなら、その前後は別のcueである。ここでは「止まった」と言える
長さと下限を先に宣言し、観測として返すだけである。意味的role・同一性・公開可否は
扱わない。

Music推定はモデルの出力なので、短い脱落を止まりと読み違え得る。だから最小の長さを
宣言で要求する。それでも作曲上の長い休符を止まりと読む可能性は残る。ここで返すのは
根拠であって証明ではない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD = "music-estimate-interior-gap-v1"
_SILENCE_FLOOR_DBFS = -200.0


class MusicGapError(ValueError):
    """止まりの読み取りの宣言、または渡された音声が契約に合わない。"""


@dataclass(frozen=True)
class MusicGapRecipe:
    window_seconds: float
    hop_seconds: float
    floor_dbfs: float
    minimum_gap_seconds: float
    edge_margin_seconds: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise MusicGapError("未対応の止まり読み取りschemaです")
        for field in ("window_seconds", "hop_seconds", "minimum_gap_seconds"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise MusicGapError(f"{field}は有限な正数である必要があります")
        value = self.edge_margin_seconds
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise MusicGapError("edge_margin_secondsは0以上の有限な秒数である必要があります")
        if self.hop_seconds > self.window_seconds:
            raise MusicGapError("hop_secondsはwindow_seconds以下である必要があります")
        value = self.floor_dbfs
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value >= 0):
            raise MusicGapError("floor_dbfsは0未満の有限な値である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-music-gap-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _levels(samples: np.ndarray, sample_rate: int, recipe: MusicGapRecipe) -> tuple[np.ndarray, int, int]:
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
        raise MusicGapError("(channels, frames)の音声が必要です")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise MusicGapError("音声に有限でない値があります")
    window = max(1, int(round(recipe.window_seconds * sample_rate)))
    hop = max(1, int(round(recipe.hop_seconds * sample_rate)))
    if data.shape[1] < window:
        raise MusicGapError("止まりを測るには窓より長い音声が必要です")
    values = data.astype(np.float64, copy=False)
    count = 1 + (values.shape[1] - window) // hop
    rms = np.array([float(np.sqrt(np.mean(np.square(values[:, index * hop:index * hop + window]))))
                    for index in range(count)])
    with np.errstate(divide="ignore"):
        levels = np.where(rms > 0, 20.0 * np.log10(np.maximum(rms, 1e-300)), _SILENCE_FLOOR_DBFS)
    return np.maximum(levels, _SILENCE_FLOOR_DBFS), window, hop


def find_interior_gaps(
    samples: np.ndarray, sample_rate: int, recipe: MusicGapRecipe,
) -> dict:
    """候補の内側だけを見る。端の立ち上がり・余韻は止まりとして数えない。"""
    recipe.validate()
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise MusicGapError("sample_rateは正の整数である必要があります")
    levels, window, hop = _levels(samples, sample_rate, recipe)
    seconds = np.arange(levels.size) * hop / sample_rate
    duration = np.asarray(samples).shape[-1] / sample_rate
    below = levels < recipe.floor_dbfs

    gaps = []
    index = 0
    while index < levels.size:
        if not below[index]:
            index += 1
            continue
        last = index
        while last + 1 < levels.size and below[last + 1]:
            last += 1
        start = float(seconds[index])
        end = float(seconds[last] + window / sample_rate)
        index = last + 1
        if end - start < recipe.minimum_gap_seconds:
            continue
        if start <= recipe.edge_margin_seconds or end >= duration - recipe.edge_margin_seconds:
            continue
        gaps.append({"start_seconds": start, "end_seconds": end, "seconds": end - start,
                     "minimum_dbfs": float(levels[last])})
    return {
        "recipe_id": recipe.recipe_id,
        "duration_seconds": duration,
        "level_dbfs_median": float(np.median(levels)),
        "interior_gaps": gaps,
    }


def extent_from_gaps(reading: dict) -> dict:
    """止まりの観測を、cueの範囲の読み取りと同じ形へ直す。"""
    gaps = reading.get("interior_gaps")
    if not isinstance(gaps, list):
        raise MusicGapError("止まりの観測がありません")
    evidence = {"interior_gap_count": len(gaps), "interior_gaps": gaps,
                "music_gap_recipe_id": reading.get("recipe_id"),
                "level_dbfs_median": reading.get("level_dbfs_median")}
    if gaps:
        # 音楽が止まってまた始まる以上、その前後を一つのcueとしては公開できない。
        return {"state": "multiple_cues", "evidence": {**evidence, "reason": "music_stops_inside"}}
    return {"state": "unresolved", "evidence": {**evidence, "reason": "no_interior_music_gap"}}


RELATIVE_DIP_METHOD = "music-estimate-relative-dip-v1"


@dataclass(frozen=True)
class MusicDipRecipe:
    """音楽が止まらずに減衰して立ち上がる形を、候補自身の音量に対して測る。

    完成mixでは、曲の終わりと次の始まりが無音を挟まずに繋がることがある。絶対の下限では
    捉えられないので、候補自身の中央値からどれだけ下がったかで見る。観測であって、
    曲の境目の証明ではない。
    """

    window_seconds: float
    hop_seconds: float
    depth_db: float
    rise_db: float
    rise_window_seconds: float
    minimum_separation_seconds: float
    edge_margin_seconds: float
    schema_version: int = SCHEMA_VERSION
    method: str = RELATIVE_DIP_METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != RELATIVE_DIP_METHOD:
            raise MusicGapError("未対応の落ち込み読み取りschemaです")
        for field in ("window_seconds", "hop_seconds", "depth_db", "rise_db",
                      "rise_window_seconds", "minimum_separation_seconds"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value <= 0):
                raise MusicGapError(f"{field}は有限な正数である必要があります")
        value = self.edge_margin_seconds
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise MusicGapError("edge_margin_secondsは0以上の有限な秒数である必要があります")
        if self.hop_seconds > self.window_seconds:
            raise MusicGapError("hop_secondsはwindow_seconds以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-music-dip-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def find_relative_dips(
    samples: np.ndarray, sample_rate: int, recipe: MusicDipRecipe,
) -> dict:
    """候補自身の中央値から深く落ち、両側で立ち上がる箇所を強い順に返す。"""
    recipe.validate()
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise MusicGapError("sample_rateは正の整数である必要があります")
    levels, window, hop = _levels(samples, sample_rate, recipe)
    seconds = np.arange(levels.size) * hop / sample_rate
    duration = np.asarray(samples).shape[-1] / sample_rate
    median = float(np.median(levels))
    span = max(1, int(round(recipe.rise_window_seconds * sample_rate / hop)))
    separation = max(1, int(round(recipe.minimum_separation_seconds * sample_rate / hop)))

    found = []
    for index in range(levels.size):
        position = float(seconds[index])
        if position <= recipe.edge_margin_seconds or position >= duration - recipe.edge_margin_seconds:
            continue
        depth = median - float(levels[index])
        if depth < recipe.depth_db:
            continue
        before = levels[max(0, index - span):index]
        after = levels[index + 1:index + 1 + span]
        if before.size == 0 or after.size == 0:
            continue
        if (float(before.max()) - float(levels[index]) < recipe.rise_db
                or float(after.max()) - float(levels[index]) < recipe.rise_db):
            continue
        found.append({"seconds": position, "level_dbfs": float(levels[index]),
                      "depth_db": depth})

    chosen: list[dict] = []
    for row in sorted(found, key=lambda item: (-item["depth_db"], item["seconds"])):
        index = int(round(row["seconds"] * sample_rate / hop))
        if all(abs(index - int(round(other["seconds"] * sample_rate / hop))) >= separation
               for other in chosen):
            chosen.append(row)
    return {
        "recipe_id": recipe.recipe_id,
        "duration_seconds": duration,
        "level_dbfs_median": median,
        "dips": sorted(chosen, key=lambda item: item["seconds"]),
    }
