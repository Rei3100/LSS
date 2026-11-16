"""名前付き4stemの歌唱stemの取り分から、cueに歌唱があるかを観測する。

`docs/QUALITY_AND_VALIDATION.md` の「観測属性とLSSの採用policyを分ける」に従う。
ここが返すのは観測された取り分と、先に宣言した帯に対する位置だけで、
`dramatic_score` か `singing_score` かという採用judgementではない。

二つのしきい値で挟み、その間は `unresolved` のまま残す。どちらとも言えない区間を
無理にどちらかへ倒さない。歌唱stemはモデルの出力であり、意味的正解ではない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import soundfile as sf

SCHEMA_VERSION = 1
METHOD = "named-four-stem-singing-energy-share-v1"
STATES = ("absent", "present", "unresolved")
_BLOCK = 1 << 18
_SILENCE_FLOOR_DBFS = -300.0


class SingingShareError(ValueError):
    """取り分の観測、または先に宣言したRecipeが契約に合わない。"""


def _energy(path: Path) -> tuple[float, int, int]:
    total, frames = 0.0, 0
    with sf.SoundFile(str(path)) as stream:
        channels, rate = stream.channels, stream.samplerate
        while True:
            block = stream.read(_BLOCK, dtype="float64", always_2d=True)
            if block.shape[0] == 0:
                break
            if not np.isfinite(block).all():
                raise SingingShareError(f"有限でないsampleがあります: {path}")
            total += float(np.sum(block ** 2))
            frames += block.shape[0]
    return total, frames, (channels, rate)


def measure_files(vocals_path: Path, instrumental_path: Path) -> dict:
    """歌唱stemと器楽stemのファイルから取り分を測る。判断は持たない。"""
    vocals, vocal_frames, vocal_shape = _energy(Path(vocals_path))
    instrumental, instrumental_frames, instrumental_shape = _energy(Path(instrumental_path))
    if vocal_frames != instrumental_frames or vocal_shape != instrumental_shape:
        raise SingingShareError("歌唱stemと器楽stemの時間軸・左右・rateが一致しません")
    return _share(vocals, instrumental, vocal_frames, vocal_shape)


def measure(vocals: np.ndarray, instrumental: np.ndarray) -> dict:
    """(channels, frames)の歌唱stemと器楽stemから取り分を測る。判断は持たない。"""
    left, right = np.asarray(vocals), np.asarray(instrumental)
    if left.ndim != 2 or left.shape != right.shape or left.shape[1] == 0:
        raise SingingShareError("同じ形の(channels, frames)が必要です")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise SingingShareError("有限でないsampleがあります")
    return _share(float(np.sum(left.astype(np.float64) ** 2)),
                  float(np.sum(right.astype(np.float64) ** 2)),
                  int(left.shape[1]), (int(left.shape[0]), None))


def _share(vocals: float, instrumental: float, frames: int, shape) -> dict:
    total = vocals + instrumental
    samples = max(frames * (shape[0] or 1), 1)
    rms = math.sqrt(total / samples)
    return {"schema_version": SCHEMA_VERSION, "method": METHOD,
            "singing_energy": vocals, "instrumental_energy": instrumental,
            "singing_share": (vocals / total) if total > 0 else None,
            # 取り分だけでは、音楽がほぼ無い区間の雑音どうしの比を歌唱と読み違える。
            "music_rms_dbfs": _SILENCE_FLOOR_DBFS if rms <= 0 else 20.0 * math.log10(rms),
            "music_present": total > 0, "frames": frames,
            "channels": shape[0], "samplerate": shape[1]}


@dataclass(frozen=True)
class SingingShareRecipe:
    """歌唱ありと歌唱なしを分ける帯。既定値を持たず、両端の宣言を要求する。"""

    maximum_absent_share: float
    minimum_present_share: float
    minimum_music_dbfs: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def to_dict(self) -> dict:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise SingingShareError("未対応の歌唱取り分schemaです")
        for name in ("maximum_absent_share", "minimum_present_share"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0 <= float(value) <= 1):
                raise SingingShareError(f"{name}は0以上1以下の有限な値である必要があります")
        if self.maximum_absent_share > self.minimum_present_share:
            raise SingingShareError("歌唱なしの上限が歌唱ありの下限を超えています")
        if (isinstance(self.minimum_music_dbfs, bool) or not isinstance(self.minimum_music_dbfs, Real)
                or not math.isfinite(self.minimum_music_dbfs) or self.minimum_music_dbfs >= 0):
            raise SingingShareError("minimum_music_dbfsは0未満の有限な値である必要があります")
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-singing-share-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def classify(measured: dict, recipe: SingingShareRecipe) -> dict:
    """観測した取り分を帯へ当てる。帯の間はunresolvedのまま返す。"""
    recipe.to_dict()
    if not isinstance(measured, dict) or measured.get("method") != METHOD:
        raise SingingShareError("この観測が作った値が必要です")
    share = measured["singing_share"]
    if share is None or not measured["music_present"]:
        # 音楽がまったく無い区間へ、歌唱の有無を言わない。
        state, reason = "unresolved", "no_music_energy"
    elif measured["music_rms_dbfs"] < recipe.minimum_music_dbfs:
        # 判断できるだけの音楽が無い。取り分の比だけで歌唱ありと言わない。
        state, reason = "unresolved", "music_level_too_low"
    elif share <= recipe.maximum_absent_share:
        state, reason = "absent", "below_absent_band"
    elif share >= recipe.minimum_present_share:
        state, reason = "present", "above_present_band"
    else:
        state, reason = "unresolved", "inside_undecided_band"
    return {"schema_version": SCHEMA_VERSION, "method": METHOD, "recipe_id": recipe.recipe_id,
            "singing": state, "reason": reason, "singing_share": share,
            "music_rms_dbfs": measured["music_rms_dbfs"], "is_observation_not_role": True}
