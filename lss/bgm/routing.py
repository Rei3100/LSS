"""異なる推定の不一致で切り替えるDevelopment stage。意味役割は判定しない。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

ENERGY_BLOCK_FRAMES = 1 << 18


@dataclass(frozen=True)
class DisagreementRoutingRecipe:
    window_frames: int = 22050
    hop_frames: int = 11025
    minimum_disagreement_share: float = 0.1
    minimum_music_share: float = 0.2
    minimum_music_rms: float = 0.001
    schema_version: int = 1
    method: str = "common-stereo-disagreement-routing"
    transition: str = "linear-between-window-centers"

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("未対応のrouting schemaです")
        if self.method != "common-stereo-disagreement-routing" or self.transition != "linear-between-window-centers":
            raise ValueError("未対応のrouting methodまたはtransitionです")
        for name in ("window_frames", "hop_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name}は正の整数である必要があります")
        if self.hop_frames > self.window_frames:
            raise ValueError("hop_framesはwindow_frames以下である必要があります")
        for name in ("minimum_disagreement_share", "minimum_music_share", "minimum_music_rms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}は有限な正数である必要があります")
        if self.minimum_music_share > 1 or self.minimum_disagreement_share > 1:
            raise ValueError("shareの下限は1以下である必要があります")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-disagreement-route-v1-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RoutingResult:
    audio: np.ndarray
    music_weights: np.ndarray
    windows: tuple[dict, ...]
    recipe_id: str


def _checked_inputs(instrumental: np.ndarray, stems: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    if set(stems) != {"Music", "Speech", "Effects"}:
        raise ValueError("Music/Speech/Effectsの全推定が必要です")
    arrays = []
    for value in (instrumental, stems["Music"], stems["Speech"], stems["Effects"]):
        data = np.asarray(value)
        if (data.ndim != 2 or data.shape[0] != 2 or data.shape[1] == 0
                or not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data)
                or not np.isfinite(data).all()):
            raise ValueError("routingには空でない有限stereo配列が必要です")
        arrays.append(data.astype(np.float64, copy=False))
    parent, music, speech, effects = arrays
    if any(value.shape != parent.shape for value in arrays):
        raise ValueError("routing入力のtimelineが一致しません")
    return arrays


def _energy_prefix(values: np.ndarray, subtract: np.ndarray | None = None) -> np.ndarray:
    """二乗・差の一時配列だけを分割する。累積和の順序は全長のまま保つ。"""
    frames = values.shape[1]
    prefix = np.empty(frames + 1, dtype=np.float64)
    prefix[0] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, frames, ENERGY_BLOCK_FRAMES):
            end = min(frames, start + ENERGY_BLOCK_FRAMES)
            block = values[:, start:end]
            if subtract is not None:
                block = block - subtract[:, start:end]
            prefix[start + 1:end + 1] = np.mean(block * block, axis=0)
        # ブロックごとの和を足すと丸め順序が変わるため、同じsample順のcumsumを1回行う。
        np.cumsum(prefix[1:], out=prefix[1:])
    return prefix


def _window_evidence(arrays: list[np.ndarray], recipe: DisagreementRoutingRecipe) -> tuple[dict, ...]:
    parent, music, speech, effects = arrays
    frames = parent.shape[1]
    # prefix sumは同じsample gridで全窓を集計する。resampleや正規化はしない。
    prefixes = [_energy_prefix(value) for value in (music, speech, effects)]
    prefixes.append(_energy_prefix(music, subtract=parent))
    if any(not np.isfinite(value).all() for value in prefixes):
        raise ValueError("routingのenergy集計が有限範囲を超えました")
    centers = np.arange(0, frames, recipe.hop_frames, dtype=np.int64)
    observations = []
    left_radius = recipe.window_frames // 2
    for center in centers:
        start = max(0, int(center) - left_radius)
        end = min(frames, int(center) + recipe.window_frames - left_radius)
        music_energy, speech_energy, effects_energy, difference_energy = [
            max(0.0, float(value[end] - value[start])) for value in prefixes
        ]
        total = music_energy + speech_energy + effects_energy
        share = music_energy / total if total > 0 else 0.0
        disagreement = difference_energy / music_energy if music_energy > 0 else 0.0
        rms = float(np.sqrt(music_energy / (end - start)))
        use_music = (rms >= recipe.minimum_music_rms and share >= recipe.minimum_music_share
                     and disagreement >= recipe.minimum_disagreement_share)
        observations.append({"center_frame": int(center), "start_frame": start, "end_frame": end,
                             "music_share": share, "disagreement_share": disagreement,
                             "music_rms": rms, "use_music": bool(use_music)})
    return tuple(observations)


def analyze_stereo(instrumental: np.ndarray, stems: Mapping[str, np.ndarray],
                   recipe: DisagreementRoutingRecipe) -> tuple[dict, ...]:
    """窓の判断だけを返す。区間分岐のために全長の中間音声・係数を作らない。"""
    recipe.validate()
    return _window_evidence(_checked_inputs(instrumental, stems), recipe)


def route_stereo(instrumental: np.ndarray, stems: Mapping[str, np.ndarray],
                 recipe: DisagreementRoutingRecipe) -> RoutingResult:
    """同じtimelineの推定だけから左右共通の係数を作る。入力・gain・形式は不変。

    MusicとInstrumentalの差は歌唱の証明ではない。Speech/Effectsによる説明も
    同じproviderの証拠であり、独立した意味検証には使えない。
    """
    recipe.validate()
    arrays = _checked_inputs(instrumental, stems)
    parent, music = arrays[:2]
    observations = _window_evidence(arrays, recipe)
    centers = [row["center_frame"] for row in observations]
    decisions = [float(row["use_music"]) for row in observations]
    weights = np.interp(np.arange(parent.shape[1]), centers, decisions)
    routed = (1 - weights[np.newaxis, :]) * parent + weights[np.newaxis, :] * music
    if not np.isfinite(routed).all():
        raise ValueError("routing結果が有限ではありません")
    return RoutingResult(routed, weights, observations, recipe.recipe_id)
