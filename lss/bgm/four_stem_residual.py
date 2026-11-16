"""4stem推定で説明できる原本残差だけを既存Musicへ補うDevelopment Stage。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FourStemResidualRecipe:
    basis: str
    blend: float = 0.5
    window_frames: int = 44100
    hop_frames: int = 22050
    minimum_explained_fraction: float = 0.1
    minimum_channel_explained_fraction: float | None = None
    schema_version: int = 1
    method: str = "source-fitted-four-stem-positive-residual"

    def to_dict(self) -> dict:
        if (self.schema_version != 1 or self.method != "source-fitted-four-stem-positive-residual"
                or self.basis not in {"music_vocals", "combined_music"}
                or isinstance(self.blend, bool) or not isinstance(self.blend, (int, float))
                or not np.isfinite(self.blend) or not 0 < self.blend <= 1
                or type(self.window_frames) is not int or self.window_frames <= 0
                or type(self.hop_frames) is not int or not 0 < self.hop_frames <= self.window_frames
                or isinstance(self.minimum_explained_fraction, bool)
                or not isinstance(self.minimum_explained_fraction, (int, float))
                or not np.isfinite(self.minimum_explained_fraction)
                or not 0 <= self.minimum_explained_fraction <= 1
                or self.minimum_channel_explained_fraction is not None and (
                    isinstance(self.minimum_channel_explained_fraction, bool)
                    or not isinstance(self.minimum_channel_explained_fraction, (int, float))
                    or not np.isfinite(self.minimum_channel_explained_fraction)
                    or not 0 <= self.minimum_channel_explained_fraction <= 1)):
            raise ValueError("4stem残差回収Recipeが不正です")
        payload = asdict(self)
        if self.minimum_channel_explained_fraction is None:
            # 既に固定したRecipe IDを、追加gateを使わない場合は維持する。
            payload.pop("minimum_channel_explained_fraction")
        return payload

    @property
    def recipe_id(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-four-stem-residual-v1-" + hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _stereo(value, shape=None) -> np.ndarray:
    result = np.asarray(value)
    if (result.ndim != 2 or result.shape[0] != 2 or result.shape[1] == 0
            or result.dtype.kind not in "fiu" or not np.isfinite(result).all()
            or shape is not None and result.shape != shape):
        raise ValueError("同じ時間軸の有限ステレオが必要です")
    return result.astype(np.float64, copy=False)


def recover_four_stem_residual(original: np.ndarray, music: np.ndarray,
                               estimates: Mapping[str, np.ndarray],
                               recipe: FourStemResidualRecipe) -> tuple[np.ndarray, dict]:
    """参照や正解labelを見ず、原本−既存Musicを正に説明する4stem基底だけを加える。

    係数は左右共通で0..1。局所残差の二乗誤差を10%以上説明できる窓だけを使う。
    これは意味役割の判定ではなく、台詞等の漏れも回収し得るためDevelopment候補に限る。
    """
    recipe.to_dict()
    source = _stereo(original)
    base = _stereo(music, source.shape)
    required = {"music_instrumental", "music_vocals", "speech", "sfx"}
    if set(estimates) != required:
        raise ValueError("名前付き4stem推定がすべて必要です")
    values = {role:_stereo(value, source.shape) for role,value in estimates.items()}
    basis = (values["music_vocals"] if recipe.basis == "music_vocals"
             else values["music_instrumental"] + values["music_vocals"])
    residual = source - base
    accumulated = np.zeros_like(base)
    normalization = np.zeros(source.shape[1], dtype=np.float64)
    windows = []
    for start in range(0, source.shape[1], recipe.hop_frames):
        end = min(source.shape[1], start + recipe.window_frames)
        length = end - start
        weight = np.sin(np.pi * (np.arange(length) + 0.5) / length) ** 2
        x, y = basis[:, start:end], residual[:, start:end]
        dot = float(np.sum(weight * np.sum(x * y, axis=0)))
        denominator = float(np.sum(weight * np.sum(x * x, axis=0)))
        residual_energy = float(np.sum(weight * np.sum(y * y, axis=0)))
        alpha = float(np.clip(dot / denominator, 0, 1)) if denominator > 0 else 0.0
        reduction = max(0.0, 2 * alpha * dot - alpha * alpha * denominator)
        explained = reduction / residual_energy if residual_energy > 0 else 0.0
        channels = []
        for channel in range(2):
            channel_dot = float(np.sum(weight * x[channel] * y[channel]))
            channel_denominator = float(np.sum(weight * x[channel] * x[channel]))
            channel_residual_energy = float(np.sum(weight * y[channel] * y[channel]))
            channel_reduction = max(0.0, 2 * alpha * channel_dot - alpha * alpha * channel_denominator)
            if channel_residual_energy > 0:
                channels.append(channel_reduction / channel_residual_energy)
            else:
                channels.append(1.0 if alpha * alpha * channel_denominator <= 1e-30 else 0.0)
        channel_gate = (recipe.minimum_channel_explained_fraction is None
                        or all(value >= recipe.minimum_channel_explained_fraction for value in channels))
        accepted = explained >= recipe.minimum_explained_fraction and alpha > 0 and channel_gate
        if accepted:
            accumulated[:, start:end] += weight * (recipe.blend * alpha * x)
            normalization[start:end] += weight
        windows.append({"start_frame":start, "end_frame":end, "alpha":alpha,
                        "explained_fraction":explained,
                        "channel_explained_fraction":channels, "accepted":bool(accepted)})
        if end == source.shape[1]:
            break
    contribution = np.divide(accumulated, normalization, out=np.zeros_like(accumulated),
                             where=normalization > 0)
    candidate = base + contribution
    before_error = float(np.sum(residual ** 2))
    after_error = float(np.sum((source - candidate) ** 2))
    fallback = after_error > before_error + max(1e-12, before_error * 1e-12)
    if fallback:
        candidate = base.copy()
        contribution.fill(0)
        after_error = before_error
    if not np.isfinite(candidate).all():
        raise ValueError("4stem残差回収に非有限値があります")
    return candidate, {"recipe_id":recipe.recipe_id,
        "artifact_kind":"source_fitted_four_stem_residual_candidate",
        "semantic_role":"unresolved", "publication_allowed":False,
        "basis":recipe.basis, "window_count":len(windows),
        "accepted_window_count":sum(row["accepted"] for row in windows),
        "source_error_before":before_error, "source_error_after":after_error,
        "recovered_energy":float(np.sum(contribution ** 2)),
        "source_consistency_fallback":fallback, "windows":windows}
