"""音楽が続く区間の中で、曲が替わる位置を探す。

1話単体の候補づくり（`episode_cues`）は、音楽が途切れずにつながる曲どうしを一つの区間に
まとめてしまう（実番組の既知の劇伴で、始まり29件中3件しか候補の端にならなかった）。
このStageは分離器の音楽stemの上で、時刻ごとに前後の違いを三つの観点で測り、合わせる。

- 大きさの段差（前後2秒）
- 音色の違い（前後8秒の平均）
- ステレオの広がりの違い（前後4秒。曲ごとに混ぜ方が違う）

一つの特徴では分けられない（実番組の正解付き境目58件と内側141点で、単独のAUCは0.57〜0.70）。
三つを話の中で標準化して平均すると0.74になる。重みは当てはめず、等しく扱う
（`.state/validation/lv-bgm-cue-boundaries-sfc-r1/`）。

返すのは「曲が替わる位置の候補」で、確定した境目ではない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from lss.bgm.spectral_features import SpectralFeatureRecipe, timbre_and_chroma

FRAME_SECONDS = 0.25
FEATURES = SpectralFeatureRecipe(n_fft=2048, stft_hop_frames=512, feature_seconds=0.5,
                                 feature_hop_seconds=FRAME_SECONDS, band_count=24,
                                 low_hz=60.0, high_hz=8000.0)


class CueBoundaryError(ValueError):
    """曲の切れ目探しの宣言か材料が契約に合わない。"""


@dataclass(frozen=True)
class CueBoundaryRecipe:
    """どの長さの区間を、どれだけ強い変化で、どの間隔以上で切るか。"""

    minimum_interval_seconds: float
    minimum_gap_seconds: float
    score_threshold: float

    def validate(self) -> None:
        for name, low, high in (("minimum_interval_seconds", 0.0, 3600.0),
                                ("minimum_gap_seconds", 1.0, 600.0), ("score_threshold", -10.0, 10.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not low <= float(value) <= high:
                raise CueBoundaryError(f"{name}は{low}以上{high}以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frame_features(data: np.ndarray, rate: int) -> dict[str, np.ndarray]:
    """0.25秒ごとの大きさ（dB）・音色・ステレオの広がり（dB）。"""
    step = int(round(FRAME_SECONDS * rate))
    count = data.shape[0] // step
    frames = data[:count * step].reshape(count, step, data.shape[1])
    level = 10.0 * np.log10(np.mean(frames ** 2, axis=(1, 2)) + 1e-20)
    if data.shape[1] >= 2:
        mid = (frames[:, :, 0] + frames[:, :, 1]) / 2
        side = (frames[:, :, 0] - frames[:, :, 1]) / 2
        width = 10.0 * np.log10((np.sum(side ** 2, axis=1) + 1e-20) / (np.sum(mid ** 2, axis=1) + 1e-20))
    else:
        width = np.zeros(count)
    timbre, _ = timbre_and_chroma(data.T, rate, FEATURES)
    length = min(count, timbre.shape[0])
    return {"level": level[:length], "width": width[:length], "timbre": timbre[:length]}


def _window_mean(values: np.ndarray, start: int, end: int) -> np.ndarray:
    return values[max(0, start):max(max(0, start) + 1, end)].mean(axis=0)


def change_scores(features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """1秒ごとの位置（区間先頭からの秒）と、三つの変化を標準化して平均した値。"""
    per_second = int(round(1.0 / FRAME_SECONDS))
    count = features["level"].shape[0]
    positions = np.arange(8 * per_second, count - 8 * per_second, per_second)
    if positions.size == 0:
        return np.array([]), np.array([])
    rows = []
    for index in positions:
        level_before = _window_mean(features["level"], index - 2 * per_second, index)
        level_after = _window_mean(features["level"], index, index + 2 * per_second)
        timbre_before = _window_mean(features["timbre"], index - 8 * per_second, index)
        timbre_after = _window_mean(features["timbre"], index, index + 8 * per_second)
        norm = float(np.linalg.norm(timbre_before) * np.linalg.norm(timbre_after))
        timbre = 1.0 - float(np.dot(timbre_before, timbre_after)) / norm if norm > 0 else 0.0
        width_before = _window_mean(features["width"], index - 4 * per_second, index)
        width_after = _window_mean(features["width"], index, index + 4 * per_second)
        rows.append((abs(float(level_after - level_before)), timbre, abs(float(width_after - width_before))))
    values = np.asarray(rows)
    spread = values.std(axis=0)
    spread[spread == 0] = 1.0
    scores = ((values - values.mean(axis=0)) / spread).mean(axis=1)
    return positions * FRAME_SECONDS, scores


def split_positions(music: Path, start_seconds: float, end_seconds: float,
                    recipe: CueBoundaryRecipe) -> list[dict]:
    """区間の中で曲が替わる位置の候補（原本の時刻）を強い順ではなく時刻順に返す。"""
    import soundfile as sf

    recipe.validate()
    if end_seconds - start_seconds < float(recipe.minimum_interval_seconds):
        return []
    with sf.SoundFile(str(music)) as stream:
        rate = stream.samplerate
        stream.seek(int(round(start_seconds * rate)))
        data = stream.read(int(round((end_seconds - start_seconds) * rate)), dtype="float64", always_2d=True)
    positions, scores = change_scores(frame_features(data, rate))
    chosen: list[int] = []
    for index in np.argsort(-scores):
        if scores[index] < recipe.score_threshold:
            break
        if all(abs(positions[index] - positions[other]) >= recipe.minimum_gap_seconds for other in chosen):
            chosen.append(int(index))
    return [{"seconds": round(start_seconds + float(positions[index]), 3),
             "score": round(float(scores[index]), 4)} for index in sorted(chosen)]
