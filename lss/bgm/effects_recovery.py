"""Effectsに残る持続成分を回収するDevelopment Stage。意味役割を判定しない。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
from scipy.ndimage import maximum_filter, median_filter, uniform_filter
from scipy.signal import istft, stft


@dataclass(frozen=True)
class EffectsRecoveryRecipe:
    n_fft: int = 2048
    hop_frames: int = 512
    time_median_frames: int = 31
    frequency_median_bins: int = 31
    harmonic_margin: float = 2.0
    require_music_support: bool = True
    support_time_frames: int = 15
    support_frequency_bins: int = 5
    minimum_phase_coherence: float | None = None
    coherence_time_frames: int = 9
    coherence_frequency_bins: int = 3
    schema_version: int = 1
    method: str = "effects-persistence-recovery"

    def validate(self) -> None:
        if self.schema_version != 1 or self.method != "effects-persistence-recovery":
            raise ValueError("未対応のEffects回収Recipeです")
        for name in ("n_fft", "hop_frames", "time_median_frames", "frequency_median_bins",
                     "support_time_frames", "support_frequency_bins", "coherence_time_frames",
                     "coherence_frequency_bins"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}には正の整数が必要です")
        if self.n_fft < 4 or self.hop_frames > self.n_fft // 2:
            raise ValueError("FFTとhopの関係が不正です")
        if type(self.require_music_support) is not bool:
            raise ValueError("Music支持の有無にはboolが必要です")
        if not np.isfinite(self.harmonic_margin) or self.harmonic_margin < 1:
            raise ValueError("持続成分のmarginは1以上の有限値が必要です")
        if self.minimum_phase_coherence is not None and (
                isinstance(self.minimum_phase_coherence, bool)
                or not isinstance(self.minimum_phase_coherence, (int, float))
                or not np.isfinite(self.minimum_phase_coherence)
                or not 0 <= self.minimum_phase_coherence <= 1):
            raise ValueError("位相coherenceには0〜1の有限値が必要です")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        content = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return "bgm-effects-recovery-v1-" + hashlib.sha256(content).hexdigest()[:16]


def recover_effects(original: np.ndarray, estimates: Mapping[str, np.ndarray],
                    recipe: EffectsRecoveryRecipe) -> tuple[dict[str, np.ndarray], dict]:
    """同じ左右maskでEffectsからMusicへ移す。原本・時間軸・3stemの和を保持する。

    このStageは候補のみ。持続する効果音や台詞も回収し得るので、採用には不在・歌唱・
    stereoの比較が必要。Music支持は回収の条件であり、元のMusicを削除しない。
    """
    recipe.validate()
    source = np.asarray(original)
    if (source.ndim != 2 or source.shape[0] != 2 or source.shape[1] == 0
            or source.dtype.kind not in "fiu" or not np.isfinite(source).all()
            or set(estimates) != {"music", "speech", "effects"}):
        raise ValueError("原本と同じ時間軸の有限ステレオ3stemが必要です")
    result = {}
    for role, values in estimates.items():
        values = np.asarray(values)
        if values.shape != source.shape or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("原本と同じ時間軸の有限ステレオ3stemが必要です")
        result[role] = values.astype(np.float64, copy=True)
    frames = source.shape[1]
    padded_frames = max(frames, recipe.n_fft)
    options = {"nperseg": recipe.n_fft, "noverlap": recipe.n_fft - recipe.hop_frames,
               "boundary": "zeros", "padded": True, "axis": -1}
    effects = np.pad(result["effects"], ((0, 0), (0, padded_frames - frames)))
    _, _, spectrum = stft(effects, **options)
    magnitude = np.sqrt(np.sum(np.abs(spectrum) ** 2, axis=0))
    harmonic = median_filter(magnitude, size=(1, recipe.time_median_frames), mode="nearest")
    transient = median_filter(magnitude, size=(recipe.frequency_median_bins, 1), mode="nearest")
    mask = np.divide(harmonic ** 2, harmonic ** 2 + (recipe.harmonic_margin * transient) ** 2,
                     out=np.zeros_like(harmonic), where=harmonic ** 2 + transient ** 2 > 0)
    music_spectrum = None
    if recipe.require_music_support or recipe.minimum_phase_coherence is not None:
        music = np.pad(result["music"], ((0, 0), (0, padded_frames - frames)))
        _, _, music_spectrum = stft(music, **options)
    if recipe.require_music_support:
        support = maximum_filter(np.sqrt(np.sum(np.abs(music_spectrum) ** 2, axis=0)),
                                 size=(recipe.support_frequency_bins, recipe.support_time_frames), mode="nearest")
        mask *= np.divide(support ** 2, support ** 2 + magnitude ** 2,
                          out=np.zeros_like(support), where=support ** 2 + magnitude ** 2 > 0)
    if recipe.minimum_phase_coherence is not None:
        size = (recipe.coherence_frequency_bins, recipe.coherence_time_frames)
        cross = uniform_filter(np.real(np.sum(music_spectrum * np.conj(spectrum), axis=0)),
                               size=size, mode="nearest")
        music_power = uniform_filter(np.sum(np.abs(music_spectrum) ** 2, axis=0), size=size, mode="nearest")
        effects_power = uniform_filter(magnitude ** 2, size=size, mode="nearest")
        denominator = np.sqrt(np.maximum(music_power, 0) * np.maximum(effects_power, 0))
        coherence = np.clip(np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0), -1, 1)
        minimum = recipe.minimum_phase_coherence
        if minimum == 1:
            mask *= coherence >= 1 - 1e-12
        else:
            mask *= np.clip((coherence - minimum) / (1 - minimum), 0, 1)
    _, recovered = istft(spectrum * mask[None, :, :], nperseg=recipe.n_fft,
                         noverlap=recipe.n_fft - recipe.hop_frames, input_onesided=True,
                         boundary=True, time_axis=-1, freq_axis=-2)
    recovered = recovered[:, :frames]
    if recovered.shape != source.shape or not np.isfinite(recovered).all():
        raise ValueError("Effects回収の結果が原本の有限ステレオ時間軸と一致しません")
    result["music"] += recovered
    result["effects"] -= recovered
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("Effects回収の加算が有限範囲を超えました")
    return result, {"recipe_id": recipe.recipe_id, "recovered_energy": float(np.sum(recovered ** 2)),
                    "mean_mask": float(np.mean(mask)), "semantic_role": "unresolved",
                    "publication_allowed": False}
