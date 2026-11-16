"""原本から係数を推定する参照録音条件付きの再構成。Development専用。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.signal import correlate, correlation_lags

from lss.bgm.identity import RecordingMatch


@dataclass(frozen=True)
class RecordingRecoveryRecipe:
    sample_rate: int = 44100
    fir_taps: int = 9
    fit_window_frames: int = 176400
    fit_hop_frames: int = 88200
    boundary_fade_frames: int = 4410
    alignment_probe_frames: int = 88200
    maximum_alignment_refinement_frames: int = 88
    timestamp_tolerance_frames: int = 44
    ridge: float = 0.001
    blend: float = 1.0
    schema_version: int = 1
    method: str = "recording-conditioned-source-reconstruction"

    def to_dict(self) -> dict:
        for name in ("sample_rate", "fir_taps", "fit_window_frames", "fit_hop_frames", "alignment_probe_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name}には正の整数が必要です")
        for name in ("boundary_fade_frames", "maximum_alignment_refinement_frames", "timestamp_tolerance_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name}には非負の整数が必要です")
        if (self.fir_taps % 2 != 1 or self.fit_hop_frames > self.fit_window_frames // 2
                or self.fit_window_frames < self.fir_taps):
            raise ValueError("FIRと重ねる窓の関係が不正です")
        if (type(self.ridge) not in (int, float) or not np.isfinite(self.ridge) or self.ridge <= 0
                or type(self.blend) not in (int, float) or not np.isfinite(self.blend) or not 0 < self.blend <= 1
                or self.schema_version != 1 or self.method != "recording-conditioned-source-reconstruction"):
            raise ValueError("未対応または不正な録音再構成Recipeです")
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        content = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-recording-recovery-v1-" + hashlib.sha256(content.encode()).hexdigest()[:16]


def _stereo(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if (data.ndim != 2 or data.shape[0] != 2 or data.shape[1] == 0
            or data.dtype.kind not in "fiu" or not np.isfinite(data).all()):
        raise ValueError("同じrateの有限ステレオ音声が必要です")
    return data.astype(np.float64, copy=False)


def reconstruct_recording(
    original: np.ndarray, music: np.ndarray, reference: np.ndarray,
    match: RecordingMatch, recipe: RecordingRecoveryRecipe,
) -> tuple[np.ndarray, dict]:
    """正の録音一致区間だけを再構成し、他の区間は元のMusicを保つ。

    参照は音の基底として使う。独立した左右FIR係数は必ず原本から推定し、基底をそのまま
    コピーしない。これは参照条件付き再構成であり、意味判定・正式公開を行わない。
    呼出側は参照・原本・一致根拠のhashとrate、query時刻の原本への対応を検証する。
    """
    recipe.to_dict()
    source, base, ref = map(_stereo, (original, music, reference))
    if source.shape != base.shape:
        raise ValueError("原本とMusicの時間軸が一致しません")
    times = (match.query_start_seconds, match.query_end_seconds, match.reference_start_seconds)
    if (match.identity_state != "known_exact" or any(
            type(value) not in (int, float) or not np.isfinite(value) for value in times)):
        raise ValueError("正の既知録音一致と有限の時刻が必要です")
    start, end, ref_start = [round(value * recipe.sample_rate) for value in times]
    tolerance = recipe.timestamp_tolerance_frames
    if (not 0 <= start < end <= source.shape[1] + tolerance or ref_start < 0
            or ref_start + end - start > ref.shape[1] + tolerance):
        raise ValueError("録音一致区間が原本・参照の時間軸から外れています")
    # 照合記録の時刻はミリ秒に丸められている。裏付けのない端は原候補のまま残す。
    end = min(end, source.shape[1], start + ref.shape[1] - ref_start)
    if end <= start:
        raise ValueError("録音一致区間に原本と参照の重なりがありません")
    length = end - start
    probe = min(length, recipe.alignment_probe_frames)
    score = sum(correlate(source[ch, start:start + probe], ref[ch, ref_start:ref_start + probe],
                          mode="full", method="fft") for ch in range(2))
    lags = correlation_lags(probe, probe, mode="full")
    allowed = np.abs(lags) <= recipe.maximum_alignment_refinement_frames
    evidence = np.abs(score[allowed])
    # 冒頭が無音なら全lagの相関がゼロになる。先頭の候補lagをずれの根拠にしない。
    lag = int(lags[allowed][np.argmax(evidence)]) if np.any(evidence) else 0
    refined = ref_start - lag
    # ミリ秒に丸めた一致時刻を詰める際、参照の外側をゼロや別の音で補完しない。
    head = max(0, -refined)
    tail = max(0, refined + length - ref.shape[1])
    start, end, ref_start = start + head, end - tail, refined + head
    length = end - start
    if length <= 0 or not np.any(ref[:, ref_start:ref_start + length]):
        raise ValueError("録音再構成に使える参照区間がありません")
    half = recipe.fir_taps // 2
    aligned = ref[:, ref_start:ref_start + length]
    padded = np.pad(aligned, ((0, 0), (half, half)))
    predicted = np.zeros((2, length), np.float64)
    weight_sum = np.zeros(length, np.float64)
    coefficients = []
    for begin in range(-recipe.fit_hop_frames, length, recipe.fit_hop_frames):
        lo, hi = max(begin, 0), min(begin + recipe.fit_window_frames, length)
        if hi <= lo:
            continue
        # 窓の端を非ゼロにして短い区間も覆う。左右で同じ重み・窓位置を使う。
        full_weight = np.hanning(recipe.fit_window_frames + 2)[1:-1]
        weight = full_weight[lo - begin:hi - begin]
        row = []
        for ch in range(2):
            design = np.lib.stride_tricks.sliding_window_view(padded[ch, lo:hi + recipe.fir_taps - 1], recipe.fir_taps)
            gram = design.T @ (weight[:, None] * design)
            regularizer = max(recipe.ridge * float(np.trace(gram)) / recipe.fir_taps, 1e-12)
            coeff = np.linalg.solve(gram + regularizer * np.eye(recipe.fir_taps),
                                    design.T @ (weight * source[ch, start + lo:start + hi]))
            predicted[ch, lo:hi] += weight * (design @ coeff)
            row.append(coeff.tolist())
        coefficients.append({"start_frame": start + lo, "end_frame": start + hi, "left_right_fir": row})
        weight_sum[lo:hi] += weight
    if np.any(weight_sum <= 0):
        raise ValueError("録音再構成の窓が区間を覆っていません")
    predicted /= weight_sum[None, :]
    fade = min(recipe.boundary_fade_frames, length // 2)
    envelope = np.full(length, recipe.blend, np.float64)
    if fade:
        ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        envelope[:fade] *= ramp
        envelope[-fade:] *= ramp[::-1]
    result = base.copy()
    result[:, start:end] = (1 - envelope) * base[:, start:end] + envelope * predicted
    if not np.isfinite(result).all():
        raise ValueError("録音再構成に非有限値があります")
    return result, {
        "recipe_id": recipe.recipe_id,
        "artifact_kind": "reference_conditioned_reconstruction",
        "semantic_role": "unresolved", "publication_allowed": False,
        "reference_id": match.reference_id, "reference_sha256": match.reference_sha256,
        "source_start_frame": start, "source_end_frame": end, "reference_start_frame": ref_start,
        "alignment_refinement_frames": lag,
        "source_reference_explained": max(0.0, 1.0 - float(np.sum((source[:, start:end] - predicted) ** 2))
                                          / max(float(np.sum(source[:, start:end] ** 2)), 1e-30)),
        "local_fir": coefficients,
    }
