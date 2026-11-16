"""相補stemがなくても可能な、補正しないsample alignmentの観測。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

POLICY = {"maximum_lag_frames": 32, "minimum_correlation": 0.99,
          "minimum_peak_margin": 1e-6, "window_frames": 44100,
          "fractions": [0.0, 0.25, 0.5, 0.75, 1.0], "channels": "independently_observed",
          "time_or_gain_correction": "none", "DC_policy": "subtract_each_window_mean_for_diagnostic_only"}


def observe_pair(reference: np.ndarray, candidate: np.ndarray) -> list[dict]:
    if np.iscomplexobj(reference) or np.iscomplexobj(candidate):
        raise ValueError("alignment観測には実数音声が必要です")
    reference, candidate = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape[0] != 2 or reference.shape[1] == 0
            or reference.shape != candidate.shape or not np.isfinite(reference).all()
            or not np.isfinite(candidate).all()):
        raise ValueError("alignment観測には同じ時間軸の有限ステレオが必要です")
    limit = min(POLICY["maximum_lag_frames"], (reference.shape[1] - 1) // 4)
    frames, observations = reference.shape[1], []
    for channel in (0, 1):
        raw = reference[channel] - reference[channel].mean()
        estimate = candidate[channel] - candidate[channel].mean()
        if np.sum(raw ** 2) == 0 or np.sum(estimate ** 2) == 0:
            observations.append({"channel": channel, "status": "unresolved_no_time_information",
                                 "best_lag_frames": None, "zero_lag_correlation": None})
            continue
        scores = []
        for lag in range(-limit, limit + 1):
            left = raw[max(0, lag):min(frames, frames + lag)]
            right = estimate[max(0, -lag):min(frames, frames - lag)]
            denominator = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
            scores.append(float(np.sum(left * right) / denominator) if denominator else 0.0)
        best = int(np.argmax(scores))
        margin = scores[best] - sorted(scores)[-2] if len(scores) > 1 else 0.0
        if scores[best] < POLICY["minimum_correlation"]:
            status = "unresolved_low_correlation"
        elif margin < POLICY["minimum_peak_margin"]:
            status = "unresolved_ambiguous_peak"
        else:
            status = "observed_zero_lag" if best == limit else "observed_nonzero_lag"
        observations.append({"channel": channel, "status": status, "best_lag_frames": best - limit,
                             "zero_lag_correlation": scores[limit], "best_correlation": scores[best],
                             "peak_margin": margin, "maximum_lag_frames": limit})
    return observations


def observe_files(reference: Path, candidates: list[Path], *, gains: list[float] | None = None) -> list[dict]:
    """同じ時間軸の複数stemの和も観測できる。和の一致は役割や品質の証明ではない。"""
    if not candidates:
        raise ValueError("観測するcacheが必要です")
    infos = [sf.info(path) for path in [reference] + candidates]
    if (infos[0].frames <= 0 or any(info.samplerate != 44100 or info.channels != 2
            or info.frames != infos[0].frames for info in infos)):
        raise ValueError("全cacheは同じ44.1kHz stereo timelineが必要です")
    scales = gains if gains is not None else [1.0] * len(candidates)
    if len(scales) != len(candidates) or any(not np.isfinite(value) or value <= 0 for value in scales):
        raise ValueError("各cacheの宣言済み正の利得が必要です")
    width = min(POLICY["window_frames"], infos[0].frames)
    starts = sorted({int((infos[0].frames - width) * fraction) for fraction in POLICY["fractions"]})
    results = []
    for start in starts:
        raw = sf.read(reference, start=start, frames=width, dtype="float64", always_2d=True)[0].T
        estimate = sum(sf.read(path, start=start, frames=width, dtype="float64", always_2d=True)[0].T * gain
                       for path, gain in zip(candidates, scales))
        results.append({"start_frame": start, "frames": width, "channels": observe_pair(raw, estimate)})
    return results
