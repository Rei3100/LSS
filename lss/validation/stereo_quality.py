"""ステレオtargetの保持と誤差を分けて測るDevelopment指標。"""
from __future__ import annotations

import numpy as np


def energy(values: np.ndarray) -> float:
    return float(np.sum(np.square(values, dtype=np.float64)))


def db_ratio(numerator: float, denominator: float) -> float:
    # 0の比は有限floorで保存し、無音targetの品質指標自体はNoneにする。
    return float(10 * np.log10(max(numerator, 1e-30) / max(denominator, 1e-30)))


def _projection(estimate: np.ndarray, reference: np.ndarray) -> tuple[float | None, float | None]:
    reference_energy = energy(reference)
    if reference_energy == 0:
        return None, None
    gain = float(np.sum(estimate * reference, dtype=np.float64) / reference_energy)
    if gain <= 0:
        return -300.0, None
    return db_ratio(energy(gain * reference), energy(estimate - gain * reference)), float(20 * np.log10(gain))


def stereo_quality(estimate: np.ndarray, reference: np.ndarray) -> dict:
    """(2,frames)。gain/EQ/time補正を施さず、SIと非scale-invariant誤差を併記。"""
    estimate, reference = np.asarray(estimate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (
        estimate.ndim != 2 or estimate.shape[0] != 2 or estimate.shape[1] == 0
        or estimate.shape != reference.shape
        or not np.isfinite(estimate).all() or not np.isfinite(reference).all()
    ):
        raise ValueError("品質比較には同じtimelineの有限stereo配列が必要です")
    pairs = {
        "joint": (estimate, reference),
        "left": (estimate[0], reference[0]),
        "right": (estimate[1], reference[1]),
        "mid": (estimate.mean(axis=0), reference.mean(axis=0)),
        "side": ((estimate[0] - estimate[1]) / 2, (reference[0] - reference[1]) / 2),
    }
    projections = {name: _projection(*pair) for name, pair in pairs.items()}
    left_gain, right_gain = projections["left"][1], projections["right"][1]

    def correlation(value: np.ndarray) -> float | None:
        denominator = np.sqrt(energy(value[0]) * energy(value[1]))
        return float(np.sum(value[0] * value[1]) / denominator) if denominator > 0 else None

    def width(value: np.ndarray) -> float | None:
        mid_energy = energy(value.mean(axis=0))
        side_energy = energy((value[0] - value[1]) / 2)
        return db_ratio(side_energy, mid_energy) if mid_energy > 0 else None

    estimate_corr, reference_corr = correlation(estimate), correlation(reference)
    estimate_width, reference_width = width(estimate), width(reference)
    spatial_loss = {
        name: energy(ref) > 0 and energy(est) == 0
        for name, (est, ref) in pairs.items() if name in {"left", "right", "mid", "side"}
    }
    return {
        "si_sdr_db": {name: result[0] for name, result in projections.items()},
        "scale_dependent_snr_db": (
            db_ratio(energy(reference), energy(estimate - reference)) if energy(reference) > 0 else None
        ),
        "projection_gain_db": {name: result[1] for name, result in projections.items()},
        "relative_channel_gain_error_db": (
            float(left_gain - right_gain) if left_gain is not None and right_gain is not None else None
        ),
        "side_to_mid_balance_error_db": (
            estimate_width - reference_width
            if estimate_width is not None and reference_width is not None
            and energy(pairs["side"][1]) > 0 else None
        ),
        "spatial_component_lost": spatial_loss,
        "interchannel_correlation_error": (
            estimate_corr - reference_corr
            if estimate_corr is not None and reference_corr is not None else None
        ),
    }
