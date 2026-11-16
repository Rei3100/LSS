"""既知の伴奏・歌唱componentを使い、保持・残留・誤差を分ける診断。"""
from __future__ import annotations

import numpy as np

from lss.validation.stereo_quality import db_ratio, energy, stereo_quality


def instrumental_quality(estimate: np.ndarray, target: np.ndarray, vocals: np.ndarray) -> dict:
    """同一時間軸の(2, frames)。補正した音声を採点せず、線形fitは診断に留める。

    componentへの投影では説明できない誤差を別に残す。無対象時は品質点を作らず、
    出力energyと入力歌唱energyの比を返す。相関が強すぎるcomponentの分解は保留する。
    """
    estimate, target, vocals = (np.asarray(value, dtype=np.float64) for value in (estimate, target, vocals))
    fidelity = stereo_quality(estimate, target)
    if vocals.shape != target.shape or not np.isfinite(vocals).all():
        raise ValueError("歌唱componentも同じ時間軸の有限ステレオ配列が必要です")
    target_energy, vocals_energy = energy(target), energy(vocals)
    result = {
        "fidelity": fidelity,
        "target_present": target_energy > 0,
        "target_absent_output_to_vocals_db": (
            db_ratio(energy(estimate), vocals_energy)
            if target_energy == 0 and vocals_energy > 0 else None
        ),
        "component_fit": None,
    }
    if target_energy == 0 or vocals_energy == 0:
        return result
    # 正規化はfitの条件数だけを整える。採点音声やreferenceのgainは変えない。
    scales = np.sqrt([target_energy, vocals_energy])
    columns = np.column_stack([target.ravel(), vocals.ravel()]) / scales
    fit, _, rank, singular = np.linalg.lstsq(columns, estimate.ravel(), rcond=None)
    if rank != 2 or singular[-1] / singular[0] < 1e-6:
        return result
    target_gain, vocals_gain = fit / scales
    unexplained = estimate - target_gain * target - vocals_gain * vocals
    result["component_fit"] = {
        "role": "diagnostic_only_not_corrected_score",
        "target_gain": float(target_gain),
        "vocals_gain": float(vocals_gain),
        "vocals_transfer_db": db_ratio(vocals_gain ** 2, 1.0),
        "unexplained_error_to_target_db": db_ratio(energy(unexplained), target_energy),
        "condition_number": float(singular[0] / singular[-1]),
    }
    return result
