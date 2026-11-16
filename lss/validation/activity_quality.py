"""固定した制御参照で候補区間の取り逃しと保存信号の欠損を分けて測る。"""
from __future__ import annotations

import numpy as np

from lss.validation.stereo_quality import energy, stereo_quality


def activity_quality(estimate: np.ndarray, reference: np.ndarray, intervals: list[tuple[int, int]]) -> dict:
    """同一timelineの有限stereo。非ゼロ参照は制御proxyであり意味精度ではない。"""
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    # 共通validatorも兼ねる。区間による切り捨て前の誤差を別に保持する。
    signal_quality = stereo_quality(estimate, reference)
    frames = reference.shape[1]
    selected = np.zeros(frames, dtype=bool)
    for start, end in intervals:
        if (
            not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool)
            or start < 0 or end <= start or end > frames
        ):
            raise ValueError("候補区間は参照timeline内の空でない整数区間である必要があります")
        selected[start:end] = True
    active = np.any(reference != 0, axis=0)
    active_frames = int(active.sum())
    reference_energy = energy(reference)
    retained = np.where(selected[np.newaxis, :], estimate, 0.0)
    return {
        "scope": "exact_nonzero_controlled_reference_not_semantic_accuracy",
        "target_present": reference_energy > 0,
        "reference_nonzero_frames": active_frames,
        "candidate_union_frames": int(selected.sum()),
        "candidate_frame_fraction": float(selected.mean()),
        "reference_frame_retained_fraction": (
            float((selected & active).sum() / active_frames) if active_frames else None
        ),
        "reference_energy_retained_fraction": (
            energy(reference[:, selected]) / reference_energy if reference_energy else None
        ),
        "candidate_frames_outside_reference": int((selected & ~active).sum()),
        "retained_signal_energy_without_target": energy(retained[:, ~active]),
        "separation_fidelity": signal_quality if reference_energy else None,
        "separation_and_activity_fidelity": stereo_quality(retained, reference) if reference_energy else None,
    }
