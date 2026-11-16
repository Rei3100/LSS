"""歌と伴奏の両方が欲しい制御ケースで、全体の改善に隠れる成分損傷を診断する。"""
from __future__ import annotations

import math

import numpy as np

from lss.validation.instrumental_quality import instrumental_quality
from lss.validation.stereo_quality import energy


def wanted_components(estimate: np.ndarray, accompaniment: np.ndarray, singing: np.ndarray) -> dict:
    """既存の2成分fitを診断だけに使う。補正した音声を採点・保存しない。"""
    measured = instrumental_quality(estimate, accompaniment, singing)
    fit = measured["component_fit"]
    jointly_present = energy(accompaniment) > 0 and energy(singing) > 0
    return {
        "scope": "controlled_additive_wanted_components_diagnostic_not_corrected_score",
        "state": "determined" if fit is not None else "undetermined" if jointly_present else "not_jointly_present",
        "wanted_roles": ["accompaniment", "singing"],
        "gains": {"accompaniment": fit["target_gain"], "singing": fit["vocals_gain"]} if fit else None,
        "condition_number": fit["condition_number"] if fit else None,
    }


def component_regressions(before: dict, after: dict, *, maximum_extra_gain_error_db: float = 0.1) -> list[dict]:
    """相関で成分が識別できない場合は保留し、観測できる追加損傷だけを返す。"""
    if (type(maximum_extra_gain_error_db) not in (int, float)
            or not math.isfinite(maximum_extra_gain_error_db) or maximum_extra_gain_error_db < 0):
        raise ValueError("成分の追加利得誤差は0以上の有限値が必要です")
    if before["state"] != "determined" or after["state"] != "determined":
        return []
    failures = []
    for role in ("accompaniment", "singing"):
        prior, current = before["gains"][role], after["gains"][role]
        if not math.isfinite(prior) or not math.isfinite(current):
            raise ValueError("成分の診断利得は有限値である必要があります")
        if prior <= 0:
            continue
        if current <= 0:
            failures.append({"component": role, "reason": "new_nonpositive_wanted_gain"})
            continue
        extra = abs(20 * math.log10(current)) - abs(20 * math.log10(prior))
        if extra > maximum_extra_gain_error_db + 1e-12:
            failures.append({"component": role, "extra_gain_error_db": extra,
                             "maximum_extra_gain_error_db": maximum_extra_gain_error_db})
    return failures
