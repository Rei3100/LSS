"""全体の改善で局所のステレオ損傷を隠さないDevelopment比較。"""
from __future__ import annotations

import math

COMPONENT_SI_CEILING_DB = 120.0

def spatial_regressions(before: dict, after: dict, *, maximum_extra_gain_error_db: float = 0.1,
                        maximum_component_si_regression_db: float = 0.1) -> list[dict]:
    """同じreferenceに対するfidelity。品質補正や親との幅の一致は要求しない。

    参照に対する左右・Mid/Sideの投影gain誤差とSI誤差の悪化を別に検出する。
    参照の無音componentはgain/SIがNoneなので、意味のない点を作らない。
    SIの120dB超は数値精度近傍として同じ天井へ切り詰め、丸め誤差の比を採否にしない。
    """
    for value in (maximum_extra_gain_error_db, maximum_component_si_regression_db):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("許容する追加誤差は0以上の有限値が必要です")
    failures = []

    def metric(name: str, component: str, prior: float | None, current: float | None, *, absolute: bool,
               tolerance: float):
        if (prior is not None and not math.isfinite(prior)) or current is not None and not math.isfinite(current):
            raise ValueError("ステレオ診断の指標は有限値かNoneである必要があります")
        if prior is None:
            return
        if current is None:
            failures.append({"metric": name, "component": component, "reason": "became_unmeasurable"})
            return
        delta = (abs(current) - abs(prior) if absolute else
                 min(prior, COMPONENT_SI_CEILING_DB) - min(current, COMPONENT_SI_CEILING_DB))
        if delta > tolerance + 1e-12:
            failures.append({"metric": name, "component": component, "extra_error_db": delta,
                             "maximum_extra_error_db": tolerance})

    for component in ("left", "right", "mid", "side"):
        if after["spatial_component_lost"][component] and not before["spatial_component_lost"][component]:
            failures.append({"metric": "spatial_component_lost", "component": component, "reason": "new_complete_loss"})
        metric("projection_gain_db", component, before["projection_gain_db"][component], after["projection_gain_db"][component],
               absolute=True, tolerance=maximum_extra_gain_error_db)
        metric("si_sdr_db", component, before["si_sdr_db"][component], after["si_sdr_db"][component],
               absolute=False, tolerance=maximum_component_si_regression_db)
    for name in ("relative_channel_gain_error_db", "side_to_mid_balance_error_db"):
        metric(name, "stereo", before[name], after[name], absolute=True, tolerance=maximum_extra_gain_error_db)
    return failures
