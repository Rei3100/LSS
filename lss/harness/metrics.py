"""ledger決定変数の算出（D-6②・D-9）。

vendor/msst/utils/metrics.py を単一の正典として薄くラップする（再実装しない＝A10）。
決定変数=sdr/l1_freq/aura_mrstft、記録専用=bleedless/fullness（採用条件には使わない）。
"""
from __future__ import annotations

import sys

import numpy as np

from lss.paths import resource_root

_MSST_ROOT = resource_root() / "vendor" / "msst"
if str(_MSST_ROOT) not in sys.path:
    sys.path.insert(0, str(_MSST_ROOT))

from utils.metrics import (  # noqa: E402  (vendor委譲のためsys.path挿入後にimportする)
    AuraMRSTFT_metric as _aura_mrstft_metric,
    L1Freq_metric as _l1_freq_metric,
    bleed_full as _bleed_full,
    sdr as _sdr,
)


def compute_decision_metrics(reference: np.ndarray, estimate: np.ndarray, device: str = "cpu") -> dict[str, float]:
    """3決定変数(sdr/l1_freq/aura_mrstft)＋記録専用2つ(bleedless/fullness)をまとめて返す。

    reference/estimate: shape (num_channels, num_samples) のステレオ波形（vendor品と同じ規約）。
    """
    sdr_value = float(_sdr(reference[np.newaxis, ...], estimate[np.newaxis, ...])[0])
    l1_freq_value = float(_l1_freq_metric(reference, estimate, device=device))
    aura_mrstft_value = float(_aura_mrstft_metric(reference, estimate, device=device))
    bleedless_value, fullness_value = _bleed_full(reference, estimate, device=device)
    return {
        "sdr": sdr_value,
        "l1_freq": l1_freq_value,
        "aura_mrstft": aura_mrstft_value,
        "bleedless": float(bleedless_value),
        "fullness": float(fullness_value),
    }
