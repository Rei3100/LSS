"""親のMusic成分を弱めるbranchを、clip全体で保留するDevelopment用の保護。"""
from __future__ import annotations

import math
import copy
from typing import Mapping

import numpy as np

POLICY = {"schema_version": 1, "method": "all_music_components_nondecreasing_parent_projection_and_energy",
          "scope": "entire_prepared_clip", "components": ["joint", "left", "right", "mid", "side"],
          "minimum_projection_gain": 1.0, "minimum_energy_ratio": 1.0,
          "rounding_tolerance": 1e-12, "semantic_identity": "unresolved"}


def preserve_parent(parent: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> tuple[dict, dict]:
    """参照・正解labelを使わず、拒否時は全stemを共通に親へ戻す。品質の証明ではない。"""
    roles = {"music", "speech", "effects"}
    if set(parent) != roles or set(candidate) != roles:
        raise ValueError("親保護には同じ3stemが必要です")
    shape = np.asarray(parent["music"]).shape
    for stems in (parent, candidate):
        for values in stems.values():
            values = np.asarray(values)
            if (values.shape != shape or values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0
                    or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
                raise ValueError("親保護には同じ時間軸の有限ステレオが必要です")
    music = [np.asarray(stems["music"], dtype=np.float64) for stems in (parent, candidate)]
    try:
        with np.errstate(over="raise", invalid="raise"):
            components = [{"joint": values, "left": values[0], "right": values[1],
                           "mid": (values[0] + values[1]) * 0.5,
                           "side": (values[0] - values[1]) * 0.5} for values in music]
            observations = []
            for name in POLICY["components"]:
                prior, after = (value[name] for value in components)
                pe, ce = float(np.sum(prior ** 2)), float(np.sum(after ** 2))
                dot = float(np.sum(prior * after))
                if not all(math.isfinite(value) for value in (pe, ce, dot)):
                    raise ValueError("親保護のenergy集計が有限範囲を超えました")
                gain, ratio = (dot / pe, ce / pe) if pe > 0 else (None, None)
                if gain is not None and not (math.isfinite(gain) and math.isfinite(ratio)):
                    raise ValueError("親保護の相対診断が有限範囲を超えました")
                accepted = (ce == 0 if pe == 0 else gain >= 1 - POLICY["rounding_tolerance"]
                            and ratio >= 1 - POLICY["rounding_tolerance"])
                observations.append({"component": name, "projection_gain": gain,
                                     "energy_ratio": ratio, "accepted": bool(accepted)})
    except FloatingPointError as exc:
        raise ValueError("親保護のenergy集計が有限範囲を超えました") from exc
    enabled = all(row["accepted"] for row in observations)
    chosen = candidate if enabled else parent
    return {role: np.asarray(values, dtype=np.float64).copy() for role, values in chosen.items()}, {
        "policy": copy.deepcopy(POLICY), "enabled": enabled, "observations": observations,
        "claim_boundary": "parent_consistency_only_not_target_preservation_or_semantic_validation"}
