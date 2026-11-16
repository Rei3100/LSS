"""Apply a repair only inside its declared sample intervals, with exact outside PCM."""
from __future__ import annotations

import numpy as np


def _ordered(frames, spans):
    ordered = sorted(spans)
    previous = -1
    for low, high in ordered:
        if type(low) is not int or type(high) is not int or not 0 <= low < high <= frames or low < previous:
            raise ValueError("Regions must be disjoint integer intervals within the audio")
        previous = high
    return ordered


def context_spans(frames, spans, padding, *, alignment=1):
    """Merge needed context on the global analysis lattice, without widening repair authority."""
    if (type(frames) is not int or frames < 1 or type(padding) is not int or padding < 0
            or type(alignment) is not int or alignment < 1):
        raise ValueError("Context layout needs integer frames, padding and alignment")
    result = []
    for a, b in _ordered(frames, spans):
        low = max(0, ((a-padding)//alignment)*alignment)
        high = min(frames, ((b+padding+alignment-1)//alignment)*alignment)
        if result and low <= result[-1][1]:
            result[-1][1] = max(result[-1][1], high)
        else:
            result.append([low, high])
    return result


def inside_weight(frames, spans, rate, transition_seconds, *, guard_seconds=0., start=0, length=None):
    if type(frames) is not int or frames < 1 or type(rate) is not int or rate < 1:
        raise ValueError("Region layout needs positive integer frames and rate")
    if length is None:
        length = frames - start
    if type(start) is not int or type(length) is not int or not 0 <= start < start + length <= frames:
        raise ValueError("Region tile is outside the audio")
    if not np.isfinite([transition_seconds, guard_seconds]).all() or min(transition_seconds, guard_seconds) < 0:
        raise ValueError("Region transitions must be finite and nonnegative")
    out = np.zeros(length, dtype=np.float64)
    for low, high in _ordered(frames, spans):
        a, b = max(start, low), min(start + length, high)
        if a >= b:
            continue
        index = np.arange(a, b)
        begin = low + (round(rate * guard_seconds) if low else 0)
        end = high - (round(rate * guard_seconds) if high < frames else 0)
        fade = min(max(1, round(rate * transition_seconds)), max(0, (end - begin) // 2))
        w = np.ones(b - a)
        w[(index < begin) | (index >= end)] = 0
        if fade > 1:
            if low:
                m = (index >= begin) & (index < begin + fade)
                w[m] = .5 - .5 * np.cos(np.pi * (index[m] - begin) / (fade - 1))
            if high < frames:
                m = (index >= end - fade) & (index < end)
                w[m] = .5 - .5 * np.cos(np.pi * (end - 1 - index[m]) / (fade - 1))
        out[a - start:b - start] = w
    return out


def blend(base, replacement, weights):
    """Keep base outside support; assign replacement exactly at full weight."""
    base, replacement, weights = np.asarray(base), np.asarray(replacement), np.asarray(weights)
    if (base.ndim != 2 or base.shape != replacement.shape or weights.shape != (len(base),)
            or not np.isfinite(base).all() or not np.isfinite(replacement).all()
            or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1))):
        raise ValueError("Region blend needs aligned finite audio and weights in [0, 1]")
    # A float32 estimate must not quantize a higher-precision original core.
    # Promotion preserves every existing base sample exactly as a value.
    out = base.astype(np.result_type(base.dtype, replacement.dtype, np.float32), copy=True)
    partial = (weights > 0) & (weights < 1)
    out[partial] = base[partial] + (replacement[partial] - base[partial]) * weights[partial, None]
    out[weights == 1] = replacement[weights == 1]
    return out
