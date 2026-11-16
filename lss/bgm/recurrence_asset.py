"""One better asset from every published occurrence of the same score recording.

The published-cue memory (`lss.bgm.cue_memory`) knows which published cues, across episodes, are the same
recording. Each of those cues is a separated estimate of the same music with *different* dialogue/effect residue
and different separation drop-outs: a line spoken over bar 5 in one episode is silent in another. Where two or
more occurrences cover the same moment, the occurrences that agree with the others are the clean ones there.

This module aligns the occurrences of one recurring group on the clock of its longest member (a real broadcast
edit, so the musical order is one that actually aired), measures for every moment how much of each occurrence the
other occurrences explain (`identity`'s short-FIR waveform explanation; residue that is not shared is not
explained), and builds a consensus in which each moment comes mostly from the occurrences that are clean there.

- No separation model, no reference recording and no original mixture are used; only published CurrentBest cues.
- Alignment comes from the memory's verified links (piecewise: TV edits jump inside a recording), composed through
  other members when a member is not linked to the longest one, refined to the sample.
- The base occurrence's clock and channels are kept. A moment is rebuilt by weighted consensus only where at least
  three occurrences play (`MIN_CONSENSUS_LAYERS`): with two, residue in one and a drop-out in the other look alike.
  Where exactly two play the base is only attenuated, per time-frequency tile, towards the other copy where that is
  lower (`PAIR_GAMMA`, `PAIR_FLOOR`; additive residue is the common case, a drop-out can cost at most 12 dB).
  Elsewhere the base (or the only occurrence) is used as is.
- An occurrence the memory links at another tempo (editors stretch a score to a scene, pitch kept) is laid on the
  canvas by `warp` (its own grains, each placed where it matches the base). That re-timed copy is never averaged
  in; it only witnesses magnitude where the base is the only constant-tempo occurrence playing: there the base is
  attenuated towards the lowest such copy with the two-occurrence rule (short grain-join notches ignored). A
  recording heard only at different tempos thereby gets an asset at all.
- A head or tail that another resolved occurrence continues past the base's ends extends the asset, only from an
  aligned passage that reaches that end.
- The quality figures are a consistency measure between occurrences, not a truth measure.

The asset is published once per recurring group (`update`) as an additional library file next to the episode
folders. Episode cues, their CurrentBest and cue identity are never changed; the asset is regenerated when the
group's members change, and a new revision replaces the previous one only when it rests on at least as much
multi-occurrence evidence (the previous file is kept in the state for rollback).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
METHOD = "recurrence-consensus-v3"

# Consistency measurement (mono, `identity` analysis rate).
QUALITY_RATE = 11025
QUALITY_WINDOW_SECONDS = 2.0
QUALITY_HOP_SECONDS = 0.5
QUALITY_TAPS = 16
# Extending a matched passage while it keeps explaining the base at least this much.
EXTEND_EXPLAINED = 0.5
EXTEND_STEP_SECONDS = 0.5
# An aligned passage within this of a base end may carry the asset past it.
EDGE_SECONDS = 0.5
# Synthesis.
N_FFT = 4096
HOP = N_FFT // 4
GAIN_HALF_WINDOW_SECONDS = 1.0
GAIN_LIMIT = 4.0
# With two occurrences "one carries residue" and "the other dropped music" look the same (each explains the
# other one-sidedly), so the weighted consensus is used only where at least this many occurrences play.
MIN_CONSENSUS_LAYERS = 3
# Where exactly two occurrences play, the base is only ever attenuated, tile by tile, towards the other copy:
# base * clip((|other| / |base|) ** PAIR_GAMMA, PAIR_FLOOR, 1). Residue is additive, so the lower of two aligned
# copies is usually the cleaner one; a drop-out in the other copy can pull the base down only part of the way
# (gamma < 1 and at most -12 dB; measured: 0.4 / 0.25 gains on every real pair and the control, gamma 0.7-1.0 or no
# floor costs blocks), and the base's own phase, detail and clock are kept. Falsy PAIR_GAMMA disables it.
PAIR_GAMMA = 0.4
PAIR_FLOOR = 0.25
MODES = ("weighted", "select", "weighted_magnitude")
DEFAULT_MODE = "weighted"
WEIGHT_POWER = 2.0
# A layer the other occurrences explain less than this (median over the cue) is not this music there: misplaced
# or another recording. It is left out of the asset.
MIN_LAYER_CONSISTENCY = 0.2
# Matched runs pair across the two sides of a link by equal length within this.
PAIR_TOLERANCE_SECONDS = 0.01
# Fine alignment search around the link's coarse offset.
FINE_SEARCH_SECONDS = 0.05
FINE_WINDOW_SECONDS = 6.0
# An occurrence at another tempo (pitch kept: editors stretch with copied grains) is laid on the canvas clock by
# overlap-adding its own short grains, each taken where it best matches the base at that canvas moment within a
# small search around the link's tempo line. Measured on real corpus-a stretched pairs (1-8 % tempo): the warped
# occurrence explains 0.36-0.59 of the base per 2 s window, the same procedure on an unrelated cue 0.02-0.09.
SCALE_TOLERANCE = 1e-4
WARP_GRAIN_SECONDS = 0.046
WARP_SEARCH_SECONDS = 0.012
# A warped layer only witnesses magnitude; its short notches are ignored by a max filter over this many STFT
# bins and frames.
WITNESS_BINS = 3
WITNESS_FRAMES = 5


class RecurrenceAssetError(ValueError):
    """The group cannot be aligned into one asset."""


@dataclass
class Layer:
    """One occurrence passage on the canvas: canvas sample ``t`` plays member sample ``scale * t + lag``.

    ``scale`` differs from 1 for an occurrence played at another tempo; `warp` renders such a layer onto the canvas
    clock as a derived signal (``member`` is then the derived key, ``source`` the occurrence and ``tempo`` the scale).
    """
    member: str
    start: int
    end: int
    lag: int
    via: str = "direct"
    scale: float = 1.0
    source: str | None = None
    tempo: float | None = None

    @property
    def occurrence(self) -> str:
        return self.source or self.member

    def to_dict(self, rate: int) -> dict:
        row = {"member": self.occurrence, "start_seconds": round(self.start / rate, 6),
               "end_seconds": round(self.end / rate, 6), "lag_seconds": round(self.lag / rate, 6), "via": self.via}
        if self.tempo is not None:
            row["tempo_scale"] = round(self.tempo, 5)
        return row


@dataclass
class Plan:
    base: str
    rate: int
    length: int
    layers: list[Layer] = field(default_factory=list)


def recipe() -> dict:
    return {"method": METHOD, "quality_rate": QUALITY_RATE, "quality_window_seconds": QUALITY_WINDOW_SECONDS,
            "quality_hop_seconds": QUALITY_HOP_SECONDS, "quality_taps": QUALITY_TAPS,
            "extend_explained": EXTEND_EXPLAINED, "edge_seconds": EDGE_SECONDS, "n_fft": N_FFT, "hop": HOP,
            "gain_half_window_seconds": GAIN_HALF_WINDOW_SECONDS, "gain_limit": GAIN_LIMIT,
            "mode": DEFAULT_MODE, "weight_power": WEIGHT_POWER, "min_consensus_layers": MIN_CONSENSUS_LAYERS, "pair_gamma": PAIR_GAMMA, "pair_floor": PAIR_FLOOR,
            "min_layer_consistency": MIN_LAYER_CONSISTENCY, "pairing": "equal-length-v1",
            "tempo": {"scale_tolerance": SCALE_TOLERANCE, "warp_grain_seconds": WARP_GRAIN_SECONDS,
                      "warp_search_seconds": WARP_SEARCH_SECONDS, "warp": "base-guided-grains-v1"}}


def recipe_id() -> str:
    body = json.dumps(recipe(), sort_keys=True, separators=(",", ":"))
    return f"bgm-recurrence-asset-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ alignment

def _stretched(link: dict) -> bool:
    return abs(float(link.get("rate") or 1.0) - 1.0) > SCALE_TOLERANCE


def _pieces(link: dict, first: str) -> list[tuple[float, float, float, float]]:
    """(start, end, other start, ratio) in ``first``'s clock: the other member's time at ``first`` time ``x`` in
    the piece is ``other start + ratio * (x - start)``.

    A constant-offset run has the same length on both sides (ratio 1). Records written before the memory kept the
    two sides paired list them sorted independently, so a range is paired with the other side's range of the same
    length; a range whose length is not unique on the other side is left out rather than guessed. A link at another
    tempo (``rate``, `cue_memory` stretched links, always paired) maps each run linearly from its two ranges.
    """
    mine, theirs = (link["a_ranges"], link["b_ranges"]) if link["a"] == first else (link["b_ranges"], link["a_ranges"])
    if _stretched(link):
        if not link.get("paired"):
            return []
        return [(float(low), float(high), float(other_low), (float(other_high) - float(other_low)) / (high - low))
                for (low, high), (other_low, other_high) in zip(mine, theirs) if high > low and other_high > other_low]
    if link.get("paired"):
        return [(float(low), float(high), float(other_low), 1.0)
                for (low, high), (other_low, _) in zip(mine, theirs)]
    found = []
    for low, high in mine:
        same = [other for other in theirs if abs((other[1] - other[0]) - (high - low)) < PAIR_TOLERANCE_SECONDS]
        if len(same) == 1:
            found.append((float(low), float(high), float(same[0][0]), 1.0))
    return found


def plan(members: dict[str, dict], links: list[dict], rate: int) -> Plan:
    """Place every reachable member's linked passages on the clock of the longest member.

    ``members`` maps asset id to ``{"duration_seconds", ...}``; ``links`` are cue-memory links among them.
    Breadth-first from the base: a member is placed through the first already-placed member it is linked to.
    """
    if len(members) < 2:
        raise RecurrenceAssetError("統合には同じ録音の公開cueが2件以上必要です")
    base = max(members, key=lambda key: (members[key]["duration_seconds"], key))
    length = int(round(members[base]["duration_seconds"] * rate))
    placed: dict[str, list[Layer]] = {base: [Layer(base, 0, length, 0, "base")]}
    frontier = [base]
    # constant-offset links first: a member reachable both ways is placed without a tempo change
    ordered = sorted(links, key=_stretched)
    while frontier:
        current = frontier.pop(0)
        for link in ordered:
            if current not in (link["a"], link["b"]):
                continue
            other = link["b"] if link["a"] == current else link["a"]
            if other not in members or other in placed:
                continue
            found = []
            for low, high, other_low, ratio in _pieces(link, current):
                for layer in placed[current]:
                    via = "direct" if current == base else f"via:{current}"
                    if ratio == 1.0 and layer.scale == 1.0:
                        # current's time x is canvas time x - layer.lag
                        start = max(layer.start, int(round(low * rate)) - layer.lag)
                        end = min(layer.end, int(round(high * rate)) - layer.lag)
                        if end - start >= rate:
                            found.append(Layer(other, start, end,
                                               layer.lag + int(round((other_low - low) * rate)), via))
                        continue
                    # current's sample x = scale * canvas t + lag; other's sample = ratio * (x - low) + other_low
                    start = max(layer.start, int(np.ceil((low * rate - layer.lag) / layer.scale)))
                    end = min(layer.end, int(np.floor((high * rate - layer.lag) / layer.scale)))
                    if end - start < rate:
                        continue
                    scale = ratio * layer.scale
                    lag = ratio * (layer.lag - low * rate) + other_low * rate
                    found.append(Layer(other, start, end, int(round(lag)), via,
                                       scale=1.0 if abs(scale - 1.0) <= SCALE_TOLERANCE else scale))
            if found:
                placed[other] = found
                frontier.append(other)
    layers = [layer for key in placed for layer in placed[key]]
    return Plan(base=base, rate=rate, length=length, layers=layers)


def _guided_grains(guide: np.ndarray, source: np.ndarray, low: int, high: int, scale: float, lag: float,
                   rate: int) -> np.ndarray:
    """``source`` (samples, channels) rendered on the canvas span [low, high): grains of ``source`` near
    ``scale * t + lag``, each placed where it best correlates with ``guide`` (canvas mono), Hann overlap-add."""
    from scipy.signal import correlate

    size = int(WARP_GRAIN_SECONDS * rate) // 2 * 2
    hop = size // 2
    search = int(WARP_SEARCH_SECONDS * rate)
    window = np.hanning(size + 1)[:-1]
    mono = source.mean(axis=1).astype(np.float64)
    out = np.zeros((high - low + size, source.shape[1]))
    norm = np.zeros(high - low + size)
    ones = np.ones(size)
    for position in range(low, high, hop):
        nominal = int(round(scale * position + lag))
        first, last = nominal - search, nominal + search + size
        if first < 0 or last > mono.size or position + size > guide.size:
            continue
        target = guide[position:position + size]
        region = mono[first:last]
        energy = np.sqrt(np.maximum(np.convolve(region * region, ones, "valid"), 0.0)) * (np.linalg.norm(target)
                                                                                           + 1e-9) + 1e-9
        offset = int(np.argmax(correlate(region, target, mode="valid", method="fft") / energy))
        begin = first + offset
        out[position - low:position - low + size] += source[begin:begin + size] * window[:, None]
        norm[position - low:position - low + size] += window
    out = out[:high - low]
    norm = norm[:high - low]
    out[norm > 1e-3] /= norm[norm > 1e-3, None]
    out[norm <= 1e-3] = 0.0
    return out


def warp(found: Plan, audio: dict[str, np.ndarray], mono: dict[str, np.ndarray]) -> Plan:
    """Lay every tempo-changed layer on the canvas clock as a derived signal; constant layers are unchanged.

    The derived signal spans everywhere the layer's tempo line stays inside its occurrence (so `refine` can still
    grow the matched run over passages it explains); the layer becomes a constant one (lag 0) of that signal.
    """
    base = mono[found.base].astype(np.float64)
    layers = []
    for number, layer in enumerate(found.layers):
        if layer.scale == 1.0:
            layers.append(layer)
            continue
        source = audio[layer.member]
        low = max(0, int(np.ceil(-layer.lag / layer.scale)))
        high = min(found.length, int(np.floor((source.shape[0] - layer.lag) / layer.scale)))
        if high - low < found.rate:
            continue
        key = f"{layer.member}@tempo{number}"
        canvas = np.zeros((found.length, source.shape[1]), np.float32)
        canvas[low:high] = _guided_grains(base, source.astype(np.float64), low, high, layer.scale, layer.lag,
                                          found.rate)
        audio[key] = canvas
        mono[key] = canvas.mean(axis=1)
        layers.append(Layer(key, max(layer.start, low), min(layer.end, high), 0, layer.via, source=layer.member,
                            tempo=layer.scale))
    return Plan(base=found.base, rate=found.rate, length=found.length, layers=layers)


def _fine_lag(target: np.ndarray, source: np.ndarray, start: int, lag: int, rate: int) -> tuple[int, float]:
    """Sample correction so that ``source[t + lag + d]`` best matches ``target[t]`` near ``start``."""
    window = int(FINE_WINDOW_SECONDS * rate)
    search = int(FINE_SEARCH_SECONDS * rate)
    begin = max(0, start, search - lag)
    stop = min(len(target), begin + window, len(source) - lag - search)
    if stop - begin < rate // 2:
        return 0, 0.0
    segment = target[begin:stop].astype(np.float64)
    region = source[begin + lag - search:stop + lag + search].astype(np.float64)
    corr = np.correlate(region, segment, mode="valid")
    energy = np.sqrt(np.convolve(region ** 2, np.ones(segment.size), "valid")) * np.linalg.norm(segment) + 1e-12
    corr = corr / energy
    best = int(np.argmax(corr))
    return best - search, float(corr[best])


def refine(found: Plan, mono: dict[str, np.ndarray]) -> Plan:
    """Correct each layer's lag to the sample against the base, then extend it while it still explains the base."""
    base = mono[found.base]
    rate = found.rate
    step = int(EXTEND_STEP_SECONDS * rate)
    window = int(QUALITY_WINDOW_SECONDS * rate)
    for layer in found.layers:
        if layer.member == found.base:
            continue
        source = mono[layer.member]
        if layer.tempo is None:
            # a warped layer's grains already sit on the base to the sample
            delta, _ = _fine_lag(base, source, layer.start, layer.lag, rate)
            layer.lag += delta
        layer.start = max(layer.start, -layer.lag, 0)
        layer.end = min(layer.end, len(source) - layer.lag, found.length)
        # grow both ways over passages the matcher's conservative run left out
        while layer.start - step >= max(0, -layer.lag):
            low = layer.start - step
            if explained(base[low:low + window], source[low + layer.lag:low + layer.lag + window]) < EXTEND_EXPLAINED:
                break
            layer.start = low
        while layer.end + step <= min(found.length, len(source) - layer.lag):
            high = layer.end + step
            if explained(base[high - window:high], source[high - window + layer.lag:high + layer.lag]) < EXTEND_EXPLAINED:
                break
            layer.end = high
    return found


def edge_extensions(found: Plan, durations: dict[str, int], resolved: dict[str, bool]) -> tuple[int, int, dict]:
    """Head/tail samples another resolved occurrence carries past the base's ends from an aligned passage."""
    edge = int(EDGE_SECONDS * found.rate)
    head, tail, chosen = 0, 0, {}
    for layer in found.layers:
        if layer.member == found.base or not resolved.get(layer.member):
            continue
        if layer.start <= edge:
            before = layer.start + layer.lag
            if before > head:
                head, chosen["head"] = before, layer
        if layer.end >= found.length - edge:
            after = durations[layer.member] - (layer.end + layer.lag)
            if after > tail:
                tail, chosen["tail"] = after, layer
    return head, tail, chosen


# ------------------------------------------------------------------------------------------------ quality

def explained(target: np.ndarray, source: np.ndarray, taps: int = QUALITY_TAPS) -> float:
    """Share of ``target`` a short linear filter of ``source`` explains (0 when silent or unrelated)."""
    from numpy.lib.stride_tricks import sliding_window_view

    size = min(target.size, source.size)
    if size < 4 * taps:
        return 0.0
    y = target[:size].astype(np.float64)
    energy = float(y @ y)
    if energy < 1e-9 * size:
        return 0.0
    half = taps // 2
    padded = np.pad(source[:size].astype(np.float64), (half, taps - half - 1))
    design = sliding_window_view(padded, taps)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return max(0.0, 1.0 - float(residual @ residual) / energy)


def coverage(found: Plan, length: int, offset: int) -> np.ndarray:
    """Boolean (layers, samples) of where each layer plays on a canvas ``offset`` samples before the base."""
    mask = np.zeros((len(found.layers), length), bool)
    for number, layer in enumerate(found.layers):
        mask[number, layer.start + offset:layer.end + offset] = True
    return mask


def consistency(found: Plan, mono: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per layer and window: median share of it that each other layer covering the window explains.

    Returns (window centres in base samples, values (layers, windows) with NaN where fewer than two layers play).
    """
    from scipy.signal import resample_poly

    rate = found.rate
    factor = max(1, rate // QUALITY_RATE)
    low = {key: resample_poly(value.astype(np.float64), 1, factor) for key, value in mono.items()}
    window, hop = int(QUALITY_WINDOW_SECONDS * rate), int(QUALITY_HOP_SECONDS * rate)
    starts = np.arange(0, max(1, found.length - window + 1), hop)
    values = np.full((len(found.layers), starts.size), np.nan)
    for column, start in enumerate(starts):
        playing = [n for n, layer in enumerate(found.layers) if layer.start <= start and start + window <= layer.end
                   and start + layer.lag >= 0 and start + window + layer.lag <= mono[layer.member].size]
        if len(playing) < 2:
            continue
        signals = {n: low[found.layers[n].member][(start + found.layers[n].lag) // factor:
                                                  (start + found.layers[n].lag + window) // factor]
                   for n in playing}
        for n in playing:
            # a constant layer is judged by constant layers only (a re-timed copy must not move consensus weights);
            # a tempo layer by every other layer
            judges = [o for o in playing if o != n and (found.layers[n].tempo is not None
                                                         or found.layers[o].tempo is None)]
            if judges:
                values[n, column] = float(np.median([explained(signals[n], signals[o]) for o in judges]))
    return starts + window // 2, values


def drop_inconsistent(found: Plan, mono: dict[str, np.ndarray], centres: np.ndarray, values: np.ndarray):
    """Leave out layers the others do not explain (misplaced passage or another recording), then re-measure.

    Returns (plan, centres, values, [(layer, median consistency)] left out). The base layer always stays.
    """
    dropped = []
    keep = []
    for number, layer in enumerate(found.layers):
        row = values[number]
        median = None if np.all(np.isnan(row)) else float(np.nanmedian(row))
        if layer.via != "base" and median is not None and median < MIN_LAYER_CONSISTENCY:
            dropped.append((layer, round(median, 4)))
        else:
            keep.append(layer)
    if not dropped:
        return found, centres, values, []
    kept = Plan(base=found.base, rate=found.rate, length=found.length, layers=keep)
    centres, values = consistency(kept, mono)
    return kept, centres, values, dropped


# ------------------------------------------------------------------------------------------------ synthesis

def _stft(x: np.ndarray) -> np.ndarray:
    from scipy.signal import stft

    return stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP, axis=0)[2]  # (bins, channels, frames)


def _istft(z: np.ndarray, length: int) -> np.ndarray:
    from scipy.signal import istft

    y = istft(z, nperseg=N_FFT, noverlap=N_FFT - HOP, time_axis=-1, freq_axis=0)[1]
    y = np.asarray(y).T if np.asarray(y).ndim == 2 else np.asarray(y)[:, None]
    out = np.zeros((length, z.shape[1]), np.float64)
    out[:min(length, y.shape[0])] = y[:length]
    return out


def _band_edges(bins: int, rate: int) -> list[tuple[int, int]]:
    edges = np.unique(np.round(np.geomspace(1, bins - 1, 33)).astype(int))
    edges = np.concatenate(([0], edges, [bins]))
    return [(int(low), int(high)) for low, high in zip(edges[:-1], edges[1:]) if high > low]


def pair_gain(gamma: float, floor: float = PAIR_FLOOR):
    """Two-occurrence tile policy: ``fn(A, B, qa, qb, mask)`` -> A attenuated where B is lower (A is the base)."""
    def fn(A, B, qa, qb, mask):
        a = np.abs(A).mean(axis=1, keepdims=True)
        b = np.abs(B).mean(axis=1, keepdims=True)
        return A * np.clip(((b + 1e-9) / (a + 1e-9)) ** gamma, floor, 1.0)
    return fn


def synthesize(found: Plan, audio: dict[str, np.ndarray], weights_by_layer: np.ndarray, centres: np.ndarray,
               *, head: int = 0, tail: int = 0, extension: dict | None = None,
               mode: str = DEFAULT_MODE, min_layers: int = MIN_CONSENSUS_LAYERS,
               pair=None) -> tuple[np.ndarray, dict]:
    """The consensus asset (samples, channels) on a canvas ``head`` samples before the base start.

    ``audio`` holds full-rate (samples, channels) arrays at ``found.rate``. ``weights_by_layer`` are consistency
    values per layer and window (`consistency`); where a moment has one layer it is used unchanged.
    """
    if mode not in MODES:
        raise RecurrenceAssetError(f"unknown mode {mode}")
    channels = audio[found.base].shape[1]
    length = head + found.length + tail
    layers = list(found.layers)
    extra = []
    if head and extension and "head" in extension:
        layer = extension["head"]
        extra.append(Layer(layer.member, -head, layer.start, layer.lag, "head"))
    if tail and extension and "tail" in extension:
        layer = extension["tail"]
        extra.append(Layer(layer.member, layer.end, found.length + tail, layer.lag, "tail"))
    all_layers = layers + extra
    frames = 1 + length // HOP + 1
    spectra, present = [], []
    for layer in all_layers:
        signal = np.zeros((length, channels), np.float64)
        source = audio[layer.member]
        low = max(layer.start, -layer.lag, -head)
        high = min(layer.end, source.shape[0] - layer.lag, found.length + tail)
        if high <= low:
            spectra.append(None)
            present.append(np.zeros(frames, bool))
            continue
        block = source[low + layer.lag:high + layer.lag]
        if block.shape[1] != channels:
            block = np.repeat(block.mean(axis=1, keepdims=True), channels, axis=1)
        signal[low + head:high + head] = block
        z = _stft(signal)
        spectra.append(z)
        centre = np.arange(z.shape[-1]) * HOP
        # a frame counts only when its whole window lies inside the passage
        present.append((centre - N_FFT // 2 >= low + head) & (centre + N_FFT // 2 <= high + head))
    frames = next(z.shape[-1] for z in spectra if z is not None)
    present = [p[:frames] if p.size >= frames else np.pad(p, (0, frames - p.size)) for p in present]
    centre_samples = np.arange(frames) * HOP - head
    weight = np.zeros((len(all_layers), frames))
    for number in range(len(all_layers)):
        if number < len(layers):
            values = weights_by_layer[number]
            known = ~np.isnan(values)
            weight[number] = np.interp(centre_samples, centres[known], values[known]) if known.any() else 0.0
        weight[number][~present[number]] = np.nan
    base_index = next(n for n, layer in enumerate(all_layers) if layer.member == found.base and layer.via == "base")
    bins = spectra[base_index].shape[0]
    reference = spectra[base_index]
    bands = _band_edges(bins, found.rate)
    half = max(1, int(GAIN_HALF_WINDOW_SECONDS * found.rate / HOP))
    # Level each layer to the base per band with a running median of magnitude ratios over shared frames.
    leveled = []
    for number, z in enumerate(spectra):
        if z is None or number == base_index or number >= len(layers):
            leveled.append(z)
            continue
        shared = present[number] & present[base_index]
        gains = np.ones((bins, frames))
        ref_mag = np.abs(reference).mean(axis=1)
        mag = np.abs(z).mean(axis=1)
        band_ref = np.stack([ref_mag[low:high].mean(axis=0) for low, high in bands])
        band_mag = np.stack([mag[low:high].mean(axis=0) for low, high in bands])
        ratio = np.full(band_ref.shape, np.nan)
        ratio[:, shared] = (band_ref[:, shared] + 1e-9) / (band_mag[:, shared] + 1e-9)
        smooth = np.ones_like(band_ref)
        for t in range(frames):
            if not present[number][t]:
                continue
            window = ratio[:, max(0, t - half):t + half + 1]
            with np.errstate(all="ignore"):
                value = np.nanmedian(window, axis=1)
            if np.all(np.isnan(value)):
                with np.errstate(all="ignore"):
                    value = np.nanmedian(ratio, axis=1)
            smooth[:, t] = np.clip(np.nan_to_num(value, nan=1.0), 1 / GAIN_LIMIT, GAIN_LIMIT)
        for (low, high), row in zip(bands, smooth):
            gains[low:high] = row
        leveled.append(z * gains[:, None, :])
    # A tempo-changed layer is a re-timed copy (grain joins, small timing wobble): its complex values must not be
    # averaged in, but its magnitude still witnesses where the base carries residue the recording does not have.
    witness = [n for n, layer in enumerate(all_layers) if layer.tempo is not None and leveled[n] is not None]
    out = np.zeros_like(reference)
    used = np.zeros(frames, np.int64)
    for t in range(frames):
        playing = [n for n in range(len(all_layers)) if leveled[n] is not None and present[n][t]
                   and n not in witness]
        if not playing:
            continue
        used[t] = len(playing)
        if len(playing) == 1:
            out[..., t] = leveled[playing[0]][..., t]
            continue
        if len(playing) < min_layers and base_index in playing:
            out[..., t] = leveled[base_index][..., t]
            continue
        q = np.array([0.0 if np.isnan(weight[n, t]) else weight[n, t] for n in playing])
        if mode == "select":
            out[..., t] = leveled[playing[int(np.argmax(q))]][..., t]
            continue
        clipped = np.clip(q, 0.0, 0.995)
        w = (clipped / (1 - clipped)) ** WEIGHT_POWER
        if w.sum() <= 0:
            w = np.ones_like(w)
        stack = np.stack([leveled[n][..., t] for n in playing])
        if mode == "weighted":
            out[..., t] = np.tensordot(w, stack, axes=1) / w.sum()
        else:
            best = stack[int(np.argmax(q))]
            phase = best / (np.abs(best) + 1e-12)
            out[..., t] = phase * np.tensordot(w, np.abs(stack), axes=1) / w.sum()
    if pair is None and PAIR_GAMMA:
        pair = pair_gain(PAIR_GAMMA)
    pair_frames = 0
    if pair is not None and pair is not False and min_layers > 2:
        # frames where exactly the base and one other layer play: pair(A, B, qa, qb, mask) -> (bins, ch, frames)
        for number in range(len(all_layers)):
            if number == base_index or leveled[number] is None or number in witness:
                continue
            mask = np.array([used[t] == 2 and present[base_index][t] and present[number][t] for t in range(frames)])
            if not mask.any():
                continue
            qa = np.nan_to_num(weight[base_index], nan=0.0)
            qb = np.nan_to_num(weight[number], nan=0.0)
            out[..., mask] = pair(leveled[base_index], leveled[number], qa, qb, mask)[..., mask]
            pair_frames += int(mask.sum())
    tempo_frames = 0
    if witness and PAIR_GAMMA:
        # Where tempo-changed copies play and the base is the only constant-tempo occurrence, the base is attenuated
        # (never raised) towards the lowest of them with the two-occurrence rule.
        mask = np.zeros(frames, bool)
        lowest = None
        from scipy.ndimage import maximum_filter

        for number in witness:
            # grain joins and re-timing leave short notches; residue (speech, effects) lasts much longer
            magnitude = maximum_filter(np.abs(leveled[number]).mean(axis=1, keepdims=True),
                                       size=(WITNESS_BINS, 1, WITNESS_FRAMES), mode="nearest")
            magnitude[..., ~present[number]] = np.inf
            lowest = magnitude if lowest is None else np.minimum(lowest, magnitude)
            mask |= present[number]
        # only where no constant-tempo occurrence shares the moment: next to the pair rule or the consensus a
        # re-timed witness measured +0.13 dB / -0.02 dB with 2-11 % of blocks worse (semi-synthetic control)
        mask &= used == 1
        if mask.any():
            current = np.abs(out).mean(axis=1, keepdims=True)
            gain = np.clip(((lowest + 1e-9) / (current + 1e-9)) ** PAIR_GAMMA, PAIR_FLOOR, 1.0)
            out[..., mask] = (out * gain)[..., mask]
            tempo_frames = int(mask.sum())
    # frames at passage edges that no layer fully covers: fall back to whatever partially plays (base first)
    signal = _istft(out, length)
    fallback = np.zeros((length, channels))
    filled = np.zeros(length, bool)
    for layer in [all_layers[base_index]] + [l for n, l in enumerate(all_layers) if n != base_index]:
        source = audio[layer.member]
        low = max(layer.start, -layer.lag, -head)
        high = min(layer.end, source.shape[0] - layer.lag, found.length + tail)
        if high <= low:
            continue
        span = np.arange(low + head, high + head)
        free = span[~filled[span]]
        block = source[free - head + layer.lag]
        if block.shape[1] != channels:
            block = np.repeat(block.mean(axis=1, keepdims=True), channels, axis=1)
        fallback[free] = block
        filled[free] = True
    covered = np.zeros(length, bool)
    for t in np.nonzero(used)[0]:
        covered[max(0, t * HOP - HOP // 2):t * HOP + HOP // 2] = True
    # inner frames come from the STFT consensus; uncovered edge samples from the plain occurrence
    signal[~covered] = fallback[~covered]
    multi = int(np.sum(used >= 2))
    consensus = int(np.sum(used >= min_layers))
    return signal.astype(np.float32), {"frames": int(frames), "multi_occurrence_frames": multi,
                                       "consensus_seconds": round(consensus * HOP / found.rate, 3),
                                       "pair_seconds": round(pair_frames * HOP / found.rate, 3),
                                       "tempo_seconds": round(tempo_frames * HOP / found.rate, 3),
                                       "multi_occurrence_seconds": round(multi * HOP / found.rate, 3),
                                       "head_seconds": round(head / found.rate, 6),
                                       "tail_seconds": round(tail / found.rate, 6),
                                       "length_seconds": round(length / found.rate, 6)}


# ------------------------------------------------------------------------------------------------ build

def _decode(path: Path) -> tuple[np.ndarray, int]:
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return samples, int(rate)


def build(members: dict[str, dict], links: list[dict], *, loader=_decode) -> tuple[np.ndarray, int, dict]:
    """The consensus asset of one group: (samples, rate, explanation). ``members`` are cue-memory members."""
    audio, rates = {}, set()
    for key, member in members.items():
        audio[key], rate = loader(Path(member["path"]))
        rates.add(rate)
    if len(rates) != 1:
        raise RecurrenceAssetError("同じ録音の公開cueの標本化周波数が一致しません")
    rate = rates.pop()
    mono = {key: value.mean(axis=1) for key, value in audio.items()}
    found = refine(warp(plan(members, links, rate), audio, mono), mono)
    centres, quality = consistency(found, mono)
    found, centres, quality, dropped = drop_inconsistent(found, mono, centres, quality)
    base_row = quality[next(n for n, layer in enumerate(found.layers) if layer.via == "base")]
    if not np.all(np.isnan(base_row)) and float(np.nanmedian(base_row)) < MIN_LAYER_CONSISTENCY:
        # the longest cue is not what the others play: its clock cannot carry the asset
        raise RecurrenceAssetError("最長の公開cueが他の使用箇所と一致しません")
    durations = {key: value.shape[0] for key, value in audio.items()}
    resolved = {key: not member.get("unresolved_extent") for key, member in members.items()}
    head, tail, chosen = edge_extensions(found, durations, resolved)
    signal, info = synthesize(found, audio, quality, centres, head=head, tail=tail, extension=chosen)
    placed = sorted({layer.occurrence for layer in found.layers})
    explanation = {
        **info, "method": METHOD, "recipe_id": recipe_id(), "base": found.base,
        "members_used": placed, "members_not_aligned": sorted(set(members) - set(placed)),
        "layers_left_out": [{**layer.to_dict(rate), "source_name": members[layer.occurrence].get("source_name"),
                             "median_consistency": value} for layer, value in dropped],
        "layers": [{**layer.to_dict(rate), "source_name": members[layer.occurrence].get("source_name"),
                    "median_consistency": (None if np.all(np.isnan(quality[number])) else
                                           round(float(np.nanmedian(quality[number])), 4))}
                   for number, layer in enumerate(found.layers)],
        "evidence_seconds": round(info["consensus_seconds"] + info["pair_seconds"] + info["tempo_seconds"]
                                  + info["head_seconds"] + info["tail_seconds"], 3)}
    return signal, rate, explanation


# ------------------------------------------------------------------------------------------------ publication

DIRECTORY = "bgm-recurrence-assets"
FOLDER_NAME = "BGM 統合版"
ALBUM = "LSS BGM 統合版"
ASSET_TAG = "LSS_BGM_RECURRENCE_ASSET"
SOURCE_TYPE = "bgm_recurrence_asset"
# Less than this of rebuilt or added audio is not a new asset: the longest occurrence already is that file.
MIN_EVIDENCE_SECONDS = 2.0
_PROVISIONAL = re.compile(r"^\d{2}:\d{2} ")
_SUFFIX = re.compile(r"（[^（）]*劇伴#\d+[^（）]*）")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root(state_root: Path) -> Path:
    return Path(state_root) / DIRECTORY


def _record_path(state_root: Path, number: str) -> Path:
    return root(state_root) / f"{number}.json"


def read_record(state_root: Path, number: str) -> dict | None:
    try:
        data = json.loads(_record_path(state_root, number).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RecurrenceAssetError(f"統合版の記録を読み込めません: {number}") from exc
    if (not isinstance(data, dict) or data.get("type") != "bgm_recurrence_asset"
            or data.get("schema_version") != SCHEMA_VERSION):
        raise RecurrenceAssetError(f"統合版の記録の形式が違います: {number}")
    return data


def _write_record(state_root: Path, number: str, data: dict) -> None:
    path = _record_path(state_root, number)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False).encode("utf-8")
    with NamedTemporaryFile(dir=path.parent, prefix=".lss-recurrence-", suffix=".tmp", delete=False) as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def signature(members: dict[str, dict]) -> str:
    body = json.dumps({"recipe_id": recipe_id(), "members": sorted((key, value["pcm_sha256"])
                                                                   for key, value in members.items())},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def target_path(base_member: dict, number: str) -> Path:
    """``<library>/BGM 統合版/劇伴#N 統合版.flac`` beside the episode folders (stable across revisions)."""
    return Path(base_member["path"]).parent.parent / FOLDER_NAME / f"劇伴#{number} 統合版.flac"


def title_for(base_member: dict, number: str) -> str:
    """The group's presentation title from the longest occurrence's current title, without its episode clock."""
    from mutagen.flac import FLAC

    try:
        title = (FLAC(base_member["path"]).get("TITLE") or [""])[0]
    except Exception:  # noqa: BLE001 - an unreadable title only loses the description
        title = ""
    title = _SUFFIX.sub("", _PROVISIONAL.sub("", title)).replace(" (区切り未確定)", "").strip()
    # One musical identity carries one title: the consolidated version is told apart by its album, folder and file
    # name, never by a label or variation note inside TITLE. A group with no name yet keeps its label as a placeholder.
    return title if title else f"劇伴#{number}（統合版）"


def _ours(path: Path, number: str) -> bool:
    """The file at ``path`` is a recurrence asset LSS wrote for this group (never touch anything else)."""
    from mutagen.flac import FLAC

    try:
        value = json.loads(FLAC(path)[ASSET_TAG][0])
    except Exception:  # noqa: BLE001
        return False
    return isinstance(value, dict) and str(value.get("label_number")) == str(number)


def _tags(number: str, title: str, explanation: dict, members: dict[str, dict]) -> dict:
    provenance = {
        "method": METHOD, "recipe_id": recipe_id(), "label": f"劇伴#{number}", "label_number": int(number),
        "base": explanation["base"], "consensus_seconds": explanation["consensus_seconds"],
        "head_seconds": explanation["head_seconds"], "tail_seconds": explanation["tail_seconds"],
        "members": [{"asset_id": key, "source_name": members[key].get("source_name"),
                     "source_sha256": members[key].get("source_sha256"),
                     "start_seconds": members[key].get("start_seconds"),
                     "pcm_sha256": members[key]["pcm_sha256"]} for key in explanation["members_used"]]}
    return {"TITLE": title, "ALBUM": ALBUM, "LSS_SOURCE_TYPE": SOURCE_TYPE, "LSS_STEM": "bgm",
            ASSET_TAG: json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}


def _publish(target: Path, signal: np.ndarray, rate: int, tags: dict) -> dict:
    """Stage, finalize (full LSS tag set, loudness tags, receipt) and atomically switch."""
    import soundfile
    from mutagen.flac import FLAC

    from lss.core import audio_identity
    from lss.output import finalizer

    target.parent.mkdir(parents=True, exist_ok=True)
    # Tags LSS never writes (COMMENT, RATING, a player's custom field) are not LSS's to remove: the rebuilt file is
    # written fresh, so they are carried over explicitly. Everything LSS writes comes from `tags` and the finalizer.
    carried = {}
    if target.exists():
        try:
            old = FLAC(target)
            dynamic = finalizer.metadata_keys(old)
            carried = {key.upper(): list(value) for key, value in old.items()
                       if not (key.upper().startswith("LSS_") or key.upper() in finalizer.WRITABLE_KEYS
                               or key.upper() in dynamic or key.upper().startswith("REPLAYGAIN_"))}
        except Exception:  # noqa: BLE001 - an unreadable old file is simply rebuilt
            carried = {}
    temporary = None
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=".lss-recurrence-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        if peak > 0.999:
            # the consensus may sum slightly over full scale: scale (never clip) and record it
            signal = signal * (0.999 / peak)
        soundfile.write(str(temporary), signal, rate, format="FLAC", subtype="PCM_24")
        if carried:
            staged = FLAC(temporary)
            for key, values in carried.items():
                staged[key] = values
            staged.save()
        record = finalizer.finalize_staged(temporary, tags=tags)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    audio = audio_identity.audio_identity(target)
    return {"path": str(target), "pcm_sha256": audio["pcm_sha256"], "frames": audio["frames"],
            "scaled_from_peak": round(peak, 6) if peak > 0.999 else None,
            "loudness_lufs": record["measurement"].get("loudness_lufs")}


def _keep_previous(state_root: Path, current: dict) -> str | None:
    """Copy the replaced revision's file into the state (one revision back) and return where it is."""
    path = Path(current["path"])
    if not path.exists():
        return None
    place = root(state_root) / "previous" / f"{current['pcm_sha256']}.flac"
    place.parent.mkdir(parents=True, exist_ok=True)
    if not place.exists():
        shutil.copy2(path, place)
    return str(place)


def update(state_root: Path, *, data: dict | None = None, builder=build) -> dict:
    """Build or refresh the recurrence asset of every recurring group in the published-cue memory.

    Returns ``{"state", "assets": [...]}``; each asset row says ``published`` / ``reused`` / ``kept_previous`` /
    ``no_consensus`` / ``path_occupied`` / ``failed``. One group's failure never stops another group, and nothing
    here changes an episode cue, its CurrentBest or the memory.
    """
    from lss.assets.naming import _locked
    from lss.bgm import cue_memory

    data = cue_memory.read(state_root) if data is None else data
    lock = root(state_root) / "assets.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with _locked(lock):
        # The consensus file goes beside a member's library: only a member that is still its source's current
        # published file (CurrentBest public_path) can name that library; remembered paths alone are not targets.
        present = {key: dict(value) for key, value in data["members"].items()
                   if cue_memory._still_present(dict(value)) and _is_current_public(state_root, value)}
        for number, entry in sorted(data["labels"].items(), key=lambda item: int(item[0])):
            if entry.get("merged_into"):
                continue
            members = {key: present[key] for key in entry.get("members") or [] if key in present}
            if len(members) < 2:
                continue
            links = [link for link in data["links"] if link["a"] in members and link["b"] in members]
            rows.append(_update_group(state_root, number, members, links, builder))
    published = [row for row in rows if row["state"] in ("published", "reused", "kept_previous")]
    return {"state": "checked", "recipe_id": recipe_id(), "assets": rows, "published_count": len(published)}


def _is_current_public(state_root: Path, member: dict) -> bool:
    from lss.bgm import current_best, write_targets

    row = current_best.published_files(state_root, member.get("source_sha256") or "").get(str(member.get("cue_key")))
    if row is None or row.get("pcm_sha256") != member.get("pcm_sha256"):
        return False
    try:
        # The asset is written beside this library: the row must be this state's own public file (`write_targets`).
        public = write_targets.checked(state_root, row, str(member.get("cue_key")))
        return public.resolve() == Path(member["path"]).resolve()
    except (OSError, RuntimeError, KeyError, TypeError, write_targets.WriteTargetRefused):
        return False


def _current_still_published(current: dict | None) -> dict | None:
    from lss.core import audio_identity

    if current is None:
        return None
    try:
        same = audio_identity.audio_identity(Path(current["path"]))["pcm_sha256"] == current["pcm_sha256"]
    except Exception:  # noqa: BLE001
        same = False
    # A removed or re-encoded file is no longer this revision; a new build may publish again, never blindly over it.
    return current if same else None


def _update_group(state_root: Path, number: str, members: dict, links: list[dict], builder) -> dict:
    sig = signature(members)
    try:
        record = read_record(state_root, number)
    except RecurrenceAssetError:
        record = None
    current = _current_still_published((record or {}).get("current"))
    row = {"label": f"劇伴#{number}", "label_number": int(number), "members": len(members)}
    if current is not None and current.get("signature") == sig:
        return {**row, "state": "reused", "path": current["path"], "evidence_seconds": current["evidence_seconds"],
                "consensus_seconds": current["consensus_seconds"]}
    candidate = (record or {}).get("candidate") or {}
    if candidate.get("signature") == sig and candidate.get("state") in ("no_consensus", "kept_previous"):
        # this exact member set was already judged; do not rebuild on every START
        return {**row, "state": candidate["state"], "path": (current or {}).get("path"),
                "evidence_seconds": candidate["evidence_seconds"]}
    try:
        signal, rate, explanation = builder(members, links)
    except Exception as exc:  # noqa: BLE001 - one group's failure is recorded, never raised
        return {**row, "state": "failed", "error": type(exc).__name__, "message": str(exc)[:200]}
    evidence = explanation["evidence_seconds"]
    base = members[explanation["base"]]
    target = target_path(base, number)
    data = record or {"type": "bgm_recurrence_asset", "schema_version": SCHEMA_VERSION,
                      "label_number": int(number), "history": []}
    judged = {"signature": sig, "evidence_seconds": evidence, "judged_at": _now(), "explanation": explanation}
    if evidence < MIN_EVIDENCE_SECONDS:
        data["candidate"] = {**judged, "state": "no_consensus"}
        _write_record(state_root, number, data)
        return {**row, "state": "no_consensus", "evidence_seconds": evidence, "path": (current or {}).get("path")}
    if current is not None and evidence < current["evidence_seconds"]:
        data["candidate"] = {**judged, "state": "kept_previous"}
        _write_record(state_root, number, data)
        return {**row, "state": "kept_previous", "evidence_seconds": current["evidence_seconds"],
                "path": current["path"], "candidate_evidence_seconds": evidence}
    if target.exists() and not (current is not None and Path(current["path"]) == target) and not _ours(target, number):
        data["candidate"] = {**judged, "state": "path_occupied"}
        _write_record(state_root, number, data)
        return {**row, "state": "path_occupied", "path": str(target)}
    previous_copy = _keep_previous(state_root, current) if current is not None else None
    try:
        published = _publish(target, signal, rate, _tags(number, title_for(base, number), explanation, members))
    except Exception as exc:  # noqa: BLE001
        return {**row, "state": "failed", "error": type(exc).__name__, "message": str(exc)[:200]}
    if current is not None:
        data["history"].append({**current, "replaced_at": _now(), "kept_copy": previous_copy})
    data["current"] = {**published, "signature": sig, "evidence_seconds": evidence,
                       "consensus_seconds": explanation["consensus_seconds"], "published_at": _now(),
                       "explanation": explanation,
                       "member_pcm": {key: value["pcm_sha256"] for key, value in members.items()}}
    data.pop("candidate", None)
    _write_record(state_root, number, data)
    return {**row, "state": "published", "path": published["path"], "evidence_seconds": evidence,
            "consensus_seconds": explanation["consensus_seconds"], "replaced_previous": current is not None}
