"""Conservative original islands on one continuous, declared sample grid.

Separator disagreement is uncertainty, never accompaniment absence. This stage
does not amplify a weak estimate or run an extra separator. It restores original
samples only where channel, transient and frequency-band checks agree that the
first estimate and the selected processed witness follow the source. Digital source silence is the sole
absence authority. All fades live inside the admitted region.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
import hashlib
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from lss.separation import regions


@dataclass(frozen=True)
class Policy:
    revision: str = "selective-original-islands-v7"
    window_seconds: float = .25
    transient_seconds: float = .025
    maximum_residual_db: float = -40.
    maximum_transient_residual_db: float = -35.
    maximum_band_residual_db: float = -25.
    band_relative_floor: float = 1e-6
    bin_relative_floor: float = 1e-6
    maximum_bin_residual_db: float = -18.
    maximum_diffuse_residual_fraction: float = .25
    maximum_gain_reduction_similarity: float = .995
    minimum_original_seconds: float = .5
    transition_seconds: float = .08
    guard_seconds: float = .05

    def to_dict(self):
        return {**asdict(self), "absence": "exact_aligned_source_zero_only",
                "quietness_is_absence": False, "gain_compensation": "none",
                "unknown": "retain_processed", "transition": "raised_cosine_inside_original",
                "alignment": "global_rational_grid_polyphase_with_context_v1"}


POLICY = Policy()
LEGACY_V6_POLICY = {**POLICY.to_dict(), "revision": "selective-original-islands-v6"}
LEGACY_V6_POLICY.pop("maximum_gain_reduction_similarity")
LEGACY_V5_POLICY = {**LEGACY_V6_POLICY, "revision": "selective-original-islands-v5"}
LEGACY_V5_POLICY.pop("maximum_diffuse_residual_fraction")


def revision():
    # Freeze both the choices and executable implementation before reuse lookup.
    source = Path(__file__)
    if not source.is_file():
        from lss.paths import resource_root
        source = resource_root() / "lss" / "separation" / "selective.py"
    from lss.core.runtime import text_source_sha256
    region_source = Path(regions.__file__)
    if not region_source.is_file():
        from lss.paths import resource_root
        region_source = resource_root() / "lss" / "separation" / "regions.py"
    body = {"policy": POLICY.to_dict(), "implementation_sha256": text_source_sha256(source),
            "region_implementation_sha256": text_source_sha256(region_source)}
    body["policy_id"] = "selective-v1-" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def _validate(source, estimate, rate):
    source, estimate = np.asarray(source), np.asarray(estimate)
    if type(rate) is not int or rate <= 0:
        raise ValueError("Selective composition needs a positive integer rate")
    if (source.shape != estimate.shape or source.ndim != 2 or source.shape[1] != 2
            or len(source) == 0 or not np.isfinite(source).all() or not np.isfinite(estimate).all()):
        raise ValueError("Selective composition needs aligned finite stereo samples")
    return source, estimate


def _within(residual, source, db):
    # Relative tests have no absolute loudness floor. A tiny instrument is still
    # a signal; a silent channel cannot hide contamination in the other channel.
    return bool(np.all(np.sum(residual ** 2, axis=0)
                       <= np.sum(source ** 2, axis=0) * 10 ** (db / 10)))


def _clean(source, estimate, rate):
    x, delta = source.astype(np.float64), (source.astype(np.float64) - estimate)
    if not _within(delta, x, POLICY.maximum_residual_db):
        return False
    # A scaled-down estimate supplies no authority to restore the removed
    # component: wanted attenuation and collinear unwanted sound are ambiguous.
    # Preserve that estimate instead of turning a level difference into content.
    energy = np.sum(x*x, axis=0)
    removed_energy = np.sum(delta*delta, axis=0)
    projection = np.sum(delta*x, axis=0)
    norm = np.sqrt(energy * removed_energy)
    if np.any((removed_energy > energy*1e-12)
              & (projection >= norm*POLICY.maximum_gain_reduction_similarity)):
        return False
    step = max(1, round(rate * POLICY.transient_seconds))
    for low in range(0, len(x), step):
        if not _within(delta[low:low+step], x[low:low+step], POLICY.maximum_transient_residual_db):
            return False
    # Do not let bass energy conceal a missing quiet high instrument or a
    # localized high-frequency contaminant. Keep stereo channels independent.
    taper = np.hanning(len(x))[:, None] if len(x) > 2 else np.ones((len(x), 1))
    spectrum = np.abs(np.fft.rfft(x * taper, axis=0)) ** 2
    error = np.abs(np.fft.rfft(delta * taper, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(x), 1 / rate)
    total = np.sum(spectrum, axis=0)
    for low, high in ((0, 200), (200, 1000), (1000, 4000), (4000, rate / 2 + 1)):
        bins = (frequencies >= low) & (frequencies < high)
        energy = np.sum(spectrum[bins], axis=0)
        important = energy > total * POLICY.band_relative_floor
        if np.any(important & (np.sum(error[bins], axis=0)
                              > energy * 10 ** (POLICY.maximum_band_residual_db / 10))):
            return False
        # Low total energy does not make removed broadband material wanted.
        # Effective spectral support catches diffuse subthreshold residuals
        # which the old per-bin floor hid underneath strong musical tones.
        removed = error[bins]
        if len(removed) >= 8:
            mass = removed.sum(axis=0)
            for channel in range(x.shape[1]):
                if mass[channel] <= total[channel] * 1e-12:
                    continue
                p = removed[:, channel] / mass[channel]
                positive = p > 0
                support = np.exp(-np.sum(p[positive] * np.log(p[positive]))) / len(p)
                if support >= POLICY.maximum_diffuse_residual_fraction:
                    return False
    # A faint narrowband sound can share a broad band with a loud instrument.
    # Test energetic bins too: the broad-band denominator must not authorize
    # restoring a contaminant which the provider already reduced substantially.
    important = spectrum > total[None, :] * POLICY.bin_relative_floor
    if np.any(important & (error > spectrum * 10 ** (POLICY.maximum_bin_residual_db / 10))):
        return False
    return True


def _runs(flags, step, frames, minimum):
    result = []
    low = None
    for index, flag in enumerate(list(flags) + [False]):
        if flag and low is None:
            low = index * step
        elif not flag and low is not None:
            high = min(index * step, frames)
            if high - low >= minimum:
                result.append([low, high])
            low = None
    return result


def plan(source, estimate, rate, *, allowed=None, witness=None):
    """Admit original islands without undoing a selected later repair.

    A later processed witness is an additional veto, never permission to undo
    the first estimate. Semantic permission also only vetoes original use.
    """
    source, estimate = _validate(source, estimate, rate)
    if witness is not None:
        _, witness = _validate(source, witness, rate)
    step = max(1, round(rate * POLICY.window_seconds))
    count = (len(source) + step - 1) // step
    if allowed is not None and (np.asarray(allowed).shape != (count,)
                                or np.asarray(allowed).dtype != np.dtype(bool)):
        raise ValueError("Semantic permissions must cover every selective window")
    clean, silent = [], []
    for index, low in enumerate(range(0, len(source), step)):
        x, y = source[low:low+step], estimate[low:low+step]
        zero = not np.any(x != 0)
        silent.append(zero)
        clean.append(bool(not zero and (allowed is None or allowed[index]) and _clean(x, y, rate)
                          and (witness is None or _clean(x, witness[low:low+step], rate))))
    raw = _runs(clean, step, len(source), round(rate * POLICY.minimum_original_seconds))
    zeros = _runs(silent, step, len(source), round(rate * POLICY.minimum_original_seconds))
    return {"policy": POLICY.to_dict(), "frames": len(source), "sample_rate": rate,
            "original_regions": raw, "silent_regions": zeros,
            "original_witness": "first_and_selected_processed" if witness is not None else "selected_estimate",
            "uncertain_windows": count - sum(clean) - sum(silent),
            "absence_scope": "digital_source_silence; voice_only_is_unresolved"}


def weight(frames, regions, rate):
    """Raised-cosine transitions strictly inside each admitted island."""
    from lss.separation.regions import inside_weight
    return inside_weight(frames, regions, rate, POLICY.transition_seconds, guard_seconds=POLICY.guard_seconds)


def _region_weight(index, low, high, frames, rate):
    return regions.inside_weight(frames, [[low, high]], rate, POLICY.transition_seconds,
                                 guard_seconds=POLICY.guard_seconds, start=int(index[0]), length=len(index))


def render(source, processed, receipt):
    """Replay a frozen plan; every sample outside its weights stays unchanged."""
    source, processed = _validate(source, processed, receipt["sample_rate"])
    if len(source) != receipt["frames"] or receipt["policy"] not in (POLICY.to_dict(), LEGACY_V6_POLICY, LEGACY_V5_POLICY):
        raise ValueError("Selective plan layout or policy changed")
    original = weight(len(source), receipt["original_regions"], receipt["sample_rate"])
    silent = weight(len(source), receipt["silent_regions"], receipt["sample_rate"])
    if np.any((original > 0) & (silent > 0)):
        raise ValueError("Original and absent regions overlap")
    for low, high in receipt["silent_regions"]:
        if np.any(source[low:high] != 0):
            raise ValueError("Silence requires exact source-zero evidence")
    w = np.maximum(original, silent)
    return regions.blend(processed, source, w)


def aligned_read(path, rate, start, frames, *, dtype="float32"):
    """Read bounded native context on a global rational grid, never reset phase."""
    with sf.SoundFile(path) as stream:
        native = stream.samplerate
        divisor = gcd(native, rate)
        up, down = rate // divisor, native // divisor
        if native == rate:
            stream.seek(start)
            data = stream.read(frames, dtype=dtype, always_2d=True)
        else:
            # scipy's default polyphase filter has 10*max(up,down) taps per
            # side. Extra context on the rational input lattice makes tiled
            # reads exactly equivalent to a full-length resample.
            margin = 32 * down
            low = max(0, (start // up - 32) * down)
            high = min(stream.frames, ((start + frames + up - 1) // up) * down + margin)
            stream.seek(low)
            native_data = stream.read(high-low, dtype=dtype, always_2d=True)
            data = resample_poly(native_data, up, down, axis=0)
            offset = start - low * up // down
            data = data[offset:offset+frames]
        if stream.channels == 1:
            data = np.repeat(data, 2, axis=1)
    if data.shape != (frames, 2) or not np.isfinite(data).all():
        raise ValueError("Source does not cover the declared stereo output grid")
    return data


def process(source_path, estimate_path, destination):
    """Bounded-memory staging; leave provider/source files untouched on failure."""
    source_info, info = sf.info(source_path), sf.info(estimate_path)
    expected = (source_info.frames * info.samplerate + source_info.samplerate - 1) // source_info.samplerate
    if info.frames != expected or info.channels != 2:
        raise ValueError("Selective estimate layout does not match source")
    rate, frames = info.samplerate, info.frames
    # First scan records only decisions. Rendering later uses global regions,
    # so block boundaries do not introduce fades or restart a clean run.
    step = max(1, round(rate * POLICY.window_seconds))
    clean, silent = [], []
    with sf.SoundFile(estimate_path) as stream:
        for start in range(0, frames, step * 32):
            size = min(step * 32, frames-start)
            source = aligned_read(source_path, rate, start, size)
            estimate = stream.read(size, dtype="float32", always_2d=True)
            _validate(source, estimate, rate)
            for low in range(0, size, step):
                x, y = source[low:low+step], estimate[low:low+step]
                zero = not np.any(x != 0)
                silent.append(zero)
                clean.append(not zero and _clean(x, y, rate))
    receipt = {"policy": POLICY.to_dict(), "frames": frames, "sample_rate": rate,
               "original_regions": _runs(clean, step, frames, round(rate * POLICY.minimum_original_seconds)),
               "silent_regions": _runs(silent, step, frames, round(rate * POLICY.minimum_original_seconds)),
               "source_rate": source_info.samplerate, "source_resampled": source_info.samplerate != rate,
               "uncertain_windows": len(clean)-sum(clean)-sum(silent),
               "absence_scope": "digital_source_silence; voice_only_is_unresolved"}
    if not receipt["original_regions"] and not receipt["silent_regions"]:
        return receipt
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(estimate_path) as stream, sf.SoundFile(destination, mode="w", samplerate=rate,
            channels=2, subtype="FLOAT", format="WAV") as target:
        for start in range(0, frames, step * 32):
            size = min(step * 32, frames-start)
            source = aligned_read(source_path, rate, start, size)
            estimate = stream.read(size, dtype="float32", always_2d=True)
            # Keep global weights (at most an 8-second tile in memory).
            out = estimate.copy()
            for key in ("original_regions", "silent_regions"):
                for low, high in receipt[key]:
                    a, b = max(start, low), min(start+size, high)
                    if a >= b:
                        continue
                    w = _region_weight(np.arange(a, b), low, high, frames, rate)
                    sel = slice(a-start, b-start)
                    x, y = source[sel], estimate[sel]
                    out[sel] = regions.blend(y, x, w)
            target.write(out)
    receipt["staged_path"] = str(destination)
    return receipt
