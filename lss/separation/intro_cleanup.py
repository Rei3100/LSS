"""Bounded Instrumental intro refinement; no gate, level recovery, or BGM use.

Quietness alone cannot authorize treatment. A weak introductory estimate under
an active source can nominate a short second separation. The accepted refinement
is localized before the first substantial backing and cannot change later PCM.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

POLICY = {"revision": "instrumental-intro-refinement-v1", "scan_seconds": 5.0,
          "context_seconds": 10.0, "frame_seconds": 0.05,
          "minimum_source_dbfs": -50.0, "maximum_estimate_dbfs": -55.0,
          "maximum_estimate_to_source_db": -30.0,
          "backing_minimum_dbfs": -50.0, "backing_minimum_to_source_db": -20.0,
          "minimum_eligible_seconds": 0.15, "transition_seconds": 0.05,
          "minimum_residue_reduction_db": 3.0,
          "normalization": "none", "quietness_is_absence": False,
          "scope": "Instrumental_leading_prefix_only"}


def _db(energy):
    return 10.0 * np.log10(max(float(energy), 1e-30))


def nomination(source: np.ndarray, estimate: np.ndarray, rate: int) -> dict | None:
    """Nominate a prefix using aligned stereo samples; do not declare absence."""
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("Intro analysis needs a positive sample rate")
    source, estimate = np.asarray(source), np.asarray(estimate)
    if source.shape != estimate.shape or source.ndim != 2 or source.shape[1] != 2 \
            or not np.isfinite(source).all() or not np.isfinite(estimate).all():
        raise ValueError("Intro analysis needs aligned finite stereo samples")
    step = max(1, round(rate * POLICY["frame_seconds"]))
    stop = min(len(source), round(rate * POLICY["scan_seconds"]))
    eligible = []
    backing = stop
    for start in range(0, stop, step):
        end = min(start+step, stop)
        mix = float(np.mean(np.square(source[start:end], dtype=np.float64)))
        target = float(np.mean(np.square(estimate[start:end], dtype=np.float64)))
        mix_db, target_db = _db(mix), _db(target)
        ratio = _db(target/max(mix, 1e-30))
        if target_db >= POLICY["backing_minimum_dbfs"] \
                and ratio >= POLICY["backing_minimum_to_source_db"]:
            backing = start
            break
        if mix_db >= POLICY["minimum_source_dbfs"] \
                and target_db <= POLICY["maximum_estimate_dbfs"] \
                and ratio <= POLICY["maximum_estimate_to_source_db"]:
            eligible.append((start,end))
    if sum(end-start for start,end in eligible) < rate*POLICY["minimum_eligible_seconds"]:
        return None
    return {"start":eligible[0][0], "end":backing, "eligible":eligible,
            "observation":"weak_leading_estimate_under_active_source_not_absence"}


def refine(estimate: np.ndarray, second: np.ndarray, region: dict, rate: int):
    """Crossfade within the nominated prefix; keep every other sample exactly."""
    if estimate.shape != second.shape or not np.isfinite(second).all():
        raise ValueError("Intro refinement needs a complete finite estimate")
    start,end = region["start"],region["end"]
    if not 0 <= start < end <= len(estimate):
        raise ValueError("Intro refinement region is outside the estimate")
    fade = min(round(rate*POLICY["transition_seconds"]), (end-start)//2)
    weight = np.ones(end-start, dtype=estimate.dtype)
    if fade:
        ramp = np.linspace(0,1,fade,endpoint=True,dtype=estimate.dtype)
        weight[:fade] = ramp
        weight[-fade:] = ramp[::-1]
    result = estimate.copy()
    result[start:end] += (second[start:end]-estimate[start:end])*weight[:,None]
    before = sum(float(np.sum(np.square(estimate[a:b],dtype=np.float64)))
                 for a,b in region["eligible"])
    after = sum(float(np.sum(np.square(result[a:b],dtype=np.float64)))
                for a,b in region["eligible"])
    reduction = _db(before/max(after,1e-30))
    accepted = bool(before > 0 and reduction >= POLICY["minimum_residue_reduction_db"])
    return (result if accepted else estimate), {"adopted":accepted,
            "residue_reduction_db":float(reduction), "start_sample":start,"end_sample":end,
            "scope":"introductory_prefix", "no_gain_compensation":True}


def process(source_path: Path, estimate_path: Path, work: Path, *, separate) -> dict:
    """Read bounded context, optionally separate it, and stage a full-length stem."""
    info = sf.info(estimate_path)
    frames = min(info.frames, round(info.samplerate*POLICY["context_seconds"]))
    estimate, rate = sf.read(estimate_path, frames=frames, dtype='float32',always_2d=True)
    source_info = sf.info(source_path)
    source,source_rate = sf.read(source_path,
            frames=min(source_info.frames, int(np.ceil(frames*source_info.samplerate/rate))+160),
            dtype='float32',always_2d=True)
    if source.shape[1] == 1:
        source = np.repeat(source,2,axis=1)
    if source_rate != rate:
        divisor = gcd(source_rate,rate)
        source = resample_poly(source,rate//divisor,source_rate//divisor,axis=0)
    source = source[:len(estimate)]
    region = nomination(source,estimate,rate)
    if region is None:
        return {"adopted":False,"reason":"no_eligible_intro","policy":POLICY}
    work.mkdir(parents=True,exist_ok=False)
    crop = work/'intro.wav'
    sf.write(crop,estimate,rate,subtype='FLOAT')
    second = separate(crop,work/'second')
    values,second_rate = sf.read(second,dtype='float32',always_2d=True)
    if second_rate != rate:
        raise ValueError("Intro refinement changed sample rate")
    refined,receipt = refine(estimate,values,region,rate)
    receipt.update(policy=POLICY)
    if not receipt["adopted"]:
        return receipt
    # Never mutate the original raw stem until the complete replacement is
    # flushed. FLOAT serialization retains float32 provider samples exactly.
    staged = work/'refined.wav'
    with sf.SoundFile(estimate_path) as source_file, sf.SoundFile(staged,mode='w',
            samplerate=rate,channels=info.channels,subtype='FLOAT',format='WAV') as target:
        target.write(refined)
        source_file.seek(len(estimate))
        while len(block := source_file.read(1<<18,dtype='float32',always_2d=True)):
            target.write(block)
    staged_info = sf.info(staged)
    if (staged_info.frames,staged_info.channels,staged_info.samplerate) != \
            (info.frames,info.channels,info.samplerate):
        raise ValueError("Intro refinement changed the full stem layout")
    receipt["refined_path"] = str(staged)
    return receipt
