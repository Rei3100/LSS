"""Nominated local recording presence, with no identity or absence authority.

A usage title narrows the search; phase alignment and time-shift controls must
agree over multiple waveform windows. Support can coexist with another cue.
Nothing here authorizes EXACT suppression, replacement or a donor render.
"""
from dataclasses import asdict, dataclass
import re
import unicodedata

import numpy as np


@dataclass(frozen=True)
class Policy:
    revision: str = "nominated-recording-component-support-v4"
    search_before_seconds: float = 12.
    maximum_seconds: float = 180.
    window_seconds: float = .5
    hop_seconds: float = .1
    seed_fraction: float = .06
    present_fraction: float = .025
    minimum_seed_seconds: float = 3.
    control_ratio: float = 4.
    maximum_hole_seconds: float = 1.
    minimum_rendered_tail_fraction: float = .2
    stereo_projection: str = "difference_with_rendered_and_mono_tail_veto"


POLICY = Policy()


def revision():
    return asdict(POLICY)


def key(text):
    value = re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text)).casefold()
    return value.replace("tvanimever", "animever")


def alignment(query, reference):
    """Reference sample = query sample + returned offset; do not fit content."""
    size = 1 << (len(query)+len(reference)-1).bit_length()
    cross = np.fft.rfft(reference, size)*np.fft.rfft(query[::-1], size)
    cc = np.fft.irfft(cross/(np.abs(cross)+1e-12), size)[:len(query)+len(reference)-1]
    return int(np.argmax(cc))-(len(query)-1)


def trace(query, reference, offset, rate):
    """Windowed coherent power and displaced-clock negative controls."""
    width, hop = round(POLICY.window_seconds*rate), round(POLICY.hop_seconds*rate)
    starts = np.arange(0, len(query)-width+1, hop)
    x = query.astype(np.float64)
    def sums(values):
        prefix = np.r_[0., np.cumsum(values)]
        return prefix[starts+width]-prefix[starts]
    sx, ex = sums(x), sums(x*x)
    ex -= sx*sx/width
    def fractions(shift):
        y = np.zeros(len(x))
        a, b = max(0, -shift), min(len(x), len(reference)-shift)
        if b > a:
            y[a:b] = reference[a+shift:b+shift]
        sy = sums(y)
        ey = sums(y*y)-sy*sy/width
        dot = sums(x*y)-sx*sy/width
        valid = (starts+shift >= 0) & (starts+width+shift <= len(reference))
        return np.where(valid, np.maximum(dot,0)**2/np.maximum(ex*ey,1e-30),0.)
    values = fractions(offset)
    control = np.maximum.reduce([fractions(offset+round(s*rate)) for s in (-3.7,-1.3,1.3,3.7)])
    return (starts+width/2)/rate, values, control


def _runs(flags):
    result, low = [], None
    for i, flag in enumerate([*flags, False]):
        if flag and low is None:
            low = i
        elif not flag and low is not None:
            result.append((low, i))
            low = None
    return result


def observe(query, reference, rate, *, origin=0., offset=None):
    """Measure positive component intervals; weak windows remain unknown."""
    query, reference = np.asarray(query), np.asarray(reference)
    if (query.ndim != 1 or reference.ndim != 1 or type(rate) is not int or rate < 100
            or min(len(query), len(reference)) < round(POLICY.window_seconds*rate)
            or not np.isfinite(query).all() or not np.isfinite(reference).all()):
        raise ValueError("Recording support needs finite mono waveforms on one grid")
    offset = alignment(query, reference) if offset is None else int(offset)
    times, fraction, control = trace(query, reference, offset, rate)
    seed = (fraction >= POLICY.seed_fraction) & (fraction >= POLICY.control_ratio*control)
    present = (fraction >= POLICY.present_fraction) & (fraction >= POLICY.control_ratio*control)
    if seed.sum()*POLICY.hop_seconds < POLICY.minimum_seed_seconds:
        return {"state": "alignment_not_corroborated", "spans": [], "offset_seconds": offset/rate-origin,
                "seed_seconds": float(seed.sum()*POLICY.hop_seconds)}
    # Only bounded internal observation holes; a track interruption stays in
    # the trace, rather than being asserted silent or borrowed from a donor.
    connected = present.copy()
    for a, b in _runs(~present):
        if a and b < len(present) and (b-a)*POLICY.hop_seconds <= POLICY.maximum_hole_seconds:
            connected[a:b] = True
    spans = []
    for a, b in _runs(connected):
        if seed[a:b].sum()*POLICY.hop_seconds < POLICY.minimum_seed_seconds:
            continue
        low = max(0., times[a]-POLICY.window_seconds/2)+origin
        high = min(len(query)/rate, times[b-1]+POLICY.window_seconds/2)+origin
        spans.append({"start_seconds": low, "end_seconds": high,
                      "seed_seconds": float(seed[a:b].sum()*POLICY.hop_seconds),
                      "median_fraction": float(np.median(fraction[a:b])),
                      "median_control": float(np.median(control[a:b]))})
    return {"state": "component_observed" if spans else "alignment_not_corroborated", "spans": spans,
            "offset_seconds": offset/rate-origin, "trace_origin": origin, "times": times.tolist(),
            "fraction": fraction.tolist(), "control": control.tolist(), "present": present.tolist(),
            "identity_authority": "none", "absence_authority": "none"}


def tail_supported(support,a,b):
    """Presence in the mixture alone cannot admit a noisy rendered tail."""
    trace_data=support.get("render_trace") or {}
    times=np.asarray(trace_data.get("times",[]))+trace_data.get("origin",0.)
    values=np.asarray(trace_data.get("fraction",[]))
    control=np.asarray(trace_data.get("control",[]))
    mask=(times>=a-POLICY.hop_seconds/2)&(times<=b)
    return bool(mask.any() and float(np.median(values[mask]))>=POLICY.minimum_rendered_tail_fraction
                and float(np.median(values[mask]))>=POLICY.control_ratio*float(np.median(control[mask])))


def confirmed_music_windows(components,count):
    """Local measured presence can corroborate a classifier hole, not absence.

    Require at least 80% observed subwindows in a source second. Duplicate
    offsets and references cannot increase coverage. Unknown tails stay unknown.
    """
    lattice=round(1/POLICY.hop_seconds)
    covered=np.zeros(count*lattice,dtype=bool)
    for record in (components or {}).get('records',[]):
        if record.get('state')!='component_observed':continue
        if record.get('music_corroborated') is False:continue
        times=np.asarray(record['times'])+record['trace_origin']
        fraction,control=np.asarray(record['fraction']),np.asarray(record['control'])
        positive=(fraction>=POLICY.present_fraction)&(fraction>=POLICY.control_ratio*control)
        bins=np.floor(times[positive]*lattice).astype(int)
        bins=bins[(bins>=0)&(bins<len(covered))]
        covered[bins]=True
    return covered.reshape(count,lattice).mean(axis=1)>=.8


def observe_stereo(query, music, reference, rate, *, origin=0.):
    """Additional coherent music witness when centered interference masks mono.

    A side-only source match is insufficient. Require three seconds of the
    same clock in selected separator music too. Mono rendered-tail checks stay
    intact: a clean side cannot certify a noisy whole stereo tail.
    """
    values = [np.asarray(x) for x in (query, music, reference)]
    if any(x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all() for x in values):
        raise ValueError("Stereo presence needs finite two-channel waveforms")
    query, music, reference = values
    if query.shape != music.shape:
        raise ValueError("Stereo presence music must cover the same source window")
    x, y, rendered = query[:, 0]-query[:, 1], reference[:, 0]-reference[:, 1], music[:, 0]-music[:, 1]
    offsets = {alignment(x, y)}
    for delta in (10., 16., 22., 30.):
        a, b = round(delta*rate), min(len(x), round((delta+8)*rate))
        if b-a >= 3*rate:
            offsets.add(alignment(x[a:b], y)-a)
    records, admitted = [], []
    for offset in sorted(offsets):
        if any(abs(offset-previous) < round(.05*rate) for previous in admitted):
            continue
        observed = observe(x, y, rate, origin=origin, offset=offset)
        if observed["state"] != "component_observed":
            continue
        times, fraction, control = trace(rendered, y, offset, rate)
        positive = (fraction >= POLICY.seed_fraction) & (fraction >= POLICY.control_ratio*control)
        if positive.sum()*POLICY.hop_seconds < POLICY.minimum_seed_seconds:
            continue
        render_seen = observe(rendered, y, rate, origin=origin, offset=offset)
        if render_seen["state"] != "component_observed":
            continue
        spans = []
        for source_span in observed["spans"]:
            for render_span in render_seen["spans"]:
                a = max(source_span["start_seconds"],render_span["start_seconds"])
                b = min(source_span["end_seconds"],render_span["end_seconds"])
                if a < b:
                    spans.append({**source_span,"start_seconds":a,"end_seconds":b})
        if not spans:
            continue
        observed["spans"] = spans
        admitted.append(offset)
        mono_times, mono_fraction, mono_control = trace(music.mean(1), reference.mean(1), offset, rate)
        # Presence authority requires both source and the selected music at
        # each local clock cell, rather than votes anywhere in the long span.
        source_fraction = np.asarray(observed["fraction"])
        source_control = np.asarray(observed["control"])
        joint = ((source_fraction >= POLICY.present_fraction)
                 & (source_fraction >= POLICY.control_ratio*source_control)
                 & (fraction >= POLICY.present_fraction) & (fraction >= POLICY.control_ratio*control))
        observed["fraction"] = np.where(joint, np.minimum(source_fraction, fraction), 0.).tolist()
        observed["control"] = np.maximum(source_control, control).tolist()
        observed["render_trace"] = {"origin": origin, "times": mono_times.tolist(),
                                     "fraction": mono_fraction.tolist(), "control": mono_control.tolist()}
        observed["projection_render_trace"] = {"origin": origin, "times": times.tolist(),
                                                "fraction": fraction.tolist(), "control": control.tolist()}
        observed["projection"] = "stereo_difference"
        observed["music_corroborated"] = True
        records.append(observed)
    return records


def nominate(query, library, usage, rate, *, music=None, source_path=None, music_path=None):
    """Reuse exact-SHA reference metadata and loaders; titles only nominate."""
    records = []
    for row in usage.get("onsets", []):
        if row.get("role") != "score":
            continue
        title = key(row["track"])
        for ref in library.index.references:
            meta = library.metadata.get(ref.sha256) or {}
            tags = meta.get("tags") or {}
            labels = [meta.get("display_title"), meta.get("file_name"), tags.get("title")]
            if not title or not any(isinstance(t, str) and title in key(t) for t in labels):
                continue
            try:
                reference = library.samples(ref.sha256)
            except (OSError, ValueError, RuntimeError):
                continue
            if len(reference)<rate:
                continue
            low = max(0., row["predicted_seconds"]-POLICY.search_before_seconds)
            high = min(len(query)/rate, low+min(POLICY.maximum_seconds, len(reference)/rate+24))
            a, b = round(low*rate), round(high*rate)
            if b-a < rate:
                continue
            offsets = {alignment(query[a:b], reference)}
            # Program edits can change the offset later. Early independent
            # seeds avoid allowing a louder late excerpt to hide the intro.
            for delta in (-2., 4., 10., 18.):
                first = max(a, round((row["predicted_seconds"]+delta)*rate))
                last = min(b, first+8*rate)
                if last-first >= 3*rate:
                    offsets.add(alignment(query[first:last],reference)-(first-a))
            admitted = []
            for shift in sorted(offsets):
                if any(abs(shift-previous)<round(.05*rate) for previous in admitted):
                    continue
                observed = observe(query[a:b], reference, rate, origin=a/rate, offset=shift)
                if observed["state"] == "component_observed":
                    admitted.append(shift)
                    if music is not None and len(music)>=b:
                        times,values,control=trace(music[a:b],reference,shift,rate)
                        observed["render_trace"]={"origin":a/rate,"times":times.tolist(),
                                                  "fraction":values.tolist(),"control":control.tolist()}
                    records.append({"onset": row, "reference_sha256": ref.sha256, **observed})
            if not admitted:
                records.append({"onset":row,"reference_sha256":ref.sha256,
                                "state":"alignment_not_corroborated","spans":[]})
            if source_path is not None and music_path is not None:
                from pathlib import Path
                from lss.bgm import identity
                try:
                    path = Path(ref.path)
                    if not path.is_file():
                        loader = library.index.loader
                        if not hasattr(loader, "path"):
                            continue
                        path = loader.path(ref.sha256)
                    if identity._sha256(path) != ref.sha256:
                        continue
                    stereo_ref = identity.decode_stereo_window(path, 0, len(reference))
                    stereo_query = identity.decode_stereo_window(source_path, a, b-a)
                    stereo_music = identity.decode_stereo_window(music_path, a, b-a)
                    if identity._sha256(path) != ref.sha256:
                        continue
                    records.extend({"onset": row, "reference_sha256": ref.sha256, **record}
                                   for record in observe_stereo(stereo_query, stereo_music, stereo_ref, rate, origin=a/rate))
                except (OSError, ValueError, RuntimeError):
                    continue  # optional stereo evidence cannot remove the mono/local result
    return {"policy": revision(), "records": records, "identity_authority": "none", "absence_authority": "none"}
