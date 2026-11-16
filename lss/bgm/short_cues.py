"""Positive source-only support for physically bounded short musical assets.

Classifier gaps alone do not establish a complete short cue. Require continuous
Music observations on both source and separated music, plus measured quiet
boundaries on both channels of that actual separated signal. Existing content,
reference, program-insert, integrity and CurrentBest gates still decide publication.
No waveform is changed and no original-mixture permission is granted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from lss.bgm import cue_semantics as cs


@dataclass(frozen=True)
class Policy:
    revision: str = "physically-bounded-short-music-v6"
    minimum_seconds: float = 2.0
    ordinary_minimum_seconds: float = 10.0
    boundary_search_seconds: float = .5
    quiet_seconds: float = .02
    quiet_dbfs: float = -60.
    music_probability: float = .5


POLICY = Policy()


def revision():
    return asdict(POLICY)


def _boundary(stream, seconds, side):
    rate = stream.samplerate
    center = round(seconds * rate)
    reach = round(POLICY.boundary_search_seconds * rate)
    width = max(1, round(POLICY.quiet_seconds * rate))
    first, last = ((center-reach, center) if side == "start" else (center, center+reach))
    # Missing context is not silence; never pad a clipped asset with zeros.
    if first < 0 or last > stream.frames or stream.channels != 2:
        return {"confirmed": False, "reason": "boundary_context_missing"}
    stream.seek(first)
    value = stream.read(last-first, dtype="float64", always_2d=True)
    if len(value) != last-first or not np.isfinite(value).all():
        return {"confirmed": False, "reason": "boundary_signal_invalid"}
    energy = np.vstack([np.zeros(2), np.cumsum(value*value, axis=0)])
    power = (energy[width:]-energy[:-width])/width
    # A quiet left channel cannot vouch for the right channel.
    quiet = np.all(power <= 10**(POLICY.quiet_dbfs/10), axis=1)
    indexes = np.flatnonzero(quiet)
    if not len(indexes):
        return {"confirmed": False, "reason": "no_quiet_boundary"}
    index = int(indexes[-1] if side == "start" else indexes[0])
    adjusted = (first+index+width)/rate if side == "start" else (first+index)/rate
    return {"confirmed": True, "seconds": adjusted,
            "quiet_span_seconds": [(first+index)/rate, (first+index+width)/rate]}


def support(segment, evidence, mixture, music, *, music_origin_seconds=0.):
    """Return a scoped positive receipt, or an explicit unsuccessful observation."""
    a, b = float(segment["start_seconds"]), float(segment["end_seconds"])
    receipt = {"policy": revision(), "span_seconds": [a, b], "confirmed": False,
               "original_mixture_authority": "none", "absence_authority": "none"}
    if not POLICY.minimum_seconds <= b-a < POLICY.ordinary_minimum_seconds:
        return {**receipt, "reason": "outside_short_duration"}
    if any(segment[side]["kind"] not in (cs.EDGE_SILENCE, "edit_boundary")
           for side in ("start_edge", "end_edge")):
        return {**receipt, "reason": "short_edges_unresolved"}
    # Re-measure the actual child interval. Edit splitting intentionally keeps
    # parent observations, which cannot establish a child's duration/continuity.
    first, last = int(np.floor(a)), int(np.ceil(b))
    mix = np.asarray(mixture)
    if (first < 0 or last > evidence.count or mix.ndim != 2
            or mix.shape[1] <= cs.MUSIC_CLASS or last > len(mix) or first >= last):
        return {**receipt, "reason": "short_observations_missing"}
    # Continuity is measured on the 1-s windows lying wholly inside the child. A short
    # cue rarely starts on the window grid; a window straddling its onset or release
    # mostly holds the silence before/after it and cannot show a gap in the music.
    # At least two whole windows are required; straddling windows still bound the
    # continuation checks below.
    inner_first, inner_last = int(np.ceil(a - 1e-6)), int(np.floor(b + 1e-6))
    if inner_last - inner_first < 2:
        inner_first, inner_last = first, last
    source = mix[inner_first:inner_last, cs.MUSIC_CLASS]
    if (not np.isfinite(source).all() or not np.all(source >= POLICY.music_probability)
            or not np.all(evidence.music_probability[inner_first:inner_last] >= POLICY.music_probability)
            or not np.all(evidence.music_db[inner_first:inner_last] >= POLICY.quiet_dbfs)):
        return {**receipt, "reason": "short_music_not_continuous"}
    source_edges = {}
    for side, index in (("start", first-1), ("end", last)):
        if segment[side+"_edge"]["kind"] == "edit_boundary":
            # split_at_edit_boundaries establishes this from actual all-channel
            # source silence. Adjacent 3-second tagging windows can overlap the
            # different music on its other side; they cannot undo an audio cut.
            source_edges[side] = {"authority": "source_digital_silence"}
            continue
        if index < 0 or index >= len(mix):
            return {**receipt, "reason": "source_boundary_observation_missing"}
        probability = float(mix[index, cs.MUSIC_CLASS])
        if not np.isfinite(probability) or probability >= POLICY.music_probability:
            return {**receipt, "reason": "source_music_continues_outside_short"}
        source_edges[side] = {"outside_window": index, "music_probability": probability,
                              "authority": "no_observed_source_music_continuation"}
    receipt["source_boundaries"] = source_edges
    import soundfile as sf
    try:
        with sf.SoundFile(str(music)) as stream:
            edges = {side: _boundary(stream, t-music_origin_seconds, side)
                     for side, t in (("start", a), ("end", b))}
            for edge in edges.values():
                if edge["confirmed"]:
                    edge["seconds"] += music_origin_seconds
                    edge["quiet_span_seconds"] = [t+music_origin_seconds for t in edge["quiet_span_seconds"]]
    except (OSError, RuntimeError, ValueError):
        return {**receipt, "reason": "short_waveform_unavailable"}
    receipt["boundaries"] = edges
    receipt["observed_windows"] = [first, last]
    receipt["confirmed"] = all(edge["confirmed"] for edge in edges.values())
    if receipt["confirmed"]:
        low, high = edges["start"]["seconds"], edges["end"]["seconds"]
        # Search only outside the proposed body and include its complete original
        # waveform. An interior quiet notch can never authorize trimming it.
        covered = sorted((max(low, x), min(high, y)) for x, y in segment.get("candidates", [])
                         if x < high and y > low)
        cursor = low
        for x, y in covered:
            if x > cursor+1e-7:
                break
            cursor = max(cursor, y)
        if (cursor < high-1e-7 or high-low >= POLICY.ordinary_minimum_seconds
                or int(np.floor(low)) != first or int(np.ceil(high)) != last):
            return {**receipt, "confirmed": False, "reason": "short_extension_not_supported"}
        receipt["observed_span_seconds"] = [a, b]
        receipt["span_seconds"] = [low, high]
    receipt["reason"] = "bounded_short_music" if receipt["confirmed"] else "short_boundary_not_confirmed"
    return receipt


def apply(segments, evidence, mixture, music, recipe, source_seconds, *, music_origin_seconds=0.):
    """Re-evaluate only positively supported short children; preserve all others."""
    from dataclasses import replace
    from lss.bgm import reference_relations
    # A checked segment that no reference explains carries state "unknown"; only that (or no check at all)
    # may take the acoustic short route. (Comparing with "UNKNOWN" excluded every short cue once references
    # were checked: real ep1 eyecatch, 2026-10-19.)
    result = []
    for segment in segments:
        a, b = float(segment["start_seconds"]), float(segment["end_seconds"])
        if not POLICY.minimum_seconds <= b-a < POLICY.ordinary_minimum_seconds:
            result.append(segment)
            continue
        # Reference/duplicate/role decisions are authoritative and cannot be
        # laundered through the acoustic short-cue route. "music_not_continuous" is
        # admitted: on a short span it usually comes from a window straddling the
        # onset, and `support` re-measures continuity strictly (every whole inner
        # window >= 0.5 Music on source and separated music) before anything changes.
        if (segment["semantic_class"] not in (cs.WANTED, cs.UNCERTAIN)
                or set(segment.get("reasons", [])) - {"shorter_than_wanted_minimum", "possible_change_inside",
                                                      "music_not_continuous"}
                or segment.get("reference_relation", {}).get("state") not in (None, reference_relations.UNKNOWN)):
            result.append(segment)
            continue
        observation = support(segment, evidence, mixture, music, music_origin_seconds=music_origin_seconds)
        item = {**segment, "short_music": observation}
        if observation["confirmed"]:
            low, high = observation["span_seconds"]
            previous_end = float(result[-1]["end_seconds"]) if result else 0.
            next_start = (float(segments[len(result)+1]["start_seconds"])
                          if len(result)+1 < len(segments) else source_seconds)
            if low < previous_end or high > next_start:
                item["short_music"] = {**observation, "confirmed": False, "reason": "short_extension_overlaps_neighbor"}
                result.append(item)
                continue
            item.update(start_seconds=low, end_seconds=high,
                        start_edge={**segment["start_edge"], "short_boundary_previous_seconds": a},
                        end_edge={**segment["end_edge"], "short_boundary_previous_seconds": b})
            # The body is judged on the same whole windows that established continuity
            # (a straddling onset window mostly holds the silence before the cue).
            body = [float(np.ceil(low - 1e-6)), float(np.floor(high + 1e-6))]
            if body[1] - body[0] < 2.0:
                body = [low, high]
            measured = cs.decide([{**item, "core_seconds": body}], evidence,
                                 replace(recipe, minimum_wanted_seconds=POLICY.minimum_seconds),
                                 source_seconds=source_seconds)[0]
            # Do not inherit a long parent's possible-change observation.
            item.update({key: measured[key] for key in ("semantic_class", "reasons", "features")}, core_seconds=[low, high])
        result.append(item)
    return result


def confirmed(segment):
    """A proof for a different span/policy cannot authorize this candidate."""
    row = segment.get("short_music") or {}
    return (row.get("confirmed") is True and row.get("policy") == revision()
            and row.get("span_seconds") == [float(segment["start_seconds"]), float(segment["end_seconds"])])
