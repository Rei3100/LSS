"""Audio-checked positive nominations and covered recording continuity.

Keep the established offline presence detector. External positions only nominate
bounded searches: an acoustic transition is required; rejected nominations stay
in Evidence. A continuous, waveform-verified *covered interval* can veto an
internal split. Neither observation excludes audio or creates EXACT relations.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from lss.bgm import cue_semantics as cs


@dataclass(frozen=True)
class Policy:
    revision: str = "audio-checked-positive-cue-formation-v6"
    search_seconds: int = 6
    minimum_local_distance: float = .08
    strong_local_distance: float = .19
    minimum_gap_between_cuts: float = 3.
    continuation_seconds: int = 8
    continuation_similarity: float = .92
    continuation_music_probability: float = .25
    fine_hop_seconds: float = .025
    fine_window_seconds: float = .1
    fine_search_seconds: float = 1.
    nominated_head_probability: float = .4
    nominated_head_anchor_seconds: int = 4


POLICY = Policy()


def revision():
    return asdict(POLICY)


def _distance(a, b):
    norm = float(np.linalg.norm(a)*np.linalg.norm(b))
    return max(0., 1-float(a@b)/norm) if norm else 0.


def _local(e, at):
    return max(_distance(e.embeddings[max(0, at-h):at].mean(0),
                         e.embeddings[at:min(e.count, at+h)].mean(0))
               for h in (2, 4) if at > 0 and at < e.count)


def _continuity(matches, at):
    """Do not extend a positive waveform match beyond its measured coverage."""
    covering = [m for m in matches if m.get("identity_state") == "known_exact"
                and m["query_start_seconds"]+.5 < at < m["query_end_seconds"]-.5]
    if not covering:
        return False
    # Conflicting recordings/offsets are uncertainty, never a continuity veto.
    return len({(m["reference_sha256"], round(m["reference_start_seconds"]
                - m["query_start_seconds"], 1)) for m in covering}) == 1


def _fine_position(music, seconds, kind):
    if music is None:
        return seconds
    import soundfile as sf
    with sf.SoundFile(str(music)) as stream:
        rate = stream.samplerate
        a = max(0, round((seconds-POLICY.fine_search_seconds)*rate))
        b = min(stream.frames, round((seconds+POLICY.fine_search_seconds)*rate))
        stream.seek(a)
        x = stream.read(b-a, dtype="float64", always_2d=True)
    hop, width = max(1, round(rate*POLICY.fine_hop_seconds)), max(1, round(rate*POLICY.fine_window_seconds))
    if len(x) < width:
        return seconds
    starts = np.arange(0, len(x)-width+1, hop)
    energy = np.mean(x*x, axis=1)
    prefix = np.r_[0., np.cumsum(energy)]
    db = 10*np.log10(np.maximum((prefix[starts+width]-prefix[starts])/width, 1e-20))
    times = (a+starts+width/2)/rate
    if kind == "change":
        # A localized notch nominates a sample-accurate switch; no whole cue
        # is trimmed to the notch. Without a notch keep the semantic position.
        i = int(np.argmin(db))
        if float(np.median(db)-db[i]) >= 12:
            return float(times[i])
        return seconds
    present = db >= max(-60., float(np.percentile(db, 90))-35.)
    crossings = np.flatnonzero(np.diff(present.astype(int)) == (1 if kind == "start" else -1))
    if not len(crossings):
        return seconds
    t = times[crossings+1]
    closest = float(t[np.argmin(np.abs(t-seconds))])
    return closest


def _recover_components(parts, candidates, components, observations):
    """Recover covered music without converting weak correlation to absence.

    A support envelope can expand or reconnect a classifier omission. It never
    shortens an existing tail. An incoming recording can nominate a cut only
    where no other measured recording still occupies its interior.
    """
    from lss.bgm import recording_support as rs
    supports = []
    records = (components or {}).get("records", [])
    corroborated = {(r["reference_sha256"],r["onset"]["predicted_seconds"]) for r in records
                    if r.get("state") == "component_observed" and any(
                        abs(s["start_seconds"]-r["onset"]["predicted_seconds"]) <= POLICY.search_seconds*2
                        for s in r["spans"])}
    for record in records:
        if record.get("state") != "component_observed":
            continue
        onset = record["onset"]
        if (record["reference_sha256"],onset["predicted_seconds"]) not in corroborated:
            continue  # a louder late edit cannot establish a missing intro
        for span in record["spans"]:
            supports.append({**span, "onset": onset, "reference_sha256": record["reference_sha256"],
                             "offset_seconds": record["offset_seconds"],"render_trace":record.get("render_trace"),
                             "role_unresolved":record.get("music_corroborated") is False})
    result = [dict(p, start_edge=dict(p["start_edge"]), end_edge=dict(p["end_edge"])) for p in parts]
    for previous,current in zip(result,result[1:]):
        edge=current["start_edge"]
        if (previous["end_seconds"] != current["start_seconds"]
                or edge.get("authority") != "audio_checked_positive_nomination"):
            continue
        incoming=[s for s in supports if s["onset"]["track"] == edge.get("track_label")
                  and abs(s["start_seconds"]-current["start_seconds"]) <= 1.]
        if not incoming:continue
        support=min(incoming,key=lambda s:abs(s["start_seconds"]-current["start_seconds"]))
        begin=support["start_seconds"]+rs.POLICY.window_seconds/2
        outgoing=[s for s in supports if s["reference_sha256"] != support["reference_sha256"]
                  and s["start_seconds"] < begin and s["end_seconds"] >= begin-1.]
        overlap=[s for s in outgoing if s["end_seconds"]-rs.POLICY.window_seconds/2 > begin+.2]
        if overlap:
            # Layered music has two component envelopes, not one forced switch.
            observations.append({"state":"overlapping_recording_components","seconds":begin,
                                 "references":[support["reference_sha256"],*[s["reference_sha256"] for s in overlap]]})
            continue
        if outgoing and min(abs(s["end_seconds"]-rs.POLICY.window_seconds/2-begin) for s in outgoing) <= .25:
            if previous["start_seconds"] < begin < current["end_seconds"]:
                old=current["start_seconds"]
                edge={**edge,"authority":"two_recording_component_transition","coarse_seconds":old,
                      "uncertainty_seconds":rs.POLICY.window_seconds,
                      "recording_identity":"unresolved"}
                previous["end_seconds"]=current["start_seconds"]=begin
                previous["end_edge"]=current["start_edge"]=edge
                observations.append({"state":"two_component_switch_refined","before_seconds":old,"seconds":begin,
                                     "track_label":support["onset"]["track"]})
    for lo, hi, members in cs._stretches(candidates):
        local = sorted([p for p in result if p["stretch"] == [float(lo),float(hi)]],
                       key=lambda p:p["start_seconds"])
        positive = [s for s in supports if s["end_seconds"]>lo and s["start_seconds"]<hi]
        for support in positive:
            a,b = max(float(lo),support["start_seconds"]),min(float(hi),support["end_seconds"])
            # A rounded stretch can contain an unmaterialized subsecond hole.
            # Positive source evidence cannot make a missing render witness.
            covered = sorted((max(a,x),min(b,y)) for x,y in members if x<b and y>a)
            cursor=a
            for x,y in covered:
                if x>cursor+1e-6:break
                cursor=max(cursor,y)
            if cursor<b-1e-6:
                continue
            overlaps = [p for p in local if p["end_seconds"]>=a and p["start_seconds"]<=b]
            if not overlaps:
                continue  # do not introduce a disconnected cue from a title
            first,last = overlaps[0],overlaps[-1]
            if first["start_edge"]["kind"] == cs.EDGE_SILENCE and a < first["start_seconds"]:
                observations.append({"state":"recording_component_recovers_head", "span":[a,first["start_seconds"]],
                                     "track_label":support["onset"]["track"]})
                # Before an unclipped support start neither the classifier nor the
                # measured recording hears this music: the recovered head keeps a
                # silence edge. A candidate edge would hold the whole useful body
                # (real 137: 413.038 head, 57.6 s cue held). Only support clipped
                # at the materialized stretch may continue outside it.
                clipped = support["start_seconds"] <= float(lo)+1e-6
                first["start_edge"] = {"kind":cs.EDGE_CANDIDATE if clipped else cs.EDGE_SILENCE,
                                      "authority":"observed_recording_component",
                                      "previous_seconds":first["start_seconds"],"recording_identity":"unresolved"}
                first["start_seconds"] = a
            if last["end_edge"]["kind"] == cs.EDGE_SILENCE and b > last["end_seconds"]:
                if b>=float(hi)-1e-6:
                    observations.append({"state":"positive_tail_exceeds_materialized_support", "span":[last["end_seconds"],b],
                                         "track_label":support["onset"]["track"]})
                elif rs.tail_supported(support,last["end_seconds"],b):
                    observations.append({"state":"recording_component_recovers_tail", "span":[last["end_seconds"],b],
                                         "track_label":support["onset"]["track"]})
                    # b is inside the materialized stretch: the support ends here.
                    last["end_edge"] = {"kind":cs.EDGE_SILENCE,"authority":"observed_recording_component",
                                        "previous_seconds":last["end_seconds"],"recording_identity":"unresolved"}
                    last["end_seconds"] = b
                else:
                    observations.append({"state":"rendered_tail_not_corroborated", "span":[last["end_seconds"],b],
                                         "track_label":support["onset"]["track"]})
            for left,right in zip(overlaps,overlaps[1:]):
                gap_a,gap_b = left["end_seconds"],right["start_seconds"]
                competing = any(s["reference_sha256"] != support["reference_sha256"]
                                and s["start_seconds"]+rs.POLICY.window_seconds/2 < gap_b
                                and s["end_seconds"]-rs.POLICY.window_seconds/2 > gap_a for s in supports)
                incoming = any(gap_a-1 <= r["predicted_seconds"] <= gap_b+1
                               and r.get("track") != support["onset"]["track"] for r in (components or {}).get("onsets",[])) or any(gap_a-1 <= s["onset"]["predicted_seconds"] <= gap_b+1
                               and s["onset"]["track"] != support["onset"]["track"] for s in supports)
                if (not competing and not incoming and a <= gap_a < gap_b <= b
                        and gap_b-gap_a <= POLICY.continuation_seconds
                        and left["end_edge"].get("authority") not in (
                            "audio_checked_positive_nomination","two_recording_component_transition")
                        and right["start_edge"].get("authority") not in (
                            "audio_checked_positive_nomination","two_recording_component_transition")
                        and left["end_edge"]["kind"] in (cs.EDGE_SILENCE,cs.EDGE_CANDIDATE)
                        and right["start_edge"]["kind"] in (cs.EDGE_SILENCE,cs.EDGE_CANDIDATE)):
                    left["end_seconds"] = right["end_seconds"]
                    left["end_edge"] = right["end_edge"]
                    left["candidates"] = sorted({tuple(x) for x in left["candidates"]+right["candidates"]})
                    local.remove(right)
                    observations.append({"state":"recording_component_recovers_classifier_hole",
                                         "span":[gap_a,gap_b],"track_label":support["onset"]["track"],
                                         "reference_sha256":support["reference_sha256"],
                                         "identity_authority":"none"})
        # Positive support changes only its candidate stretch. Core semantics
        # include recovered audible music, without extending materialization.
        result = [p for p in result if p["stretch"] != [float(lo),float(hi)]]+local
    result.sort(key=lambda p:p["start_seconds"])
    for left,right in zip(result,result[1:]):
        if left["end_seconds"] > right["start_seconds"]:
            # Existing strong switches retain their authority and both sides.
            if right["start_edge"]["kind"] == cs.EDGE_CHANGE:
                left["end_seconds"] = right["start_seconds"]
            else:
                left["end_seconds"] = min(left["end_seconds"],right["start_seconds"])
    for part in result:
        part["core_seconds"] = [part["start_seconds"],part["end_seconds"]]
        unresolved = [[max(part["start_seconds"],s["start_seconds"]),min(part["end_seconds"],s["end_seconds"])]
                      for s in supports if s.get("role_unresolved") and s["start_seconds"] < part["end_seconds"]
                      and s["end_seconds"] > part["start_seconds"]]
        if unresolved:
            part["unresolved_recording_regions"] = unresolved
    return result


def _recover_nominated_heads(parts, candidates, evidence, usage, matches, components, mixture_music, observations,
                             *, recipe=None):
    """A near-threshold intro must not be joined to its outgoing track.

    Sparse usage only nominates. Require a strong *local* acoustic transition,
    continuous materialized music, near-threshold separator role throughout the
    omitted head, and stable classified music after it. Missing mixture tags
    retain the ordinary result. Only measured mixture Music seconds supply the
    extra role witness; lower scores stay unknown.
    """
    if mixture_music is None:
        return parts, []
    mixture_music = np.asarray(mixture_music)
    if (mixture_music.shape != (evidence.count,) or not np.isfinite(mixture_music).all()
            or np.any((mixture_music < 0)|(mixture_music > 1))):
        raise ValueError("Nominated head needs aligned mixture Music probabilities")
    result = [dict(p, start_edge=dict(p["start_edge"]), end_edge=dict(p["end_edge"])) for p in parts]
    recipe = recipe or cs.default_recipe()
    positive_seconds = set()
    for left, right in zip(result, result[1:]):
        a, b = int(np.ceil(left["end_seconds"])), int(np.floor(right["start_seconds"]))
        if (left["stretch"] != right["stretch"] or not 0 < b-a <= POLICY.continuation_seconds
                or left["end_edge"]["kind"] != cs.EDGE_SILENCE or right["start_edge"]["kind"] != cs.EDGE_SILENCE):
            continue
        anchor = min(evidence.count, b+POLICY.nominated_head_anchor_seconds, int(right["end_seconds"]))
        if anchor-b < POLICY.nominated_head_anchor_seconds or not np.all(evidence.music_probability[b:anchor] >= .5):
            continue
        nominations = [r for r in usage.get("onsets", []) if r.get("role") == "score"
                       and a-POLICY.search_seconds <= r["predicted_seconds"] <= b]
        if len(nominations) != 1:
            continue
        row = nominations[0]
        lo = max(int(left["start_seconds"])+2, a-POLICY.search_seconds)
        options = [(t, _local(evidence, t)) for t in range(lo, a+1) if not _continuity(matches, t)]
        if not options:
            continue
        t, distance = max(options, key=lambda x:x[1]*np.exp(-.5*((x[0]-row["predicted_seconds"])/3.)**2))
        if distance < POLICY.strong_local_distance:
            continue
        if (not np.all(evidence.music_db[t:b] >= -60.)
                or not np.all(evidence.music_probability[t:b] >= POLICY.nominated_head_probability)):
            continue
        if any(t < r["predicted_seconds"] < b and r.get("track") != row["track"] for r in usage.get("onsets", [])):
            continue
        # A component continuing across an incoming track may be an overlay.
        # Keep both original parts instead of enforcing a single switch.
        records = (components or {}).get("records", [])
        if any(r.get("state") == "component_observed" and r["onset"]["track"] != row["track"]
               and any(s["start_seconds"]+.25 < t < s["end_seconds"]-.25 for s in r["spans"]) for r in records):
            observations.append({"state":"nominated_head_has_overlapping_component", "seconds":t})
            continue
        # Exact union coverage; rounding a candidate stretch cannot fill a hole.
        cursor = float(t)
        for first,last in sorted(candidates):
            if last <= cursor:continue
            if first > cursor+1e-6:break
            cursor = max(cursor,float(last))
            if cursor >= right["start_seconds"]:break
        if cursor < right["start_seconds"]-1e-6:
            continue
        old = right["start_seconds"]
        witnesses = [i for i in range(t,b) if mixture_music[i] >= recipe.music_probability]
        from lss.bgm import recording_support as rs
        role_mask = rs.confirmed_music_windows(components,evidence.count)
        previous = cs.decide([right],evidence,recipe,source_seconds=evidence.count,
                             additional_music_windows=role_mask)[0]
        role_mask[witnesses] = True
        proposed = {**right,"start_seconds":float(t),"core_seconds":[float(t),right["end_seconds"]]}
        measured = cs.decide([proposed],evidence,recipe,source_seconds=evidence.count,
                             additional_music_windows=role_mask)[0]
        if (previous["features"]["music_fraction"] >= recipe.minimum_music_fraction
                and measured["features"]["music_fraction"] < recipe.minimum_music_fraction):
            observations.append({"state":"nominated_head_kept_without_downgrading_body", "span":[float(t),old],
                                 "music_corroborated_seconds":witnesses,
                                 "proposed_music_fraction":measured["features"]["music_fraction"]})
            continue
        edge = {"kind":cs.EDGE_CHANGE, "authority":"nominated_head_local_transition",
                "distance":round(distance,4), "track_label":row["track"],
                "predicted_seconds":row["predicted_seconds"], "uncertainty_seconds":1.,
                "recording_identity":"unresolved"}
        left["end_seconds"] = right["start_seconds"] = float(t)
        left["end_edge"] = dict(edge)
        right["start_edge"] = dict(edge)
        positive_seconds.update(witnesses)
        right["nominated_head"] = {"span":[float(t),old], "music_corroborated_seconds":witnesses,
                                    "unconfirmed_seconds":[i for i in range(t,b) if i not in witnesses]}
        observations.append({"state":"nominated_head_recovered_without_outgoing_join", "span":[float(t),old],
                             "previous_outgoing_end":float(a), "edge":edge, "music_corroborated_seconds":witnesses})
    for part in result:
        part["core_seconds"] = [part["start_seconds"],part["end_seconds"]]
    return result, sorted(positive_seconds)


def segment(candidates, evidence, recipe, *, usage=None, matches=(), music=None, components=None, mixture_music=None,
            stereo_continuations=()):
    candidates = list(candidates)
    base = cs.segment(candidates, evidence, recipe)
    usage = usage or {"state": "not_available", "onsets": []}
    rows = []
    # First restore low-confidence classifier holes only with agreeing flanks,
    # audible music in the gap, and no externally nominated intervening onset.
    joined = []
    for item in base:
        if joined:
            left = joined[-1]; a, b = int(left["end_seconds"]), int(item["start_seconds"])
            if 0 < b-a <= POLICY.continuation_seconds:
                lhs = evidence.embeddings[max(int(left["start_seconds"]), a-4):a].mean(0)
                rhs = evidence.embeddings[b:min(int(item["end_seconds"]), b+4)].mean(0)
                same = 1-_distance(lhs, rhs) >= POLICY.continuation_similarity
                heard = bool(np.all(evidence.music_db[a:b] >= recipe.floor_music_db)
                             and np.all(evidence.music_probability[a:b] >= POLICY.continuation_music_probability))
                nomination = any(a-1 <= r["predicted_seconds"] <= b+1 for r in usage["onsets"])
                supplementary = next((r for r in stereo_continuations
                                      if r.get("state") == "stereo_music_continuation_observed"
                                      and r.get("span") == [a, b]
                                      and r.get("flanks") == [left["core_seconds"], item["core_seconds"]]), None)
                if ((same and heard) or supplementary) and not nomination and left["stretch"] == item["stretch"]:
                    left.update(end_seconds=item["end_seconds"], end_edge=item["end_edge"],
                                core_seconds=[left["core_seconds"][0], item["core_seconds"][1]],
                                candidates=sorted({tuple(x) for x in left["candidates"]+item["candidates"]}))
                    rows.append({"state": "stereo_continuation_hole_joined" if supplementary else "continuation_hole_joined",
                                 "span": [a, b], "identity_authority": "none"})
                    continue
                rows.append({"state": "continuation_hole_refused", "span": [a, b],
                             "similarity": round(float(1-_distance(lhs, rhs)), 4),
                             "similar_enough": bool(same),
                             "min_music_probability": round(float(np.min(evidence.music_probability[a:b])), 4),
                             "min_music_db": round(float(np.min(evidence.music_db[a:b])), 2),
                             "audible": heard, "nominated_onset": nomination,
                             "same_stretch": left["stretch"] == item["stretch"], "identity_authority": "none"})
            # Only a measured continuous recording may veto a recursive cut.
            if a == b and _continuity(matches, a):
                left.update(end_seconds=item["end_seconds"], end_edge=item["end_edge"],
                            core_seconds=[left["core_seconds"][0], item["core_seconds"][1]],
                            candidates=sorted({tuple(x) for x in left["candidates"]+item["candidates"]}))
                rows.append({"state": "covered_recording_veto", "seconds": a})
                continue
        joined.append(dict(item))
    out = []
    for item in joined:
        a, b = item["start_seconds"], item["end_seconds"]
        cuts = []
        for onset in usage["onsets"]:
            predicted = onset["predicted_seconds"]
            if not a+POLICY.minimum_gap_between_cuts < predicted < b-POLICY.minimum_gap_between_cuts:
                continue
            lo, hi = max(int(a)+3, int(np.floor(predicted))-POLICY.search_seconds), min(int(b)-3, int(np.ceil(predicted))+POLICY.search_seconds)
            options = [(t, _local(evidence, t)) for t in range(lo, hi+1) if not _continuity(matches, t)]
            if not options:
                rows.append({"state": "nomination_vetoed_by_covered_continuity", "onset": onset})
                continue
            # Position prior handles noisy broadcast annotations; acoustic
            # change supplies all acceptance authority.
            t, distance = max(options, key=lambda x: x[1]*np.exp(-.5*((x[0]-predicted)/3.)**2))
            if distance < POLICY.minimum_local_distance:
                rows.append({"state": "nomination_without_acoustic_change", "onset": onset, "distance": distance})
                continue
            t = _fine_position(music, float(t), "change")
            if cuts and t-cuts[-1][0] < POLICY.minimum_gap_between_cuts:
                continue
            strong = distance >= POLICY.strong_local_distance
            edge = {"kind": cs.EDGE_CHANGE if strong else cs.EDGE_CANDIDATE,
                    "distance": round(distance, 4), "authority": "audio_checked_positive_nomination",
                    "predicted_seconds": predicted, "track_label": onset["track"],
                    "recording_identity": "unresolved"}
            cuts.append((t, edge))
            rows.append({"state": "acoustic_nomination_admitted", "seconds": t, "edge": edge})
        cuts.sort()
        points = [a, *[t for t, _ in cuts], b]
        for i, (low, high) in enumerate(zip(points, points[1:])):
            part = {**item, "start_seconds": low, "end_seconds": high, "core_seconds": [low, high],
                    "start_edge": dict(item["start_edge"] if i == 0 else cuts[i-1][1]),
                    "end_edge": dict(item["end_edge"] if i == len(cuts) else cuts[i][1])}
            for key, kind in (("start_seconds", "start"), ("end_seconds", "end")):
                edge = part["start_edge" if kind == "start" else "end_edge"]
                if edge["kind"] == cs.EDGE_SILENCE:
                    refined = _fine_position(music, part[key], kind)
                    # Stay inside materialized candidates and the neighboring
                    # part. Unknown waveform outside support is never zero.
                    refined = min(item["stretch"][1], max(item["stretch"][0], refined))
                    if low-1 <= refined <= high+1 and part["start_seconds"] < (refined if kind == "end" else part["end_seconds"]):
                        edge["coarse_seconds"] = part[key]
                        part[key] = refined
            out.append(part)
    # Refinement must not create overlapping authority or erase a short cue.
    for left, right in zip(out, out[1:]):
        if left["end_seconds"] > right["start_seconds"]:
            midpoint = (left["end_seconds"]+right["start_seconds"])/2
            left["end_seconds"] = right["start_seconds"] = midpoint
    out, head_music = _recover_nominated_heads(out,candidates,evidence,usage,matches,components,mixture_music,rows,recipe=recipe)
    if components:
        out = _recover_components(out,candidates,components,rows)
    record = {"policy": revision(), "usage": usage, "observations": rows,
              "recording_components": components,
              "nominated_music_seconds":head_music,
              "stereo_continuations":list(stereo_continuations),
              "boundary_count": len(out), "source_grid": "decoded_audio_seconds",
              "identity_authority": "none", "absence_authority": "none"}
    return out, record
