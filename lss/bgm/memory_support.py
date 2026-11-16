"""Local recording presence from LSS's own published cues of other episodes (no external table, no donor).

The published-cue memory (`lss.bgm.cue_memory`) holds the CurrentBest BGM cues LSS itself published for other
sources. When the episode being processed plays one of those recordings, the memory knows how the recording goes
on before and after the part this episode's classifier heard: a quiet intro the tagger missed, a tail under
dialogue, or a classifier hole inside one continuous recording.

This module only *measures*. For each first-pass segment it looks the segment's music up among the published cues
of other sources (`identity.match`, landmark fingerprint plus waveform verification on separated music, the easy
direction of the matcher). A verified member then becomes a local reference for the established recording-support
measurement (`recording_support.observe`: windowed coherent power against displaced-clock controls, in this
episode's own source mix, plus a render trace in this episode's own music stem). The records have the same shape
as released-reference component records, so `cue_formation` applies its existing, audited rules (head only up to a
silence edge inside the materialized stretch, tail only when the rendered music corroborates it, hole only between
silence/candidate edges with no competing recording).

- Dialogue is mostly centred: where it masks the mono measurement, the stereo-difference measurement of
  `recording_support.observe_stereo` (side channel in the source and in this episode's music, mono render veto) is
  added, exactly as for released references.
- The member's audio is never rendered, mixed or copied: what is published is still this episode's own audio.
- A prior published cue is not a release: no EXACT, redundancy, suppression or absence authority.
- Same-source members (an earlier run of this episode, or another file of the same name) are not used: an
  episode cannot corroborate itself.
- Memory never lowers the single-episode result: presence that continues before a materialized candidate stretch
  is recorded but starts the span just inside it, so a recovered head keeps a silence edge instead of a
  candidate edge (which would hold a cue that is published without memory).
- The member was published as wanted music in its own episode by LSS's own decision; that is the music-role
  witness for the aligned span (`music_corroborated`), the same role the prior occurrence plays in
  `occurrence_support`, but without any usage table.
- Members of one memory group (`劇伴#N`, the same recording heard in several episodes) are one recording identity
  here (`reference_sha256 = cue-memory-group:N`), so two copies of one recording never count as two competing
  recordings.
- `join_covered` restores one cue where a single recording, measured on one clock, plays straight through a
  first-pass split (a classifier hole or a strong embedding change inside the recording), only when the joined cue
  is not judged lower than its parts.
- Missing memory, unreadable members or failed matching leave the single-episode result unchanged.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Policy:
    revision: str = "published-cue-memory-component-support-v1"
    # A first-pass segment shorter than this is not looked up (too little music to verify a recording).
    minimum_query_seconds: float = 6.0
    # Same threshold as the memory's own links (`cue_memory.MIN_EXPLAINED`).
    minimum_explained: float = 0.5
    # How far around the member's aligned extent this episode's source is measured.
    margin_seconds: float = 12.0
    maximum_members_per_segment: int = 3
    maximum_members: int = 24
    # A memory span must reach this far into both parts to join them.
    join_flank_seconds: float = 2.0


POLICY = Policy()
JOIN_FLANK_SECONDS = POLICY.join_flank_seconds


def revision() -> dict:
    return asdict(POLICY)


def _members(state_root: Path, source_sha256: str, source_name: str | None) -> list[dict]:
    from lss.bgm import cue_memory

    data = cue_memory.read(state_root)
    label_of = {key: number for number, entry in data.get("labels", {}).items()
                if not entry.get("merged_into") for key in entry.get("members") or []}
    present = []
    for member in data.get("members", {}).values():
        if member.get("source_sha256") == source_sha256 or (source_name and member.get("source_name") == source_name):
            continue
        if cue_memory._still_present(member):
            number = label_of.get(member["asset_id"])
            present.append({**member, "recording_identity": f"cue-memory-group:{number}" if number
                            else member["pcm_sha256"]})
    return present


def _current_best_member(state_root: Path, member: dict, cache: dict[str, dict | None]) -> dict | None:
    """Return the exact current accepted row for a peer; memory membership alone is not current membership."""
    from lss.bgm import current_best

    sha = member["source_sha256"]
    if sha not in cache:
        try:
            record = current_best.read(state_root, sha)
            cache[sha] = record["current"] if record is not None else None
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            cache[sha] = None
    current = cache[sha]
    if not current:
        return None
    path = Path(member["path"])
    for row in current.get("files") or []:
        try:
            same_path = Path(row["public_path"]).resolve() == path.resolve()
        except (OSError, RuntimeError):
            same_path = False
        if (same_path and row.get("cue_key") == member.get("cue_key")
                and row.get("pcm_sha256") == member.get("pcm_sha256")):
            return {key: row.get(key) for key in ("cue_key", "public_path", "pcm_sha256", "run_key",
                                                   "recipe_id", "start_seconds", "end_seconds", "adopted_at")}
    return None


def _measurement_component_ranges(cells: list[int], sample_rate: int) -> list[dict[str, list[int]]]:
    """Exact half-open query sample ranges read by `_Evidence.measure` for each cell."""
    from lss.bgm import reference_relations

    lo, hi = cells[0] * sample_rate, (cells[-1] + 1) * sample_rate
    full = 2 * sample_rate
    ranges = []
    for cell in cells:
        first = int(round((cell - 0.5) * sample_rate))
        size = full
        if min(first + full, hi) - max(first, lo) >= sample_rate:
            size = min(first + full, hi) - max(first, lo)
            first = max(first, lo)
        ranges.append({"music": [first, first + size],
                       "interference": [first - reference_relations._HALF,
                                        first - reference_relations._HALF + size + reference_relations.TAPS - 1]})
    return ranges


def _sample_hash(audio: np.ndarray, bounds: list[int]) -> str:
    first, last = max(0, int(bounds[0])), min(audio.size, int(bounds[1]))
    return hashlib.sha256(np.ascontiguousarray(audio[first:last]).tobytes()).hexdigest()


def _file_stat_identity(path: Path) -> tuple[int, ...] | None:
    """A cheap observation guard around decoding; PCM identity remains authoritative."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            getattr(stat, "st_dev", 0), getattr(stat, "st_ino", 0))


def _peer_waveform_binding(path: Path, member: dict, current_row: dict | None,
                           reference: np.ndarray, before_stat: tuple[int, ...] | None,
                           after_stat: tuple[int, ...] | None, state_root: Path) -> dict:
    """Bind the decoded diagnostic waveform to the original peer and its still-current publication row."""
    from lss.bgm import identity
    from lss.core import audio_identity

    mono = np.ascontiguousarray(np.asarray(reference, dtype=np.float32))
    mono_digest = hashlib.sha256(mono.tobytes()).hexdigest()
    binding = {"original_pcm_sha256": member.get("pcm_sha256"),
               "decoded_measurement_reference": {"sample_rate": identity.SAMPLE_RATE,
                                                  "dtype": "float32", "sample_count": int(mono.size),
                                                  "sha256": mono_digest},
               "file_stat_identity_before_decode": list(before_stat) if before_stat is not None else None,
               "file_stat_identity_after_validation": list(after_stat) if after_stat is not None else None,
               "validated": False}
    if (before_stat is None or after_stat is None or before_stat != after_stat
            or current_row is None or current_row.get("pcm_sha256") != member.get("pcm_sha256")):
        binding["state"] = "held_file_or_currentbest_changed"
        return binding
    try:
        observed = audio_identity.audio_identity(path)
        final_stat = _file_stat_identity(path)
        current = _current_best_member(state_root, member, {})
    except (OSError, ValueError, RuntimeError, TypeError):
        binding["state"] = "held_peer_identity_unverified"
        return binding
    binding["original_pcm_recheck"] = {key: observed.get(key) for key in
                                       ("pcm_sha256", "frames", "samplerate", "channels")}
    binding["file_stat_identity_after_pcm_recheck"] = list(final_stat) if final_stat is not None else None
    binding["current_best_membership_rechecked"] = current
    if (final_stat != after_stat or observed.get("pcm_sha256") != member.get("pcm_sha256")
            or current != current_row):
        binding["state"] = "held_file_or_currentbest_changed"
        return binding
    binding["validated"] = True
    binding["state"] = "bound_to_currentbest_original_pcm"
    return binding


def _peer_query_component_measurement(query: np.ndarray, music: np.ndarray, reference: np.ndarray,
                                      interference: dict[str, np.ndarray] | None, match, segment_low: float,
                                      segment_high: float,
                                      source_sha256: str, member: dict, current_row: dict | None,
                                      evidence=None, waveform_binding: dict | None = None) -> dict:
    """Raw full-query-window component metrics for one verified peer alignment; never an authority decision."""
    from lss.bgm import identity, reference_relations

    rate = identity.SAMPLE_RATE
    requested_start, requested_end = float(segment_low), float(segment_high)
    start = max(0.0, requested_start + float(match.query_start_seconds))
    end = min(query.size / rate, requested_start + float(match.query_end_seconds))
    matched_start, matched_end = start, end
    first, last = int(round(requested_start * rate)), int(round(requested_end * rate))
    first, last = max(0, first), min(query.size, last)
    low_cell, high_cell = int(np.floor(requested_start)), int(np.ceil(requested_end))
    cells = list(range(low_cell, high_cell))
    base = {"candidate_kind": "current_best_peer", "candidate_status": "HELD",
            "query_source_sha256": source_sha256,
            "requested_query_window_seconds": [requested_start, requested_end],
            "requested_query_window_samples": [first, last], "query_sample_rate": rate,
            "query_analysis_pcm_sha256": _sample_hash(query, [first, last]),
            "peer_member": {key: member.get(key) for key in ("asset_id", "source_sha256", "source_name", "cue_key",
                                                               "path", "pcm_sha256", "recording_identity",
                                                               "start_seconds", "duration_seconds")},
            "current_best_membership": current_row,
            "peer_waveform_binding": waveform_binding,
            "requested_measurement_scope": {"segment_seconds": [requested_start, requested_end],
                                            "cell_seconds": [low_cell, high_cell],
                                            "unknown_query_edge_intervals_seconds":
                                            ([[float(low_cell), requested_start]] if requested_start > low_cell else [])
                                            + ([[requested_end, float(high_cell)]] if requested_end < high_cell else []),
                                            "edge_cells_without_fractional_isolation":
                                            ([low_cell] if requested_start > low_cell else []) +
                                            ([high_cell - 1] if requested_end < high_cell else [])},
            "matched_alignment": {"query_segment_start_seconds": requested_start,
                                  "matcher_fragment_seconds": [matched_start, matched_end],
                                  "matcher_query_seconds": [float(match.query_start_seconds),
                                                            float(match.query_end_seconds)],
                                  "reference_offset_seconds":
                                  float(match.offset_seconds - segment_low),
                                  "waveform_explained": match.waveform_explained,
                                  "identity_evidence": match.evidence_kind},
            "peer_source_quality_certificate": None,
            "donor_state": "held_peer_source_quality_certificate_missing",
            "authority": {"identity": "none", "availability": "none", "absence": "none", "donor": "none"},
            "consumer": "diagnostic_only_no_cue_or_release_decision"}
    required = ("speech", "effects")
    missing = [name for name in required if interference is None or interference.get(name) is None]
    if current_row is None:
        return {**base, "state": "held_not_current_best_member", "missing_components": []}
    if waveform_binding is None or not waveform_binding.get("validated"):
        return {**base, "state": "held_peer_waveform_identity_unverified", "missing_components": []}
    if missing:
        return {**base, "state": "held_missing_query_component", "missing_components": missing}
    arrays = {"query": np.asarray(query), "music": np.asarray(music),
              **{name: np.asarray(interference[name]) for name in required}}
    if any(value.ndim != 1 or value.size != query.size or not np.isfinite(value).all()
           for value in arrays.values()):
        return {**base, "state": "held_query_component_grid_mismatch", "missing_components": []}
    if not cells:
        return {**base, "state": "held_empty_query_window", "missing_components": []}
    evidence = evidence or reference_relations._Evidence(music, [interference["speech"], interference["effects"]])
    measured = evidence.measure(reference, match.offset_seconds - segment_low, cells)
    read_ranges = _measurement_component_ranges(cells, rate)
    requested_bounds = [int(cells[0] * rate), int(cells[-1] * rate + rate)]
    records = []
    for index, cell in enumerate(cells):
        bounds = measured.reference_ranges[index]
        ranges = read_ranges[index]
        reference_inside = bounds is not None and 0 <= bounds[0] < bounds[1] <= reference.size
        music_inside = 0 <= ranges["music"][0] < ranges["music"][1] <= music.size
        interference_inside = all(0 <= ranges["interference"][0] < ranges["interference"][1] <= arrays[name].size
                                 for name in ("speech", "effects"))
        query_inside = requested_bounds[0] <= first <= last <= requested_bounds[1] and last <= query.size
        valid = reference_inside and music_inside and interference_inside and query_inside
        matched = (cell >= matched_start and cell + 1 <= matched_end)
        values = {"query_cell_seconds": cell,
                  "reference_input_samples_11025": list(bounds) if bounds is not None else None,
                  "query_component_input_samples": {"music": ranges["music"],
                                                     "speech": ranges["interference"],
                                                     "effects": ranges["interference"]},
                  "matched_fragment_covers_cell": bool(matched),
                  "alignment_scope": "matched_fragment" if matched else "extrapolated_for_diagnostic_only",
                  "state": ("measured" if valid else
                            "unknown_reference_fir_outside_peer" if not reference_inside else
                            "unknown_query_component_boundary")}
        for name in ("joint", "novelty", "inside", "absence", "joint_reference", "interference"):
            value = measured.__getattribute__(name)[index]
            values[name] = (bool(valid) if name == "inside" and not valid else
                            bool(value) if name == "inside" else
                            float(value) if valid and np.isfinite(value) else None)
        records.append(values)
    all_ranges = read_ranges
    requested_component_bounds = {
        "query": requested_bounds,
        "music": [min(x["music"][0] for x in all_ranges), max(x["music"][1] for x in all_ranges)],
        "speech": [min(x["interference"][0] for x in all_ranges),
                   max(x["interference"][1] for x in all_ranges)],
        "effects": [min(x["interference"][0] for x in all_ranges),
                    max(x["interference"][1] for x in all_ranges)]}
    component_bounds = {name: [max(0, bounds[0]), min(arrays[name].size, bounds[1])]
                        for name, bounds in requested_component_bounds.items()}
    hashes = {name: _sample_hash(arrays[name], bounds) for name, bounds in component_bounds.items()}
    return {**base, "state": "measured_peer_query_components_held_no_quality_certificate",
            "query_components": ["music", "speech", "effects"], "component_window_pcm_sha256": hashes,
            "component_window_sample_ranges": component_bounds,
            "requested_component_sample_ranges": requested_component_bounds,
            "unknown_cell_count": sum(row["state"] != "measured" for row in records),
            "reference_sample_rate": rate, "cells": records}


def _held_peer_diagnostic_failure(source_sha256: str, member: dict, current_row: dict | None,
                                  match, low: float, high: float, error: Exception) -> dict:
    """A diagnostic-only fault is visible but cannot erase ordinary memory-support records."""
    return {"candidate_kind": "current_best_peer", "candidate_status": "HELD",
            "state": "held_peer_diagnostic_error", "error": type(error).__name__,
            "query_source_sha256": source_sha256,
            "requested_query_window_seconds": [float(low), float(high)],
            "peer_member": {key: member.get(key) for key in
                            ("asset_id", "source_sha256", "source_name", "cue_key", "path", "pcm_sha256")},
            "current_best_membership": current_row,
            "matched_alignment": {"reference_offset_seconds": float(match.offset_seconds - low),
                                  "waveform_explained": match.waveform_explained,
                                  "identity_evidence": match.evidence_kind},
            "peer_source_quality_certificate": None,
            "authority": {"identity": "none", "availability": "none", "absence": "none", "donor": "none"},
            "consumer": "diagnostic_only_no_cue_or_release_decision"}


def _inside_stretches(spans: list[dict], stretches, step: float) -> list[dict]:
    """Spans that begin at or before a materialized stretch they continue into start just inside it."""
    moved = []
    for span in spans:
        for low, high in stretches or ():
            if span["start_seconds"] <= low + 1e-6 < span["end_seconds"] and span["end_seconds"] > low + step:
                moved.append({"before_stretch": [span["start_seconds"], float(low)]})
                span["start_seconds"] = float(low) + step
    return moved


def nominate(query: np.ndarray, music: np.ndarray, segments: list[dict], state_root: Path, source_sha256: str,
             *, loader=None, stretches=None, source_name: str | None = None, stereo=None,
             stereo_loader=None, query_interference: dict[str, np.ndarray] | None = None,
             query_interference_loader=None) -> dict:
    """Component records for `cue_formation.segment(components=...)` from published cues of other sources.

    ``query`` and ``music`` are this episode's source mix and first-pass music stem, mono at
    ``identity.SAMPLE_RATE``; ``segments`` are the first-pass segments (seconds); ``stretches`` the materialized
    candidate stretches ``(low, high)`` seconds; ``source_name`` this source's file stem; ``stereo(start, frames)``
    returns this episode's (source, music) stereo windows at that rate (optional); ``stereo_loader(path, frames)`` a
    member's stereo audio (defaults to `identity.decode_stereo_window`).
    """
    from lss.bgm import cue_memory, identity, recording_support as rs

    rate = identity.SAMPLE_RATE
    receipt = {"policy": revision(), "records": [], "used_members": [], "lookups": [],
               "peer_query_component_measurements": [],
               "identity_authority": "none", "absence_authority": "none", "donor_authority": "none"}
    members = _members(state_root, source_sha256, source_name)
    if not members:
        receipt["state"] = "no_published_cues_of_other_sources"
        return receipt
    loader = loader or (lambda path: identity.decode_mono(Path(path)))
    by_id = {m["asset_id"]: m for m in members}
    items = [(m["asset_id"], m["path"], m["asset_id"],
              cue_memory._fingerprint(state_root, m["pcm_sha256"], None, Path(m["path"]))) for m in members]
    index = identity.ReferenceIndex._from_fingerprints(items, lambda location: loader(location))
    stereo_loader = stereo_loader or (lambda path, frames: identity.decode_stereo_window(Path(path), 0, frames))
    seen: set[tuple[str, int]] = set()
    diagnostic_seen: set[tuple[str, int, int, int]] = set()
    current_best_cache: dict[str, dict | None] = {}
    interference_loaded = query_interference is not None
    interference_cache = query_interference
    component_evidence = None

    def peer_interference() -> dict[str, np.ndarray] | None:
        nonlocal interference_loaded, interference_cache
        if not interference_loaded:
            interference_loaded = True
            try:
                interference_cache = query_interference_loader() if query_interference_loader is not None else None
            except (OSError, ValueError, RuntimeError):
                interference_cache = None
        return interference_cache

    def peer_evidence(interference: dict[str, np.ndarray] | None):
        nonlocal component_evidence
        if component_evidence is None and interference is not None:
            speech, effects = interference.get("speech"), interference.get("effects")
            if (speech is not None and effects is not None and np.asarray(speech).shape == query.shape
                    and np.asarray(effects).shape == query.shape and music.shape == query.shape
                    and np.isfinite(speech).all() and np.isfinite(effects).all() and np.isfinite(music).all()):
                from lss.bgm import reference_relations

                component_evidence = reference_relations._Evidence(music, [speech, effects])
        return component_evidence

    audio: dict[str, np.ndarray] = {}
    audio_bindings: dict[str, tuple[tuple[int, ...] | None, tuple[int, ...] | None]] = {}
    for segment in segments:
        low, high = float(segment["start_seconds"]), float(segment["end_seconds"])
        if high - low < POLICY.minimum_query_seconds:
            continue
        a, b = round(low * rate), min(len(music), round(high * rate))
        if b - a < POLICY.minimum_query_seconds * rate:
            continue
        found = [m for m in identity._collapse_alternatives(identity.match(music[a:b], index, collapse=False))
                 if m.identity_state == "known_exact" and m.waveform_explained is not None
                 and m.waveform_explained >= POLICY.minimum_explained]
        found.sort(key=lambda m: (m.query_end_seconds - m.query_start_seconds, m.waveform_explained), reverse=True)
        receipt["lookups"].append({"segment": [low, high], "verified": len(found)})
        for match in found[:POLICY.maximum_members_per_segment]:
            member = by_id[match.reference_sha256]
            # Member clock = segment clock + offset; the member's first sample in this episode's clock:
            member_start = low - match.offset_seconds
            key = (member["asset_id"], round(member_start))
            if key in seen:
                continue
            if len(audio) >= POLICY.maximum_members and member["asset_id"] not in audio:
                break
            seen.add(key)
            try:
                reference = audio.get(member["asset_id"])
                if reference is None:
                    before_stat = _file_stat_identity(Path(member["path"]))
                    reference = audio[member["asset_id"]] = np.asarray(loader(member["path"]), dtype=np.float32)
                    after_stat = _file_stat_identity(Path(member["path"]))
                    audio_bindings[member["asset_id"]] = (before_stat, after_stat)
            except (OSError, ValueError, RuntimeError):
                continue
            diagnostic_start = max(low, low + float(match.query_start_seconds))
            diagnostic_end = min(high, low + float(match.query_end_seconds))
            diagnostic_key = (member["asset_id"], round(diagnostic_start * rate),
                              round(diagnostic_end * rate), round((match.offset_seconds - low) * rate))
            if diagnostic_key not in diagnostic_seen:
                diagnostic_seen.add(diagnostic_key)
                current_row = None
                try:
                    current_row = _current_best_member(state_root, member, current_best_cache)
                    interference = peer_interference() if current_row is not None else None
                    evidence = peer_evidence(interference) if current_row is not None else None
                    before_stat, after_stat = audio_bindings.get(member["asset_id"], (None, None))
                    binding = (_peer_waveform_binding(Path(member["path"]), member, current_row, reference,
                                                      before_stat, after_stat, state_root)
                               if current_row is not None else None)
                    diagnostic = _peer_query_component_measurement(
                        query, music, reference, interference, match, low, high, source_sha256, member,
                        current_row, evidence, binding)
                except Exception as exc:  # noqa: BLE001 - optional diagnostic cannot erase local records
                    diagnostic = _held_peer_diagnostic_failure(source_sha256, member, current_row, match,
                                                               low, high, exc)
                receipt["peer_query_component_measurements"].append(diagnostic)
            window_low = max(0.0, member_start - POLICY.margin_seconds)
            window_high = min(len(query) / rate, member_start + len(reference) / rate + POLICY.margin_seconds)
            x, y = round(window_low * rate), round(window_high * rate)
            if y - x < 3 * rate:
                continue
            shift = round((window_low - member_start) * rate)   # reference sample = window sample + shift
            try:
                observed = rs.observe(query[x:y], reference, rate, origin=x / rate, offset=shift)
                trace_times, trace_values, trace_control = rs.trace(music[x:y], reference, shift, rate)
            except ValueError:
                continue
            heard = (trace_values >= rs.POLICY.seed_fraction) & (trace_values >= rs.POLICY.control_ratio * trace_control)
            label = f"memory:{member['asset_id']}"
            common = {"reference_sha256": member["recording_identity"], "reference_type": "published_cue_memory",
                      "member": {"asset_id": member["asset_id"], "source_sha256": member["source_sha256"],
                                 "source_name": member.get("source_name"), "pcm_sha256": member["pcm_sha256"],
                                 "start_seconds": member.get("start_seconds"),
                                 "duration_seconds": member["duration_seconds"]},
                      "lookup": {"segment": [low, high], "waveform_explained": match.waveform_explained,
                                 "offset_seconds": match.offset_seconds, "z": match.z},
                      "music_corroborated": True, "music_presence_state": "published_wanted_music_of_member",
                      "availability_authority": "none", "donor_authority": "none"}
            if stereo is not None:
                try:
                    source_window, music_window = stereo(x, y - x)
                    member_stereo = stereo_loader(member["path"], len(reference))
                    for record in rs.observe_stereo(source_window, music_window, member_stereo, rate, origin=x / rate):
                        moved = _inside_stretches(record["spans"], stretches, rs.POLICY.hop_seconds)
                        first = min(s["start_seconds"] for s in record["spans"])
                        receipt["records"].append({**record, **common,
                                                   "onset": {"track": label, "predicted_seconds": first,
                                                             "role": "score"},
                                                   "presence_before_materialized_stretch": moved})
                except (OSError, ValueError, RuntimeError):
                    pass  # optional stereo evidence never removes the mono result
            if (observed["state"] != "component_observed"
                    or heard.sum() * rs.POLICY.hop_seconds < rs.POLICY.minimum_seed_seconds):
                receipt["records"].append({"onset": {"track": label, "predicted_seconds": member_start,
                                                     "role": "score"},
                                           "reference_sha256": member["recording_identity"],
                                           "state": "alignment_not_corroborated", "spans": []})
                continue
            continued_before = _inside_stretches(observed["spans"], stretches, rs.POLICY.hop_seconds)
            observed["render_trace"] = {"origin": x / rate, "times": trace_times.tolist(),
                                        "fraction": trace_values.tolist(), "control": trace_control.tolist()}
            # The memory measured this alignment; the onset it nominates is where the measured presence begins
            # (or the member's own first sample when that lies inside the measured window).
            first = min(s["start_seconds"] for s in observed["spans"])
            onset = max(member_start, first) if member_start <= first + rs.POLICY.search_before_seconds else first
            receipt["records"].append({
                **observed, **common, "onset": {"track": label, "predicted_seconds": onset, "role": "score"},
                "presence_before_materialized_stretch": continued_before})
            receipt["used_members"].append({"asset_id": member["asset_id"], "pcm_sha256": member["pcm_sha256"]})
    receipt["state"] = "checked"
    return receipt


# Edges whose authority is a measured switch between recordings (released-reference nominations); never rejoined.
SWITCH_AUTHORITIES = ("audio_checked_positive_nomination", "two_recording_component_transition",
                      "nominated_head_local_transition")
# Lower rank = better for the listener; a join may not produce a class ranked worse than its best part.
_RANK = {"UNKNOWN_OR_UNRELEASED_WANTED_BGM": 0, "UNCERTAIN": 1}


def join_covered(parts: list[dict], records: list[dict], evidence, recipe, *, additional_music_windows=None,
                 observations: list | None = None, onsets=()) -> list[dict]:
    """Join adjacent first-pass parts that one memory recording, on one clock, plays straight through.

    A join needs: both parts in one materialized stretch; a gap no longer than the formation's continuation limit;
    one measured span of one record (one recording, one offset) reaching at least ``JOIN_FLANK_SECONDS`` into both
    parts; no record of another recording beginning inside the boundary region (an incoming cue). The joined cue is
    kept only when `cue_semantics.decide` judges it no lower than its parts (WANTED if any part was WANTED).
    """
    from lss.bgm import cue_formation, cue_semantics as cs

    spans = [(span["start_seconds"], span["end_seconds"], record["reference_sha256"], record["onset"]["predicted_seconds"])
             for record in records if record.get("state") == "component_observed"
             and record.get("reference_type") == "published_cue_memory" and record.get("music_corroborated")
             for span in record.get("spans", [])]
    result = [dict(part) for part in sorted(parts, key=lambda p: p["start_seconds"])]
    index = 0
    while index + 1 < len(result):
        left, right = result[index], result[index + 1]
        a, b = left["end_seconds"], right["start_seconds"]
        if (left.get("stretch") != right.get("stretch") or b - a > cue_formation.POLICY.continuation_seconds
                or b < a - 1e-6):
            index += 1
            continue
        covering = [identity for low, high, identity, _ in spans
                    if low <= a - JOIN_FLANK_SECONDS and high >= b + JOIN_FLANK_SECONDS]
        incoming = (any(identity not in covering and a - 1.0 <= onset <= b + 1.0 for _, _, identity, onset in spans)
                    or any(a - 1.0 <= row["predicted_seconds"] <= b + 1.0 for row in onsets))
        switch = (left["end_edge"].get("authority") in SWITCH_AUTHORITIES
                  or right["start_edge"].get("authority") in SWITCH_AUTHORITIES)
        if not covering or incoming or switch:
            index += 1
            continue
        joined = {**left, "end_seconds": right["end_seconds"], "end_edge": dict(right["end_edge"]),
                  "core_seconds": [left["start_seconds"], right["end_seconds"]],
                  "candidates": sorted({tuple(x) for x in left.get("candidates", []) + right.get("candidates", [])})}
        unresolved = (left.get("unresolved_recording_regions") or []) + (right.get("unresolved_recording_regions") or [])
        if unresolved:
            joined["unresolved_recording_regions"] = unresolved
        before = cs.decide([left, right], evidence, recipe, source_seconds=float(evidence.count),
                           additional_music_windows=additional_music_windows)
        after = cs.decide([joined], evidence, recipe, source_seconds=float(evidence.count),
                          additional_music_windows=additional_music_windows)[0]
        best = min(_RANK.get(item["semantic_class"], 2) for item in before)
        if _RANK.get(after["semantic_class"], 2) > best:
            if observations is not None:
                observations.append({"state": "memory_join_kept_split_without_lowering", "span": [a, b],
                                     "joined_reasons": after["reasons"], "recording": covering[0]})
            index += 1
            continue
        if observations is not None:
            observations.append({"state": "memory_continuous_recording_joined", "span": [a, b],
                                 "recording": covering[0], "previous_edges": [left["end_edge"], right["start_edge"]],
                                 "identity_authority": "none"})
        result[index:index + 2] = [joined]
    return result


def never_lower(first: list[dict], formed: list[dict], evidence, recipe, *, additional_music_windows=None,
                baseline_music_windows=None,
                observations: list | None = None) -> list[dict]:
    """The memory-supported parts, except where they would lose first-pass WANTED material.

    A part with the same bounds as a first-pass part is that first-pass part (shared component rules also widen
    every part's measured core; memory must not change parts it did not move). Parts are grouped into regions of
    overlapping first-pass and memory parts; a region where some first-pass WANTED part is not inside a WANTED
    memory part returns to its first-pass parts. ``baseline_music_windows`` preserves independently measured
    first-pass music (such as stereo continuity) when judging what memory may not lose.
    """
    from lss.bgm import cue_semantics as cs

    def same(a, b):
        return abs(a["start_seconds"] - b["start_seconds"]) < 1e-6 and abs(a["end_seconds"] - b["end_seconds"]) < 1e-6

    formed = [next((f for f in first if same(f, p)), p) for p in formed]
    decide = lambda items, extra: cs.decide(items, evidence, recipe, source_seconds=float(evidence.count),
                                           additional_music_windows=extra) if items else []
    first_decided = decide(first, baseline_music_windows)
    formed_decided = decide(formed, additional_music_windows)
    items = sorted([("first", i) for i in range(len(first))] + [("formed", i) for i in range(len(formed))],
                   key=lambda item: (first if item[0] == "first" else formed)[item[1]]["start_seconds"])
    regions, end = [], None
    for kind, i in items:
        part = (first if kind == "first" else formed)[i]
        if regions and part["start_seconds"] < end - 1e-6:
            regions[-1].append((kind, i))
            end = max(end, part["end_seconds"])
        else:
            regions.append([(kind, i)])
            end = part["end_seconds"]
    result = []
    for region in regions:
        old = [i for kind, i in region if kind == "first"]
        new = [i for kind, i in region if kind == "formed"]
        wanted_new = [formed_decided[i] for i in new if formed_decided[i]["semantic_class"] == cs.WANTED]
        lost = [first_decided[i] for i in old if first_decided[i]["semantic_class"] == cs.WANTED
                and not any(w["start_seconds"] <= first_decided[i]["start_seconds"] + 1e-6
                            and w["end_seconds"] >= first_decided[i]["end_seconds"] - 1e-6 for w in wanted_new)]
        if lost:
            if observations is not None:
                observations.append({"state": "memory_change_reverted_without_lowering",
                                     "span": [min(first[i]["start_seconds"] for i in old),
                                              max(first[i]["end_seconds"] for i in old)],
                                     "lost_wanted": [[x["start_seconds"], x["end_seconds"]] for x in lost]})
            result.extend(first[i] for i in old)
        else:
            result.extend(formed[i] for i in new)
    return sorted(result, key=lambda p: p["start_seconds"])


def final_check(first: list[dict], parts: list[dict], windows, evidence, recipe, *,
                baseline_music_windows=None,
                observations: list | None = None):
    """(parts, windows) for the final decision, never lower than the first pass.

    The memory's confirmed music windows are kept only inside parts memory moved (unchanged parts are decided on
    their own evidence, as without memory). If the final decision would still lose any first-pass WANTED part, the
    whole first pass is returned with its baseline windows. Independent baseline windows survive the memory
    window scoping; without a baseline, the established no-windows fallback is unchanged.
    """
    import numpy as np

    from lss.bgm import cue_semantics as cs

    def same(a, b):
        return abs(a["start_seconds"] - b["start_seconds"]) < 1e-6 and abs(a["end_seconds"] - b["end_seconds"]) < 1e-6

    if windows is not None:
        windows = np.asarray(windows, dtype=bool).copy()
        keep = np.zeros_like(windows)
        for part in parts:
            if not any(same(part, f) for f in first):
                low, high = int(np.floor(part["start_seconds"])), int(np.ceil(part["end_seconds"]))
                keep[max(0, low):min(len(keep), high)] = True
        windows &= keep
    decide = lambda items, extra: cs.decide(items, evidence, recipe, source_seconds=float(evidence.count),
                                           additional_music_windows=extra) if items else []
    if baseline_music_windows is not None:
        windows = np.asarray(baseline_music_windows, dtype=bool) | (windows if windows is not None else False)
    before = decide(first, baseline_music_windows)
    after = decide(parts, windows)
    wanted = [w for w in after if w["semantic_class"] == cs.WANTED]
    lost = [b for b in before if b["semantic_class"] == cs.WANTED
            and not any(w["start_seconds"] <= b["start_seconds"] + 1e-6 and w["end_seconds"] >= b["end_seconds"] - 1e-6
                        for w in wanted)]
    if lost:
        if observations is not None:
            observations.append({"state": "memory_result_reverted_to_first_pass",
                                 "lost_wanted": [[x["start_seconds"], x["end_seconds"]] for x in lost]})
        return list(first), baseline_music_windows
    return parts, windows
