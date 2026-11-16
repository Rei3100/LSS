"""Veto mono reference authority when the saved stereo Music has unexplained material.

This observes the existing fingerprint grid and never renders audio or grants a relation.
Both reference channels jointly explain each Music channel; channel swap, gain and short
linear EQ remain explainable. A separate side residual prevents strong mid from hiding
anti-phase material. Nonmusic stems cannot explain away wanted stereo material here.
"""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from lss.bgm import identity

METHOD = "joint-lr-fir-side-residual-v2"
TAPS = 17
WINDOW_SECONDS = .5
MIN_MEAN_POWER = 1e-9
MIN_FIT_FRAMES = 4 * TAPS


def policy() -> dict:
    return {"method": METHOD, "taps": TAPS, "window_seconds": WINDOW_SECONDS,
            "minimum_mean_power": MIN_MEAN_POWER,
            "minimum_fit_frames": MIN_FIT_FRAMES, "tail_window": "overlap_previous_instead_of_short_fit",
            "residual_limit": "relation_recipe.exact_max_local_clean_residual",
            "scope": "whole_span_for_release_otherwise_final_donor_ranges",
            "authority": "veto_only", "input": "saved_stereo_music"}


def _path(library, sha: str) -> Path:
    reference = library.index.references[library.owner(sha)]
    path = Path(reference.path)
    if not path.is_file():
        path = Path(library.index.loader.path(sha))
    if identity._sha256(path) != sha:
        raise ValueError("Stereo reference bytes differ from the registered recording")
    return path


def _pieces(relation: dict, start: float, end: float) -> list[tuple]:
    reference = relation.get("reference") or {}
    if relation["state"] == "exact_released_recording":
        return [(start, end, reference["reference_sha256"], reference["offset_seconds"])]
    # An edit's offset changes. Do not silently reuse its main offset across other pieces.
    rows = sorted((max(start, r["start_seconds"]), min(end, r["end_seconds"]),
                   r["reference_sha256"], r["offset_seconds"])
                  for r in relation.get("regions") or []
                  if r.get("reference_sha256") and r.get("offset_seconds") is not None
                  and r["end_seconds"] > start and r["start_seconds"] < end)
    cursor = start
    for first, last, _, _ in rows:
        if first > cursor + 1e-6:
            raise ValueError("Stereo reference accounting has an unmeasured gap")
        cursor = max(cursor, last)
    if cursor < end - 1e-6:
        raise ValueError("Stereo reference accounting does not cover the cue")
    return rows


def _measure(query: np.ndarray, reference: np.ndarray, first_frame: int) -> dict:
    if query.shape != reference.shape or query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("Stereo reference observation has a short or changed timeline")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("Stereo reference observation is not finite")
    if len(query) < MIN_FIT_FRAMES:
        raise ValueError("Stereo observation is too short for a constrained reference fit")
    half = TAPS // 2
    padded = np.pad(reference.astype(np.float64), ((half, half), (0, 0)))
    views = [sliding_window_view(padded[:, channel], TAPS) for channel in (0, 1)]
    step = round(WINDOW_SECONDS * identity.SAMPLE_RATE)
    result = {"windows": 0, "max_residual": 0., "max_side_residual": 0., "worst_window_seconds": None}
    for first in range(0, len(query), step):
        if len(query) - first < step:
            first = max(0, len(query) - step)
        last = min(first + step, len(query))
        y = query[first:last].astype(np.float64)
        energy = float(np.sum(y * y))
        if energy < MIN_MEAN_POWER * y.size:
            continue
        design = np.hstack([view[first:last] for view in views])
        gram = design.T @ design
        gram[np.diag_indices_from(gram)] += 1e-9 * (np.trace(gram) / gram.shape[0] + 1e-12)
        predicted = design @ np.linalg.solve(gram, design.T @ y)
        residual = y - predicted
        ratio = float(np.sum(residual * residual) / energy)
        side = (y[:, 0] - y[:, 1]) * .5
        side_residual = (residual[:, 0] - residual[:, 1]) * .5
        side_energy = float(side @ side)
        side_ratio = (float(side_residual @ side_residual) / side_energy
                      if side_energy >= MIN_MEAN_POWER * len(side) else 0.)
        result["windows"] += 1
        if max(ratio, side_ratio) > max(result["max_residual"], result["max_side_residual"]):
            result["worst_window_seconds"] = [(first_frame + first) / identity.SAMPLE_RATE,
                                                (first_frame + last) / identity.SAMPLE_RATE]
        result["max_residual"] = max(result["max_residual"], ratio)
        result["max_side_residual"] = max(result["max_side_residual"], side_ratio)
    return result


def guard(segments: list[dict], *, music: Path | None, library, recipe) -> list[dict]:
    from lss.bgm import reference_reconstruction

    result = []
    for item in segments:
        relation = item.get("reference_relation") or {}
        if relation.get("state") not in ("exact_released_recording", "released_recording_edit",
                                          "released_recording_component_variant"):
            result.append(item)
            continue
        observed = {"method": METHOD, "state": "verified", "pieces": [],
                    "limit": recipe.exact_max_local_clean_residual}
        released = (relation["state"] == "exact_released_recording"
                    or (relation["state"] == "released_recording_edit"
                        and bool((relation.get("released_cover") or {}).get("complete"))))
        if released:
            observed["scope"] = "whole_span"
        else:
            planned = reference_reconstruction.plan(relation, item["start_seconds"], item["end_seconds"])
            if not planned["eligible"]:
                # A descriptive relation grants neither suppression nor replacement authority.
                # Do not discard its partial identity because unrelated Music is unexplained.
                observed.update(state="not_required_no_authority", scope="none")
                result.append({**item, "reference_relation": {**relation, "stereo_accounting": observed}})
                continue
            observed["scope"] = "final_donor_ranges"
        try:
            if music is None:
                raise ValueError("Stereo Music is unavailable")
            music = Path(music)
            music_sha = identity._sha256(music)
            observed["music_sha256"] = music_sha
            pieces = (_pieces(relation, item["start_seconds"], item["end_seconds"]) if released else
                      [(d["start_seconds"], d["end_seconds"], d["reference_sha256"], d["offset_seconds"])
                       for d in planned["donors"]])
            for first, last, sha, offset in pieces:
                a, b = round(first * identity.SAMPLE_RATE), round(last * identity.SAMPLE_RATE)
                ref_start = a + round(offset * identity.SAMPLE_RATE)
                if b <= a or ref_start < 0:
                    raise ValueError("Stereo reference range is outside its clock")
                path = _path(library, sha)
                query = identity.decode_stereo_window(music, a, b - a)
                reference = identity.decode_stereo_window(path, ref_start, b - a)
                if len(query) != b - a or len(reference) != b - a:
                    raise ValueError("Stereo reference decode is incomplete")
                row = _measure(query, reference, a)
                row.update(span_seconds=[first, last], reference_sha256=sha, offset_seconds=offset)
                observed["pieces"].append(row)
                if not row["windows"]:
                    raise ValueError("Stereo reference has no measured music windows")
                if max(row["max_residual"], row["max_side_residual"]) > observed["limit"]:
                    observed["state"] = "unexplained_stereo_material"
                if identity._sha256(path) != sha:
                    raise ValueError("Stereo reference changed during verification")
            if not observed["pieces"] or identity._sha256(music) != music_sha:
                raise ValueError("Stereo Music changed or has no scoped accounting")
        except (OSError, ValueError, AttributeError, KeyError, np.linalg.LinAlgError, identity.DecodeError) as exc:
            observed.update(state="unverified", error=type(exc).__name__)
        updated = {**relation, "stereo_accounting": observed}
        if observed["state"] != "verified":
            updated.update(state="unknown", basis="stereo_reference_accounting_not_established",
                           mono_state=relation["state"], donor_reconstructable=False,
                           authority_hold_reasons=[*(relation.get("authority_hold_reasons") or []),
                                                   "stereo_reference_accounting_not_established"])
            if relation.get("released_cover"):
                updated["released_cover"] = {**relation["released_cover"], "complete": False}
        result.append({**item, "reference_relation": updated})
    return result
