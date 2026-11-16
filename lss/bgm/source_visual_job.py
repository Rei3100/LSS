"""Attach source-bound visual observations as ancillary evidence to held cues."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from . import frame_text, source_visual


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_identity() -> dict:
    files = [Path(__file__), Path(source_visual.__file__), Path(frame_text.__file__), frame_text._SCRIPT]
    hashes = {}
    for path in files:
        try:
            hashes[path.name] = _sha256(path)
        except OSError:
            hashes[path.name] = None
    return {"policy_id": "held-program-insert-source-visual-v1", "files_sha256": hashes}


def _unavailable(summary: dict, reason: str, exc: Exception | None = None) -> dict:
    summary.update(status="UNAVAILABLE", reason=reason)
    if exc is not None:
        summary["error_type"] = type(exc).__name__
    return summary


def observe(source, source_sha256: str, timeline: dict | None, decided: list[dict], run_dir) -> dict:
    """Observe only unresolved held program-insert regions; never alter cue decisions."""
    policy = _policy_identity()
    summary = {"status": "NOT_NEEDED", "policy": policy, "regions": []}
    if not isinstance(decided, list):
        return _unavailable(summary, "invalid_decided_segments")
    targets = [item for item in decided if (item.get("scope") or {}).get("disposition") == "HELD" and
               (item.get("scope") or {}).get("reason") == "program_insert_role_unresolved"]
    if not targets:
        return summary
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256.lower()):
        raise ValueError("source_sha256 must be SHA-256 hex")
    if not isinstance(timeline, dict) or timeline.get("method") != "first_decoded_audio_pts":
        return _unavailable(summary, "audio_timeline_origin_unavailable")
    offset = timeline.get("offset_seconds")
    if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(offset):
        return _unavailable(summary, "audio_timeline_origin_unavailable")
    numerator, denominator = timeline.get("offset_numerator"), timeline.get("offset_denominator")
    if numerator is not None or denominator is not None:
        if (not isinstance(numerator, int) or isinstance(numerator, bool) or
                not isinstance(denominator, int) or isinstance(denominator, bool) or denominator == 0 or
                not math.isclose(numerator / denominator, offset, rel_tol=0, abs_tol=1e-12)):
            return _unavailable(summary, "audio_timeline_origin_unavailable")
    if denominator is not None and denominator < 0:
        return _unavailable(summary, "audio_timeline_origin_unavailable")
    regions = []
    target_pairs = []
    for index, item in enumerate(decided):
        scope = item.get("scope") or {}
        if scope.get("disposition") != "HELD" or scope.get("reason") != "program_insert_role_unresolved":
            continue
        identifier = scope.get("candidate_id") or item.get("cue_id") or f"segment-{index:03d}"
        start, end = item.get("start_seconds"), item.get("end_seconds")
        if (not isinstance(start, (int, float)) or isinstance(start, bool) or not math.isfinite(start) or
                not isinstance(end, (int, float)) or isinstance(end, bool) or not math.isfinite(end) or
                start < 0 or end <= start):
            return _unavailable(summary, "invalid_target_region")
        regions.append({"id": str(identifier), "start_seconds": float(start), "end_seconds": float(end)})
        target_pairs.append((item, str(identifier)))
    if len({region["id"] for region in regions}) != len(regions):
        return _unavailable(summary, "duplicate_target_region_id")
    if not regions:
        return summary
    work_dir = Path(os.path.abspath(os.fspath(run_dir))) / f"sv-{uuid.uuid4().hex[:12]}"
    try:
        report = source_visual.observe_regions(source, source_sha256, regions, float(offset), work_dir)
    except source_visual.SourceVisualIntegrityError:
        raise
    except Exception as exc:  # ancillary availability cannot fail audio processing
        return _unavailable(summary, "source_visual_runtime_failure", exc)
    if not isinstance(report, dict):
        return _unavailable(summary, "malformed_source_visual_report")

    reported_source = report.get("source_sha256")
    if reported_source is not None and (not isinstance(reported_source, str) or
                                        reported_source.lower() != source_sha256.lower()):
        raise source_visual.SourceVisualIntegrityError("source visual report was bound to a different source")
    after_source = report.get("source_sha256_after")
    if after_source is not None and (not isinstance(after_source, str) or
                                     after_source.lower() != source_sha256.lower()):
        raise source_visual.SourceVisualIntegrityError("source visual report source changed during observation")
    if report.get("source_unchanged") is False:
        raise source_visual.SourceVisualIntegrityError("source visual report says source changed during observation")
    reported_offset = report.get("audio_pts_origin_seconds")
    if reported_offset is not None and (not isinstance(reported_offset, (int, float)) or
                                        isinstance(reported_offset, bool) or reported_offset != float(offset)):
        raise source_visual.SourceVisualIntegrityError("source visual report used a different audio timeline origin")
    if (reported_source is None or after_source is None or report.get("source_unchanged") is not True or
            reported_offset is None):
        return _unavailable(summary, "malformed_source_visual_binding")
    if report.get("status") not in ("OK", "UNAVAILABLE"):
        return _unavailable(summary, "malformed_source_visual_status")
    reported_regions = report.get("regions")
    if not isinstance(reported_regions, list) or not all(isinstance(region, dict) for region in reported_regions):
        return _unavailable(summary, "malformed_source_visual_regions")
    if any(not isinstance(region.get("id"), str) for region in reported_regions):
        return _unavailable(summary, "malformed_source_visual_regions")
    expected_by_id = {region["id"]: region for region in regions}
    returned_by_id = {region["id"]: region for region in reported_regions}
    if len(returned_by_id) != len(reported_regions):
        raise source_visual.SourceVisualIntegrityError("source visual report repeated a region ID")
    if set(returned_by_id) != set(expected_by_id):
        raise source_visual.SourceVisualIntegrityError("source visual report region IDs did not match requested regions")
    for region_id, expected in expected_by_id.items():
        returned = returned_by_id[region_id]
        if (returned.get("start_seconds") != expected["start_seconds"] or
                returned.get("end_seconds") != expected["end_seconds"]):
            raise source_visual.SourceVisualIntegrityError("source visual report region bounds did not match request")
        if returned.get("status") not in ("OK", "UNAVAILABLE"):
            return _unavailable(summary, "malformed_source_visual_region")

    report_path = work_dir / "report.json"
    bound_report = {"policy": policy, "observation": report}
    try:
        report_path.write_text(json.dumps(bound_report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        report_sha256 = _sha256(report_path)
    except (OSError, TypeError, ValueError) as exc:
        return _unavailable(summary, "source_visual_report_save_failure", exc)
    for item, region_id in target_pairs:
        item["source_visual_evidence"] = {"report_path": str(report_path), "report_sha256": report_sha256,
                                         "region_id": region_id, "coverage": "sampled_only",
                                         "role_authority": "none"}
        summary["regions"].append({"region_id": region_id, "status": returned_by_id[region_id].get("status")})
    summary.update(status=report.get("status", "UNAVAILABLE"), report_path=str(report_path),
                   report_sha256=report_sha256, coverage="sampled_only", role_authority="none")
    return summary
