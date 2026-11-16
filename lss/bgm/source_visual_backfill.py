"""Independent ancillary cache for normal START, never an accepted-run rewrite."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

from . import source_visual_job as job
from .source_visual import SourceVisualIntegrityError
from ..core import session

SCHEMA_VERSION = 1
RETRY_SECONDS = 24 * 60 * 60
SIDECAR = "source-visual-backfill.json"


def _targets(segments: list[dict]) -> list[dict]:
    return [{"id": str((item.get("scope") or {}).get("candidate_id") or item.get("cue_id") or
                       f"segment-{index:03d}"),
             "start_seconds": item.get("start_seconds"), "end_seconds": item.get("end_seconds")}
            for index, item in enumerate(segments)
            if (item.get("scope") or {}).get("disposition") == "HELD" and
            (item.get("scope") or {}).get("reason") == "program_insert_role_unresolved"]


def _valid_report(summary: dict, binding: dict, run_dir: Path) -> bool:
    """A checksum alone cannot attest the source, timeline or requested population."""
    try:
        path = Path(summary["report_path"]).resolve()
        if not path.is_relative_to(run_dir.resolve()) or job._sha256(path) != summary["report_sha256"]:
            return False
        raw = json.loads(path.read_text(encoding="utf-8"))
        report = raw["observation"]
        timeline = binding["timeline"]
        if (raw["policy"] != binding["policy"] or not isinstance(timeline, dict) or
                timeline.get("method") != "first_decoded_audio_pts" or
                report.get("source_sha256") != binding["source_sha256"] or
                report.get("source_sha256_after") != binding["source_sha256"] or
                report.get("source_unchanged") is not True or
                report.get("audio_pts_origin_seconds") != timeline.get("offset_seconds") or
                report.get("status") != summary.get("status") or
                report.get("status") not in ("OK", "UNAVAILABLE")):
            return False
        regions = report["regions"]
        return (isinstance(regions, list) and
                [{name: row[name] for name in ("id", "start_seconds", "end_seconds")}
                 for row in regions] == binding["targets"] and
                all(row.get("status") in ("OK", "UNAVAILABLE") for row in regions))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _reusable(record: dict, binding: dict, run_dir: Path, now: float) -> bool:
    if record.get("binding") != binding:
        return False
    summary = record.get("summary") or {}
    if summary.get("policy") != binding["policy"]:
        return False
    if summary.get("status") == "UNAVAILABLE":
        created = record.get("created_at")
        if not isinstance(created, (int, float)) or not 0 <= now - created < RETRY_SECONDS:
            return False
        # Typed failures before a report exists are deliberately negative-cached.
        if not summary.get("report_path"):
            return isinstance(summary.get("reason"), str) and bool(summary["reason"])
    return _valid_report(summary, binding, run_dir)


def ensure(source, source_sha256: str, run: dict, run_path: Path) -> dict:
    """Called only by START under the run lock; outcome/readback never calls this.

    Ordinary ancillary failures retain usable audio. Integrity failures propagate.
    All accepted records, labels, PCM and CurrentBest remain untouched.
    """
    try:
        result = _ensure(source, source_sha256, run, Path(run_path))
    except SourceVisualIntegrityError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result = {"status": "UNAVAILABLE", "reason": "ancillary_cache_unavailable",
                  "error_type": type(exc).__name__}
    if job._sha256(Path(source)) != source_sha256:
        raise session.InputChangedError("BGM ancillary observation source changed")
    return result


def _ensure(source, source_sha256: str, run: dict, run_path: Path) -> dict:
    directory = run_path.parent
    reference = (run.get("semantics") or {}).get("record") or {}
    folder = (directory / "semantics").resolve()
    path = (folder / str(reference.get("name", ""))).resolve()
    if not reference or path.parent != folder or job._sha256(path) != reference.get("sha256"):
        return {"status": "UNAVAILABLE", "reason": "accepted_semantics_unavailable"}
    semantics = json.loads(path.read_text(encoding="utf-8"))
    segments = semantics["segments"]
    targets = _targets(segments)
    if not targets:
        return {"status": "NOT_NEEDED"}
    binding = {"schema_version": SCHEMA_VERSION, "policy": job._policy_identity(),
               "source_sha256": source_sha256,
               "timeline": (run.get("source") or {}).get("audio_timeline_origin"), "targets": targets}
    now = time.time()
    sidecar = directory / SIDECAR
    try:
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
        if _reusable(cached, binding, directory, now):
            return cached["summary"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    summary = semantics.get("source_visual") or {}
    seed = (summary.get("status") == "OK" or not sidecar.exists())
    if not seed or summary.get("policy") != binding["policy"] or not _valid_report(summary, binding, directory):
        summary = job.observe(source, source_sha256, binding["timeline"], deepcopy(segments), directory)
    # Even provider unavailability must not conceal mutation during START.
    if job._sha256(Path(source)) != source_sha256:
        raise session.InputChangedError("BGM ancillary observation source changed")
    session._write_json_atomic(sidecar, {"binding": binding, "created_at": now, "summary": summary})
    return summary
