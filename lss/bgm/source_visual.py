"""Optional, source-bound still-frame OCR evidence.

This module records pixels and uncertain OCR only. It does not infer scene,
cast, role, title, publication, or identity.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
from numbers import Real

from . import frame_text
from ..paths import resource_root

_FRAME_PTS = re.compile(r"^frame-(-?\d+)\.png$")
_TIMEOUT = 30


class SourceVisualIntegrityError(RuntimeError):
    """Source or decoder executable changed during observation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    flags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) |
             getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False,
                          creationflags=flags)


def _regions(value: object) -> list[dict]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("regions must be a nonempty sequence")
    result = []
    ids = set()
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("each region needs a nonempty string id")
        start, end = item.get("start_seconds"), item.get("end_seconds")
        if not _finite(start) or not _finite(end) or start < 0 or end <= start:
            raise ValueError("region bounds must be finite, positive-duration seconds")
        if item["id"] in ids:
            raise ValueError("region ids must be unique")
        ids.add(item["id"])
        result.append({"id": item["id"], "start_seconds": float(start),
                       "end_seconds": float(end)})
    return result


def observe_regions(source, expected_source_sha256, regions,
                    audio_pts_origin_seconds, work_dir, ffmpeg=None):
    """Sample 3 interior frames per region, bind OCR records to source bytes.

    ``work_dir`` must not exist. The caller owns the resulting directory.
    Audio origin is an explicit known offset in video PTS seconds.
    """
    source_path = Path(os.fspath(source))  # retain lexical spelling (including aliases)
    expected = str(expected_source_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected_source_sha256 must be SHA-256 hex")
    if not _finite(audio_pts_origin_seconds):
        raise ValueError("audio_pts_origin_seconds must be an explicit finite offset")
    region_list = _regions(regions)
    output = Path(work_dir)
    output.mkdir(parents=True, exist_ok=False)
    executable = Path(ffmpeg) if ffmpeg is not None else resource_root() / "bin" / "ffmpeg.exe"
    try:
        source_before = _sha256(source_path)
        if source_before != expected:
            raise SourceVisualIntegrityError("source did not match expected SHA-256")
        ffmpeg_available = executable.is_file()
        ffmpeg_before = _sha256(executable) if ffmpeg_available else None
    except OSError as exc:
        raise SourceVisualIntegrityError("source or ffmpeg executable unavailable") from exc

    report = {"status": "OK", "evidence_kind": "raw_pixel_and_uncertain_ocr",
              "absence_inference": "not_authorized", "role_authority": "none",
              "source_sha256": source_before,
              "source_path_lexical": os.fspath(source), "audio_pts_origin_seconds": float(audio_pts_origin_seconds),
              "ffmpeg_path": os.fspath(executable), "ffmpeg_sha256": ffmpeg_before,
              "regions": []}
    try:
        for region in region_list:
            region_record = {**region, "status": "OK", "frames": []}
            for fraction in (1 / 6, 1 / 2, 5 / 6):
                requested = (float(audio_pts_origin_seconds) + region["start_seconds"] +
                             (region["end_seconds"] - region["start_seconds"]) * fraction)
                frame_dir = output / f"frame-{len(report['regions'])}-{len(region_record['frames'])}"
                frame_dir.mkdir()
                frame_path = frame_dir / "frame.png"
                args = [os.fspath(executable), "-nostdin", "-v", "error", "-threads", "1",
                        "-filter_threads", "1", "-hwaccel", "none", "-accurate_seek",
                        "-ss", f"{requested:.6f}", "-copyts", "-i", os.fspath(source_path),
                        "-map", "0:v:0", "-an", "-frames:v", "1", "-fps_mode", "passthrough", "-threads", "1",
                        "-enc_time_base", "1:1000000", "-frame_pts", "1", "-y",
                        os.fspath(frame_path.parent / "frame-%d.png")]
                frame = {"requested_video_pts_seconds": requested, "fraction": fraction,
                         "ffmpeg_argv": args}
                if not ffmpeg_available:
                    frame.update(status="UNAVAILABLE", reason="ffmpeg_unavailable")
                    region_record["frames"].append(frame)
                    continue
                try:
                    completed = _run(args, _TIMEOUT)
                    frame["ffmpeg_returncode"] = completed.returncode
                    frame["ffmpeg_stdout"] = completed.stdout[-1000:]
                    frame["ffmpeg_stderr"] = completed.stderr[-2000:]
                    matches = list(frame_path.parent.glob("frame-*.png"))
                    actual_path = matches[0] if len(matches) == 1 else None
                    if completed.returncode != 0 or actual_path is None:
                        frame.update(status="UNAVAILABLE", reason="frame_decode_failed")
                    else:
                        match = _FRAME_PTS.fullmatch(actual_path.name)
                        if not match:
                            frame.update(status="UNAVAILABLE", reason="invalid_frame_pts_name")
                        else:
                            actual_pts = int(match.group(1)) / 1_000_000
                            filename_pts = int(match.group(1))
                            frame.update(filename_pts_microseconds=filename_pts,
                                         actual_video_pts_seconds=actual_pts,
                                         frame_path=str(actual_path), frame_sha256=_sha256(actual_path))
                            if not (region["start_seconds"] + float(audio_pts_origin_seconds) <= actual_pts <
                                    region["end_seconds"] + float(audio_pts_origin_seconds)):
                                frame.update(status="UNAVAILABLE", reason="frame_outside_region")
                            else:
                                raw = frame_text.observe(actual_path, frame["frame_sha256"])
                                frame["ocr"] = raw
                                if raw.get("status") == "OK":
                                    frame["text_state"] = ("empty_not_absence" if not raw.get("recognized_text", "")
                                                           else "text_observed")
                                frame.update(status=raw.get("status", "UNAVAILABLE"),
                                             reason=raw.get("reason"))
                except (OSError, subprocess.TimeoutExpired) as exc:
                    frame.update(status="UNAVAILABLE", reason=type(exc).__name__)
                except frame_text.FrameTextIntegrityError:
                    raise
                except Exception as exc:
                    frame.update(status="UNAVAILABLE", reason="ocr_runtime_failure",
                                 error_type=type(exc).__name__)
                region_record["frames"].append(frame)
            if all(f["status"] == "UNAVAILABLE" for f in region_record["frames"]):
                region_record["status"] = "UNAVAILABLE"
                region_record["reason"] = "no_usable_frame_observations"
            report["regions"].append(region_record)
    except frame_text.FrameTextIntegrityError as exc:
        raise SourceVisualIntegrityError(str(exc)) from exc
    finally:
        try:
            source_after = _sha256(source_path)
            ffmpeg_exists_after = executable.is_file()
            ffmpeg_after = _sha256(executable) if ffmpeg_exists_after else None
        except OSError as exc:
            raise SourceVisualIntegrityError("source or ffmpeg executable became unavailable") from exc
        if (source_after != source_before or ffmpeg_after != ffmpeg_before or
                ffmpeg_exists_after != (ffmpeg_before is not None)):
            raise SourceVisualIntegrityError("source or ffmpeg executable changed during observation")
    if report["regions"] and all(region["status"] == "UNAVAILABLE" for region in report["regions"]):
        report["status"] = "UNAVAILABLE"
        report["reason"] = "no_usable_region_observations"
    report.update(source_sha256_after=source_after, source_unchanged=True,
                  ffmpeg_sha256_after=ffmpeg_after, coverage="sampled_instants_only")
    return report
