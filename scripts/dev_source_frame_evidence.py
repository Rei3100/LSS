"""Create bounded, hash-bound source-video frame evidence for Development use."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root



SHOWINFO_PTS = re.compile(r"showinfo.*?\bpts_time:([^\s]+)")
STATE_ROOT = dev_state_root()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_hidden(command: list[str]) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, object] = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace"}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return subprocess.run(command, **kwargs)  # type: ignore[arg-type]


def _parse_sha(value: str, flag: str) -> str:
    value = value.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{flag} must be a 64-character SHA-256 hex digest")
    return value


def extract(source_text: str, expected_source_hash: str, ffmpeg_text: str,
            timestamps: list[float], output_text: str) -> Path:
    source = Path(source_text)  # Intentionally never resolve: retain caller's C: alias.
    ffmpeg = Path(ffmpeg_text)
    output = Path(output_text)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if STATE_ROOT.resolve() not in output.resolve().parents:
        raise ValueError("--output must stay beneath the development state root")
    if not source.is_absolute() or source.drive.upper() != "C:":
        raise ValueError("--source must be an absolute C: alias path")
    if not source.is_file():
        raise ValueError("--source must name an existing file")
    if not ffmpeg.is_file():
        raise ValueError("--ffmpeg must name an existing ffmpeg executable")
    expected_source_hash = _parse_sha(expected_source_hash, "--expected-source-sha256")
    if not timestamps or any(not math.isfinite(t) or t < 0 for t in timestamps):
        raise ValueError("provide one or more non-negative --timestamp values")
    if sha256_file(source) != expected_source_hash:
        raise ValueError("source SHA-256 does not match --expected-source-sha256")

    output.mkdir(parents=True, exist_ok=False)
    source_hash_before = sha256_file(source)
    ffmpeg_hash = sha256_file(ffmpeg)
    records: list[dict[str, object]] = []
    try:
        for index, timestamp in enumerate(timestamps, 1):
            pattern = output / f"frame-{index}-%d.png"
            command = [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "info",
                       "-threads", "1", "-hwaccel", "none", "-accurate_seek", "-ss", f"{timestamp:.6f}",
                       "-copyts", "-i", str(source), "-threads", "1", "-filter_threads", "1",
                       "-map", "0:v:0", "-vf", "showinfo",
                       "-frames:v", "1", "-fps_mode", "passthrough", "-enc_time_base", "1:1000000",
                       "-frame_pts", "1", str(pattern)]
            result = run_hidden(command)
            candidates = list(output.glob(f"frame-{index}-*.png"))
            points = [float(v) for v in SHOWINFO_PTS.findall(result.stderr)]
            if result.returncode != 0 or len(candidates) != 1:
                raise RuntimeError(f"ffmpeg frame extraction failed for timestamp {timestamp}: {result.stderr[-2000:]}")
            image = candidates[0]
            match = re.fullmatch(rf"frame-{index}-(\d+)\.png", image.name)
            saved_pts_us = int(match.group(1)) if match else None
            records.append({"requested_video_pts_seconds": timestamp,
                            "file": image.name,
                            "sha256": sha256_file(image),
                            "size_bytes": image.stat().st_size,
                            "filename_pts_microseconds": saved_pts_us,
                            "filename_pts_identifies_saved_frame": saved_pts_us is not None,
                            "showinfo_candidate_pts_seconds": points,
                            "returncode": result.returncode})
        source_hash_after = sha256_file(source)
        manifest = {
            "format": "lss-dev-source-frame-evidence-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Development-only visual observation; not Product input, scene inference, or exact audio/video alignment",
            "source_argument": source_text,
            "source_sha256_before": source_hash_before,
            "source_sha256_after": source_hash_after,
            "source_unchanged": source_hash_before == source_hash_after == expected_source_hash,
            "ffmpeg_argument": ffmpeg_text,
            "ffmpeg_sha256": ffmpeg_hash,
            "decode": {"threads": 1, "hwaccel": "none", "input_seek": "-ss before -i with -accurate_seek and -copyts",
                       "video_stream": "0:v:0", "output": "PNG image2 with -frame_pts 1 and encoder time base 1:1000000",
                       "pts_claim": "Integer filename PTS is the muxed PNG packet PTS; candidate showinfo values are retained separately."},
            "frames": records,
        }
        if not manifest["source_unchanged"]:
            raise RuntimeError("source changed during extraction")
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path
    except Exception:
        # Preserve partial evidence for diagnosis; never overwrite or silently clean it.
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="absolute C: alias to the finished source video")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--ffmpeg", required=True, help="path to ffmpeg.exe")
    parser.add_argument("--timestamp", action="append", type=float, required=True, help="source-video PTS seconds; repeatable")
    parser.add_argument("--output", required=True, help="new output directory")
    args = parser.parse_args(argv)
    try:
        manifest = extract(args.source, args.expected_source_sha256, args.ffmpeg, args.timestamp, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"source-frame evidence failed: {exc}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
