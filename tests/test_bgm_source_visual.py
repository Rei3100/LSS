import hashlib
import subprocess
from pathlib import Path

import pytest

from lss.bgm import frame_text, source_visual as sv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"finished stereo source")
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"decoder")
    monkeypatch.setattr(sv, "_run", lambda args, timeout: None)
    return source, ffmpeg


def run_with_frames(monkeypatch, source, ffmpeg, *, ocr=None, mutate=None):
    calls = []

    def runner(args, timeout):
        calls.append(args)
        target = args[-1]
        requested = float(args[args.index("-ss") + 1])
        output = Path(target.replace("%d", str(round(requested * 1_000_000))))
        output.write_bytes(b"png")
        if mutate:
            mutate()
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(sv, "_run", runner)
    monkeypatch.setattr(frame_text, "observe", ocr or
                        (lambda path, expected: {"status": "UNAVAILABLE", "reason": "no_text"}))
    return calls


def test_maps_audio_origin_and_uses_saved_pts_filename(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    calls = run_with_frames(monkeypatch, source, ffmpeg)
    out = tmp_path / "fresh"
    result = sv.observe_regions(source, digest(source),
                               [{"id": "opening", "start_seconds": 10, "end_seconds": 16}],
                               2.0, out, ffmpeg)
    frames = result["regions"][0]["frames"]
    assert [f["requested_video_pts_seconds"] for f in frames] == [13, 15, 17]
    assert [f["actual_video_pts_seconds"] for f in frames] == [13, 15, 17]
    assert [f["filename_pts_microseconds"] for f in frames] == [13_000_000, 15_000_000, 17_000_000]
    assert len(calls) == 3
    assert all("-hwaccel" in args and "none" in args and "-copyts" in args for args in calls)
    assert all(args[args.index("-fps_mode") + 1] == "passthrough" for args in calls)
    assert result["evidence_kind"] == "raw_pixel_and_uncertain_ocr"
    assert result["absence_inference"] == "not_authorized" and result["role_authority"] == "none"
    assert result["status"] == "UNAVAILABLE"
    assert all(f["status"] == "UNAVAILABLE" and f["reason"] == "no_text" for f in frames)


def test_existing_output_is_never_reused(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    out = tmp_path / "already-owned"
    out.mkdir()
    with pytest.raises(FileExistsError):
        sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 1,
                              "end_seconds": 2}], 0, out, ffmpeg)


def test_outside_pts_is_unavailable_without_ocr(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    calls = run_with_frames(monkeypatch, source, ffmpeg,
                            ocr=lambda *args: pytest.fail("outside frame must not reach OCR"))
    def place_early(args, timeout):
        output = Path(args[-1].replace("%d", "999999"))
        output.write_bytes(b"png")
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(sv, "_run", place_early)
    result = sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 10,
                              "end_seconds": 12}], 2, tmp_path / "out", ffmpeg)
    assert result["regions"][0]["status"] == "UNAVAILABLE"
    assert all(frame["reason"] == "frame_outside_region" for frame in result["regions"][0]["frames"])


def test_exact_end_pts_is_outside_half_open_region(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    run_with_frames(monkeypatch, source, ffmpeg,
                    ocr=lambda *args: pytest.fail("end boundary must not reach OCR"))
    def place_at_end(args, timeout):
        Path(args[-1].replace("%d", "2000000")).write_bytes(b"png")
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(sv, "_run", place_at_end)
    result = sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 1,
                              "end_seconds": 2}], 0, tmp_path / "out", ffmpeg)
    assert all(frame["reason"] == "frame_outside_region" for frame in result["regions"][0]["frames"])


def test_empty_successful_ocr_is_not_absence(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    run_with_frames(monkeypatch, source, ffmpeg,
                    ocr=lambda path, expected: {"status": "OK", "recognized_text": "", "width": 5, "height": 6})
    result = sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 1,
                              "end_seconds": 2}], 0, tmp_path / "out", ffmpeg)
    assert all(frame["text_state"] == "empty_not_absence" for frame in result["regions"][0]["frames"])


def test_decoder_failure_and_missing_ffmpeg_are_unavailable(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(sv, "_run", lambda args, timeout: subprocess.CompletedProcess(args, 1, "", "no video"))
    decoded = sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 1,
                                 "end_seconds": 2}], 0, tmp_path / "decode", ffmpeg)
    assert decoded["status"] == "UNAVAILABLE"
    assert all(frame["reason"] == "frame_decode_failed" for frame in decoded["regions"][0]["frames"])
    ffmpeg.unlink()
    missing = sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 1,
                                 "end_seconds": 2}], 0, tmp_path / "missing", ffmpeg)
    assert missing["status"] == "UNAVAILABLE"
    assert all(frame["reason"] == "ffmpeg_unavailable" for frame in missing["regions"][0]["frames"])


def test_ffmpeg_mutation_raises_integrity_error(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    def mutate_decoder(args, timeout):
        requested = float(args[args.index("-ss") + 1])
        Path(args[-1].replace("%d", str(round(requested * 1_000_000)))).write_bytes(b"png")
        ffmpeg.write_bytes(b"changed decoder")
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(sv, "_run", mutate_decoder)
    with pytest.raises(sv.SourceVisualIntegrityError, match="changed"):
        sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 1,
                           "end_seconds": 2}], 0, tmp_path / "mutated", ffmpeg)


def test_source_mutation_raises_integrity_error_and_keeps_partial_output(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    out = tmp_path / "partial"
    def mutate():
        source.write_bytes(b"changed")
    run_with_frames(monkeypatch, source, ffmpeg, mutate=mutate)
    with pytest.raises(sv.SourceVisualIntegrityError):
        sv.observe_regions(source, hashlib.sha256(b"finished stereo source").hexdigest(),
                           [{"id": "r", "start_seconds": 1, "end_seconds": 2}],
                           0, out, ffmpeg)
    assert out.exists()
    assert list(out.rglob("*.png"))


def test_invalid_or_implicit_offsets_rejected_before_output(tmp_path, monkeypatch):
    source, ffmpeg = setup_inputs(tmp_path, monkeypatch)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        sv.observe_regions(source, digest(source), [{"id": "r", "start_seconds": 1,
                              "end_seconds": 2}], None, out, ffmpeg)
    assert not out.exists()
