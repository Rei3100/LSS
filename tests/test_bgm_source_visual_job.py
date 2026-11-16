import hashlib
import json
from pathlib import Path

import pytest

from lss.bgm import source_visual as sv
from lss.bgm import source_visual_job as job


def _cue(identifier, disposition="HELD", reason="program_insert_role_unresolved"):
    return {"start_seconds": 4.0, "end_seconds": 9.0, "semantic_class": "UNKNOWN",
            "scope": {"candidate_id": identifier, "disposition": disposition, "reason": reason}}


def _report(digest, origin, regions, **overrides):
    return {"status": "OK", "source_sha256": digest, "source_sha256_after": digest,
            "source_unchanged": True, "audio_pts_origin_seconds": origin,
            "regions": regions, "coverage": "sampled_instants_only", **overrides}


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_targets_does_no_work(tmp_path, monkeypatch):
    monkeypatch.setattr(sv, "observe_regions", lambda *a, **k: pytest.fail("must not start visual work"))
    result = job.observe(tmp_path / "missing", "bad", None, [_cue("x", reason="other")], tmp_path)
    assert result["status"] == "NOT_NEEDED"


@pytest.mark.parametrize("timeline", [None, {"method": "standalone_audio_sample_zero", "offset_seconds": 0}])
def test_unknown_or_standalone_timeline_is_unavailable_without_work(tmp_path, monkeypatch, timeline):
    monkeypatch.setattr(sv, "observe_regions", lambda *a, **k: pytest.fail("must not decode without known origin"))
    result = job.observe(tmp_path / "missing", "a" * 64, timeline, [_cue("x")], tmp_path)
    assert result["status"] == "UNAVAILABLE"
    assert result["reason"] == "audio_timeline_origin_unavailable"


def test_only_exact_held_insert_targets_and_report_is_bound(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    candidates = [_cue("insert"), _cue("published", disposition="PUBLISHED"), _cue("other", reason="other")]
    def observe(path, digest, regions, origin, work_dir):
        assert regions == [{"id": "insert", "start_seconds": 4.0, "end_seconds": 9.0}]
        assert origin == 3.25
        assert not work_dir.exists()
        work_dir.mkdir()
        return _report(digest, origin, [{"id": "insert", "start_seconds": 4.0,
                "end_seconds": 9.0, "status": "OK", "frames": [{"ocr": {"recognized_text": "raw"}}]}])
    monkeypatch.setattr(sv, "observe_regions", observe)
    result = job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 3.25}, candidates, tmp_path)
    assert candidates[0]["scope"] == {"candidate_id": "insert", "disposition": "HELD",
                                       "reason": "program_insert_role_unresolved"}
    assert "source_visual_evidence" in candidates[0]
    assert all("source_visual_evidence" not in cue for cue in candidates[1:])
    ref = candidates[0]["source_visual_evidence"]
    report = Path(ref["report_path"])
    assert _hash(report) == ref["report_sha256"]
    assert ref["coverage"] == "sampled_only" and ref["role_authority"] == "none"
    assert result["status"] == "OK"
    assert json.loads(report.read_text(encoding="utf-8"))["observation"]["source_sha256"] == _hash(source)


def test_source_visual_integrity_error_propagates(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    def fail(*args, **kwargs):
        raise sv.SourceVisualIntegrityError("changed")
    monkeypatch.setattr(sv, "observe_regions", fail)
    with pytest.raises(sv.SourceVisualIntegrityError, match="changed"):
        job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 0},
                    [_cue("x")], tmp_path)


def test_cross_source_binding_is_rejected_by_provider(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    def fail(*args, **kwargs):
        raise sv.SourceVisualIntegrityError("source did not match expected SHA-256")
    monkeypatch.setattr(sv, "observe_regions", fail)
    with pytest.raises(sv.SourceVisualIntegrityError):
        job.observe(source, "0" * 64, {"method": "first_decoded_audio_pts", "offset_seconds": 0},
                    [_cue("x")], tmp_path)


def test_normal_provider_report_uses_fresh_dir_and_keeps_raw_report(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    calls = []
    def observe(path, digest, regions, origin, work_dir):
        calls.append(work_dir)
        work_dir.mkdir()
        return _report(digest, origin, [{"id": "x", "start_seconds": 4.0,
                                         "end_seconds": 9.0, "status": "UNAVAILABLE"}], status="UNAVAILABLE")
    monkeypatch.setattr(sv, "observe_regions", observe)
    cue = _cue("x")
    job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 0}, [cue], tmp_path)
    assert len(calls) == 1
    assert calls[0].is_dir()
    assert (calls[0] / "report.json").is_file()
    job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 0}, [cue], tmp_path)
    assert len(calls) == 2 and calls[0] != calls[1]


def test_two_held_segments_use_candidate_ids_without_cue_ids(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    cues = [_cue("segment-022"), _cue("segment-023")]
    cues[1]["start_seconds"], cues[1]["end_seconds"] = 12.0, 17.0
    def observe(path, digest, regions, origin, work_dir):
        assert [region["id"] for region in regions] == ["segment-022", "segment-023"]
        work_dir.mkdir()
        return _report(digest, origin, [{**region, "status": "OK"} for region in regions])
    monkeypatch.setattr(sv, "observe_regions", observe)
    job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 2}, cues, tmp_path)
    assert [cue["source_visual_evidence"]["region_id"] for cue in cues] == ["segment-022", "segment-023"]


def test_wrong_source_in_returned_report_is_integrity_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    def observe(path, digest, regions, origin, work_dir):
        work_dir.mkdir()
        return _report("0" * 64, origin, [{**regions[0], "status": "OK"}])
    monkeypatch.setattr(sv, "observe_regions", observe)
    with pytest.raises(sv.SourceVisualIntegrityError, match="different source"):
        job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 2}, [_cue("x")], tmp_path)


def test_wrong_region_bounds_are_integrity_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    def observe(path, digest, regions, origin, work_dir):
        work_dir.mkdir()
        return _report(digest, origin, [{**regions[0], "end_seconds": 10.0, "status": "OK"}])
    monkeypatch.setattr(sv, "observe_regions", observe)
    with pytest.raises(sv.SourceVisualIntegrityError, match="bounds"):
        job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 2}, [_cue("x")], tmp_path)


def test_duplicate_returned_region_never_attaches_ambiguous_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    cue = _cue("x")
    def observe(path, digest, regions, origin, work_dir):
        work_dir.mkdir()
        row = {**regions[0], "status": "OK"}
        return _report(digest, origin, [row, dict(row)])
    monkeypatch.setattr(sv, "observe_regions", observe)
    with pytest.raises(sv.SourceVisualIntegrityError, match="repeated"):
        job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 2}, [cue], tmp_path)
    assert "source_visual_evidence" not in cue


def test_provider_oserror_and_report_save_error_are_ancillary(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    cue = _cue("x")
    def fail(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(sv, "observe_regions", fail)
    result = job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 0}, [cue], tmp_path)
    assert result["status"] == "UNAVAILABLE" and "source_visual_evidence" not in cue
    monkeypatch.setattr(sv, "observe_regions", lambda path, digest, regions, origin, work: (work.mkdir() or _report(
        digest, origin, [{**regions[0], "status": "OK"}])))
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    result = job.observe(source, _hash(source), {"method": "first_decoded_audio_pts", "offset_seconds": 0}, [cue], tmp_path)
    assert result["status"] == "UNAVAILABLE" and "source_visual_evidence" not in cue


def test_inconsistent_or_nonfinite_timeline_is_unavailable_without_work(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"source")
    monkeypatch.setattr(sv, "observe_regions", lambda *a, **k: pytest.fail("invalid time must not start provider"))
    for timeline in ({"method": "first_decoded_audio_pts", "offset_seconds": float("nan")},
                     {"method": "first_decoded_audio_pts", "offset_seconds": 2,
                      "offset_numerator": 3, "offset_denominator": 2}):
        cue = _cue("x")
        result = job.observe(source, _hash(source), timeline, [cue], tmp_path)
        assert result["status"] == "UNAVAILABLE" and "source_visual_evidence" not in cue
