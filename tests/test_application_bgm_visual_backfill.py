"""Normal START against a frozen legacy eligible semantic fixture.

Audio and lifecycle use the normal product with the existing GPU-only doubles.
The short fixture's metadata is migrated before freezing to represent an old
held program-insert run, without allocating a 600-second inference fixture.
"""
import json

import pytest

from lss import application
from lss.bgm import source_visual_backfill as cache, source_visual_job as job, source_visual as visual
from lss.bgm import current_best
from lss.core import session
from tests.test_application_bgm_candidates import gpu, _run, _source


@pytest.mark.parametrize("unavailable", [False, True])
def test_normal_start_backfills_only_ancillary_files_and_readback_is_read_only(
        tmp_path, gpu, monkeypatch, unavailable):
    source = _source(tmp_path / "episode.wav")
    gpu.tagger.noncontinuous_music = True
    first = _run(tmp_path, source)
    assert first.disposition is application.ResultDisposition.HELD, first.message
    run_path = first.record_path
    run = json.loads(run_path.read_text(encoding="utf-8"))
    path = run_path.parent / "semantics" / run["semantics"]["record"]["name"]
    semantics = json.loads(path.read_text(encoding="utf-8"))
    for segment in semantics["segments"]:
        segment["scope"]["disposition"] = "HELD"
        segment["scope"]["reason"] = "program_insert_role_unresolved"
    semantics.pop("source_visual", None)
    session._write_json_atomic(path, semantics)
    run["semantics"]["record"]["sha256"] = job._sha256(path)
    run["source"]["audio_timeline_origin"].update(method="first_decoded_audio_pts", offset_seconds=0.)
    session._write_json_atomic(run_path, run)
    # This is the freeze boundary: accepted run, semantics, PCM and CurrentBest.
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    calls = []
    def observe(source, digest, regions, origin, work_dir):
        calls.append(regions)
        work_dir.mkdir()
        return {"status": "UNAVAILABLE" if unavailable else "OK", "source_sha256": digest,
                "source_sha256_after": digest, "source_unchanged": True,
                "audio_pts_origin_seconds": origin,
                "regions": [{**row, "status": "UNAVAILABLE" if unavailable else "OK"} for row in regions]}
    monkeypatch.setattr(visual, "observe_regions", observe)
    second = _run(tmp_path, source)
    assert second.disposition is application.ResultDisposition.HELD, second.message
    assert second.reused and len(gpu.calls) == 1 and len(calls) == 1
    assert all(p.read_bytes() == content for p, content in before.items())
    assert (run_path.parent / cache.SIDECAR).is_file()
    third = _run(tmp_path, source)
    assert third.reused and len(calls) == 1
    frozen = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(cache, "ensure", lambda *a, **k: pytest.fail("readback cannot backfill"))
    application._bgm_outcome(run, run_path, reused=True, source=source)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == frozen


def test_pinned_normal_start_reaches_backfill_without_reinference_or_currentbest_changes(tmp_path, gpu, monkeypatch):
    source = _source(tmp_path / "pinned.wav")
    first = _run(tmp_path, source)
    assert first.disposition is application.ResultDisposition.PUBLISHED, first.message
    run = json.loads(first.record_path.read_text(encoding="utf-8"))
    record_path = current_best.record_path(tmp_path / "state", job._sha256(source))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["current"]["pinned"] = True
    record["current"]["evaluated_runs"] = []
    session._write_json_atomic(record_path, record)
    semantics = first.record_path.parent / "semantics" / run["semantics"]["record"]["name"]
    protected = [record_path, first.record_path, semantics, *first.outputs]
    before = {p: p.read_bytes() for p in protected}
    calls = []
    original = cache.ensure
    def ensure(*args):
        calls.append(args[3])
        return original(*args)
    monkeypatch.setattr(cache, "ensure", ensure)
    monkeypatch.setattr(visual, "observe_regions", lambda *a, **k: pytest.fail("no eligible held targets"))
    second = _run(tmp_path, source)
    assert second.disposition is application.ResultDisposition.PUBLISHED and second.reused, second.message
    assert calls == [first.record_path] and len(gpu.calls) == 1
    assert all(p.read_bytes() == content for p, content in before.items())

