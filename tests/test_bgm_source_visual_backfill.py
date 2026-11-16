"""START ancillary migration, cache falsifiers and immutable accepted evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from lss.bgm import source_visual_backfill as cache, source_visual_job as job, source_visual as visual
from lss.core import session


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"finished stereo source")
    directory = tmp_path / "run"
    (directory / "semantics").mkdir(parents=True)
    semantics = {"segments": [{"start_seconds": 4., "end_seconds": 9.,
                  "scope": {"candidate_id": "segment-022", "disposition": "HELD",
                            "reason": "program_insert_role_unresolved"}}]}
    path = directory / "semantics" / "cue-semantics.json"
    session._write_json_atomic(path, semantics)
    run = {"source": {"sha256": _hash(source), "audio_timeline_origin": {
            "method": "first_decoded_audio_pts", "offset_seconds": .007,
            "offset_numerator": 7, "offset_denominator": 1000}},
           "semantics": {"record": {"name": path.name, "sha256": _hash(path)}}}
    run_path = directory / "run.json"
    session._write_json_atomic(run_path, run)
    return source, run, run_path


def _provider(monkeypatch, *, unavailable=False):
    calls = []
    def observe(source, digest, regions, origin, directory):
        calls.append(regions)
        directory.mkdir()
        return {"status": "UNAVAILABLE" if unavailable else "OK", "source_sha256": digest,
                "source_sha256_after": digest, "source_unchanged": True,
                "audio_pts_origin_seconds": origin,
                "regions": [{**row, "status": "UNAVAILABLE" if unavailable else "OK"} for row in regions]}
    monkeypatch.setattr(visual, "observe_regions", observe)
    return calls


def test_backfill_and_second_start_preserve_accepted_bytes(tmp_path, monkeypatch):
    source, run, run_path = _fixture(tmp_path)
    calls = _provider(monkeypatch)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    original = deepcopy(run)
    first = cache.ensure(source, _hash(source), run, run_path)
    second = cache.ensure(source, _hash(source), run, run_path)
    assert first == second and first["status"] == "OK" and len(calls) == 1
    assert run == original
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (run_path.parent / cache.SIDECAR).is_file()
    assert first["coverage"] == "sampled_only" and first["role_authority"] == "none"


@pytest.mark.parametrize("corruption", ["hash", "targets", "timeline", "policy", "raw_binding"])
def test_invalid_binding_is_not_a_cache_hit(tmp_path, monkeypatch, corruption):
    source, run, run_path = _fixture(tmp_path)
    calls = _provider(monkeypatch)
    cache.ensure(source, _hash(source), run, run_path)
    sidecar = run_path.parent / cache.SIDECAR
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if corruption == "hash":
        record["summary"]["report_sha256"] = "0" * 64
    elif corruption == "raw_binding":
        raw_path = Path(record["summary"]["report_path"])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw["observation"]["source_sha256"] = "0" * 64
        session._write_json_atomic(raw_path, raw)
        record["summary"]["report_sha256"] = _hash(raw_path)
    else:
        record["binding"][corruption] = "invalid"
    session._write_json_atomic(sidecar, record)
    assert cache.ensure(source, _hash(source), run, run_path)["status"] == "OK"
    assert len(calls) == 2


def test_typed_unavailability_is_bounded_and_does_not_respawn(tmp_path, monkeypatch):
    source, run, run_path = _fixture(tmp_path)
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise OSError("provider unavailable")
    monkeypatch.setattr(visual, "observe_regions", fail)
    now = [1000.]
    monkeypatch.setattr(cache.time, "time", lambda: now[0])
    assert cache.ensure(source, _hash(source), run, run_path)["status"] == "UNAVAILABLE"
    cache.ensure(source, _hash(source), run, run_path)
    assert len(calls) == 1
    now[0] += cache.RETRY_SECONDS
    cache.ensure(source, _hash(source), run, run_path)
    assert len(calls) == 2


def test_valid_fresh_observations_seed_without_decode(tmp_path, monkeypatch):
    source, run, run_path = _fixture(tmp_path)
    calls = _provider(monkeypatch)
    path = run_path.parent / "semantics" / "cue-semantics.json"
    semantics = json.loads(path.read_text(encoding="utf-8"))
    semantics["source_visual"] = job.observe(source, _hash(source), run["source"]["audio_timeline_origin"],
                                            semantics["segments"], run_path.parent)
    session._write_json_atomic(path, semantics)
    run["semantics"]["record"]["sha256"] = _hash(path)
    result = cache.ensure(source, _hash(source), run, run_path)
    assert result["status"] == "OK" and len(calls) == 1


def test_integrity_failures_and_source_mutation_propagate(tmp_path, monkeypatch):
    source, run, run_path = _fixture(tmp_path)
    def fail(*args, **kwargs):
        raise visual.SourceVisualIntegrityError("wrong binding")
    monkeypatch.setattr(visual, "observe_regions", fail)
    with pytest.raises(visual.SourceVisualIntegrityError):
        cache.ensure(source, _hash(source), run, run_path)
    def mutate(*args, **kwargs):
        source.write_bytes(b"changed")
        raise OSError("unavailable")
    monkeypatch.setattr(visual, "observe_regions", mutate)
    with pytest.raises(session.InputChangedError):
        cache.ensure(source, run["source"]["sha256"], run, run_path)


def test_report_unavailable_is_cached(tmp_path, monkeypatch):
    source, run, run_path = _fixture(tmp_path)
    calls = _provider(monkeypatch, unavailable=True)
    for _ in range(2):
        assert cache.ensure(source, _hash(source), run, run_path)["status"] == "UNAVAILABLE"
    assert len(calls) == 1


def test_expired_unavailable_seed_retries_provider(tmp_path, monkeypatch):
    source, run, run_path = _fixture(tmp_path)
    calls = _provider(monkeypatch, unavailable=True)
    path = run_path.parent / "semantics" / "cue-semantics.json"
    semantics = json.loads(path.read_text(encoding="utf-8"))
    semantics["source_visual"] = job.observe(source, _hash(source), run["source"]["audio_timeline_origin"],
                                            semantics["segments"], run_path.parent)
    session._write_json_atomic(path, semantics)
    run["semantics"]["record"]["sha256"] = _hash(path)
    now = [1000.]
    monkeypatch.setattr(cache.time, "time", lambda: now[0])
    cache.ensure(source, _hash(source), run, run_path)
    assert len(calls) == 1
    now[0] += cache.RETRY_SECONDS
    cache.ensure(source, _hash(source), run, run_path)
    assert len(calls) == 2
