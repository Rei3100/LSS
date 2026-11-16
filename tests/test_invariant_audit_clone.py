"""Invariant 3 (file ownership): processing the same content from another location never edits, moves or deletes
the files a stale CurrentBest/run record still names in the library it was copied from.

`_verified_current_files` already relocates a kept CurrentBest to the processed source's own folder on the reuse
path; the supersession path must do the same before it retires or keeps anything.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from lss import application
from lss.bgm import current_best
from tests.test_application_bgm_candidates import _run, _source, gpu  # noqa: F401
from tests.test_application_bgm_current_best import _legacy, V7_WORKFLOW


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(folder):
    return {str(p.relative_to(folder)): (_digest(p), p.stat().st_mtime_ns) for p in sorted(folder.rglob("*"))
            if p.is_file()}


def _older_with_record(tmp_path, monkeypatch, gpu, source):
    gpu.tagger.singing = 0.0
    _, old, old_run = _legacy(tmp_path, monkeypatch, source, workflow=V7_WORKFLOW, music_from=7)
    monkeypatch.setattr(current_best, "cue_improvements", lambda item: ["dialogue_residual_whole_cue"])
    return old, old_run


def test_supersession_from_another_location_never_touches_the_library_a_legacy_run_names(tmp_path, gpu,
                                                                                          monkeypatch):
    library_a = tmp_path / "A"
    library_a.mkdir()
    source_a = _source(library_a / "episode.wav")
    old, old_run = _older_with_record(tmp_path, monkeypatch, gpu, source_a)
    before = _snapshot(library_a)
    source_b = tmp_path / "B" / "episode.wav"
    source_b.parent.mkdir()
    shutil.copyfile(source_a, source_b)
    outcome = _run(tmp_path, source_b)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert _snapshot(library_a) == before                       # A: nothing edited, moved or deleted
    for path in outcome.outputs:
        assert path.resolve().is_relative_to((tmp_path / "B" / "LSS").resolve())
    record = current_best.read(tmp_path / "state", old_run["source"]["sha256"])
    for item in record["current"]["files"]:
        assert item["public_path"].startswith(str(tmp_path / "B" / "LSS"))


def test_supersession_from_another_location_never_touches_the_library_a_current_best_names(tmp_path, gpu,
                                                                                            monkeypatch):
    library_a = tmp_path / "A"
    library_a.mkdir()
    source_a = _source(library_a / "episode.wav")
    old, old_run = _older_with_record(tmp_path, monkeypatch, gpu, source_a)
    sha = old_run["source"]["sha256"]
    # A CurrentBest (copied state) that still names library A's public file.
    row = {"public_path": str(old), "pcm_sha256": current_best.pcm_sha256(old),
           "cue_key": old_run["published"][0]["key"], "start_seconds": old_run["published"][0]["start_seconds"],
           "end_seconds": old_run["published"][0]["end_seconds"], "run_key": old.parent.name,
           "recipe_id": old_run["recipe_id"], "workflow": V7_WORKFLOW,
           "run_path": str(next((tmp_path / "state" / "bgm-candidates").glob("*/run/run.json"))),
           "adoption": "first_publication", "improvements": [], "extent": None}
    row["run_key"] = Path(row["run_path"]).parents[1].name
    current_best._write_json_atomic(current_best.record_path(tmp_path / "state", sha), {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION, "source_sha256": sha,
        "current": {"files": [row], "pinned": False, "evaluated_runs": [row["run_key"]]}, "history": []})
    before = _snapshot(library_a)
    source_b = tmp_path / "B" / "episode.wav"
    source_b.parent.mkdir()
    shutil.copyfile(source_a, source_b)
    outcome = _run(tmp_path, source_b)
    assert outcome.disposition is application.ResultDisposition.PUBLISHED, outcome.message
    assert _snapshot(library_a) == before
    for path in outcome.outputs:
        assert path.resolve().is_relative_to((tmp_path / "B" / "LSS").resolve())
    record = current_best.read(tmp_path / "state", sha)
    assert all(item["public_path"].startswith(str(tmp_path / "B" / "LSS")) for item in record["current"]["files"])
    assert json.dumps(record).count(json.dumps(str(tmp_path / "A" / "LSS"))[1:-1]) >= 1   # history still names where it came from


def test_rollback_from_another_location_restores_beside_that_source_and_never_writes_the_old_library(
        tmp_path, gpu, monkeypatch):
    """Invariants 1-3: a rollback whose history names another library restores beside the processed source."""
    library_a = tmp_path / "A"
    library_a.mkdir()
    source_a = _source(library_a / "episode.wav")
    old, old_run = _older_with_record(tmp_path, monkeypatch, gpu, source_a)
    old_pcm = current_best.pcm_sha256(old)
    first = _run(tmp_path, source_a)                                     # improved cue replaces the old one at A
    assert first.disposition is application.ResultDisposition.PUBLISHED and not old.exists()
    before = _snapshot(library_a)
    source_b = tmp_path / "B" / "episode.wav"
    source_b.parent.mkdir()
    shutil.copyfile(source_a, source_b)
    input_before = (source_b.read_bytes(), source_b.stat().st_mtime_ns)
    restored = application.restore_previous_bgm_result(source_b, state_root=tmp_path / "state")
    assert restored.disposition is application.ResultDisposition.PUBLISHED, restored.message
    assert _snapshot(library_a) == before                                 # A: nothing written, moved or deleted
    [here] = restored.outputs
    assert here.resolve().is_relative_to((tmp_path / "B" / "LSS").resolve())
    assert current_best.pcm_sha256(here) == old_pcm                      # the prior CurrentBest's exact audio
    assert (source_b.read_bytes(), source_b.stat().st_mtime_ns) == input_before


def test_the_original_input_is_never_modified_by_any_job_refresh_or_rollback(tmp_path, gpu, monkeypatch):
    """Invariant 2: bytes and mtime of the finished input are unchanged after publish, replace, reuse, rollback."""
    source = _source(tmp_path / "episode.wav")
    stamp = (_digest(source), source.stat().st_mtime_ns)
    _older_with_record(tmp_path, monkeypatch, gpu, source)
    assert (_digest(source), source.stat().st_mtime_ns) == stamp
    _run(tmp_path, source)                                               # supersession
    _run(tmp_path, source)                                               # reuse + presentation refresh
    application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    application.restore_previous_bgm_result(source, state_root=tmp_path / "state")
    assert (_digest(source), source.stat().st_mtime_ns) == stamp
