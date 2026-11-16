"""A presentation refresh writes only to the CurrentBest public_path; a kept-but-missing cue never fails a switch."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm import current_best, cue_memory, scene_job, scene_names
from tests.test_bgm_scene_job import FACTS, FakeModels, _cue, _write_current_best_for_recovery

SHA = "a" * 64
RUN = "b" * 64


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_refresh_never_writes_the_library_the_cue_memory_still_names(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_WORK_INFO", "0")
    monkeypatch.setattr(scene_names, "source_facts", lambda source, tags=None: {**FACTS, "episode": 137})
    monkeypatch.setattr(scene_job, "sound_words", lambda path: "短調（暗め・切なめ）、強弱はなだらか")
    state, library_a, library_b = tmp_path / "stateB", tmp_path / "A" / "ep", tmp_path / "B" / "ep"
    cue = _cue(library_a, "s1", 400)
    shutil.copytree(library_a, library_b)
    file_a, file_b = cue["path"], library_b / cue["path"].name
    before_a, before_b = _digest(file_a), _digest(file_b)
    # CurrentBest of the relocated state names library B; the cue memory (copied along) still names library A.
    _write_current_best_for_recovery(state, tmp_path / "series-alpha-137.mkv", "s1", [{**cue, "path": file_b}])
    memory = cue_memory._empty()
    memory["members"][cue["asset_id"]] = {"asset_id": cue["asset_id"], "path": str(file_a), "source_sha256": "s1",
                                          "pcm_sha256": cue["pcm_sha256"], "cue_key": cue["asset_id"][8:]}
    monkeypatch.setattr(cue_memory, "read", lambda state_root: memory)
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=FakeModels(["ミズウオ大作戦"]))
    assert _digest(file_a) == before_a                      # nothing under A is modified
    assert _digest(file_b) != before_b and FLAC(file_b)["TITLE"] == ["ミズウオ大作戦（仮題）"]


def test_refresh_skips_a_cue_that_is_not_a_current_published_file(tmp_path):
    state = tmp_path / "state"
    cue = _cue(tmp_path / "A", "s1", 400)
    before = _digest(cue["path"])
    scene_names.record_cue(state, cue["asset_id"], pcm_sha256=cue["pcm_sha256"], source_sha256="s1",
                           facts={**FACTS, "episode": 137}, start_seconds=400.0, end_seconds=420.0, heard=[],
                           order={"use": None, "direction": "コミカルに", "category": None, "name": "ミズウオ大作戦"},
                           model={"id": "fake"}, music={})
    memory = cue_memory._empty()
    memory["members"][cue["asset_id"]] = {"path": str(cue["path"])}
    # No CurrentBest record at all: the memory path alone is never a write target, and the job does not raise.
    result = scene_job.refresh_work(state, FACTS["work_key"], memory)
    assert result["refreshed"] == 0 and _digest(cue["path"]) == before
    assert result["pending"][0]["error"] == "not_a_current_published_path"


def _flac(path: Path, seed: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.random.default_rng(seed).uniform(-0.1, 0.1, (4410, 2)), 44100, format="FLAC", subtype="PCM_24")
    return current_best.pcm_sha256(path)


def _row(public: Path, pcm: str, key: str) -> dict:
    return {"public_path": str(public), "pcm_sha256": pcm, "cue_key": key, "run_key": RUN, "recipe_id": "r",
            "workflow": "w", "run_path": "run.json", "adoption": "test"}


def _switch(root: Path, kept_row: dict, new_row: dict | None = None, staged_file: Path | None = None):
    state = root / "state"
    previous = {"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
                "source_sha256": SHA, "current": {"files": [kept_row]}, "history": []}
    current_best._write_json_atomic(current_best.record_path(state, SHA), previous)
    files = [kept_row] + ([{**new_row, "placed_now": True}] if new_row else [])
    staged = {new_row["public_path"]: staged_file} if new_row else {}
    journal = current_best.plan_switch(state, SHA, {"files": files, "pinned": False}, staged,
                                       root / ".lss-bgm-history", retire=[], record=previous,
                                       previous_current=previous["current"], mode="recipe_supersession")
    return state, previous, journal


def _scenario(tmp_path):
    kept_pcm = _flac(tmp_path / "orig.flac", 1)
    staged = tmp_path / "stage" / "new.flac"
    new_pcm = _flac(staged, 2)
    kept, new_public = tmp_path / "lib" / "kept.flac", tmp_path / "lib" / "new.flac"
    state, previous, journal = _switch(tmp_path, _row(kept, kept_pcm, "old"), _row(new_public, new_pcm, "new"), staged)
    return state, previous, journal, kept, kept_pcm, new_public


def test_a_deleted_kept_cue_is_regenerated_from_its_run_record(tmp_path):
    state, _, journal, kept, kept_pcm, new_public = _scenario(tmp_path)
    calls = []

    def regenerate(item, public):
        calls.append(item["cue_key"])
        public.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path / "orig.flac", public)

    record = current_best.execute(state, journal, regenerate)
    assert calls == ["old"] and current_best.pcm_sha256(kept) == kept_pcm and new_public.is_file()
    assert [item["cue_key"] for item in record["current"]["files"]] == ["old", "new"]
    assert current_best.read(state, SHA)["current"]["files"] == record["current"]["files"]


def test_an_unrebuildable_kept_cue_is_excluded_as_unknown_and_the_run_still_publishes(tmp_path):
    state, _, journal, kept, kept_pcm, new_public = _scenario(tmp_path)

    def regenerate(item, public):
        raise current_best.CurrentBestError("no run")

    record = current_best.execute(state, journal, regenerate)
    assert [item["cue_key"] for item in record["current"]["files"]] == ["new"] and new_public.is_file()
    unknown = record["current"]["unavailable"]
    assert len(unknown) == 1 and unknown[0]["cue_key"] == "old" and unknown[0]["state"] == "UNKNOWN"
    assert unknown[0]["pcm_sha256"] == kept_pcm            # the full former row is kept, not forgotten
    # Without any regenerator the outcome is the same.
    other = tmp_path / "second"
    state2, _, journal2 = _switch(other, _row(kept, kept_pcm, "old"))
    assert current_best.execute(state2, journal2)["current"]["unavailable"][0]["state"] == "UNKNOWN"


def test_a_kept_file_with_different_pcm_is_displaced_to_history_and_the_cue_rebuilt(tmp_path):
    kept_pcm = _flac(tmp_path / "orig.flac", 1)
    kept = tmp_path / "lib" / "kept.flac"
    _flac(kept, 99)                                         # same name, other sound: not LSS's recorded file
    before = kept.read_bytes()
    state, previous, journal = _switch(tmp_path, _row(kept, kept_pcm, "old"))
    rebuilt = []

    def regenerate(item, public):
        rebuilt.append(public)
        public.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path / "orig.flac", public)

    record = current_best.execute(state, journal, regenerate)
    [moved] = record["displaced"]
    assert Path(moved["moved_to"]).read_bytes() == before        # the user's file survives, never deleted
    assert rebuilt == [kept] and kept.read_bytes() != before


def test_a_regeneration_with_other_pcm_is_removed_and_excluded(tmp_path):
    kept_pcm = _flac(tmp_path / "orig.flac", 1)
    _flac(tmp_path / "other.flac", 5)
    kept = tmp_path / "lib" / "kept.flac"

    def regenerate(item, public):
        public.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path / "other.flac", public)

    state, _, journal = _switch(tmp_path, _row(kept, kept_pcm, "old"))
    record = current_best.execute(state, journal, regenerate)
    assert record["current"]["files"] == [] and not kept.exists()
    assert record["current"]["unavailable"][0]["reason"] == "regenerated_pcm_differs"
