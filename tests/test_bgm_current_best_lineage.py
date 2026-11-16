"""Lineage of CurrentBest history: two different audios of the same run key and public name must keep separate
history copies (a re-run of the same Recipe on the same source reuses the run key; GPU inference may differ)."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess

import numpy as np
import soundfile as sf
import pytest
from mutagen.flac import FLAC

from lss.bgm import current_best, current_best_lineage

SHA = "a" * 64
RUN = "b" * 64


def _flac(path: Path, seed: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.random.default_rng(seed).uniform(-0.1, 0.1, (4410, 2)), 44100, format="FLAC", subtype="PCM_24")
    return current_best.pcm_sha256(path)


def _row(public: Path, pcm: str) -> dict:
    return {"public_path": str(public), "pcm_sha256": pcm, "cue_key": "cue", "run_key": RUN, "recipe_id": "r",
            "workflow": "w", "run_path": "run.json", "adoption": "test"}


def test_same_run_key_and_name_with_different_audio_keep_separate_history_copies(tmp_path):
    state, history, public = tmp_path / "state", tmp_path / "LSS" / ".lss-bgm-history", tmp_path / "LSS" / "x.flac"
    pcm_a = _flac(public, 1)
    pcm_b = _flac(tmp_path / "staged" / "x.flac", 2)
    row_a, row_b = _row(public, pcm_a), _row(public, pcm_b)
    # A is adopted, then a re-run of the same run key replaces it by B (A is retired into history).
    journal = current_best.plan_switch(
        state, SHA, {"files": [{**row_b, "placed_now": True}], "pinned": False, "evaluated_runs": [RUN]},
        {str(public): tmp_path / "staged" / "x.flac"}, history, retire=[row_a], record=None,
        previous_current={"files": [row_a], "evaluated_runs": [RUN]}, mode="recipe_supersession")
    current_best.execute(state, journal)
    # Rollback to A retires B into the same run key and public name.
    record = current_best_lineage.restore_previous(state, SHA, history)
    assert current_best.pcm_sha256(public) == pcm_a
    [retired_b] = record["history"][-1]["files"]
    assert retired_b["pcm_sha256"] == pcm_b
    assert current_best.pcm_sha256(Path(retired_b["archive_path"])) == pcm_b
    # Going forward again must bring back B's exact audio from its own history copy, without any regenerator.
    forward = current_best_lineage.restore_previous(state, SHA, history)
    assert current_best.pcm_sha256(public) == pcm_b and forward["current"]["files"][0]["pcm_sha256"] == pcm_b


@pytest.mark.parametrize("restore", [current_best.restore_previous, current_best_lineage.restore_previous])
@pytest.mark.skipif(os.name != "nt", reason="Windows junction alias regression")
def test_rollback_keeps_same_physical_audio_referenced_through_a_junction(tmp_path, restore):
    physical, alias = tmp_path / "physical", tmp_path / "alias"
    physical.mkdir()
    subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(physical)], check=True,
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    old_rows, new_rows, before = [], [], {}
    for index in range(2):
        path = physical / f"kept-{index}.flac"
        pcm = _flac(path, index+10)
        tags = FLAC(path)
        tags["TITLE"] = f"USER TITLE {index}"
        tags.save()
        before[path.name] = path.read_bytes()
        old_rows.append(_row(alias / path.name, pcm))
        new_rows.append(_row(path, pcm))
    extra = physical / "added.flac"
    extra_pcm = _flac(extra, 20)
    new_rows.append(_row(extra, extra_pcm))
    state, history = tmp_path / "state", tmp_path / ".lss-bgm-history"
    record = {"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
              "source_sha256": SHA, "current": {"files": new_rows, "evaluated_runs": [RUN]},
              "history": [{"files": old_rows, "evaluated_runs": [RUN]}]}
    current_best._write_json_atomic(current_best.record_path(state, SHA), record)
    restored = restore(state, SHA, history)
    assert len(restored["current"]["files"]) == 2 and restored["current"]["pinned"]
    for path in physical.glob("kept-*.flac"):
        assert path.read_bytes() == before[path.name]
    assert len(list(physical.glob("kept-*.flac"))) == 2
    assert all(current_best.pcm_sha256(Path(row["public_path"])) == row["pcm_sha256"] for row in old_rows)
    assert not extra.exists()
    archived = [row for row in restored["history"][-1]["files"] if row.get("archive_path")]
    assert len(archived) == 1 and archived[0]["pcm_sha256"] == extra_pcm
    forward = restore(state, SHA, history)
    assert len(forward["current"]["files"]) == 3 and current_best.pcm_sha256(extra) == extra_pcm
    assert all((physical / name).read_bytes() == content for name, content in before.items())


def test_same_pcm_at_distinct_public_paths_is_still_a_real_rollback_move(tmp_path):
    old, new = tmp_path / "old.flac", tmp_path / "new.flac"
    pcm = _flac(old, 50)
    new.write_bytes(old.read_bytes())
    state, history = tmp_path / "state", tmp_path / ".lss-bgm-history"
    record = {"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
              "source_sha256": SHA, "current": {"files": [_row(new, pcm)], "evaluated_runs": [RUN]},
              "history": [{"files": [_row(old, pcm)], "evaluated_runs": [RUN]}]}
    current_best._write_json_atomic(current_best.record_path(state, SHA), record)
    restored = current_best_lineage.restore_previous(state, SHA, history)
    assert Path(restored["current"]["files"][0]["public_path"]) == old
    assert current_best.pcm_sha256(old) == pcm and not new.exists()


def test_execute_excludes_an_unrebuildable_missing_kept_file_as_unknown_instead_of_failing(tmp_path):
    missing = _row(tmp_path / "missing.flac", "0"*64)
    state = tmp_path / "state"
    previous = {"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
                "source_sha256": SHA, "current": {"files": [missing]}, "history": []}
    current_best._write_json_atomic(current_best.record_path(state, SHA), previous)
    journal = current_best.plan_switch(state, SHA, {"files": [missing], "pinned": True}, {},
                                       tmp_path / ".lss-bgm-history", retire=[], record=previous,
                                       previous_current=None, mode="restored_previous")
    record = current_best.execute(state, journal)
    assert record["current"]["files"] == []
    assert [row["public_path"] for row in record["current"]["unavailable"]] == [missing["public_path"]]
    assert record["current"]["unavailable"][0]["state"] == "UNKNOWN"
    assert current_best.read(state, SHA) == record


def _kept_switch(tmp_path, row, regenerate):
    state = tmp_path / "state"
    previous = {"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
                "source_sha256": SHA, "current": {"files": [row]}, "history": []}
    current_best._write_json_atomic(current_best.record_path(state, SHA), previous)
    journal = current_best.plan_switch(state, SHA, {"files": [row], "pinned": True}, {},
                                       tmp_path / "LSS" / ".lss-bgm-history", retire=[], record=previous,
                                       previous_current=None, mode="restored_previous")
    return current_best.execute(state, journal, regenerate)


def test_a_different_file_at_a_kept_cue_path_is_displaced_and_the_recorded_cue_placed(tmp_path):
    public = tmp_path / "LSS" / "kept.flac"
    recorded = _flac(tmp_path / "ref" / "kept.flac", 1)
    stray = _flac(public, 2)
    stray_bytes = public.read_bytes()
    foreign = tmp_path / "LSS" / "foreign.flac"
    _flac(foreign, 3)
    foreign_bytes = foreign.read_bytes()
    reference = tmp_path / "ref" / "kept.flac"
    record = _kept_switch(tmp_path, _row(public, recorded), lambda item, dest: dest.write_bytes(reference.read_bytes()))
    assert current_best.pcm_sha256(public) == recorded != stray
    [moved] = record["displaced"]
    assert moved["public_path"] == str(public) and Path(moved["moved_to"]).read_bytes() == stray_bytes
    assert "displaced" in Path(moved["moved_to"]).parts
    assert foreign.read_bytes() == foreign_bytes and [r["public_path"] for r in record["current"]["files"]] == [str(public)]


def test_the_recorded_file_at_a_kept_cue_path_is_left_untouched(tmp_path):
    public = tmp_path / "LSS" / "kept.flac"
    recorded = _flac(public, 1)
    before = public.read_bytes()
    record = _kept_switch(tmp_path, _row(public, recorded), lambda item, dest: pytest.fail("must not regenerate"))
    assert public.read_bytes() == before and not record.get("displaced")
    assert not (tmp_path / "LSS" / ".lss-bgm-history" / "displaced").exists()


@pytest.mark.parametrize("fail_placement", [False, True])
def test_replacement_preserves_foreign_public_bytes_even_with_a_valid_archive(tmp_path, monkeypatch, fail_placement):
    state, history = tmp_path / "state", tmp_path / "LSS" / current_best.HISTORY_DIR
    public, staged = tmp_path / "LSS" / "cue.flac", tmp_path / "staged" / "cue.flac"
    old_pcm, new_pcm = _flac(public, 11), _flac(staged, 12)
    old_bytes = public.read_bytes()
    old, new = _row(public, old_pcm), _row(public, new_pcm)
    previous = {"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
                "source_sha256": SHA, "current": {"files": [old]}, "history": []}
    current_best._write_json_atomic(current_best.record_path(state, SHA), previous)
    journal = current_best.plan_switch(
        state, SHA, {"files": [{**new, "placed_now": True}], "evaluated_runs": [RUN]},
        {str(public): staged}, history, retire=[old], record=previous,
        previous_current=previous["current"], mode="recipe_supersession")
    archive = Path(journal["retire"][0]["archive_path"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(old_bytes)
    _flac(public, 13)
    foreign_bytes = public.read_bytes()
    replace = os.replace

    def place_or_fail(source, target):
        if fail_placement and Path(source) == staged:
            raise OSError("placement failed")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", place_or_fail)
    if fail_placement:
        with pytest.raises(OSError, match="placement failed"):
            current_best.execute(state, journal)
        assert public.read_bytes() == old_bytes
        assert current_best.read(state, SHA) == previous
    else:
        current_best.execute(state, journal)
        assert current_best.pcm_sha256(public) == new_pcm
    displaced = list((history / "displaced").rglob("*.flac"))
    assert [path.read_bytes() for path in displaced] == [foreign_bytes]
    assert archive.read_bytes() == old_bytes


def test_rollback_displaces_a_foreign_file_that_arrives_after_placement(tmp_path, monkeypatch):
    state, history = tmp_path / "state", tmp_path / "LSS" / current_best.HISTORY_DIR
    public, staged = tmp_path / "LSS" / "cue.flac", tmp_path / "staged" / "cue.flac"
    old_pcm, new_pcm = _flac(public, 21), _flac(staged, 22)
    old_bytes = public.read_bytes()
    old, new = _row(public, old_pcm), _row(public, new_pcm)
    previous = {"type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
                "source_sha256": SHA, "current": {"files": [old]}, "history": []}
    current_best._write_json_atomic(current_best.record_path(state, SHA), previous)
    journal = current_best.plan_switch(
        state, SHA, {"files": [{**new, "placed_now": True}], "evaluated_runs": [RUN]},
        {str(public): staged}, history, retire=[old], record=previous,
        previous_current=previous["current"], mode="recipe_supersession")
    write_record = current_best._write_json_atomic
    archive = Path(journal["retire"][0]["archive_path"])
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(old_bytes)
    _flac(public, 24)
    first_foreign_bytes = public.read_bytes()
    foreign_bytes = None

    def fail_commit(path, record):
        nonlocal foreign_bytes
        if Path(path) == current_best.record_path(state, SHA) and record is journal["next_record"]:
            _flac(public, 23)
            foreign_bytes = public.read_bytes()
            raise OSError("record failed")
        return write_record(path, record)

    monkeypatch.setattr(current_best, "_write_json_atomic", fail_commit)
    with pytest.raises(OSError, match="record failed"):
        current_best.execute(state, journal)
    assert public.read_bytes() == old_bytes
    assert current_best.read(state, SHA) == previous
    assert sorted(path.read_bytes() for path in (history / "displaced").rglob("*.flac")) == sorted(
        [first_foreign_bytes, foreign_bytes])
