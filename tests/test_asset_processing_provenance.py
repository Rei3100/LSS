"""移動後も処理来歴を保ち、完了時の内容がない場合は版を捏造しない。"""
import json
import shutil
import sqlite3
from contextlib import closing

import pytest
from mutagen.flac import FLAC

from lss.assets import catalog, history, relocate, reprocess
from lss.core import session
from tests.test_reprocess_candidates import CURRENT, _output, _source


def _move(output, state, tmp_path):
    moved = tmp_path / "moved" / output.name
    moved.parent.mkdir(exist_ok=True)
    shutil.move(output, moved)
    relocate.record_relocations(relocate.find_moved([moved.parent], state_root=state), state_root=state)
    return moved


def test_moved_current_processing_stays_preferred_to_an_intact_older_model(tmp_path):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "song.flac", b"source")
    old = _output(tmp_path / "old.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    current = _output(tmp_path / "current.flac", model="model_new", model_sha="n" * 64, config_sha="c" * 64)
    for output, when, recipe in [(old, "2026-08-01", "separation-v1-old"), (current, "2026-09-01", "separation-v1-current")]:
        session.mark_completed(sha, recipe, when, source_path=source, outputs=[output], state_root=state)
    before = session.completion_ledger_path(state).read_bytes()
    _move(current, state, tmp_path)
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["revisions"][1]["processing_state"] == "current"
    assert result["current_best"] == 1 and result["current_best_is_current_processing"]
    assert reprocess.reprocess_candidates(state_root=state, current=CURRENT)["counts"] == {"already_current": 1}
    database = catalog.build(state_root=state, database=tmp_path / "catalog.sqlite")
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT model_id FROM results WHERE recipe_id='separation-v1-current'").fetchone() == ("model_new",)
    assert session.completion_ledger_path(state).read_bytes() == before


def test_moved_older_config_remains_a_reprocess_candidate(tmp_path):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "song.flac", b"source")
    output = _output(tmp_path / "output.flac", model="model_new", model_sha="n" * 64, config_sha="d" * 64)
    session.mark_completed(sha, "separation-v1-old", source_path=source, outputs=[output], state_root=state)
    _move(output, state, tmp_path)
    [candidate] = reprocess.reprocess_candidates(state_root=state, current=CURRENT)["candidates"]
    assert candidate["reason"] == "config_updated"
    assert candidate["previous"]["config_sha256"] == "d" * 64


def test_changed_original_tags_are_not_trusted_when_the_verified_file_has_moved(tmp_path):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "song.flac", b"source")
    output = _output(tmp_path / "output.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-old", source_path=source, outputs=[output], state_root=state)
    moved = _move(output, state, tmp_path)
    _output(output, model="model_new", model_sha="n" * 64, config_sha="c" * 64)
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["revisions"][0]["processing_state"] == "model_updated"
    assert result["revisions"][0]["outputs_intact"]
    [candidate] = reprocess.reprocess_candidates(state_root=state, current=CURRENT)["candidates"]
    assert candidate["previous"]["model_id"] == "model_old"
    moved.write_bytes(b"changed too")
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["revisions"][0]["processing_state"] == "processing_unknown"
    assert result["current_best"] is None


@pytest.mark.parametrize("missing", ["model_sha256", "config_sha256"])
def test_incomplete_matching_processing_is_unverifiable(missing):
    values = {"model_id": CURRENT.model_id, "model_sha256": CURRENT.model_sha256, "config_sha256": CURRENT.config_sha256}
    values[missing] = None
    incomplete = reprocess.ProcessingIdentity(**values)
    assert incomplete.compare(CURRENT) == "processing_unverifiable"
    assert CURRENT.compare(incomplete) == "processing_unverifiable"


def test_invalid_stage_type_is_unresolved_instead_of_crashing_history(tmp_path):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "song.flac", b"source")
    output = _output(tmp_path / "output.flac", model="model_new", model_sha="n" * 64, config_sha="c" * 64)
    tags = FLAC(output)
    tags["LSS_RECIPE_REVISION"] = json.dumps({"stage_graph": ["invalid stage"]})
    tags.save()
    session.mark_completed(sha, "separation-v1-current", source_path=source, outputs=[output], state_root=state)
    result = history.history(sha, state_root=state, current=CURRENT)
    assert result["revisions"][0]["processing_state"] == "processing_unknown"
