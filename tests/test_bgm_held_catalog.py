"""保留cueを正式成果や録音へ昇格せず、分岐・全所在・破損からの旧目録保持を確認する。"""
import json
from pathlib import Path
import shutil
import sqlite3

from mutagen.flac import FLAC
import pytest

from lss.assets import catalog, held_catalog, relocate
from lss.harness.fetch import file_sha256
from tests.test_application_bgm_reference_residual import bundle, case, execute


@pytest.fixture
def saved(bundle, tmp_path):
    outcome = execute(bundle)
    assert outcome.disposition == "HELD", outcome.message
    state = tmp_path / "state"
    database = catalog.build(state_root=state, outcome_records=[outcome.record_path])
    return outcome, state, database


def test_three_representations_keep_one_occurrence_and_no_published_result(saved):
    outcome, state, database = saved
    assert held_catalog.summary(database=database) == {
        "held_results": 1, "primary_cues": 1, "music_alternatives": 1, "reference_residuals": 1,
        "logical_occurrences": 1, "published_outputs": 0}
    ordinary = catalog.summary(database=database)
    assert ordinary["sources"] == 1
    assert ordinary["results"] == ordinary["derivatives"] == ordinary["recordings"] == ordinary["occurrences"] == 0
    source = json.loads(outcome.record_path.read_bytes())["source"]["sha256"]
    rows = held_catalog.candidates_of(source, database=database)
    primary = next(row for row in rows if row["role"] == "primary")
    assert primary["primary_candidate_id"] is None
    for row in rows:
        assert row["semantic_role"] == "unresolved_music_like" and row["publication_state"] == "not_published"
        assert row["offset_numerator"] == 7 and row["offset_denominator"] == 1000
        if row["role"] != "primary":
            assert row["primary_candidate_id"] == primary["candidate_id"]
    before = outcome.record_path.read_bytes()
    catalog.build(state_root=state, outcome_records=[outcome.record_path, outcome.record_path])
    assert held_catalog.summary(database=database)["held_results"] == 1
    assert outcome.record_path.read_bytes() == before


@pytest.mark.parametrize("damage", ["record", "candidate", "plan", "evidence", "published"])
def test_invalid_required_held_record_does_not_replace_good_catalog(saved, damage):
    outcome, state, database = saved
    before = database.read_bytes()
    target = {"record": outcome.record_path, "candidate": outcome.held_artifacts[0],
              "plan": outcome.record_path.parent / "plan.json", "evidence": outcome.evidence_path,
              "published": outcome.record_path}[damage]
    if damage == "published":
        record = json.loads(target.read_bytes()); record["publication_allowed"] = True
        target.write_text(json.dumps(record), encoding="utf-8")
    else:
        target.write_bytes(b"broken")
    with pytest.raises(held_catalog.CatalogHeldError):
        catalog.build(state_root=state, outcome_records=[outcome.record_path])
    assert database.read_bytes() == before
    assert held_catalog.summary(database=database)["logical_occurrences"] == 1


@pytest.mark.parametrize("error", [FileNotFoundError("消えた記録"), PermissionError("読めない記録")])
def test_unreadable_explicit_outcome_does_not_become_an_empty_catalog(saved, monkeypatch, error):
    outcome, state, database = saved
    before = database.read_bytes(); read = Path.read_bytes
    def blocked(path):
        if path == outcome.record_path:
            raise error
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", blocked)
    with pytest.raises(held_catalog.CatalogHeldError):
        catalog.build(state_root=state, outcome_records=[outcome.record_path])
    assert database.read_bytes() == before


@pytest.mark.parametrize("changed", ["record", "candidate", "source"])
def test_late_held_input_change_preserves_previous_database(saved, monkeypatch, changed):
    outcome, state, database = saved
    before = database.read_bytes(); read_ledger = catalog.load_tsv
    source = Path(json.loads(outcome.record_path.read_bytes())["source"]["path"])
    target = {"record": outcome.record_path, "candidate": outcome.held_artifacts[0], "source": source}[changed]
    def late(path):
        result = read_ledger(path) if Path(path).exists() else ([], [])
        target.write_bytes(target.read_bytes() + b"late")
        return result
    monkeypatch.setattr(catalog, "load_tsv", late)
    with pytest.raises(held_catalog.CatalogHeldError, match="構築中"):
        catalog.build(state_root=state, outcome_records=[outcome.record_path])
    assert database.read_bytes() == before
    assert not list(database.parent.glob("*.building"))


def test_stale_optional_branch_keeps_parent_and_declared_lineage(saved):
    outcome, state, database = saved
    outcome.reference_residual_evidence_path.write_bytes(b"broken")
    catalog.build(state_root=state, outcome_records=[outcome.record_path])
    counts = held_catalog.summary(database=database)
    assert counts["primary_cues"] == counts["music_alternatives"] == 1 and counts["reference_residuals"] == 0
    with sqlite3.connect(database) as connection:
        declared, unavailable = connection.execute("SELECT declared_branches, unavailable_branches FROM held_results").fetchone()
    assert "held_reference_residuals" in json.loads(declared)
    assert json.loads(unavailable) == ["reference_residual"]


def test_changed_semantic_tags_cannot_promote_a_primary_candidate_by_refreshing_hashes(saved):
    outcome, state, database = saved
    before = database.read_bytes(); audio = outcome.held_artifacts[0]
    tags = FLAC(audio); metadata = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    metadata["semantic_role"] = "background_score"
    tags["LSS_CANDIDATE_PROVENANCE"] = json.dumps(metadata); tags.save()
    run = json.loads(outcome.evidence_path.read_bytes())
    run["candidates"][0]["asset"]["sha256"] = file_sha256(audio)
    outcome.evidence_path.write_text(json.dumps(run), encoding="utf-8")
    record = json.loads(outcome.record_path.read_bytes())
    record["held_artifacts"][0]["sha256"] = file_sha256(audio)
    record["evidence"]["sha256"] = file_sha256(outcome.evidence_path)
    outcome.record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(held_catalog.CatalogHeldError, match="semantic_role"):
        catalog.build(state_root=state, outcome_records=[outcome.record_path])
    assert database.read_bytes() == before


def test_moved_candidate_keeps_recorded_origin_and_all_current_locations(saved, tmp_path):
    outcome, state, database = saved
    original = outcome.held_reference_residual_artifacts[0]
    archive = tmp_path / "archive"; archive.mkdir()
    first, second = archive / "renamed.flac", archive / "copy.flac"
    shutil.move(original, first); shutil.copyfile(first, second)
    moves = relocate.find_moved([archive], state_root=state, outcome_records=[outcome.record_path])
    relocate.record_relocations(moves, state_root=state)
    catalog.build(state_root=state, outcome_records=[outcome.record_path])
    source = json.loads(outcome.record_path.read_bytes())["source"]["sha256"]
    row = next(row for row in held_catalog.candidates_of(source, database=database) if row["role"] == "reference_residual")
    assert row["recorded_path"] == str(original)
    assert set(row["current_paths"]) == {str(first), str(second)}
    assert held_catalog.summary(database=database)["logical_occurrences"] == 1


def test_held_queries_are_read_only_and_missing_database_is_not_a_zero(tmp_path):
    missing = tmp_path / "absent.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        held_catalog.summary(database=missing)
    with pytest.raises(sqlite3.OperationalError):
        held_catalog.candidates_of("a" * 64, database=missing)
    assert not missing.exists()
