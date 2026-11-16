"""取得台帳が読めない場合に、所在・再取得元・目録を空の状態で確定しない。"""
import json
from pathlib import Path

import pytest

from lss.acquisition import service, store
from lss.assets import catalog, reprocess
from lss.core import session
from scripts import list_reprocess_candidates
from tests.test_acquisition import FakeBackend, NO_WAIT, _flac, roots
from tests.test_reprocess_candidates import CURRENT, _output


@pytest.fixture
def acquired(roots, tmp_path):
    state, _ = roots
    [entry] = service.acquire("https://example.test/x1", backend=FakeBackend({
        "x1": _flac(tmp_path / "media.flac"),
    }), pacing=NO_WAIT)
    source = entry.source
    ledger = store._ledger_path(source.logical_source, state)
    output = _output(tmp_path / "result.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(source.sha256, "separation-v1-old", source_path=source.path, outputs=[output], state_root=state)
    database = catalog.build(state_root=state, database=tmp_path / "catalog.sqlite")
    return state, source, ledger, output, database


@pytest.mark.parametrize("error", [PermissionError("取得台帳を読めません"), ValueError("取得台帳が不正です")])
def test_ledger_read_error_preserves_catalog_and_does_not_report_missing_provenance(
    acquired, tmp_path, monkeypatch, error,
):
    state, source, ledger, output, database = acquired
    before = database.read_bytes(), ledger.read_bytes(), output.read_bytes()
    read = Path.read_text
    def fail_read(path, *args, **kwargs):
        if path == ledger:
            raise error
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(type(error)):
        store.records_for_sha256(source.sha256, state)
    with pytest.raises(type(error)):
        catalog.build(state_root=state, database=database)
    source.path.unlink()
    with pytest.raises(type(error)):
        reprocess.reprocess_candidates(state_root=state, current=CURRENT, verify_sources=True)
    assert (database.read_bytes(), ledger.read_bytes(), output.read_bytes()) == before
    assert not list(tmp_path.glob(".*.building"))


def test_corrupt_ledger_does_not_replace_a_previous_cli_export(acquired, tmp_path, monkeypatch, capsys):
    state, source, ledger, _, _ = acquired
    source.path.unlink()
    ledger.write_bytes(b"{interrupted")
    destination = tmp_path / "candidates.json"
    destination.write_bytes(b"previous valid export")
    monkeypatch.setattr(reprocess, "current_identity", lambda: CURRENT)

    assert list_reprocess_candidates.main(["--verify-sources", "--out", str(destination)]) == 1
    assert json.loads(capsys.readouterr().err)["error_code"] == "reprocess_listing_failed"
    assert destination.read_bytes() == b"previous valid export" and ledger.read_bytes() == b"{interrupted"


@pytest.mark.parametrize("damage", ["list", "schema", "source", "record_source", "record_type"])
def test_invalid_ledger_structure_is_not_reused_or_overwritten(acquired, damage):
    state, source, ledger, _, _ = acquired
    payload = json.loads(ledger.read_bytes())
    if damage == "list":
        payload = []
    elif damage == "schema":
        payload["schema"] = 999
    elif damage == "source":
        payload["logical_source"] = "example:other"
    elif damage == "record_source":
        payload["revisions"][0]["logical_source"] = "example:other"
    else:
        payload["revisions"] = [None]
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    before = ledger.read_bytes()

    with pytest.raises(ValueError, match="取得台帳"):
        store.load_revisions(source.logical_source, state)
    with pytest.raises(ValueError, match="取得台帳"):
        list(store.iter_records(state))
    with pytest.raises(ValueError, match="取得台帳"):
        store._append_revision(source.record, state)
    assert ledger.read_bytes() == before


def test_directory_listing_error_is_not_an_empty_acquisition_state(acquired, monkeypatch):
    state, _, _, _, _ = acquired
    iterdir = Path.iterdir
    def fail_scan(path):
        if path == store.ledger_dir(state):
            raise PermissionError("取得台帳を列挙できません")
        return iterdir(path)
    monkeypatch.setattr(Path, "iterdir", fail_scan)

    with pytest.raises(PermissionError, match="列挙"):
        list(store.iter_records(state))


def test_missing_acquisition_state_is_a_valid_empty_state(tmp_path):
    assert list(store.iter_records(tmp_path)) == []
    assert store.records_for_sha256("a" * 64, tmp_path) == []


def test_optional_tag_enrichment_can_skip_bad_unrelated_ledger(acquired, tmp_path):
    state, source, _, _, _ = acquired
    other = store._ledger_path("example:other", state)
    other.write_bytes(b"{interrupted")
    tags = store.inherited_tags_for(source.path, state)
    assert tags["TITLE"] == "曲 x1"
    assert other.read_bytes() == b"{interrupted"


def test_url_lookup_does_not_hide_unreadable_acquisition_history(acquired):
    state, source, ledger, _, _ = acquired
    ledger.write_bytes(b"{interrupted")
    with pytest.raises(ValueError):
        store.find_valid_by_url("https://example.test/x1", state)
    assert source.path.is_file()
