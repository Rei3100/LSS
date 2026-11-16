"""読めない履歴を空として返さず、CLIでも以前の再処理一覧を保全する。"""
import json

import pytest

from lss.assets import history, reprocess
from lss.core import session
from scripts import list_reprocess_candidates as cli
from tests.test_reprocess_candidates import CURRENT, _output, _source


@pytest.fixture
def completed(tmp_path):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "source.flac", b"source")
    output = _output(tmp_path / "output.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-old", source_path=source, outputs=[output], state_root=state)
    return state, sha, output


@pytest.mark.parametrize("module", [history, reprocess])
@pytest.mark.parametrize("error", [PermissionError("ledger locked"), ValueError("ledger invalid"),
                                    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "ledger invalid UTF-8")])
def test_unreadable_ledger_is_an_error_and_all_stored_results_survive(completed, monkeypatch, module, error):
    state, sha, output = completed
    before = session.completion_ledger_path(state).read_bytes(), output.read_bytes()

    def unreadable(path):
        raise error

    with monkeypatch.context() as scoped:
        scoped.setattr(module, "load_tsv", unreadable)
        with pytest.raises(type(error), match="ledger"):
            if module is history:
                history.history(sha, state_root=state, current=CURRENT)
            else:
                reprocess.reprocess_candidates(state_root=state, current=CURRENT)
    assert (session.completion_ledger_path(state).read_bytes(), output.read_bytes()) == before
    assert history.history(sha, state_root=state, current=CURRENT)["current_best"] == 0
    assert len(reprocess.reprocess_candidates(state_root=state, current=CURRENT)["candidates"]) == 1


def test_new_state_without_a_ledger_has_an_empty_history_and_candidate_list(tmp_path):
    assert history.history("a" * 64, state_root=tmp_path, current=CURRENT)["revisions"] == []
    assert reprocess.reprocess_candidates(state_root=tmp_path, current=CURRENT)["candidates"] == []


def test_cli_failure_preserves_an_existing_candidate_export(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "candidates.json"
    destination.write_bytes(b"previous verified candidate export")

    def unreadable(**kwargs):
        raise PermissionError("ledger locked")

    monkeypatch.setattr(cli, "reprocess_candidates", unreadable)
    assert cli.main(["--out", str(destination)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    error = json.loads(captured.err)
    assert error["status"] == "FAILED" and error["error_code"] == "reprocess_listing_failed"
    assert "ledger locked" in error["message"]
    assert destination.read_bytes() == b"previous verified candidate export"


def test_cli_export_failure_preserves_previous_file_and_cleans_only_its_staging_file(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "candidates.json"
    destination.write_bytes(b"previous candidate export")
    monkeypatch.setattr(cli, "reprocess_candidates", lambda **k: {"counts": {}, "candidates": []})

    def disk_failure(fd):
        raise OSError("保存容量不足")

    monkeypatch.setattr(cli.session.os, "fsync", disk_failure)
    assert cli.main(["--out", str(destination)]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["error_code"] == "reprocess_export_failed"
    assert destination.read_bytes() == b"previous candidate export"
    assert not list(tmp_path.glob(".candidates.json-*.tmp"))


def test_cli_complete_export_replaces_previous_file_after_serialization(tmp_path, monkeypatch):
    destination = tmp_path / "exports" / "candidates.json"
    result = {"current": {}, "counts": {"model_updated": 1}, "candidates": [{"reason": "model_updated"}]}
    monkeypatch.setattr(cli, "reprocess_candidates", lambda **k: result)
    assert cli.main(["--out", str(destination)]) == 0
    assert json.loads(destination.read_bytes()) == result
    assert not list(destination.parent.glob(".candidates.json-*.tmp"))
