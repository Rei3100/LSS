"""Read-only query path coverage for the local memory index."""
from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "devmem.py"


@pytest.fixture
def dm():
    spec = importlib.util.spec_from_file_location("devmem_readonly_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def config_with_index(dm, tmp_path):
    cfg = dm.Config(repo=tmp_path, db_dir=tmp_path / "cache")
    con = dm.connect(cfg)
    dm._add_source(con, "fixture", "repo", 0, 0,
                   [(1, 1, "Wanted", "standalone searchable fixture text")],
                   "fixture.md", "today")
    con.commit()
    con.close()
    return cfg


def args(**kwargs):
    defaults = dict(no_refresh=True, text="searchable fixture", k=5, mode="bm25", json=True, budget=1)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_no_refresh_uses_enforced_readonly_connection_and_preserves_db(dm, tmp_path, monkeypatch, capsys):
    cfg = config_with_index(dm, tmp_path)
    before = cfg.db.read_bytes()
    opened = []
    original = dm.connect_readonly

    def tracked(config):
        con = original(config)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("UPDATE src SET size=size+1")
        opened.append(con)
        return con

    monkeypatch.setattr(dm, "connect_readonly", tracked)
    monkeypatch.setattr(dm, "connect", lambda *a, **k: pytest.fail("no-refresh used RW connect"))
    monkeypatch.setattr(dm, "refresh", lambda *a, **k: pytest.fail("no-refresh refreshed"))
    assert dm.cmd_query(args(), cfg) == 0
    assert 'standalone searchable fixture text' in capsys.readouterr().out
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("UPDATE src SET size=1")
    assert cfg.db.read_bytes() == before


def test_no_refresh_closes_connection_after_query_failure(dm, tmp_path, monkeypatch):
    cfg = config_with_index(dm, tmp_path)
    opened = []
    original = dm.connect_readonly
    monkeypatch.setattr(dm, "connect_readonly", lambda c: opened.append(original(c)) or opened[-1])
    monkeypatch.setattr(dm, "search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("search failed")))
    with pytest.raises(RuntimeError, match="search failed"):
        dm.cmd_query(args(), cfg)
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


def test_missing_index_returns_two_without_creating_directory(dm, tmp_path, capsys):
    cfg = dm.Config(repo=tmp_path, db_dir=tmp_path / "missing-cache")
    assert dm.cmd_query(args(), cfg) == 2
    assert not cfg.db_dir.exists()
    assert "no index yet" in capsys.readouterr().err


def test_refresh_path_still_refreshes_and_uses_rw_connection(dm, tmp_path, monkeypatch, capsys):
    cfg = config_with_index(dm, tmp_path)
    calls = []
    original = dm.connect
    monkeypatch.setattr(dm, "connect_readonly", lambda *a: pytest.fail("refresh used RO connection"))
    monkeypatch.setattr(dm, "connect", lambda *a, **k: calls.append("connect") or original(*a, **k))
    monkeypatch.setattr(dm, "stale_kinds", lambda *a: {"repo"})
    monkeypatch.setattr(dm, "refresh", lambda *a, **k: calls.append(("refresh", k["kinds"])))
    assert dm.cmd_query(args(no_refresh=False), cfg) == 0
    assert calls == ["connect", ("refresh", {"repo"}), "connect"]
    assert "standalone searchable fixture text" in capsys.readouterr().out
