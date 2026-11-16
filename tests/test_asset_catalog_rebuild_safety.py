"""目録の同時再構築・接続失敗・確定失敗でも、完成済みDBを守る。"""
from pathlib import Path

import pytest

from lss.assets import catalog


def test_overlapping_rebuilds_do_not_share_or_delete_staging_database(tmp_path, monkeypatch):
    database = tmp_path / "catalog.sqlite"
    state = tmp_path / "state"
    nested = False
    snapshots = []

    def read_ledger(_path):
        nonlocal nested
        if not nested:
            nested = True
            catalog.build(state_root=state, database=database)
            snapshots.append(catalog.summary(database=database))
        return [], []

    monkeypatch.setattr(catalog, "load_tsv", read_ledger)
    catalog.build(state_root=state, database=database)
    assert len(snapshots) == 1
    assert catalog.summary(database=database) == snapshots[0]
    assert not list(tmp_path.glob("*.building"))


def test_rebuild_keeps_unrelated_staging_file(tmp_path):
    database = tmp_path / "catalog.sqlite"
    foreign = tmp_path / ".catalog.sqlite.building"
    foreign.write_bytes(b"another build owns this")
    catalog.build(state_root=tmp_path / "state", database=database)
    assert foreign.read_bytes() == b"another build owns this"
    assert list(tmp_path.glob("*.building")) == [foreign]


def test_failed_database_connection_removes_own_staging_only(tmp_path, monkeypatch):
    database = catalog.build(state_root=tmp_path / "state", database=tmp_path / "catalog.sqlite")
    before = database.read_bytes()

    def fail_connect(_path):
        raise catalog.sqlite3.OperationalError("cannot open database")

    monkeypatch.setattr(catalog.sqlite3, "connect", fail_connect)
    with pytest.raises(catalog.sqlite3.OperationalError, match="cannot open"):
        catalog.build(state_root=tmp_path / "state", database=database)
    assert database.read_bytes() == before
    assert not list(tmp_path.glob("*.building"))


def test_failed_publication_preserves_previous_catalog_and_cleans_staging(tmp_path, monkeypatch):
    database = catalog.build(state_root=tmp_path / "state", database=tmp_path / "catalog.sqlite")
    before = database.read_bytes()
    real_replace = Path.replace

    def fail_replace(path, target):
        if target == database:
            raise PermissionError("catalog is open")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(PermissionError, match="catalog is open"):
        catalog.build(state_root=tmp_path / "state", database=database)
    assert database.read_bytes() == before
    assert not list(tmp_path.glob("*.building"))
