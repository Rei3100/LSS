"""録音照合記録の破損・読取失敗・構築中の変更で完成済みの目録を欠かさない。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lss.assets import catalog
from tests.test_asset_catalog import _identity_record


@pytest.fixture
def populated_catalog(tmp_path):
    record = _identity_record(tmp_path / "identity.json", "a" * 64, "b" * 64,
                              [(10.0, 30.0, 1.0, "known_exact")])
    database = catalog.build(state_root=tmp_path / "state", identity_records=[record],
                             database=tmp_path / "catalog.sqlite")
    return record, database, database.read_bytes()


@pytest.mark.parametrize("content", [b"{", b"[]", b"null", b'{"type":"other"}',
                                      b'{"run":{},"matches":[]}'])
def test_invalid_identity_record_preserves_existing_occurrences(populated_catalog, tmp_path, content):
    record, database, before = populated_catalog
    record.write_bytes(content)
    with pytest.raises(catalog.CatalogIdentityError, match="identity.json"):
        catalog.build(state_root=tmp_path / "state", identity_records=[record], database=database)
    assert database.read_bytes() == before
    assert catalog.summary(database=database)["occurrences"] == 1
    assert not list(tmp_path.glob("*.building"))


@pytest.mark.parametrize("error", [FileNotFoundError("missing identity"), PermissionError("locked identity")])
def test_unreadable_requested_record_is_not_silently_omitted(populated_catalog, tmp_path, monkeypatch, error):
    record, database, before = populated_catalog
    read = Path.read_bytes

    def unreadable(path):
        if path == record:
            raise error
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(catalog.CatalogIdentityError, match="identity.json"):
        catalog.build(state_root=tmp_path / "state", identity_records=[record], database=database)
    assert database.read_bytes() == before


@pytest.mark.parametrize("field,value", [("query_start_seconds", -1.0), ("query_end_seconds", 10.0),
                                        ("query_end_seconds", float("nan")),
                                        ("reference_start_seconds", float("inf")),
                                        ("query_start_seconds", True), ("waveform_explained", 1.2),
                                        ("reference_sha256", "wrong")])
def test_invalid_match_identity_and_timeline_are_not_indexed(populated_catalog, tmp_path, field, value):
    record, database, before = populated_catalog
    payload = json.loads(record.read_bytes())
    payload["matches"][0][field] = value
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(catalog.CatalogIdentityError):
        catalog.build(state_root=tmp_path / "state", identity_records=[record], database=database)
    assert database.read_bytes() == before


@pytest.mark.parametrize("change", ["rewrite", "delete"])
def test_identity_changes_during_build_preserve_previous_catalog(populated_catalog, tmp_path, monkeypatch, change):
    record, database, before = populated_catalog
    read_ledger = catalog.load_tsv

    def changed(path):
        if change == "rewrite":
            record.write_bytes(record.read_bytes() + b"\n")
        else:
            record.unlink()
        return read_ledger(path)

    monkeypatch.setattr(catalog, "load_tsv", changed)
    with pytest.raises(catalog.CatalogIdentityError):
        catalog.build(state_root=tmp_path / "state", identity_records=[record], database=database)
    assert database.read_bytes() == before
    assert not list(tmp_path.glob("*.building"))


def test_repeated_record_paths_do_not_duplicate_an_occurrence(populated_catalog, tmp_path):
    record, database, _ = populated_catalog
    catalog.build(state_root=tmp_path / "state", identity_records=[record, record], database=database)
    assert catalog.summary(database=database)["occurrences"] == 1


@pytest.mark.parametrize("damage", ["unknown_episode", "duplicate_episode", "bad_source", "bad_span", "checksum"])
def test_invalid_series_evidence_preserves_existing_catalog(populated_catalog, tmp_path, damage):
    _, database, before = populated_catalog
    payload = {"type": "series_identity_annotation",
               "episodes": [{"key": e, "source_sha256": e * 64} for e in "ab"],
               "shared_recordings": [{"first": "a", "second": "b", "occurrences": [
                   {"first_start_seconds": 10.0, "first_end_seconds": 30.0,
                    "second_start_seconds": 40.0, "second_end_seconds": 60.0}]}]}
    if damage == "unknown_episode":
        payload["shared_recordings"][0]["second"] = "missing"
    elif damage == "duplicate_episode":
        payload["episodes"].append(payload["episodes"][0])
    elif damage == "bad_source":
        payload["episodes"][0]["source_sha256"] = None
    elif damage == "bad_span":
        payload["shared_recordings"][0]["occurrences"][0]["second_end_seconds"] = float("inf")
    else:
        payload["payload_sha256"] = "c" * 64
    record = tmp_path / "series_identity.json"
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(catalog.CatalogIdentityError):
        catalog.build(state_root=tmp_path / "state", identity_records=[record], database=database)
    assert database.read_bytes() == before
