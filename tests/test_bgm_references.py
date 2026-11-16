"""作品ごとの参照録音の集まり。内容で確かめ、消えたものは取得物から探し、黙って外さない。"""
from __future__ import annotations

import json

import pytest

from lss.acquisition import store
from lss.bgm import identity_run, references
from tests.test_acquisition import NO_WAIT, FakeBackend, _flac
from tests.test_bgm_identity_run import _mono, completed_run  # noqa: F401


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv(store.LIBRARY_ENV, str(tmp_path / "library"))
    return tmp_path / "state"


def test_local_references_are_added_once_and_resolved_by_content(roots, tmp_path):
    folder = tmp_path / "ost"
    folder.mkdir()
    first = _flac(folder / "a.flac", seed=1)
    _flac(folder / "b.flac", seed=2)
    assert len(references.add_local("series_e", [folder])) == 2
    assert references.add_local("series_e", [first]) == []  # 同じ内容は重ねない

    resolution = references.resolve("series_e")
    assert sorted(p.name for p in resolution.paths) == ["a.flac", "b.flac"]
    assert all(origin == {"kind": "local"} for origin in resolution.origins.values())

    first.write_bytes(first.read_bytes() + b"\0")  # 内容が変わったものは使わず、見つからない録音として返す
    resolution = references.resolve("series_e")
    assert [p.name for p in resolution.paths] == ["b.flac"]
    assert len(resolution.missing) == 1 and resolution.missing[0]["reacquire_url"] is None
    assert len(references.load("series_e")["entries"]) == 2  # 記録からは外さない


def test_acquired_references_keep_their_origin_and_are_found_after_the_library_moves(roots, tmp_path, monkeypatch):
    backend = FakeBackend({"ost1": _flac(tmp_path / "ost1.flac", seed=4)})
    added, failures = references.add_acquired("series_e", "https://example.test/ost1", backend=backend, pacing=NO_WAIT)
    assert failures == [] and len(added) == 1
    origin = added[0]["origin"]
    assert origin["kind"] == "acquisition" and origin["logical_source"] == "example:ost1"
    assert origin["reacquire_url"] == "https://example.test/ost1"
    assert len(references.resolve("series_e").paths) == 1

    # 集まりに記録した場所が消えても、同じ内容の取得物が取得の記録に残っていればそれを使う。
    payload = references.load("series_e")
    payload["entries"][0]["path"] = str(tmp_path / "moved-away.flac")
    references._save(payload, None)
    resolution = references.resolve("series_e")
    assert len(resolution.paths) == 1 and resolution.missing == []


def test_invalid_set_ids_are_refused(roots):
    with pytest.raises(references.ReferenceSetError):
        references.load("../escape")


def test_identity_records_the_origin_of_references_from_a_set(roots, completed_run, tmp_path):  # noqa: F811
    run_path, reference_paths = completed_run
    references.add_local("episode-refs", reference_paths)
    resolution = references.resolve("episode-refs")
    record_path = identity_run.annotate_run(run_path, resolution.paths, loader=_mono,
                                            reference_origins=resolution.origins)
    evidence = json.loads(record_path.read_text(encoding="utf-8"))["external_evidence_used"]
    assert {item["origin"]["kind"] for item in evidence} == {"local"}
