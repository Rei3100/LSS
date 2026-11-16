"""出所だけが変わった参照のcacheを安全に更新し、波形照合を繰り返さない。"""
import json

import pytest

from lss.bgm import identity, identity_run
from lss.harness.fetch import file_sha256
from tests.test_application_bgm_identity import routed_result


def test_reference_origin_changes_refresh_metadata_without_repeating_matches(routed_result, monkeypatch):
    _, _, run, reference, *_ = routed_result
    path = identity_run.annotate_run(run, [reference])
    before = json.loads(path.read_bytes())
    origins = {file_sha256(reference): {"kind": "local", "catalog": {"record": "library-r2"}}}
    monkeypatch.setattr(identity, "match", lambda *a, **k: pytest.fail("出所の変更で再照合しない"))
    identity_run.annotate_run(run, [reference], reference_origins=origins)
    after = json.loads(path.read_bytes())
    assert after["matches"] == before["matches"] and after["activity_runs"] == before["activity_runs"]
    assert after["created_at"] == before["created_at"] and "metadata_updated_at" in after
    assert after["external_evidence_used"][0]["origin"] == origins[file_sha256(reference)]
    stamp = path.stat().st_mtime_ns
    identity_run.annotate_run(run, [reference], reference_origins=origins)
    assert path.stat().st_mtime_ns == stamp
    identity_run.annotate_run(run, [reference])
    assert path.stat().st_mtime_ns == stamp
    assert json.loads(path.read_bytes())["external_evidence_used"][0]["origin"] == origins[file_sha256(reference)]


def test_origin_is_a_snapshot_of_the_request_not_a_mutable_caller_dict(routed_result, monkeypatch):
    _, _, run, reference, *_ = routed_result
    sha = file_sha256(reference)
    origins = {sha: {"catalog": {"record": "initial-library"}}}
    previous_match = identity.match

    def changed_match(*a, **k):
        origins[sha]["catalog"]["record"] = "changed-during-match"
        return previous_match(*a, **k)

    monkeypatch.setattr(identity, "match", changed_match)
    path = identity_run.annotate_run(run, [reference], reference_origins=origins)
    assert json.loads(path.read_bytes())["external_evidence_used"][0]["origin"]["catalog"]["record"] == "initial-library"


def test_late_reference_change_during_metadata_refresh_preserves_previous_annotation(routed_result, monkeypatch):
    _, _, run, reference, *_ = routed_result
    path = identity_run.annotate_run(run, [reference])
    before = path.read_bytes()
    original_verify = identity_run.verify_annotation_inputs

    def changed_verify(*a, **k):
        reference.write_bytes(reference.read_bytes() + b"changed")
        original_verify(*a, **k)

    monkeypatch.setattr(identity_run, "verify_annotation_inputs", changed_verify)
    with pytest.raises(identity_run.IdentityRunError, match="参照録音"):
        identity_run.annotate_run(run, [reference], reference_origins={file_sha256(reference): {"kind": "local"}})
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".identity.json-*.tmp"))


def test_overlapping_annotation_writes_have_separate_staging_files(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text('{"generation": "previous"}', encoding="utf-8")

    def first_validation():
        def second_validation():
            assert len(list(tmp_path.glob(".identity.json-*.tmp"))) == 2
        identity_run._write_annotation(path, {"generation": "second"}, second_validation)
        assert json.loads(path.read_bytes())["generation"] == "second"

    identity_run._write_annotation(path, {"generation": "first"}, first_validation)
    assert json.loads(path.read_bytes())["generation"] == "first"
    assert not list(tmp_path.glob(".identity.json-*.tmp"))
