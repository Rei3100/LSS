"""再処理候補から移動済み原本を辿り、改変・所在不明と同内容の複数コピーを区別する。"""
import shutil

import pytest

from lss.assets import reprocess, relocate
from lss.core import session
from tests.test_reprocess_candidates import CURRENT, _source, _output


def _case(tmp_path):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "source.flac", b"source content")
    output = _output(tmp_path / "output.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-old", source_path=source, outputs=[output], state_root=state)
    return state, source, sha


@pytest.mark.parametrize("verify", [True, False])
def test_moved_input_exposes_all_verified_copies_without_changing_source_provenance(tmp_path, verify):
    state, source, sha = _case(tmp_path)
    before = session.completion_ledger_path(state).read_bytes()
    archive = tmp_path / "archive"
    archive.mkdir()
    moved = archive / "renamed.flac"
    copy = archive / "copy.flac"
    shutil.move(source, moved)
    shutil.copyfile(moved, copy)
    relocate.record_relocations(relocate.find_moved([archive], state_root=state), state_root=state)
    [candidate] = reprocess.reprocess_candidates(state_root=state, current=CURRENT, verify_sources=verify)["candidates"]
    assert candidate["source_status"] == "moved"
    assert candidate["source_path"] == str(source.resolve())
    assert candidate["source_locations"] == sorted([str(moved.resolve()), str(copy.resolve())])
    assert candidate["input_sha256"] == sha
    assert session.completion_ledger_path(state).read_bytes() == before


def test_changed_original_does_not_hide_a_verified_copy(tmp_path):
    state, source, _ = _case(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    moved = archive / "original.flac"
    shutil.move(source, moved)
    relocate.record_relocations(relocate.find_moved([archive], state_root=state), state_root=state)
    source.write_bytes(b"different content now at original path")
    [candidate] = reprocess.reprocess_candidates(state_root=state, current=CURRENT, verify_sources=True)["candidates"]
    assert candidate["source_status"] == "changed_with_verified_copy"
    assert candidate["source_locations"] == [str(moved.resolve())]
    moved.write_bytes(b"changed copy too")
    [candidate] = reprocess.reprocess_candidates(state_root=state, current=CURRENT, verify_sources=True)["candidates"]
    assert candidate["source_status"] == "changed" and candidate["source_locations"] == []


@pytest.mark.parametrize("verify,status", [(False, "present_unverified"), (True, "present")])
def test_original_location_keeps_verification_state_explicit(tmp_path, verify, status):
    state, source, _ = _case(tmp_path)
    [candidate] = reprocess.reprocess_candidates(state_root=state, current=CURRENT, verify_sources=verify)["candidates"]
    assert candidate["source_status"] == status
    assert candidate["source_locations"] == [str(source.resolve())]


def test_empty_config_hash_is_unverifiable(tmp_path):
    assert reprocess.ProcessingIdentity(CURRENT.model_id, CURRENT.model_sha256, "").compare(CURRENT) == "processing_unverifiable"
