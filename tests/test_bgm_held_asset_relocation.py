"""保留音声と原本の移動を内容で追跡し、候補の来歴・件数・非公開状態を保つ。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lss import application
from lss.assets import relocate
from tests.test_application_bgm_identity import routed_result, _execute


def _archive(case, outcome, tmp_path, *, copies=False):
    archive = tmp_path / "archive"
    archive.mkdir()
    source = case[0].sources[0]
    originals = [source, *outcome.held_artifacts]
    locations = {}
    for index, original in enumerate(originals):
        moved = archive / f"renamed-{index}{original.suffix}"
        shutil.move(original, moved)
        paths = [moved.resolve()]
        if copies:
            copy = archive / f"copy-{index}{original.suffix}"
            shutil.copyfile(moved, copy)
            paths.append(copy.resolve())
        locations[original] = tuple(sorted(paths))
    return archive, locations


@pytest.mark.parametrize("copies", [False, True])
def test_moved_bgm_inputs_are_discovered_and_read_back_with_all_locations(routed_result, tmp_path, copies):
    original = _execute(routed_result)
    before = original.record_path.read_bytes()
    archive, locations = _archive(routed_result, original, tmp_path, copies=copies)
    state = tmp_path / "isolated-state"
    # BGM保留結果は完了台帳の正式成果物ではない。台帳の無い状態でも検索できる。
    moves = relocate.find_moved([archive], state_root=state, outcome_records=[original.record_path])
    assert len(moves) == 3 and sum(m["kind"] == "source" for m in moves) == 1
    assert all(len(m["found_at"]) == (2 if copies else 1) for m in moves)
    relocation_record = relocate.record_relocations(moves, state_root=state)
    relocation_before = relocation_record.read_bytes()
    outcome = application.read_bgm_development_outcome(original.record_path, state_root=state)
    assert outcome.disposition == application.ResultDisposition.HELD and outcome.reused is True
    assert len(outcome.held_artifacts) == 2 and outcome.outputs == ()
    assert outcome.source_locations == locations[routed_result[0].sources[0]]
    assert outcome.held_artifact_locations == {p: locations[p] for p in original.held_artifacts}
    assert outcome.held_artifacts == tuple(locations[p][0] for p in original.held_artifacts)
    assert original.record_path.read_bytes() == before
    assert relocation_record.read_bytes() == relocation_before


def test_changed_original_can_use_a_registered_exact_copy(routed_result, tmp_path):
    original = _execute(routed_result)
    archive, locations = _archive(routed_result, original, tmp_path)
    state = tmp_path / "state"
    relocate.record_relocations(relocate.find_moved([archive], state_root=state,
                                                   outcome_records=[original.record_path]), state_root=state)
    routed_result[0].sources[0].write_bytes(b"another source at old path")
    original.held_artifacts[0].write_bytes(b"another candidate at old path")
    outcome = application.read_bgm_development_outcome(original.record_path, state_root=state)
    assert outcome.disposition == application.ResultDisposition.HELD
    assert outcome.source_locations == locations[routed_result[0].sources[0]]
    assert outcome.held_artifact_locations[original.held_artifacts[0]] == locations[original.held_artifacts[0]]


def test_same_filename_with_different_content_is_not_discovered_or_reused(routed_result, tmp_path):
    original = _execute(routed_result)
    archive, locations = _archive(routed_result, original, tmp_path)
    moved = locations[original.held_artifacts[0]][0]
    moved.write_bytes(b"unrelated sound")
    state = tmp_path / "state"
    moves = relocate.find_moved([archive], state_root=state, outcome_records=[original.record_path])
    assert len(moves) == 2
    relocate.record_relocations(moves, state_root=state)
    assert application.read_bgm_development_outcome(original.record_path, state_root=state).disposition == (
        application.ResultDisposition.FAILED)


def test_late_moved_candidate_change_is_rejected_without_record_update(routed_result, tmp_path, monkeypatch):
    original = _execute(routed_result)
    before = original.record_path.read_bytes()
    archive, locations = _archive(routed_result, original, tmp_path)
    state = tmp_path / "state"
    relocate.record_relocations(relocate.find_moved([archive], state_root=state,
                                                   outcome_records=[original.record_path]), state_root=state)
    verify = application._verify_bgm_inputs

    def changed(*a, **k):
        locations[original.held_artifacts[0]][0].write_bytes(b"changed after locating")
        verify(*a, **k)

    monkeypatch.setattr(application, "_verify_bgm_inputs", changed)
    outcome = application.read_bgm_development_outcome(original.record_path, state_root=state)
    assert outcome.disposition == application.ResultDisposition.FAILED
    assert original.record_path.read_bytes() == before


@pytest.mark.parametrize("content", [b"{", b"[]", b'{"type":"other"}'])
def test_unreadable_requested_held_record_does_not_become_no_missing_assets(routed_result, tmp_path, content):
    original = _execute(routed_result)
    original.record_path.write_bytes(content)
    with pytest.raises(relocate.RelocationError, match="BGM保留結果"):
        relocate.find_moved([tmp_path], state_root=tmp_path / "state", outcome_records=[original.record_path])


def test_held_outcome_change_during_search_is_rejected(routed_result, tmp_path, monkeypatch):
    original = _execute(routed_result)
    archive, _ = _archive(routed_result, original, tmp_path)
    digest = relocate.session.input_sha256_of
    changed = False

    def changed_hash(path):
        nonlocal changed
        if not changed:
            changed = True
            original.record_path.write_bytes(original.record_path.read_bytes() + b"\n")
        return digest(path)

    monkeypatch.setattr(relocate.session, "input_sha256_of", changed_hash)
    with pytest.raises(relocate.RelocationError, match="変更"):
        relocate.find_moved([archive], state_root=tmp_path / "state", outcome_records=[original.record_path])


@pytest.mark.parametrize("error", [PermissionError("locked ledger"), ValueError("bad ledger")])
def test_unreadable_completion_ledger_is_not_treated_as_no_missing_assets(tmp_path, monkeypatch, error):
    def fail(path):
        raise error

    monkeypatch.setattr(relocate, "load_tsv", fail)
    with pytest.raises(relocate.RelocationError, match="完了台帳"):
        relocate.find_moved([tmp_path], state_root=tmp_path / "state")
