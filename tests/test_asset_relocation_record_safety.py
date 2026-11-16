"""壊れた移動来歴を空記録で上書きせず、再実行と不正候補で既存記録を守る。"""
import json
from pathlib import Path

import pytest

from lss.assets import relocate


ENTRY = {"sha256": "a" * 64, "kind": "source", "path": "C:/archive/source.flac"}
MOVE = {"sha256": "b" * 64, "kind": "output", "recorded_paths": ["C:/old/output.flac"],
        "found_at": ["C:/archive/output.flac"]}


@pytest.mark.parametrize("payload", [None, [], {}, {"schema": True, "entries": []},
    {"schema": 2, "entries": []}, {"schema": 1, "entries": None},
    {"schema": 1, "entries": [None]}, {"schema": 1, "entries": [{**ENTRY, "kind": []}]},
    {"schema": 1, "entries": [{**ENTRY, "sha256": []}]},
    {"schema": 1, "entries": [ENTRY], "extra": float("nan")}])
def test_invalid_existing_record_is_retained_and_not_treated_as_a_fresh_history(tmp_path, payload):
    path = tmp_path / relocate.RECORD_NAME
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with pytest.raises(relocate.RelocationError):
        relocate.record_relocations([MOVE], state_root=tmp_path)
    assert path.read_bytes() == before
    assert relocate.current_locations(ENTRY["sha256"], state_root=tmp_path) == []


def test_broken_json_is_not_overwritten_even_when_no_move_was_found(tmp_path):
    path = tmp_path / relocate.RECORD_NAME
    path.write_bytes(b'{"entries": [incomplete recoverable bytes')
    before = path.read_bytes()
    with pytest.raises(relocate.RelocationError, match="上書きせず保持"):
        relocate.record_relocations([], state_root=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("invalid", [None, {**MOVE, "found_at": "not a list"}, {**MOVE, "found_at": [None]},
    {**MOVE, "sha256": []}, {**MOVE, "kind": {}}, {**MOVE, "recorded_paths": [None]}])
def test_invalid_incoming_move_does_not_publish_earlier_partial_additions(tmp_path, invalid):
    path = tmp_path / relocate.RECORD_NAME
    path.write_text(json.dumps({"schema": 1, "entries": [ENTRY]}))
    before = path.read_bytes()
    with pytest.raises(relocate.RelocationError):
        relocate.record_relocations([MOVE, invalid], state_root=tmp_path)
    assert path.read_bytes() == before


def test_permission_error_while_reading_cannot_reset_existing_history(tmp_path, monkeypatch):
    path = tmp_path / relocate.RECORD_NAME
    path.write_text(json.dumps({"schema": 1, "entries": [ENTRY]}))
    before = path.read_bytes()
    read_text = Path.read_text

    def inaccessible(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("record temporarily locked")
        return read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", inaccessible)
    with pytest.raises(relocate.RelocationError, match="読めません"):
        relocate.record_relocations([MOVE], state_root=tmp_path)
    assert path.read_bytes() == before


def test_duplicate_replay_keeps_original_provenance_and_file_unchanged(tmp_path):
    path = relocate.record_relocations([MOVE], state_root=tmp_path)
    original = path.read_bytes(), path.stat().st_mtime_ns
    relocate.record_relocations([MOVE], state_root=tmp_path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert len(json.loads(path.read_bytes())["entries"]) == 1


def test_no_moves_does_not_create_an_empty_history(tmp_path):
    path = relocate.record_relocations([], state_root=tmp_path)
    assert not path.exists()
