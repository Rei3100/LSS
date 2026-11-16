"""SSD pinning retains bytes after cache eviction and never falls back to archive reads."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_ai_storage as storage


def fixture(tmp_path, monkeypatch):
    archive = Path(('E' + chr(58) + '/archive/test.gguf'))
    source_dir = tmp_path / "stage"
    source_dir.mkdir()
    source = source_dir / "test.gguf"
    source.write_bytes(b"verified SSD weights")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {"source": str(archive), "files": {source.name: {"bytes": source.stat().st_size, "sha256": sha}}}
    (source_dir / storage.stage.META).write_text(json.dumps(metadata))
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"models": {"test": {"path": str(archive)}}}))
    overlay = tmp_path / "registry.local.json"
    overlay.write_text("{}", encoding="utf-8")
    def load_config(path):
        data = json.loads(Path(path).read_text())
        data["_machine_overlay"] = str(overlay)
        return data
    monkeypatch.setattr(storage, "load_config", load_config)
    monkeypatch.setattr(storage.local_llm, "parse_registry", lambda data: {"test": SimpleNamespace(
        path=archive, size_bytes=source.stat().st_size, sha256=None)})
    monkeypatch.setattr(storage.stage, "entries", lambda policy: [{"path": str(source_dir)}])
    policy = {**machine_policy(tmp_path), "local_ai_hot_models": str(tmp_path / "hot"), "hot_volume": tmp_path.anchor}
    return source, registry, policy


@pytest.mark.skipif(sys.platform != "win32", reason="explicit Windows SSD/HDD volumes")
def test_pin_survives_stage_eviction_without_copy(tmp_path, monkeypatch):
    source, registry, policy = fixture(tmp_path, monkeypatch)
    before = registry.read_bytes()
    result = storage.pin("test", registry=registry, policy=policy)
    target = Path(result["path"])
    assert source.samefile(target)
    source.unlink()
    assert target.read_bytes() == b"verified SSD weights"
    assert result["bytes_copied_from_hdd"] == 0
    assert registry.read_bytes() == before
    assert json.loads((tmp_path / "registry.local.json").read_text())["models"]["test"]["archive_path"] == ('E' + chr(58) + '/archive/test.gguf')


@pytest.mark.skipif(sys.platform != "win32", reason="explicit Windows SSD/HDD volumes")
def test_absent_stage_refuses_before_archive_hash(tmp_path, monkeypatch):
    _, registry, policy = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(storage.stage, "entries", lambda policy: [])
    monkeypatch.setattr(storage.local_llm, "_sha256", lambda path: pytest.fail("must not read archive"))
    with pytest.raises(ValueError, match="refusing archive read"):
        storage.pin("test", registry=registry, policy=policy)


@pytest.mark.skipif(sys.platform != "win32", reason="explicit Windows SSD/HDD volumes")
def test_corrupt_stage_is_not_pinned(tmp_path, monkeypatch):
    source, registry, policy = fixture(tmp_path, monkeypatch)
    before = registry.read_bytes()
    source.write_bytes(b"changed SSD weights!")
    with pytest.raises(ValueError, match="hash mismatch"):
        storage.pin("test", registry=registry, policy=policy)
    assert registry.read_bytes() == before
