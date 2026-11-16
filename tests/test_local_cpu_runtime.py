"""Explicit CPU runtime trials cannot redirect GPU jobs or hash before refusal."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_llm as llm


def test_default_runtime_does_not_change_launcher(monkeypatch):
    import dev_hot_paths
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: pytest.fail('Default touched override'))
    assert llm._cpu_runtime_launcher(None, cpu=False) is None
    assert llm._cpu_runtime_launcher(None, cpu=True) is None


def test_gpu_override_refuses_before_model_access(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, 'get_model', lambda *a: pytest.fail('Looked up model'))
    with pytest.raises(llm.RegistryError, match='requires CPU mode'):
        with llm.serving('unused', runtime_dir=tmp_path):
            pytest.fail('Started a server')


def test_missing_cpu_runtime_refuses_before_model_access(tmp_path, monkeypatch):
    import dev_hot_paths
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: None)
    monkeypatch.setattr(llm, 'get_model', lambda *a: pytest.fail('Hashed or looked up model'))
    with pytest.raises(llm.ServerFailed, match='missing'):
        with llm.serving('unused', cpu=True, runtime_dir=tmp_path):
            pytest.fail('Started a server')


def test_runtime_storage_refusal_precedes_file_probe(tmp_path, monkeypatch):
    import dev_hot_paths
    class Refused(RuntimeError):
        pass
    def reject(*args):
        raise Refused()
    monkeypatch.setattr(dev_hot_paths, 'require_hot', reject)
    original = Path.is_file
    def is_file(path):
        if path.name == 'llama-server.exe':
            pytest.fail('Probed file before hot-volume check')
        return original(path)
    monkeypatch.setattr(Path, 'is_file', is_file)
    with pytest.raises(Refused):
        llm._cpu_runtime_launcher(tmp_path, cpu=True)


def test_existing_cpu_executable_is_selected_without_execution(tmp_path, monkeypatch):
    import dev_hot_paths
    calls = []
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: calls.append(a))
    exe = tmp_path/'llama-server.exe'
    exe.write_bytes(b'fixture only, never execute')
    assert llm._cpu_runtime_launcher(tmp_path, cpu=True) == [str(exe.resolve())]
    assert len(calls) == 1
