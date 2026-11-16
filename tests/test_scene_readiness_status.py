"""Read-only native readiness and safe windowed-EXE dispatch, with no real inference."""
import json
import pytest
import lss_gui
from lss import paths
from lss.bgm import scene_native, scene_runtime
from lss.core import gpu_lease


@pytest.fixture
def empty_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'is_frozen', lambda: True)
    monkeypatch.setattr(scene_native, 'native_root', lambda: tmp_path / 'missing-native')
    monkeypatch.delenv('LSS_SCENE_BACKEND', raising=False)
    def forbidden(*a, **k): pytest.fail('read-only status crossed an execution boundary')
    monkeypatch.setattr(scene_native, 'install', forbidden)
    monkeypatch.setattr(scene_runtime, 'create_models', forbidden)
    monkeypatch.setattr(gpu_lease, 'acquire', forbidden)
    monkeypatch.setattr(lss_gui, '_configure_runtime_caches', forbidden)
    monkeypatch.setattr(lss_gui, '_run_scene_naming', forbidden)
    return tmp_path


def test_empty_frozen_is_supported_but_not_ready_without_any_write(empty_frozen):
    before=set(empty_frozen.iterdir())
    result=scene_runtime.read_only_status()
    assert result['status']=='observed' and result['available'] is False
    assert result['native_ready'] is False and result['native_bootstrap_supported'] is True
    assert result['native_problem_codes']==['no_manifest']
    assert set(empty_frozen.iterdir())==before


def test_bad_manifest_is_unknown_and_exception_text_is_not_exposed(empty_frozen, monkeypatch):
    def bad(*a): raise ValueError('secret should never enter diagnostics')
    monkeypatch.setattr(scene_native, '_manifest_problems', bad)
    result=scene_runtime.read_only_status()
    assert result['status']=='unresolved' and result['available'] is None
    assert 'secret' not in json.dumps(result)


def test_problem_paths_are_not_reported(empty_frozen, monkeypatch):
    monkeypatch.setattr(scene_native, '_manifest_problems', lambda *a:['changed:private-location/file'])
    result=scene_runtime.read_only_status()
    assert result['native_problem_codes']==['changed']
    assert 'private-location' not in json.dumps(result)


def test_windowed_report_requires_a_new_file_and_never_initializes_gui(empty_frozen):
    report=empty_frozen/'readiness.json'
    assert lss_gui.main(['--scene-naming-status', str(report)])==0
    before=report.read_bytes()
    assert lss_gui.main(['--scene-naming-status', str(report)])==1
    assert report.read_bytes()==before
    assert lss_gui.main(['--scene-naming-status'])==1
    assert lss_gui.main(['--scene-naming-status',str(report),'extra'])==1


def test_oversized_manifest_does_not_enter_readiness_reader(empty_frozen, monkeypatch):
    root=empty_frozen/'native';root.mkdir()
    (root/scene_native.MANIFEST).write_bytes(b' '*(256*1024+1))
    monkeypatch.setattr(scene_native,'native_root',lambda:root)
    monkeypatch.setattr(scene_native,'_manifest_problems',lambda *a:pytest.fail('large manifest read'))
    result=scene_runtime.read_only_status()
    assert result['available'] is None and result['native_problem_codes']==['manifest_too_large']


def test_changing_readiness_is_not_certified_as_ready(empty_frozen, monkeypatch):
    monkeypatch.setattr(scene_runtime, 'backend', lambda: 'native')
    result=scene_runtime.read_only_status()
    assert result['status']=='unresolved' and result['available'] is None
    assert result['native_problem_codes']==['readiness_changed_during_observation']
