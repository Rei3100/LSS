from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import lss_gui
from lss import paths
from lss.bgm import scene_job, scene_native, scene_runtime, scene_names
from lss.core import gpu_lease


def _fresh(monkeypatch, tmp_path):
    root = tmp_path / 'native'
    monkeypatch.setenv('LSS_SCENE_NATIVE_DIR', str(root))
    monkeypatch.delenv('LSS_SCENE_BACKEND', raising=False)
    monkeypatch.setattr(paths, 'is_frozen', lambda: True)
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'original source')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(scene_job, '_published', lambda *a: [{'asset_id': 'bgm-cue:one', 'pcm_sha256': 'a' * 64}])
    monkeypatch.setattr(scene_names, 'cue_record', lambda *a: None)
    monkeypatch.setattr(scene_job, '_pending_retry_with_budget', lambda *a, **k: None)
    monkeypatch.setattr(scene_job, '_next_transient_retry_after', lambda *a, **k: None)
    monkeypatch.setattr(scene_native, '_bootstrap_capacity', lambda *a: None)
    monkeypatch.setattr(gpu_lease, 'acquire', lambda *a, **k: pytest.fail('GPU lease during acquisition'))
    return root, source, digest


def test_frozen_start_reaches_worker_bootstrap_then_naming(monkeypatch, tmp_path):
    root, source, digest = _fresh(monkeypatch, tmp_path)
    state = tmp_path / 'state'
    ready = {'value': False}
    events = []
    monkeypatch.setattr(scene_native, 'ready', lambda *a: ready['value'])
    def install(path):
        events.append('install')
        ready['value'] = True
    monkeypatch.setattr(scene_native, 'install', install)
    monkeypatch.setattr(scene_job, 'name_source_checked', lambda *a: events.append('name') or {'named': 1})
    monkeypatch.setattr(lss_gui, '_configure_runtime_caches', lambda: None)
    monkeypatch.setattr(lss_gui, '_ensure_worker_streams', lambda: None)
    monkeypatch.setattr(paths, 'resource_root', lambda: tmp_path)
    def launch(command, **kwargs):
        assert command[1] == '--scene-naming'
        assert kwargs['stdin'] == subprocess.DEVNULL
        assert lss_gui.main(command[1:]) == 0
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    assert scene_job.start_in_background(state, source, digest)['state'] == 'started'
    assert events == ['install', 'name']
    assert scene_runtime.available()
    scene_native.ensure_ready(root)
    assert events == ['install', 'name']


def test_invalid_source_prevents_acquisition_and_preserves_existing_state(monkeypatch, tmp_path):
    root, source, digest = _fresh(monkeypatch, tmp_path)
    state = tmp_path / 'state'; state.mkdir()
    accepted = state / 'accepted.json'; accepted.write_text('{"kept":true}')
    monkeypatch.setattr(scene_native, 'install', lambda *a: pytest.fail('must not acquire'))
    monkeypatch.setattr(scene_job, 'name_source_checked', lambda *a: pytest.fail('must not infer'))
    source.write_bytes(b'different source')
    result = scene_job.run_background_worker(state, source, digest)
    assert result['state'] == 'bootstrap_failed' and result['message'] == 'source_identity_mismatch'
    assert accepted.read_text() == '{"kept":true}' and not root.exists()


def test_no_pending_or_forced_unsupported_backend_does_not_acquire(monkeypatch, tmp_path):
    root, source, digest = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('must not start child'))
    monkeypatch.setattr(scene_native, 'install', lambda *a: pytest.fail('must not acquire'))
    monkeypatch.setattr(scene_job, '_published', lambda *a: [])
    assert scene_job.start_in_background(tmp_path / 'state', source, digest)['state'] == 'current'
    monkeypatch.setenv('LSS_SCENE_BACKEND', 'transformers')
    assert scene_job.start_in_background(tmp_path / 'state', source, digest)['state'] == 'not_available'
    assert not root.exists()


def test_acquisition_failure_preserves_state_and_is_throttled(monkeypatch, tmp_path):
    root, source, digest = _fresh(monkeypatch, tmp_path)
    calls = []
    def fail(path):
        calls.append('download')
        raise OSError('offline')
    monkeypatch.setattr(scene_native, 'install', fail)
    monkeypatch.setattr(scene_job, 'name_source_checked', lambda *a: pytest.fail('must not infer'))
    first = scene_job.run_background_worker(tmp_path / 'state', source, digest)
    second = scene_job.run_background_worker(tmp_path / 'state', source, digest)
    assert first['state'] == second['state'] == 'bootstrap_failed'
    assert calls == ['download'] and second['message'] == 'scene_native_bootstrap_retry_pending'
    assert json.loads((root / 'bootstrap-status.json').read_text())['retry_after'] > 0


def test_concurrent_workers_share_one_install(monkeypatch, tmp_path):
    root, _, _ = _fresh(monkeypatch, tmp_path)
    ready = {'value': False}
    started, release = threading.Event(), threading.Event()
    calls, errors = [], []
    monkeypatch.setattr(scene_native, 'ready', lambda *a: ready['value'])
    def install(path):
        calls.append('install'); started.set()
        assert release.wait(5)
        ready['value'] = True
    monkeypatch.setattr(scene_native, 'install', install)
    def run():
        try: scene_native.ensure_ready(root)
        except Exception as exc: errors.append(exc)
    first = threading.Thread(target=run); first.start(); assert started.wait(5)
    second = threading.Thread(target=run); second.start()
    release.set(); first.join(10); second.join(10)
    assert not first.is_alive() and not second.is_alive() and not errors and calls == ['install']


def test_bootstrap_refuses_separation_cache(monkeypatch, tmp_path):
    from lss.harness import fetch
    cache = tmp_path / 'separation-cache'
    monkeypatch.setattr(fetch, 'model_cache_dir', lambda: cache)
    monkeypatch.setattr(scene_native, 'install', lambda *a: pytest.fail('must not acquire'))
    with pytest.raises(scene_native.SceneModelsError, match='inside_separation_cache'):
        scene_native.ensure_ready(cache / 'scene-native')
    assert not cache.exists()


def test_forced_missing_backend_worker_does_not_infer(monkeypatch, tmp_path):
    _, source, digest = _fresh(monkeypatch, tmp_path)
    monkeypatch.setenv('LSS_SCENE_BACKEND', 'transformers')
    monkeypatch.setattr(scene_job, 'name_source_checked', lambda *a: pytest.fail('must not infer'))
    assert scene_job.run_background_worker(tmp_path / 'state', source, digest)['state'] == 'not_available'


def test_startup_recovery_handoff_bootstraps_only_eligible_source(monkeypatch, tmp_path):
    from lss.bgm import current_best
    root, source, digest = _fresh(monkeypatch, tmp_path)
    state = tmp_path / 'state'
    directory = state / current_best.DIRECTORY
    directory.mkdir(parents=True)
    (directory / (digest + '.json')).write_text('{}')
    monkeypatch.setattr(scene_job, '_current_best_run_sources', lambda _: ({digest: [source]}, []))
    ready = {'value': False}
    events = []
    monkeypatch.setattr(scene_native, 'ready', lambda *a: ready['value'])
    def install(path):
        events.append('install'); ready['value'] = True
    monkeypatch.setattr(scene_native, 'install', install)
    monkeypatch.setattr(scene_job, 'name_source_checked', lambda *a: events.append('name') or {'named': 1})
    monkeypatch.setattr(lss_gui, '_configure_runtime_caches', lambda: None)
    monkeypatch.setattr(paths, 'resource_root', lambda: tmp_path)
    def launch(command, **kwargs):
        assert command[1] == '--scene-naming-recovery'
        assert lss_gui.main(command[1:]) == 0
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    assert scene_job.start_recovery_in_background(state)['state'] == 'started'
    assert events == ['install', 'name']


def test_unknown_forced_backend_never_silently_falls_back(monkeypatch, tmp_path):
    _, source, digest = _fresh(monkeypatch, tmp_path)
    monkeypatch.setenv("LSS_SCENE_BACKEND", "unsupported")
    monkeypatch.setattr(scene_runtime, "native_available", lambda: True)
    assert scene_runtime.backend() is None
    assert scene_job.start_in_background(tmp_path / "state", source, digest)["state"] == "not_available"


def test_low_disk_fails_before_download(monkeypatch, tmp_path):
    root, source, digest = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(scene_native, '_bootstrap_capacity', lambda _: (_ for _ in ()).throw(
        scene_native.SceneModelsError('scene_native_bootstrap_insufficient_disk_space')))
    monkeypatch.setattr(scene_native, 'install', lambda *a: pytest.fail('must not download'))
    monkeypatch.setattr(scene_job, 'name_source_checked', lambda *a: pytest.fail('must not infer'))
    result = scene_job.run_background_worker(tmp_path / 'state', source, digest)
    assert result['state'] == 'bootstrap_failed'
    assert result['message'] == 'scene_native_bootstrap_insufficient_disk_space'


def test_startup_recovery_with_no_pending_cue_never_bootstraps(monkeypatch, tmp_path):
    _, source, digest = _fresh(monkeypatch, tmp_path)
    monkeypatch.setattr(scene_job, '_current_best_run_sources', lambda _: ({digest: [source]}, []))
    monkeypatch.setattr(scene_job, '_published', lambda *a: [])
    monkeypatch.setattr(scene_native, 'install', lambda *a: pytest.fail('no pending cue: no acquisition'))
    result = scene_job.run_current_best_recovery(tmp_path / 'state')
    assert result['started'] == 0 and result['state'] == 'recovery_complete'
