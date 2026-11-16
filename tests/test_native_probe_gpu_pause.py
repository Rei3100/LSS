"""A queued probe waits before importing or launching any naming worker."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import probe_native_packaged_first_use as probe
import dev_quiet_mode


def test_queued_probe_defers_before_source_import_or_exe_launch(tmp_path, monkeypatch):
    output = tmp_path / 'result'
    output.mkdir()
    monkeypatch.setattr(probe, 'OUT', output)
    monkeypatch.setenv('LSS_NATIVE_FIRST_USE_SUPERVISED', str(output))
    def defer():
        raise dev_quiet_mode.GPUDeferred('explicit one-hour pause')
    class ReachedWait(Exception):
        pass
    def wait(seconds):
        assert seconds == 30
        raise ReachedWait()
    monkeypatch.setattr(dev_quiet_mode, 'require_gpu_allowed', defer)
    monkeypatch.setattr(probe.time, 'sleep', wait)
    monkeypatch.setattr(probe.importlib.util, 'spec_from_file_location',
                        lambda *a, **k: pytest.fail('Source helper loaded during GPU pause'))
    with pytest.raises(ReachedWait):
        probe.worker()
    assert json.loads((output / 'progress.json').read_text())['gpu_used'] is False


def test_failed_readonly_exe_proof_cannot_launch_naming_worker(tmp_path, monkeypatch):
    import batch
    import dev_hot_paths
    output = tmp_path / 'out'
    proof = tmp_path / 'proof'
    proof.mkdir()
    (proof / 'supervisor-result.json').write_text(json.dumps({'status': 'failed'}))
    checks = tmp_path / 'release-evidence/native-resume-probe-contracts-20261028-r2'
    checks.mkdir(parents=True)
    (checks / 'result.json').write_text(json.dumps({'status': 'passed'}))
    provenance = tmp_path / 'release-evidence/native-resume-provenance-contracts-20261028'
    provenance.mkdir(parents=True)
    (provenance / 'result.json').write_text(json.dumps({'status': 'passed'}))
    offline = tmp_path / 'release-evidence/native-offline-resume-contracts-20261028-r3'
    offline.mkdir(parents=True)
    (offline / 'result.json').write_text(json.dumps({'status': 'passed'}))
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    monkeypatch.setattr(probe, 'wait_for_restored_model', lambda: None)
    monkeypatch.setattr(probe, 'OUT', output)
    monkeypatch.setattr(probe, 'READONLY', proof / 'result.json')
    monkeypatch.setattr(sys, 'argv', ['probe'])
    monkeypatch.setattr(dev_quiet_mode, 'require_gpu_allowed', lambda: None)
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: None)
    monkeypatch.setattr(batch, 'run_command', lambda *a, **k: pytest.fail('naming worker launched after failed proof'))
    assert probe.main() == 1
    result = json.loads((output / 'supervisor-result.json').read_text())
    assert result['status'] == 'failed' and result['error_type'] == 'RuntimeError'


@pytest.mark.parametrize('override', [None, 'native'])
def test_packaged_probe_preserves_private_env_without_forcing_backend(override):
    env = {'LSS_STATE_DIR': 'private-state'}
    if override:
        env['LSS_SCENE_BACKEND'] = override
    seen, record = [], {}
    marker = object()
    def factory(*args, **kwargs):
        seen.append(kwargs['env'])
        return marker
    child = probe.product_default_child(factory, record)
    assert child(['owned.exe'], env=env) is marker
    assert seen == [{'LSS_STATE_DIR': 'private-state'}]
    assert record['worker_backend_override_absent'] is True
    assert env.get('LSS_SCENE_BACKEND') == override


def test_packaged_probe_rejects_an_unexpected_helper_backend():
    child = probe.product_default_child(lambda *a, **k: pytest.fail('unexpected worker launch'), {})
    with pytest.raises(ValueError, match='Unexpected helper'):
        child(['owned.exe'], env={'LSS_SCENE_BACKEND': 'transformers'})


@pytest.mark.parametrize('key', [*probe.PROTECTED, 'names'])
def test_resume_rejects_each_changed_protected_fixture_field(key):
    expected = {name: 'original-' + name for name in (*probe.PROTECTED, 'names')}
    before = {**expected, 'native_ready': False, key: 'changed'}
    with pytest.raises(ValueError, match='Protected interrupted'):
        probe.verify_resume_baseline(before, expected)


def test_resume_requires_incomplete_cache_and_accepts_unchanged_guards():
    expected = {name: 'original-' + name for name in (*probe.PROTECTED, 'names')}
    probe.verify_resume_baseline({**expected, 'native_ready': False}, expected)
    for value in (True, None, 'unknown'):
        with pytest.raises(ValueError, match='unfinished'):
            probe.verify_resume_baseline({**expected, 'native_ready': value}, expected)


def historical_fixture(tmp_path, monkeypatch):
    import importlib.util
    old = tmp_path / 'old' / 'LSS.exe'
    ffmpeg = old.parent / '_internal/bin/ffmpeg.exe'
    ffmpeg.parent.mkdir(parents=True)
    old.write_bytes(b'old exe')
    ffmpeg.write_bytes(b'old ffmpeg')
    spec = importlib.util.spec_from_file_location('fixture_helper', probe.HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    exe_hash = helper.digest(old)
    manifest = {'source_commit': probe.OLD_COMMIT, 'exe_sha256': exe_hash,
                'files_sha256': {'_internal/bin/ffmpeg.exe': {'sha256': helper.digest(ffmpeg)}}}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(probe, 'OLD_EXE', old)
    monkeypatch.setattr(probe, 'OLD_FROZEN', manifest_path)
    original = {'source_commit': probe.OLD_COMMIT, 'exe_sha256': exe_hash,
                'worker_probe': {'exe': str(old), 'exe_sha256': exe_hash}}
    return helper, old, ffmpeg, original


def test_historical_provenance_allows_only_verified_exact_files(tmp_path, monkeypatch):
    helper, old, ffmpeg, original = historical_fixture(tmp_path, monkeypatch)
    case = tmp_path / 'case'
    case.mkdir()
    record = case / 'record.json'
    new = tmp_path / 'new.exe'
    new.write_bytes(b'new')
    record.write_text(json.dumps([str(old), str(ffmpeg), str(new)]))
    validate = helper.validate_tree
    with probe.historical_provenance(helper, original):
        helper.validate_tree(case, allowed_external=[new])
        record.write_text(json.dumps([str(old), str(tmp_path / 'outside')]))
        with pytest.raises(ValueError, match='outside path'):
            helper.validate_tree(case, allowed_external=[new])
    assert helper.validate_tree is validate


@pytest.mark.parametrize('corrupt', ['exe', 'ffmpeg'])
def test_bad_historical_hash_blocks_before_worker(tmp_path, monkeypatch, corrupt):
    helper, old, ffmpeg, original = historical_fixture(tmp_path, monkeypatch)
    (old if corrupt == 'exe' else ffmpeg).write_bytes(b'changed')
    helper.snapshot = lambda *args: pytest.fail('snapshot reached after failed proof')
    helper.launch_worker = lambda *args: pytest.fail('worker launched after failed proof')
    with pytest.raises(ValueError, match='provenance mismatch'):
        with probe.historical_provenance(helper, original):
            helper.launch_worker()


def test_offline_model_budget_requires_exact_verified_immutable_stamp():
    from restore_native_model_offline import TARGET, SIZE, SHA
    observed = {'bytes': SIZE, 'mtime_ns': 1, 'file_id': 2, 'volume_id': 3}
    proof = {'status': 'verified', 'complete_model_verified': True, 'target': str(TARGET),
             'expected_bytes': SIZE, 'expected_sha256': SHA, 'copied_sha256': SHA,
             'target_after': observed.copy()}
    probe.verify_restore_receipt(proof, observed, True)
    for key, bad in [('status', 'running'), ('complete_model_verified', None),
                     ('copied_sha256', 'wrong'), ('target', 'C:/outside.gguf')]:
        with pytest.raises(ValueError):
            probe.verify_restore_receipt({**proof, key: bad}, observed, True)
    with pytest.raises(ValueError):
        probe.verify_restore_receipt(proof, {**observed, 'file_id': 9}, True)
    with pytest.raises(ValueError):
        probe.verify_restore_receipt(proof, observed, False)
