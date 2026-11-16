import json
from datetime import datetime, timezone
import pytest
from scripts import dev_quiet_mode as quiet


def test_explicit_timed_gpu_pause_expires_at_its_deadline(tmp_path, monkeypatch):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'schema': 1, 'enabled': True,
                               'gpu_pause_until_utc': '2026-10-27T17:43:09+00:00'}))
    monkeypatch.setattr(quiet, 'POLICY', path)
    monkeypatch.setattr(quiet, 'utc_now', lambda: datetime(2026, 10, 27, 17, 43, 8, tzinfo=timezone.utc))
    with pytest.raises(quiet.GPUDeferred):
        quiet.require_gpu_allowed()
    monkeypatch.setattr(quiet, 'utc_now', lambda: datetime(2026, 10, 27, 17, 43, 9, tzinfo=timezone.utc))
    quiet.require_gpu_allowed()


@pytest.mark.parametrize('deadline', ['bad', '2026-10-27T17:43:09', True, 0])
def test_invalid_timed_pause_cannot_enable_gpu(tmp_path, monkeypatch, deadline):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'schema': 1, 'enabled': True, 'gpu_pause_until_utc': deadline}))
    monkeypatch.setattr(quiet, 'POLICY', path)
    with pytest.raises(quiet.GPUDeferred):
        quiet.require_gpu_allowed()


def test_absent_and_explicit_disabled_allow_gpu(tmp_path, monkeypatch):
    path = tmp_path/'policy.json'
    monkeypatch.setattr(quiet, 'POLICY', path)
    quiet.require_gpu_allowed()
    path.write_text(json.dumps({'schema':1,'enabled':False}))
    quiet.require_gpu_allowed()


@pytest.mark.parametrize('raw', ['broken', '[]', '{"schema":1,"enabled":"false"}', '{"schema":1,"enabled":true}', '{"schema":true,"enabled":false}'])
def test_enabled_or_unreadable_policy_cannot_start_gpu(tmp_path, monkeypatch, raw):
    path=tmp_path/'policy.json';path.write_text(raw)
    monkeypatch.setattr(quiet,'POLICY',path)
    with pytest.raises(quiet.GPUDeferred): quiet.require_gpu_allowed()


def test_serving_defers_before_any_model_or_gpu_access(monkeypatch):
    import local_llm
    import dev_quiet_mode
    def refuse(): raise dev_quiet_mode.GPUDeferred('sleep')
    monkeypatch.setattr(dev_quiet_mode,'require_gpu_allowed',refuse)
    monkeypatch.setattr(local_llm,'get_model',lambda *a:pytest.fail('model lookup before sleep guard'))
    with pytest.raises(dev_quiet_mode.GPUDeferred):
        with local_llm.serving('unused-model'): pytest.fail('served')


def test_gpu_entry_defers_before_storage_or_launch(monkeypatch):
    import gpu_run
    import dev_quiet_mode
    def refuse(): raise dev_quiet_mode.GPUDeferred('sleep')
    monkeypatch.setattr(dev_quiet_mode,'require_gpu_allowed',refuse)
    monkeypatch.setattr(gpu_run.subprocess,'run',lambda *a,**k:pytest.fail('child before sleep guard'))
    assert gpu_run.main(['--','never-launch.exe']) == 3


def test_cpu_route_does_not_need_gpu_permission(monkeypatch):
    import local_llm
    import dev_quiet_mode
    monkeypatch.setattr(dev_quiet_mode,'require_gpu_allowed',lambda:pytest.fail('CPU checked GPU permission'))
    class ReachedCPUPath(Exception): pass
    def model(*a): raise ReachedCPUPath()
    monkeypatch.setattr(local_llm,'get_model',model)
    with pytest.raises(ReachedCPUPath):
        with local_llm.serving('unused',cpu=True): pytest.fail('served')


def test_draft_queue_defers_before_admission_or_large_file_hashing(tmp_path, monkeypatch):
    import local_ai_queue
    import dev_quiet_mode
    def refuse(): raise dev_quiet_mode.GPUDeferred('sleep')
    monkeypatch.setattr(dev_quiet_mode,'require_gpu_allowed',refuse)
    def forbidden(*a,**k): pytest.fail('large-work boundary crossed')
    with pytest.raises(dev_quiet_mode.GPUDeferred):
        local_ai_queue.run_queue([],object(),tmp_path,admit=forbidden,identify=forbidden)
