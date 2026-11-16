"""Sleep mode stops automatic disk work before path discovery or task launch."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dev_quiet_mode as quiet
import storage_budget as budget
import storage_tiers as tiers


@pytest.mark.parametrize('raw', ['{"schema":1,"enabled":true}', 'broken',
                               '{"schema":1,"enabled":false,"defer_automatic_maintenance":true}',
                               '{"schema":1,"enabled":false,"defer_automatic_maintenance":"false"}',
                               '{"schema":true,"enabled":false}'])
@pytest.mark.parametrize('entry', [['auto', '--detach'], ['scheduled']])
def test_auto_and_nudge_stop_before_storage_discovery(tmp_path, monkeypatch, capsys, raw, entry):
    path = tmp_path / 'sleep.json'
    path.write_text(raw)
    monkeypatch.setattr(quiet, 'POLICY', path)
    def forbidden(*args, **kwargs):
        pytest.fail('Sleep mode crossed the storage/task boundary')
    monkeypatch.setattr(tiers, 'load_policy', forbidden)
    monkeypatch.setattr(tiers, 'main_root', forbidden)
    monkeypatch.setattr(budget, '_live_policy', forbidden)
    monkeypatch.setattr(budget.subprocess, 'run', forbidden)
    assert tiers.main(entry) == 0
    assert json.loads(capsys.readouterr().out) == {'status': 'deferred', 'reason': 'sleep_quiet_mode'}
    assert budget.nudge({}, reason='test') == {'status': 'quiet_deferred'}


@pytest.mark.parametrize('enabled', [False, None])
def test_disabled_or_absent_policy_preserves_normal_entry(tmp_path, monkeypatch, enabled):
    path = tmp_path / 'sleep.json'
    if enabled is not None:
        path.write_text(json.dumps({'schema': 1, 'enabled': enabled}))
    monkeypatch.setattr(quiet, 'POLICY', path)
    class ReachedNormalPath(Exception):
        pass
    def reached():
        raise ReachedNormalPath()
    monkeypatch.setattr(tiers, 'load_policy', reached)
    with pytest.raises(ReachedNormalPath):
        tiers.main(['auto'])


def test_explicit_command_reaches_existing_capacity_policy(tmp_path, monkeypatch):
    path = tmp_path / 'sleep.json'
    path.write_text('{"schema":1,"enabled":true}')
    monkeypatch.setattr(quiet, 'POLICY', path)
    class ReachedNormalPath(Exception):
        pass
    def reached():
        raise ReachedNormalPath()
    monkeypatch.setattr(tiers, 'load_policy', reached)
    with pytest.raises(ReachedNormalPath):
        tiers.main(['maintain'])


def test_non_utf8_policy_defers_gpu_and_automatic_storage(tmp_path, monkeypatch):
    path = tmp_path / 'sleep.json'
    path.write_bytes(b'\xff\xfe\x00')
    monkeypatch.setattr(quiet, 'POLICY', path)
    with pytest.raises(quiet.GPUDeferred):
        quiet.require_gpu_allowed()
    monkeypatch.setattr(tiers, 'load_policy', lambda: pytest.fail('Decoded invalid policy as permission'))
    assert tiers.main(['auto']) == 0
    assert budget.nudge({}, reason='test') == {'status': 'quiet_deferred'}


def test_awake_low_load_policy_allows_gpu_and_explicit_hdd_work(tmp_path, monkeypatch):
    path = tmp_path / 'awake.json'
    path.write_text('{"schema":1,"enabled":false,"defer_automatic_maintenance":true}')
    monkeypatch.setattr(quiet, 'POLICY', path)
    quiet.require_gpu_allowed()
    assert quiet.quiet_requested()
    class ReachedExplicitWork(Exception):
        pass
    def reached():
        raise ReachedExplicitWork()
    monkeypatch.setattr(tiers, 'load_policy', reached)
    with pytest.raises(ReachedExplicitWork):
        tiers.main(['maintain'])
