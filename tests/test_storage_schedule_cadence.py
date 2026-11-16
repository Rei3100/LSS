"""Cadence falsifiers: healthy ticks never enter expensive disk discovery."""
import json
from pathlib import Path
import sys

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import storage_tiers as tiers
import storage_layout
import storage_retention
import storage_inventory

REAL_CLEAN_HOT = tiers.clean_hot
REAL_REFRESH = storage_inventory.refresh


@pytest.fixture
def scheduled_case(tmp_path, monkeypatch):
    policy = {**machine_policy(tmp_path), 'hot_volume': ('C' + chr(58) + '\\'), 'fast_volumes': [('C' + chr(58) + '\\'), ('D' + chr(58) + '\\')],
              'start_free_gb': 60, 'local_state_root': str(tmp_path / 'state'),
              'volume_limits': {('C' + chr(58) + '\\'): {'start_free_gb': 60}, ('D' + chr(58) + '\\'): {'start_free_gb': 120}},
              'scheduled_cadence': {'ssd_every_hours': 24, 'hdd_every_hours': 168}}
    policy['model_stores'] = [('C' + chr(58) + '/synthetic-protected-models')]
    clock, free, calls = [1800000000.0], [500.0], []
    monkeypatch.setattr(tiers.time, 'time', lambda: clock[0])
    monkeypatch.setattr(tiers, 'free_gb', lambda volume: free[0])
    monkeypatch.setattr(tiers, 'uncommitted_storage_code', lambda: [])
    monkeypatch.setattr(tiers, 'background_priority', lambda: None)
    monkeypatch.setattr(tiers, 'cleanups_due', lambda p: calls.append('cleanup_discovery') or [])
    monkeypatch.setattr(tiers, 'clean_hot', lambda *a, **k: calls.append('ssd'))
    monkeypatch.setattr(storage_layout, 'maintain', lambda *a, **k: calls.append('layout'))
    monkeypatch.setattr(storage_retention, 'maintain', lambda *a, **k: calls.append('retention'))
    monkeypatch.setattr(tiers, 'maintain', lambda *a, **k: calls.append('tiers'))
    monkeypatch.setattr(storage_inventory, 'refresh', lambda *a, **k: calls.append('inventory'))
    return policy, clock, free, calls, tmp_path


def run(case):
    policy, _, _, _, root = case
    return tiers.scheduled(policy, root)


def test_healthy_not_due_tick_never_discovers_targets(scheduled_case):
    case = scheduled_case
    assert run(case) == 0
    assert {'ssd', 'layout', 'retention', 'tiers', 'inventory'} <= set(case[3])
    case[3].clear()
    assert run(case) == 0
    assert case[3] == []


def test_ssd_due_does_not_enter_hdd_stages(scheduled_case):
    case = scheduled_case
    assert run(case) == 0
    case[1][0] += 25 * 3600
    case[3].clear()
    assert run(case) == 0
    assert case[3] == ['cleanup_discovery', 'ssd']


def test_pressure_relief_keeps_hdd_cold(scheduled_case, monkeypatch):
    case = scheduled_case
    assert run(case) == 0
    case[2][0] = 10
    case[3].clear()
    def relieve(*args, **kwargs):
        case[3].append('ssd')
        case[2][0] = 500
    monkeypatch.setattr(tiers, 'clean_hot', relieve)
    assert run(case) == 0
    assert case[3] == ['cleanup_discovery', 'ssd']


def test_unrelieved_pressure_allows_archival(scheduled_case):
    case = scheduled_case
    assert run(case) == 0
    case[2][0] = 10
    case[3].clear()
    assert run(case) == 0
    assert {'ssd', 'layout', 'retention', 'tiers', 'inventory'} <= set(case[3])


@pytest.mark.parametrize('stage', ['ssd', 'layout'])
def test_failure_keeps_its_group_due(scheduled_case, monkeypatch, stage):
    case = scheduled_case
    def fail(*args, **kwargs):
        case[3].append(stage)
        raise OSError('known failure')
    if stage == 'ssd':
        monkeypatch.setattr(tiers, 'clean_hot', fail)
    else:
        monkeypatch.setattr(storage_layout, 'maintain', fail)
    assert run(case) == 1
    case[3].clear()
    assert run(case) == 1
    assert stage in case[3]
    record = json.loads((tiers.local_state(case[0]) / 'last-scheduled.json').read_text())
    assert record['status'] == 'failed'
    assert any('known failure' in error['error'] for error in record['errors'])


def test_pending_recovery_enters_hdd_stage_before_cadence(scheduled_case):
    case = scheduled_case
    assert run(case) == 0
    (tiers.local_state(case[0]) / 'journal.json').write_text('{}')
    case[3].clear()
    assert run(case) == 0
    assert {'layout', 'retention', 'tiers', 'inventory'} <= set(case[3])


def test_missing_cadence_preserves_existing_schedule(scheduled_case):
    case = scheduled_case
    case[0].pop('scheduled_cadence')
    assert run(case) == 0
    case[3].clear()
    assert run(case) == 0
    assert {'ssd', 'layout', 'retention', 'tiers', 'inventory'} <= set(case[3])


def test_unavailable_process_inventory_never_stamps_ssd_success(scheduled_case, monkeypatch):
    case = scheduled_case
    attempts = []
    monkeypatch.setattr(tiers, 'clean_hot', REAL_CLEAN_HOT)
    monkeypatch.setattr(tiers, 'process_texts', lambda: attempts.append(True) or ['<process list unavailable>'])
    assert run(case) == 1
    assert run(case) == 1
    assert len(attempts) == 2
    record = json.loads((tiers.local_state(case[0]) / 'last-scheduled.json').read_text())
    assert record['status'] == 'failed'
    assert any(error['step'] == 'clean_hot' for error in record['errors'])


def test_inventory_walk_error_is_retained_and_retried(scheduled_case, monkeypatch):
    case = scheduled_case
    root = case[4] / 'inventory'
    root.mkdir()
    case[0]['dependency_inventory'] = {'roots': [str(root)], 'every_hours': 168}
    attempts = []
    def denied(path, *, onerror):
        attempts.append(path)
        onerror(PermissionError('known inventory denial'))
        return iter(())
    monkeypatch.setattr(storage_inventory, 'refresh', REAL_REFRESH)
    monkeypatch.setattr(storage_inventory.os, 'walk', denied)
    assert run(case) == 1
    state = tiers.local_state(case[0])
    assert json.loads((state / 'dependency-inventory.json').read_text())['errors']
    assert run(case) == 1
    assert len(attempts) == 2
    record = json.loads((state / 'last-scheduled.json').read_text())
    assert any(error['step'] == 'dependency_inventory' for error in record['errors'])


def test_ssd_cleanup_does_not_resolve_cold_model_roots(scheduled_case, monkeypatch):
    case = scheduled_case
    case[0]['model_stores'] = [('E' + chr(58) + '/cold/models'), ('D' + chr(58) + '/hot/models')]
    checked, received = [], []
    def link(path):
        checked.append(str(path))
        assert path.anchor.casefold() != ('e' + chr(58) + '\\')
        return False
    monkeypatch.setattr(tiers, 'is_link', link)
    monkeypatch.setattr(tiers, 'clean_hot', lambda p, **kwargs: received.extend(p['model_stores']))
    assert run(case) == 0
    assert received == [('D' + chr(58) + '/hot/models')]


def test_hourly_group_runs_on_every_tick(scheduled_case):
    case = scheduled_case
    case[0]['scheduled_cadence'] = {'ssd_every_hours': 1, 'hdd_every_hours': 24}
    assert run(case) == 0
    case[1][0] += 3600 - 60
    case[3].clear()
    assert run(case) == 0
    assert case[3] == ['cleanup_discovery', 'ssd']
