"""Scheduler ownership, unattended maintenance and hot-asset falsifiers."""
import json
import sys
import time
from pathlib import Path

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import storage_tiers as tiers


@pytest.fixture(autouse=True)
def storage_code_is_reviewed(monkeypatch):
    # Dirty-checkout refusal has its own tests; these cases exercise scheduling.
    monkeypatch.setattr(tiers, 'uncommitted_storage_code', lambda: [])


def policy(tmp_path):
    return {**machine_policy(tmp_path), 'local_state_root': str(tmp_path / 'state'), 'hot_volume': str(tmp_path.anchor),
            'start_free_gb': 100, 'hot_directory_names': ['venv', '.venv']}


def test_session_auto_cannot_launch_cleanup(tmp_path, monkeypatch):
    import dev_quiet_mode
    monkeypatch.setattr(dev_quiet_mode, 'quiet_requested', lambda: False)
    monkeypatch.setattr(tiers, 'load_policy', lambda: dict(policy(tmp_path), scheduler_only=True))
    monkeypatch.setattr(tiers, 'main_root', lambda: tmp_path)
    monkeypatch.setattr(tiers, 'cleanups_due', lambda *_: pytest.fail('agent launched cleanup'))
    assert tiers.main(['auto', '--detach']) == 0


def test_scheduler_runs_without_agent_and_records_partial_failure(tmp_path, monkeypatch):
    p = policy(tmp_path)
    calls = []
    monkeypatch.setattr(tiers, 'background_priority', lambda: None)
    monkeypatch.setattr(tiers, 'free_gb', lambda *_: 35)
    monkeypatch.setattr(tiers, 'developing', lambda *_: pytest.fail('must not depend on agent'))
    monkeypatch.setattr(tiers, 'cleanups_due', lambda *_: [{'name': 'broken'}])
    def broken(*a, **kw):
        raise RuntimeError('tool failed')
    monkeypatch.setattr(tiers, 'run_cleanup', broken)
    monkeypatch.setattr(tiers, 'clean_hot', lambda *a, **kw: calls.append('hot'))
    monkeypatch.setattr(tiers, 'maintain', lambda *a, **kw: calls.append('tiers'))
    assert tiers.scheduled(p, tmp_path) == 1
    assert calls == ['hot', 'tiers']
    receipt = json.loads((tmp_path / 'state/last-scheduled.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['errors'] == [{'step': 'broken', 'error': 'tool failed'}]


def test_failed_tool_is_not_stamped_as_success(tmp_path, monkeypatch):
    p = policy(tmp_path)
    monkeypatch.setattr(tiers, 'free_gb', lambda *_: 35)
    monkeypatch.setattr(tiers.subprocess, 'run', lambda *a, **kw: type('R', (), {'returncode': 1})())
    with pytest.raises(RuntimeError):
        tiers.run_cleanup(p, {'name': 'tool', 'commands': ['false']}, log=lambda *_: None)
    assert not tiers._stamp(p, 'tool').exists()
    assert list((tmp_path / 'state/receipts').glob('cleanup-tool-*.json'))


def test_overlapping_scheduler_never_mutates(tmp_path, monkeypatch):
    p = policy(tmp_path)
    monkeypatch.setattr(tiers, 'cleanups_due', lambda *_: pytest.fail('overlap performed work'))
    with tiers.Lock(tiers.local_state(p) / '.scheduled.lock'):
        assert tiers.scheduled(p, tmp_path) == 2


@pytest.mark.parametrize('free,expected', [(150, 'verify'), (35, 'purge')])
def test_reusable_cache_is_only_purged_under_pressure(tmp_path, monkeypatch, free, expected):
    p = policy(tmp_path)
    calls = []
    monkeypatch.setattr(tiers, 'free_gb', lambda *_: free)
    monkeypatch.setattr(tiers.subprocess, 'run', lambda cmd, **kw: calls.append(cmd[-1]) or type('R', (), {'returncode': 0})())
    tiers.run_cleanup(p, {'name': 'cache', 'commands': ['verify'], 'pressure_commands': ['purge']}, log=lambda *_: None)
    assert expected in calls[0]


def test_aged_cleanup_refuses_nested_junction(tmp_path):
    base = tmp_path / 'cache'
    base.mkdir()
    target = tmp_path / 'user-data'
    target.mkdir()
    tiers.make_junction(base / 'linked', target)
    assert tiers._aged_entries({'exists': str(base / 'linked'), 'older_than_days': 0}, 1e12) == []


def test_hot_environment_prevents_parent_demotion(tmp_path, monkeypatch):
    base = tmp_path / 'results'
    (base / 'venv').mkdir(parents=True)
    (base / 'venv/model.bin').write_bytes(b'hot')
    cold = base / 'finished'
    cold.mkdir()
    (cold / 'evidence.json').write_bytes(b'cold')
    p = dict(policy(tmp_path), min_idle_hours=0, min_candidate_gb=0,
             candidates=[{'glob': 'results', 'class': 'work'}])
    monkeypatch.setattr(tiers, '_checkout_of', lambda *_: None)
    import storage_budget
    monkeypatch.setattr(storage_budget, 'protected_paths', lambda *_: [])
    monkeypatch.setattr(storage_budget, 'completed_paths', lambda *_: [])
    # An explicit now: a file time and time.time() in the same Windows clock tick can round 1 ulp apart.
    rows = tiers.candidates(p, tmp_path, now=time.time() + 60, processes=[])
    assert [r.path for r in rows if r.eligible] == [str(cold)]
    assert tiers._never(base / 'venv/model.bin', tmp_path, p)


def test_same_model_name_and_size_do_not_authorize_deletion(tmp_path):
    original = tmp_path / 'store/bandit-split.ckpt'
    candidate = tmp_path / 'scratch/bandit-split.ckpt'
    original.parent.mkdir()
    candidate.parent.mkdir()
    original.write_bytes(b'good')
    candidate.write_bytes(b'else')
    stored = tiers.stored_model_index({'model_stores': [str(original.parent)]})
    assert not tiers.verified_model_copy(str(candidate), 4, stored)
    candidate.write_bytes(b'good')
    assert tiers.verified_model_copy(str(candidate), 4, stored)


def test_dependency_inventory_distinguishes_copies_versions_and_shared_bytes(tmp_path):
    import os
    import storage_inventory
    for name, data in [('a/torch_cuda.dll', b'cuda12'), ('b/torch_cuda.dll', b'cuda12'),
                       ('c/torch_cuda.dll', b'cuda13')]:
        path = tmp_path / name
        path.parent.mkdir()
        path.write_bytes(data)
    (tmp_path / 'd').mkdir()
    os.link(tmp_path / 'a/torch_cuda.dll', tmp_path / 'd/torch_cuda.dll')
    report = storage_inventory.inventory([tmp_path])
    assert len(report['files']) == 4
    assert len(report['verified_groups']) == 1
    assert report['verified_groups'][0]['duplicate_bytes'] == 6
    assert report['verified_groups'][0]['distinct_physical_files'] == 2
    assert (tmp_path / 'c/torch_cuda.dll').read_bytes() == b'cuda13'


def test_inventory_hash_budget_never_claims_unverified_duplicate(tmp_path):
    import storage_inventory
    for folder in ['a', 'b']:
        path = tmp_path / folder
        path.mkdir()
        (path / 'ffmpeg.exe').write_bytes(b'ffmpeg')
    report = storage_inventory.inventory([tmp_path], hash_budget_gb=0)
    assert report['verified_groups'] == []
    assert all(r['verification'] == 'deferred_hash_budget' for r in report['files'])


def test_inventory_resumes_deferred_hashes_without_rereading_prior_files(tmp_path):
    import storage_inventory
    for folder in ['a', 'b']:
        path = tmp_path / folder
        path.mkdir()
        (path / 'ffmpeg.exe').write_bytes(b'ffmpeg')
    first = storage_inventory.inventory([tmp_path], hash_budget_gb=6e-9)
    assert not first['verified_groups']
    second = storage_inventory.inventory([tmp_path], hash_budget_gb=6e-9, prior=first)
    assert second['hash_bytes'] == 6
    assert second['verified_groups'][0]['duplicate_bytes'] == 6
