"""Owned acceptance leftovers are reclaimed without deleting active work."""
import contextlib
import json
from pathlib import Path
import sys

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import storage_tiers as tiers


def test_reclaim_idle_scratch_keep_active_cwd_and_link_target(tmp_path, monkeypatch):
    busy, idle, original = [tmp_path / name for name in ('accept-busy', 'accept-idle', 'models')]
    (busy / 'nested').mkdir(parents=True)
    idle.mkdir()
    original.mkdir()
    (busy / 'nested/work.bin').write_bytes(b'active work')
    (idle / 'scratch.bin').write_bytes(b'disposable scratch')
    (original / 'weights.bin').write_bytes(b'permanent weights')
    tiers.make_junction(idle / 'models', original)
    policy = {**machine_policy(tmp_path), 'hot_volume': tmp_path.anchor, 'start_free_gb': 60,
              'min_idle_hours': 2, 'pressure_free_gb': 40, 'pressure_min_idle_hours': 1,
              'local_state_root': str(tmp_path / 'state')}
    monkeypatch.setattr(tiers, 'process_texts', lambda: ['python.exe worker.py'])
    monkeypatch.setattr(tiers, 'process_cwds', lambda: ([(1234, 'python.exe', str(busy / 'nested'))], 0))
    monkeypatch.setattr(tiers, 'superseded_models', lambda *args: [])
    monkeypatch.setattr(tiers, 'temp_orphans', lambda *args: [busy, idle])
    monkeypatch.setattr(tiers, 'storage_mutation', lambda *args: contextlib.nullcontext())
    result = tiers.clean_hot(policy, apply=True, log=lambda line: None)
    assert (busy / 'nested/work.bin').read_bytes() == b'active work'
    assert not idle.exists()
    assert (original / 'weights.bin').read_bytes() == b'permanent weights'
    assert len(result['deleted']) == 1


def test_cwd_collection_failure_never_deletes(tmp_path, monkeypatch):
    scratch = tmp_path / 'accept-old'
    scratch.mkdir()
    monkeypatch.setattr(tiers, 'process_texts', lambda: ['python.exe worker.py'])
    def unavailable():
        raise OSError('cwd inventory unavailable')
    monkeypatch.setattr(tiers, 'process_cwds', unavailable)
    monkeypatch.setattr(tiers, 'superseded_models', lambda *args: [])
    monkeypatch.setattr(tiers, 'temp_orphans', lambda *args: [scratch])
    with pytest.raises(OSError, match='cwd inventory unavailable'):
        tiers.clean_hot({**machine_policy(tmp_path), 'hot_volume': tmp_path.anchor, 'min_idle_hours': 2,
                         'start_free_gb': 60, 'pressure_free_gb': 40,
                         'pressure_min_idle_hours': 1}, apply=True, log=lambda line: None)
    assert scratch.exists()


def test_generated_acceptance_retention_is_one_day(tmp_path):
    import os
    import time

    roots = [tmp_path / name for name in ('hot', 'work', 'post-deploy')]
    created = []
    for root in roots:
        scratch = root / 'accept-generated'
        scratch.mkdir(parents=True)
        (scratch / 'data.bin').write_bytes(b'synthetic acceptance output')
        created.append(scratch)
    written = time.time()
    for scratch in created:
        os.utime(scratch / 'data.bin', (written, written))
        os.utime(scratch, (written, written))
    policy = {'temp_orphans': [{'root': str(root), 'prefixes': ['accept-'], 'min_idle_hours': 24}
                               for root in roots]}
    assert len(policy['temp_orphans']) == 3
    assert all(rule['min_idle_hours'] == 24 for rule in policy['temp_orphans'])
    assert tiers.temp_orphans(policy, now=written + 24 * 3600 - 1) == []
    assert set(tiers.temp_orphans(policy, now=written + 24 * 3600 + 1)) == set(created)
