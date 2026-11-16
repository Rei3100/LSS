"""Three-volume placement and copy/switch/recovery falsifiers."""
import json
import os
from pathlib import Path
import sys

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import storage_layout as layout
import storage_tiers as tiers
import storage_budget as budget


def policy(tmp_path):
    return {**machine_policy(tmp_path), 'hot_volume': str(tmp_path.anchor), 'local_state_root': str(tmp_path / 'state'),
            'layout': {'destination_root': str(tmp_path / 'D'), 'minimum_free_gb': 0}}


def tree(tmp_path):
    source = tmp_path / 'C/models'
    source.mkdir(parents=True)
    (source / 'model.bin').write_bytes(b'exact wanted weights')
    return source, tmp_path / 'D/models'


def test_verified_switch_keeps_old_path_and_cold_junction(tmp_path):
    p = policy(tmp_path)
    source, target = tree(tmp_path)
    cold = tmp_path / 'E/archive'
    cold.mkdir(parents=True)
    (cold / 'record.json').write_text('{}')
    tiers.make_junction(source / 'cold', cold)
    result = layout.migrate(p, source, target, apply=True)
    assert result['status'] == 'migrated_verified'
    assert tiers.is_link(source) and source.resolve() == target
    assert (source / 'model.bin').read_bytes() == b'exact wanted weights'
    assert (target / 'cold').resolve() == cold
    assert (cold / 'record.json').read_text() == '{}'
    assert layout.migrate(p, source, target, apply=True)['status'] == 'already_placed'


def test_existing_old_pc_data_is_never_overwritten(tmp_path):
    p = policy(tmp_path)
    source, target = tree(tmp_path)
    target.mkdir(parents=True)
    (target / 'user.bin').write_bytes(b'old PC data')
    with pytest.raises(OSError, match='Existing destination'):
        layout.migrate(p, source, target, apply=True)
    assert (target / 'user.bin').read_bytes() == b'old PC data'
    assert not tiers.is_link(source)


def test_failed_verification_leaves_source_untouched(tmp_path, monkeypatch):
    p = policy(tmp_path)
    source, target = tree(tmp_path)
    def fail(*a, **kw):
        raise OSError('copy verification failed')
    monkeypatch.setattr(tiers, 'copy_verified', fail)
    with pytest.raises(OSError):
        layout.migrate(p, source, target, apply=True)
    assert (source / 'model.bin').read_bytes() == b'exact wanted weights'
    assert not tiers.is_link(source)


def test_interrupted_switch_recovers_without_recopying_or_losing_source(tmp_path, monkeypatch):
    p = policy(tmp_path)
    source, target = tree(tmp_path)
    discard = layout._discard_owned
    monkeypatch.setattr(layout, '_discard_owned', lambda *_: (_ for _ in ()).throw(OSError('interrupted after switch')))
    with pytest.raises(OSError):
        layout.migrate(p, source, target, apply=True)
    assert tiers.is_link(source)
    monkeypatch.setattr(layout, '_discard_owned', discard)
    result = layout.migrate(p, source, target, apply=True)
    assert result['status'] == 'migrated_recovered'
    assert (source / 'model.bin').read_bytes() == b'exact wanted weights'


def test_worktree_foundation_excluded_from_data_scope(tmp_path):
    base = tmp_path / 'C/work'
    (base / 'wt').mkdir(parents=True)
    (base / 'models').mkdir()
    p = policy(tmp_path)
    p['layout']['scopes'] = [{'source': str(base), 'target': str(tmp_path / 'D/work'), 'exclude': ['wt']}]
    assert [src.name for src, dst in layout.pairs(p)] == ['models']


def test_acceptance_scratch_belongs_to_retention_not_layout(tmp_path, monkeypatch):
    p = policy(tmp_path)
    base, destination = tmp_path / 'C/work', tmp_path / 'D/work'
    for directory, payload in ((base / 'accept-example', b'C scratch'),
                               (destination / 'accept-example', b'D scratch')):
        directory.mkdir(parents=True)
        (directory / 'data.bin').write_bytes(payload)
    (base / 'models').mkdir()
    p['layout']['scopes'] = [{'source': str(base), 'target': str(destination),
                              'exclude_prefixes': ['accept-']}]
    assert [src.name for src, dst in layout.pairs(p)] == ['models']
    assert (base / 'accept-example/data.bin').read_bytes() == b'C scratch'
    assert (destination / 'accept-example/data.bin').read_bytes() == b'D scratch'
    # The existing age-based cleanup still owns eligible scratch on both volumes.
    p['temp_orphans'] = [{'root': str(root), 'prefixes': ['accept-'], 'min_idle_hours': 72}
                         for root in (base, destination)]
    future = __import__('time').time() + 73 * 3600
    assert set(tiers.temp_orphans(p, now=future)) == {
        base / 'accept-example', destination / 'accept-example'}


def test_capacity_charges_resolved_destination_not_alias(tmp_path, monkeypatch):
    c, d = ('C' + chr(58) + '\\'), ('D' + chr(58) + '\\')
    p = {'hot_volume': c, 'fast_volumes': [c, d], 'work_volume': d,
         'local_state_root': str(tmp_path / 'state'), 'storage_budget': {'floor_gb': 30},
         'volume_limits': {d: {'budget_floor_gb': 80}}}
    monkeypatch.delenv(budget.TOKEN_ENV, raising=False)
    monkeypatch.setattr(tiers, 'on_volume', lambda path, volume: volume == d)
    monkeypatch.setattr(tiers, 'free_gb', lambda volume: 85 if volume == d else 999)
    with pytest.raises(budget.BudgetError, match='D:'):
        with budget.reserve(6, paths=[tmp_path], policy=p, wait=False):
            pytest.fail('C free space admitted a full D volume')
    with budget.reserve(5, paths=[tmp_path], policy=p, wait=False):
        row = json.loads(next((tmp_path / 'state/budget/active').glob('*.json')).read_text())
        assert row['volumes'] == [d]


def test_broken_directory_junction_is_preserved_without_reading_target(tmp_path):
    p = policy(tmp_path)
    source, target = tree(tmp_path)
    cold = tmp_path / 'E/missing'
    cold.mkdir(parents=True)
    tiers.make_junction(source / 'old-reference', cold)
    cold.rmdir()
    layout.migrate(p, source, target, apply=True)
    assert tiers.is_link(target / 'old-reference')
    assert not cold.exists()


def test_prepared_model_preserves_archive_and_avoids_rehash_on_noop(tmp_path, monkeypatch):
    import hashlib
    p = policy(tmp_path)
    archive = tmp_path / 'E/model.gguf'
    archive.parent.mkdir()
    archive.write_bytes(b'archive weights')
    target = tmp_path / 'D/models/model.gguf'
    p['layout']['prepare_files'] = [{'source': str(archive), 'target': str(target), 'bytes': 15,
                                    'sha256': hashlib.sha256(b'archive weights').hexdigest()}]
    monkeypatch.setattr(tiers, 'process_texts', lambda: [])
    monkeypatch.setattr(tiers, 'process_cwds', lambda: ([], 0))
    layout.maintain(p, tmp_path)
    assert archive.read_bytes() == target.read_bytes() == b'archive weights'
    monkeypatch.setattr(tiers, '_hash', lambda *_: pytest.fail('unchanged prepared file reread'))
    layout.maintain(p, tmp_path)


def test_fast_guard_accepts_both_ssds_but_rejects_archive():
    import dev_hot_paths as hot
    import ntpath
    resolve = lambda p: str(p)
    volume = lambda p: ntpath.splitdrive(str(p))[0]
    assert hot.require_hot(('D' + chr(58) + '/models/one'), 'model', [('C' + chr(58) + '/'), ('D' + chr(58) + '/')], resolver=resolve, volume_resolver=volume)['ok']
    with pytest.raises(hot.HotPathError):
        hot.require_hot(('E' + chr(58) + '/archive/one'), 'model', [('C' + chr(58) + '/'), ('D' + chr(58) + '/')], resolver=resolve, volume_resolver=volume)


def test_protected_package_stays_protected_through_data_ssd_alias(tmp_path):
    p = policy(tmp_path)
    source, target = tree(tmp_path)
    (source / 'LSS.exe').write_bytes(b'candidate')
    (source / '_internal').mkdir()
    p.update(delete_package_copies=True, protected_packages=[str(source)])
    layout.migrate(p, source, target, apply=True)
    assert not tiers.is_package_copy(target, p, tmp_path)


def test_new_build_outputs_use_data_root_and_existing_output_is_untouched(tmp_path):
    import storage_paths
    repo = tmp_path / 'C/source'
    (repo / 'build').mkdir(parents=True)
    (repo / 'build/old.bin').write_bytes(b'existing')
    p = {'build_root': str(tmp_path / 'D/builds')}
    rows = storage_paths.prepare_build(repo, p)
    assert (repo / 'build/old.bin').read_bytes() == b'existing'
    assert not tiers.is_link(repo / 'build')
    assert tiers.is_link(repo / 'dist')
    assert (repo / 'dist').resolve().is_relative_to(tmp_path / 'D')


def test_changed_source_refreshes_only_its_unchanged_owned_copy(tmp_path, monkeypatch):
    p = policy(tmp_path)
    source, target = tree(tmp_path)
    original = tiers.copy_verified
    def changing(*a, **kw):
        result = original(*a, **kw)
        (source / 'model.bin').write_bytes(b'new source weights')
        return result
    monkeypatch.setattr(tiers, 'copy_verified', changing)
    with pytest.raises(OSError, match='Source changed'):
        layout.migrate(p, source, target, apply=True)
    assert not tiers.is_link(source)
    monkeypatch.setattr(tiers, 'copy_verified', original)
    layout.migrate(p, source, target, apply=True)
    assert (source / 'model.bin').read_bytes() == b'new source weights'


def test_stage_capacity_uses_its_own_volume_not_system_free_space(monkeypatch):
    import stage
    p = {'hot_volume': ('C' + chr(58) + '\\'), 'stage': {'root': ('D' + chr(58) + '/stage'), 'min_free_after_gb': 30}}
    calls = []
    monkeypatch.setattr(stage, 'cache_volume', lambda _: ('D' + chr(58) + '\\'))
    monkeypatch.setattr(stage, 'total_bytes', lambda _: 0)
    monkeypatch.setattr(stage, 'prune', lambda *a, **kw: None)
    monkeypatch.setattr(tiers, 'free_gb', lambda volume: calls.append(volume) or (20 if volume == ('D' + chr(58) + '\\') else 999))
    with pytest.raises(stage.StageRefused, match='D: free'):
        stage._make_room(p, 1, set())
    assert set(calls) == {('D' + chr(58) + '\\')}


def test_layout_leaves_room_for_existing_jobs_peak_growth(tmp_path, monkeypatch):
    p = policy(tmp_path)
    p['storage_budget'] = {'floor_gb': 0}
    p['layout']['minimum_free_gb'] = 100
    source, target = tree(tmp_path)
    monkeypatch.setattr(tiers, 'free_gb', lambda *_: 150)
    monkeypatch.delenv(budget.TOKEN_ENV, raising=False)
    with budget.reserve(100, paths=[tmp_path / 'other-job'], policy=p, wait=False):
        with pytest.raises(OSError, match='capacity floor'):
            layout.migrate(p, source, target, apply=True)
    assert not tiers.is_link(source)
    assert (source / 'model.bin').exists()
