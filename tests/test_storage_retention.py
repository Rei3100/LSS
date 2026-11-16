"""Scheduled reclamation keeps dependencies and modern Product recovery intact."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import storage_retention as retain
import storage_tiers as tiers
import disk_cleanup as disk


def policy(tmp_path):
    return {**machine_policy(tmp_path), 'local_state_root': str(tmp_path / 'local'), 'retention': {
        'validation_idle_hours': 72, 'legacy_idle_hours': 336,
        'legacy_candidates_root': str(tmp_path / 'product/bgm-candidates')}}


def legacy(tmp_path):
    base = tmp_path / 'product/bgm-candidates' / ('a' * 64)
    (base / 'run/artifacts').mkdir(parents=True)
    for name, value in [('activity-recipe.json', {'method': 'windowed_rms_relative_hysteresis'}),
                        ('run/intent.json', {'corpus_role': 'product', 'source_sha256': 'b' * 64}),
                        ('run/separation.json', {'artifacts': []})]:
        (base / name).write_text(json.dumps(value))
    (base / 'run/artifacts/song.flac').write_bytes(b'kept-audio')
    (base / '.owner.lock').write_bytes(b'0')
    for p in [base, *base.rglob('*')]:
        os.utime(p, (1, 1))
    return base


def test_old_legacy_payload_archives_but_current_or_recent_never_does(tmp_path):
    p = policy(tmp_path)
    base = legacy(tmp_path)
    assert retain.legacy_payloads(p) == [base / 'run/artifacts']
    (base / 'run/run.json').write_text('{}')
    assert retain.legacy_payloads(p) == []


def test_archive_uses_product_owner_lock_and_keeps_audio(tmp_path, monkeypatch):
    p = policy(tmp_path)
    base = legacy(tmp_path)
    monkeypatch.setattr(tiers, 'process_texts', lambda: [])
    monkeypatch.setattr(tiers, 'process_cwds', lambda: ([], 0))
    called = []
    def move(policy, payload):
        with pytest.raises(RuntimeError):
            with tiers.Lock(base / '.owner.lock'):
                pass
        called.append(payload)
        return {'bytes': 10}
    monkeypatch.setattr(tiers, 'demote_one', move)
    assert retain.archive_legacy(p, apply=True)['bytes'] == 10
    assert called == [base / 'run/artifacts']
    assert (base / 'run/artifacts/song.flac').read_bytes() == b'kept-audio'


def test_validation_reclaims_only_old_classified_outputs(tmp_path, monkeypatch):
    p = policy(tmp_path)
    base = tmp_path / '.state/validation/case'
    base.mkdir(parents=True)
    rows = []
    for name, cls, age in [('music.wav', 'RETIRE', 100), ('speech.wav', 'KEEP', 100),
                           ('effects.wav', 'RETIRE', 1)]:
        path = base / name
        path.write_bytes(b'audio')
        os.utime(path, (time.time() - age * 3600,) * 2)
        rows.append({'path': str(path.relative_to(tmp_path)), 'bytes': 5, 'class': cls})
    (base / 'report.json').write_text('{}')
    monkeypatch.setattr(disk, 'ROOT', tmp_path)
    monkeypatch.setattr(disk, 'validation_media_candidates', lambda: (rows, []))
    monkeypatch.setattr(tiers, 'process_texts', lambda: [])
    monkeypatch.setattr(tiers, 'process_cwds', lambda: ([], 0))
    result = retain.validation_gc(p, tmp_path, apply=True)
    assert result['bytes'] == 5
    assert not (base / 'music.wav').exists()
    assert (base / 'speech.wav').exists() and (base / 'effects.wav').exists()
    assert (base / 'report.json').exists()
    assert result['deleted'][0]['sha256']


def test_broken_junction_is_not_entered_by_validation_scan(tmp_path):
    root = tmp_path / 'validation'
    root.mkdir()
    target = tmp_path / 'external'
    target.mkdir()
    tiers.make_junction(root / 'archived', target)
    target.rmdir()
    (root / 'report.json').write_text('{}')
    assert list(disk.local_files(root)) == [root / 'report.json']


def test_unreadable_dependency_index_never_authorizes_gc(tmp_path, monkeypatch):
    monkeypatch.setattr(disk, 'ROOT', tmp_path)
    monkeypatch.setattr(disk, 'validation_media_candidates', lambda: ([], ['denied plan.json']))
    with pytest.raises(OSError, match='dependencies'):
        retain.validation_gc(policy(tmp_path), tmp_path, apply=True)


def test_old_small_evidence_uses_its_own_demotion_threshold(tmp_path, monkeypatch):
    old = tmp_path / 'evidence/old'
    recent = tmp_path / 'evidence/recent'
    old.mkdir(parents=True)
    recent.mkdir()
    p = dict(policy(tmp_path), min_candidate_gb=0.2, min_idle_hours=2,
             hot_volume=str(tmp_path.anchor), target_free_gb=2,
             candidates=[{'glob': 'evidence/*', 'class': 'evidence',
                          'min_candidate_gb': 0.02, 'min_idle_hours': 168}])
    monkeypatch.setattr(tiers, '_checkout_of', lambda *_: None)
    monkeypatch.setattr(tiers, 'survey', lambda path: tiers.Tree(bytes=50_000_000, files=1,
        newest=time.time() - (200 if path == old else 20) * 3600))
    import storage_budget
    monkeypatch.setattr(storage_budget, 'protected_paths', lambda *_: [])
    monkeypatch.setattr(storage_budget, 'completed_paths', lambda *_: [])
    rows = tiers.candidates(p, tmp_path, processes=[])
    assert [r.path for r in rows if r.eligible] == [str(old)]
    moved = []
    monkeypatch.setattr(tiers, 'candidates', lambda *a, **kw: rows)
    monkeypatch.setattr(tiers, 'free_gb', lambda *_: 0)
    monkeypatch.setattr(tiers, 'background_priority', lambda: None)
    monkeypatch.setattr(tiers, '_regenerable_roots', lambda *_: [])
    monkeypatch.setattr(tiers, 'demote_one', lambda p, source: moved.append(source) or {'bytes': 50_000_000, 'seconds': 0})
    tiers.demote(p, tmp_path, apply=True, log=lambda *_: None)
    assert moved == [old]


def test_failed_validation_does_not_block_legacy_or_advance_success_stamp(tmp_path, monkeypatch):
    p = policy(tmp_path)
    p['retention']['every_hours'] = 24
    called = []
    def failed(*a, **kw):
        raise OSError('unreadable reference')
    monkeypatch.setattr(retain, 'validation_gc', failed)
    monkeypatch.setattr(retain, 'archive_legacy', lambda *a, **kw: called.append('legacy') or {'moved': []})
    with pytest.raises(RuntimeError, match='incomplete'):
        retain.maintain(p, tmp_path, log=lambda *_: None)
    assert called == ['legacy']
    assert not (tmp_path / 'local/last-retention.json').exists()
    assert (tmp_path / 'local/last-retention-attempt.json').exists()


def test_legacy_retention_survives_a_fast_data_root_junction(tmp_path):
    p = policy(tmp_path)
    base = legacy(tmp_path)
    original = base.parent
    relocated = tmp_path / 'data-ssd/bgm-candidates'
    relocated.parent.mkdir()
    original.rename(relocated)
    tiers.make_junction(original, relocated)
    p.update(hot_volume=tmp_path.anchor, fast_volumes=[tmp_path.anchor])
    assert retain.legacy_payloads(p) == [relocated / base.name / 'run/artifacts']


def test_legacy_archive_never_moves_product_state_when_hdd_moves_are_off(tmp_path, monkeypatch):
    p = {**policy(tmp_path), 'move_to_hdd': False}
    base = legacy(tmp_path)
    monkeypatch.setattr(tiers, 'demote_one', lambda *a: pytest.fail('Product state must not move to the HDD'))
    report = retain.archive_legacy(p, apply=True)
    assert report['moved'] == [] and report['skipped'] == 'move_to_hdd is false'
    assert (base / 'run/artifacts/song.flac').read_bytes() == b'kept-audio'
