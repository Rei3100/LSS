"""A case index cannot silently change denominators, truth or cached correspondence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts import research_tests as research


def test_real_catalog_separates_actual_condition_banks_and_bgm_singing():
    data = research.catalog()
    banks = {b['id']: b for b in data['banks']}
    assert len(banks['condition-6s']['cases']) == 7
    assert len(banks['explosion-context']['cases']) == 6
    assert not any('sound_effect' in c['conditions'] for c in banks['condition-6s']['cases'])
    selected = research.plan(data, 'bgm-separation', ['order', 'context'])
    assert [p['id'] for p in selected['patterns']] == ['order', 'context']
    assert not selected['missing_conditions']
    assert all(c['truth'] == 'controlled-clean-component' for c in selected['groups'][0]['cases'])
    assert any('wanted-singing' in c['conditions'] for c in selected['groups'][0]['cases'])
    assert 'Independent generic' in selected['uncovered'][0]
    assert selected['promotion_allowed'] is False


def test_unknown_question_pattern_and_missing_audio_oracle_are_not_passes():
    data = research.catalog()
    for profile, patterns in [('unknown', []), ('instrumental', ['typo'])]:
        with pytest.raises(ValueError):
            research.plan(data, profile, patterns)
    selected = research.plan(data, 'scene-naming', [])
    assert selected['status'] == 'NO_REGISTERED_AUDIO_CASES'
    assert selected['uncovered']


def test_cli_has_compact_discovery_and_explicit_full_plan(capsys):
    assert research.main(['plan', 'instrumental']) == 0
    compact = capsys.readouterr().out
    assert 'SELECTED_NOT_EXECUTED' in compact and 'UNRESOLVED' in compact
    assert 'input_sha256' not in compact
    assert research.main(['--json', 'plan', 'instrumental']) == 0
    full = json.loads(capsys.readouterr().out)
    assert len(full['groups'][0]['cases']) == 12
    assert research.main(['plan', 'unknown']) == 2
    assert 'REFUSED' in capsys.readouterr().err


def _bank(tmp_path):
    cases = []
    for i, absent in enumerate([False, True]):
        rng = np.random.default_rng(i)
        target = np.zeros((1024, 2)) if absent else rng.normal(0, .05, (1024, 2))
        vocals = rng.normal(0, .05, (1024, 2))
        artifacts = {}
        for role, values in [('input', target+vocals), ('target', target), ('vocals', vocals)]:
            p = tmp_path / f'{i}-{role}.wav'
            sf.write(p, values, 8000, subtype='FLOAT')
            artifacts[role] = {'path': str(p), 'sha256': research.digest(p)}
        cases.append({'id': f'case{i}', 'truth': 'controlled_known_components',
                      'conditions': ['target-absent' if absent else 'target-present'],
                      'sample_rate': 8000, 'frames': 1024, 'artifacts': artifacts})
    native = tmp_path / 'native.json'
    native.write_text(json.dumps({'cases': [{'case_id': c['id']} for c in cases]}))
    bank = {'id': 'instrumental-controls', 'cases': cases,
            'definition': {'path': str(native), 'sha256': research.digest(native)}, 'limits': []}
    return {'banks': [bank], 'profiles': []}, bank


def _protocol(data, bank):
    outputs = {c['id']: {**c['artifacts']['target'],
                         'input_sha256': c['artifacts']['input']['sha256'],
                         'encoding': 'float-wav'} for c in bank['cases']}
    return {'schema_version': 1, 'bank_sha256': research.identity(bank),
            'baseline': {'identity': 'oracle', 'target': 'Instrumental', 'outputs': copy.deepcopy(outputs)},
            'candidate': {'identity': 'same-oracle', 'target': 'Instrumental', 'outputs': copy.deepcopy(outputs)}}


def test_check_detects_missing_changed_and_relabelled_cases(tmp_path):
    data, bank = _bank(tmp_path)
    assert research.check(data, hash_assets=True)['status'] == 'INDEX_VALID'
    bank['cases'][0]['id'] = 'silently-relabelled'
    assert any('case sets differ' in e for e in research.check(data)['errors'])
    bank['cases'][0]['artifacts']['input']['sha256'] = '0'*64
    assert any('HASH_MISMATCH' in e for e in research.check(data, hash_assets=True)['errors'])
    Path(bank['cases'][0]['artifacts']['target']['path']).unlink()
    assert any('MISSING' in e for e in research.check(data)['errors'])


@pytest.mark.parametrize('mutation', ['missing-case', 'extra-case', 'changed-input', 'wrong-bank', 'bgm-target'])
def test_compare_rejects_denominator_and_target_changes_before_audio(tmp_path, mutation):
    data, bank = _bank(tmp_path)
    protocol = _protocol(data, bank)
    if mutation == 'missing-case':
        protocol['candidate']['outputs'].pop('case0')
    elif mutation == 'extra-case':
        protocol['candidate']['outputs']['other'] = {}
    elif mutation == 'changed-input':
        protocol['candidate']['outputs']['case0']['input_sha256'] = '0'*64
    elif mutation == 'wrong-bank':
        protocol['bank_sha256'] = '0'*64
    else:
        protocol['candidate']['target'] = 'BGM'
    with pytest.raises(ValueError):
        research.compare_instrumental(data, protocol)


def test_cached_compare_keeps_absence_and_damage_visible(tmp_path):
    data, bank = _bank(tmp_path)
    protocol = _protocol(data, bank)
    # A clean no-op remains unchanged; added voice in target-absent output regresses.
    protocol['candidate']['outputs']['case1'].update(bank['cases'][1]['artifacts']['vocals'])
    result = research.compare_instrumental(data, protocol)
    assert result['status'] == 'DEVELOPMENT_DIAGNOSTIC'
    assert result['promotion_allowed'] is False
    assert result['regression_cases'] == ['case1']
    assert result['cases'][1]['baseline']['fidelity']['scale_dependent_snr_db'] is None


def test_changed_audio_and_channel_collapse_are_refused(tmp_path):
    data, bank = _bank(tmp_path)
    protocol = _protocol(data, bank)
    row = protocol['candidate']['outputs']['case0']
    row['sha256'] = '0'*64
    with pytest.raises(ValueError, match='Changed audio'):
        research.compare_instrumental(data, protocol)
    p = tmp_path / 'mono.wav'
    sf.write(p, np.ones(1024)*.01, 8000, subtype='FLOAT')
    row.update(path=str(p), sha256=research.digest(p))
    with pytest.raises(ValueError, match='native stereo'):
        research.compare_instrumental(data, protocol)


def test_duplicate_json_identities_are_refused(tmp_path):
    p = tmp_path / 'duplicate.json'
    p.write_text('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        research.read(p)
    data = research.catalog()
    data['banks'].append(copy.deepcopy(data['banks'][0]))
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate banks'):
        research.catalog(p)


@pytest.mark.parametrize('change', ['audio', 'implementation'])
def test_no_receipt_can_certify_files_changed_during_comparison(tmp_path, monkeypatch, change):
    data, bank = _bank(tmp_path)
    protocol = _protocol(data, bank)
    real = research.digest
    watched = (bank['cases'][0]['artifacts']['input']['path'] if change == 'audio'
               else str(research.ROOT / 'lss/validation/stereo_quality.py'))
    calls = 0
    def changed(path):
        nonlocal calls
        value = real(path)
        if str(path) == watched:
            calls += 1
            if calls > 1:
                return '0'*64
        return value
    monkeypatch.setattr(research, 'digest', changed)
    with pytest.raises(ValueError, match='changed during comparison'):
        research.compare_instrumental(data, protocol)


def test_relative_evidence_and_public_resources_use_their_own_roots(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    home = tmp_path / "evidence-home"
    home.mkdir()
    monkeypatch.setenv("LSS_DEV_STATE", str(home / ".state"))
    for relative in ("release-evidence/run/result.json", "boundary-evidence/run/result.json",
                     "ep001-truth/answer.json", "ledger/research_tests.json", "ledger/test_set.tsv"):
        assert research.local_path(relative, checkout) == home / relative
    for relative in ("ledger/models.tsv", "ledger/model_stem_semantics.tsv",
                     "ledger/test_set.tsv.example", "lss/validation/stereo_quality.py"):
        assert research.local_path(relative, checkout) == checkout / relative
    absolute = tmp_path / "absolute.json"
    assert research.local_path(str(absolute), checkout) == absolute
