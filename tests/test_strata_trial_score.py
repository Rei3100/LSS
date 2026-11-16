"""Scoring transport retains truncation and refuses altered trial receipts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('strata_trial_score', SCRIPTS / 'strata_trial_score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def fixture(tmp_path, finish='stop'):
    case = {'id': 'off/fix-example', 'prompt': 'repair', 'oracle': {'type': 'draft-code-json'}}
    identity = {'installation': {'model_files': []}, 'variant': 'quiet-exact'}
    request = {'messages': [{'role': 'user', 'content': 'repair'}]}
    response = {'choices': [{'finish_reason': finish, 'message': {'content': '{"code":"bad"}'}}]}
    key = score.trials.digest({'identity': identity, 'case': case, 'request': request})
    row = {'id': case['id'], 'key': key, 'request': request, 'response': response,
           'response_digest': score.trials.digest(response), 'status': 'draft-ready'}
    for name, value in [('summary.json', {'status': 'completed', 'completed': 1, 'total': 1}),
                        ('identity.json', identity), ('cases.json', [case]), (key + '.json', row)]:
        score.trials.atomic(tmp_path / name, value)
    return tmp_path / (key + '.json')


def test_export_preserves_length_finish_for_existing_rejection(tmp_path):
    fixture(tmp_path, 'length')
    result = score.export(tmp_path, 'off')
    assert result['replays'][0]['finish_reason'] == 'length'


def test_changed_reply_cannot_enter_historical_oracles(tmp_path):
    path = fixture(tmp_path)
    row = json.loads(path.read_text(encoding='utf-8'))
    row['response']['choices'][0]['message']['content'] = '{"code":"replacement"}'
    score.trials.atomic(path, row)
    with pytest.raises(ValueError, match='response integrity'):
        score.export(tmp_path, 'off')


def test_partial_matrix_does_not_claim_full_denominator(tmp_path):
    fixture(tmp_path)
    score.trials.atomic(tmp_path / 'summary.json', {'status': 'running', 'completed': 1, 'total': 43})
    with pytest.raises(ValueError, match='incomplete matrix'):
        score.export(tmp_path, 'off')
