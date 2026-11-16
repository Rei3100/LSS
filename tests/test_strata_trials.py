"""Frozen trial failures and drafts never masquerade as accepted repairs."""
import importlib.util
import json
import hashlib
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('strata_trials', SCRIPTS / 'strata_trials.py')
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def reply(content, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}


def test_truncated_correct_choice_still_fails():
    case = {'oracle': {'type': 'exact', 'value': 'keep'}}
    assert trial.judge(case, reply('keep', 'length')) == {'status': 'incomplete', 'passed': False}


def test_parseable_code_is_only_a_draft():
    result = trial.judge({'oracle': {'type': 'draft-code-json'}}, reply('{"code":"def wrong(): return False"}'))
    assert result['passed'] is None
    assert result['repair_evaluation'] == 'pending'
    assert result['status'] == 'draft-ready'


def test_tool_call_wrong_revision_fails():
    response = reply(None, 'tool_calls')
    response['choices'][0]['message']['tool_calls'] = [{'function': {
        'name': 'select_current_best', 'arguments': '{"revision_id":"held_cue_999"}'}}]
    case = {'oracle': {'type': 'tool', 'name': 'select_current_best', 'arguments': {'revision_id': 'accepted_cue_001'}}}
    assert trial.judge(case, response)['passed'] is False


def test_modified_response_never_hits_resume_cache():
    response = reply('keep')
    row = {'key': 'key', 'response': response, 'response_digest': trial.digest(response), 'status': 'complete'}
    assert trial.reusable(row, 'key')
    response['choices'][0]['message']['content'] = 'remove'
    assert not trial.reusable(row, 'key')


def test_incomplete_reply_cannot_be_reused():
    response = reply('keep', 'length')
    row = {'key': 'key', 'response': response, 'response_digest': trial.digest(response), 'status': 'incomplete'}
    assert not trial.reusable(row, 'key')


def test_matrix_retains_distinct_repeats_and_historical_denominators():
    cases = trial.matrix()
    ids = [case['id'] for case in cases]
    assert len(ids) == len(set(ids))
    assert sum(c['family'] == 'historical-repair' for c in cases) == 12
    assert sum(c['family'] == 'decision' for c in cases) == 12
    assert sum(c['family'] == 'long-context' for c in cases) == 9


def test_explicit_variant_wins_over_helper_module_default(tmp_path, monkeypatch):
    home = tmp_path / 'runtime'
    home.mkdir()
    (home / 'lss-install-coder.json').write_text('{"model_files": []}')
    standard = home / 'user-state/Strata/launch/iq1_m-8k.json'
    quiet = home / 'variants/quiet-exact/user-state/Strata/launch/iq1_m-8k.json'
    standard.parent.mkdir(parents=True)
    quiet.parent.mkdir(parents=True)
    standard.write_bytes(b'standard config')
    quiet.write_bytes(b'quiet config')
    out = tmp_path / 'out'
    out.mkdir()
    monkeypatch.setattr(trial, 'output_directory', lambda output: out)
    monkeypatch.setattr(trial.runtime, 'paths', lambda profile: (home, tmp_path / 'model.gguf'))
    monkeypatch.setattr(trial.runtime, 'ACTIVE_VARIANT', 'standard')
    monkeypatch.setattr(trial, 'matrix', lambda: [])
    trial.run('test-model', out, variant='quiet-exact')
    identity = json.loads((out / 'identity.json').read_text(encoding='utf-8'))
    assert identity['variant'] == 'quiet-exact'
    assert identity['effective_config_sha256'] == hashlib.sha256(b'quiet config').hexdigest()
