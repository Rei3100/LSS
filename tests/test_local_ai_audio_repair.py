"""Real historical frontend replay and its closed execution surface."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import local_ai_audio_repair as oracle


@pytest.mark.parametrize('body', [
    'import os\n return os.listdir(signal_path)',
    'return open(signal_path)',
    'return signal_path.read_bytes()',
    'return signal_path.__class__',
    'return getattr(signal_path, "read_bytes")()',
    'return eval("1")',
    'return sf.write(signal_path, signal_path, 44100)',
    'return np.load(signal_path)',
    'return np.memmap(signal_path)',
    'return np.isfinite.__globals__',
    'return __import__("os")',
    'return resample_poly.__call__(signal_path, 1, 1)',
    'global sf\n return signal_path',
    'sf = signal_path\n return sf',
    'signal_path[0] = 1\n return signal_path',
    'return (lambda: signal_path)()',
    'while True:\n  pass',
    'return [0] * 1000000000',
])
def test_unsafe_candidate_surface_rejected(body):
    source = 'def model_input(signal_path):\n ' + body.replace('\n', '\n ')
    with pytest.raises((ValueError, SyntaxError)):
        oracle.validate_source(source)


def test_exact_historical_test_fails_before_and_passes_after():
    target = oracle.load_target()
    assert oracle.definition(oracle.git_show(f'{oracle.COMMIT}:{oracle.TEST}'), oracle.TEST_NAME) == target['test']
    before = oracle.evaluate_code(target['before'])
    after = oracle.evaluate_code(target['after'])
    assert not before['pass'] and 'unexpected keyword argument' in before['error'], before
    assert after['pass'] and after['torch_imported'] is False, after


def test_read_gate_denies_nonfixture_even_if_validated():
    source = oracle.load_target()['after'].replace('sf.read(str(signal_path)', 'sf.read("C:/outside.wav"')
    result = oracle.evaluate_code(source)
    assert not result['pass']
    assert 'only exact oracle fixture read' in result['error']


def test_worker_timeout_is_enforced():
    result = oracle.evaluate_code(oracle.load_target()['after'], timeout=0.001)
    assert result == {'pass': False, 'error': 'timeout'}


def test_prompt_contains_only_before_source_and_public_contract():
    target = oracle.load_target()
    request, = oracle.generate_requests()
    prompt = request['prompt']
    assert target['before'] in prompt
    assert target['after'] not in prompt
    assert target['test'] not in prompt
    assert oracle.TEST_NAME not in prompt
    assert 'default_rng' not in prompt
    assert request['provenance']['commit'] == oracle.COMMIT


def test_saved_audit_has_one_new_functional_control_and_original_provenance():
    report = json.loads((dev_evidence_root(ROOT) / 'release-evidence/local-ai-challengers-r1/audio-repair/audit.json').read_text())
    assert report['eligible_functional_targets_added'] == 1
    assert report['model_runs'] == 0
    assert report['provenance']['test_name'] == oracle.TEST_NAME
    assert report['limits']['memory_bytes'] <= 1024**3
