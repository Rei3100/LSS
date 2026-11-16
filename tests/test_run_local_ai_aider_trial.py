import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_local_ai_aider_trial as runner


def test_historical_controls_and_fixed_candidate_use_same_oracle():
    task = runner.load_task()
    controls = runner.check_controls(task)
    assert controls['baseline']['pass'] is False
    assert controls['positive_control']['pass'] is True
    assert runner.replay.evaluate_code(task['after'], '_normalise', task['constants'], task['cases'])['pass']
    prompt = runner.contract(task)
    assert task['after'] not in prompt and str(task['cases']) not in prompt


def test_invalid_control_prevents_serving(tmp_path, monkeypatch):
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    cases = tmp_path / 'cases'; cases.mkdir()
    monkeypatch.setattr(runner, 'EVIDENCE', evidence)
    monkeypatch.setattr(runner, 'CASE_PARENT', cases)
    monkeypatch.setattr(runner, 'load_task', lambda: {'commit':'x','before':'old','after':'new','constants':{},'cases':[]})
    import dev_hot_paths
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: None)
    @contextlib.contextmanager
    def forbidden(*args, **kwargs):
        pytest.fail('server must not start')
        yield
    result = runner.run('no-start', serve=forbidden, evaluate=lambda *a: {'pass': True})
    assert result['stage'] == 'controls' and result['status'] == 'unmeasured'
    assert not (cases / 'aider-case-no-start').exists()


def test_child_environment_does_not_inherit_api_credentials(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-test-value')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'secret-test-value')
    env = runner.child_environment()
    assert env['OPENAI_API_KEY'] == 'local-dummy'
    assert 'ANTHROPIC_API_KEY' not in env
    assert env['CUDA_VISIBLE_DEVICES'] == '-1' and env['PYTHONUTF8'] == '1'


def test_child_preserves_windows_runtime_locations_without_repurposing_home(monkeypatch):
    monkeypatch.setenv('USERPROFILE', ('C' + chr(58) + '/synthetic-home'))
    monkeypatch.setenv('LOCALAPPDATA', ('C' + chr(58) + '/synthetic-home/AppData/Local'))
    env = runner.child_environment()
    assert env['USERPROFILE'] == ('C' + chr(58) + '/synthetic-home')
    assert env['LOCALAPPDATA'] == ('C' + chr(58) + '/synthetic-home/AppData/Local')
    assert 'HOME' not in env


def test_label_and_existing_case_refuse_before_model(tmp_path, monkeypatch):
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    cases = tmp_path / 'cases'; cases.mkdir()
    monkeypatch.setattr(runner, 'EVIDENCE', evidence)
    monkeypatch.setattr(runner, 'CASE_PARENT', cases)
    import dev_hot_paths
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: None)
    (cases / 'aider-case-run1').mkdir()
    with pytest.raises(ValueError, match='immutable'): runner.run('run1')
    with pytest.raises(ValueError): runner.run('../escape')


def test_fake_proposal_preserves_fixture_and_evaluates(tmp_path, monkeypatch):
    import hashlib
    import dev_hot_paths
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    cases = tmp_path / 'cases'; cases.mkdir()
    runtime_dir = tmp_path / 'runtime'; runtime_dir.mkdir()
    exe = runtime_dir / 'llama-server.exe'; exe.write_bytes(b'runtime')
    python = tmp_path / 'python.exe'; python.write_bytes(b'python')
    model_path = tmp_path / 'model.gguf'; model_path.write_bytes(b'model')
    outside = tmp_path / 'user-file.txt'; outside.write_text('unchanged')
    monkeypatch.setattr(runner, 'EVIDENCE', evidence)
    monkeypatch.setattr(runner, 'CASE_PARENT', cases)
    monkeypatch.setattr(runner, 'RUNTIME', python)
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: None)
    monkeypatch.setattr(runner.local_llm, 'llama_dir', lambda: runtime_dir)
    model = SimpleNamespace(name='qwen3.5-4b', path=model_path, sha256=hashlib.sha256(b'model').hexdigest())
    monkeypatch.setattr(runner.local_llm, 'get_model', lambda _: model)
    task = runner.load_task()
    monkeypatch.setattr(runner, 'load_task', lambda: task)
    @contextlib.contextmanager
    def serve(*args, **kwargs):
        assert kwargs['ctx'] == 16384
        yield 'http://127.0.0.1:5000/v1'
    def child(command, **kwargs):
        case = Path(command[command.index('--case') + 1])
        snapshot = runner.guarded.snapshot_case(runner.guarded.CasePaths(case))
        receipt = {'status':'proposal_only', 'case_sha256':snapshot, 'proposal':task['after'], 'identity':{'version':'fake'}}
        kwargs['stdout'].write(json.dumps(receipt).encode())
        Path(command[command.index('--transcript') + 1]).write_text('{}')
        return SimpleNamespace(returncode=0)
    result = runner.run('fake-run', serve=serve, child=child)
    assert result['status'] == 'evaluated' and result['oracle']['pass']
    assert result['case_before_sha256'] == result['case_after_sha256']
    assert outside.read_text() == 'unchanged'
    assert (cases / 'aider-case-fake-run/candidate.py').read_text(encoding='utf-8') == task['before']
