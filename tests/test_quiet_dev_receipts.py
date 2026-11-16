"""Quiet job failures publish terminal receipts without overwriting other work."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import quiet_dev_checks as quiet


@pytest.fixture
def owned(tmp_path, monkeypatch):
    import dev_hot_paths
    monkeypatch.setattr(quiet, 'ROOT', tmp_path)
    monkeypatch.setattr(quiet, "dev_evidence_root", lambda checkout_root: checkout_root)
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *args: None)
    return tmp_path / 'release-evidence' / 'case'


def test_failure_before_first_task_publishes_terminal_receipt(owned, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError('private signed URL must not be copied')
    monkeypatch.setattr(quiet, '_run_claimed', failure)
    with pytest.raises(RuntimeError):
        quiet.run(owned)
    data = json.loads((owned/'result.json').read_text())
    assert data['status'] == 'failed' and data['error_type'] == 'RuntimeError'
    assert 'signed' not in (owned/'result.json').read_text()


def test_existing_success_is_never_overwritten(owned):
    owned.mkdir(parents=True)
    result = owned/'result.json'
    result.write_text('{"status":"passed","guard":"keep exact bytes"}')
    before = result.read_bytes()
    with pytest.raises(ValueError):
        quiet.run(owned)
    assert result.read_bytes() == before


def test_failed_dependency_is_immediately_visible_to_next_job(owned):
    dependency = owned.parent/'dependency.json'
    dependency.parent.mkdir(parents=True)
    dependency.write_text('{"status":"failed"}')
    with pytest.raises(RuntimeError, match='Required quiet check failed'):
        quiet.run(owned, after=dependency)
    assert json.loads((owned/'result.json').read_text())['status'] == 'failed'


def test_failed_guard_never_freezes_or_hashes_candidate(owned, monkeypatch):
    import storage_budget
    monkeypatch.setattr(storage_budget, 'reserve', lambda *args, **kwargs: nullcontext())
    calls = []
    def failed_guard(command, **kwargs):
        calls.append(command)
        assert '-m' in command and 'pytest' in command
        return {'returncode': 1}
    monkeypatch.setattr(quiet, 'run_capped', failed_guard)
    result = quiet.run(owned, finalize=True)
    assert result['status'] == 'failed' and len(calls) == 1
    assert result['skipped_steps'] == ['bridge-and-freeze']


def test_exception_retains_completed_step_evidence(owned, monkeypatch):
    def failure(output, **kwargs):
        (output/'result.json').write_text('{"status":"running","steps":[{"name":"done","returncode":0}]}')
        raise OSError('later failure')
    monkeypatch.setattr(quiet, '_run_claimed', failure)
    with pytest.raises(OSError):
        quiet.run(owned)
    result = json.loads((owned/'result.json').read_text())
    assert result['status'] == 'failed' and result['steps'] == [{'name': 'done', 'returncode': 0}]
