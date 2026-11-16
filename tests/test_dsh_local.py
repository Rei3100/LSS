from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dsh_local as subject


def test_patch_routes_to_local_strata_with_sandbox_and_without_deepseek_account():
    text = subject.patch('m-id', Path.home() / 'w')
    assert 'baseURL: http://127.0.0.1:18766/v1' in text and '- id: m-id' in text
    assert 'mode: workspace-write' in text and text.count('disabled: true') == len(subject.DISABLED)


def test_workspace_must_sit_under_the_user_profile(tmp_path):
    assert subject.check_workspace(Path.home() / 'x') == (Path.home() / 'x').resolve()
    with pytest.raises(ValueError):
        subject.check_workspace('C:/LSS-Work/x' if sys.platform == 'win32' else '/opt/x')


def test_run_refuses_a_server_with_a_small_context(monkeypatch):
    monkeypatch.setattr(subject, 'health', lambda: {'max_context': 8192, 'loaded': True, 'model': 'm'})
    with pytest.raises(RuntimeError):
        subject.run(Path.home() / 'w', 'task', 'true')
