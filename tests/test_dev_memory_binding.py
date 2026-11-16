"""A redirected user profile cannot select a divergent cache after binding."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import dev as cli


def binding(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    directory = tmp_path / '.cache/local-ai/devmem'
    directory.mkdir(parents=True)
    (directory / 'devmem.sqlite').touch()
    path = tmp_path / '.state/devmem-index.json'
    path.parent.mkdir()
    value = {'schema': 1, 'repo': str(tmp_path), 'index_dir': str(directory)}
    path.write_text(json.dumps(value))
    return path, value, directory


def test_bound_cache_wins_over_redirected_profile(tmp_path, monkeypatch):
    path, value, directory = binding(tmp_path, monkeypatch)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'different-profile'))
    assert cli.developer_memory_config().db_dir == directory
    assert not (tmp_path / 'different-profile').exists()


@pytest.mark.parametrize('change', ['schema-bool', 'other-repo', 'outside-cache', 'missing-database', 'large'])
def test_invalid_binding_never_silently_falls_back(tmp_path, monkeypatch, change):
    path, value, directory = binding(tmp_path, monkeypatch)
    if change == 'schema-bool':
        value['schema'] = True
    elif change == 'other-repo':
        value['repo'] = str(tmp_path / 'another-repo')
    elif change == 'outside-cache':
        value['index_dir'] = str(tmp_path / 'external')
    elif change == 'missing-database':
        (directory / 'devmem.sqlite').unlink()
    path.write_text(json.dumps(value) if change != 'large' else ' ' * 16385)
    with pytest.raises(ValueError):
        cli.developer_memory_config()


def test_explicit_override_remains_available_without_reading_binding(tmp_path, monkeypatch):
    path, value, directory = binding(tmp_path, monkeypatch)
    path.write_text('invalid')
    explicit = tmp_path / 'explicit'
    assert cli.developer_memory_config(explicit).db_dir == explicit


def test_absent_binding_keeps_unbound_profile_default(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    import devmem
    assert cli.developer_memory_config().db_dir == devmem.Config().db_dir
