from __future__ import annotations

import ntpath

import pytest

from scripts import dev_hot_paths as hot


def path_resolver(mapping):
    return lambda path: mapping.get(str(path), str(path))


def windows_volume(path):
    return ntpath.splitdrive(str(path))[0].upper()


def test_require_hot_accepts_a_path_whose_link_resolves_to_the_hot_volume():
    resolve = path_resolver({('E' + chr(58) + '\\cache\\model.gguf'): ('C' + chr(58) + '\\synthetic-work\\local-ai-models\\model.gguf')})

    result = hot.require_hot(('E' + chr(58) + '\\cache\\model.gguf'), "model", ('C' + chr(58) + '\\'),
                             resolver=resolve, volume_resolver=windows_volume)

    assert result["ok"] is True
    assert result["resolved"] == ('C' + chr(58) + '\\synthetic-work\\local-ai-models\\model.gguf')
    assert result["volume"] == result["hot_volume"] == "C:"


def test_require_hot_raises_with_resolved_path_when_target_is_on_another_volume():
    resolve = path_resolver({('C' + chr(58) + '\\alias\\job'): ('E' + chr(58) + '\\synthetic-dev\\active\\job')})

    with pytest.raises(hot.HotPathError) as caught:
        hot.require_hot(('C' + chr(58) + '\\alias\\job'), "output", ('C' + chr(58) + '\\'),
                        resolver=resolve, volume_resolver=windows_volume)

    assert caught.value.record == {
        "label": "output", "input": ('C' + chr(58) + '\\alias\\job'), "resolved": ('E' + chr(58) + '\\synthetic-dev\\active\\job'),
        "volume": "E:", "hot_volume": "C:", "ok": False,
    }


def test_inspect_paths_reports_all_paths_without_stopping_at_first_failure():
    resolve = path_resolver({"linked": ('E' + chr(58) + '\\cold\\linked')})
    examined = []

    def record_resolver(path):
        examined.append(str(path))
        return resolve(path)

    report = hot.inspect_paths([("repo", ('C' + chr(58) + '\\repo')), ("output", "linked"),
                                 ("model", ('E' + chr(58) + '\\cold\\model.gguf'))], ('C' + chr(58) + '\\'),
                                resolver=record_resolver, volume_resolver=windows_volume)

    assert report["ok"] is False
    assert report["checked"] == 3
    assert [row["ok"] for row in report["paths"]] == [True, False, False]
    assert ('C' + chr(58) + '\\repo') in examined and "linked" in examined and ('E' + chr(58) + '\\cold\\model.gguf') in examined


def test_non_windows_volume_check_uses_injected_physical_mount_identity():
    resolve = path_resolver({"alias": "/mnt/ssd/models/one.gguf"})
    physical_volume = lambda path: "ssd-volume-id" if str(path).startswith("/mnt/ssd") else "hdd-volume-id"

    result = hot.require_hot("alias", "model", "/mnt/ssd", resolver=resolve,
                             volume_resolver=physical_volume)

    assert result["ok"] is True
    assert result["volume"] == "ssd-volume-id"


def test_default_targets_include_active_registry_models_but_skip_archive_only(tmp_path, monkeypatch):
    policy = tmp_path / "storage_policy.json"
    policy.write_text(('{"hot_volume":"C' + chr(58) + '\\\\","stage":{"root":"C' + chr(58) + '/stage"}}'), encoding="utf-8")
    registry = tmp_path / "models.json"
    registry.write_text(
        ('{"models":{"active":{"path":"C' + chr(58) + '/models/active.gguf"},')
        + ('"archive":{"path":"E' + chr(58) + '/archive/old.gguf","archive_only":true}}}'), encoding="utf-8")
    monkeypatch.setattr(hot, "_git_common_dir", lambda _repo: tmp_path / ".git")

    paths, volume = hot.default_paths(tmp_path, prefix=('C' + chr(58) + '/venv'), base_prefix=('C' + chr(58) + '/python'),
                                      temp_dir=('C' + chr(58) + '/temp'), environ={}, registry_path=registry,
                                      policy_path=policy)

    assert volume == ('C' + chr(58) + '\\')
    labels = {label for label, _ in paths}
    assert {"repository", "git_common_dir", "python_prefix", "python_base_prefix",
            "temp_directory", "model_stage_cache", "model:active"} <= labels
    assert "model:archive" not in labels


def test_require_hot_records_unconfigured_fast_volumes_as_not_checked(monkeypatch):
    monkeypatch.setattr(hot, 'load_config', lambda path: {'hot_volume': ('C' + chr(58) + '\\'), '_machine_overlay': None})

    def never(path):
        raise AssertionError('volume resolver must not run')

    result = hot.require_hot(('X' + chr(58) + '\\anything'), 'label', resolver=lambda p: str(p), volume_resolver=never)

    assert result['ok'] is True
    assert result['checked'] is False
    assert result['label'] == 'label'
    assert result['reason'] is not None and len(result['reason']) > 0

    monkeypatch.setattr(hot, 'load_config', lambda path: {'hot_volume': ('C' + chr(58) + '\\'), 'fast_volumes': [('C' + chr(58) + '\\')]})

    with pytest.raises(hot.HotPathError):
        hot.require_hot(('E' + chr(58) + '\\job'), 'label', resolver=lambda p: str(p), volume_resolver=windows_volume)
