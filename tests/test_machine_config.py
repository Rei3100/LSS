"""Public config, private overlay isolation and storage mutation admission."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import machine_config as config
from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import storage_tiers as tiers

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def files(tmp_path, monkeypatch):
    public = tmp_path / "public" / "storage_policy.json"
    public.parent.mkdir()
    public.write_text(json.dumps({"nested": {"keep": 1, "replace": [1]}, "scalar": 1}), encoding="utf-8")
    private = tmp_path / "private"
    overlay = private / "config" / "storage_policy.local.json"
    overlay.parent.mkdir(parents=True)
    monkeypatch.setenv("LSS_PRIVATE_ROOT", str(private))
    monkeypatch.delenv("LSS_TEST_MODE", raising=False)
    return public, overlay


def test_recursive_merge_replaces_lists_scalars_and_null(files):
    public, overlay = files
    public_before = public.read_bytes()
    overlay.write_text(json.dumps({"nested": {"replace": [2], "new": True}, "scalar": None}), encoding="utf-8")
    assert config.load_config(public) == {
        "nested": {"keep": 1, "replace": [2], "new": True}, "scalar": None,
        "_machine_overlay": str(overlay),
    }
    assert public.read_bytes() == public_before


def test_object_and_scalar_can_replace_each_other(files):
    public, overlay = files
    overlay.write_text('{"nested": 9, "scalar": {"new": [3]}}', encoding="utf-8")
    result = config.load_config(public)
    assert result["nested"] == 9 and result["scalar"] == {"new": [3]}


def test_test_mode_never_opens_overlay(files, monkeypatch):
    public, overlay = files
    overlay.write_text("invalid JSON", encoding="utf-8")
    monkeypatch.setenv("LSS_TEST_MODE", "1")
    original = Path.read_text
    def read(path, *args, **kwargs):
        assert path != overlay, "test mode opened a private overlay"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    assert config.load_config(public)["_machine_overlay"] is None


@pytest.mark.parametrize("text", ["{broken", "[]", "null", '"text"', "42"])
def test_invalid_overlay_raises_without_fallback(files, text):
    public, overlay = files
    overlay.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        config.load_config(public)


@pytest.mark.parametrize("private_root", [None, "relative-private"])
def test_unset_or_relative_root_does_not_select_overlay(files, monkeypatch, private_root):
    public, overlay = files
    overlay.write_text('{"scalar": 99}', encoding="utf-8")
    if private_root is None:
        monkeypatch.delenv("LSS_PRIVATE_ROOT")
    else:
        monkeypatch.setenv("LSS_PRIVATE_ROOT", private_root)
    result = config.load_config(public)
    assert result["scalar"] == 1 and result["_machine_overlay"] is None


def test_missing_overlay_cannot_be_claimed_by_public_json(files):
    public, _ = files
    public.write_text('{"_machine_overlay": "claimed", "scalar": 1}', encoding="utf-8")
    assert config.load_config(public)["_machine_overlay"] is None


def test_non_object_public_config_raises(files):
    public, _ = files
    public.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        config.load_config(public)


@pytest.mark.parametrize("key", ["never", "protected_packages", "model_stores", "frozen_roots"])
@pytest.mark.parametrize("value", [None, [], [""], "not-a-list"])
def test_overlay_with_empty_or_invalid_protections_still_refuses(tmp_path, key, value):
    policy = machine_policy(tmp_path)
    policy[key] = value
    with pytest.raises(config.StorageMutationRefused, match=key):
        config.require_machine_overlay(policy)


def test_explicit_temporary_machine_scope_is_accepted(tmp_path):
    config.require_machine_overlay(machine_policy(tmp_path))


@pytest.mark.parametrize("entry", [
    "demote_one", "promote", "recover", "clean_hot", "dedupe", "prune",
    "delete_disposables", "clean_scratch", "demote", "retire_worktree",
    "promote_worktree", "run_cleanup", "scheduled", "stage_relief", "maintain",
    "retention_validation", "retention_archive", "retention_maintain",
    "layout_migrate", "layout_maintain", "stage_prune", "stage_prune_for_free",
    "stage_sweep", "stage_evict", "stage_get", "disk_prune", "disk_retire",
    "disk_validation", "disk_provider", "disk_media", "model_pin", "cli_apply",
])
def test_mutating_entry_points_refuse_public_only_policy(tmp_path, monkeypatch, entry):
    import disk_cleanup as disk
    import local_ai_storage
    import stage
    import storage_layout as layout
    import storage_retention as retention
    policy = tiers.load_policy()
    assert policy["_machine_overlay"] is None  # conftest enables test mode.
    monkeypatch.setattr(tiers, "load_policy", lambda *args: policy)
    monkeypatch.setattr(tiers, "uncommitted_storage_code", lambda: [])
    source = tmp_path / "source.bin"
    source.write_bytes(b"keep")
    monkeypatch.setattr(tiers, "on_hot_volume", lambda *args: False)
    actions = {
        "demote_one": lambda: tiers.demote_one(policy, source),
        "promote": lambda: tiers.promote(policy, source),
        "recover": lambda: tiers.recover(policy),
        "clean_hot": lambda: tiers.clean_hot(policy, apply=True),
        "dedupe": lambda: tiers.dedupe(policy, [tmp_path], apply=True),
        "prune": lambda: tiers.prune(policy, tmp_path, apply=True),
        "delete_disposables": lambda: tiers.delete_disposables(policy, [source]),
        "clean_scratch": lambda: tiers.clean_scratch(policy, apply=True),
        "demote": lambda: tiers.demote(policy, tmp_path, apply=True),
        "retire_worktree": lambda: tiers.retire_worktree(policy, tmp_path, SimpleNamespace(path=source)),
        "promote_worktree": lambda: tiers.promote_worktree(policy, tmp_path, source, apply=True),
        "run_cleanup": lambda: tiers.run_cleanup(policy, {"commands": ["unused"]}),
        "scheduled": lambda: tiers.scheduled(policy, tmp_path),
        "stage_relief": lambda: tiers.stage_relief(policy),
        "maintain": lambda: tiers.maintain(policy, tmp_path),
        "retention_validation": lambda: retention.validation_gc(policy, tmp_path, apply=True),
        "retention_archive": lambda: retention.archive_legacy(policy, apply=True),
        "retention_maintain": lambda: retention.maintain(policy, tmp_path),
        "layout_migrate": lambda: layout.migrate(policy, source, tmp_path / "target", apply=True),
        "layout_maintain": lambda: layout.maintain(policy, tmp_path),
        "stage_prune": lambda: stage.prune(policy),
        "stage_prune_for_free": lambda: stage.prune_for_free(policy),
        "stage_sweep": lambda: stage._sweep(policy),
        "stage_evict": lambda: stage._evict(policy, source),
        "stage_get": lambda: stage.get(policy, source, force=True),
        "disk_prune": lambda: disk.prune(True, 0),
        "disk_retire": lambda: disk.retire(True),
        "disk_validation": lambda: disk.validation_media_gc(True),
        "disk_provider": lambda: disk.provider_media_gc(True),
        "disk_media": lambda: disk._media_gc([], [], True, tmp_path, None, "unused"),
        "model_pin": lambda: local_ai_storage.pin("unused", policy=policy),
        "cli_apply": lambda: tiers.main(["prune", "--apply"]),
    }
    with pytest.raises(RuntimeError, match="no machine overlay"):
        actions[entry]()
    assert source.read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [source]


def test_public_read_only_reports_dry_runs_and_budget_still_work(tmp_path, monkeypatch):
    import stage
    import storage_budget as budget
    import storage_retention as retention
    policy = tiers.load_policy()
    policy["local_state_root"] = str(tmp_path / "state")
    policy["stage"]["root"] = str(tmp_path / "stage")
    monkeypatch.setattr(tiers, "free_gb", lambda *args: 1000.0)
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    monkeypatch.setattr(tiers, "process_cwds", lambda: ([], 0))
    for name in (budget.TOKEN_ENV, "LSS_STORAGE_POLICY", "LSS_STORAGE_POLICY_JSON"):
        monkeypatch.delenv(name, raising=False)
    with budget.reserve(0, paths=[tmp_path], policy=policy, wait=False, _allow_nudge=False):
        assert tiers.status(policy, tmp_path)["candidates"] == []
    assert tiers.demote(policy, tmp_path, apply=False) == []
    assert tiers.prune(policy, tmp_path, apply=False)["dirs"] == 0
    assert tiers.clean_hot(policy, apply=False)["dry_run"] is True
    assert tiers.clean_scratch(policy, apply=False)["bytes"] == 0
    assert stage.entries(policy) == []
    assert retention.legacy_payloads(policy) == []
    assert tiers.audit(policy)["status"] == "archive_not_configured"


def test_model_without_path_is_unavailable_and_cannot_start(tmp_path, monkeypatch, capsys):
    import local_llm as llm
    models = llm.load_registry()
    assert models and all(model.path is None for model in models.values())
    model = next(iter(models.values()))
    assert "unavailable" in llm.availability(model)
    with pytest.raises(llm.RegistryError, match="no machine path"):
        llm.get_model(model.name)
    monkeypatch.setattr(llm, "_launcher", lambda: pytest.fail("started without model path"))
    monkeypatch.setattr(llm, "llama_dir", lambda: pytest.fail("looked for runtime without model path"))
    with pytest.raises(llm.RegistryError, match="no machine path"):
        with llm.serving(model.name, cpu=True):
            pytest.fail("model started")
    assert llm.cmd_models(SimpleNamespace(json=True)) == 0
    rows = json.loads(capsys.readouterr().out)
    assert all(row["path"] is None and "unavailable" in row["available"] for row in rows)


def test_hot_paths_skip_unconfigured_models_and_report_unavailable(tmp_path, monkeypatch, capsys):
    from scripts import dev_hot_paths as hot
    monkeypatch.setattr(hot, "_git_common_dir", lambda repo: tmp_path / ".git")
    paths, _ = hot.default_paths(tmp_path)
    assert not any(label.startswith("model:") for label, path in paths)
    monkeypatch.setattr(hot, "inspect_paths", lambda *args: {"ok": True})
    assert hot.main(["--repo", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["unavailable_models"]
    assert all(row["reason"] == "no machine path configured" for row in report["unavailable_models"])
