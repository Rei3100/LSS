"""Local model deletion needs positive, recoverable provenance and every independent safety guard."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import storage_tiers as tiers


def age(path, hours=24):
    then = time.time() - hours * 3600
    for directory, names, files in os.walk(path):
        for name in names + files:
            os.utime(Path(directory) / name, (then, then))
        os.utime(directory, (then, then))


def model(store, name="abandoned-model", *, status="retired", provenance=True, cache=False):
    path = store / name
    path.mkdir()
    payloads = {"weights.gguf": b"public reproducible weights"}
    if cache:
        payloads[".cache/blob"] = b"not disposable by name"
    for relative, data in payloads.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if provenance:
        record = {"source": "example/public-model", "revision": "a" * 40, "status": status,
                  "total_bytes": sum(map(len, payloads.values())),
                  "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                            for name, data in payloads.items()}}
        (path / tiers.MODEL_PROVENANCE).write_text(json.dumps(record), encoding="utf-8")
    age(path)
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    state = tmp_path / "isolated-state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state))
    monkeypatch.delenv("LSS_STORAGE_RESERVATION", raising=False)
    store, profiles, repo = (tmp_path / name for name in ("models", "profiles", "repo"))
    for path in (store, profiles, repo):
        path.mkdir()
    policy = {**machine_policy(tmp_path), "hot_volume": tmp_path.anchor, "min_idle_hours": 2, "move_to_hdd": False,
              "local_state_root": str(state / "storage"),
              "storage_budget": {"floor_gb": 0, "default_job_gb": 0},
              "unused_local_models": {"roots": [str(store)], "profile_dirs": [str(profiles)], "keep": []}}
    tracked = []
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        if command == ["git", "-C", str(repo), "ls-files", "-z"]:
            return SimpleNamespace(stdout="\0".join(tracked).encode("utf-8"))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(tiers.subprocess, "run", run)
    monkeypatch.setattr(tiers, "main_root", lambda: repo)
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    monkeypatch.setattr(tiers, "process_cwds", lambda: ([], 0))
    return SimpleNamespace(policy=policy, store=store, profiles=profiles, repo=repo, tracked=tracked, state=state)


def clean(case, *, apply=True, log=lambda line: None):
    return tiers.clean_hot(case.policy, apply=apply, log=log)


def test_unknown_folder_and_cache_are_kept_and_receipted(case):
    path = model(case.store, provenance=False, cache=True)
    lines = []
    result = clean(case, log=lines.append)
    assert result["deleted"] == [] and result["bytes"] == 0
    assert result["kept"][0]["reason"] == "unknown provenance"
    assert any("kept: unknown provenance" in line for line in lines)
    assert (path / ".cache/blob").is_file()
    receipt = next((case.state / "storage/receipts").glob("tier-clean-hot-*.json"))
    assert json.loads(receipt.read_text())["kept"] == result["kept"]


def test_keep_list_protects_classified_model_and_cache(case):
    path = model(case.store, cache=True)
    case.policy["unused_local_models"]["keep"] = [path.name.upper()]
    result = clean(case)
    assert result["deleted"] == [] and result["kept"][0]["reason"] == "keep list"
    assert (path / ".cache/blob").is_file()


def test_verified_idle_model_is_deleted_with_reacquisition_receipt(case, monkeypatch):
    path = model(case.store, status="retired", cache=True)
    remove = tiers._remove_tree

    def remove_after_receipt(target):
        receipt = next((case.state / "storage/receipts").glob("tier-local-model-*.json"))
        assert json.loads(receipt.read_text())["provenance"]["revision"] == "a" * 40
        remove(target)

    monkeypatch.setattr(tiers, "_remove_tree", remove_after_receipt)
    result = clean(case)
    assert not path.exists()
    assert [r["path"] for r in result["deleted"]] == [str(path)]
    receipt = json.loads(Path(result["deleted"][0]["provenance_receipt"]).read_text())
    assert receipt["provenance"]["status"] == "retired"
    assert receipt["provenance"]["files"][".cache/blob"]["sha256"]


@pytest.mark.parametrize("location", ["profile", "preset", "script", "config", "root_config", "filename"])
def test_saved_and_tracked_references_keep_model(case, location):
    path = model(case.store)
    if location in {"profile", "preset"}:
        reference = case.profiles / ("nested/preset.json" if location == "preset" else "launch.json")
    else:
        relative = {"script": "scripts/recover.py", "config": "configs/models.yaml",
                    "root_config": "models.json", "filename": f"scripts/{path.name}.py"}[location]
        case.tracked.append(relative)
        reference = case.repo / relative
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("{}" if location == "filename" else json.dumps({
        "model": str(path / "weights.gguf").upper()}), encoding="utf-8")
    result = clean(case)
    assert path.is_dir() and result["deleted"] == []
    assert result["kept"][0]["reason"] == "saved profile/preset or tracked script/config reference"


@pytest.mark.parametrize("usage", ["command_path", "command_name", "cwd", "unknown_cwd", "unknown_processes"])
def test_running_or_unknown_process_inventory_keeps_model(case, monkeypatch, usage):
    path = model(case.store)
    if usage == "command_path":
        monkeypatch.setattr(tiers, "process_texts", lambda: [f"server --model {path.as_posix().upper()}/weights.gguf"])
    elif usage == "command_name":
        monkeypatch.setattr(tiers, "process_texts", lambda: [f"server --model {path.name.upper()}"])
    elif usage == "cwd":
        monkeypatch.setattr(tiers, "process_cwds", lambda: ([(123, "server", str(path))], 0))
    elif usage == "unknown_cwd":
        monkeypatch.setattr(tiers, "process_cwds", lambda: ([], 1))
    else:
        monkeypatch.setattr(tiers, "process_texts", lambda: ["<process list unavailable>"])
        with pytest.raises(OSError, match="process inventory unavailable"):
            clean(case)
        assert path.is_dir()
        return
    result = clean(case)
    assert path.is_dir() and result["deleted"] == []
    assert "process" in result["kept"][0]["reason"]


@pytest.mark.parametrize("defect", ["missing_hash", "invalid_hash", "wrong_hash", "wrong_size", "total",
                                   "source", "revision", "mutable_revision", "extra_payload", "missing_payload",
                                   "cache_not_recorded", "invalid_json", "non_object"])
def test_incomplete_or_mismatched_provenance_never_deletes(case, defect):
    path = model(case.store)
    manifest = path / tiers.MODEL_PROVENANCE
    data = json.loads(manifest.read_text())
    if defect == "missing_hash":
        del data["files"]["weights.gguf"]["sha256"]
    elif defect in {"invalid_hash", "wrong_hash"}:
        data["files"]["weights.gguf"]["sha256"] = "invalid" if defect == "invalid_hash" else "b" * 64
    elif defect == "wrong_size":
        data["files"]["weights.gguf"]["bytes"] += 1
        data["total_bytes"] += 1
    elif defect == "total":
        data["total_bytes"] += 1
    elif defect in {"source", "revision"}:
        del data[defect]
    elif defect == "mutable_revision":
        data["revision"] = "main"
    elif defect in {"extra_payload", "cache_not_recorded"}:
        target = path / ("unrecorded.bin" if defect == "extra_payload" else ".cache/blob")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"unknown, possibly unrecoverable data")
    elif defect == "missing_payload":
        (path / "weights.gguf").unlink()
    manifest.write_text("{" if defect == "invalid_json" else json.dumps([] if defect == "non_object" else data))
    age(path)
    result = clean(case)
    assert path.is_dir() and result["deleted"] == []
    assert result["kept"][0]["reason"].startswith("unknown provenance")


@pytest.mark.parametrize("status", ["trial", "active", "needed_later", None])
def test_needed_or_unclassified_status_is_kept(case, status):
    path = model(case.store, status=status)
    result = clean(case)
    assert path.is_dir() and result["deleted"] == []
    assert result["kept"][0]["reason"] == "status is not retired"


def test_nested_recent_activity_and_model_specific_idle_threshold_keep_model(case):
    path = model(case.store)
    (path / "nested/deep").mkdir(parents=True)
    recent = path / "nested/deep/recent.txt"
    recent.write_text("active work")
    age(path)
    os.utime(recent, None)  # direct entries remain old
    assert "not idle" in clean(case)["kept"][0]["reason"]
    age(path, 3)
    case.policy["unused_local_models"]["min_idle_hours"] = 48
    assert "not idle" in clean(case)["kept"][0]["reason"]


@pytest.mark.parametrize("failure", ["git", "profile_read", "oversize"])
def test_reference_discovery_failure_keeps_model(case, monkeypatch, failure):
    path = model(case.store)
    if failure == "git":
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "git ls-files")
        monkeypatch.setattr(tiers.subprocess, "run", fail)
    else:
        reference = case.profiles / "launch.json"
        reference.write_text("{}")
        if failure == "oversize":
            reference.write_bytes(b" " * (2 * 1024 * 1024 + 1))
        else:
            original_read = Path.read_text
            def read(p, *args, **kwargs):
                if p == reference:
                    raise PermissionError("unreadable profile")
                return original_read(p, *args, **kwargs)
            monkeypatch.setattr(Path, "read_text", read)
    result = clean(case)
    assert path.is_dir() and result["deleted"] == []
    assert result["kept"][0]["reason"] == "reference scan unavailable"


def test_dry_run_reports_eligible_and_unknown_without_writes(case):
    eligible = model(case.store)
    unknown = model(case.store, "unclassified-model", provenance=False)
    lines = []
    result = clean(case, apply=False, log=lines.append)
    assert result["dry_run"] and [r["path"] for r in result["deleted"]] == [str(eligible)]
    assert result["kept"][0]["path"] == str(unknown)
    assert any("would delete" in line for line in lines)
    assert any("kept: unknown provenance" in line for line in lines)
    assert eligible.is_dir() and unknown.is_dir() and not case.state.exists()


def test_status_reports_unknown_provenance(case, monkeypatch, capsys):
    path = model(case.store, provenance=False)
    case.policy.update(start_free_gb=1, target_free_gb=2, min_candidate_gb=0)
    monkeypatch.setattr(tiers, "candidates", lambda *args: [])
    monkeypatch.setattr(tiers, "load_policy", lambda: case.policy)
    monkeypatch.setitem(sys.modules, "stage", SimpleNamespace(total_bytes=lambda p: 0, settings=lambda p: {
        "max_gb": 1, "evict_first_below_free_gb": 1}))
    report = tiers.status(case.policy, case.repo)
    assert report["local_models"][0]["reason"] == "unknown provenance"
    assert tiers.main(["status"]) == 0
    assert f"{path}: kept: unknown provenance" in capsys.readouterr().out


def test_late_reference_is_rechecked_under_mutation_guard(case, monkeypatch):
    path = model(case.store)
    original = tiers.storage_mutation
    @contextmanager
    def guard(policy, paths):
        with original(policy, paths):
            (case.profiles / "new-preset.json").write_text(json.dumps({"model": str(path)}))
            yield
    monkeypatch.setattr(tiers, "storage_mutation", guard)
    result = clean(case)
    assert path.is_dir() and result["deleted"] == []
    assert "reference" in result["kept"][0]["reason"]


def test_superseded_and_temp_rules_cannot_bypass_local_model_safety(case):
    older = model(case.store, "weights-r1", provenance=False)
    newer = model(case.store, "weights-r2", provenance=False)
    case.policy.update(model_stores=[str(case.store)], superseded_models=True,
                       temp_orphans={"root": str(case.store), "min_idle_hours": 2})
    result = clean(case)
    assert result["deleted"] == [] and older.is_dir() and newer.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction safety")
def test_linked_payload_keeps_model_and_target(case):
    import _winapi
    path = model(case.store)
    target = case.repo / "permanent"
    target.mkdir()
    (target / "keep.bin").write_bytes(b"irreplaceable")
    _winapi.CreateJunction(str(target), str(path / "linked"))
    result = clean(case)
    assert result["deleted"] == [] and path.is_dir()
    assert (target / "keep.bin").read_bytes() == b"irreplaceable"


def test_policy_keeps_hdd_moves_disabled():
    policy = json.loads(tiers.POLICY_PATH.read_text())
    assert policy["move_to_hdd"] is False


def test_shipped_policy_never_deletes_unknown_temp_items_by_age_alone():
    """%TEMP% items of unknown origin are UNKNOWN: only owned prefixes may be reclaimed by age."""
    policy = json.loads((Path(tiers.__file__).with_name("storage_policy.json")).read_text(encoding="utf-8"))
    for rule in policy.get("temp_orphans", []):
        if "%TEMP%" in str(rule.get("root", "")).upper():
            assert rule.get("prefixes"), rule
