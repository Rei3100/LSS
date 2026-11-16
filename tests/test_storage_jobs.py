"""Real wrapper and mover boundaries, with private storage policy and tiny outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.machine_config_fixtures import machine_policy

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import storage_budget as budget
import storage_tiers as tiers


@pytest.fixture
def policy(tmp_path, monkeypatch):
    for key in (budget.TOKEN_ENV, "LSS_STORAGE_POLICY_JSON", "LSS_STORAGE_POLICY"):
        monkeypatch.delenv(key, raising=False)
    return {**machine_policy(tmp_path), "schema_version": 1, "hot_volume": tmp_path.anchor,
            "local_state_root": str(tmp_path / "private-state"), "archive_root": str(tmp_path / "archive"),
            "receipts_root": str(tmp_path / "receipts"), "start_free_gb": 0, "target_free_gb": 0,
            "min_idle_hours": 48, "min_candidate_gb": 0,
            "candidates": [{"glob": "outputs/*", "class": "work"}],
            "storage_budget": {"floor_gb": 0, "default_job_gb": 0}}


def child_env(policy):
    return {**os.environ, "LSS_STORAGE_POLICY_JSON": json.dumps(policy), "PYTHONUTF8": "1"}


def test_batch_reserves_each_step_and_releases_before_next(policy, tmp_path):
    output = tmp_path / "outputs" / "job"
    output.mkdir(parents=True)
    queue = tmp_path / "queue.json"
    steps = []
    for number in (1, 2):
        code = ("import os,pathlib; "
                f"pathlib.Path({str(output / str(number))!r}).write_text(os.environ[{budget.TOKEN_ENV!r}])")
        steps.append({"name": str(number), "cmd": [sys.executable, "-c", code], "duty": 1,
                      "required": True, "storage_gb": 0, "archive_paths": [str(output)]})
    queue.write_text(json.dumps({"steps": steps}), encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPTS / "batch.py"), "run", str(queue)],
                            capture_output=True, text=True, env=child_env(policy), timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "1").read_text() != (output / "2").read_text()
    assert json.loads(queue.with_suffix(".receipt.json").read_text())["status"] == "ok"
    assert budget.protected_paths(policy) == []
    assert budget.completed_paths(policy)


def test_active_output_is_kept_then_completed_output_moves(policy, tmp_path, monkeypatch):
    output = tmp_path / "outputs" / "job"
    output.mkdir(parents=True)
    payload = output / "result.bin"
    payload.write_bytes(b"preserve me")
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    with budget.reserve(0, paths=[output], archive_paths=[output], policy=policy):
        rows = tiers.candidates(policy, tmp_path, processes=[])
        assert not rows[0].eligible and "reservation" in rows[0].reason
        with pytest.raises(OSError, match="reservation"):
            tiers.demote_one(policy, output)
        assert payload.read_bytes() == b"preserve me"
    rows = tiers.candidates(policy, tmp_path, processes=[])
    assert rows[0].eligible and rows[0].reason.startswith("explicit output completed")
    records = tiers.demote(policy, tmp_path, apply=True, until_free_gb=float("inf"), completed_only=True)
    assert len(records) == 1 and tiers.is_link(output)
    assert payload.read_bytes() == b"preserve me"
    assert budget.completed_paths(policy) == []


def test_completion_does_not_authorize_later_writes_or_foreign_roots(policy, tmp_path):
    output = tmp_path / "outputs" / "job"
    output.mkdir(parents=True)
    payload = output / "new.bin"
    payload.write_bytes(b"old")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    with budget.reserve(0, paths=[output, foreign], archive_paths=[output, foreign], policy=policy):
        pass
    stamp = time.time() + 10
    os.utime(payload, (stamp, stamp))
    rows = tiers.candidates(policy, tmp_path, processes=[])
    assert len(rows) == 1 and not rows[0].eligible
    assert "written" in rows[0].reason


def test_dev_receipt_is_written_before_completion_request(policy, tmp_path, monkeypatch):
    import dev
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    monkeypatch.setattr(budget, "_policy", lambda: policy)
    monkeypatch.setattr(dev, "identity", lambda: {"head": "stable"})
    monkeypatch.setattr(dev, "worker_count", lambda *args: 1)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_one.py").write_text("def test_one(): pass")

    def fake_pytest(command, environment, log_path, stream):
        assert environment[budget.TOKEN_ENV]
        assert budget.protected_paths(policy)
        log_path.write_text("1 passed")
        xml = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml="))
        Path(xml).write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="test_one"/></testsuite></testsuites>')
        return 0

    monkeypatch.setattr(dev, "run_pytest", fake_pytest)
    args = argparse.Namespace(changed=None, changed_list=False, paths=["tests/test_one.py"], suite="local",
                              keyword=None, collect_only=False, stream=False, workers="1", storage_gb=0)
    assert dev.run_tests(args) == 0
    [(directory, stamp)] = budget.completed_paths(policy)
    report = Path(directory) / "report.json"
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "PASSED"
    assert report.stat().st_mtime <= stamp


def test_direct_gpu_entry_uses_process_tree_supervisor_before_gpu_acquisition(monkeypatch):
    import gpu_run
    monkeypatch.delenv(budget.TOKEN_ENV, raising=False)
    seen = []
    monkeypatch.setattr(gpu_run.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or argparse.Namespace(returncode=7))
    monkeypatch.setattr(gpu_run, "_run_reserved", lambda *_: pytest.fail("GPU acquired before supervision"))
    assert gpu_run.main(["--storage-gb", "2", "--", sys.executable, "-c", "pass"]) == 7
    assert str(SCRIPTS / "storage_budget.py") in seen[0]
    assert "--required-gb" in seen[0] and "2.0" in seen[0]
