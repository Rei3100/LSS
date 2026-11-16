"""Temporary-only storage admission falsifiers: concurrency, dead owners and corruption."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import storage_budget as budget
import storage_tiers as tiers


@pytest.fixture(autouse=True)
def isolated_sleep_policy(tmp_path, monkeypatch):
    import dev_quiet_mode
    monkeypatch.setattr(dev_quiet_mode, 'POLICY', tmp_path / 'absent-sleep-policy.json')


@pytest.fixture
def policy(tmp_path, monkeypatch):
    monkeypatch.delenv(budget.TOKEN_ENV, raising=False)
    monkeypatch.delenv("LSS_STORAGE_POLICY_JSON", raising=False)
    monkeypatch.delenv("LSS_STORAGE_POLICY", raising=False)
    monkeypatch.setattr(tiers, "free_gb", lambda _: 40)
    return dict(local_state_root=str(tmp_path / "state"), hot_volume=str(tmp_path.anchor),
                storage_budget=dict(floor_gb=30, default_job_gb=5))


def test_capacity_and_explicit_owned_receipt(policy, tmp_path):
    owned = tmp_path / "outputs"
    with budget.reserve(6, paths=[tmp_path], archive_paths=[owned], policy=policy) as held:
        assert not held.inherited
        assert str(tmp_path) in budget.protected_paths(policy)
        with pytest.raises(budget.BudgetError, match="reserved 6.00"):
            with budget.reserve(5, policy=policy, wait=False):
                pytest.fail("oversubscribed")
        with budget.reserve(4, policy=policy, wait=False):
            pass
        with pytest.raises(budget.BudgetError, match="overlaps"):
            with budget.mutation_guard([owned], policy):
                pytest.fail("mover entered active output")
    assert budget.protected_paths(policy) == []
    hints = budget.completed_paths(policy)
    assert [p for p, stamp in hints] == [str(owned)]
    receipt = json.loads(next((tmp_path / "state/budget/pending").glob("*.json")).read_text())
    assert receipt["status"] == "completed"
    assert receipt["paths"] == [str(tmp_path)]
    assert receipt["archive_paths"] == [str(owned)]


def test_nested_reuses_live_owner_and_rejects_expansion(policy, tmp_path, monkeypatch):
    with budget.reserve(1, paths=[tmp_path], policy=policy) as parent:
        monkeypatch.setenv(budget.TOKEN_ENV, parent.token)
        with budget.reserve(paths=[tmp_path / "child"], policy=policy) as child:
            assert child.inherited and child.required_gb == 1
            assert len(list((tmp_path / "state/budget/active").glob("*.json"))) == 1
        with budget.reserve(paths=[tmp_path / "child"], archive_paths=[tmp_path / "child"], policy=policy):
            pass
        assert budget.completed_paths(policy)[0][0] == str(tmp_path / "child")
        for kwargs in [dict(required_gb=2), dict(paths=[tmp_path.parent])]:
            with pytest.raises(budget.BudgetError, match="refused"):
                with budget.reserve(policy=policy, **kwargs):
                    pytest.fail("nested expansion")
    with pytest.raises(budget.BudgetError, match="stale"):
        with budget.reserve(policy=policy):
            pytest.fail("stale inherited token")


@pytest.mark.parametrize("contents", ["{broken", "{}", '{"schema_version": 1}'])
def test_malformed_registry_fails_closed(policy, tmp_path, contents):
    active = tmp_path / "state/budget/active"
    active.mkdir(parents=True)
    (active / ("a" * 32 + ".json")).write_text(contents)
    with pytest.raises(budget.BudgetError, match="malformed"):
        budget.protected_paths(policy)
    with pytest.raises(budget.BudgetError):
        with budget.reserve(0, policy=policy, wait=False):
            pytest.fail("corrupt state admitted")


def _holder(policy, tmp_path):
    source = ("import sys,json;sys.path.insert(0,sys.argv[1]);"
              "import storage_budget as b,storage_tiers as t;t.free_gb=lambda _:40;"
              "p=json.loads(sys.argv[2]);"
              "ctx=b.reserve(8,paths=[sys.argv[3]],archive_paths=[sys.argv[3]],policy=p);"
              "ctx.__enter__();print('READY',flush=True);sys.stdin.readline();ctx.__exit__(None,None,None)")
    child = subprocess.Popen([sys.executable, "-c", source, str(Path(budget.__file__).parent),
                              json.dumps(policy), str(tmp_path / "owned")],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert child.stdout.readline().strip() == "READY"
    return child


def test_cross_process_lease_capacity_and_crash_recovery(policy, tmp_path):
    child = _holder(policy, tmp_path)
    try:
        with pytest.raises(budget.BudgetError, match="reserved 8.00"):
            with budget.reserve(3, policy=policy, wait=False):
                pytest.fail("cross-process oversubscription")
        assert str(tmp_path / "owned") in budget.protected_paths(policy)
        child.kill()
        child.wait(timeout=10)
        for _ in range(100):  # Windows releases a killed process's byte-range lock a moment after wait() returns
            if not budget.protected_paths(policy):
                break
            time.sleep(0.05)
        assert budget.protected_paths(policy) == []
        assert budget.completed_paths(policy)[0][0] == str(tmp_path / "owned")
        with budget.reserve(10, policy=policy, wait=False):
            pass
        receipts = list((tmp_path / "state/budget/receipts").glob("*.json"))
        assert any(json.loads(p.read_text())["status"] == "owner_exited" for p in receipts)
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_wait_rechecks_capacity(policy, monkeypatch):
    calls = []
    monkeypatch.setattr(tiers, "free_gb", lambda _: 34 if not calls else 40)
    monkeypatch.setattr(budget.time, "sleep", lambda _: calls.append(True))
    with budget.reserve(5, policy=policy, log=lambda _: None):
        assert calls


def test_wait_logs_and_nudges_once_until_capacity_changes(policy, monkeypatch):
    polls = []
    messages = []
    nudges = []
    monkeypatch.setattr(tiers, "free_gb", lambda _: 34 if len(polls) < 3 else 40)
    monkeypatch.setattr(budget.time, "sleep", lambda _: polls.append(True))
    monkeypatch.setattr(budget, "nudge", lambda *a, **kw: nudges.append(kw["reason"]))
    with budget.reserve(5, policy=policy, log=messages.append):
        pass
    assert len(polls) == 3
    assert len(messages) == 1
    assert nudges == ["capacity_wait"]


def test_temporary_and_custom_policy_never_launch_live_task(policy, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("temporary policy triggered live task")
    monkeypatch.setattr(budget.subprocess, "run", forbidden)
    assert budget.nudge(policy, reason="test")["status"] == "custom_policy_skipped"
    monkeypatch.setattr(tiers, "load_policy", lambda: policy)
    monkeypatch.setenv("LSS_STORAGE_POLICY", "explicit.json")
    assert budget.nudge(policy, reason="test")["status"] == "custom_policy_skipped"
    monkeypatch.delenv("LSS_STORAGE_POLICY")
    monkeypatch.setenv("LSS_STORAGE_POLICY_JSON", json.dumps(policy))
    assert budget.nudge(policy, reason="test")["status"] == "custom_policy_skipped"


def test_live_nudge_uses_existing_task_outside_lock_and_throttles(policy, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(budget, "_live_policy", lambda _: True)
    def requested(command, **kwargs):
        calls.append((command, kwargs))
        # A maintenance task must be able to acquire the gate during invocation.
        with budget._mutex(budget._root(policy)):
            pass
        return subprocess.CompletedProcess(command, 0, "SUCCESS", "")
    monkeypatch.setattr(budget.subprocess, "run", requested)
    with budget.reserve(0, paths=[tmp_path], archive_paths=[tmp_path / "outputs"], policy=policy):
        pass
    assert len(calls) == 1
    assert calls[0][0] == ["schtasks.exe", "/Run", "/TN", "LSS Storage Tiers"]
    assert calls[0][1]["timeout"] == 10
    assert calls[0][1]["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert calls[0][1]["errors"] == "replace"
    assert calls[0][1]["encoding"] == ("oem" if os.name == "nt" else "utf-8")
    assert budget.nudge(policy, reason="another")["status"] == "throttled"
    assert len(calls) == 1
    status = json.loads((tmp_path / "state/budget/nudge.json").read_text())
    assert status["reason"] == "explicit_output_completed"


def test_nudge_failure_does_not_replace_job_failure(policy, monkeypatch, tmp_path):
    messages = []
    monkeypatch.setattr(budget, "_live_policy", lambda _: True)
    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)
    monkeypatch.setattr(budget.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="original job failure"):
        with budget.reserve(0, paths=[tmp_path], archive_paths=[tmp_path / "outputs"],
                            policy=policy, log=messages.append):
            raise RuntimeError("original job failure")
    assert any("nudge unavailable" in message for message in messages)
    assert budget.protected_paths(policy) == []


@pytest.mark.parametrize("returncode,expected", [(0, "requested"), (1, "failed")])
def test_nudge_absent_diagnostics_cannot_abort_admission(policy, monkeypatch, returncode, expected):
    monkeypatch.setattr(budget, "_live_policy", lambda _: True)
    monkeypatch.setattr(budget.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, returncode, None, None))
    result = budget.nudge(policy, reason="capacity_wait", log=lambda _: None)
    assert result["status"] == expected
    assert result["returncode"] == returncode
    assert result["detail"] == ""


def test_cli_nested_policy_exit_and_receipt(policy, tmp_path):
    environment = dict(os.environ, LSS_STORAGE_POLICY_JSON=json.dumps(policy))
    command = budget.command([sys.executable, "-c", "raise SystemExit(7)"],
                             paths=[tmp_path], archive_paths=[tmp_path / "result"], required_gb=0)
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == 7, result.stderr
    receipts = list((tmp_path / "state/budget/receipts").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "failed"
    assert budget.protected_paths(policy) == []


def test_invalid_estimates_and_archive_scope(policy, tmp_path):
    for amount in [-1, float("nan"), float("inf"), True]:
        with pytest.raises(budget.BudgetError):
            with budget.reserve(amount, policy=policy):
                pass
    with pytest.raises(budget.BudgetError, match="contained"):
        with budget.reserve(paths=[tmp_path / "a"], archive_paths=[tmp_path / "b"], policy=policy):
            pass


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
@pytest.mark.parametrize("kill_runner", [False, True])
def test_cli_owns_descendants_on_exit_or_runner_death(policy, tmp_path, kill_runner):
    heartbeat = tmp_path / "heartbeat"
    writer = ("import time,pathlib; p=pathlib.Path(" + repr(str(heartbeat)) + "); "
              "end=time.monotonic()+5; "
              "exec('while time.monotonic()<end:\\n p.write_text(str(time.monotonic()))\\n time.sleep(.03)')")
    parent = ("import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(writer) + "]);"
              + ("time.sleep(10)" if kill_runner else "time.sleep(.8)"))
    environment = dict(os.environ, LSS_STORAGE_POLICY_JSON=json.dumps(policy))
    runner = subprocess.Popen(budget.command([sys.executable, "-c", parent], required_gb=0),
                              env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 5
        while not heartbeat.exists() and runner.poll() is None and time.monotonic() < deadline:
            time.sleep(.03)
        assert heartbeat.exists(), runner.communicate(timeout=3)
        if kill_runner:
            runner.kill()
        runner.communicate(timeout=5)
        time.sleep(.2)
        stopped = heartbeat.read_text()
        time.sleep(.2)
        assert heartbeat.read_text() == stopped, "descendant outlived reservation supervisor"
        assert budget.protected_paths(policy) == []
    finally:
        if runner.poll() is None:
            runner.kill()
        runner.communicate(timeout=5)
