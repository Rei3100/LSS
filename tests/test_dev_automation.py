from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_gpu_run_holds_the_lease_for_the_command_and_returns_its_exit_code(tmp_path, monkeypatch):
    import contextlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpu_run", SCRIPTS / "gpu_run.py")
    gpu_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_run)
    from lss.core import gpu_lease

    held = []

    @contextlib.contextmanager
    def lease(on_wait=None):
        held.append("in")
        yield
        held.append("out")
    monkeypatch.setattr(gpu_lease, "acquire", lease)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import storage_budget  # this test is about the lease; storage admission has its own tests
    monkeypatch.setenv(storage_budget.TOKEN_ENV, "supervised")
    monkeypatch.setattr(storage_budget, "reserve", lambda *a, **k: contextlib.nullcontext(
        argparse.Namespace(child_env=lambda: dict(os.environ))))
    monkeypatch.setattr(gpu_run, "HOLDER", tmp_path / "holder.json")
    probe = tmp_path / "probe.py"
    probe.write_text("import sys; sys.exit(7)", encoding="utf-8")
    assert gpu_run.main(["--purpose", "test", "--", sys.executable, str(probe)]) == 7
    assert held == ["in", "out"] and not (tmp_path / "holder.json").exists()

def test_gpu_run_in_process_keeps_lease_holder_pid_and_script_arguments(tmp_path, monkeypatch):
    import contextlib
    import importlib.util
    import json
    import os

    spec = importlib.util.spec_from_file_location("gpu_run_inprocess", SCRIPTS / "gpu_run.py")
    gpu_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_run)
    from lss.core import gpu_lease

    held = []

    @contextlib.contextmanager
    def lease(on_wait=None):
        held.append("in")
        try:
            yield
        finally:
            held.append("out")

    monkeypatch.setattr(gpu_lease, "acquire", lease)
    holder = tmp_path / "holder.json"
    monkeypatch.setattr(gpu_run, "HOLDER", holder)
    script = tmp_path / "probe.py"
    script.write_text(
        "import json, os, sys\n"
        "row=json.load(open(sys.argv[2], encoding='utf-8'))\n"
        "assert sys.argv[1] == 'probe-arg'\n"
        "assert row['execution_mode'] == 'in_process'\n"
        "assert row['pid'] == os.getpid()\n"
        "assert row['command'][1].endswith('probe.py')\n"
        "sys.exit(7)\n",
        encoding="utf-8",
    )
    old_argv = sys.argv
    result = gpu_run.main(["--purpose", "in-process test", "--in-process", "--",
                           sys.executable, str(script), "probe-arg", str(holder)])

    assert result == 7
    assert sys.argv is old_argv
    assert held == ["in", "out"] and not holder.exists()


def test_gpu_run_in_process_restores_argv_and_releases_lease_on_exception(tmp_path, monkeypatch):
    import contextlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpu_run_inprocess_error", SCRIPTS / "gpu_run.py")
    gpu_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_run)
    from lss.core import gpu_lease

    held = []

    @contextlib.contextmanager
    def lease(on_wait=None):
        held.append("in")
        try:
            yield
        finally:
            held.append("out")

    monkeypatch.setattr(gpu_lease, "acquire", lease)
    holder = tmp_path / "holder.json"
    monkeypatch.setattr(gpu_run, "HOLDER", holder)
    script = tmp_path / "raises.py"
    script.write_text("raise RuntimeError('probe failed')\n", encoding="utf-8")
    old_argv = sys.argv

    with pytest.raises(RuntimeError, match="probe failed"):
        gpu_run.main(["--in-process", "--", sys.executable, str(script)])

    assert sys.argv is old_argv
    assert held == ["in", "out"] and not holder.exists()


def test_gpu_run_invalid_in_process_command_does_not_acquire_lease(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpu_run_invalid_inprocess", SCRIPTS / "gpu_run.py")
    gpu_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_run)
    from lss.core import gpu_lease

    monkeypatch.setattr(gpu_lease, "acquire", lambda **kwargs: pytest.fail("lease must not be acquired"))
    script = tmp_path / "probe.py"
    script.write_text("pass\n", encoding="utf-8")
    other_python = tmp_path / "other-python.exe"
    other_python.write_bytes(b"not executed")
    with pytest.raises(ValueError, match="current Python executable"):
        gpu_run.main(["--in-process", "--", str(other_python), str(script)])


def test_lanes_lists_recent_agent_branches():
    result = subprocess.run([sys.executable, str(SCRIPTS / "lanes.py"), "--hours", "100000", "--limit", "3"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and result.stdout.startswith("Active lanes")
