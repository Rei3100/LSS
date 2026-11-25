"""scripts/batch.py: serial queue with receipts, logs, `after` markers, done marker; failures do not stop the queue; a failed required step or failed upstream marker fails the run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "scripts" / "batch.py"


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _run(queue: Path, expect: int = 0, *, storage_policy: dict | None = None) -> None:
    # Generic queue behavior; Codex admission has separate coverage.
    env = {k: v for k, v in os.environ.items()
           if k not in ("LSS_STORAGE_RESERVATION", "CODEX_SESSION_ID", "CODEX_THREAD_ID")}
    env.update(LSS_BATCH_POLL="0.1", LSS_MIN_COMMIT_HEADROOM_GB="0", LSS_MIN_FREE_RAM_GB="0")
    if storage_policy is not None:
        env["LSS_STORAGE_POLICY_JSON"] = json.dumps(storage_policy)
    done = subprocess.run([sys.executable, str(BATCH), "run", str(queue)], capture_output=True, text=True,
                          env=env, timeout=120)
    assert done.returncode == expect, done.stderr


def test_queue_runs_in_order_and_survives_failure(tmp_path, queue_storage_policy):
    queue = tmp_path / "q.json"
    queue.write_text(json.dumps({"done": "DONE", "steps": [
        {"name": "make marker", "cmd": _py("print('one'); open('MARK', 'w').write('x')"), "duty": 1},
        {"name": "fails", "cmd": _py("import sys; print('boom'); sys.exit(3)"), "duty": 1},
        {"name": "after marker", "cmd": _py("print('three')"), "after": ["MARK"], "duty": 1},
        {"name": "capped", "cmd": _py("print('four')")},
    ]}), encoding="utf-8")
    _run(queue, storage_policy=queue_storage_policy)
    receipt = json.loads((tmp_path / "q.receipt.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in receipt["steps"]] == ["make marker", "fails", "after marker", "capped"]
    assert [s["rc"] for s in receipt["steps"]] == [0, 3, 0, 0]
    assert receipt["ended"] and all(s["start"] <= s["end"] and s["duration"] >= 0 for s in receipt["steps"])
    assert (tmp_path / "q-make_marker.log").read_text().strip() == "one"
    assert "boom" in (tmp_path / "q-fails.log").read_text()
    assert (tmp_path / "q-after_marker.log").read_text().strip() == "three"
    assert (tmp_path / "q-capped.log").read_text().strip() == "four"
    lines = (tmp_path / "DONE").read_text().splitlines()
    assert len(lines) == 5 and lines[-1] == "status ok"
    assert receipt["status"] == "ok"
    assert [ln.startswith(p) for ln, p in zip(lines, ["make marker 0 ", "fails 3 ", "after marker 0 ", "capped 0 "])] == [True] * 4
    table = subprocess.run([sys.executable, str(BATCH), "status", str(queue)], capture_output=True, text=True)
    assert "FAILED" in table.stdout and "ok" in table.stdout


def test_required_step_failure_stops_queue(tmp_path, queue_storage_policy):
    queue = tmp_path / "r.json"
    queue.write_text(json.dumps({"done": "DONE", "steps": [
        {"name": "must", "cmd": _py("import sys; sys.exit(1)"), "required": True, "duty": 1},
        {"name": "never", "cmd": _py("print('x')"), "duty": 1},
    ]}), encoding="utf-8")
    _run(queue, expect=1, storage_policy=queue_storage_policy)
    receipt = json.loads((tmp_path / "r.receipt.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in receipt["steps"]] == ["must"]
    assert receipt["status"] == "failed must rc=1"
    marker = (tmp_path / "DONE").read_text().splitlines()          # still written, so watchers wake
    assert marker[0].startswith("must 1 ") and marker[-1] == "status failed must rc=1"


def test_queue_after_a_failed_marker_runs_nothing_and_fails(tmp_path):
    (tmp_path / "UP").write_text("must 1 0.1\nstatus failed must rc=1\n", encoding="utf-8")
    queue = tmp_path / "d.json"
    queue.write_text(json.dumps({"done": "DONE", "steps": [
        {"name": "dependent", "cmd": _py("open('RAN', 'w').write('x')"), "after": ["UP"], "duty": 1},
    ]}), encoding="utf-8")
    _run(queue, expect=1)
    receipt = json.loads((tmp_path / "d.receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed after UP" and receipt["steps"][0]["rc"] is None
    assert "UP" in receipt["steps"][0]["skipped"] and not (tmp_path / "RAN").exists()
    assert (tmp_path / "DONE").read_text().splitlines()[-1] == "status failed after UP"


def test_old_format_marker_without_status_counts_as_ok(tmp_path, queue_storage_policy):
    (tmp_path / "UP").write_text("must 0 0.1\n", encoding="utf-8")
    queue = tmp_path / "o.json"
    queue.write_text(json.dumps({"steps": [
        {"name": "dependent", "cmd": _py("open('RAN', 'w').write('x')"), "after": ["UP"], "duty": 1},
    ]}), encoding="utf-8")
    _run(queue, storage_policy=queue_storage_policy)
    assert (tmp_path / "RAN").exists()
    assert json.loads((tmp_path / "o.receipt.json").read_text(encoding="utf-8"))["status"] == "ok"


def test_a_job_waits_for_memory_headroom_and_starts_once_it_is_there(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import gpu_run

    readings = iter([(2.0, 1.0), (3.0, 20.0), (50.0, 20.0)])
    monkeypatch.setattr(gpu_run, "memory_headroom", lambda: next(readings))
    monkeypatch.setattr(gpu_run.time, "sleep", lambda _s: None)
    lines = []
    gpu_run.wait_for_memory("job", log=lines.append)
    assert len(lines) == 1 and "job waits" in lines[0]
    assert next(readings, None) is None                      # returned at the first reading with headroom
