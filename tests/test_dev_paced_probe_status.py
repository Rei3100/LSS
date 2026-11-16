import importlib.util
import json
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "scripts/dev_paced_probe_status.py"
SPEC = importlib.util.spec_from_file_location("dev_paced_probe_status", MODULE)
STATUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATUS)


def _write_run(tmp_path, progress):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-receipt.json").write_text(json.dumps({
        "status": "RUNNING", "gpu_executed": False,
        "lease": {"pid": 4321},
    }), encoding="utf8")
    (run_dir / "progress.jsonl").write_text(progress, encoding="utf8")
    allowance = tmp_path / "latest.json"
    allowance.write_text(json.dumps({"status": "fresh", "relative_wallclock_rate_multiplier": None,
                                     "configured_unknown_rate_probe_fraction": .05,
                                     "windows": {"secondary": {"used_percent": 47,
                                                                  "relative_wallclock_rate_multiplier": None,
                                                                  "unknown_rate_probe_fraction": .05,
                                                                  "permitted_percent_per_minute": .004}}}),
                         encoding="utf8")
    return run_dir, allowance


def test_progress_event_overrides_stale_false_receipt_and_ignores_partial_tail(tmp_path):
    event = {"phase": "forward_rest", "elapsed_seconds": 65., "wall_time": 100.,
             "completed_at": 100., "sleep_seconds": 60.,
             "cuda_memory_observation": {"peak_memory_allocated_bytes": 99,
                                          "current_memory_allocated_bytes": 12}}
    run_dir, allowance = _write_run(tmp_path, json.dumps(event) + "\n{" )
    result = STATUS.snapshot(run_dir, allowance, now=105., process_probe=lambda _: "running")
    assert result["receipt_gpu_executed_snapshot"] is False
    assert result["gpu_execution_observed_from_progress"] is True
    assert result["progress"]["event_count"] == 1
    assert result["progress"]["observed_inter_forward_intervals"] == 0
    assert result["progress"]["latest_cuda_memory_observation"]["current_memory_allocated_bytes"] == 12
    assert result["process_visibility"] == "running"
    assert result["allowance"]["configured_unknown_rate_probe_fraction"] == .05
    assert result["allowance"]["relative_wallclock_rate_multiplier"] is None
    assert result["allowance"]["windows"]["secondary"]["permitted_percent_per_minute"] == .004


def test_no_forward_events_remains_unknown_not_gpu_false_claim(tmp_path):
    run_dir, allowance = _write_run(tmp_path, '{"phase":"separating"}\n')
    result = STATUS.snapshot(run_dir, allowance, now=105., process_probe=lambda _: "unknown")
    assert result["receipt_gpu_executed_snapshot"] is False
    assert result["gpu_execution_observed_from_progress"] is False
    assert result["gpu_execution_observation"] == "unknown_no_forward_event"
    assert result["progress"]["event_count"] == 0
    assert result["process_visibility"] == "unknown"


def test_following_forward_event_confirms_rest_and_reports_minimum_delta(tmp_path):
    rows = [
        {"phase": "forward_rest", "elapsed_seconds": 5., "wall_time": 100., "completed_at": 100.},
        {"phase": "forward_rest", "elapsed_seconds": 66., "wall_time": 161., "completed_at": 161.},
        {"phase": "forward_rest", "elapsed_seconds": 127., "wall_time": 222., "completed_at": 222.},
    ]
    run_dir, allowance = _write_run(tmp_path, "".join(json.dumps(row) + "\n" for row in rows))
    result = STATUS.snapshot(run_dir, allowance, now=230., process_probe=lambda _: "running")
    assert result["progress"]["observed_inter_forward_intervals"] == 2
    assert result["progress"]["inter_forward_wallclock_gaps_seconds"] == [61., 61.]
    assert result["progress"]["minimum_observed_inter_forward_wallclock_gap_seconds"] == 61.
