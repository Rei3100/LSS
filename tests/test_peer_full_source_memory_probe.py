import importlib.util
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "scene-web-evidence/peer_full_source_memory_probe.py"
SPEC = importlib.util.spec_from_file_location("peer_full_source_memory_probe", MODULE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def valid_envelope():
    analysis = {"path": "analysis.wav", "sha256": "a" * 64, "container_sha256": "a" * 64,
                "sample_rate": 44100, "channels": 2, "frames": 500000, "subtype": "FLOAT"}
    workload = {"blocks": 1, "channels": 2, "maximum_forwards": 20,
                "per_block": [{"block_index": 0, "source_frames": [0, 500000]}]}
    source_alias = str(probe.EXPECTED_SOURCE)
    prepared = {"status": "PREPARED_GPU_NOT_RUN",
                "source": {"path": source_alias, "container_sha256": probe.EXPECTED_SOURCE_SHA256},
                "analysis_copy": analysis, "expected_forward_workload": workload}
    cadence = [{"sleep_seconds": 60., "rest_elapsed_seconds": 60.1}]
    worker = {"status": "COMPLETED_CUDA_WITHOUT_FALLBACK", "execution_profile": probe.PROFILE,
              "blocks": None, "input_role": "mixture_analysis_copy",
              "analysis": {"sha256": analysis["container_sha256"], "sample_rate": 44100,
                           "channels": 2, "frames": 500000},
              "stems": {role: {"path": f"run/worker/{role}.wav", "sha256": role[0] * 64}
                        for role in probe.ROLES},
              "execution_policy": {"effective": {"deterministic_algorithms": True,
                  "deterministic_warn_only": False, "cudnn_deterministic": True,
                  "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8",
                  "matmul_allow_tf32": False}, "cadence_events": cadence}}
    run = {"status": "DEV_ONLY_FULL_SOURCE_SFC_REGEN_COMPLETE", "gpu_executed": True,
           "source": {"path": source_alias, "container_sha256": probe.EXPECTED_SOURCE_SHA256},
           "prepared_receipt": "prepare-receipt.json", "output": "run",
           "input_role": "mixture_analysis_copy", "profile": probe.PROFILE, "blocks": None,
           "expected_forward_workload": workload, "forward_count": 1,
           "stems": {role: {"path": f"run/worker/{role}.wav", "container_sha256": role[0] * 64}
                     for role in probe.ROLES}}
    return run, prepared, worker


def test_running_receipt_is_rejected_before_prepared_or_worker_paths(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-receipt.json").write_text(json.dumps({"status": "RUNNING", "gpu_executed": False,
                                                            "prepared_receipt": "", "worker_receipt": ""}))
    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("incomplete receipt must be gated before resolving dependent paths")
    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    with pytest.raises(probe.ProbeError, match="not a completed GPU result"):
        probe.load_completed_receipts(run_dir)


def test_incomplete_pacing_or_missing_full_grid_is_rejected():
    run, prepared, worker = valid_envelope()
    worker["execution_policy"]["cadence_events"] = []
    with pytest.raises(probe.ProbeError, match="forward/rest receipt"):
        probe.validate_completion_envelope(run, prepared, worker)

    run, prepared, worker = valid_envelope()
    prepared["expected_forward_workload"]["per_block"][0]["source_frames"] = [1, 500000]
    with pytest.raises(probe.ProbeError, match="gap or invalid block"):
        probe.validate_completion_envelope(run, prepared, worker)


def test_tampered_source_binding_is_rejected():
    run, prepared, worker = valid_envelope()
    prepared["source"]["container_sha256"] = "b" * 64
    with pytest.raises(probe.ProbeError, match="Prepared source binding differs"):
        probe.validate_completion_envelope(run, prepared, worker)


def test_positive_synthetic_prepare_schema_and_source_alias_never_resolve(monkeypatch):
    # Synthetic metadata envelope; no source, analysis or stem file is opened.
    run, prepared, worker = valid_envelope()
    analysis = prepared["analysis_copy"]
    analysis.update({"decoded_rate": 48000, "analysis_rate": 44100,
                    "resample": "scipy.signal.resample_poly up=147 down=160",
                    "path": str(Path('synthetic-analysis') / 'analysis.wav'),
                    "frames": 65268000, "sha256": "b" * 64, "container_sha256": "b" * 64})
    worker["analysis"].update({"sha256": "b" * 64, "sample_rate": 44100,
                               "channels": 2, "frames": 65268000})
    ranges = [[index * 5292000, min((index + 1) * 5292000, 65268000)] for index in range(13)]
    css_counts = [21, *([23] * 11), 8]
    workload = {"blocks": 13, "channels": 2, "css_forwards_per_channel": 282,
                "maximum_forwards": 564, "maximum_scheduled_rest_seconds": 33840.0,
                "maximum_scheduled_rest_hours": 9.4, "cadence_seconds_after_forward": 60.0,
                "note": "Upper bound assumes every CSS segment is non-silent; zero segments skip model forwards.",
                "per_block": [{"block_index": i, "source_frames": span,
                               "context_extended_frames": (5821200 if i == 0 else
                                   2293200 if i == 12 else 6350400),
                               "css_forwards_per_channel": css_counts[i]}
                              for i, span in enumerate(ranges)]}
    prepared["expected_forward_workload"] = deepcopy(workload)
    prepared["runtime_bindings"] = {"modules": {}}
    run["expected_forward_workload"] = deepcopy(workload)
    run["forward_count"] = 1
    # Compare the source alias lexically without following a possible junction.
    expected = os.path.normcase(os.path.abspath(str(probe.EXPECTED_SOURCE)))
    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("readable source aliases must not be resolved")
    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    identity = probe.validate_completion_envelope(run, prepared, worker)
    assert identity["source_path"] == expected
    assert identity["analysis"]["frames"] == 65268000


@pytest.mark.parametrize("bad_rest", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rest_is_rejected(bad_rest):
    run, prepared, worker = valid_envelope()
    worker["execution_policy"]["cadence_events"][0]["rest_elapsed_seconds"] = bad_rest
    with pytest.raises(probe.ProbeError, match="Paced rest"):
        probe.validate_completion_envelope(run, prepared, worker)


@pytest.mark.parametrize("bad_window", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_held_window_is_rejected(bad_window):
    with pytest.raises(probe.ProbeError, match="invalid held span"):
        probe.windows_from_plan({"cues": [{"cue_id": "bad", "disposition": "held",
                                            "start_seconds": bad_window, "end_seconds": 2.0}]})


def test_output_inventory_detects_one_byte_tamper(tmp_path):
    payload = tmp_path / "proof.json"
    payload.write_bytes(b"a")
    receipt = {"outputs_sha256": {"proof.json": hashlib.sha256(b"a").hexdigest()}}
    assert probe.validate_inventory(tmp_path, receipt)["all_hashes_verified"] is True
    payload.write_bytes(b"b")
    with pytest.raises(probe.ProbeError, match="inventory mismatch"):
        probe.validate_inventory(tmp_path, receipt)


def test_short_window_is_retained_beside_eligible_zero_verified_lookup():
    segments = [
        {"cue_id": "short-jingle", "start_seconds": 1.0, "end_seconds": 5.5},
        {"cue_id": "full-query", "start_seconds": 10.0, "end_seconds": 20.0},
    ]
    lookups = [{"segment": [10.0, 20.0], "verified": 0}]
    accounting = probe.account_candidate_lookups(segments, lookups)
    assert [row["lookup_status"] for row in accounting["candidate_windows"]] == [
        "not_checked_due_to_native_min_query_seconds", "eligible_for_native_lookup"]
    assert accounting["eligible_windows"] == [[10.0, 20.0]]
    assert accounting["lookup_count"] == 1
    assert accounting["verified_alignments"] == 0
    assert probe.classify_memory_measurement(
        {"state": "checked", "lookups": lookups, "records": [],
         "peer_query_component_measurements": []}, accounting) == "NOMINATION_NONE"


def test_all_short_windows_are_not_misreported_as_nomination_none():
    segments = [{"cue_id": "jingle", "start_seconds": 0.0, "end_seconds": 5.999}]
    accounting = probe.account_candidate_lookups(segments, [])
    assert accounting["candidate_windows"][0]["lookup_status"] == "not_checked_due_to_native_min_query_seconds"
    assert probe.classify_memory_measurement({"state": "checked", "lookups": [], "records": [],
                                              "peer_query_component_measurements": []}, accounting) == "NO_ELIGIBLE_WINDOWS"


def test_automatic_window_mapping_keeps_every_held_candidate_only():
    plan = {"cues": [
        {"cue_id": "cue-001", "disposition": "held", "start_seconds": 18.25, "end_seconds": 41.0},
        {"cue_id": "cue-002", "disposition": "rejected_music_not_sustained",
         "start_seconds": 55.0, "end_seconds": 61.0},
        {"cue_id": "cue-003", "disposition": "held", "start_seconds": 92.0, "end_seconds": 106.5},
    ]}
    segments, spans = probe.windows_from_plan(plan)
    assert [row["cue_id"] for row in segments] == ["cue-001", "cue-003"]
    assert spans == [(18.25, 41.0), (92.0, 106.5)]
    assert all("offset" not in row for row in segments)
