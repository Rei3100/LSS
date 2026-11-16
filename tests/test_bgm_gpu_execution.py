"""Focused CPU controls for the internal SFC deterministic/cadence boundary."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import gpu_execution, sfc_analysis


class _Cudnn:
    deterministic = False
    benchmark = True
    allow_tf32 = True


class _Matmul:
    allow_tf32 = True


class _CudaBackend:
    matmul = _Matmul()


class _Torch:
    __version__ = "fake-torch"
    backends = type("Backends", (), {"cudnn": _Cudnn(), "cuda": _CudaBackend()})()

    def __init__(self):
        self.deterministic = False
        self.warn_only = False
        self.backends = type("Backends", (), {"cudnn": _Cudnn(), "cuda": _CudaBackend()})()
        self.backends.cuda.matmul = _Matmul()

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn_only

    def use_deterministic_algorithms(self, enabled, warn_only=False):
        self.deterministic, self.warn_only = enabled, warn_only


def test_profile_allowlist_is_narrow_and_profile_identity_is_distinct():
    assert gpu_execution.resolve_execution_profile(None) is None
    baseline = gpu_execution.resolve_execution_profile("strict_deterministic")
    paced = gpu_execution.resolve_execution_profile("strict_deterministic_paced60")
    assert baseline.cadence_seconds == 0 and paced.cadence_seconds == 60
    assert gpu_execution.profile_identity(None) is None
    assert gpu_execution.profile_identity(baseline.name) != gpu_execution.profile_identity(paced.name)
    for bad in ("", "paced", "strict_deterministic_paced30", 60, True):
        with pytest.raises(ValueError):
            gpu_execution.resolve_execution_profile(bad)


def test_configure_sets_cublas_before_torch_load_and_restores_flags(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "old-value")
    torch = _Torch()
    loaded = []

    def loader():
        loaded.append(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        return torch

    with gpu_execution.configure_execution("strict_deterministic", torch_loader=loader) as receipt:
        assert loaded == [":4096:8"]
        assert receipt["profile"] == "strict_deterministic"
        assert receipt["effective"]["deterministic_algorithms"] is True
        assert receipt["effective"]["deterministic_warn_only"] is False
        assert receipt["effective"]["cudnn_deterministic"] is True
        assert receipt["effective"]["cudnn_benchmark"] is False
        assert receipt["effective"]["matmul_allow_tf32"] is False
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == "old-value"
    assert torch.deterministic is False and torch.warn_only is False
    assert torch.backends.cudnn.deterministic is False and torch.backends.cudnn.benchmark is True
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_strict_profile_rejects_deterministic_warn_only_even_when_enabled(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _Torch()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with pytest.raises(RuntimeError, match="deterministic_warn_only"):
        gpu_execution.assert_profile_active("strict_deterministic", torch)


def test_default_configuration_has_no_import_or_environment_side_effect(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = []
    with gpu_execution.configure_execution(None, torch_loader=lambda: calls.append("loaded")) as receipt:
        assert receipt is None
    assert calls == []
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_paced_job_restores_startup_workspace_and_next_default_is_not_strict(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _Torch()
    with gpu_execution.configure_execution("strict_deterministic_paced60", torch_loader=lambda: torch):
        assert torch.deterministic is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.deterministic is False and torch.warn_only is False
    assert torch.backends.cudnn.deterministic is False and torch.backends.cudnn.benchmark is True
    assert torch.backends.cuda.matmul.allow_tf32 is True

    with gpu_execution.configure_execution(None, torch_loader=lambda: pytest.fail("default must not configure torch")) as receipt:
        assert receipt is None
    assert torch.deterministic is False
    assert torch.backends.cudnn.deterministic is False and torch.backends.cudnn.benchmark is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


class _HookHandle:
    def __init__(self, model, hook):
        self.model, self.hook = model, hook
        self.removed = False

    def remove(self):
        self.model.hooks.remove(self.hook)
        self.removed = True


class _Model:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return _HookHandle(self, hook)

    def forward(self, output):
        for hook in list(self.hooks):
            hook(self, (), output)
        return output


class _Cuda:
    def __init__(self, events):
        self.events = events

    def synchronize(self):
        self.events.append("synchronize")

    def empty_cache(self):
        self.events.append("empty_cache")

    def memory_allocated(self):
        self.events.append("allocated")
        return 123

    def memory_reserved(self):
        self.events.append("reserved")
        return 456


def test_paced_hook_observes_intact_forward_synchronizes_then_rests_and_is_removable():
    order = []
    torch = type("Torch", (), {"cuda": _Cuda(order)})()
    model, rows = _Model(), []
    ticks = iter((10.0, 70.0))
    monotonic_now = [0.0]

    def sleeper(seconds):
        order.append(("sleep", seconds))
        monotonic_now[0] += seconds

    handle = gpu_execution.attach_forward_cadence(
        model, torch, "strict_deterministic_paced60", rows,
        sleeper=sleeper, clock=lambda: next(ticks), monotonic=lambda: monotonic_now[0])
    output = object()
    assert model.forward(output) is output
    assert order == ["synchronize", "empty_cache", "allocated", "reserved", ("sleep", 60.0)]
    assert rows == [{"completed_at": 10.0, "allocated_bytes": 123, "reserved_bytes": 456,
                     "sleep_seconds": 60.0, "rest_elapsed_seconds": 60.0,
                     "rest_completed_at": 70.0}]
    gpu_execution.remove_forward_hook(handle)
    assert handle.removed and model.hooks == []


def test_paced_hook_uses_monotonic_deadline_when_wall_clock_moves_back_and_sleep_wakes_early():
    order = []
    torch = type("Torch", (), {"cuda": _Cuda(order)})()
    model, rows = _Model(), []
    monotonic_now = [100.0]
    wall_ticks = iter((900.0, 20.0))
    requested_sleeps = []

    def sleeper(seconds):
        requested_sleeps.append(seconds)
        # Simulate an OS sleep that returns early once; the hook must wait out the remainder.
        monotonic_now[0] += 25.0 if len(requested_sleeps) == 1 else seconds

    handle = gpu_execution.attach_forward_cadence(
        model, torch, "strict_deterministic_paced60", rows,
        sleeper=sleeper, clock=lambda: next(wall_ticks), monotonic=lambda: monotonic_now[0])
    output = object()
    assert model.forward(output) is output
    assert requested_sleeps == [60.0, 35.0]
    assert rows[0]["completed_at"] == 900.0
    assert rows[0]["rest_completed_at"] == 20.0
    assert rows[0]["rest_elapsed_seconds"] == 60.0
    assert order == ["synchronize", "empty_cache", "allocated", "reserved"]
    gpu_execution.remove_forward_hook(handle)
    assert handle.removed and model.hooks == []


def test_default_and_strict_baseline_install_no_forward_hook():
    model = _Model()
    torch = type("Torch", (), {"cuda": _Cuda([])})()
    for profile in (None, "strict_deterministic"):
        assert gpu_execution.attach_forward_cadence(model, torch, profile, []) is None
    assert model.hooks == []


def test_analyzer_id_binds_explicit_profile_but_omits_default_profile(monkeypatch):
    monkeypatch.setattr(sfc_analysis, "implementation_artifacts", lambda **_kwargs: (("stub", "a" * 64),))
    baseline = sfc_analysis.analyzer_revision()
    strict = sfc_analysis.analyzer_revision("strict_deterministic")
    paced = sfc_analysis.analyzer_revision("strict_deterministic_paced60")
    assert "execution_profile" not in baseline
    assert strict["execution_profile"] == "strict_deterministic"
    assert paced["execution_profile"] == "strict_deterministic_paced60"
    assert len({baseline["analyzer_id"], strict["analyzer_id"], paced["analyzer_id"]}) == 3


def test_worker_rejects_unknown_execution_profile_before_gpu(tmp_path, monkeypatch):
    from lss.core import worker

    source = tmp_path / "analysis.wav"
    sf.write(source, np.zeros((441, 2), dtype=np.float32), 44100, subtype="FLOAT")
    monkeypatch.setattr(worker.gpu_lease, "acquire", lambda **_kwargs: pytest.fail("GPU lease requested"))
    result = worker.handle_bgm_analysis_job({
        "cmd": "bgm_analysis", "analysis_path": str(source),
        "analysis_sha256": sfc_analysis._sha256(source), "analyzer_id": "irrelevant",
        "execution_profile": "unbounded-or-unknown",
    }, destination_dir=tmp_path / "out")
    assert result["status"] == "error" and result["error_code"] == "invalid_job"


def test_worker_receipt_binds_explicit_profile_and_runtime_flags(tmp_path, monkeypatch):
    from lss.core import worker

    source = tmp_path / "analysis.wav"
    sf.write(source, np.zeros((441, 2), dtype=np.float32), 44100, subtype="FLOAT")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"cached test weights")
    expected_profile = "strict_deterministic_paced60"
    revision = sfc_analysis.analyzer_revision(expected_profile)

    class Lease:
        def close(self):
            pass

    @contextmanager
    def configured(profile):
        assert profile == expected_profile
        yield {"profile": profile, "cadence_seconds_after_forward": 60.0,
               "effective": {"deterministic_algorithms": True}}

    monkeypatch.setattr(worker.gpu_lease, "acquire", lambda **_kwargs: Lease())
    monkeypatch.setattr(worker.runtime, "capture_cuda_runtime", lambda *_args, **_kwargs:
                        type("Runtime", (), {"to_dict": lambda self: {"runtime_id": "runtime"}})())
    monkeypatch.setattr(worker.vram, "reset_peak_vram", lambda: None)
    monkeypatch.setattr(worker.vram, "peak_vram_mb", lambda: 1.0)
    monkeypatch.setattr(worker.vram, "release_unused_vram", lambda: None)
    monkeypatch.setattr(worker.fetch, "file_sha256", sfc_analysis._sha256)
    monkeypatch.setattr(worker.session, "_write_json_atomic", lambda path, value:
                        Path(path).write_text(json.dumps(value), encoding="utf-8"))
    monkeypatch.setattr(sfc_analysis, "ensure_weights", lambda **_kwargs: weights)
    monkeypatch.setattr(gpu_execution, "configure_execution", configured)
    monkeypatch.setattr(sfc_analysis, "separate_stems", lambda *_args, **kwargs: {
        "analysis_sha256": sfc_analysis._sha256(source), "frames": 441, "channels": 2,
        "sample_rate": 44100, "blocks": None, "stems": {}, "attention_modules_patched": 12,
        "execution_profile": kwargs["execution_profile"], "cadence_events": [{"sleep_seconds": 60.0}],
    })
    result = worker.handle_bgm_analysis_job({
        "cmd": "bgm_analysis", "analysis_path": str(source),
        "analysis_sha256": sfc_analysis._sha256(source), "analyzer_id": revision["analyzer_id"],
        "execution_profile": expected_profile,
    }, destination_dir=output_dir)

    assert result["status"] == "ok", result
    receipt = json.loads((output_dir / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["execution_profile"] == expected_profile
    assert receipt["analyzer"]["execution_profile"] == expected_profile
    assert receipt["execution_policy"]["effective"]["deterministic_algorithms"] is True
    assert receipt["execution_policy"]["cadence_events"] == [{"sleep_seconds": 60.0}]
