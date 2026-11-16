from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lss.bgm import scene_models
from lss.core import gpu_lease


class _Lease:
    def __init__(self, held):
        self.held = held
        self.held[0] = True
        self.closed = False

    def close(self):
        self.closed = True
        self.held[0] = False


def _fake_torch(monkeypatch):
    cuda_calls = []
    cuda = SimpleNamespace(
        set_per_process_memory_fraction=lambda *args: cuda_calls.append(("fraction", args)),
        empty_cache=lambda: cuda_calls.append(("empty_cache",)),
    )
    torch = SimpleNamespace(float16="fp16", bfloat16="bf16", cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return cuda_calls


def _fake_transformers(monkeypatch, events, held=None):
    def pipeline(*args, **kwargs):
        if held is not None:
            assert held[0]
        events.append(("pipeline", kwargs["model"]))
        return lambda *a, **k: {"chunks": []}

    class Factory:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            events.append((cls.__name__, path))
            return SimpleNamespace(device="cuda:0", generate=lambda *a, **k: None)

    class AutoTokenizer(Factory):
        pass

    class AutoModelForCausalLM(Factory):
        pass

    transformers = SimpleNamespace(
        pipeline=pipeline,
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _lease_tracker(monkeypatch, events=None):
    held = [False]
    leases = []

    def acquire():
        if events is not None:
            events.append(("acquire",))
        lease = _Lease(held)
        leases.append(lease)
        return lease

    monkeypatch.setattr(gpu_lease, "acquire", acquire)
    return held, leases


def test_asr_weights_are_ensured_before_lease_and_pipeline_load(monkeypatch, tmp_path):
    events = []
    held, leases = _lease_tracker(monkeypatch, events)
    _fake_torch(monkeypatch)
    _fake_transformers(monkeypatch, events, held)

    def ensure(spec):
        assert not held[0]
        events.append(("ensure", spec["repository"]))
        return tmp_path

    monkeypatch.setattr(scene_models, "ensure", ensure)
    # transcribe() stats the bundled executable at resource_root()/bin/ffmpeg.exe (identity receipt).
    ffmpeg = tmp_path / "bin" / "ffmpeg.exe"
    ffmpeg.parent.mkdir()
    ffmpeg.write_bytes(b"fake")
    monkeypatch.setattr("lss.paths.resource_root", lambda: tmp_path)
    monkeypatch.setattr("lss.paths.is_frozen", lambda: True)
    monkeypatch.setattr(scene_models.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=np.zeros(1600, dtype=np.float32).tobytes()))

    model = scene_models.SceneModels()
    assert model.transcribe(Path("episode.mkv"), [(0., 1.)]) == []

    assert events == [("ensure", scene_models.ASR["repository"]), ("acquire",), ("pipeline", str(tmp_path))]
    assert held[0] and len(leases) == 1
    model.release()
    assert not held[0] and leases[0].closed


def test_asr_to_llm_switch_closes_old_lease_before_weight_ensure(monkeypatch):
    events = []
    held, leases = _lease_tracker(monkeypatch, events)
    _fake_torch(monkeypatch)
    _fake_transformers(monkeypatch, events, held)
    old = _Lease(held)
    model = scene_models.SceneModels()
    model._asr = object()
    model._lease = old

    def ensure(spec):
        events.append(("ensure", spec["repository"], held[0], model._asr))
        return Path("llm-snapshot")

    monkeypatch.setattr(scene_models, "ensure", ensure)
    model._load_llm()

    assert old.closed and not events[0][2] and events[0][3] is None
    assert events[0][:2] == ("ensure", scene_models.LLM["repository"])
    assert events[1] == ("acquire",)
    assert events[2] == ("AutoTokenizer", "llm-snapshot")
    assert events[3] == ("AutoModelForCausalLM", "llm-snapshot")
    assert held[0] and len(leases) == 1
    model.release()
    assert not held[0] and leases[0].closed


def test_llm_download_failure_releases_old_asr_before_raising(monkeypatch):
    held, _ = _lease_tracker(monkeypatch)
    _fake_torch(monkeypatch)
    _fake_transformers(monkeypatch, [])
    old = _Lease(held)
    model = scene_models.SceneModels()
    model._asr = object()
    model._lease = old

    def fail_ensure(spec):
        assert not held[0] and model._asr is None
        raise OSError("download failed")

    monkeypatch.setattr(scene_models, "ensure", fail_ensure)
    with pytest.raises(OSError, match="download failed"):
        model._load_llm()
    assert old.closed and not held[0] and model._lease is None


def test_repeated_llm_order_reuses_loaded_model_and_lease(monkeypatch):
    events = []
    held, leases = _lease_tracker(monkeypatch)
    _fake_torch(monkeypatch)
    _fake_transformers(monkeypatch, events)
    model = scene_models.SceneModels()
    model._llm = SimpleNamespace(device="cuda:0")
    model._tok = object()
    model._lease = _Lease(held)
    monkeypatch.setattr(scene_models, "ensure", lambda spec: pytest.fail("loaded LLM must be reused"))

    model._load_llm()
    first = model._llm
    model._load_llm()
    assert model._llm is first and held[0] and len(leases) == 0


def test_failed_lease_acquisition_happens_after_ensure_and_never_loads(monkeypatch, tmp_path):
    events = []
    _fake_torch(monkeypatch)
    _fake_transformers(monkeypatch, events)
    monkeypatch.setattr(scene_models, "ensure", lambda spec: events.append(("ensure", spec["repository"])) or tmp_path)

    def fail_acquire():
        events.append(("acquire",))
        raise gpu_lease.GpuLeaseUnavailableError("busy")

    monkeypatch.setattr(gpu_lease, "acquire", fail_acquire)
    # transcribe() stats the bundled executable at resource_root()/bin/ffmpeg.exe (identity receipt).
    ffmpeg = tmp_path / "bin" / "ffmpeg.exe"
    ffmpeg.parent.mkdir()
    ffmpeg.write_bytes(b"fake")
    monkeypatch.setattr("lss.paths.resource_root", lambda: tmp_path)
    monkeypatch.setattr("lss.paths.is_frozen", lambda: True)
    monkeypatch.setattr(scene_models.subprocess, "run", lambda *a, **k: pytest.fail("pipeline must not run"))

    with pytest.raises(gpu_lease.GpuLeaseUnavailableError):
        scene_models.SceneModels().transcribe(Path("episode.mkv"), [(0., 1.)])
    assert events == [("ensure", scene_models.ASR["repository"]), ("acquire",)]
