import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scene-web-evidence" / "run_cooperative_sfc_probe.py"
SPEC = importlib.util.spec_from_file_location("cooperative_sfc_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_forward_hook_synchronizes_samples_memory_then_sleeps_and_preserves_output():
    order = []

    class Cuda:
        def synchronize(self):
            order.append("sync")

        def memory_allocated(self):
            order.append("allocated")
            return 123

        def empty_cache(self):
            order.append("empty_cache")

        def memory_reserved(self):
            order.append("reserved")
            return 456

    class Torch:
        cuda = Cuda()

    events = []
    original_output = object()
    hook = probe.paced_hook(Torch, events, lambda seconds: order.append(("sleep", seconds)))
    assert hook(object(), (), original_output) is None
    assert order == ["sync", "empty_cache", "allocated", "reserved", ("sleep", 60.0)]
    assert events[0]["allocated_bytes"] == 123
    assert events[0]["reserved_bytes"] == 456
    assert events[0]["sleep_seconds"] == 60.0


def test_output_root_must_be_fresh_and_inside_state(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".state").mkdir(parents=True)
    good = probe.validate_output(repo / ".state" / "attempt-1", repo)
    assert good == (repo / ".state" / "attempt-1").resolve()

    outside = tmp_path / "outside"
    with pytest.raises(RuntimeError, match="new directory inside"):
        probe.validate_output(outside, repo)

    good.mkdir()
    with pytest.raises(RuntimeError, match="new directory inside"):
        probe.validate_output(good, repo)


def test_input_check_uses_mono_decode_and_records_native_stereo_separately():
    import numpy as np

    mono = np.array([0.25, -0.5], dtype=np.float32)
    native = np.column_stack((mono, -mono)).astype(np.float32)
    meta = {"decoded_float32_sha256": hashlib.sha256(mono.tobytes()).hexdigest()}
    verified = probe.verify_decoded_input(meta, native, 48000, mono)
    assert verified["decoded_mono_float32_sha256"] == meta["decoded_float32_sha256"]
    assert verified["native_pcm_float32_sha256"] == hashlib.sha256(native.tobytes()).hexdigest()
    assert verified["native_rate"] == 48000
    assert verified["native_channels"] == 2
    with pytest.raises(RuntimeError, match="mono-decoded"):
        probe.verify_decoded_input(meta, native, 48000, native)


def test_lease_requires_named_direct_parent(tmp_path, monkeypatch):
    holder = tmp_path / "holder.json"
    holder.write_text(json.dumps({"purpose": "test lease", "pid": 41}), encoding="utf-8")
    assert probe.require_lease(holder, parent_pid=41) == "test lease"
    monkeypatch.setattr(probe, "_venv_launcher_parent", lambda parent, owner: False)
    with pytest.raises(RuntimeError, match="direct parent"):
        probe.require_lease(holder, parent_pid=42)
    monkeypatch.setattr(probe, "_venv_launcher_parent", lambda parent, owner: parent == 42 and owner == 41)
    assert probe.require_lease(holder, parent_pid=42) == "test lease"
