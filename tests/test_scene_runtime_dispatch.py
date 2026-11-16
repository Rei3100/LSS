from __future__ import annotations

import builtins
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import lss_gui
from lss import paths
from lss.bgm import scene_job, scene_models
from lss.core import gpu_lease


def _launch(monkeypatch, tmp_path, frozen):
    calls = []
    monkeypatch.setattr(paths, "is_frozen", lambda: frozen)
    monkeypatch.setattr(scene_job, "available", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *args, **kwargs: True)

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=2718)

    monkeypatch.setattr(subprocess, "Popen", popen)
    result = scene_job.start_in_background(tmp_path / "state", tmp_path / "episode.mkv", "a" * 64)
    return result, calls


def test_background_launch_uses_frozen_executable_flag_or_development_module(monkeypatch, tmp_path):
    frozen, frozen_calls = _launch(monkeypatch, tmp_path, True)
    assert frozen["state"] == "started"
    command, kwargs = frozen_calls[0]
    assert command == [sys.executable, "--scene-naming", str((tmp_path / "state").resolve()),
                       str((tmp_path / "episode.mkv").resolve()), "a" * 64]
    assert kwargs["stdout"].name.endswith(".log") and kwargs["stderr"] == subprocess.STDOUT
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)

    monkeypatch.undo()
    development, development_calls = _launch(monkeypatch, tmp_path, False)
    assert development["state"] == "started"
    assert development_calls[0][0] == [sys.executable, "-m", "lss.bgm.scene_job",
                                        str((tmp_path / "state").resolve()),
                                        str((tmp_path / "episode.mkv").resolve()), "a" * 64]


def test_missing_scene_model_dependency_does_not_spawn_a_background_process(monkeypatch, tmp_path):
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setenv("LSS_SCENE_BACKEND", "transformers")  # A forced unavailable backend never falls back.
    monkeypatch.setattr(scene_job, "needs_naming", lambda *args, **kwargs: True)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: None if name == "bitsandbytes" else real_find_spec(name))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must stay unavailable"))

    assert scene_job.start_in_background(tmp_path / "state", tmp_path / "episode.mkv", "b" * 64) == {
        "state": "not_available"}


def test_scene_naming_cli_runs_without_importing_gui(monkeypatch, tmp_path):
    resource_root = tmp_path / "resources"
    resource_root.mkdir()
    calls = []
    monkeypatch.setattr(lss_gui, "_configure_runtime_caches", lambda: None)
    monkeypatch.setattr(lss_gui, "_ensure_worker_streams", lambda: None)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "resource_root", lambda: resource_root)
    monkeypatch.setattr(scene_job, "run_background_worker", lambda *args: calls.append(args) or {"named": 1})
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lss.ui.app":
            raise AssertionError("scene one-shot must not initialize the GUI")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    previous = Path.cwd()
    assert lss_gui.main(["--scene-naming", str(tmp_path / "state"), str(tmp_path / "episode.mkv"), "c" * 64]) == 0
    assert calls == [(tmp_path / "state", tmp_path / "episode.mkv", "c" * 64)]
    assert Path.cwd() == previous


def test_scene_naming_recovery_cli_runs_without_importing_gui(monkeypatch, tmp_path):
    resource_root = tmp_path / "resources"
    resource_root.mkdir()
    calls = []
    monkeypatch.setattr(lss_gui, "_configure_runtime_caches", lambda: None)
    monkeypatch.setattr(lss_gui, "_ensure_worker_streams", lambda: None)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(paths, "resource_root", lambda: resource_root)
    monkeypatch.setattr(scene_job, "run_current_best_recovery",
                        lambda root: calls.append(root) or {"state": "recovery_complete", "started": 0})
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "lss.ui.app":
            raise AssertionError("scene naming recovery must not initialize the GUI")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert lss_gui.main(["--scene-naming-recovery", str(tmp_path / "state")]) == 0
    assert calls == [tmp_path / "state"]


def test_host_started_dispatches_recovery_from_a_daemon_thread(monkeypatch):
    import threading
    from lss.ui.web.host import Host

    threads = []
    called = []

    class DeferredThread:
        def __init__(self, *, target, name, daemon):
            self.target, self.name, self.daemon = target, name, daemon
            threads.append(self)

        def start(self):
            called.append(("start", self.name))

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    host = Host()
    host._pump = lambda: None
    host._recover_current_best_scene_names = lambda: called.append(("recover",))

    host._started()

    assert [thread.name for thread in threads] == ["lss-web-push", "lss-scene-name-recovery-launch"]
    assert all(thread.daemon for thread in threads)
    assert called == [("start", "lss-web-push"), ("start", "lss-scene-name-recovery-launch")]
    threads[1].target()
    assert called[-1] == ("recover",)


def test_scene_model_does_not_touch_cuda_when_gpu_lease_cannot_be_acquired(monkeypatch):
    cuda_calls = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        set_per_process_memory_fraction=lambda *args: cuda_calls.append(args)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fail_acquire():
        raise gpu_lease.GpuLeaseUnavailableError("injected lease failure")

    monkeypatch.setattr(gpu_lease, "acquire", fail_acquire)
    with pytest.raises(gpu_lease.GpuLeaseUnavailableError):
        scene_models.SceneModels()._gpu()
    assert cuda_calls == []


@pytest.mark.parametrize("failure_stage", ["cache", "naming", "arguments"])
def test_scene_background_failure_returns_without_gui_or_unhandled_exception(monkeypatch, tmp_path, capsys, failure_stage):
    import json
    real_import = builtins.__import__
    def reject_gui(name, *args, **kwargs):
        if name == "lss.ui.app":
            pytest.fail("background failure must not fall through to GUI")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_gui)
    def fail(*args):
        raise scene_job.SourceIdentityMismatch("原本が変更されました")
    monkeypatch.setattr(lss_gui, "_configure_runtime_caches", fail if failure_stage == "cache" else lambda: None)
    monkeypatch.setattr(lss_gui, "_ensure_worker_streams", lambda: None)
    monkeypatch.setattr(lss_gui, "_run_scene_naming", fail)
    args = ["--scene-naming", str(tmp_path), str(tmp_path / "source.mkv"), "a" * 64]
    if failure_stage == "arguments":
        args = ["--scene-naming"]
    assert lss_gui.main(args) == 1
    result = json.loads(capsys.readouterr().err)
    assert result["state"] == "failed" and result["operation"] == "scene_naming"
    assert result["error_type"] == ("ValueError" if failure_stage == "arguments" else "SourceIdentityMismatch")


def test_scene_background_failure_survives_an_unusable_diagnostic_stream(monkeypatch):
    import io
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(lss_gui.sys, "stderr", stream)
    assert lss_gui.main(["--scene-naming"]) == 1


@pytest.mark.parametrize("failure_stage", ["cache", "naming"])
def test_scene_job_binds_missing_streams_to_append_log_before_failures(monkeypatch, tmp_path, failure_stage):
    import json

    state = tmp_path / "state"
    digest = "a" * 64
    log_path = state / "bgm-scene-names" / "jobs" / f"{digest[:16]}.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("previous receipt\n", encoding="utf-8")
    monkeypatch.setattr(lss_gui.sys, "stdout", None)
    monkeypatch.setattr(lss_gui.sys, "stderr", None)

    def fail(*args):
        raise scene_job.SourceIdentityMismatch("原本が変更されました")

    monkeypatch.setattr(lss_gui, "_configure_runtime_caches", fail if failure_stage == "cache" else lambda: None)
    monkeypatch.setattr(lss_gui, "_run_scene_naming", fail)

    assert lss_gui.main(["--scene-naming", str(state), str(tmp_path / "episode.mkv"), digest]) == 1
    assert lss_gui.sys.stdout is lss_gui.sys.stderr
    lss_gui.sys.stdout.flush()
    lss_gui.sys.stdout.close()
    contents = log_path.read_text(encoding="utf-8")
    assert contents.startswith("previous receipt\n")
    assert all(ord(char) < 128 for char in contents)
    payload = json.loads(contents.splitlines()[-1])
    assert payload["state"] == "failed" and payload["operation"] == "scene_naming"
    assert payload["error_type"] == "SourceIdentityMismatch"
    assert payload["message"] == "原本が変更されました"


def test_scene_job_writes_success_json_to_log_when_windowed_streams_are_none(monkeypatch, tmp_path):
    import json

    state = tmp_path / "state"
    digest = "b" * 64
    log_path = state / "bgm-scene-names" / "jobs" / f"{digest[:16]}.log"
    monkeypatch.setattr(lss_gui.sys, "stdout", None)
    monkeypatch.setattr(lss_gui.sys, "stderr", None)
    monkeypatch.setattr(lss_gui, "_configure_runtime_caches", lambda: None)
    monkeypatch.setattr(scene_job, "run_background_worker",
                        lambda *args: {"state": "complete", "attempts": 1, "result": {"named": 2}})

    assert lss_gui.main(["--scene-naming", str(state), str(tmp_path / "episode.mkv"), digest]) == 0
    assert lss_gui.sys.stdout is lss_gui.sys.stderr
    lss_gui.sys.stdout.flush()
    lss_gui.sys.stdout.close()
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload == {"state": "complete", "attempts": 1, "result": {"named": 2}}


def test_scene_job_rebinds_closed_streams_to_its_log(monkeypatch, tmp_path):
    import io
    import json

    out, err = io.StringIO(), io.StringIO()
    out.close()
    err.close()
    monkeypatch.setattr(lss_gui.sys, "stdout", out)
    monkeypatch.setattr(lss_gui.sys, "stderr", err)
    monkeypatch.setattr(lss_gui, "_configure_runtime_caches", lambda: None)
    monkeypatch.setattr(lss_gui, "_run_scene_naming", lambda *args: 0)
    digest = "c" * 64
    state = tmp_path / "state"
    log_path = state / "bgm-scene-names" / "jobs" / f"{digest[:16]}.log"

    assert lss_gui.main(["--scene-naming", str(state), str(tmp_path / "episode.mkv"), digest]) == 0
    assert lss_gui.sys.stdout is lss_gui.sys.stderr
    lss_gui.sys.stdout.write("closed stream recovered\n")
    lss_gui.sys.stdout.flush()
    lss_gui.sys.stdout.close()
    assert "closed stream recovered" in log_path.read_text(encoding="utf-8")


def test_scene_job_rejects_hostile_hash_before_constructing_log_path_or_routing_gui(monkeypatch, tmp_path, capsys):
    real_import = builtins.__import__

    def reject_gui(name, *args, **kwargs):
        if name == "lss.ui.app":
            pytest.fail("invalid background args must not fall through to GUI")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_gui)
    monkeypatch.setattr(lss_gui, "_scene_job_log_path", lambda *a: pytest.fail("must validate before path construction"))
    hostile = "a" * 59 + r"\..\x"
    assert len(hostile) == 64
    assert lss_gui.main(["--scene-naming", str(tmp_path / "state"), str(tmp_path / "episode.mkv"), hostile]) == 1
    import json
    payload = json.loads(capsys.readouterr().err)
    assert payload["state"] == "failed" and payload["message"] == "scene_naming_source_hash_invalid"
    assert not (tmp_path / "state").exists()


def test_scene_job_log_setup_failure_is_contained_and_uses_usable_stderr(monkeypatch, tmp_path, capsys):
    import json

    def fail_log_setup(*args):
        raise PermissionError("log denied")

    monkeypatch.setattr(lss_gui, "_bind_scene_job_log", fail_log_setup)
    monkeypatch.setattr(lss_gui, "_run_scene_naming", lambda *a: pytest.fail("log setup failure must stop worker"))

    assert lss_gui.main(["--scene-naming", str(tmp_path / "state"), str(tmp_path / "episode.mkv"), "d" * 64]) == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["state"] == "failed" and payload["error_type"] == "PermissionError"


def test_scene_model_closes_lease_even_if_model_cleanup_fails(monkeypatch):
    closed = []

    class Lease:
        def close(self):
            closed.append(True)

    model = scene_models.SceneModels()
    model._lease = Lease()

    def fail_cleanup(*args):
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(model, "_drop", fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failure"):
        model.release()

    assert closed == [True] and model._lease is None


def test_scene_transcription_uses_the_bundled_ffmpeg_resource(monkeypatch, tmp_path):
    bundled_ffmpeg = tmp_path / "resources" / "bin" / "ffmpeg.exe"
    bundled_ffmpeg.parent.mkdir(parents=True)
    bundled_ffmpeg.write_bytes(b"fixture binary marker")
    monkeypatch.setattr(paths, "resource_root", lambda: bundled_ffmpeg.parents[1])
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16=object()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **k: None))
    model = scene_models.SceneModels()
    model._asr = lambda *args, **kwargs: {"chunks": []}
    monkeypatch.setattr(model, "_gpu", lambda: None)
    monkeypatch.setattr(model, "_drop", lambda *args: None)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=(b"\0" * 4) * 1600)

    monkeypatch.setattr(scene_models.subprocess, "run", run)
    assert model.transcribe(tmp_path / "episode.mkv", [(0.0, 1.0)]) == []
    decode_call = next(call for call in calls if "-f" in call[0])
    assert decode_call[0][0] == str(bundled_ffmpeg)
    assert decode_call[1]["timeout"] == 600


def test_scene_transcription_receipt_binds_the_real_decoded_pipeline_array(monkeypatch, tmp_path):
    import hashlib
    import numpy as np
    import soundfile as sf

    bundled_ffmpeg = Path(__file__).resolve().parents[1] / "bin" / "ffmpeg.exe"
    source = tmp_path / "stereo-fixture.wav"
    time_axis = np.arange(16000, dtype=np.float32) / 16000
    stereo = np.column_stack((0.2 * np.sin(2 * np.pi * 440 * time_axis),
                              0.1 * np.sin(2 * np.pi * 660 * time_axis))).astype(np.float32)
    sf.write(source, stereo, 16000, subtype="FLOAT")
    monkeypatch.setattr(paths, "resource_root", lambda: bundled_ffmpeg.parents[1])
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16=object()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **k: None))
    monkeypatch.setattr(scene_models.importlib.metadata, "version", lambda name: f"{name}-fixture-version")
    scene_models._ffmpeg_identity.cache_clear()
    model = scene_models.SceneModels()
    captured = []

    def asr(inputs, **kwargs):
        captured.append((inputs["raw"].copy(), inputs["sampling_rate"], kwargs))
        return {"chunks": [{"timestamp": (0.1, 0.3), "text": " これはいい "}]}

    model._asr = asr
    monkeypatch.setattr(model, "_gpu", lambda: None)
    monkeypatch.setattr(model, "_drop", lambda *args: None)
    output = model.transcribe(source, [(0.0, 0.6), (0.5, 1.0)])

    assert output == [{"start": 0.1, "end": 0.3, "text": "これはいい"}]
    audio, sample_rate, options = captured[0]
    receipt = model.last_transcription_receipt
    window = receipt["windows"][0]
    assert sample_rate == 16000 and audio.dtype == np.float32 and audio.size == 16000
    assert hashlib.sha256(audio.tobytes()).hexdigest() == window["input"]["sha256"]
    assert window["input"] == {"sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                               "byte_length": audio.nbytes, "sample_count": audio.size,
                               "dtype": "float32-le", "sampling_rate": 16000,
                               "accepted_by_existing_minimum": True}
    assert window["source_start_seconds"] == 0.0 and window["source_end_seconds"] == 1.0
    assert window["ffmpeg_argv"] == [str(bundled_ffmpeg), "-v", "error", "-ss", "0.000", "-t", "1.000",
                                     "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    assert receipt["ffmpeg"]["binary_sha256"] == hashlib.sha256(bundled_ffmpeg.read_bytes()).hexdigest()
    assert receipt["ffmpeg"]["identity_status"] == "unchanged_at_observation_points"
    assert receipt["ffmpeg"]["version"].startswith("ffmpeg version")
    assert receipt["asr"]["revision"] == {k: scene_models.ASR[k] for k in
                                           ("repository", "revision", "chunk_seconds", "batch")}
    assert receipt["asr"]["generate_kwargs"] == {"language": "ja", "task": "transcribe"}
    assert receipt["asr"]["runtime"]["packages"]["transformers"]
    assert scene_models._ffmpeg_identity.cache_info().misses == 1
    assert options["chunk_length_s"] == scene_models.ASR["chunk_seconds"]
    assert options["batch_size"] == scene_models.ASR["batch"]
    assert not any(key in receipt for key in ("audio", "raw", "pcm", "samples"))


def test_ffmpeg_identity_rebinds_replaced_binary_at_the_same_path(monkeypatch, tmp_path):
    import hashlib

    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"harmless executable version one")
    scene_models._ffmpeg_identity.cache_clear()

    def version(command, **kwargs):
        return SimpleNamespace(stdout=b"ffmpeg version fixture " + Path(command[0]).read_bytes())

    monkeypatch.setattr(scene_models.subprocess, "run", version)
    old_stat = scene_models._file_stat_identity(executable)
    old_binding = scene_models._ffmpeg_identity(str(executable.resolve()), old_stat)
    executable.write_bytes(b"harmless executable replacement version two with different size")
    new_stat = scene_models._file_stat_identity(executable)
    new_binding = scene_models._ffmpeg_identity(str(executable.resolve()), new_stat)

    assert old_binding["binary_sha256"] == hashlib.sha256(b"harmless executable version one").hexdigest()
    assert new_binding["binary_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert old_binding["binary_sha256"] != new_binding["binary_sha256"]
    assert old_binding["version"] != new_binding["version"]
    assert old_stat != new_stat and scene_models._ffmpeg_identity.cache_info().misses == 2


def test_scene_transcription_marks_binary_unverified_if_path_changes_during_decode(monkeypatch, tmp_path):
    import hashlib

    executable = tmp_path / "resources" / "bin" / "ffmpeg.exe"
    executable.parent.mkdir(parents=True)
    initial_bytes = b"harmless fixture executable before decode"
    executable.write_bytes(initial_bytes)
    monkeypatch.setattr(paths, "resource_root", lambda: executable.parents[1])
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16=object()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **k: None))
    scene_models._ffmpeg_identity.cache_clear()
    model = scene_models.SceneModels()
    model._asr = lambda *args, **kwargs: {"chunks": []}
    monkeypatch.setattr(model, "_gpu", lambda: None)
    monkeypatch.setattr(model, "_drop", lambda *args: None)

    def run(command, **kwargs):
        if "-version" in command:
            return SimpleNamespace(stdout=b"ffmpeg version fixture")
        executable.write_bytes(b"harmless fixture executable replaced during decode")
        return SimpleNamespace(stdout=(b"\0" * 4) * 1600)

    monkeypatch.setattr(scene_models.subprocess, "run", run)
    model.transcribe(tmp_path / "episode.mkv", [(0.0, 1.0)])
    identity = model.last_transcription_receipt["ffmpeg"]

    assert identity["identity_status"] == "unverified_changed_during_decode"
    assert identity["binary_sha256"] is None and identity["version"] is None
    assert identity["binary_sha256_at_binding"] == hashlib.sha256(initial_bytes).hexdigest()
