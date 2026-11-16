"""Native scene-name backend: selection, revision, process lifecycle (fake engines), frozen launch, install."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lss import paths
from lss.bgm import scene_job, scene_models, scene_native, scene_runtime
from lss.core import gpu_lease

FFMPEG = Path(__file__).resolve().parents[1] / "bin" / "ffmpeg.exe"

FAKE_SERVER = textwrap.dedent('''
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    port, answer, record = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            open(record, "a", encoding="utf-8").write(json.dumps(body, ensure_ascii=False) + "\\n")
            data = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')
FAKE_WHISPER = textwrap.dedent('''
    import json, sys
    wav, output = sys.argv[1], sys.argv[2]
    rows = [{"offsets": {"from": 100, "to": 900}, "text": " 行こう "}, {"offsets": {"from": 1000, "to": 1500}, "text": ""}]
    json.dump({"transcription": rows}, open(output + ".json", "w", encoding="utf-8"), ensure_ascii=False)
''')


def _alive(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return code.value == 259  # STILL_ACTIVE


def _wait_gone(pid: int, seconds: float = 15.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def _fake_archive_proof(root, folder, monkeypatch):
    """Create pinned fake archives, then verify every installed engine member against them."""
    folder.mkdir(parents=True, exist_ok=True)
    archives = {}
    for engine, names in ((scene_native.WHISPER_ENGINE, scene_native.WHISPER_IDENTITY),
                          (scene_native.LLAMA_ENGINE, scene_native.LLAMA_IDENTITY)):
        for i, spec in enumerate(engine["archives"]):
            path = folder / spec["url"].rsplit("/", 1)[1]
            with zipfile.ZipFile(path, "w") as bundle:
                if i == 0:
                    for name in names:
                        bundle.write(root / engine["directory"] / name, spec["strip"] + name)
            monkeypatch.setitem(spec, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
            monkeypatch.setitem(spec, "bytes", path.stat().st_size)
            archives[path.name] = str(path)
    return archives


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """A native root whose files are tiny fakes; the pins are patched to match them."""
    root = tmp_path / "native"
    contents = {}
    for engine, names in ((scene_native.WHISPER_ENGINE, scene_native.WHISPER_IDENTITY),
                          (scene_native.LLAMA_ENGINE, scene_native.LLAMA_IDENTITY)):
        for name in names:
            path = root / engine["directory"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fake {engine['name']} {name}".encode())
    for spec, payload in ((scene_native.ASR_SPEC, b"asr weights"), (scene_native.LLM_SPEC, b"llm weights")):
        path = scene_native._weights_path(root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        contents[spec["file"]] = payload
        monkeypatch.setitem(spec, "bytes", len(payload))
        monkeypatch.setitem(spec, "sha256", hashlib.sha256(payload).hexdigest())
    archives = _fake_archive_proof(root, tmp_path / "verified-archives", monkeypatch)
    scene_native.write_manifest(root, local=archives, log=lambda *_: None)
    monkeypatch.setenv("LSS_SCENE_NATIVE_DIR", str(root))
    monkeypatch.setattr(scene_native, "_gpu_free_mib", lambda: None)
    return root


@pytest.fixture
def no_lease(monkeypatch):
    lease = SimpleNamespace(closed=0, close=lambda: setattr(lease, "closed", lease.closed + 1))
    monkeypatch.setattr(gpu_lease, "acquire", lambda *a, **k: lease)
    return lease


def _serve(monkeypatch, tmp_path, models, answer='{"name":"x"}'):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    record = tmp_path / "requests.jsonl"
    monkeypatch.setattr(models, "_llama_argv", lambda port: [sys.executable, str(script), str(port), answer, str(record)])
    return record


# --- selection ---------------------------------------------------------------------------------------------------------

def test_native_is_chosen_when_installed_and_works_frozen_or_from_source(fake_root, monkeypatch):
    monkeypatch.delenv("LSS_SCENE_BACKEND", raising=False)
    monkeypatch.setattr(scene_runtime, "transformers_available", lambda: True)
    for frozen in (True, False):
        monkeypatch.setattr(paths, "is_frozen", lambda frozen=frozen: frozen)
        assert scene_runtime.backend() == "native" and scene_runtime.available()
        assert isinstance(scene_runtime.create_models(), scene_native.NativeSceneModels)
        assert scene_job.available() is True


def test_without_native_files_the_frozen_app_has_no_backend_and_source_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_SCENE_NATIVE_DIR", str(tmp_path / "absent"))
    monkeypatch.delenv("LSS_SCENE_BACKEND", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    assert scene_runtime.backend() is None and scene_job.available() is False
    monkeypatch.setattr(paths, "is_frozen", lambda: False)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert scene_runtime.backend() == "transformers"
    assert type(scene_runtime.create_models()) is scene_models.SceneModels


def test_forced_backend_is_never_swapped(fake_root, monkeypatch):
    monkeypatch.setenv("LSS_SCENE_BACKEND", "transformers")
    monkeypatch.setattr(scene_runtime, "transformers_available", lambda: False)
    assert scene_runtime.backend() is None
    monkeypatch.setenv("LSS_SCENE_BACKEND", "native")
    assert scene_runtime.backend() == "native"
    (fake_root / scene_native.MANIFEST).unlink()
    assert scene_runtime.backend() is None


def test_a_changed_or_missing_engine_file_makes_native_unavailable(fake_root):
    assert scene_native.ready(fake_root)
    weights = scene_native._weights_path(fake_root, scene_native.LLM_SPEC)
    weights.write_bytes(b"llm weights changed")
    assert any(p.startswith("changed:") for p in scene_native._manifest_problems(fake_root))
    weights.unlink()
    assert not scene_native.ready(fake_root)


# --- revision ------------------------------------------------------------------------------------------------------------

def test_revision_records_backend_binaries_and_weight_hashes_and_differs_from_transformers(fake_root, monkeypatch, tmp_path):
    native = scene_native.revision(fake_root)
    assert native["backend"] == scene_native.BACKEND
    assert native["asr"]["sha256"] == scene_native.ASR_SPEC["sha256"]
    assert native["llm"]["sha256"] == scene_native.LLM_SPEC["sha256"] and native["llm"]["thinking"] is False
    assert native["llm"]["engine"]["tag"] == scene_native.LLAMA_ENGINE["tag"]
    assert native["asr"]["engine"]["files"]["whisper-cli.exe"] == hashlib.sha256(b"fake whisper.cpp whisper-cli.exe").hexdigest()
    transformers = scene_models.SceneModels().revision()
    assert "backend" not in transformers and native != transformers
    cue = {"pcm_sha256": "p"}
    keys = {scene_job._retry_key(fake_root, "s", cue, rev) for rev in (native, transformers)}
    assert len(keys) == 2, "an unresolved attempt of the other backend is retried, not silently equal"
    engine = fake_root / scene_native.LLAMA_ENGINE["directory"] / "llama-server.exe"
    engine.write_bytes(b"upgraded")
    archives = _fake_archive_proof(fake_root, tmp_path / "updated-archives", monkeypatch)
    scene_native.write_manifest(fake_root, local=archives, log=lambda *_: None)
    assert scene_native.revision(fake_root) != native, "a different binary is a different revision"


def test_scene_job_revision_follows_the_selected_backend(fake_root, monkeypatch):
    monkeypatch.delenv("LSS_SCENE_BACKEND", raising=False)
    assert scene_job._current_model_revision(None)["backend"] == scene_native.BACKEND
    monkeypatch.setenv("LSS_SCENE_BACKEND", "transformers")
    monkeypatch.setattr(scene_runtime, "transformers_available", lambda: True)
    assert "backend" not in scene_job._current_model_revision(None)


# --- lifecycle ------------------------------------------------------------------------------------------------------------

def test_llm_runs_through_the_chat_template_without_thinking_and_release_kills_the_server(
        fake_root, no_lease, monkeypatch, tmp_path):
    models = scene_native.NativeSceneModels()
    record = _serve(monkeypatch, tmp_path, models, answer='{"use":"食事","direction":"静かに","category":"日常","name":"ひととき","name_basis":"context_music"}')
    order = models.order("PROMPT")
    assert order["name"] == "ひととき" and order["name_basis"] == "context_music"
    request = json.loads(record.read_text(encoding="utf-8").splitlines()[0])
    assert request["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["max_tokens"] == scene_models.LLM["max_new_tokens"] == 260
    assert request["temperature"] == 0 and request["top_k"] == 1 and "response_format" not in request
    assert request["t_max_predict_ms"] == scene_models.LLM["max_seconds"] * 1000
    pid = models._server.process.pid
    models.order("again")  # same server, no second start
    assert models._server.process.pid == pid
    models.release()
    assert _wait_gone(pid) and models._server is None and no_lease.closed == 1
    assert not scene_native._LIVE


def test_exception_in_the_caller_still_leaves_no_process_when_release_runs(fake_root, no_lease, monkeypatch, tmp_path):
    models = scene_native.NativeSceneModels()
    _serve(monkeypatch, tmp_path, models)
    pid = None
    with pytest.raises(RuntimeError, match="boom"):
        try:
            models.order("x")
            pid = models._server.process.pid
            raise RuntimeError("boom")
        finally:
            models.release()
    assert _wait_gone(pid) and not scene_native._LIVE


def test_server_that_exits_or_never_gets_healthy_is_stopped_and_reported(fake_root, no_lease, monkeypatch):
    models = scene_native.NativeSceneModels()
    monkeypatch.setattr(models, "_llama_argv", lambda port: [sys.executable, "-c", "import sys; sys.exit(3)"])
    with pytest.raises(scene_models.SceneModelsError, match="exit 3"):
        models.order("x")
    assert models._server is None and not scene_native._LIVE
    sleeper = [sys.executable, "-c", "import time; time.sleep(600)"]
    monkeypatch.setattr(models, "_llama_argv", lambda port: sleeper)
    monkeypatch.setattr(scene_native, "HEALTH_SECONDS", 1.5)
    with pytest.raises(scene_models.SceneModelsError, match="時間内"):
        models.order("x")
    assert models._server is None and not scene_native._LIVE
    models.release()


def test_lease_failure_starts_no_process(fake_root, monkeypatch):
    def refuse(*a, **k):
        raise gpu_lease.GpuLeaseUnavailableError("injected")

    monkeypatch.setattr(gpu_lease, "acquire", refuse)
    monkeypatch.setattr(scene_native, "_Child", lambda *a, **k: pytest.fail("a process was started without the lease"))
    models = scene_native.NativeSceneModels()
    with pytest.raises(gpu_lease.GpuLeaseUnavailableError):
        models.order("x")
    with pytest.raises(gpu_lease.GpuLeaseUnavailableError):
        models.transcribe(Path("missing.mkv"), [(0.0, 1.0)])


def test_not_enough_free_vram_fails_before_a_model_is_started(fake_root, no_lease, monkeypatch):
    monkeypatch.setattr(scene_native, "_gpu_free_mib", lambda: 1000)
    monkeypatch.setattr(scene_native, "_Child", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(scene_models.SceneModelsError, match="GPU"):
        scene_native.NativeSceneModels().order("x")


def test_children_die_with_their_parent_process(tmp_path):
    """The Job Object (kill-on-close) removes a child even when this Python process is killed without cleanup."""
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(textwrap.dedent(f'''
        import os, sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from pathlib import Path
        from lss.bgm import scene_native
        child = scene_native._Child([sys.executable, "-c", "import time; time.sleep(600)"], cwd=Path({str(tmp_path)!r}),
                                    log=Path({str(tmp_path / "child.log")!r}), env=dict(os.environ))
        Path({str(pidfile)!r}).write_text(str(child.process.pid))
        os._exit(0)  # no atexit, no finally
    '''), encoding="utf-8")
    subprocess.run([sys.executable, str(script)], check=True, timeout=60)
    pid = int(pidfile.read_text())
    assert _wait_gone(pid), "child outlived its parent"


def test_transcribe_decodes_windows_runs_one_process_each_and_shifts_times(fake_root, no_lease, monkeypatch, tmp_path):
    source = tmp_path / "ep.wav"
    seconds = 40
    sf.write(source, (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000 * seconds) / 16000)).astype(np.float32), 16000, subtype="FLOAT")
    monkeypatch.setattr(paths, "resource_root", lambda: FFMPEG.parents[1])
    scene_models._ffmpeg_identity.cache_clear()
    script = tmp_path / "fake_whisper.py"
    script.write_text(FAKE_WHISPER, encoding="utf-8")
    seen = []
    monkeypatch.setattr(scene_native.NativeSceneModels, "_whisper_argv",
                        lambda self, wav, output: seen.append(Path(wav).name) or [sys.executable, str(script), str(wav), str(output)])
    models = scene_native.NativeSceneModels()
    lines = models.transcribe(source, [(0.0, 5.0), (3.0, 8.0), (20.0, 25.0)])
    assert seen == ["window0.wav", "window1.wav"], "overlapping windows merge; one process per merged window"
    assert lines == [{"start": 0.1, "end": 0.9, "text": "行こう"}, {"start": 20.1, "end": 20.9, "text": "行こう"}]
    receipt = models.last_transcription_receipt
    assert receipt["schema"] == "scene-asr-input-lineage-r1" and receipt["asr"]["backend"] == scene_native.BACKEND
    assert receipt["asr"]["revision"] == scene_native.revision(fake_root)["asr"]
    assert [w["source_start_seconds"] for w in receipt["windows"]] == [0.0, 20.0]
    assert receipt["windows"][0]["input"]["sample_count"] == 16000 * 8
    assert not scene_native._LIVE
    models.release()


def test_failed_whisper_process_raises_a_runtime_error_the_job_treats_as_asr_unavailable(
        fake_root, no_lease, monkeypatch, tmp_path):
    source = tmp_path / "ep.wav"
    sf.write(source, np.zeros(16000 * 5, dtype=np.float32), 16000, subtype="FLOAT")
    monkeypatch.setattr(paths, "resource_root", lambda: FFMPEG.parents[1])
    scene_models._ffmpeg_identity.cache_clear()
    monkeypatch.setattr(scene_native.NativeSceneModels, "_whisper_argv",
                        lambda self, wav, output: [sys.executable, "-c", "import sys; sys.exit(7)"])
    with pytest.raises(RuntimeError, match="exit 7"):
        scene_native.NativeSceneModels().transcribe(source, [(0.0, 3.0)])
    assert not scene_native._LIVE


# --- frozen launch -----------------------------------------------------------------------------------------------------------

def test_frozen_app_launches_the_scene_job_through_its_own_executable_when_native_is_installed(
        fake_root, monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("LSS_SCENE_BACKEND", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setattr(scene_job, "needs_naming", lambda *a, **k: True)
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kw: calls.append(command) or SimpleNamespace(pid=4242))
    result = scene_job.start_in_background(tmp_path / "state", tmp_path / "ep.mkv", "d" * 64)
    assert result["state"] == "started"
    assert calls[0][:2] == [sys.executable, "--scene-naming"] and "-m" not in calls[0]
    (fake_root / scene_native.MANIFEST).unlink()
    assert not scene_runtime.available()  # Readiness remains false until the worker verifies installation.
    assert scene_job.start_in_background(tmp_path / "state", tmp_path / "ep.mkv", "d" * 64)["state"] == "started"
    assert calls[-1][:2] == [sys.executable, "--scene-naming"]


def test_startup_recovery_needs_the_native_files_not_a_hugging_face_cache(fake_root, tmp_path, monkeypatch):
    monkeypatch.delenv("LSS_SCENE_BACKEND", raising=False)
    monkeypatch.setenv("LSS_SCENE_MODEL_DIR", str(tmp_path / "no-hf-cache"))
    assert scene_job._scene_models_cached() is True


# --- install -------------------------------------------------------------------------------------------------------------------

def test_install_extracts_only_the_needed_members_and_verifies_pins(tmp_path, monkeypatch):
    root = tmp_path / "native"
    archives = {}
    for engine, strip, names in ((scene_native.WHISPER_ENGINE, "Release/", scene_native.WHISPER_IDENTITY + ("bench.exe",)),
                                 (scene_native.LLAMA_ENGINE, "", scene_native.LLAMA_IDENTITY + ("llama-bench.exe", "LICENSE.txt"))):
        spec = engine["archives"][0]
        path = tmp_path / spec["url"].rsplit("/", 1)[1]
        with zipfile.ZipFile(path, "w") as bundle:
            for name in names:
                bundle.writestr(strip + name, f"{name}".encode())
        monkeypatch.setitem(spec, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        monkeypatch.setitem(spec, "bytes", path.stat().st_size)
        archives[path.name] = str(path)
        for extra in engine["archives"][1:]:
            path = tmp_path / extra["url"].rsplit("/", 1)[1]
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr("cudart64_13.dll", b"cudart")
            monkeypatch.setitem(extra, "sha256", hashlib.sha256(path.read_bytes()).hexdigest())
            monkeypatch.setitem(extra, "bytes", path.stat().st_size)
            archives[path.name] = str(path)
    for spec, payload in ((scene_native.ASR_SPEC, b"a"), (scene_native.LLM_SPEC, b"b")):
        path = tmp_path / spec["file"]
        path.write_bytes(payload)
        monkeypatch.setitem(spec, "bytes", 1)
        monkeypatch.setitem(spec, "sha256", hashlib.sha256(payload).hexdigest())
        archives[spec["file"]] = str(path)
    manifest = scene_native.install(root, local=archives, log=lambda *_: None)
    assert scene_native.ready(root) and manifest["schema"] == scene_native.MANIFEST_SCHEMA
    assert not (root / scene_native.WHISPER_ENGINE["directory"] / "bench.exe").exists()
    llama_license = root / scene_native.LLAMA_ENGINE["directory"] / "LICENSE.txt"
    assert llama_license.read_bytes() == b"LICENSE.txt"
    assert all("LICENSE" not in relative for relative in manifest["archive_verified_files"])
    runtime_dll = root / scene_native.LLAMA_ENGINE["directory"] / "cudart64_13.dll"
    assert runtime_dll.is_file()
    assert runtime_dll.relative_to(root).as_posix() in manifest["archive_verified_files"]
    runtime_dll.write_bytes(b"changed runtime dll")
    assert not scene_native.ready(root)
    runtime_dll.write_bytes(b"cudart")
    # Selected license payloads are pinned and checked even though readiness records runtime binaries only.
    llama_license.write_bytes(b"tampered license")
    (root / scene_native.MANIFEST).unlink()
    with pytest.raises(scene_models.SceneModelsError, match="engine_file_identity_mismatch:LICENSE.txt"):
        scene_native.install(root, local=archives, log=lambda *_: None)
    llama_license.write_bytes(b"LICENSE.txt")
    # Unexpected DLLs remain a hard member-set failure.
    unexpected_dll = root / scene_native.LLAMA_ENGINE["directory"] / "unexpected.dll"
    unexpected_dll.write_bytes(b"unexpected")
    with pytest.raises(scene_models.SceneModelsError, match="engine_member_set_mismatch"):
        scene_native.install(root, local=archives, log=lambda *_: None)
    unexpected_dll.unlink()
    # An existing executable cannot be blessed merely by hashing its own bytes.
    exe = root / scene_native.WHISPER_ENGINE["directory"] / scene_native.WHISPER_ENGINE["executable"]
    exe.write_bytes(b"untrusted engine")
    with pytest.raises(scene_models.SceneModelsError, match="engine_file_identity_mismatch"):
        scene_native.install(root, local=archives, log=lambda *_: None)
    assert not (root / scene_native.MANIFEST).exists()
    exe.write_bytes(scene_native.WHISPER_ENGINE["executable"].encode())
    bad = dict(archives, **{scene_native.ASR_SPEC["file"]: str(tmp_path / scene_native.LLM_SPEC["file"])})
    shutil_target = scene_native._weights_path(root, scene_native.ASR_SPEC)
    shutil_target.unlink()
    with pytest.raises(scene_models.SceneModelsError, match="固定した実体"):
        scene_native.install(root, local=bad, log=lambda *_: None)


def test_native_weights_are_outside_the_evicting_model_cache(monkeypatch, tmp_path):
    from lss.core import models_store

    monkeypatch.delenv("LSS_SCENE_NATIVE_DIR", raising=False)
    root = scene_native.native_root()
    cache = Path(models_store.cache_dir()).resolve()
    assert cache not in root.resolve().parents and root.resolve() != cache
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    assert scene_native.native_root().name == "scene-native"


def test_legacy_manifest_is_unready_until_archive_verified_then_warm_is_offline(fake_root, tmp_path, monkeypatch):
    manifest_path = fake_root / scene_native.MANIFEST
    current = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert current['schema'] == scene_native.MANIFEST_SCHEMA and scene_native.ready(fake_root)
    legacy = dict(current, schema='scene-native-manifest-r1')
    legacy.pop('archive_verified_files'); legacy.pop('archive_sha256')
    manifest_path.write_text(json.dumps(legacy), encoding='utf-8')
    assert not scene_native.ready(fake_root)
    archives = {p.name: str(p) for p in (tmp_path / 'verified-archives').iterdir()}
    real_install = scene_native.install
    installs = []
    def local_install(root):
        installs.append(1)
        return real_install(root, local=archives, log=lambda *_: None)
    monkeypatch.setattr(scene_native, 'install', local_install)
    monkeypatch.setattr(scene_native, '_bootstrap_capacity', lambda _: None)
    monkeypatch.setattr(scene_native, '_download', lambda *a, **k: pytest.fail('verified local archives should be reused'))
    scene_native.ensure_ready(fake_root)
    assert scene_native.ready(fake_root) and installs == [1]
    scene_native.ensure_ready(fake_root)
    assert installs == [1]


@pytest.mark.parametrize('bad_proof', ['missing', 'empty', 'incomplete', 'wrong_archive'])
def test_manifest_with_incomplete_archive_proof_is_not_ready(fake_root, bad_proof):
    manifest_path = fake_root / scene_native.MANIFEST
    doc = json.loads(manifest_path.read_text(encoding='utf-8'))
    if bad_proof == 'missing': doc.pop('archive_verified_files')
    elif bad_proof == 'empty': doc['archive_verified_files'] = []
    elif bad_proof == 'incomplete': doc['archive_verified_files'] = doc['archive_verified_files'][:1]
    else: doc['archive_sha256'] = {}
    manifest_path.write_text(json.dumps(doc), encoding='utf-8')
    assert not scene_native.ready(fake_root)
