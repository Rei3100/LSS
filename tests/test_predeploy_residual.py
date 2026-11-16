"""Residual pre-deploy risks of the hidden scene-naming worker, with mocked engines/network and tiny files.

1. whisper.cpp receives only ASCII paths under the native root (never %TEMP%), cleaned on success/failure/stop.
2. A deploy/rollback stop request ends the worker between cues, download blocks and waits; partials are kept.
3. A pinned archive whose bytes no longer match backs off exponentially per pinned identity (supply problem).
4. Complete weights are hashed once per (size, mtime_ns); a ``.part`` file is never trusted from the cache.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lss.bgm import scene_job, scene_models, scene_names, scene_native as native, scene_runtime, scene_stop
from tests.test_bgm_scene_job import (FakeModels, _cue, library,  # noqa: F401  fixtures
                                      no_external_network_for_model_fixture_tests)

NON_ASCII = "ユーザー😀"


@pytest.fixture(autouse=True)
def fresh_stop_state(monkeypatch):
    monkeypatch.setattr(scene_stop, "_enabled_at", None)


# --- 1. ASCII ASR scratch ------------------------------------------------------------------------------------

class FakeWhisper:
    """Stands in for ``_Child``: records argv, writes whisper's JSON output, exits with ``code``."""
    started: list = []
    code = 0
    on_wait = None

    def __init__(self, argv, *, cwd, log, env):
        self.argv = argv
        FakeWhisper.started.append(self)
        output = Path(argv[argv.index("-of") + 1])
        assert Path(argv[argv.index("-f") + 1]).is_file()
        self.scratch = output.parent
        output.with_name(output.name + ".json").write_text(json.dumps({"transcription": [
            {"offsets": {"from": 0, "to": 900}, "text": "こんにちは"}]}), encoding="utf-8")
        self.process = SimpleNamespace(wait=self._wait, poll=lambda: None)

    def _wait(self, timeout=None):
        if FakeWhisper.on_wait:
            FakeWhisper.on_wait()
        return FakeWhisper.code

    def stop(self):
        pass

    def tail(self, characters=600):
        return ""


@pytest.fixture
def asr(monkeypatch, tmp_path):
    root = tmp_path / "native"
    root.mkdir()
    user_temp = tmp_path / NON_ASCII / "AppData" / "Local" / "Temp"
    user_temp.mkdir(parents=True)
    # %TEMP% contains the Windows user name: anything placed there is a non-ASCII path.
    monkeypatch.setattr(native.tempfile, "tempdir", str(user_temp))
    monkeypatch.setattr(native, "_manifest_problems", lambda _root: [])
    monkeypatch.setattr(native, "_require_vram", lambda _mib: None)
    monkeypatch.setattr(scene_models, "_ffmpeg_identity",
                        lambda exe, stat: {"identity_status": "unchanged_at_binding_checks"})
    monkeypatch.setattr(native.subprocess, "run", lambda argv, **k: SimpleNamespace(
        stdout=np.zeros(16000, dtype=np.float32).tobytes()))
    monkeypatch.setattr(native, "_Child", FakeWhisper)
    monkeypatch.setattr(FakeWhisper, "started", [])
    monkeypatch.setattr(FakeWhisper, "code", 0)
    monkeypatch.setattr(FakeWhisper, "on_wait", None)
    models = native.NativeSceneModels(root=root)
    gpu = []
    monkeypatch.setattr(models, "_gpu", lambda: gpu.append(True))
    return SimpleNamespace(root=root, models=models, gpu=gpu, user_temp=user_temp, source=tmp_path / "ep.mkv")


def _engine_paths(child) -> list[str]:
    argv = child.argv
    return [argv[argv.index(flag) + 1] for flag in ("-m", "-f", "-of")]


def _scratch_left(root: Path) -> list[Path]:
    return sorted((root / native.ASR_SCRATCH).glob("asr-*"))


def test_engine_paths_are_ascii_under_the_native_root_and_removed_after_success(asr):
    rows = asr.models.transcribe(asr.source, [(10.0, 20.0)])
    assert [row["text"] for row in rows] == ["こんにちは"]
    assert len(FakeWhisper.started) == 1
    for value in _engine_paths(FakeWhisper.started[0]):
        assert value.isascii(), value
        assert NON_ASCII not in value
    assert FakeWhisper.started[0].scratch.parent == asr.root / native.ASR_SCRATCH
    assert _scratch_left(asr.root) == [] and list(asr.user_temp.iterdir()) == []


def test_scratch_is_removed_when_the_engine_fails(asr):
    FakeWhisper.code = 1
    with pytest.raises(native.SceneModelsError):
        asr.models.transcribe(asr.source, [(10.0, 20.0)])
    assert FakeWhisper.started and _scratch_left(asr.root) == []


def test_scratch_is_removed_when_a_stop_is_requested_mid_window(asr):
    scene_stop.enable()
    FakeWhisper.on_wait = lambda: scene_stop.request(asr.root)
    with pytest.raises(scene_stop.StopRequested):
        asr.models.transcribe(asr.source, [(10.0, 20.0), (100.0, 110.0)])
    assert len(FakeWhisper.started) == 1 and _scratch_left(asr.root) == []


def test_non_ascii_native_root_is_refused_by_name_before_the_gpu(asr, tmp_path):
    asr.models.root = tmp_path / NON_ASCII / "scene-native"
    with pytest.raises(native.NativeScratchUnavailable) as caught:
        asr.models.transcribe(asr.source, [(10.0, 20.0)])
    assert caught.value.reason_code == "scene_native_asr_path_not_ascii"
    assert asr.gpu == [] and FakeWhisper.started == []


def test_stale_scratch_of_a_killed_worker_is_removed(asr):
    stale = asr.root / native.ASR_SCRATCH / "asr-killed"
    stale.mkdir(parents=True)
    (stale / "window0.wav").write_bytes(b"x")
    old = time.time() - native.ASR_SCRATCH_STALE_SECONDS - 60
    os.utime(stale, (old, old))
    asr.models.transcribe(asr.source, [(10.0, 20.0)])
    assert not stale.exists()


def test_job_records_the_named_cause_instead_of_an_anonymous_asr_failure(tmp_path, library):
    cue = _cue(tmp_path / "ep137", "s1", 400)
    library["s1"].append(cue)

    class NonAsciiRoot(FakeModels):
        def transcribe(self, source, windows):
            raise native.NativeScratchUnavailable(native.NativeScratchUnavailable.reason_code)

        def order(self, prompt):
            return {"name_basis": "context_music", "name": "はじまりの余韻", "direction": None,
                    "use": None, "category": None, "evidence": None}

    state = tmp_path / "state"
    scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=NonAsciiRoot([]))
    receipt = scene_names.cue_record(state, cue["asset_id"], cue["pcm_sha256"])["attempt_identity"][
        "transcription_receipt"]
    assert receipt["state"] == "ASR_UNAVAILABLE" and receipt["reason"] == "scene_native_asr_path_not_ascii"


# --- 2. stop request ----------------------------------------------------------------------------------------

class Response(io.BytesIO):
    def __init__(self, body, status, content_range=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


def _ranged_server(body: bytes, *, after_request=None):
    seen = []

    def fetch(request, **kwargs):
        value = request.get_header("Range")
        seen.append(value)
        if after_request:
            after_request(len(seen))
        lo, hi = map(int, value[6:].split("-"))
        return Response(body[lo:hi + 1], 206, f"bytes {lo}-{hi}/{len(body)}")
    return fetch, seen


@pytest.fixture
def small_download(monkeypatch, tmp_path):
    root = tmp_path / "native"
    root.mkdir()
    monkeypatch.setenv("LSS_SCENE_NATIVE_DIR", str(root))
    monkeypatch.setattr(native, "DOWNLOAD_RANGE_BYTES", 4)
    monkeypatch.setattr(native, "_download_ssl_context", lambda: object())
    monkeypatch.setattr(native.shutil, "disk_usage", lambda _: SimpleNamespace(free=1 << 50))
    return root


def test_stop_between_download_blocks_keeps_the_resumable_prefix(monkeypatch, small_download):
    root = small_download
    body = b"abcdefghijkl"
    target = root / "downloads" / "engine.zip"
    scene_stop.enable()
    fetch, seen = _ranged_server(body, after_request=lambda n: n == 2 and scene_stop.request(root))
    monkeypatch.setattr(native.urllib.request, "urlopen", fetch)
    with pytest.raises(scene_stop.StopRequested):
        native._download("https://example.invalid/e.zip", target, sha256=hashlib.sha256(body).hexdigest(),
                         size=len(body))
    part = target.with_name(target.name + ".part")
    assert seen == ["bytes=0-3", "bytes=4-7"] and part.read_bytes() == body[:8] and not target.exists()
    scene_stop.clear(root)
    fetch, seen = _ranged_server(body)
    monkeypatch.setattr(native.urllib.request, "urlopen", fetch)
    native._download("https://example.invalid/e.zip", target, sha256=hashlib.sha256(body).hexdigest(),
                     size=len(body))
    assert seen[0] == "bytes=8-11" and target.read_bytes() == body


def test_stop_during_bootstrap_records_no_failure_or_retry_delay(monkeypatch, small_download):
    root = small_download
    monkeypatch.setattr(native, "_bootstrap_capacity", lambda _root: None)
    from lss.harness import fetch as harness_fetch

    monkeypatch.setattr(harness_fetch, "model_cache_dir", lambda: root.parent / "separation-cache")

    def install(_root):
        raise scene_stop.StopRequested("scene_worker_stop_requested")
    monkeypatch.setattr(native, "install", install)
    with pytest.raises(scene_stop.StopRequested):
        native.ensure_ready(root)
    assert not (root / "bootstrap-status.json").exists()


def test_a_request_older_than_the_worker_never_stops_it(small_download):
    root = small_download
    path = scene_stop.request(root)
    old = time.time() - 600
    os.utime(path, (old, old))
    scene_stop.enable()
    assert not scene_stop.stop_requested(root)
    scene_stop.request(root)
    assert scene_stop.stop_requested(root)


def test_stop_checks_are_inert_outside_a_worker(small_download):
    scene_stop.request(small_download)
    assert not scene_stop.stop_requested(small_download)
    scene_stop.check(small_download)


def test_stop_between_cues_keeps_named_cues_and_leaves_the_rest_pending(monkeypatch, tmp_path, library):
    root = tmp_path / "native"
    monkeypatch.setenv("LSS_SCENE_NATIVE_DIR", str(root))
    cues = [_cue(tmp_path / "ep137", "s1", 400), _cue(tmp_path / "ep137", "s1", 600, tone=330)]
    library["s1"] += cues

    class StopAfterFirst(FakeModels):
        def order(self, prompt):
            scene_stop.request(root)
            return super().order(prompt)

    models = StopAfterFirst(["ひとつめ", "ふたつめ"])
    state = tmp_path / "state"
    scene_stop.enable()
    with pytest.raises(scene_stop.StopRequested):
        scene_job.name_source(state, tmp_path / "series-alpha-137.mkv", "s1", models=models)
    assert ("release",) in models.calls
    assert scene_names.cue_record(state, cues[0]["asset_id"], cues[0]["pcm_sha256"])["name"] == "ひとつめ"
    assert scene_names.cue_record(state, cues[1]["asset_id"], cues[1]["pcm_sha256"]) is None


def test_worker_returns_stopped_without_waiting_out_a_retry_delay(monkeypatch, tmp_path):
    root = tmp_path / "native"
    monkeypatch.setenv("LSS_SCENE_NATIVE_DIR", str(root))
    sleeps = []

    def run(*args, wait, **kwargs):
        def fake_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:
                scene_stop.request(root)
        assert wait is scene_stop.wait
        wait(3600, sleep=fake_sleep)
        return {"state": "complete"}
    monkeypatch.setattr(scene_job, "_run_background_worker", run)
    result = scene_job.run_background_worker(tmp_path / "state", tmp_path / "a.mkv", "a" * 64)
    assert result == {"state": "stopped", "reason": "scene_worker_stop_requested"}
    assert len(sleeps) == 3


def test_deploy_helper_requests_then_waits_bounded(tmp_path):
    root = tmp_path / "LSS" / "scene-native"
    alive = [True, True, False]
    clock = iter(range(100))
    result = scene_stop.request_and_wait(root, lambda: alive.pop(0), timeout=60, poll=0.5,
                                         sleep=lambda _s: None, clock=lambda: next(clock))
    assert result["state"] == "stopped" and (root / scene_stop.STOP_REQUEST).is_file()
    clock = iter(range(0, 1000, 10))
    result = scene_stop.request_and_wait(root, lambda: True, timeout=60, sleep=lambda _s: None,
                                         clock=lambda: next(clock))
    assert result["state"] == "timeout" and result["waited_seconds"] >= 60
    scene_stop.clear(root)
    assert not (root / scene_stop.STOP_REQUEST).exists()


def test_deploy_script_requests_the_stop_before_moving_files():
    script = (Path(__file__).resolve().parents[1] / "deploy-to-freesoft.ps1").read_text(encoding="utf-8-sig")
    assert '"stop-request"' in script and "--scene-naming" in script
    assert script.count("Stop-TargetProcesses\n") == 2  # deploy and rollback (the definition ends in " {")
    deploy_stop = script.index("    Stop-TargetProcesses\n", script.index("after_stage"))
    assert deploy_stop < script.index("Move-Item -LiteralPath $deployRoot -Destination $previousRoot")
    rollback = script.index("if ($Rollback) {")
    assert script.index("Stop-TargetProcesses", rollback) < script.index("Recover-InterruptedDeploy\n", rollback)


# --- 3. supply mismatch backoff ----------------------------------------------------------------------------

def test_mismatching_pinned_archive_backs_off_exponentially_and_is_recorded_as_supply(monkeypatch, small_download):
    root = small_download
    spec = native.WHISPER_ENGINE["archives"][0]
    served = b"not the pinned archive"
    monkeypatch.setitem(spec, "bytes", len(served))
    requests = []

    def fetch(request, **kwargs):
        requests.append(request.get_header("Range"))
        lo, hi = map(int, request.get_header("Range")[6:].split("-"))
        return Response(served[lo:hi + 1], 206, f"bytes {lo}-{hi}/{len(served)}")
    monkeypatch.setattr(native.urllib.request, "urlopen", fetch)
    now = [1_000_000.0]
    monkeypatch.setattr(native.time, "time", lambda: now[0])
    target = root / "downloads" / spec["url"].rsplit("/", 1)[1]
    fetch_pinned = lambda: native._fetch_pinned(root, spec["url"], target, sha256=spec["sha256"],
                                                size=spec["bytes"])
    delays = []
    for _ in range(4):
        before = len(requests)
        with pytest.raises(native.SupplyMismatch) as caught:
            fetch_pinned()
        assert len(requests) > before
        delays.append(caught.value.retry_after - now[0])
        # Within the backoff no byte is requested again.
        now[0] += delays[-1] - 1
        attempted = len(requests)
        with pytest.raises(native.SupplyMismatch, match="supply_backoff"):
            fetch_pinned()
        assert len(requests) == attempted
        now[0] += 1
    assert delays == [600, 1200, 2400, 4800]
    row = json.loads((root / native.SUPPLY_STATUS).read_text())["assets"]
    assert [value["reason"] for value in row.values()] == ["pinned_asset_supply_mismatch"]


def test_supply_backoff_is_capped(monkeypatch, small_download):
    root = small_download
    spec = native.LLAMA_ENGINE["archives"][0]
    identity = native._asset_identity(spec["url"], spec["sha256"], spec["bytes"])
    native._write_supply(root, {identity: {"failures": 40, "retry_after": 0}})
    monkeypatch.setattr(native, "_download", lambda *a, **k: (_ for _ in ()).throw(native.SupplyMismatch("x")))
    with pytest.raises(native.SupplyMismatch) as caught:
        native._fetch_pinned(root, spec["url"], root / "downloads" / "a.zip", sha256=spec["sha256"],
                             size=spec["bytes"])
    assert caught.value.retry_after - time.time() <= native.SUPPLY_BACKOFF_CAP_SECONDS + 5


def test_supply_backoff_resets_on_a_new_pin_and_on_success(monkeypatch, small_download):
    root = small_download
    spec = native.WHISPER_ENGINE["archives"][0]
    old_identity = native._asset_identity(spec["url"], "0" * 64, spec["bytes"])
    native._write_supply(root, {old_identity: {"failures": 5, "retry_after": time.time() + 10 ** 6}})
    assert native._read_supply(root) == {}  # an identity this build no longer pins carries no history
    body = b"pinned-bytes"
    monkeypatch.setitem(spec, "bytes", len(body))
    monkeypatch.setitem(spec, "sha256", hashlib.sha256(body).hexdigest())
    identity = native._asset_identity(spec["url"], spec["sha256"], spec["bytes"])
    native._write_supply(root, {identity: {"failures": 2, "retry_after": 0}})
    fetch, _ = _ranged_server(body)
    monkeypatch.setattr(native.urllib.request, "urlopen", fetch)
    native._fetch_pinned(root, spec["url"], root / "downloads" / "w.zip", sha256=spec["sha256"], size=len(body))
    assert native._read_supply(root) == {} and not (root / native.SUPPLY_STATUS).exists()


def test_bootstrap_status_and_readiness_show_a_supply_problem(monkeypatch, small_download):
    root = small_download
    monkeypatch.setattr(native, "_bootstrap_capacity", lambda _root: None)
    from lss.harness import fetch as harness_fetch

    monkeypatch.setattr(harness_fetch, "model_cache_dir", lambda: root.parent / "separation-cache")

    def install(_root):
        exc = native.SupplyMismatch("取得したファイルが期待と一致しません: x.zip")
        exc.retry_after = time.time() + 86400
        raise exc
    monkeypatch.setattr(native, "install", install)
    with pytest.raises(native.SupplyMismatch):
        native.ensure_ready(root)
    status = json.loads((root / "bootstrap-status.json").read_text(encoding="utf-8"))
    assert status["reason"] == "pinned_asset_supply_mismatch" and status["retry_after"] > time.time() + 86000
    monkeypatch.setattr(native, "native_root", lambda: root)
    report = scene_runtime.read_only_status()
    assert "pinned_asset_supply_mismatch" in report["native_problem_codes"]


# --- 4. verified-hash cache --------------------------------------------------------------------------------

def test_capacity_check_hashes_complete_weights_once_per_stat(monkeypatch, tmp_path):
    root = tmp_path / "native"
    hashed = []
    real = native._sha256
    monkeypatch.setattr(native, "_sha256", lambda path: hashed.append(Path(path).name) or real(path))
    monkeypatch.setattr(native.shutil, "disk_usage", lambda _: SimpleNamespace(free=1 << 50))
    for spec, body in ((native.ASR_SPEC, b"asr-weights"), (native.LLM_SPEC, b"llm-weights!")):
        path = native._weights_path(root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        monkeypatch.setitem(spec, "bytes", len(body))
        monkeypatch.setitem(spec, "sha256", hashlib.sha256(body).hexdigest())
    native._bootstrap_capacity(root)
    native._bootstrap_capacity(root)
    native._bootstrap_capacity(root)
    assert sorted(hashed) == sorted([native.ASR_SPEC["file"], native.LLM_SPEC["file"]])
    # A changed stat (same size, new mtime) is measured again, and a changed byte is not trusted.
    path = native._weights_path(root, native.LLM_SPEC)
    path.write_bytes(b"llm-weights?")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    hashed.clear()
    assert native._verified_sha256(root, path) != native.LLM_SPEC["sha256"]
    assert hashed == [native.LLM_SPEC["file"]]


def test_part_files_are_never_answered_from_the_cache(monkeypatch, tmp_path):
    root = tmp_path / "native"
    part = root / "models" / "w.gguf.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"prefix")
    hashed = []
    real = native._sha256
    monkeypatch.setattr(native, "_sha256", lambda path: hashed.append(path) or real(path))
    native._verified_sha256(root, part)
    native._verified_sha256(root, part)
    assert len(hashed) == 2
    assert not (root / native.VERIFIED_CACHE).exists()


def test_install_skips_rehashing_verified_weights(monkeypatch, tmp_path):
    root = tmp_path / "native"
    for engine in (native.WHISPER_ENGINE, native.LLAMA_ENGINE):
        directory = native._engine_dir(root, engine)
        directory.mkdir(parents=True)
        (directory / engine["executable"]).write_bytes(b"exe")
    for spec, body in ((native.ASR_SPEC, b"asr"), (native.LLM_SPEC, b"llm")):
        path = native._weights_path(root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        monkeypatch.setitem(spec, "bytes", len(body))
        monkeypatch.setitem(spec, "sha256", hashlib.sha256(body).hexdigest())
        native._verified_sha256(root, path)
    hashed = []
    monkeypatch.setattr(native, "_sha256", lambda path: hashed.append(path) or "0" * 64)
    monkeypatch.setattr(native, "_fetch_pinned", lambda *a, **k: pytest.fail("verified weights re-fetched"))
    monkeypatch.setattr(native, "write_manifest", lambda *a, **k: {"written": True})
    assert native.install(root) == {"written": True}
    assert hashed == []
