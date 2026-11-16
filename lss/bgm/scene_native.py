"""Native (no Python ML stack) backend of the scene-name models: whisper.cpp ASR and llama.cpp LLM.

The Transformers backend (``scene_models.SceneModels``: torch, bitsandbytes, ~30 GB of bf16 weights) cannot be frozen
into LSS.exe. This backend runs the same two models as separate native programs, one at a time, inside the same GPU
lease, with the same interface (``transcribe``, ``order``, ``prefix``, ``release``, ``revision``). Prompts, parsing,
sampling (greedy), token limit (260), no thinking, the repeat filter and the ASR windows are those of
``SceneModels``; only the engines differ. ``scene_runtime`` chooses the backend and keeps results apart.

Engines and weights (all public, official; pinned by SHA-256, recorded in ``revision()``):

- ASR: ``whisper-cli`` of ggml-org/whisper.cpp release ``b5454`` (Windows x64 CUDA 12.4 bundle; no sm_120 code in it,
  so the first run on a Blackwell GPU JIT-compiles from PTX for ~2 min, later runs use the driver cache) with
  ``kotoba-tech/kotoba-whisper-v2.0-ggml`` (``ggml-kotoba-whisper-v2.0.bin``, f16, the same weights as the pinned
  Transformers ``kotoba-tech/kotoba-whisper-v2.0``). One process per merged window; it exits before the LLM starts.
- LLM: ``llama-server`` of ggml-org/llama.cpp ``b11878`` (CUDA 13.4, native sm_120) with ``Qwen/Qwen3-14B-GGUF``
  ``Qwen3-14B-Q4_K_M.gguf`` (official 4-bit K-quant of the same Qwen3-14B). Prompts are sent through the model's own
  chat template with thinking disabled; there is no grammar or JSON constraint (the Transformers path has none either).

Location: ``<native root>/`` = ``LSS_SCENE_NATIVE_DIR``, else ``<app root>/scene-native`` (frozen) or
``<repo>/.scene-native`` (source). It is deliberately NOT under ``models_store.cache_dir()``: that directory is counted
against the 20 GB model cache cap and evicted least-recently-used, so ~10 GB of naming weights there would evict the
separation models. ``deploy-to-freesoft.ps1`` keeps ``scene-native`` across deployments like ``models``.
Layout: ``native-manifest.json`` (sizes, mtimes and SHA-256 of every pinned file, written by ``install``), ``whisper.cpp/<tag>/``,
``llama.cpp/<tag>/``, ``models/<repository>/<revision>/<file>``. ``ready()`` compares sizes and mtimes with the
manifest (never rehashes 10 GB); ``python -m lss.bgm.scene_native install|verify`` downloads / rehashes.

Processes: every child is assigned to a Windows Job Object with kill-on-close, runs without a window, writes to
a log under ``<root>/logs`` and is stopped on ``release()``, on any exception, at interpreter exit and, if this
process dies, by the OS. GPU use is inside ``gpu_lease`` (failure to take it raises; nothing is started).
``CUDA_CACHE_MAXSIZE`` is raised for the children: with the 256 MB default the JIT cache of the CUDA 12.4 whisper
bundle thrashes and every start recompiles (measured: 28 s -> 2.7 s per 60 s of audio).
"""
from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import re
import shutil
import ssl
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from contextlib import contextmanager

import numpy as np

from lss.bgm import scene_models, scene_stop
from lss.bgm.scene_models import LLM, SceneModelsError

BACKEND = "native-whisper.cpp-llama.cpp"
MANIFEST = "native-manifest.json"
MANIFEST_SCHEMA = "scene-native-manifest-r2"
ASR_SPEC = {"repository": "kotoba-tech/kotoba-whisper-v2.0-ggml", "revision": "e3a0cf6a62b95911703cfb97d819292e058f12c3",
            "file": "ggml-kotoba-whisper-v2.0.bin", "bytes": 1519521155, "precision": "f16", "license": "Apache-2.0",
            "sha256": "eff70a8a236e731abba774ba71e1f6d0fce53302137208c32207e694e0bf4546",
            "source_model": scene_models.ASR["repository"] + "@" + scene_models.ASR["revision"]}
LLM_SPEC = {"repository": "Qwen/Qwen3-14B-GGUF", "revision": "530227a7d994db8eca5ab5ced2fb692b614357fd",
            "file": "Qwen3-14B-Q4_K_M.gguf", "bytes": 9001752960, "quantization": "Q4_K_M", "license": "Apache-2.0",
            "sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
            "source_model": LLM["repository"] + "@" + LLM["revision"], "thinking": LLM["thinking"]}
WHISPER_ENGINE = {
    "name": "whisper.cpp", "tag": "b5454", "directory": "whisper.cpp/b5454-cuda12.4", "executable": "whisper-cli.exe",
    "archives": [{"url": "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-win-cuda-12.4.0-x64.zip",
                  "sha256": "afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a", "bytes": 684913404,
                  "strip": "Release/"}],
    "members": r"(whisper-cli\.exe|whisper\.dll|ggml[^/]*\.dll|cudart64_12\.dll|cublas(Lt)?64_12\.dll|nvrtc[^/]*\.dll)",
    "license": "MIT"}
LLAMA_ENGINE = {
    "name": "llama.cpp", "tag": "b11878", "directory": "llama.cpp/b11878-cuda13.4", "executable": "llama-server.exe",
    "archives": [{"url": "https://github.com/ggml-org/llama.cpp/releases/download/b11878/llama-b11878-bin-win-cuda-13.4-x64.zip",
                  "sha256": "640d5c4177bb477c5b08e2c0200794b30ab693262e97f8d4c2642addde46b837", "bytes": 160067623,
                  "strip": ""},
                 {"url": "https://github.com/ggml-org/llama.cpp/releases/download/b11878/cudart-llama-bin-win-cuda-13.4-x64.zip",
                  "sha256": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668", "bytes": 423535356,
                  "strip": ""}],
    "members": r"(llama-server(-impl)?\.(exe|dll)|llama(-common)?\.dll|mtmd\.dll|ggml[^/]*\.dll|libomp\.dll|"
               r"cudart64_13\.dll|cublas(Lt)?64_13\.dll|LICENSE[^/]*)",
    "license": "MIT"}
# Files whose size/mtime/SHA-256 identify an engine (the executable and the CUDA backend it loads).
WHISPER_IDENTITY = ("whisper-cli.exe", "ggml-cuda.dll", "whisper.dll")
LLAMA_IDENTITY = ("llama-server.exe", "llama-server-impl.dll", "ggml-cuda.dll", "llama.dll")
# whisper-cli decoding parameters (recorded in the receipt; changing one changes ``revision()``).
WHISPER_ARGS = ["-l", "ja", "-t", "4", "-bs", "5", "-bo", "5", "-np", "-oj"]
LLAMA_CONTEXT = 6144
ASR_MAX_SECONDS = 600
HEALTH_SECONDS = 600
CHILD_ENV = {"CUDA_CACHE_MAXSIZE": str(4 * 1024 ** 3)}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class _NoLocalRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise urllib.error.HTTPError(request.full_url, code, message, headers, response)


def _local_open(request, *, timeout):
    """Open only numeric loopback HTTP requests without proxies or redirects."""
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1")
            or parsed.username is not None or parsed.password is not None or parsed.fragment):
        raise ValueError("Local model requests require a numeric loopback HTTP URL")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoLocalRedirect())
    return opener.open(request, timeout=timeout)


# --- location and manifest -------------------------------------------------------------------------------------

def native_root() -> Path:
    override = os.environ.get("LSS_SCENE_NATIVE_DIR")
    if override:
        return Path(override)
    from lss.paths import app_root, is_frozen

    return app_root() / ("scene-native" if is_frozen() else ".scene-native")


def _weights_path(root: Path, spec: dict) -> Path:
    return root / "models" / spec["repository"].replace("/", "--") / spec["revision"] / spec["file"]


def _engine_dir(root: Path, engine: dict) -> Path:
    return root / engine["directory"]


def _file_row(path: Path, sha256: str | None = None) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256 or _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for index, block in enumerate(iter(lambda: stream.read(8 * 1024 * 1024), b"")):
            if index % 32 == 31:
                scene_stop.check()
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict | None:
    try:
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) and manifest.get("schema") == MANIFEST_SCHEMA else None


def _manifest_problems(root: Path) -> list[str]:
    """Why the native runtime is not usable here (empty = ready). Stat-only: never hashes the weights."""
    manifest = _read_manifest(root)
    if manifest is None:
        return ["no_manifest"]
    files = manifest.get("files") or {}
    wanted = {}
    for engine, names in ((WHISPER_ENGINE, WHISPER_IDENTITY), (LLAMA_ENGINE, LLAMA_IDENTITY)):
        for name in names:
            wanted[f"{engine['directory']}/{name}"] = None
    expected_archives = {spec["url"]: spec["sha256"] for engine in (WHISPER_ENGINE, LLAMA_ENGINE)
                         for spec in engine["archives"]}
    if manifest.get("archive_sha256") != expected_archives:
        return ["engine_archive_proof_missing_or_mismatched"]
    archive_files = manifest.get("archive_verified_files")
    if (not isinstance(archive_files, list) or not archive_files
            or not all(isinstance(item, str) for item in archive_files)
            or len(set(archive_files)) != len(archive_files)):
        return ["invalid_verified_engine_paths"]
    required_engine_paths = {f"{engine['directory']}/{name}" for engine, names in
                             ((WHISPER_ENGINE, WHISPER_IDENTITY), (LLAMA_ENGINE, LLAMA_IDENTITY)) for name in names}
    if not required_engine_paths.issubset(archive_files):
        return ["verified_engine_identity_incomplete"]
    for relative in archive_files:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not any(
                relative.startswith(engine["directory"] + "/") for engine in (WHISPER_ENGINE, LLAMA_ENGINE)):
            return ["invalid_verified_engine_path"]
        row = files.get(relative)
        digest = row.get("sha256") if isinstance(row, dict) else None
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            return ["invalid_verified_engine_sha256"]
        wanted[relative] = None
    for engine in (WHISPER_ENGINE, LLAMA_ENGINE):
        directory = _engine_dir(root, engine)
        try:
            actual = {(directory / path.name).relative_to(root).as_posix() for path in directory.iterdir()
                      if path.is_file() and path.suffix.lower() in (".exe", ".dll")}
        except OSError:
            return ["verified_engine_directory_unreadable"]
        recorded = {name for name in archive_files if name.startswith(engine["directory"] + "/")}
        if actual != recorded:
            return ["verified_engine_member_set_mismatch"]
    wanted[_weights_path(root, ASR_SPEC).relative_to(root).as_posix()] = ASR_SPEC
    wanted[_weights_path(root, LLM_SPEC).relative_to(root).as_posix()] = LLM_SPEC
    problems = []
    for relative, spec in wanted.items():
        row = files.get(relative)
        path = root / relative
        try:
            stat = path.stat()
        except OSError:
            problems.append(f"missing:{relative}")
            continue
        if not row or row.get("size") != stat.st_size or row.get("mtime_ns") != stat.st_mtime_ns:
            problems.append(f"changed:{relative}")
        elif spec is not None and (stat.st_size != spec["bytes"] or row.get("sha256") != spec["sha256"]):
            problems.append(f"not_pinned:{relative}")
    return problems


def ready(root: Path | None = None) -> bool:
    root = Path(root) if root else native_root()
    return not (root / "bootstrap-in-progress").exists() and not _manifest_problems(root)


def models_cached() -> bool:
    return ready()


def _identity(root: Path) -> dict:
    manifest = _read_manifest(root) or {}
    files = manifest.get("files") or {}

    def engine(spec: dict, names: tuple) -> dict:
        return {"name": spec["name"], "tag": spec["tag"],
                "files": {name: (files.get(f"{spec['directory']}/{name}") or {}).get("sha256") for name in names},
                "version": (manifest.get("versions") or {}).get(spec["name"])}

    return {"whisper": engine(WHISPER_ENGINE, WHISPER_IDENTITY), "llama": engine(LLAMA_ENGINE, LLAMA_IDENTITY)}


def revision(root: Path | None = None) -> dict:
    """What produced a result, comparable with ``SceneModels.revision()`` (a different backend is a different value)."""
    root = Path(root) if root else native_root()
    identity = _identity(root)
    return {"backend": BACKEND,
            "asr": {"repository": ASR_SPEC["repository"], "revision": ASR_SPEC["revision"],
                    "file": ASR_SPEC["file"], "sha256": ASR_SPEC["sha256"], "precision": ASR_SPEC["precision"],
                    "engine": identity["whisper"], "args": WHISPER_ARGS},
            "llm": {"repository": LLM_SPEC["repository"], "revision": LLM_SPEC["revision"], "file": LLM_SPEC["file"],
                    "sha256": LLM_SPEC["sha256"], "quantization": LLM_SPEC["quantization"],
                    "thinking": LLM["thinking"], "engine": identity["llama"], "context": LLAMA_CONTEXT}}


# --- install -------------------------------------------------------------------------------------------------------

def _download_ssl_context() -> ssl.SSLContext:
    """Use verified system trust plus certifi roots when certifi is available."""
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return context
    context.load_verify_locations(cafile=certifi.where())
    return context


DOWNLOAD_RANGE_BYTES = 8 * 1024 ** 2
VERIFIED_CACHE = "verified-hashes.json"
SUPPLY_STATUS = "supply-status.json"
SUPPLY_BACKOFF_SECONDS = 300
SUPPLY_BACKOFF_CAP_SECONDS = 7 * 24 * 60 * 60


class SupplyMismatch(SceneModelsError):
    """A pinned upstream asset arrived complete but is not the pinned bytes: a supply problem, not a network one.

    Retrying soon cannot help (the pin is fixed in this build), so ``_fetch_pinned`` backs the asset off
    exponentially; only a new pinned identity (a new LSS build) or a later matching download resets it.
    """
    reason_code = "pinned_asset_supply_mismatch"
    retry_after: float | None = None


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_json_dict(path: Path) -> dict:
    try:
        if path.stat().st_size > 256 * 1024:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _verified_sha256(root: Path, path: Path) -> str:
    """SHA-256 of a complete file under ``root``, re-hashed only when its (size, mtime_ns) changed.

    The cache (``verified-hashes.json``) maps the root-relative path to the stat and the digest measured for it;
    a ``.part`` file (still being written) or a file outside ``root`` is always hashed and never cached.
    """
    root, path = Path(root), Path(path)
    stat = path.stat()
    try:
        key = path.relative_to(root).as_posix()
    except ValueError:
        key = None
    if key is None or path.name.endswith(".part"):
        return _sha256(path)
    cache_path = root / VERIFIED_CACHE
    row = _read_json_dict(cache_path).get(key)
    if (isinstance(row, dict) and row.get("size") == stat.st_size and row.get("mtime_ns") == stat.st_mtime_ns
            and _is_sha256(row.get("sha256"))):
        return row["sha256"]
    digest = _sha256(path)
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns):
        cache = _read_json_dict(cache_path)
        cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
        try:
            _write_json_atomic(cache_path, cache)
        except OSError:
            pass  # the cache only saves time; the measured digest is still returned
    return digest


def _weight_url(spec: dict) -> str:
    return f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{spec['file']}"


def _asset_identity(url: str, sha256: str, size: int) -> str:
    return f"{url}#sha256={sha256}#bytes={size}"


def _pinned_asset_identities() -> set[str]:
    archives = {_asset_identity(spec["url"], spec["sha256"], spec["bytes"])
                for engine in (WHISPER_ENGINE, LLAMA_ENGINE) for spec in engine["archives"]}
    return archives | {_asset_identity(_weight_url(spec), spec["sha256"], spec["bytes"]) for spec in (ASR_SPEC, LLM_SPEC)}


def _read_supply(root: Path) -> dict:
    """Backoff rows of currently pinned assets only: a new pinned identity starts without history."""
    assets = _read_json_dict(Path(root) / SUPPLY_STATUS).get("assets")
    pinned = _pinned_asset_identities()
    return {key: row for key, row in (assets or {}).items()
            if key in pinned and isinstance(row, dict)} if isinstance(assets, dict) else {}


def _write_supply(root: Path, assets: dict) -> None:
    path = Path(root) / SUPPLY_STATUS
    if not assets:
        path.unlink(missing_ok=True)
        return
    _write_json_atomic(path, {"schema": "scene-native-supply-r1", "assets": assets})


def _fetch_pinned(root: Path, url: str, target: Path, *, sha256: str, size: int) -> None:
    """``_download`` behind the per-asset supply backoff; a verified complete file needs no request."""
    identity = _asset_identity(url, sha256, size)
    if target.is_file() and target.stat().st_size == size and _verified_sha256(root, target) == sha256:
        assets = _read_supply(root)
        if assets.pop(identity, None) is not None:
            _write_supply(root, assets)
        return
    assets = _read_supply(root)
    row = assets.get(identity) or {}
    retry_after = row.get("retry_after")
    if isinstance(retry_after, (int, float)) and time.time() < retry_after:
        exc = SupplyMismatch(f"scene_native_pinned_asset_supply_backoff:{target.name}")
        exc.retry_after = float(retry_after)
        raise exc
    try:
        _download(url, target, sha256=sha256, size=size)
    except SupplyMismatch as exc:
        previous = row.get("failures")
        failures = (previous if isinstance(previous, int) and previous > 0 else 0) + 1
        delay = min(SUPPLY_BACKOFF_CAP_SECONDS, SUPPLY_BACKOFF_SECONDS * 2 ** failures)
        exc.retry_after = time.time() + delay
        assets = _read_supply(root)
        assets[identity] = {"reason": SupplyMismatch.reason_code, "file": target.name, "failures": failures,
                            "last_failed_at": time.time(), "retry_after": exc.retry_after}
        _write_supply(root, assets)
        raise
    assets = _read_supply(root)
    if assets.pop(identity, None) is not None:
        _write_supply(root, assets)


def _download(url: str, target: Path, *, sha256: str, size: int) -> None:
    """Resume bounded HTTPS ranges; publish only after full pinned SHA-256 proof.

    A server ignoring Range retains the full-response fallback. Every206 must
    identify exactly the requested bytes and total before any partial is changed.
    Interrupted valid prefixes remain resumable; no parallel requests are used.
    """
    if target.is_file() and target.stat().st_size == size and _sha256(target) == sha256:
        return
    # Explicit cached-only runs may verify existing bytes but never create a download or prefix.
    # A malformed nonzero value also refuses acquisition rather than silently enabling it.
    if os.environ.get("LSS_SCENE_NO_ACQUIRE", "0") != "0":
        raise SceneModelsError("scene_native_acquisition_disabled")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        raise SceneModelsError("scene_native_download_oversized_partial")
    # A kept prefix already occupies its bytes: admission needs only the rest (else an interrupted
    # first-use download on a nearly full drive could never resume).
    if shutil.disk_usage(target.parent).free < size - have + 4 * 1024 ** 3:
        raise SceneModelsError("scene_native_bootstrap_insufficient_disk_space")
    context = _download_ssl_context()
    while have < size:
        end = min(size, have + DOWNLOAD_RANGE_BYTES) - 1
        request = urllib.request.Request(url, headers={
            "User-Agent": "LSS-scene-native/1", "Range": f"bytes={have}-{end}"})
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            if response.status == 206:
                expected = f"bytes {have}-{end}/{size}"
                if response.headers.get("Content-Range") != expected:
                    raise SceneModelsError("scene_native_download_invalid_content_range")
                count, mode, required = have, "ab", end + 1
            elif response.status == 200:
                count, mode, required = 0, "wb", size
            else:
                raise SceneModelsError("scene_native_download_unexpected_status")
            with part.open(mode) as stream:
                while count < required:
                    block = response.read(min(1 << 20, required - count))
                    if not block:
                        raise SceneModelsError("scene_native_download_truncated_response")
                    if len(block) > required - count:
                        raise SceneModelsError("scene_native_download_oversized_response")
                    stream.write(block)
                    count += len(block)
                    # Stop between blocks: the appended prefix stays a valid resumable ``.part``.
                    if scene_stop.stop_requested():
                        stream.flush()
                        raise scene_stop.StopRequested("scene_worker_stop_requested")
                if response.read(1):
                    raise SceneModelsError("scene_native_download_oversized_response")
            have = count
    if part.stat().st_size != size or _sha256(part) != sha256:
        part.unlink(missing_ok=True)
        raise SupplyMismatch(f"取得したファイルが期待と一致しません: {target.name}")
    part.replace(target)


def _extract(archive: Path, engine: dict, spec: dict, destination: Path) -> None:
    pattern = re.compile(engine["members"])
    strip = spec["strip"]
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        selected = [member for member in bundle.infolist() if not member.is_dir()
                    and (not strip or member.filename.startswith(strip))
                    and "/" not in member.filename[len(strip):]
                    and pattern.fullmatch(member.filename[len(strip):])]
        required = sum(member.file_size for member in selected) + 4 * 1024 ** 3
        if shutil.disk_usage(destination).free < required:
            raise SceneModelsError("scene_native_bootstrap_insufficient_disk_space")
        for member in selected:
            name = member.filename
            if member.is_dir() or (strip and not name.startswith(strip)):
                continue
            relative = name[len(strip):]
            if "/" in relative or not pattern.fullmatch(relative):
                continue
            with bundle.open(member) as source, (destination / relative).open("wb") as target:
                for block in iter(lambda: source.read(1 << 20), b""):
                    target.write(block)


def install(root: Path | None = None, *, local: dict | None = None, log=print) -> dict:
    """Fetch (or take from ``local``: {url-or-weight-file-name: path}) engines and weights, then write the manifest.

    Pinned SHA-256 of every archive and weight file is verified; nothing is trusted from the manifest of another run.
    """
    root = Path(root) if root else native_root()
    local = local or {}
    work = root / "downloads"
    for engine in (WHISPER_ENGINE, LLAMA_ENGINE):
        destination = _engine_dir(root, engine)
        if (destination / engine["executable"]).is_file():
            continue
        staging = destination.with_name(destination.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        for spec in engine["archives"]:
            name = spec["url"].rsplit("/", 1)[1]
            archive = Path(local[name]) if name in local else work / name
            if name not in local:
                log(f"download {name}")
                _fetch_pinned(root, spec["url"], archive, sha256=spec["sha256"], size=spec["bytes"])
            elif _sha256(archive) != spec["sha256"]:
                raise SceneModelsError(f"{archive} は固定した実体と一致しません")
            _extract(archive, engine, spec, staging)
        staging.replace(destination)
    for spec in (ASR_SPEC, LLM_SPEC):
        target = _weights_path(root, spec)
        if (target.is_file() and target.stat().st_size == spec["bytes"]
                and _verified_sha256(root, target) == spec["sha256"]):
            continue
        given = local.get(spec["file"])
        if given:
            if _sha256(Path(given)) != spec["sha256"]:
                raise SceneModelsError(f"{given} は固定した実体と一致しません")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(given, target)
        else:
            log(f"download {spec['file']}")
            _fetch_pinned(root, _weight_url(spec), target, sha256=spec["sha256"], size=spec["bytes"])
    manifest = write_manifest(root, log=log, local=local)
    (root / "bootstrap-in-progress").unlink(missing_ok=True)
    return manifest


def _verify_engine_archives(root: Path, *, local: dict | None = None) -> list[str]:
    """Verify installed engine members against SHA-pinned archives before publishing a manifest."""
    local = local or {}
    verified = []
    for engine, names in ((WHISPER_ENGINE, WHISPER_IDENTITY), (LLAMA_ENGINE, LLAMA_IDENTITY)):
        expected = {}
        for spec in engine["archives"]:
            name = spec["url"].rsplit("/", 1)[1]
            archive = Path(local[name]) if name in local else root / "downloads" / name
            if name not in local:
                _fetch_pinned(root, spec["url"], archive, sha256=spec["sha256"], size=spec["bytes"])
            elif not archive.is_file() or _sha256(archive) != spec["sha256"]:
                raise SceneModelsError("scene_native_engine_archive_identity_mismatch")
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if member.is_dir() or not member.filename.startswith(spec["strip"]):
                        continue
                    relative = member.filename[len(spec["strip"]):]
                    if "/" in relative or not re.fullmatch(engine["members"], relative):
                        continue
                    digest = hashlib.sha256()
                    with bundle.open(member) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                    expected[relative] = digest.hexdigest()
        if not set(names).issubset(expected):
            raise SceneModelsError("scene_native_engine_archive_members_missing")
        destination = _engine_dir(root, engine)
        runtime_suffixes = (".exe", ".dll")
        actual = {path.name for path in destination.iterdir()
                  if path.is_file() and (path.name in expected or path.suffix.lower() in runtime_suffixes)}
        if actual != set(expected):
            raise SceneModelsError("scene_native_engine_member_set_mismatch")
        for name, digest in expected.items():
            if _sha256(destination / name) != digest:
                raise SceneModelsError("scene_native_engine_file_identity_mismatch:" + name)
            if Path(name).suffix.lower() in runtime_suffixes:
                verified.append((destination / name).relative_to(root).as_posix())
    return sorted(verified)


@contextmanager
def _bootstrap_lock(root: Path):
    """Cross-process install lock; waiting workers never hold a GPU lease."""
    path = root / "bootstrap.lock"
    root.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        deadline = time.monotonic() + 4 * 60 * 60
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise SceneModelsError("scene_native_bootstrap_lock_timeout")
                scene_stop.wait(1, root=root)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _bootstrap_capacity(root: Path) -> None:
    """Conservative extra space for archives, extracted binaries, weights and a 4 GiB reserve."""
    def complete_weight(spec):
        path = _weights_path(root, spec)
        # Stat-keyed cache: an unchanged 10 GB weight is not re-hashed on every admission check.
        return (path.is_file() and path.stat().st_size == spec["bytes"]
                and _verified_sha256(root, path) == spec["sha256"])

    def held(path: Path) -> int:
        # Bytes an earlier interrupted attempt already stored (complete file or resumable prefix).
        sizes = [candidate.stat().st_size for candidate in (path, path.with_name(path.name + ".part"))
                 if candidate.is_file()]
        return max(sizes, default=0)

    missing_weights = sum(max(0, spec["bytes"] - held(_weights_path(root, spec)))
                          for spec in (ASR_SPEC, LLM_SPEC) if not complete_weight(spec))
    specs = [spec for engine in (WHISPER_ENGINE, LLAMA_ENGINE) for spec in engine["archives"]]
    archives = sum(spec["bytes"] for spec in specs)
    stored_archives = sum(min(spec["bytes"], held(root / "downloads" / spec["url"].rsplit("/", 1)[1]))
                          for spec in specs)
    required = missing_weights + archives * 4 - stored_archives + 4 * 1024 ** 3
    if shutil.disk_usage(root).free < required:
        raise SceneModelsError("scene_native_bootstrap_insufficient_disk_space")


def ensure_ready(root: Path | None = None) -> None:
    """Install once under a shared lock; leave failures retryable without a download loop."""
    root = Path(root) if root else native_root()
    from lss.harness import fetch

    cache = Path(fetch.model_cache_dir()).resolve()
    resolved = root.resolve()
    if resolved == cache or resolved.is_relative_to(cache):
        raise SceneModelsError("scene_native_bootstrap_inside_separation_cache")
    if ready(root):
        return
    with _bootstrap_lock(root):
        if ready(root):
            return
        status = root / "bootstrap-status.json"
        try:
            previous = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        retry_after = previous.get("retry_after")
        if isinstance(retry_after, (int, float)) and time.time() < retry_after:
            raise SceneModelsError("scene_native_bootstrap_retry_pending")
        try:
            _bootstrap_capacity(root)
            (root / "bootstrap-in-progress").write_text("verification pending\n", encoding="utf-8")
            install(root)
            if not ready(root):
                raise SceneModelsError("scene_native_bootstrap_manifest_invalid")
        except scene_stop.StopRequested:
            raise  # a deploy/rollback stop: no failure, no retry delay; ``.part`` prefixes stay resumable
        except Exception as exc:
            retry_after = time.time() + 300
            supply_retry = getattr(exc, "retry_after", None)
            if isinstance(supply_retry, (int, float)):
                retry_after = max(retry_after, float(supply_retry))
            temporary = status.with_suffix(".tmp")
            temporary.write_text(json.dumps({"state": "failed", "retry_after": retry_after,
                "reason": getattr(exc, "reason_code", None),
                "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(status)
            raise
        status.unlink(missing_ok=True)


def write_manifest(root: Path, *, local: dict | None = None, log=print) -> dict:
    """Verify pinned archives and installed members before writing a ready manifest."""
    verified_engine_files = _verify_engine_archives(root, local=local)
    files, versions = {}, {}
    for engine, names in ((WHISPER_ENGINE, WHISPER_IDENTITY), (LLAMA_ENGINE, LLAMA_IDENTITY)):
        for name in names:
            path = _engine_dir(root, engine) / name
            files[f"{engine['directory']}/{name}"] = _file_row(path)
        executable = _engine_dir(root, engine) / engine["executable"]
        try:
            out = subprocess.run([str(executable), "--version"], capture_output=True, timeout=120, cwd=executable.parent,
                                 creationflags=NO_WINDOW, env=_child_env(executable.parent))
            text = (out.stdout + out.stderr).decode("utf-8", "replace")
            versions[engine["name"]] = next((line.strip() for line in text.splitlines() if "version" in line.lower()), None)
        except (OSError, subprocess.SubprocessError):
            versions[engine["name"]] = None
    for relative in verified_engine_files or []:
        if relative not in files:
            files[relative] = _file_row(root / relative)
    for spec in (ASR_SPEC, LLM_SPEC):
        path = _weights_path(root, spec)
        log(f"hash {spec['file']}")
        row = _file_row(path)
        if row["size"] != spec["bytes"] or row["sha256"] != spec["sha256"]:
            raise SceneModelsError(f"{spec['file']} は固定した実体と一致しません")
        files[path.relative_to(root).as_posix()] = row
    manifest = {"schema": MANIFEST_SCHEMA, "backend": BACKEND, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "files": files, "versions": versions,
                "archive_verified_files": verified_engine_files,
                "archive_sha256": {spec["url"]: spec["sha256"] for engine in (WHISPER_ENGINE, LLAMA_ENGINE)
                                   for spec in engine["archives"]},
                "pins": {"asr": {k: ASR_SPEC[k] for k in ("repository", "revision", "file", "sha256")},
                         "llm": {k: LLM_SPEC[k] for k in ("repository", "revision", "file", "sha256")},
                         "whisper_tag": WHISPER_ENGINE["tag"], "llama_tag": LLAMA_ENGINE["tag"]}}
    temporary = root / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(root / MANIFEST)
    return manifest


# --- processes -------------------------------------------------------------------------------------------------------

def _child_env(directory: Path) -> dict:
    return {**os.environ, **CHILD_ENV, "PATH": str(directory) + os.pathsep + os.environ.get("PATH", "")}


class _JobObject:
    """Windows Job Object that kills every assigned process when closed (or when this process dies)."""

    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount",
                         "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Counters), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateJobObjectW.restype = wintypes.HANDLE
        api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        self._api = api
        handle = api.CreateJobObjectW(None, None)
        if not handle:
            raise SceneModelsError("子プロセスの管理(Job Object)を作れません")
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            api.CloseHandle(handle)
            raise SceneModelsError("子プロセスの管理(Job Object)を設定できません")
        self.handle = handle

    def add(self, process: subprocess.Popen) -> None:
        if self.handle is None:
            return
        if not self._api.AssignProcessToJobObject(self.handle, int(process._handle)):  # noqa: SLF001 - Popen's own handle
            raise SceneModelsError("子プロセスをJob Objectへ入れられません")

    def close(self) -> None:
        handle, self.handle = self.handle, None
        if handle is not None:
            self._api.CloseHandle(handle)


class _Child:
    """One native process: hidden, logged, in a kill-on-close job, always stopped by ``stop()``."""

    def __init__(self, argv: list[str], *, cwd: Path, log: Path, env: dict):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.argv, self.log_path = [str(a) for a in argv], log
        self._log = log.open("ab")
        self.job = _JobObject()
        try:
            self.process = subprocess.Popen(self.argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                            stdout=self._log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            self.job.add(self.process)
        except BaseException:
            self.stop()
            raise
        _LIVE.add(self)

    def tail(self, characters: int = 600) -> str:
        try:
            data = self.log_path.read_bytes()[-4000:]
        except OSError:
            return ""
        return data.decode("utf-8", "replace")[-characters:]

    def stop(self) -> None:
        process = getattr(self, "process", None)
        try:
            self.job.close()  # kills the whole tree
            if process is not None and process.poll() is None:
                process.kill()
            if process is not None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            try:
                self._log.close()
            except OSError:
                pass
            _LIVE.discard(self)


_LIVE: set[_Child] = set()


@atexit.register
def _stop_all() -> None:
    for child in list(_LIVE):
        try:
            child.stop()
        except Exception:  # noqa: BLE001 - exit path
            pass


def _gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, timeout=20, creationflags=NO_WINDOW)
        return int(out.stdout.decode().strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _require_vram(needed_mib: int) -> None:
    """Fail instead of letting the driver spill a model into shared system memory (2026-10-16 incident)."""
    free = _gpu_free_mib()
    if free is not None and free < needed_mib:
        raise SceneModelsError("GPUのメモリが足りません")


ASR_SCRATCH = "scratch"
ASR_SCRATCH_STALE_SECONDS = 6 * 60 * 60


class NativeScratchUnavailable(SceneModelsError):
    """whisper.cpp opens ``-m``/``-f``/``-of`` through narrow (ANSI code page) paths: a non-ASCII path fails."""
    reason_code = "scene_native_asr_path_not_ascii"


def _asr_scratch_parent(root: Path) -> Path:
    """LSS-owned ASCII scratch for the WAV/JSON files handed to whisper-cli (never %TEMP%, which holds the user name).

    The weights are under the same root, so a non-ASCII root cannot work at all; it is refused by name.
    """
    parent = Path(os.path.abspath(Path(root) / ASR_SCRATCH))
    if not str(parent).isascii():
        raise NativeScratchUnavailable(NativeScratchUnavailable.reason_code)
    parent.mkdir(parents=True, exist_ok=True)
    # A worker killed mid-window (deploy force-stop, power loss) cannot run its cleanup; ASR runs one at a time
    # under the GPU lease, so only directories far older than any running window are removed.
    for stale in parent.glob("asr-*"):
        try:
            if stale.is_dir() and time.time() - stale.stat().st_mtime > ASR_SCRATCH_STALE_SECONDS:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            pass
    return parent


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --- the models ------------------------------------------------------------------------------------------------------

class NativeSceneModels(scene_models.SceneModels):
    """``SceneModels`` over native engines. ``order``/``prefix`` (prompts and parsing) are inherited unchanged."""

    def __init__(self, device: str = "cuda:0", root: Path | None = None):
        super().__init__(device)
        self.root = Path(root) if root else native_root()
        self._server: _Child | None = None
        self._port: int | None = None

    # -- interface ---------------------------------------------------------------------------------------------
    def revision(self) -> dict:
        return revision(self.root)

    def _gpu(self):
        if self._lease is None:
            from lss.core import gpu_lease

            # Same cross-process lease as the main worker; failure raises before any process is started.
            self._lease = gpu_lease.acquire()

    def _stop_server(self) -> None:
        server, self._server, self._port = self._server, None, None
        if server is not None:
            server.stop()

    def release(self):
        try:
            self._stop_server()
        finally:
            if self._lease:
                try:
                    self._lease.close()
                except Exception:  # noqa: BLE001
                    pass
            self._lease = None

    # -- ASR ---------------------------------------------------------------------------------------------------
    def _whisper_argv(self, wav: Path, output: Path) -> list[str]:
        return [str(_engine_dir(self.root, WHISPER_ENGINE) / WHISPER_ENGINE["executable"]),
                "-m", str(_weights_path(self.root, ASR_SPEC)), "-f", str(wav), "-of", str(output), *WHISPER_ARGS]

    def _run_whisper(self, wav: Path, scratch: Path, window_id: int) -> list[dict]:
        output = scratch / f"window{window_id}"
        engine_dir = _engine_dir(self.root, WHISPER_ENGINE)
        child = _Child(self._whisper_argv(wav, output), cwd=engine_dir, log=self.root / "logs" / "whisper-cli.log",
                       env=_child_env(engine_dir))
        try:
            deadline = time.monotonic() + ASR_MAX_SECONDS
            while True:
                scene_stop.check(self.root)
                try:
                    code = child.process.wait(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired as exc:
                    if time.monotonic() >= deadline:
                        raise SceneModelsError("文字起こしが時間内に終わりません") from exc
            if code != 0:
                raise SceneModelsError(f"文字起こしが失敗しました(exit {code}): {child.tail()}")
        finally:
            child.stop()
        try:
            data = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SceneModelsError("文字起こしの出力を読めません") from exc
        return [{"a": row["offsets"]["from"] / 1000.0, "b": row["offsets"]["to"] / 1000.0,
                 "text": (row.get("text") or "").strip()} for row in data.get("transcription") or []]

    def transcribe(self, source: Path, windows: list[tuple[float, float]]) -> list[dict]:
        """Same contract and receipt shape as ``SceneModels.transcribe``; whisper.cpp does the decoding."""
        import hashlib as _hashlib
        import platform

        import soundfile as sf

        problems = _manifest_problems(self.root)
        if problems:
            raise SceneModelsError("場面名のネイティブ部品が揃っていません: " + ",".join(problems[:3]))
        scratch_parent = _asr_scratch_parent(self.root)
        self._gpu()
        _require_vram(3 * 1024)
        from lss.paths import is_frozen, resource_root

        ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
        if not ffmpeg.is_file() and not is_frozen():
            ffmpeg = Path("ffmpeg")
        executable = str(ffmpeg)
        if ffmpeg.name == "ffmpeg" and not ffmpeg.is_file():
            executable = shutil.which(executable) or executable
        executable = str(Path(executable).resolve())
        ffmpeg_stat = scene_models._file_stat_identity(Path(executable))
        ffmpeg_identity = dict(scene_models._ffmpeg_identity(executable, ffmpeg_stat))
        input_windows, lines = [], []
        self.last_transcription_receipt = None
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="asr-", dir=scratch_parent) as scratch_name:
            scratch = Path(scratch_name)
            for window_id, (start, end) in enumerate(scene_models._merge(windows)):
                scene_stop.check(self.root)
                argv = [executable, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(source),
                        "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
                raw = subprocess.run(argv, capture_output=True, check=True, timeout=600, creationflags=NO_WINDOW).stdout
                audio = np.frombuffer(raw, dtype=np.float32).copy()
                input_windows.append({
                    "window_id": window_id, "source_start_seconds": start, "source_end_seconds": end, "ffmpeg_argv": argv,
                    "input": {"sha256": _hashlib.sha256(raw).hexdigest(), "byte_length": len(raw),
                              "sample_count": int(audio.size), "dtype": "float32-le", "sampling_rate": 16000,
                              "accepted_by_existing_minimum": audio.size >= 1600}})
                if audio.size < 1600:
                    continue
                wav = scratch / f"window{window_id}.wav"
                sf.write(str(wav), audio, 16000, subtype="FLOAT")
                for row in self._run_whisper(wav, scratch, window_id):
                    if not row["text"]:
                        continue
                    lines.append({"start": round(start + row["a"], 2), "end": round(start + row["b"], 2),
                                  "text": row["text"]})
        result = scene_models._drop_repeats(lines)
        try:
            ffmpeg_after = scene_models._file_stat_identity(Path(executable))
        except OSError:
            ffmpeg_after = None
        ffmpeg_identity["binary_sha256_at_binding"] = ffmpeg_identity.get("binary_sha256")
        ffmpeg_identity["stat_after_decode"] = list(ffmpeg_after) if ffmpeg_after else None
        if ffmpeg_identity.get("identity_status") == "unchanged_at_binding_checks":
            ffmpeg_identity["identity_status"] = ("unchanged_at_observation_points" if ffmpeg_after == ffmpeg_stat
                                                  else "unverified_changed_during_decode")
        if ffmpeg_identity["identity_status"] != "unchanged_at_observation_points":
            ffmpeg_identity["binary_sha256"] = ffmpeg_identity["version"] = ffmpeg_identity["version_output_sha256"] = None
        self.last_transcription_receipt = {
            "schema": "scene-asr-input-lineage-r1", "source_path": str(Path(source).resolve()),
            "ffmpeg": ffmpeg_identity,
            "asr": {"revision": revision(self.root)["asr"], "backend": BACKEND, "device": "cuda:0",
                    "timestamps": True, "language": "ja",
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                    "seconds": round(time.time() - started, 1)},
            "windows": input_windows,
            "transcript_sha256": _hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                                            separators=(",", ":")).encode("utf-8")).hexdigest()}
        return result

    # -- LLM ---------------------------------------------------------------------------------------------------
    def _llama_argv(self, port: int) -> list[str]:
        return [str(_engine_dir(self.root, LLAMA_ENGINE) / LLAMA_ENGINE["executable"]),
                "-m", str(_weights_path(self.root, LLM_SPEC)), "--host", "127.0.0.1", "--port", str(port),
                "-ngl", "99", "-c", str(LLAMA_CONTEXT), "-np", "1", "--no-webui", "--reasoning", "off"]

    def _load_llm(self) -> None:
        if self._server is not None:
            if self._server.process.poll() is None:
                self._gpu()
                return
            self._stop_server()
        problems = _manifest_problems(self.root)
        if problems:
            raise SceneModelsError("場面名のネイティブ部品が揃っていません: " + ",".join(problems[:3]))
        self._gpu()
        _require_vram(LLM_SPEC["bytes"] // (1024 * 1024) + 1536)
        port = _free_port()
        engine_dir = _engine_dir(self.root, LLAMA_ENGINE)
        server = _Child(self._llama_argv(port), cwd=engine_dir, log=self.root / "logs" / "llama-server.log",
                        env=_child_env(engine_dir))
        self._server, self._port = server, port
        try:
            deadline = time.time() + HEALTH_SECONDS
            while True:
                if server.process.poll() is not None:
                    raise SceneModelsError(f"言語モデルが起動しません(exit {server.process.returncode}): {server.tail()}")
                try:
                    with _local_open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                        if response.status == 200:
                            return
                except (urllib.error.URLError, OSError):
                    pass
                if time.time() > deadline:
                    raise SceneModelsError("言語モデルの起動が時間内に終わりません")
                scene_stop.check(self.root)
                time.sleep(0.5)
        except BaseException:
            self._stop_server()
            raise

    def _ask(self, prompt: str, max_new: int) -> str:
        self._load_llm()
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}], "max_tokens": max_new, "temperature": 0, "top_k": 1,
            "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0, "seed": 0, "stream": False, "cache_prompt": False,
            "t_max_predict_ms": LLM["max_seconds"] * 1000,
            "chat_template_kwargs": {"enable_thinking": bool(LLM["thinking"])}}).encode("utf-8")
        request = urllib.request.Request(f"http://127.0.0.1:{self._port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with _local_open(request, timeout=LLM["max_seconds"] + 60) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            tail = self._server.tail() if self._server is not None else ""
            self._stop_server()
            if "out of memory" in tail.lower():
                raise SceneModelsError("GPUのメモリが足りません") from exc
            raise SceneModelsError("言語モデルの応答を読めません") from exc


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m lss.bgm.scene_native")
    parser.add_argument("command", choices=("install", "verify", "status"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--local", action="append", default=[], metavar="NAME=PATH",
                        help="use an already downloaded archive / weight file (verified against the pinned SHA-256)")
    args = parser.parse_args(argv)
    root = args.root or native_root()
    if args.command == "status":
        print(json.dumps({"root": str(root), "problems": _manifest_problems(root)}, ensure_ascii=False))
        return 0
    if args.command == "install":
        install(root, local=dict(item.split("=", 1) for item in args.local))
    else:
        write_manifest(root, local=dict(item.split("=", 1) for item in args.local))
    print(json.dumps({"root": str(root), "problems": _manifest_problems(root), "revision": revision(root)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
