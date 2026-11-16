"""Shared local LLM runner for LSS development: llama.cpp `llama-server` behind the GPU lease, one model at a time.

    python scripts/local_llm.py install                                  # pinned llama.cpp build -> policy llama_runtime_root / b11878
    python scripts/local_llm.py models [--json]                          # registry (scripts/local_llm_models.json) and availability
    python scripts/local_llm.py serve gpt-oss-20b [--ctx N] [--wait]     # prints the base URL, holds until Ctrl+C / parent death
    python scripts/local_llm.py run gpt-oss-20b -- python job.py         # job.py sees OPENAI_BASE_URL / LOCAL_LLM_BASE_URL
    python scripts/local_llm.py chat ornith-1.0-9b --prompt-file p.txt [--schema schema.json] [--max-tokens N]

    from local_llm import serving, complete, embed, rerank                # scripts/ and tests/ (never lss/ product code)
    with serving("gpt-oss-20b", purpose="triage") as base_url:
        print(complete(base_url, [{"role": "user", "content": "..."}]).text)

What one serve does, in order:
- SSD placement: checkout/Python stay on C; work/model/runtime paths use the configured fast SSDs;
  an HDD path is refused before inference. Active weight paths come from the
  private machine overlay; archive originals are never a runtime fallback.
- RAM admission: it starts only while available physical RAM is at least the model's working set (`ram_gb`) + 4 GB
  and commit headroom is at least `LSS_MIN_COMMIT_HEADROOM_GB` (default 4, the `gpu_run` rule). Otherwise it
  refuses (exit 3) or, with `--wait`, polls. Qwen3.6-35B-A3B once drove free RAM to 63 MB on this 32 GB PC.
- Staging: already-hot weights pass through unchanged. `local_ai_storage.py pin` retains an existing
  verified SSD stage as permanent hot weights without another large copy; missing SSD bytes refuse.
- GPU lease: `lss.core.gpu_lease.acquire()` in this process, the same named mutex `gpu_run` and the LSS worker use.
  The server child never takes it. Never run this script under `gpu_run` (the parent would hold the lease this
  process waits for; that is detected and refused). The holder file `gpu_run` shows is written too.
- The server: hidden (no console), 127.0.0.1 on a free port, `-np 1 --jinja --no-webui --metrics`, f16 K/V cache (q8_0 K
  corrupts identifiers on Qwen3.6), below-normal CPU priority, inside a Windows Job Object with KILL_ON_JOB_CLOSE:
  it dies with this process however this process ends. No duty cap: the load is request-bound, and a 0.25 duty
  would suspend the server 75 % of the time.
- On exit: the job is killed, the lease released, VRAM is read back, and one row goes to
  %LOCALAPPDATA%\\LSS-dev\\local-llm\\runs.jsonl (utc, model, purpose, staged path, load seconds, RAM/VRAM peaks,
  tokens, wall, exit status).
The lease is released by the thread that took it: use `serving` from one thread.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_run  # noqa: E402  (memory admission helpers, holder file)
from lss.core import gpu_lease  # noqa: E402
from stage import stage_path  # noqa: E402
from machine_config import load_config  # noqa: E402

BUILD = "b11878"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/"
ASSETS = {
    f"llama-{BUILD}-bin-win-cuda-13.4-x64.zip": "640d5c4177bb477c5b08e2c0200794b30ab693262e97f8d4c2642addde46b837",
    "cudart-llama-bin-win-cuda-13.4-x64.zip": "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668",
}
LOCAL_ZIP_DIRS = [Path(r"C:\LSS-Work\local-llm\dl")]
REGISTRY = Path(__file__).resolve().parent / "local_llm_models.json"
HOME = Path(os.path.expandvars(r"%LOCALAPPDATA%\LSS-dev"))
DEFAULT_HOME = HOME
RAM_MARGIN_GB = 4.0
KINDS = ("chat", "embedding", "rerank")
EXIT_REFUSED = 3
CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


class RegistryError(ValueError):
    """The model registry is malformed or a model file does not match it."""


class AdmissionRefused(RuntimeError):
    """The PC does not have the RAM / commit headroom to start this model now."""


class ServerFailed(RuntimeError):
    """llama-server exited or did not become healthy."""


def _log(message: str) -> None:
    print(f"[local_llm] {message}", file=sys.stderr, flush=True)


def llama_dir() -> Path:
    if HOME != DEFAULT_HOME:  # explicit isolated runtime home (tests/embedding callers)
        return HOME / 'llama.cpp' / BUILD
    from storage_tiers import load_policy
    target = Path(load_policy().get('llama_runtime_root', str(HOME / 'llama.cpp'))) / BUILD
    legacy = HOME / 'llama.cpp' / BUILD
    return legacy if not target.is_dir() and legacy.is_dir() else target


def runs_file() -> Path:
    return HOME / "local-llm" / "runs.jsonl"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- registry --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    name: str
    path: Path | None
    size_bytes: int
    kind: str
    ctx: int
    args: tuple[str, ...]
    ram_gb: float
    request: dict = field(default_factory=dict)
    notes: str = ""
    sha256: str | None = None


def parse_registry(data: dict) -> dict[str, Model]:
    """Validate the registry document; every problem is listed in one RegistryError."""
    problems, models = [], {}
    rows = data.get("models")
    if not isinstance(rows, dict) or not rows:
        raise RegistryError("registry has no 'models' object")
    for name, row in rows.items():
        def bad(why: str) -> None:
            problems.append(f"{name}: {why}")
        if not isinstance(row, dict):
            bad("entry is not an object")
            continue
        kind = row.get("kind", "chat")
        if kind not in KINDS:
            bad(f"kind {kind!r} is not one of {KINDS}")
        path = row.get("path")
        if path is not None and (not isinstance(path, str) or not path.lower().endswith(".gguf")
                                 or not os.path.isabs(os.path.expandvars(path))):
            bad("path must be an absolute .gguf path")
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            bad("size_bytes must be a positive integer")
        ram = row.get("ram_gb")
        if not isinstance(ram, (int, float)) or isinstance(ram, bool) or ram <= 0:
            bad("ram_gb must be a positive number")
        ctx = row.get("ctx", 16384)
        if not isinstance(ctx, int) or isinstance(ctx, bool) or ctx < 256:
            bad("ctx must be an integer >= 256")
        args = row.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            bad("args must be a list of strings")
        elif any(a in ("-m", "--model", "--host", "--port", "-ctk", "-ctv", "-c", "--ctx-size") for a in args):
            bad("args must not set model, host, port, cache type or ctx (the runner owns them; use 'ctx')")
        request = row.get("request", {})
        if not isinstance(request, dict):
            bad("request must be an object")
        sha = row.get("sha256")
        if sha is not None and not (isinstance(sha, str) and len(sha) == 64):
            bad("sha256 must be 64 hex characters")
        if not any(p.startswith(f"{name}:") for p in problems):
            models[name] = Model(name, Path(os.path.expandvars(path)) if path is not None else None,
                                 size, kind, ctx, tuple(args), float(ram),
                                 request, str(row.get("notes", "")), sha)
    if problems:
        raise RegistryError("; ".join(problems))
    return models


def load_registry(path: Path | None = None) -> dict[str, Model]:
    return parse_registry(load_config(path or REGISTRY))


def get_model(name: str) -> Model:
    models = load_registry()
    if name not in models:
        raise RegistryError(f"unknown model {name!r}; known: {', '.join(sorted(models))}")
    model = models[name]
    if model.path is None:
        raise RegistryError(f"{name}: {availability(model)}")
    return model


def availability(model: Model) -> str:
    if model.path is None:
        return "unavailable (no machine path configured)"
    try:
        size = model.path.stat().st_size
    except OSError:
        return "missing"
    return "ok" if size == model.size_bytes else f"size {size} != {model.size_bytes}"


# --- memory admission ------------------------------------------------------------------------------------------------

def admit(model: Model, *, wait: bool = False, poll: float = 15.0, log=_log, cpu: bool = False) -> None:
    """Start only with available RAM >= working set + 4 GB and commit headroom >= the gpu_run minimum; otherwise
    raise AdmissionRefused, or with `wait` poll until it holds. Waiting changes when the model starts, never what it does."""
    need_ram = max(model.ram_gb, model.size_bytes / 1e9) + RAM_MARGIN_GB if cpu else model.ram_gb + RAM_MARGIN_GB
    shown = False
    while True:
        commit, ram = gpu_run.memory_headroom()
        if ram >= need_ram and commit >= gpu_run.MIN_COMMIT_GB:
            return
        why = (f"{model.name} needs {need_ram:.1f} GB available RAM ({model.ram_gb:g} + {RAM_MARGIN_GB:g}) and "
               f"{gpu_run.MIN_COMMIT_GB:g} GB commit headroom; now {ram:.1f} GB RAM, {commit:.1f} GB commit")
        if not wait:
            raise AdmissionRefused(why)
        if not shown:
            log(f"waiting: {why}")
            shown = True
        time.sleep(poll)


# --- Windows plumbing ------------------------------------------------------------------------------------------------

class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _k32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    return k


class Job:
    """A Windows Job Object that kills every process in it when closed (or when the owner dies)."""

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("the local LLM runner needs Windows")
        self._k = _k32()
        self._handle = self._k.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000                    # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._k.SetInformationJobObject(self._handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.close()
            raise OSError(f"SetInformationJobObject failed: {ctypes.get_last_error()}")

    def assign(self, pid: int) -> None:
        process = self._k.OpenProcess(0x0100 | 0x0001, False, pid)        # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not process:
            raise OSError(f"OpenProcess({pid}) failed: {ctypes.get_last_error()}")
        try:
            if not self._k.AssignProcessToJobObject(self._handle, process):
                raise OSError(f"AssignProcessToJobObject failed: {ctypes.get_last_error()}")
        finally:
            self._k.CloseHandle(process)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._k.TerminateJobObject(handle, 1)
            self._k.CloseHandle(handle)


class _Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_working_set_gb(pid: int) -> float | None:
    """Peak working set of a process in GB (None if it cannot be opened)."""
    if os.name != "nt":
        return None
    k = _k32()
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    handle = k.OpenProcess(0x1000 | 0x0010, False, pid)                   # QUERY_LIMITED_INFORMATION | VM_READ
    if not handle:
        return None
    try:
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        return counters.PeakWorkingSetSize / 2 ** 30 if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb) else None
    finally:
        k.CloseHandle(handle)


def process_table() -> dict[int, tuple[int, str]]:
    """pid -> (parent pid, image name) from a Toolhelp snapshot."""
    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]
    k = ctypes.windll.kernel32
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    table, entry = {}, Entry()
    entry.dwSize = ctypes.sizeof(Entry)
    ok = k.Process32First(snap, ctypes.byref(entry))
    while ok:
        table[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile.decode("mbcs", "replace").lower())
        ok = k.Process32Next(snap, ctypes.byref(entry))
    k.CloseHandle(snap)
    return table


def ancestors(pid: int | None = None) -> list[int]:
    table, out, p = process_table(), [], pid or os.getpid()
    while p in table and table[p][0] not in out and table[p][0] != p:
        p = table[p][0]
        out.append(p)
    return out


def _lease_holder_is_ancestor() -> bool:
    try:
        row = json.loads(Path(gpu_run.HOLDER).read_text(encoding="utf-8"))
        return int(row["pid"]) in ancestors()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _vram_mib() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=15, creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class Sampler(threading.Thread):
    """Peak VRAM (nvidia-smi) and lowest available RAM while a server runs."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval, self.vram_peak, self.ram_min, self._stop_event = interval, None, None, threading.Event()

    def sample(self) -> None:
        vram = _vram_mib()
        if vram is not None:
            self.vram_peak = vram if self.vram_peak is None else max(self.vram_peak, vram)
        ram = gpu_run.memory_headroom()[1]
        self.ram_min = ram if self.ram_min is None else min(self.ram_min, ram)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.sample()
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(20)
        self.sample()                                   # the last state before the server is killed


class CpuSampler(Sampler):
    """Watch only OS RAM; terminate this server if the 4 GB reserve is crossed."""

    def __init__(self, process, interval: float = 2.0):
        super().__init__(interval)
        self.process = process
        self.tripped = False

    def sample(self) -> None:
        ram = gpu_run.memory_headroom()[1]
        self.ram_min = ram if self.ram_min is None else min(self.ram_min, ram)
        if ram < RAM_MARGIN_GB and self.process.poll() is None:
            self.tripped = True
            self.process.kill()


# --- the server ------------------------------------------------------------------------------------------------------

def _launcher() -> list[str]:
    """The command prefix that starts llama-server (tests replace it with a fake)."""
    exe = llama_dir() / "llama-server.exe"
    if not exe.is_file():
        raise ServerFailed(f"{exe} is missing; run: python scripts/local_llm.py install")
    return [str(exe)]


def _cpu_runtime_launcher(runtime_dir, *, cpu):
    """An explicit experimental runtime never changes the shared default."""
    if runtime_dir is None:
        return None
    if not cpu:
        raise RegistryError('An explicit runtime directory requires CPU mode')
    from dev_hot_paths import require_hot
    runtime = Path(runtime_dir).resolve()
    require_hot(runtime, 'explicit CPU runtime')
    exe = runtime / 'llama-server.exe'
    if not exe.is_file():
        raise ServerFailed('Explicit CPU runtime is missing llama-server.exe')
    return [str(exe)]


def server_args(model: Model, port: int, path: Path, ctx: int | None, cpu: bool = False) -> list[str]:
    args = ["-m", str(path), "--host", "127.0.0.1", "--port", str(port), "--no-webui", "-np", "1", "--metrics",
            "-c", str(ctx or model.ctx), "-ngl", "99", "-fa", "on", "-ctk", "f16", "-ctv", "f16", "--fit", "off"]
    if model.kind == "chat":
        args.append("--jinja")
    elif model.kind == "embedding":
        args.append("--embeddings")
    else:
        args.append("--reranking")
    return args + list(model.args) + (["-ngl", "0", "--device", "none", "-t", "2", "-tb", "2", "--no-mmproj-offload"] if cpu else [])


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _NoLocalRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise urllib.error.HTTPError(request.full_url, code, message, headers, response)


def _local_open(request, *, timeout):
    """Owned loopback service only: no ambient proxy or redirect handling.

    Public model acquisition still uses its ordinary urllib path. Language,
    embedding, ranking, health and metrics traffic stays on numeric loopback.
    """
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1')
            or parsed.username is not None or parsed.password is not None or parsed.fragment):
        raise ValueError('Local model requests require a numeric loopback HTTP URL')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoLocalRedirect())
    return opener.open(request, timeout=timeout)


def _http_json(url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with _local_open(request, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_health(process: subprocess.Popen, base: str, timeout: float, log_path: Path) -> float:
    started = time.time()
    while time.time() - started < timeout:
        if process.poll() is not None:
            raise ServerFailed(f"llama-server exited with {process.returncode}: {_tail(log_path)}")
        try:
            with _local_open(f"{base}/health", timeout=3) as response:
                if response.status == 200:
                    return time.time() - started
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise ServerFailed(f"llama-server not healthy after {timeout:.0f} s: {_tail(log_path)}")


def _tail(path: Path, chars: int = 600) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-chars:].strip()
    except OSError:
        return ""


def _metric(base: str, name: str) -> int | None:
    try:
        with _local_open(f"{base}/metrics", timeout=5) as response:
            for line in response.read().decode("utf-8", "replace").splitlines():
                if line.startswith(name + " "):
                    return int(float(line.split()[1]))
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return None


def _append_row(row: dict) -> None:
    path = runs_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prune_logs(directory: Path, keep: int = 20) -> None:
    logs = sorted(directory.glob("*.log"), key=lambda p: p.stat().st_mtime)
    for old in logs[:-keep]:
        with contextlib.suppress(OSError):
            old.unlink()


@contextlib.contextmanager
def serving(name: str, *, ctx: int | None = None, purpose: str = "development", wait_memory: bool = False,
            health_timeout: float = 1800.0, normal_priority: bool = False, log=_log, cpu: bool = False,
            runtime_dir: Path | None = None):
    """Serve registry model `name` and yield its OpenAI-compatible base URL (`http://127.0.0.1:PORT/v1`)."""
    launcher_override = _cpu_runtime_launcher(runtime_dir, cpu=cpu)
    if not cpu:
        from dev_quiet_mode import require_gpu_allowed
        require_gpu_allowed()  # Before model hashing, staging, VRAM probes or lease acquisition.
    model = get_model(name)
    if model.path is None:
        raise RegistryError(f"{name}: {availability(model)}")
    started = time.time()
    row = {"utc": _utc(), "model": name, "purpose": purpose, "staged": None, "stage_seconds": None, "lease_wait_s": None,
           "load_s": None, "ram_peak_gb": None, "ram_avail_min_gb": None, "vram_before_mib": None, "vram_peak_mib": None,
           "vram_after_mib": None, "vram_release_s": None, "prompt_tokens": None, "completion_tokens": None,
           "wall_s": None, "status": "error", "device": "cpu" if cpu else "gpu",
           "runtime_dir": str(Path(launcher_override[0]).parent if launcher_override else llama_dir()),
           "runtime_override": launcher_override is not None}
    job = process = sampler = lease = None
    holder = Path(gpu_run.HOLDER)
    try:
        from dev_hot_paths import require_hot
        for label, hot_path in (("development checkout", ROOT), ("Python environment", sys.prefix),
                                ("temporary files", tempfile.gettempdir()), ("model weights", model.path),
                                ("llama runtime", Path(launcher_override[0]).parent if launcher_override else llama_dir())):
            require_hot(hot_path, label)
        problem = availability(model)
        if problem != "ok":
            raise RegistryError(f"{name}: {model.path} is {problem}")
        admit(model, wait=wait_memory, log=log, **({"cpu": True} if cpu else {}))
        t0 = time.time()
        path = Path(stage_path(model.path, purpose=f"local-llm:{name}"))
        row["stage_seconds"] = round(time.time() - t0, 2)
        row["staged"] = str(path)
        if path != model.path:
            log(f"weights staged: {path} ({row['stage_seconds']} s)")
        else:
            log(f"weights read from the original {path}")
        if not cpu:
            if _lease_holder_is_ancestor():
                raise RuntimeError("the GPU lease is held by an ancestor (gpu_run?): do not run local_llm under gpu_run")
            waited = []

            def on_wait() -> None:
                if not waited:
                    log(f"waiting for the GPU: held by {gpu_run._holder()}")
                waited.append(1)
            t0 = time.time()
            lease = gpu_lease.acquire(on_wait=on_wait)
            row["lease_wait_s"] = round(time.time() - t0, 2)
            holder.parent.mkdir(parents=True, exist_ok=True)
            holder.write_text(json.dumps({"purpose": f"local LLM {name}: {purpose}", "pid": os.getpid(),
                                          "command": ["local_llm.py", "serve", name],
                                          "since": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False), encoding="utf-8")
            admit(model, wait=wait_memory, log=log)                       # the wait for the lease may have been long
            row["vram_before_mib"] = _vram_mib()
        else:
            admit(model, wait=wait_memory, log=log, cpu=True)
        port = free_port()
        logs = HOME / "local-llm" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{time.strftime('%Y%m%dT%H%M%S')}-{name}.log"
        _prune_logs(logs)
        command = (launcher_override or _launcher()) + server_args(model, port, path, ctx, cpu=True) if cpu else _launcher() + server_args(model, port, path, ctx)
        flags = CREATE_NO_WINDOW | (0 if normal_priority else BELOW_NORMAL_PRIORITY_CLASS)
        job = Job()
        cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "none"}
        if cpu:
            for key in ('LLAMA_ARG_RPC', 'GGML_RPC', 'RPC_SERVERS'):
                cpu_env.pop(key, None)
        with open(log_path, "wb") as out:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                       creationflags=flags, **({"env": cpu_env} if cpu else {}))
        try:
            job.assign(process.pid)
        except BaseException:
            # The job cannot clean up a child that was never assigned.
            process.kill()
            process.wait(30)
            raise
        sampler = CpuSampler(process) if cpu else Sampler()
        sampler.start()
        base = f"http://127.0.0.1:{port}"
        row["load_s"] = round(_wait_health(process, base, health_timeout, log_path), 2)
        log(f"{name} ready at {base}/v1 after {row['load_s']} s (log {log_path})")
        try:
            yield f"{base}/v1"
            if cpu and sampler.tripped:
                raise AdmissionRefused("CPU server stopped: available RAM fell below 4 GB")
            row["status"] = "ok"
        except KeyboardInterrupt:
            row["status"] = "interrupted"
            raise
        finally:
            row["prompt_tokens"] = _metric(base, "llamacpp:prompt_tokens_total")
            row["completion_tokens"] = _metric(base, "llamacpp:tokens_predicted_total")
    except BaseException as exc:
        if isinstance(exc, AdmissionRefused):
            row["status"] = "refused: " + str(exc)[:280]
        elif row["status"] != "interrupted":
            row["status"] = f"error: {type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        if process is not None:
            row["ram_peak_gb"] = _round(peak_working_set_gb(process.pid))
        if sampler is not None:
            sampler.stop()
            row["vram_peak_mib"], row["ram_avail_min_gb"] = sampler.vram_peak, _round(sampler.ram_min)
        if job is not None:
            job.close()
        if process is not None:
            with contextlib.suppress(Exception):
                process.wait(30)
            if not cpu:
                row["vram_release_s"], row["vram_after_mib"] = _vram_release(row["vram_before_mib"])
        if lease is not None:
            holder.unlink(missing_ok=True)
            lease.close()
        row["wall_s"] = round(time.time() - started, 2)
        _append_row(row)
        if process is not None:
            log(f"{name} stopped: status {row['status']}, load {row['load_s']} s, RAM peak {row['ram_peak_gb']} GB, "
                f"VRAM {row['vram_before_mib']} -> peak {row['vram_peak_mib']} -> {row['vram_after_mib']} MiB "
                f"(released in {row['vram_release_s']} s)")


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _vram_release(before: int | None, timeout: float = 30.0) -> tuple[float | None, int | None]:
    """Seconds until VRAM is back within 150 MiB of what it was before the server, and the value then."""
    if before is None:
        return None, None
    started = time.time()
    while True:
        now = _vram_mib()
        if now is None:
            return None, None
        if now <= before + 150 or time.time() - started > timeout:
            return (round(time.time() - started, 2) if now <= before + 150 else None), now
        time.sleep(0.25)


# --- client ----------------------------------------------------------------------------------------------------------

@dataclass
class Reply:
    text: str
    reasoning: str
    usage: dict
    timings: dict
    raw: dict

    def __str__(self) -> str:
        return self.text


def complete(base_url: str, messages: list[dict], schema: dict | None = None, *, max_tokens: int = 1024,
             temperature: float = 0.0, extra: dict | None = None, timeout: float = 900.0) -> Reply:
    """One /chat/completions call. `schema` (JSON Schema) turns on strict constrained JSON output; `extra` is merged
    into the request body (e.g. a model's registry `request`: chat_template_kwargs)."""
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}}
    body.update(extra or {})
    raw = _http_json(f"{base_url}/chat/completions", body, timeout)
    message = raw["choices"][0]["message"]
    return Reply(message.get("content") or "", message.get("reasoning_content") or "", raw.get("usage") or {},
                 raw.get("timings") or {}, raw)


def embed(base_url: str, texts: list[str], *, timeout: float = 300.0) -> list[list[float]]:
    raw = _http_json(f"{base_url}/embeddings", {"model": "local", "input": texts}, timeout)
    return [item["embedding"] for item in sorted(raw["data"], key=lambda item: item["index"])]


def rerank(base_url: str, query: str, documents: list[str], *, timeout: float = 300.0) -> list[float]:
    """One finite relevance score per input document; reject incomplete replies.

    Response order is irrelevant. Missing, repeated or fabricated indices cannot
    silently introduce zero scores or overwrite a different document's score.
    """
    if not documents:
        return []
    raw = _http_json(f"{base_url}/rerank", {"model": "local", "query": query, "documents": documents}, timeout)
    results = raw.get("results") if isinstance(raw, dict) else None
    if not isinstance(results, list) or len(results) != len(documents):
        raise ValueError("Reranker must return exactly one result per document")
    scores = [0.0] * len(documents)
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Invalid reranker result")
        index, score = item.get("index"), item.get("relevance_score")
        if type(index) is not int or not 0 <= index < len(documents) or index in seen:
            raise ValueError("Invalid or repeated reranker document index")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Reranker relevance score must be finite and numeric")
        seen.add(index)
        scores[index] = score
    return scores


# --- install ---------------------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_zip(name: str, sha: str, downloads: Path, log) -> Path:
    for folder in [*LOCAL_ZIP_DIRS, downloads]:
        candidate = folder / name
        if candidate.is_file():
            if _sha256(candidate) == sha:
                return candidate
            log(f"{candidate} does not match the pinned sha256; ignoring it")
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / name
    partial = target.with_name(name + ".partial")
    log(f"downloading {RELEASE_URL}{name}")
    with urllib.request.urlopen(RELEASE_URL + name, timeout=60) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 22)
    if _sha256(partial) != sha:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {name} does not match the pinned sha256")
    os.replace(partial, target)
    return target


def install(log=_log) -> Path:
    """Unpack the pinned build; idempotent. Returns the directory holding llama-server.exe."""
    target = llama_dir()
    receipt = target.with_name(f"{BUILD}.receipt.json")
    try:
        done = json.loads(receipt.read_text(encoding="utf-8"))
        if done.get("assets") == ASSETS and (target / "llama-server.exe").is_file():
            log(f"already installed: {target}")
            return target
    except (OSError, ValueError):
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{BUILD}.", suffix=".partial", dir=target.parent))
    try:
        sources = {name: _find_zip(name, sha, target.parent / "downloads", log) for name, sha in ASSETS.items()}
        for name, source in sources.items():
            with zipfile.ZipFile(source) as archive:
                archive.extractall(staging)                          # zipfile drops absolute and '..' members
        if not (staging / "llama-server.exe").is_file():
            raise RuntimeError("llama-server.exe is not in the unpacked build")
        version = subprocess.run([str(staging / "llama-server.exe"), "--version"], capture_output=True, text=True,
                                 timeout=120, creationflags=CREATE_NO_WINDOW)
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
        receipt.write_text(json.dumps({
            "type": "lss_local_llm_install", "build": BUILD, "release_url": RELEASE_URL, "assets": ASSETS,
            "sources": {name: str(path) for name, path in sources.items()}, "directory": str(target),
            "version_output": (version.stdout + version.stderr).strip()[-600:], "utc": _utc()}, indent=1), encoding="utf-8")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"installed {BUILD} at {target}")
    return target


# --- parent watch and CLI --------------------------------------------------------------------------------------------

def _watched_parent(explicit: int | None) -> int | None:
    """The process whose death ends `serve`: --parent-pid, else the parent, skipping the venv launcher that re-spawns us."""
    if explicit:
        return explicit
    table = process_table()
    parent = os.getppid()
    mine = table.get(os.getpid(), (0, ""))[1]
    if parent in table and table[parent][1] == mine and table[parent][0] in table:
        parent = table[parent][0]
    return parent or None


def _alive(pid: int) -> bool:
    k = _k32()
    handle = k.OpenProcess(0x00100000, False, pid)                       # SYNCHRONIZE
    if not handle:
        return False
    try:
        return k.WaitForSingleObject(handle, 0) == 0x102                 # WAIT_TIMEOUT: still running
    finally:
        k.CloseHandle(handle)


def _serving_args(args) -> dict:
    return {"ctx": args.ctx, "purpose": args.purpose, "wait_memory": args.wait,
            "health_timeout": args.health_timeout, "normal_priority": args.normal_priority}


def cmd_models(args) -> int:
    rows = [{"name": m.name, "kind": m.kind, "available": availability(m), "size_gb": round(m.size_bytes / 1e9, 2),
             "ram_gb": m.ram_gb, "ctx": m.ctx, "args": list(m.args),
             "path": str(m.path) if m.path is not None else None} for m in load_registry().values()]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False))
        return 0
    for r in rows:
        print(f"{r['name']:24s} {r['kind']:9s} {r['available']:8s} {r['size_gb']:6.2f} GB file, RAM {r['ram_gb']:g} GB, "
              f"ctx {r['ctx']}  {' '.join(r['args'])}")
    return 0


def cmd_serve(args) -> int:
    parent = _watched_parent(args.parent_pid)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with serving(args.model, **_serving_args(args)) as base:
        state = HOME / "local-llm" / "serving.json"
        state.write_text(json.dumps({"model": args.model, "base_url": base, "pid": os.getpid(), "since": _utc()}), encoding="utf-8")
        print(base, flush=True)
        try:
            while parent is None or _alive(parent):
                time.sleep(1.0)
            _log("parent process ended; stopping")
        except KeyboardInterrupt:
            _log("interrupted; stopping")
        finally:
            state.unlink(missing_ok=True)
    return 0


def cmd_run(args) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("give the command after --")
    with serving(args.model, **_serving_args(args)) as base:
        env = {**os.environ, "OPENAI_BASE_URL": base, "OPENAI_API_BASE": base, "LOCAL_LLM_BASE_URL": base,
               "OPENAI_API_KEY": "local"}
        try:
            exe = str(shutil.which(command[0]) or command[0])
            return subprocess.call([exe, *command[1:]], env=env)
        except OSError as exc:
            _log(f"cannot start {command[0]}: {exc}")
            return 127


def _load_schema(text: str) -> dict:
    path = Path(text)
    return json.loads(path.read_text(encoding="utf-8") if path.is_file() else text)


def cmd_chat(args) -> int:
    model = get_model(args.model)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": prompt}]
    schema = _load_schema(args.schema) if args.schema else None
    with serving(args.model, **_serving_args(args)) as base:
        reply = complete(base, messages, schema, max_tokens=args.max_tokens, extra=model.request)
    sys.stdout.reconfigure(encoding="utf-8")
    print(reply.text)
    if not reply.text.strip() and reply.reasoning:
        _log(f"empty reply: all {reply.usage.get('completion_tokens')} tokens went into reasoning; raise --max-tokens")
    if reply.timings:
        _log(f"prompt {reply.usage.get('prompt_tokens')} tok at {reply.timings.get('prompt_per_second', 0):.0f} t/s, "
             f"completion {reply.usage.get('completion_tokens')} tok at {reply.timings.get('predicted_per_second', 0):.1f} t/s")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("install")
    models = sub.add_parser("models")
    models.add_argument("--json", action="store_true")

    def serve_options(p) -> None:
        p.add_argument("model")
        p.add_argument("--ctx", type=int)
        p.add_argument("--purpose", default="development")
        p.add_argument("--wait", action="store_true", help="wait for RAM headroom instead of refusing")
        p.add_argument("--health-timeout", type=float, default=1800.0)
        p.add_argument("--normal-priority", action="store_true", help="do not lower the server's CPU priority")
    serve = sub.add_parser("serve")
    serve_options(serve)
    serve.add_argument("--parent-pid", type=int, help="stop when this process ends (default: the parent)")
    run = sub.add_parser("run")
    serve_options(run)
    run.add_argument("command", nargs=argparse.REMAINDER)
    chat = sub.add_parser("chat")
    serve_options(chat)
    chat.add_argument("--prompt-file", required=True)
    chat.add_argument("--schema", help="JSON Schema as a file path or inline JSON: strict constrained output")
    chat.add_argument("--system")
    chat.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    try:
        if args.action == "install":
            install()
            return 0
        if args.action == "models":
            return cmd_models(args)
        return {"serve": cmd_serve, "run": cmd_run, "chat": cmd_chat}[args.action](args)
    except AdmissionRefused as exc:
        _log(f"refused: {exc} (use --wait to wait)")
        return EXIT_REFUSED
    except (RegistryError, ServerFailed) as exc:
        _log(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
