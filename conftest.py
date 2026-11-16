"""Test-run guards that protect the developer's machine, whatever way pytest is started.

2026-10-16 incident: a `pytest -n auto` started by a session that later crashed kept ~20 xdist workers alive for
1.5 h (about 700 MB each); the pagefile grew and C: fell from 25 GB to 4.6 GB free. `scripts/dev.py test` already
caps its workers, but a direct `pytest -n auto` did not. Here:
- `-n auto` gets the same cap as `scripts/dev.py` (half the logical cores, at most 8, one per 3 GB free commit);
- a run does not start when the temporary directory's drive has less than `LSS_TEST_MIN_FREE_GB` (default 8) free;
- a run whose starting process has gone (crashed agent or closed terminal) ends itself instead of staying orphaned.

State guard, set here before any product module is imported or a test is collected: the run is in test mode,
`LSS_STATE_DIR` is a temporary folder of this run, and the state root refuses to resolve into the user's product
state (`LSS_DEPLOY_BASE`), the development records (`LSS_DEV_STATE`, this checkout's `.state`) or any
`LSS_PROTECTED_STATE_ROOTS` entry,
links followed (`lss.paths.guard_state_root`).
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path


def _enter_test_mode() -> None:
    protected = [item for item in os.environ.get("LSS_PROTECTED_STATE_ROOTS", "").split(os.pathsep) if item.strip()]
    if os.environ.get("LSS_DEPLOY_BASE"):
        protected.append(os.path.join(os.environ["LSS_DEPLOY_BASE"], "LSS", ".state"))
    if os.environ.get("LSS_DEV_STATE"):
        protected.append(os.environ["LSS_DEV_STATE"])
    # the checkout's own `.state` (a development tree keeps real records there): a test that drops
    # `LSS_STATE_DIR` must not fall back into it
    protected.append(str(Path(__file__).resolve().parent / ".state"))
    os.environ["LSS_PROTECTED_STATE_ROOTS"] = os.pathsep.join(dict.fromkeys(protected))
    if os.environ.get("LSS_TEST_MODE") == "1" and os.environ.get("LSS_TEST_STATE_OWNER"):
        return  # a worker or child of a guarded run: keep the state folder its controller made
    base = os.environ.get("LSS_WORK_ROOT")
    if base:
        os.makedirs(os.path.join(base, "scratch"), exist_ok=True)
    state = tempfile.mkdtemp(prefix="lss-test-state-", dir=os.path.join(base, "scratch") if base else None)
    os.environ.update(LSS_TEST_MODE="1", LSS_STATE_DIR=state, LSS_TEST_STATE_OWNER=str(os.getpid()))
    atexit.register(shutil.rmtree, state, ignore_errors=True)
    _isolate_network_and_models(state)


def _isolate_network_and_models(state: str) -> None:
    """No test reaches the network or the shared model store unless the run opts in with `LSS_TEST_EXTERNAL=1`.

    2026-11-14: application tests started real detached scene-naming children (`python -m lss.bgm.scene_job`),
    which fetched 18 GB of public weights into the checkout's `.models` (a link to the shared store). The scene
    backend is therefore off, and children
    inherit this environment, so they see an empty per-run model folder and cannot download. Explicit real-model
    suites (`scripts/dev.py test --suite gpu` and the other acceptance/native suites) set the opt-in.
    """
    if os.environ.get("LSS_TEST_EXTERNAL") == "1":
        return
    blocked = "http://127.0.0.1:9"  # discard port: a connection fails at once instead of reaching a host
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[name] = blocked
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    # The source-only scene backend is "available" wherever requirements-scene.txt is installed; that made ordinary
    # application tests start real naming children. Tests that exercise backend choice set this variable themselves.
    os.environ.update(LSS_SCENE_BACKEND="none", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      LSS_MODEL_DIR=os.path.join(state, "models"),
                      LSS_SCENE_MODEL_DIR=os.path.join(state, "scene-models"))


_enter_test_mode()

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from dev import available_commit_bytes  # noqa: E402

MAX_AUTO_WORKERS = 8
MIN_FREE_GB = float(os.environ.get("LSS_TEST_MIN_FREE_GB", "8"))
PARENT_CHECK_SECONDS = 30


def auto_workers() -> int:
    return max(1, min(MAX_AUTO_WORKERS, (os.cpu_count() or 2) // 2, available_commit_bytes() // (3 * 2 ** 30)))


def pytest_xdist_auto_num_workers(config):
    return auto_workers()


def _alive(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _processes() -> dict[int, tuple[int, str]]:
    """pid -> (parent pid, executable name) on Windows (Toolhelp snapshot); empty elsewhere or on failure."""
    if sys.platform != "win32":
        return {}
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", ctypes.c_long), ("flags", wintypes.DWORD),
                    ("exe", ctypes.c_wchar * 260)]

    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return {}
    found = {}
    entry = Entry(size=ctypes.sizeof(Entry))
    try:
        ok = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            found[int(entry.pid)] = (int(entry.parent), entry.exe)
            ok = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return found


def starter() -> int:
    """The process that started this run: the parent, skipping a venv `python.exe` launcher in between."""
    parent = os.getppid()
    table = _processes()
    # `.venv\Scripts\python.exe` is a launcher that waits for the real interpreter (and `scripts/dev.py test` is one
    # more Python level), so these outlive a dead starter: watch the first non-Python ancestor
    seen = set()
    while parent in table and table[parent][1].lower() in ("python.exe", "pythonw.exe") and parent not in seen:
        seen.add(parent)
        if table[parent][0] not in table:
            break
        parent = table[parent][0]
    return parent


def _watch_parent(parent: int) -> None:
    while True:
        time.sleep(PARENT_CHECK_SECONDS)
        if not _alive(parent):
            sys.stderr.write(f"\nLSS test guard: the process that started this run ({parent}) is gone; stopping.\n")
            sys.stderr.flush()
            os._exit(3)  # xdist workers exit when their controller's channel closes


def pytest_configure(config):
    if hasattr(config, "workerinput"):  # xdist worker: the controller guards the run
        return
    free = shutil.disk_usage(tempfile.gettempdir()).free / 2 ** 30
    if free < MIN_FREE_GB:
        pytest.exit(f"LSS test guard: only {free:.1f} GB free on the temporary drive (< {MIN_FREE_GB:g} GB). "
                    "Free space first (scripts/dev_hygiene.ps1) or set LSS_TEST_MIN_FREE_GB.", returncode=4)
    parent = starter()
    if parent > 1 and os.environ.get("LSS_TEST_PARENT_WATCH", "1") != "0":
        threading.Thread(target=_watch_parent, args=(parent,), daemon=True, name="lss-parent-watch").start()


@pytest.fixture(autouse=True)
def _fresh_audio_content_identities():
    """Tests synthesize the same samples in many places; an identity verified by decoding in one test must not
    answer for byte-identical files of another (a test may change what a decoder reports for them)."""
    from lss.core import audio_identity

    with audio_identity._IDENTITIES_LOCK:
        audio_identity._BY_CONTENT.clear()
    yield


@pytest.fixture(autouse=True)
def _no_real_gui_cookie_files(monkeypatch, tmp_path_factory):
    """Tests never pick up the user's real yt-dlp GUI cookie files; a test that wants one sets its own folder."""
    from lss.acquisition import login_profile

    monkeypatch.setattr(login_profile, "GUI_COOKIE_DIR", tmp_path_factory.getbasetemp() / "no-gui-cookies")


def _local_entries():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    try:
        from local_tests import LOCAL_FILES, LOCAL_TESTS
    finally:
        sys.path.pop(0)
    return LOCAL_FILES, LOCAL_TESTS


def _local_deselected(config) -> bool:
    expression = (config.getoption("markexpr", "") or "").replace(" ", "")
    return "notlocal" in expression


def pytest_ignore_collect(collection_path, config):
    """A listed file that cannot even be imported here is not collected when `local` is deselected."""
    if not _local_deselected(config):
        return None
    files, _ = _local_entries()
    relative = Path(collection_path).resolve().relative_to(Path(__file__).resolve().parent).as_posix() \
        if Path(collection_path).resolve().is_relative_to(Path(__file__).resolve().parent) else ""
    return True if relative in files else None


def pytest_collection_modifyitems(config, items):
    """Mark `local` the tests that need material this public tree does not carry (tests/local_tests.py)."""
    files, tests = _local_entries()
    for item in items:
        path, _, rest = item.nodeid.partition("::")
        node = path + "::" + rest.split("[", 1)[0] if rest else path
        if path in files or node in tests:
            item.add_marker(pytest.mark.local)
