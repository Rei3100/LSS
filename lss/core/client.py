"""worker.pyへジョブを送るクライアント側ヘルパ(T322のUIプロセスから使われる想定)。"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import threading
from collections.abc import Callable
from multiprocessing import connection
from multiprocessing.context import AuthenticationError

from lss.core import protocol
from lss.paths import repo_root

REPO_ROOT = repo_root()
_startup_lock = threading.Lock()
# LSS を終了している。新しいworkerを起動せず、ジョブも送らない（終了後に常駐を残さない）。
_closing = threading.Event()
# このprocessが起動したworker（起動の途中で終了したとき、終了要求を届け直す）。
_spawned: subprocess.Popen | None = None
# 起動したworkerが待ち受けるまで待つ上限。入れ替えた直後のpackageの最初の起動は、冷えたディスクから
# 実行ファイル群（~5 GB）を読む。他の読み書きで混んだディスクでは25秒を超え、止めて起動し直した2回目も
# 冷えたまま超えて行が失敗した（2026-10-15、凍結3b3a456。温まった後は2.2秒）。生きているworkerを
# 途中で止めても速くはならないので、長めに待つ。
WORKER_STARTUP_SECONDS = 120.0


class ClosingError(OSError):
    """LSS の終了後に、workerの起動やジョブの送信を求められた。"""


def close() -> None:
    """LSS を終了する。以後このprocessからworkerを起動せず、ジョブを送らない。"""
    _closing.set()


def finish_worker_shutdown(*, abandon: bool, timeout: float = WORKER_STARTUP_SECONDS) -> bool:
    """このprocessが起動したworkerへ、終了要求が届くまで送り直す。届いた・既に居なければ True。

    workerの起動の途中で LSS を終了すると、最初の終了要求は届かない（まだ待ち受けていない）。
    そのままでは画面の無い常駐workerが残る。起動を待つ上限は起動の上限と同じ。
    """
    process = _spawned
    if process is None or process.poll() is not None:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if request_worker_shutdown(abandon=abandon) is not None:
            return True
        if process.poll() is not None:
            return True
        time.sleep(0.2)
    return False


def is_worker_running(address: str = protocol.ADDRESS) -> bool:
    conn = None
    try:
        conn = connection.Client(address, authkey=protocol.AUTHKEY)
        conn.send({"cmd": "ping", "protocol_version": protocol.VERSION})
        if not conn.poll(protocol.PROBE_TIMEOUT_SECONDS):
            return False
        response = conn.recv()
        return response == {"status": "ok", "protocol_version": protocol.VERSION}
    except (AuthenticationError, EOFError, FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    finally:
        if conn is not None:
            conn.close()


def request_worker_shutdown(
    address: str = protocol.ADDRESS, timeout: float = protocol.PROBE_TIMEOUT_SECONDS,
    *, abandon: bool = False,
) -> dict | None:
    """常駐ワーカーへ終了を要求する。

    ワーカーは新しいジョブを受け付けなくなり、実行中の1件を公開まで仕上げてから落ちる。
    `abandon` は利用者が処理中に終了したとき。実行中の1件は公開の前で止め、保存しない
    （公開を始めていれば最後まで運ぶ）。
    応答が無くても呼び出し側(終了処理)を止めない。
    """
    conn = None
    try:
        conn = connection.Client(address, authkey=protocol.AUTHKEY)
        command = {"cmd": "shutdown", "protocol_version": protocol.VERSION}
        if abandon:
            command["abandon"] = True
        conn.send(command)
        if not conn.poll(timeout):
            return None
        response = conn.recv()
        return response if isinstance(response, dict) else None
    except (AuthenticationError, EOFError, FileNotFoundError, ConnectionRefusedError, OSError):
        return None
    finally:
        if conn is not None:
            conn.close()


def spawn_worker(address: str = protocol.ADDRESS) -> subprocess.Popen:
    """workerを別プロセスとして起動する(D-5 L3)。

    PyInstaller onedirへ凍結された実行ツリーではsys.executableがLSS.exe自身であり
    `-m`モジュール起動を受け付けない。凍結判定(`sys.frozen`)時は同じexeを`--worker`引数付きで
    起動し、エントリポイントでそのフラグを見てworker.main()へ直接分岐させる
    (decisions.md 2026-08-14・単一onedirのままworker役を持たせるための標準パターン)。
    """
    # Strict SFC profiles need cuBLAS configured before the child imports torch. Select the
    # profile from the parent environment, validate it before launch, and pass the required
    # workspace only to that child. The UI process environment stays untouched.
    from lss.bgm import gpu_execution

    profile_value = os.environ.get("LSS_DEV_SFC_EXECUTION_PROFILE")
    profile = gpu_execution.resolve_execution_profile(profile_value)
    child_environment = None
    if profile is not None:
        required_workspace = ":4096:8"
        inherited_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if inherited_workspace not in (None, required_workspace):
            raise ValueError("Strict SFC execution profile requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        child_environment = os.environ.copy()
        child_environment["CUBLAS_WORKSPACE_CONFIG"] = required_workspace

    if getattr(sys, "frozen", False):
        args = [sys.executable, "--worker", "--address", address]
    else:
        args = [sys.executable, "-m", "lss.core.worker", "--address", address]
    options = {"cwd": str(REPO_ROOT), "creationflags": subprocess.CREATE_NO_WINDOW}
    if child_environment is not None:
        options["env"] = child_environment
    return subprocess.Popen(args, **options)


def ensure_worker_running(startup_timeout: float = 20.0, address: str = protocol.ADDRESS) -> None:
    global _spawned
    with _startup_lock:
        if _closing.is_set():
            raise ClosingError("LSS を終了しています")
        if is_worker_running(address):
            return
        proc = spawn_worker(address)
        if address == protocol.ADDRESS:
            _spawned = proc
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if is_worker_running(address):
                return
            if proc.poll() is not None:
                raise RuntimeError(f"分離ワーカーの起動に失敗しました（終了コード: {proc.returncode}）")
            if _closing.is_set():
                # 起動を待つ間に終了した。起動途中のworkerへは終了処理が終了要求を届け直す。
                raise ClosingError("LSS を終了しています")
            time.sleep(0.2)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        raise TimeoutError(f"workerが{startup_timeout}秒以内に起動しなかった")


def dispatch_job(
    job: dict, address: str = protocol.ADDRESS, connect_timeout: float = WORKER_STARTUP_SECONDS,
    on_progress: Callable[[dict], None] | None = None, transport_retries: int = 1,
) -> dict:
    if transport_retries < 0:
        raise ValueError("transport_retriesは0以上である必要があります")
    payload = dict(job)
    if on_progress is not None:
        payload["_stream_progress"] = True
    deadline = None
    while True:
        response = _dispatch_once(payload, address, connect_timeout, on_progress, transport_retries)
        if (isinstance(response, dict) and response.get("error_code") == "worker_shutting_down"
                and not _closing.is_set()):
            # 直前に終了した LSS の常駐workerに当たった（終了後すぐに起動し直した）。そのworkerは未開始の
            # ジョブを受け付けずに落ちる。行を失敗にせず、新しいworkerで続ける。
            deadline = deadline or time.monotonic() + SHUTTING_DOWN_WAIT_SECONDS
            if time.monotonic() < deadline:
                time.sleep(0.5)
                continue
        return response


# 終了中の前のworkerが待ち受けを閉じるのは要求の直後。待つのはその短い間だけ。
SHUTTING_DOWN_WAIT_SECONDS = 30.0


def _dispatch_once(payload: dict, address: str, connect_timeout: float,
                   on_progress: Callable[[dict], None] | None, transport_retries: int) -> dict:
    last_error: OSError | EOFError | None = None
    for attempt in range(transport_retries + 1):
        conn = None
        try:
            ensure_worker_running(startup_timeout=connect_timeout, address=address)
            if _closing.is_set():
                # 起動を待っている間に終了した。終了要求の後にジョブを届けない。
                raise ClosingError("LSS を終了しています")
            conn = connection.Client(address, authkey=protocol.AUTHKEY)
            conn.send(payload)
            while True:
                response = conn.recv()
                if isinstance(response, dict) and response.get("type") == "progress":
                    if on_progress is not None:
                        try:
                            on_progress({key: value for key, value in response.items() if key != "type"})
                        except Exception:
                            # 表示更新の失敗で、進行中の分離結果を受け取れなくしない。
                            pass
                    continue
                return response
        except (EOFError, OSError) as exc:
            last_error = exc
            if attempt >= transport_retries:
                raise
        finally:
            if conn is not None:
                conn.close()
    raise last_error or RuntimeError("分離ワーカーとの通信に失敗しました")
