"""起動から終了までを、本物のプロセスで1度通す。

「終了を選んだらLSSが本当に居なくなる」はプロセスが実際に終わることでしか確かめられない。
ここで見るのは3つ。

1. WebView2ホストの実行が戻る（＝終了が要求として届く）
2. プロセスが終了する（＝非デーモンのスレッドが残っていない）
3. 常駐ワーカー（別プロセス）も落ちる（＝GPUとパイプを掴んだまま残らない）
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from multiprocessing import connection
from multiprocessing.context import AuthenticationError
from pathlib import Path
from uuid import uuid4

import pytest

from lss.core import protocol

REPOSITORY = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent(
    '''
    import sys, threading, time
    sys.path.insert(0, {repository!r})

    from lss.core import client
    from lss.ui.web.host import Host

    ADDRESS = {address!r}

    # 終了要求の宛先だけを試験用のパイプへ向ける。経路は製品と同じ。
    original = client.request_worker_shutdown
    client.request_worker_shutdown = lambda *a, **k: original(ADDRESS)

    client.ensure_worker_running(address=ADDRESS)
    print("worker_running", client.is_worker_running(ADDRESS), flush=True)

    host = Host()

    def close_from_tray():
        if not host._loaded.wait(90):
            print("page_not_loaded", flush=True)
        time.sleep(0.5)
        host.quit()

    threading.Thread(target=close_from_tray, daemon=True).start()
    code = host.run(fullscreen=False, x=-4000, y=0, width=800, height=500)
    print("run_returned", code, flush=True)
    print("threads", sorted(t.name for t in threading.enumerate()
                            if t is not threading.main_thread() and not t.daemon), flush=True)
    '''
)


def _worker_alive(address: str) -> bool:
    conn = None
    try:
        conn = connection.Client(address, authkey=protocol.AUTHKEY)
        conn.send({"cmd": "ping", "protocol_version": protocol.VERSION})
        if not conn.poll(protocol.PROBE_TIMEOUT_SECONDS):
            return False
        return isinstance(conn.recv(), dict)
    except (AuthenticationError, EOFError, FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    finally:
        if conn is not None:
            conn.close()


@pytest.mark.skipif(sys.platform != "win32", reason="WebView2はWindowsだけ")
def test_quitting_ends_the_process_and_the_resident_worker(tmp_path):
    address = fr"\\.\pipe\lss-lifecycle-{uuid4().hex}"
    script = tmp_path / "child.py"
    script.write_text(
        CHILD.format(repository=str(REPOSITORY), address=address), encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["LSS_STATE_DIR"] = str(tmp_path / "state")

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPOSITORY), env=environment, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=240,
    )

    output = completed.stdout + completed.stderr
    assert "worker_running True" in completed.stdout, output
    assert "page_not_loaded" not in completed.stdout, output
    assert "run_returned 0" in completed.stdout, output
    # 残った非デーモンのスレッドはそのままプロセスの居残りになる。
    assert "threads []" in completed.stdout, output
    assert completed.returncode == 0, output

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and _worker_alive(address):
        time.sleep(0.2)
    assert not _worker_alive(address), "常駐ワーカーがLSS終了後も生き残っている"


# ---- 参照構図を実画面へ写す倍率 ---------------------------------------------

def test_the_reference_composition_fits_the_target_display() -> None:
    """2560x1440 / 150%: 論理1706.7x960 に 1440x900 を高さ基準で収める。"""
    from lss.ui.web.host import fit_factor

    assert fit_factor(2560, 1440, 1.5) == pytest.approx(960 / 900)
    assert fit_factor(1440, 900, 1.0) == pytest.approx(1.0)
    # 縦長の画面では幅が決める。余りは背景の非意味的な余白が受ける。
    assert fit_factor(1440, 1800, 1.0) == pytest.approx(1.0)
    assert fit_factor(0, 900, 1.0) is None
