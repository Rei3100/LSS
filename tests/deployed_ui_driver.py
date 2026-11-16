"""配備された実行物の画面を、隔離した状態で起動して操作する受入用の道具。

ウィンドウはOSの本物（WebView2・ファイル選択ダイアログ・トレイのメニュー窓）を使い、
操作はDevTools ProtocolのキーとポインタのイベントとWin32のダイアログ操作で行う。
利用者の画面を奪わないよう、アプリは同じウィンドウステーション内の別デスクトップ（入力を受けない
隠しデスクトップ）で起動する。製品は最小化指定を無視して前面に出るため、画面外へ移す方式（作れない
ときの代替）では一瞬の点滅と前面奪取が残る。Designのページ・ホスト・バックエンドの実装には手を入れない。
"""
from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path

import websocket

user32 = ctypes.windll.user32
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = ctypes.c_ssize_t
user32.GetDlgCtrlID.argtypes = [wintypes.HWND]
user32.GetParent.argtypes = [wintypes.HWND]
user32.GetParent.restype = wintypes.HWND

WM_SETTEXT = 0x000C
WM_COMMAND = 0x0111
IDOK = 1
_ENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


SW_SHOWMINNOACTIVE = 7
SW_SHOWNOACTIVATE = 4
SWP_NOSIZE, SWP_NOZORDER, SWP_NOACTIVATE = 0x0001, 0x0004, 0x0010


class _WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT), ("showCmd", wintypes.UINT),
                ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                ("rcNormalPosition", wintypes.RECT)]


def _offscreen_x(width: int) -> int:
    # 全モニタを含む仮想画面の左端より、さらに窓の幅だけ左（どのモニタ配置でも見えない）。
    return user32.GetSystemMetrics(76) - width - 400  # SM_XVIRTUALSCREEN


def _offscreen_window(hwnd) -> None:
    """表示中の窓を大きさを変えずに画面外へ移す。アクティブにしない。"""
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    user32.SetWindowPos(hwnd, None, _offscreen_x(rect.right - rect.left), rect.top, 0, 0,
                        SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)


def _hide_offscreen(hwnd) -> None:
    """最小化で出た窓を、画面外の位置で通常の大きさに戻す（描画とDevToolsは通常どおり動く）。"""
    placement = _WINDOWPLACEMENT()
    placement.length = ctypes.sizeof(placement)
    user32.GetWindowPlacement(hwnd, ctypes.byref(placement))
    normal = placement.rcNormalPosition
    width, height = normal.right - normal.left, normal.bottom - normal.top
    if width <= 0 or height <= 0:
        width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    x = _offscreen_x(width)
    placement.rcNormalPosition = wintypes.RECT(x, normal.top, x + width, normal.top + height)
    placement.showCmd = SW_SHOWNOACTIVATE
    placement.flags = 0
    user32.SetWindowPlacement(hwnd, ctypes.byref(placement))
    _offscreen_window(hwnd)  # 最大化などで通常位置を使わなかった場合も画面外へ


# --- 隠しデスクトップ -------------------------------------------------------------------------
kernel32 = ctypes.windll.kernel32
GENERIC_ALL = 0x10000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESHOWWINDOW = 0x00000001
STILL_ACTIVE = 259
WAIT_OBJECT_0, WAIT_TIMEOUT = 0, 0x102
user32.CreateDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPVOID, wintypes.DWORD,
                                  wintypes.DWORD, wintypes.LPVOID]
user32.CreateDesktopW.restype = wintypes.HANDLE
user32.CloseDesktop.argtypes = [wintypes.HANDLE]
user32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.LPARAM]
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

# pid -> 隠しデスクトップのハンドル。top_windows はこれがあるプロセスだけ EnumDesktopWindows で数える。
_DESKTOP_OF_PID: dict[int, int] = {}


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD), ("lpReserved2", ctypes.c_void_p),
                ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


class DesktopProcess:
    """別デスクトップで起動したプロセス。Popen と同じ pid / poll / wait / kill / returncode を持つ。"""

    def __init__(self, handle: int, pid: int, desktop: int) -> None:
        self._handle, self.pid, self._desktop = handle, pid, desktop
        self.returncode: int | None = None
        _DESKTOP_OF_PID[pid] = desktop

    def poll(self) -> int | None:
        if self.returncode is None:
            if kernel32.WaitForSingleObject(self._handle, 0) == WAIT_OBJECT_0:
                code = wintypes.DWORD()
                kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code))
                self.returncode = code.value
                self._release()
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.05)
        return self.returncode

    def kill(self) -> None:
        if self.poll() is None:
            kernel32.TerminateProcess(self._handle, 1)

    def _release(self) -> None:
        # 子（常駐ワーカー）が残っていても、デスクトップは最後のハンドルが閉じられた時点で消える。
        _DESKTOP_OF_PID.pop(self.pid, None)
        if self._handle:
            kernel32.CloseHandle(self._handle)
            self._handle = 0
        if self._desktop:
            user32.CloseDesktop(self._desktop)
            self._desktop = 0


def launch_on_hidden_desktop(command: list[str], cwd: Path, environment: dict[str, str]) -> DesktopProcess:
    """入力を受けない別デスクトップでプロセスを起動する。作れなければ OSError。"""
    name = f"LSS-test-{os.getpid()}-{time.monotonic_ns() % 10**9}"
    desktop = user32.CreateDesktopW(name, None, None, 0, GENERIC_ALL, None)
    if not desktop:
        raise ctypes.WinError()
    startup = _STARTUPINFOW()
    startup.cb = ctypes.sizeof(startup)
    startup.lpDesktop = name
    startup.dwFlags = STARTF_USESHOWWINDOW
    startup.wShowWindow = SW_SHOWNOACTIVATE
    info = _PROCESS_INFORMATION()
    block = "".join(f"{k}={v}\0" for k, v in sorted(environment.items(), key=lambda kv: kv[0].upper())) + "\0"
    environment_block = ctypes.create_unicode_buffer(block, len(block))
    line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
    ok = kernel32.CreateProcessW(None, line, None, None, False, CREATE_UNICODE_ENVIRONMENT, environment_block,
                                 str(cwd), ctypes.byref(startup), ctypes.byref(info))
    if not ok:
        error = ctypes.WinError()
        user32.CloseDesktop(desktop)
        raise error
    kernel32.CloseHandle(info.hThread)
    return DesktopProcess(info.hProcess, info.dwProcessId, desktop)


def _set_text(hwnd, text: str) -> None:
    buffer = ctypes.create_unicode_buffer(text)
    user32.SendMessageW(hwnd, WM_SETTEXT, 0, ctypes.addressof(buffer))


def _get_text(hwnd) -> str:
    length = user32.SendMessageW(hwnd, 0x000E, 0, 0)  # WM_GETTEXTLENGTH
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.SendMessageW(hwnd, 0x000D, length + 1, ctypes.addressof(buffer))  # WM_GETTEXT
    return buffer.value


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _text(hwnd) -> str:
    buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buffer, 512)
    return buffer.value


def _class(hwnd) -> str:
    buffer = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, buffer, 128)
    return buffer.value


def top_windows(pid: int) -> list[int]:
    found: list[int] = []

    def visit(hwnd, _param):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    desktop = _DESKTOP_OF_PID.get(pid)
    if desktop:  # 隠しデスクトップ上の窓は EnumWindows では見えない
        user32.EnumDesktopWindows(desktop, ctypes.cast(_ENUM(visit), ctypes.c_void_p), 0)
    else:
        user32.EnumWindows(_ENUM(visit), 0)
    return found


def child_windows(hwnd) -> list[int]:
    found: list[int] = []

    def visit(child, _param):
        found.append(child)
        return True

    user32.EnumChildWindows(hwnd, _ENUM(visit), 0)
    return found


def explorer_windows() -> list[int]:
    found: list[int] = []

    def visit(hwnd, _param):
        if user32.IsWindowVisible(hwnd) and _class(hwnd) == "CabinetWClass":
            found.append(hwnd)
        return True

    user32.EnumWindows(_ENUM(visit), 0)
    return found


def wait_for(predicate, timeout: float, what: str, interval: float = 0.1):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"時間内に成立しない: {what}")
        time.sleep(interval)


class Page:
    """DevTools Protocolの1ページ。"""

    def __init__(self, websocket_url: str) -> None:
        self._socket = websocket.create_connection(websocket_url, timeout=60, suppress_origin=True)
        self._next = 0
        self._scale: float | None = None

    def close(self) -> None:
        self._socket.close()

    def call(self, method: str, params: dict | None = None) -> dict:
        self._next += 1
        ident = self._next
        self._socket.send(json.dumps({"id": ident, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self._socket.recv())
            if message.get("id") == ident:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def eval(self, expression: str):
        result = self.call("Runtime.evaluate", {
            "expression": expression, "awaitPromise": True, "returnByValue": True,
        })
        if "exceptionDetails" in result:
            raise RuntimeError(f"ページ内の評価に失敗: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def state(self) -> dict:
        return self.eval(
            "(() => { const c = window.__lssComponent; if (!c) return null; const s = c.state;"
            " return {phase: s.phase, queue: s.queue, hist: s.hist, st: s.st, ci: s.ci, ph: s.ph,"
            " stopReq: s.stopReq, stopped: s.stopped, allDone: s.allDone, fault: s.fault,"
            " selId: s.selId, songs: c.songs()}; })()")

    def key(self, key: str, *, code: str, vk: int, shift: bool = False) -> None:
        modifiers = 8 if shift else 0
        text = key if len(key) == 1 else ""
        self.call("Input.dispatchKeyEvent", {
            "type": "keyDown", "key": key, "code": code, "windowsVirtualKeyCode": vk,
            "text": text, "unmodifiedText": text.lower(), "modifiers": modifiers,
        })
        self.call("Input.dispatchKeyEvent", {
            "type": "keyUp", "key": key, "code": code, "windowsVirtualKeyCode": vk, "modifiers": modifiers,
        })

    def _pointer_scale(self) -> float:
        """入力イベントの座標とページのCSS座標の比。WebView2のページ倍率の分だけずれる。"""
        if self._scale is None:
            self.eval(
                "window.__lssProbe = null; window.addEventListener('mousemove',"
                " e => { window.__lssProbe = [e.clientX, e.clientY]; }, {once: true}); true")
            self.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 400, "y": 300})
            client = wait_for(lambda: self.eval("window.__lssProbe"), 5, "ポインタの較正")
            self._scale = 400 / client[0]
        return self._scale

    def click_text(self, *needles: str) -> None:
        """全ての文字列を含む最も内側の可視要素を、本物のポインタイベントで押す。"""
        box = self.eval(
            "(() => { const needles = " + json.dumps(needles, ensure_ascii=False) + ";"
            " const all = [...document.querySelectorAll('[data-lss-frame] *')].filter(el => {"
            "  const t = el.textContent || ''; if (!needles.every(n => t.includes(n))) return false;"
            "  const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; });"
            " const inner = all.filter(el => !all.some(o => o !== el && el.contains(o)));"
            " const hits = inner.length ? inner : all; if (!hits.length) return null;"
            " let el = hits[hits.length - 1]; while (el && getComputedStyle(el).cursor !== 'pointer') el = el.parentElement;"
            " el = el || hits[hits.length - 1]; const r = el.getBoundingClientRect();"
            " return [r.left + r.width / 2, r.top + r.height / 2]; })()")
        if box is None:
            raise AssertionError(f"押す対象が画面に無い: {needles}")
        scale = self._pointer_scale()
        x, y = box[0] * scale, box[1] * scale
        self.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        for kind in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", {
                "type": kind, "x": x, "y": y, "button": "left", "clickCount": 1,
            })

    def screenshot(self, path: Path) -> None:
        import base64

        data = self.call("Page.captureScreenshot", {"format": "png"})["data"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data))

    def frame_intervals(self, frames: int = 240) -> dict:
        values = self.eval(
            "new Promise(resolve => { const t = []; const step = ts => { t.push(ts);"
            f" if (t.length < {frames + 1}) requestAnimationFrame(step); else resolve(t); }};"
            " requestAnimationFrame(step); })")
        gaps = sorted(b - a for a, b in zip(values, values[1:]))
        return {
            "frames": len(gaps), "median_ms": round(gaps[len(gaps) // 2], 2),
            "p95_ms": round(gaps[int(len(gaps) * 0.95)], 2), "max_ms": round(gaps[-1], 2),
        }


class DeployedApp:
    """隔離した配備版LSSの画面プロセス1つ。"""

    def __init__(self, executable: Path, environment: dict[str, str]) -> None:
        self.executable = executable
        self.environment = environment
        self.process: subprocess.Popen | DesktopProcess | None = None
        self.hidden_desktop = False
        self.port = 0
        self.page: Page | None = None
        self.command: list[str] | None = None  # 開発ツリーの画面で同じ操作をするときの起動コマンド

    def launch(self) -> Page:
        self.port = free_port()
        environment = {
            **self.environment,
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS": f"--remote-debugging-port={self.port}",
        }
        command = self.command or [str(self.executable)]
        self.process = None
        # 利用者の作業を奪わない: 既定は入力を受けない別デスクトップで起動する（LSS_UI_DESKTOP=visible で無効）。
        # 作れないときだけ、最小化・非アクティブで出して画面外へ移す従来の方式に戻る（点滅と前面奪取が残る）。
        if os.environ.get("LSS_UI_DESKTOP", "hidden") != "visible":
            try:
                self.process = launch_on_hidden_desktop(command, self.executable.parent, environment)
                self.hidden_desktop = True
            except OSError:
                self.process = None
        if self.process is None:
            self.hidden_desktop = False
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = SW_SHOWMINNOACTIVE
            self.process = subprocess.Popen(command, cwd=self.executable.parent, env=environment,
                                            startupinfo=startup)
        window = wait_for(self._main_window, 60, "画面のウィンドウ", interval=0.01)
        if not self.hidden_desktop:
            _hide_offscreen(window)
        self.page = wait_for(self._ready_page, 90, "ページの準備", interval=0.5)
        time.sleep(1.2)  # 起動の入場モーション
        return self.page

    def _ready_page(self) -> Page | None:
        # 読み込み中はabout:blankから移動し、DevToolsの接続が切れることがある。つなぎ直す。
        target = self._target("page", "index.html")
        if target is None:
            return None
        try:
            page = Page(target)
        except (OSError, websocket.WebSocketException):
            return None
        try:
            if page.eval("!!(window.__lssComponent && window.pywebview && window.pywebview.api)"):
                return page
        except (RuntimeError, OSError, websocket.WebSocketException):
            pass
        page.close()
        return None

    def _main_window(self):
        if self.process.poll() is not None:
            raise AssertionError(f"画面プロセスが終了した: {self.process.returncode}")
        for hwnd in top_windows(self.process.pid):
            if _text(hwnd) == "LSS" and _class(hwnd).startswith("WindowsForms"):
                return hwnd
        return None

    def _target(self, kind: str, url_part: str = "") -> str | None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json", timeout=2) as response:
                targets = json.load(response)
        except OSError:
            return None
        for target in targets:
            url = target.get("url", "")
            if target.get("type") != kind:
                continue
            if url_part and url_part not in url:
                continue
            if not url_part and "tray_menu" in url:
                continue
            return target["webSocketDebuggerUrl"]
        return None

    def menu_page(self) -> Page:
        return Page(wait_for(lambda: self._target("page", "tray_menu"), 30, "トレイのメニュー窓"))

    def choose_with_key(self, page: "Page", directory: Path, names: list[str] | None) -> None:
        """Aキー（ファイル）かShift+Aキー（フォルダ）で選択ダイアログを開き、選んで確定する。

        直前の選択の応答がページへ戻る前のキーは、Designが二重に開かないよう無視する。
        新しいダイアログが出るまでキーを送り直す。
        """
        previous = self._dialog()
        for _attempt in range(5):
            page.key("A" if names is None else "a", code="KeyA", vk=65, shift=names is None)
            try:
                dialog = wait_for(lambda: (lambda h: h if h and h != previous else None)(self._dialog()),
                                  4, "選択ダイアログ", interval=0.01)
                _offscreen_window(dialog)  # 前面に出さない。操作はメッセージで行う
                break
            except TimeoutError:
                continue
        else:
            raise TimeoutError("選択ダイアログが開かない")
        self._choose(dialog, directory, names)

    def _choose(self, dialog, directory: Path, names: list[str] | None) -> None:
        edit = wait_for(lambda: self._name_edit(dialog), 10, "ダイアログの名前欄")
        _set_text(edit, str(directory))
        user32.SendMessageW(dialog, WM_COMMAND, IDOK, 0)

        def closed() -> bool:
            return not user32.IsWindow(dialog) or not user32.IsWindowVisible(dialog)

        # 入力したパスへ移動すると、名前欄はそのフォルダ名に変わる。
        def moved() -> bool:
            if closed():
                return True
            current = self._name_edit(dialog)
            return current is not None and _get_text(current) == (directory.name if names is None else "")

        if names is None:
            wait_for(moved, 15, "フォルダへの移動")
            if not closed():
                user32.SendMessageW(dialog, WM_COMMAND, IDOK, 0)  # 移動した先を「フォルダーの選択」で確定
            wait_for(closed, 15, "フォルダ選択の確定")
            return
        # 開いた直後のダイアログはOKを取りこぼすことがある（再起動直後の2回目の選択で実測）。移動しなければ入れ直す。
        for _attempt in range(3):
            try:
                wait_for(lambda: (lambda e: e is not None and _get_text(e) != str(directory))(self._name_edit(dialog)),
                         6, "フォルダへの移動")
                break
            except TimeoutError:
                edit = wait_for(lambda: self._name_edit(dialog), 10, "ダイアログの名前欄")
                _set_text(edit, str(directory))
                user32.SendMessageW(dialog, WM_COMMAND, IDOK, 0)
        # 前回と同じフォルダで開いたダイアログは、そのフォルダのパスでは名前欄を変えない。名前の入力へ進む
        # （別のフォルダなら名前が見つからずダイアログが閉じないので、下の「ファイル選択の確定」で失敗する）。
        time.sleep(0.3)
        edit = wait_for(lambda: self._name_edit(dialog), 10, "ダイアログの名前欄")
        _set_text(edit, " ".join(f'"{name}"' for name in names))
        user32.SendMessageW(dialog, WM_COMMAND, IDOK, 0)
        wait_for(closed, 15, "ファイル選択の確定")

    def _dialog(self):
        for hwnd in top_windows(self.process.pid):
            if _class(hwnd) == "#32770":
                return hwnd
        return None

    @staticmethod
    def _name_edit(dialog):
        for child in child_windows(dialog):
            if _class(child) != "Edit" or not user32.IsWindowVisible(child):
                continue
            # ファイル選択は名前のコンボボックスの中、フォルダ選択はダイアログ直下の1152番。
            parent = user32.GetParent(child)
            if (parent and _class(parent) == "ComboBox") or user32.GetDlgCtrlID(child) == 1152:
                return child
        return None

    def quit_from_tray_menu(self, timeout: float = 60) -> int:
        menu = self.menu_page()
        try:
            menu.eval("window.pywebview.api.activate('quit')")
        except (RuntimeError, OSError, websocket.WebSocketException):
            pass  # 終了でページが閉じ、応答が返らないことがある
        finally:
            try:
                menu.close()
            except Exception:
                pass
        if self.page is not None:
            try:
                self.page.close()
            except Exception:
                pass
            self.page = None
        return self.process.wait(timeout=timeout)

    def open_saved_folder_from_tray_menu(self, folder: Path, timeout: float = 20) -> None:
        """トレイのメニューの「保存先を開く」を実行し、その場所のエクスプローラが開いたら閉じる。"""
        before = set(explorer_windows())
        menu = self.menu_page()
        try:
            menu.eval("window.pywebview.api.activate('folder')")
        finally:
            menu.close()

        def opened():
            return [hwnd for hwnd in explorer_windows() if hwnd not in before and _text(hwnd) in (folder.name, str(folder))]

        windows = wait_for(opened, timeout, f"保存先のエクスプローラ: {folder}")
        for hwnd in windows:
            user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE

    def worker_pids(self) -> list[int]:
        """この隔離コピーから起動された常駐ワーカーのプロセス。"""
        script = (
            "Get-CimInstance Win32_Process -Filter \"Name='LSS.exe'\" | Where-Object {"
            " $_.ExecutablePath -eq '" + str(self.executable).replace("'", "''") + "'"
            " -and $_.CommandLine -like '*--worker*' } | ForEach-Object { $_.ProcessId }")
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
        return [int(line) for line in output.split() if line.strip().isdigit()]

    def kill(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=15)
