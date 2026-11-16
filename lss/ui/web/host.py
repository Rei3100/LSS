"""凍結Designのページを、WebView2の全画面ウィンドウとしてLSSの画面にする。

ここが持つのはOS所有の境界だけ。

- 全画面・フレームレスのウィンドウ（ウィンドウモードは製品として存在しない）
- 参照構図（1440x900）を実画面へ写すページ倍率
- トレイ（最小化＝トレイへ格納、トレイのツールチップ、トレイにいる間の完了通知）
- 閉じる＝終了（凍結Design 40a。処理中の確認はページの確認カード）
- ファイル／フォルダ選択ダイアログ、ドラッグ＆ドロップの実パス、クリップボード、フォルダを開く
- バックエンドの事実をページへ届けること（`window.__lssPush`）
- 完了後の動作の実行

画面の見た目・動き・状態遷移はページ（Claude Designのコンポーネント）が持つ。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from lss.ui.web import page
from lss.ui.web import tray as tray_ui
from lss.ui.web.session import WebSession

logger = logging.getLogger("lss.ui.web")

TITLE = "LSS"
# 凍結Designの --bg（oklch(0.13 0.004 264)）。ページが描かれる前の地も同じ色にする。
BACKGROUND = "#0b0c0e"
# 1秒ごとの経過時間より細かく、進捗の連続イベントを間引く間隔。
PUSH_INTERVAL_SECONDS = 1 / 30
# 処理中に終了したとき、BGM候補の工程が次の区切りで止まるのを待つ短い猶予。
QUIT_GRACE_SECONDS = 2.0
# 利用者のフォルダへ書いている途中の片付けを待つ上限（公開cue1曲の書き出しは数秒）。
QUIT_WRITE_LIMIT_SECONDS = 60.0


def _audio_filter() -> tuple[str, ...]:
    from lss.core.session import AUDIO_EXTENSIONS

    patterns = ";".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))
    return (f"音声ファイル ({patterns})", "すべてのファイル (*.*)")


def fit_factor(width_px: int, height_px: int, dpi_scale: float) -> float | None:
    """参照構図（1440x900）を縦横比を保って収めるページ倍率。

    WebView2のCSS画素は「物理画素 / (DPI倍率 × ページ倍率)」なので、DPI倍率で割った
    論理サイズに対して構図を収める。倍率を焼き込んだ新しい固定寸法は作らない。
    """
    if width_px <= 0 or height_px <= 0 or dpi_scale <= 0:
        return None
    width, height = width_px / dpi_scale, height_px / dpi_scale
    return min(width / page.FRAME_WIDTH, height / page.FRAME_HEIGHT)


def folder_to_open(requested: str, snapshot: dict) -> Path | None:
    """ページが開きたいフォルダ。一覧の音源の場所か、その隣の LSS フォルダだけを開く。

    ページの言うパスをそのまま OS に渡さない（実行ファイルや別の場所は開かない）。
    LSS フォルダがまだ無ければ（保存していない）、音源の場所を開く。
    """
    if not isinstance(requested, str) or not requested:
        return None
    wanted = os.path.normcase(os.path.normpath(requested))
    for song in (snapshot.get("songs") or {}).values():
        source_dir = Path(str(song.get("dir") or ""))
        if not str(source_dir):
            continue
        for candidate in (source_dir, source_dir / "LSS"):
            if os.path.normcase(os.path.normpath(str(candidate))) != wanted:
                continue
            for folder in (candidate, source_dir):
                if folder.is_dir():
                    return folder
            return None
    return None


class Api:
    """ページから呼ばれる操作。pywebviewが公開するのは下線で始まらないメソッドだけ。"""

    def __init__(self, host: "Host") -> None:
        self._host = host

    def ready(self) -> None:
        self._host.page_ready()

    def snapshot(self) -> dict:
        return self._host.session.snapshot()

    def choose(self, kind: str) -> list[str]:
        return self._host.choose(kind)

    def remove(self, item_id: str) -> bool:
        return self._host.session.remove(item_id)

    def retryRow(self, item_id: str) -> bool:  # noqa: N802  ページ側の名前
        return self._host.session.retry_row(item_id)

    def start(self, wanted: dict | None = None, default: list | None = None, sounds: str | None = None) -> dict:
        """START / RESUME。`wanted` は音源ごとの欲しい結果、`default` はバッチの既定、
        `sounds` は欲しい音の入力欄の文字列。

        戻り値 ``{"started": bool, "work": 処理する音源の数}``。work が 0 なら保存済み
        だけで完了しており、ページはPROCESSINGを経ずにCOMPLETEへ進む。
        """
        self._host.completion.reset()
        return self._host.session.start(wanted, default, sounds)

    def stop(self) -> None:
        self._host.session.request_stop()

    def haltRetry(self) -> None:  # noqa: N802
        self._host.session.halt_retry()

    def haltSkip(self) -> None:  # noqa: N802
        self._host.session.halt_skip()

    def backToReady(self) -> dict:  # noqa: N802
        self._host.completion.reset()
        self._host.session.back_to_ready()
        return self._host.session.snapshot()

    def sessionComplete(self, choice: str, summary: list | None = None) -> str:  # noqa: N802
        """COMPLETE に着いた。`summary` は完了画面の件数の短句（トレイにいるときの通知に使う）。"""
        return self._host.session_complete(choice, summary)

    def wave(self, item_id: str, buckets: int | None = None) -> list[int]:
        from lss.ui import waveform

        path = self._host.session.source_path(item_id)
        if not path:
            return []
        count = buckets if isinstance(buckets, int) and 8 <= buckets <= 240 else waveform.BUCKETS
        return list(waveform.peaks(path, count))

    def copyText(self, text: str) -> bool:  # noqa: N802
        return self._host.copy_text(text)

    def openFolder(self, path: str) -> bool:  # noqa: N802
        return self._host.open_folder(path)

    def uiState(self, state: dict) -> None:  # noqa: N802
        self._host.set_ui_state(state)

    def minimize(self) -> None:
        self._host.to_tray()

    def quit(self) -> None:
        # 凍結Design 40a: 閉じる＝終了。処理中の確認はページ（Design の確認カード）が済ませている。
        self._host.quit()

    def close(self) -> None:
        self._host.quit()


class MenuApi:
    """トレイのメニュー窓から呼ばれる操作。"""

    def __init__(self, host: "Host") -> None:
        self._host = host

    def activate(self, action: str) -> None:
        self._host.menu_action(action)

    def dismiss(self) -> None:
        self._host.hide_menu()


class Host:
    def __init__(self, session: WebSession | None = None) -> None:
        from lss.ui.completion import Completion

        self.session = session or WebSession()
        self.session.set_listener(self._mark_dirty)
        self.completion = Completion(quit_app=self.quit)
        self.window = None
        self._loaded = threading.Event()
        self._dirty = threading.Event()
        self._quitting = False
        self._tray: tray_ui.Tray | None = None
        self._menu = None
        self._menu_ready = False
        self._ui: dict = {"phase": "ready"}
        self._drop_ready = False
        # トレイにしまっている間（完了の OS 通知はこのときだけ出す）。
        self._hidden = False
        # 処理中の終了（確認カードの「終了する」）と、常駐workerへ終了要求が届いたか。
        self._abandon = False
        self._worker_reached = True

    # --- 起動 -------------------------------------------------------------

    def build_page(self) -> Path:
        from lss.core.session import state_dir

        # 書き出すページ一式を上書きするだけで、フォルダは消さない（LSSが書く名前以外の物には触れない）。
        return page.build(state_dir() / "web", initial=self.session.snapshot())

    def run(self, **window_options) -> int:
        """`window_options` は検証用の上書き（全画面をやめて画面外に置く等）。"""
        import webview

        entry = self.build_page()
        options = dict(
            fullscreen=True, frameless=True, easy_drag=False, shadow=False,
            background_color=BACKGROUND, text_select=False, zoomable=False,
        )
        options.update(window_options)
        self.window = webview.create_window(TITLE, url=str(entry), js_api=Api(self), **options)
        # トレイのメニュー（アプリ内の右クリックメニューと同じ見た目）。隠したまま用意し、右クリックで出す。
        self._menu = webview.create_window(
            "LSS menu", url=str(entry.with_name("tray_menu.html")), js_api=MenuApi(self),
            hidden=True, frameless=True, on_top=True, transparent=True, easy_drag=False,
            shadow=False, text_select=False, zoomable=False, width=260, height=300)
        self._menu.events.loaded += self._on_menu_loaded
        self.window.events.shown += self._on_shown
        self.window.events.loaded += self._on_loaded
        self.window.events.closing += self._on_closing
        self.window.events.resized += lambda *_args: self._fit()
        debug = os.environ.get("LSS_WEB_DEBUG") == "1"
        webview.start(self._started, gui="edgechromium", debug=debug, private_mode=True)
        self._finish_quit()
        return 0

    def _finish_quit(self) -> None:
        """窓を閉じた後、processを終える前の片付け（窓とトレイは既に無い）。

        - 処理中に終了したとき、利用者のフォルダへ書いている途中なら、その片付けか公開が終わるまで待つ
          （上限あり）。書いていなければ封をして（以後そこへ書き始めさせない）終える。推論の結果待ちや
          内部の作業領域への書き込みは待たない。そこで切れた作りかけは次の処理が片付けてやり直す。
        - 起動の途中だったworkerへ、終了要求を届け直す（画面の無い常駐workerを残さない）。
        """
        if not self._quitting:
            return
        from lss.core import quitting

        started = time.monotonic()
        while self._abandon and self.session.running():
            spent = time.monotonic() - started
            if spent >= QUIT_WRITE_LIMIT_SECONDS:
                logger.warning("終了時の公開の片付けが%.0f秒で終わらなかった", spent)
                break
            if spent >= QUIT_GRACE_SECONDS and quitting.seal_if_idle():
                break
            time.sleep(0.05)
        if not self._worker_reached:
            try:
                from lss.core import client

                client.finish_worker_shutdown(abandon=self._abandon)
            except Exception:
                logger.exception("起動途中のworkerへ終了を届けられなかった")

    def _started(self) -> None:
        # ここはGUIループの開始直後に別スレッドで呼ばれ、まだフォームが無いことがある。
        # フォームに依存する準備（トレイ）は shown で行う。
        threading.Thread(target=self._pump, name="lss-web-push", daemon=True).start()
        threading.Thread(target=self._recover_current_best_scene_names,
                         name="lss-scene-name-recovery-launch", daemon=True).start()

    def _recover_current_best_scene_names(self) -> None:
        """Ask one detached coordinator to recover published names without blocking GUI startup."""
        try:
            from lss.core.session import state_dir
            from lss.bgm import scene_job

            result = scene_job.start_recovery_in_background(state_dir())
            if result.get("state") not in ("current", "already_running"):
                logger.info("起動時の場面名復旧: %s", result)
        except Exception:
            logger.exception("起動時の場面名復旧を開始できませんでした")

    def _on_shown(self) -> None:
        self._apply_window_icon()
        if self._tray is None:
            self._install_tray()

    def _on_loaded(self) -> None:
        self._fit()
        self._install_process_watch()
        self._install_drop()
        self._loaded.set()
        self._mark_dirty()

    def _install_process_watch(self) -> None:
        """WebView2の子プロセスの異常終了を記録する。窓なしビルドでは画面が黙って消えるため。"""
        if getattr(self, "_process_watch", False):
            return
        native = getattr(self.window, "native", None)

        def subscribe() -> None:
            core = native.webview.CoreWebView2 if native is not None and native.webview is not None else None
            if core is None:
                return

            def failed(_sender, args) -> None:
                logger.error(
                    "WebView2のプロセスが終了: kind=%s reason=%s exit=%s process=%s",
                    args.ProcessFailedKind, getattr(args, "Reason", ""), getattr(args, "ExitCode", ""),
                    getattr(args, "ProcessDescription", ""))

            core.ProcessFailed += failed
            self._process_watch = True

        self._invoke(subscribe)

    def page_ready(self) -> None:
        self._loaded.set()
        self._fit()
        self._mark_dirty()

    # --- 事実をページへ ---------------------------------------------------

    def _mark_dirty(self) -> None:
        self._dirty.set()

    def _pump(self) -> None:
        last = 0.0
        while not self._quitting:
            fired = self._dirty.wait(timeout=1.0)
            if self._quitting:
                return
            if not self._loaded.is_set():
                continue
            if not fired and not self.session.elapsed_running():
                continue
            wait = PUSH_INTERVAL_SECONDS - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            self._dirty.clear()
            last = time.monotonic()
            try:
                facts = self.session.snapshot_json()
                self.window.run_js("window.__lssPush&&window.__lssPush(" + facts + ")")
            except Exception:
                logger.exception("ページへ事実を届けられなかった")
                continue
            self._update_tray_tip(json.loads(facts))

    # --- 構図の写像 -------------------------------------------------------

    def _fit(self) -> None:
        """参照構図を縦横比を保って全画面へ写す。WebView2のページ倍率で拡大する。

        CSSのzoomやtransformで拡大すると、Designがポインタ座標から置くメニューと
        ツールチップが拡大率の分だけずれる。ページ倍率はブラウザの座標系そのものを
        変えるので、Designのコードは1440x900の世界のまま正しく動く。
        """
        native = getattr(self.window, "native", None)
        if native is None:
            return
        box: dict = {}

        def apply() -> None:
            control = native.webview
            if control is None or control.CoreWebView2 is None:
                return
            box["scale"] = float(native._scale)
            current = float(control.ZoomFactor) or 1.0
            size = native.ClientSize
            factor = fit_factor(size.Width, size.Height, box["scale"])
            if factor is not None and abs(factor - current) > 0.0005:
                control.ZoomFactor = factor

        self._invoke(apply)
        if "scale" in box and self._loaded.is_set():
            # カーソルは 100/125/150% ごとに描き分けた画像を 1 物理画素で出す（UIスレッドの外で呼ぶ）。
            try:
                self.window.run_js(
                    f"window.__lssSystemScale && window.__lssSystemScale({box['scale']})")
            except Exception:
                logger.exception("カーソルの倍率を渡せなかった")

    # --- OS所有面 ---------------------------------------------------------

    def _invoke(self, function) -> None:
        native = getattr(self.window, "native", None)
        if native is None:
            return
        from System import Func, Type

        box: dict = {}

        def run():
            try:
                function()
            except Exception as exc:  # pragma: no cover - UIスレッド上の失敗を記録だけする
                box["error"] = exc
            return None

        try:
            if native.InvokeRequired:
                native.Invoke(Func[Type](run))
            else:
                run()
        except Exception:
            logger.exception("UIスレッドでの実行に失敗")
        if "error" in box:
            logger.error("UIスレッドでの実行に失敗: %s", box["error"])

    def choose(self, kind: str) -> list[str]:
        """ファイル選択はOS所有面。見た目を偽造しない。"""
        import webview

        window = self.window
        if window is None:
            return []
        if kind == "folder":
            names = window.create_file_dialog(webview.FileDialog.FOLDER)
        else:
            names = window.create_file_dialog(
                webview.FileDialog.OPEN, allow_multiple=True, file_types=_audio_filter())
        paths = [Path(name) for name in (names or []) if name]
        return self.session.add(paths) if paths else []

    def _install_drop(self) -> None:
        if self._drop_ready or self.window is None:
            return
        from webview.dom import DOMEventHandler

        document = self.window.dom.document
        document.events.dragover += DOMEventHandler(lambda _event: None, True, False)
        document.events.drop += DOMEventHandler(self._on_drop, True, True)
        self._drop_ready = True

    def _on_drop(self, event: dict) -> None:
        files = (event.get("dataTransfer") or {}).get("files") or []
        paths = [Path(item["pywebviewFullPath"]) for item in files
                 if isinstance(item, dict) and item.get("pywebviewFullPath")]
        if not paths:
            return
        try:
            phase = self.window.evaluate_js(
                "window.__lssComponent ? window.__lssComponent.state.phase : ''")
        except Exception:
            phase = ""
        # COMPLETEでは追加しない（凍結Design: COMPLETE以外で追加できる）。
        if phase == "complete":
            return
        self.session.add(paths)

    def copy_text(self, text: str) -> bool:
        if not text:
            return False
        done = {}

        def put() -> None:
            import System.Windows.Forms as WinForms

            WinForms.Clipboard.SetText(text)
            done["ok"] = True

        self._invoke(put)
        return bool(done.get("ok"))

    def _apply_window_icon(self) -> None:
        """タスクバーとウィンドウのアイコン（Design System Assets §B の描き分けた ICO）。"""
        from lss.paths import resource_root

        icon_path = resource_root() / "assets" / "app.ico"
        native = getattr(self.window, "native", None)
        if native is None or not icon_path.is_file():
            return

        def apply() -> None:
            from System.Drawing import Icon

            native.Icon = Icon(str(icon_path))

        self._invoke(apply)

    def _install_tray(self) -> None:
        from lss.paths import resource_root

        def subscribe() -> None:
            from Microsoft.Win32 import SystemEvents

            SystemEvents.SessionEnding += lambda _s, _e: self._session_ending()

        self._invoke(subscribe)
        tray = tray_ui.Tray(
            self._invoke, resource_root() / "assets" / "system",
            on_open=self.restore, on_menu=self.show_menu)
        if tray.install():
            self._tray = tray
            tray.set_state(tray_ui.tray_state(self._ui), force=True)
            self._update_tray_tip()
        else:
            logger.error("トレイを作れなかった。最小化はタスクバーへ退く")

    # --- 画面の状態とトレイ ------------------------------------------------

    def set_ui_state(self, state: dict) -> None:
        """ページが持つ画面の状態（phase など）。トレイのアイコンとメニューが従う。"""
        if not isinstance(state, dict):
            return
        self._ui = state
        if self._tray is not None:
            self._tray.set_state(tray_ui.tray_state(state))
            self._update_tray_tip()

    def _update_tray_tip(self, snapshot: dict | None = None) -> None:
        tray = self._tray
        if tray is None:
            return
        try:
            facts = snapshot if snapshot is not None else self.session.snapshot()
            tray.set_tip(tray_ui.tray_tip(self._ui, facts))
        except Exception:
            logger.exception("トレイのツールチップを更新できなかった")

    def _on_menu_loaded(self) -> None:
        """メニュー窓の準備。pywebview は透明な窓を読み込み時に自分で表示するので、すぐ隠す。"""
        native = getattr(self._menu, "native", None)
        main = getattr(self.window, "native", None)
        if native is None:
            return

        def prepare() -> None:
            from ctypes import windll

            if not self._menu_ready:
                # タスクバーに出さない。WinForms の ShowInTaskbar はウィンドウを作り直し、
                # 中の WebView2 が白紙になるので、拡張スタイルだけを付け替える。
                hwnd = native.Handle.ToInt32()
                style = windll.user32.GetWindowLongW(hwnd, -20)
                windll.user32.SetWindowLongW(hwnd, -20, (style | 0x80) & ~0x40000)
                native.TopMost = True
                native.Deactivate += lambda _s, _e: self._later(self.hide_menu)
                self._menu_ready = True
            native.Hide()
            if main is not None and main.Visible:
                main.Activate()

        self._invoke(prepare)

    def show_menu(self, anchor: tuple) -> None:
        """トレイの右クリック。ポインタの左上に、作業領域からはみ出さないように出す。"""
        import json

        menu = self._menu
        native = getattr(menu, "native", None)
        if menu is None or native is None or self._quitting:
            return
        model = tray_ui.menu_model(self._ui, self.session.snapshot())
        try:
            size = menu.evaluate_js(
                "window.__menuRender(" + json.dumps(model, ensure_ascii=False) + ")")
        except Exception:
            logger.exception("トレイのメニューを描けなかった")
            return
        if not isinstance(size, dict):
            return
        x, y, left, top, right, bottom = anchor

        def place() -> None:
            import math
            from ctypes import windll

            if not self._menu_ready:
                return
            # アプリ内のメニューと同じ物理寸法にする（ページと同じ倍率を掛ける）。
            main = getattr(self.window, "native", None)
            zoom = 1.0
            if main is not None and main.webview is not None and main.webview.CoreWebView2 is not None:
                zoom = float(main.webview.ZoomFactor) or 1.0
            if abs(float(native.webview.ZoomFactor) - zoom) > 0.0005:
                native.webview.ZoomFactor = zoom
            scale = float(native._scale) * zoom
            width = math.ceil(size["width"] * scale)
            height = math.ceil(size["height"] * scale)
            px = max(left, min(x - width, right - width))
            py = max(top, min(y - height, bottom - height))
            windll.user32.SetWindowPos(native.Handle.ToInt32(), None, px, py, width, height, 0x0040)
            native.Show()
            native.Activate()
            native.webview.Focus()

        self._invoke(place)

    def hide_menu(self) -> None:
        native = getattr(self._menu, "native", None)
        if native is None:
            return
        self._invoke(lambda: native.Hide())
        try:
            # 次に開いたとき、アプリ内メニューと同じ入場の動きをもう一度出す。
            self._menu.run_js("window.__menuHide && window.__menuHide()")
        except Exception:
            pass

    def menu_action(self, action: str) -> None:
        self.hide_menu()
        page_call = {
            "stop": "window.__lssComponent && window.__lssComponent.requestStop()",
            "retry": "window.__lssComponent && window.__lssComponent.clearFault(true)",
            "skip": "window.__lssComponent && window.__lssComponent.skipCurrent()",
        }
        if action == "open":
            self.restore()
        elif action == "quit":
            # 処理中は終了させない（メニューにも出さない）。
            if tray_ui.tray_state(self._ui) is not tray_ui.sa.TrayState.PROCESSING:
                self.quit()
        elif action == "folder":
            for folder in tray_ui.saved_folders(self.session.snapshot()):
                try:
                    os.startfile(folder)  # noqa: S606 - 保存先をエクスプローラで開く
                except OSError:
                    logger.warning("保存先を開けなかった: %s", folder)
        elif action in page_call and self.window is not None:
            # 状態遷移はページ（Designのコンポーネント）が持つ。同じ操作をページに通す。
            self.window.run_js(page_call[action])

    def _later(self, function) -> None:
        # pywebviewの操作はUIスレッドへ自分でInvokeするので、UIスレッドの外から呼ぶ。
        threading.Thread(target=function, daemon=True).start()

    def _session_ending(self) -> None:
        # Windowsの終了・サインアウトを妨げない。
        self._quitting = True
        self.session.shutdown()

    def to_tray(self) -> None:
        if self._quitting or self.window is None:
            return
        if self._tray is None:
            self.window.minimize()
            return
        self.hide_menu()
        self._hidden = True
        self.window.hide()

    def restore(self) -> None:
        """トレイから復帰したときも全画面で戻る。"""
        if self._quitting or self.window is None:
            return
        self._hidden = False
        self.window.show()
        self._fit()

    def _on_closing(self) -> bool | None:
        if self._quitting:
            return None
        # 凍結Design 40a: 閉じる＝終了。Alt+F4 など OS 側の閉じるも、タイトルバーの閉じると
        # 同じくページに通す（処理中だけ Design の確認カードが出る）。
        if self._loaded.is_set():
            self._later(lambda: self.window is not None and self.window.run_js(
                "window.__lssComponent ? window.__lssComponent.requestQuit() : null"))
        else:
            self._later(self.quit)
        return False

    def quit(self) -> None:
        """LSSを終了する。次のSOURCEは始めず、常駐ワーカーにも終了を求める。

        処理中の1件があるとき（確認カードで「終了する」を選んだとき）は、その1件を保存しない。
        ワーカーは公開の前で止め、作りかけの一時ファイルを片付けてから落ちる。
        """
        if self._quitting and self.window is None:
            return
        abandon = self.session.running()
        self._quitting = True
        self._abandon = abandon
        self._dirty.set()
        self.session.shutdown(abandon=abandon)
        try:
            from lss.core import client

            client.close()
            self._worker_reached = client.request_worker_shutdown(abandon=abandon) is not None
        except Exception:
            pass

        tray, self._tray = self._tray, None
        if tray is not None:
            tray.dispose()
        menu, self._menu = self._menu, None
        if menu is not None:
            menu.destroy()
        window, self.window = self.window, None
        if window is not None:
            window.destroy()

    def session_complete(self, choice: str, summary: list | None = None) -> str:
        from lss.ui.presentation import WhenDone

        try:
            when_done = WhenDone(choice)
        except ValueError:
            when_done = WhenDone.NONE
        # トレイにいる間に完了したら、OS 通知を1回だけ出す。全画面なら画面が既に言っている。
        tray = self._tray
        if when_done is WhenDone.NONE and self._hidden and tray is not None:
            tray.notify(*tray_ui.completion_notice(summary))
        return self.completion.perform(when_done).value

    def open_folder(self, path: str) -> bool:
        """フォルダを開く（エクスプローラー）。一覧の音源の場所と、その LSS フォルダだけ。"""
        target = folder_to_open(path, self.session.snapshot())
        if target is None:
            return False
        try:
            os.startfile(str(target))  # noqa: S606 - 一覧の音源のフォルダをエクスプローラで開く
        except OSError:
            logger.warning("フォルダを開けなかった: %s", target)
            return False
        return True


def main() -> int:
    from lss.core.session import state_dir

    # 窓なしビルドでは標準出力が無い。OS境界の失敗（トレイ・倍率・配線）を後から読めるよう残す。
    try:
        log_dir = state_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_dir / "ui.log"), level=logging.WARNING, encoding="utf-8",
            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    except OSError:
        pass
    return Host().run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

