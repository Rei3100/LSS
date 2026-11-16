"""トレイ（アイコンとメニュー）。アイコンは Claude Design「LSS System Assets」§C。

- `tray_state` / `menu_model` は純粋な翻訳。画面の状態（ページが持つ phase など）と
  バックエンドの事実から、トレイの4状態とメニューの中身を決める。
- メニューの見た目と語彙はアプリ内の右クリックメニューに揃える（共通の文法を持つため）。
- `Tray` は WinForms の NotifyIcon と、メニューを出す小さな WebView2 窓を持つ。

メニューに出さないもの: モデル・RECIPE・プロバイダ・指標・エラーコード。
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from lss.ui import system_assets as sa

logger = logging.getLogger("lss.ui.web")


def tray_state(ui: dict) -> sa.TrayState:
    """ページの状態からトレイの4状態へ。EMPTY と STOPPED は READY の形。"""
    phase = ui.get("phase")
    if phase == "processing" and ui.get("fault"):
        return sa.TrayState.HALTED
    if phase == "processing":
        return sa.TrayState.PROCESSING
    if phase == "complete":
        return sa.TrayState.COMPLETE
    return sa.TrayState.READY


def tray_tip(ui: dict, snapshot: dict) -> str:
    """トレイのツールチップ（凍結Design 40a）。状態を1行で、件数は音源で数える。"""
    phase = ui.get("phase")
    if phase == "processing":
        if ui.get("fault"):
            return "操作が必要です"
        if ui.get("stopReq"):
            return "停止待ち"
        queue = snapshot.get("queue") or []
        st = snapshot.get("st") or {}
        processed = sum(1 for item in queue if st.get(item, "queued") != "queued")
        return f"処理中（{processed} / {len(queue)}）"
    if phase == "complete":
        return "完了"
    return "待機中"


def completion_notice(summary: list | None) -> tuple[str, str]:
    """トレイにいる間に完了したときの OS 通知（凍結Design 40a）。

    本文は完了画面が数えたものだけ（`伴奏 6 件` `BGM 9 曲` `要確認 3 件`）。数を作らない。
    """
    parts = [str(part) for part in (summary or []) if isinstance(part, str) and part]
    lines = ["、".join(parts)] if parts else []
    lines.append("クリックで LSS を開く")
    return "SESSION COMPLETE", "\n".join(lines)


def _source_name(snapshot: dict, item_id: str | None) -> str:
    song = (snapshot.get("songs") or {}).get(item_id or "")
    return song["name"] if song and song.get("name") else ""


# 行の語彙はアプリ内の右クリックメニュー・主操作・HALTカードと同じ（EN と JP の対）。
_OPEN = {"en": "OPEN", "jp": "LSS を表示", "action": "open"}
_QUIT = {"en": "QUIT", "jp": "LSS を終了", "action": "quit"}


def menu_model(ui: dict, snapshot: dict) -> dict:
    """メニューの中身。アプリ内の右クリックメニューと同じ「見出し・行・フッター」の形。

    見出しは、対象の音源があればその名前（行を右クリックしたときと同じ）、なければ LSS。
    フッターは上部ステータスの日本語側と同じ短句。当てはまらない操作は行ごと出さない
    （アプリ内のメニューと同じく、無効な行を並べない）。
    """
    state = tray_state(ui)
    queue = snapshot.get("queue") or []
    st = snapshot.get("st") or {}
    pending = sum(1 for item in queue if st.get(item) == "queued")
    if state is sa.TrayState.PROCESSING:
        processed = sum(1 for item in queue if st.get(item) != "queued")
        stopping = bool(ui.get("stopReq")) and not ui.get("allDone")
        items = [_OPEN] if stopping or ui.get("allDone") else [
            _OPEN, {"en": "STOP", "jp": "停止", "action": "stop"}]
        return {"state": state.value,
                "head": _source_name(snapshot, snapshot.get("running")) or "LSS",
                "items": items,
                "foot": "停止待ち" if stopping else f"{processed} / {len(queue)} 完了"}
    if state is sa.TrayState.HALTED:
        fault = snapshot.get("fault") or {}
        return {"state": state.value,
                "head": _source_name(snapshot, fault.get("item")) or "LSS",
                "items": [{"en": "RETRY", "jp": "再試行", "action": "retry"},
                          {"en": "SKIP", "jp": "この音源を飛ばす", "action": "skip"},
                          _OPEN, _QUIT],
                "foot": "中断"}
    if state is sa.TrayState.COMPLETE:
        saved = sum(1 for item in queue if st.get(item) == "done")
        items = [_OPEN, _QUIT]
        if saved_folders(snapshot):
            items = [{"en": "OPEN FOLDER", "jp": "保存先を開く", "action": "folder"}] + items
        return {"state": state.value, "head": "LSS", "items": items, "foot": f"{saved} 件 完了"}
    return {"state": state.value, "head": "LSS", "items": [_OPEN, _QUIT],
            "foot": f"未処理 {pending} 件" if queue and pending else ""}


def saved_folders(snapshot: dict) -> list[str]:
    """このセッションの結果がある先（各音源の隣の LSS フォルダ）。重複なし・並び順どおり。

    保存済みを再利用した行（SKIPPED）も結果はそこにある。検証済みの空出力は結果を持たない。
    """
    folders: list[str] = []
    for item in snapshot.get("queue") or []:
        state = (snapshot.get("st") or {}).get(item)
        song = (snapshot.get("songs") or {}).get(item)
        if state not in {"done", "skipped", "failed"} or not song or (
                state == "skipped" and song.get("zero")):
            continue
        # 欲しい結果ごとの到達がある音源は、伴奏を実際に保存した（または再利用した）ときだけ。
        # 保留したBGM候補は公開していないので、LSSフォルダを指さない。
        if item in (snapshot.get("tst") or {}) and not song.get("out"):
            continue
        if item not in (snapshot.get("tst") or {}) and state == "failed":
            continue
        folder = str(Path(song["dir"]) / "LSS")
        if folder not in folders:
            folders.append(folder)
    return folders


def _fit_tip(text: str) -> str:
    # .NET Framework の NotifyIcon.Text は 63 文字まで（超えると例外）。
    return text if len(text) <= 63 else text[:62] + "…"


def light_taskbar() -> bool:
    """タスクバーが明るいテーマか（Windows の SystemUsesLightTheme）。"""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "SystemUsesLightTheme")[0] == 1
    except OSError:
        return False


class Tray:
    """NotifyIcon の持ち主。アイコンの状態と動き、メニュー窓の出し入れ。

    WinForms のオブジェクトには UI スレッドでだけ触る（`invoke`）。
    """

    def __init__(self, invoke, icon_dir: Path, *, on_open, on_menu) -> None:
        self._invoke = invoke
        self._icon_dir = icon_dir
        self._on_open = on_open
        self._on_menu = on_menu
        self._notify = None
        self._icons: dict[tuple, object] = {}
        self._state = sa.TrayState.READY
        self._light = False
        self._generation = 0
        self._lock = threading.Lock()
        self._tip = "待機中"

    @property
    def installed(self) -> bool:
        return self._notify is not None

    def install(self) -> bool:
        def create() -> None:
            import clr

            clr.AddReference("System.Windows.Forms")
            import System.Windows.Forms as WinForms
            from Microsoft.Win32 import SystemEvents

            notify = WinForms.NotifyIcon()
            notify.Text = _fit_tip(self._tip)

            def clicked(_sender, args) -> None:
                if args.Button == WinForms.MouseButtons.Left:
                    threading.Thread(target=self._on_open, daemon=True).start()
                elif args.Button == WinForms.MouseButtons.Right:
                    position = WinForms.Cursor.Position
                    area = WinForms.Screen.FromPoint(position).WorkingArea
                    anchor = (position.X, position.Y, area.Left, area.Top, area.Right, area.Bottom)
                    threading.Thread(target=self._on_menu, args=(anchor,), daemon=True).start()

            notify.MouseUp += clicked
            # 完了通知のクリックでも全画面に戻る。
            notify.BalloonTipClicked += lambda _s, _e: threading.Thread(
                target=self._on_open, daemon=True).start()
            self._notify = notify
            self._light = light_taskbar()
            self._apply(self._state, None)
            notify.Visible = True
            SystemEvents.UserPreferenceChanged += lambda _s, _e: self._theme_changed()

        self._invoke(create)
        return self.installed

    def dispose(self) -> None:
        self._generation += 1

        def remove() -> None:
            notify, self._notify = self._notify, None
            if notify is not None:
                notify.Visible = False
                notify.Dispose()

        self._invoke(remove)

    def set_tip(self, text: str) -> None:
        """ツールチップ（OS が描く）。同じ文言なら触らない。"""
        if text == self._tip:
            return
        self._tip = text

        def apply() -> None:
            if self._notify is not None:
                self._notify.Text = _fit_tip(text)

        self._invoke(apply)

    def notify(self, title: str, text: str) -> None:
        """OS の通知（トレイの吹き出し。Windows 10/11 ではトースト）を1回出す。"""
        def show() -> None:
            import System.Windows.Forms as WinForms

            if self._notify is not None:
                # ToolTipIcon.None は Python の予約語と同名なので getattr で取る。
                self._notify.ShowBalloonTip(10000, title, text, getattr(WinForms.ToolTipIcon, "None"))

        self._invoke(show)

    def _theme_changed(self) -> None:
        light = light_taskbar()
        if light != self._light:
            self._light = light
            self.set_state(self._state, force=True)

    def _icon(self, state: sa.TrayState, frame: int):
        key = (state, self._light, frame)
        icon = self._icons.get(key)
        if icon is None:
            import System.Windows.Forms as WinForms
            from System.Drawing import Icon

            path = self._icon_dir / sa.tray_ico_name(state, self._light, frame)
            # Windows の小アイコン寸法（100/125/150% で 16/20/24）に描き分けた1枚を選ぶ。
            icon = Icon(str(path), WinForms.SystemInformation.SmallIconSize)
            self._icons[key] = icon
        return icon

    def _apply(self, state: sa.TrayState, frame: int | None) -> None:
        if self._notify is None:
            return
        if frame is None:
            frame = sa.COMPLETE_FINAL_FRAME if state is sa.TrayState.COMPLETE else 0
        self._notify.Icon = self._icon(state, frame)
        self.applied = (state.value, "light" if self._light else "dark", frame)

    def set_state(self, state: sa.TrayState, *, force: bool = False) -> None:
        """アイコンの切り替えは即時（クロスフェードしない）。動きは PROCESSING と COMPLETE だけ。"""
        with self._lock:
            if state is self._state and not force:
                return
            self._state = state
            self._generation += 1
            generation = self._generation
        if state is sa.TrayState.COMPLETE and not force:
            self._invoke(lambda: self._apply(state, 0))
            threading.Thread(target=self._open_once, args=(generation,), daemon=True).start()
            return
        self._invoke(lambda: self._apply(state, None))
        if state is sa.TrayState.PROCESSING:
            threading.Thread(target=self._pulse, args=(generation,), daemon=True).start()

    def _pulse(self, generation: int) -> None:
        """横ずれ 4 px と 3 px を 1.2 s ごとに2コマで交互。補間しない。"""
        frame = 0
        while True:
            time.sleep(sa.PROCESSING_STEP_SECONDS)
            if generation != self._generation:
                return
            frame = 1 - frame
            self._invoke(lambda f=frame: self._apply(sa.TrayState.PROCESSING, f))

    def _open_once(self, generation: int) -> None:
        for frame in range(1, sa.COMPLETE_FINAL_FRAME + 1):
            time.sleep(sa.COMPLETE_STEP_SECONDS)
            if generation != self._generation:
                return
            self._invoke(lambda f=frame: self._apply(sa.TrayState.COMPLETE, f))
