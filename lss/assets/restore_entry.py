"""配備版だけの環境で、保存状態のバックアップから復元する入口。

配備版は開発用Pythonを持たず、画面の無い実行ファイルは標準出力も見えない。
`LSS.exe --restore-backup <zip>`、またはバックアップのzipを `LSS.exe` へドラッグすると、
Windows標準の確認ダイアログを出してから `backup.restore_backup` を呼び、結果も同じ形で示す。
復元そのものの安全性（起動中は拒否、今の記録の退避、失敗時の巻き戻し）は `backup` が持つ。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lss.assets import backup

TITLE = "LSS 保存状態の復元"
_MB_OK = 0x0
_MB_YESNO = 0x4
_MB_ICONINFORMATION = 0x40
_MB_ICONWARNING = 0x30
_MB_ICONERROR = 0x10
_IDYES = 6


def _message_box(text: str, flags: int) -> int:
    import ctypes

    return int(ctypes.windll.user32.MessageBoxW(None, text, TITLE, flags))


def ask(text: str) -> bool:
    return _message_box(text, _MB_YESNO | _MB_ICONWARNING) == _IDYES


def notify(text: str, *, error: bool = False) -> None:
    _message_box(text, _MB_OK | (_MB_ICONERROR if error else _MB_ICONINFORMATION))


def is_backup_archive(argument: str) -> bool:
    """ドラッグされた引数が、LSSが作った保存状態のバックアップらしいか。"""
    path = Path(argument)
    return path.suffix.casefold() == ".zip" and path.name.casefold().startswith("lss-state-")


def _worker_running() -> bool:
    from lss.core import client

    return client.is_worker_running()


def _stop_worker(wait_seconds: float = 120.0) -> bool:
    """常駐ワーカーへ正常終了を頼み、止まるまで待つ。実行中の1件は公開まで仕上がる。"""
    import time

    from lss.core import client

    client.request_worker_shutdown()
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if not client.is_worker_running():
            # pipeを閉じてからプロセスが消えるまでの間に、起動中と判定されないよう少し待つ。
            # 画面が開いたままなら消えないので、そこから先の判定は復元側に任せる。
            settle = time.monotonic() + 10.0
            while time.monotonic() < settle and backup.lss_running():
                time.sleep(0.5)
            return True
        time.sleep(0.5)
    return False


def run(path: Path, *, confirm: Callable[[str], bool] = ask,
        report: Callable[..., None] = notify, restore=None, verify=None,
        worker_running: Callable[[], bool] = _worker_running,
        stop_worker: Callable[[], bool] = _stop_worker) -> int:
    """確認してから復元する。復元した・しなかった・できなかったを終了コードで返す。

    画面を閉じても常駐ワーカーは残る。動いていれば、止めてよいかを確かめてから
    正常終了を頼む。画面が開いたままなら、復元そのものが起動中として拒否する。
    """
    restore = backup.restore_backup if restore is None else restore
    verify = backup.verify_backup if verify is None else verify
    try:
        manifest = verify(Path(path))
    except backup.BackupError as exc:
        report(f"このファイルからは復元できません。\n\n{exc}", error=True)
        return 2
    question = (f"{Path(path).name}\n作成: {manifest.get('created_at')}\n記録: {len(manifest['files'])}件\n\n"
                "今の保存状態を退避してから、このバックアップの状態へ置き換えます。\n"
                "分離結果の音声ファイルは変わりません。よろしいですか？")
    if not confirm(question):
        return 1
    if worker_running():
        if not confirm("LSSの常駐ワーカーが動いています。処理中の曲があれば仕上げてから止め、"
                       "そのあと復元します。よろしいですか？"):
            return 1
        if not stop_worker():
            report("常駐ワーカーが止まりませんでした。今の保存状態はそのままです。", error=True)
            return 2
    try:
        saved = restore(Path(path))
    except backup.BackupError as exc:
        report(f"復元できませんでした。今の保存状態はそのままです。\n\n{exc}", error=True)
        return 2
    report(f"復元しました。\n置き換える前の記録は次へ退避してあります。\n{saved}")
    return 0
