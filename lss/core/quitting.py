"""利用者が処理中に LSS を終了するときの、同じprocess内の取り決め（凍結Design 40a）。

確認カードは「処理中の 1 件は中断され、保存されません。完成した結果はすべて保存されています。」と言う。

- 終了の要求（`request`）。画面のprocessで動くBGM候補の工程は区切りごとに `checkpoint` で確かめ、
  利用者のフォルダへ公開する前で止める。GPUの推論そのものは途中で切らない（常駐workerが仕上げて落ちる）。
- 利用者のフォルダへ書いている間（`writing_user_folder`、公開を最後まで運ぶ `hold_publication`）。終了処理は
  それが終わるまで待ち、終わったら封をする（`seal_if_idle`）。封をした後は、利用者のフォルダへの書き込みを
  新しく始めない（入ろうとすると `JobAbandonedError`）。終了処理が「書いていない」と判断した直後に書き始める
  ことはない。推論の結果待ち・内部の作業領域への書き込みは待たない。そこで切れても、次の処理が作りかけを
  片付けてやり直す（`application._discard_interrupted_attempt`）。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

_REQUESTED = threading.Event()
_lock = threading.Lock()
_writes = 0
_sealed = False
_held = threading.local()


class JobAbandonedError(RuntimeError):
    """利用者が処理中に LSS を終了した。公開の前なら、その1件は保存しない。"""


def request() -> None:
    _REQUESTED.set()


def requested() -> bool:
    return _REQUESTED.is_set()


def checkpoint() -> None:
    if _REQUESTED.is_set():
        raise JobAbandonedError("LSS の終了により、処理中の1件を保存せずに中断しました")


def _enter() -> None:
    global _writes
    with _lock:
        if _sealed:
            raise JobAbandonedError("LSS を終了しています。利用者のフォルダへは書き始めません")
        _writes += 1


def _leave() -> None:
    global _writes
    with _lock:
        _writes -= 1


@contextmanager
def writing_user_folder():
    """この間は利用者のフォルダ（音源の隣の LSS）へ書いている。終了処理はここを待つ。"""
    _enter()
    try:
        yield
    finally:
        _leave()


def hold_publication() -> None:
    """公開を最後まで運ぶと決めた（作業領域への書き出しを終え、終了の要求も無かった）。

    この後の完了記録とCurrentBestの切替が終わる（`release_publication`）まで、終了処理を待たせる。
    同じスレッドで2回呼んでも1回分。
    """
    if getattr(_held, "on", False):
        return
    _enter()
    _held.on = True


def release_publication() -> None:
    if getattr(_held, "on", False):
        _held.on = False
        _leave()


def writing() -> bool:
    with _lock:
        return _writes > 0


def seal_if_idle() -> bool:
    """利用者のフォルダへ書いていなければ封をして True。以後そこへの書き込みを始めさせない。"""
    global _sealed
    with _lock:
        if _writes:
            return False
        _sealed = True
        return True
