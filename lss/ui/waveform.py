"""展開した行に出す波形。実測のピークだけを描く。

凍結Canvasの波形はサイン波の合成で、見た目のための飾りだった。ハンドオフは
それを移植することを禁じている（偽のデータ表示になる）。ここは実ファイルから
ピークを読む。読めなければ空を返し、描画側は要素ごと持たない。

読むのは各区間の先頭にある短い窓だけで、全長は復号しない。
"""
from __future__ import annotations

import threading
from pathlib import Path

BUCKETS = 44
WINDOW = 2048
FLOOR = 8  # 無音区間でも帯が消えないための下限（凍結Canvasと同じ）

_cache: dict[tuple[str, int, int], tuple[int, ...]] = {}


def peaks(path: str | Path, buckets: int = BUCKETS) -> tuple[int, ...]:
    """0〜100のピーク列。読めないときは空。

    同じファイルは更新時刻が変わるまで読み直さない。
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return ()
    key = (str(target), int(stat.st_mtime), buckets)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    values = _read_peaks(target, buckets)
    _cache[key] = values
    return values


# 読み取りはGUIスレッドの外で行い、結果はここに置く。Qtのオブジェクトへ
# 別スレッドから触らない（破棄と競ると、シグナル配送でプロセスごと落ちる）。
_pending: set[str] = set()
_ready: dict[str, tuple[int, ...]] = {}
_async_lock = threading.Lock()


def request(path: str | Path) -> None:
    """ピークの実測を別スレッドへ依頼する。呼び出し側は待たない。

    同じ音源は一度しか読まない。結果は`resolved()`で取り出す。
    """
    key = str(path)
    with _async_lock:
        if key in _pending or key in _ready:
            return
        _pending.add(key)

    def run() -> None:
        try:
            values = peaks(key)
        except Exception:
            values = ()
        with _async_lock:
            _pending.discard(key)
            _ready[key] = tuple(values)

    threading.Thread(target=run, name="lss-peaks", daemon=True).start()


def resolved(path: str | Path) -> tuple[int, ...] | None:
    """読み終わっていれば実測ピーク。まだならNone。"""
    with _async_lock:
        return _ready.get(str(path))


def waiting(path: str | Path) -> bool:
    with _async_lock:
        return str(path) in _pending


def _read_peaks(target: Path, buckets: int) -> tuple[int, ...]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        return ()
    try:
        with sf.SoundFile(str(target)) as handle:
            total = len(handle)
            if total <= 0:
                return ()
            step = max(1, total // buckets)
            raw = []
            for index in range(buckets):
                start = min(total - 1, index * step)
                handle.seek(start)
                block = handle.read(min(WINDOW, total - start), dtype="float32",
                                    always_2d=True)
                if block.size == 0:
                    raw.append(0.0)
                    continue
                raw.append(float(np.abs(block).max()))
    except (OSError, RuntimeError, ValueError):
        return ()

    top = max(raw) if raw else 0.0
    if top <= 0.0:
        return tuple(FLOOR for _ in raw)
    return tuple(max(FLOOR, int(round(value / top * 100))) for value in raw)
