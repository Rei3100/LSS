"""音源の素性を読む。ヘッダだけを見て、復号はしない。

行に出すのは「読めた事実」だけ。読めなければその項目を持たない。推測で埋めない。
チャンネル数は常設情報ではないので、値そのものは持つが、2chを超えるときだけ
前面化するのは画面側（凍結Designのページ）の規則が決める。

libsndfileが直接読めないコンテナ（m4a / aac / wma 等）は、LSSが受け付ける入力で
ありながらヘッダを読めない。行が空になるのは「読めなかった」ではなく実装の穴なので、
同梱FFmpegに素性だけを聞く。復号はせず、GUIスレッドも止めない（`request`）。
"""
from __future__ import annotations

import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaFacts:
    """1つの音源から読めた素性。空文字は「読めなかった」を意味する。"""

    duration: str = ""  # m:ss（1時間超は h:mm:ss）
    media_format: str = ""  # WAV / FLAC など、コンテナ名
    sample_rate: str = ""  # kHz
    bit_depth: str = ""  # bit（可逆で意味を持つときだけ）
    megabytes: str = ""  # MB
    channels: int = 0


_cache: dict[tuple[str, int, int], MediaFacts] = {}

# subtypeから読み取れるビット深度。非可逆や可変長のものは持たない。
_BIT_DEPTH = {
    "PCM_S8": "8", "PCM_U8": "8", "PCM_16": "16", "PCM_24": "24", "PCM_32": "32",
    "FLOAT": "32", "DOUBLE": "64", "ALAC_16": "16", "ALAC_20": "20",
    "ALAC_24": "24", "ALAC_32": "32",
}


def clock(seconds: float) -> str:
    """経過も長さも1つの文法で書く。m:ss、1時間超で h:mm:ss。"""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, second = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{second:02d}"
    return f"{minutes}:{second:02d}"


def facts(path: str | Path) -> MediaFacts:
    """音源の素性。ヘッダしか読まないので一覧の描画中に呼べる。"""
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return MediaFacts()
    key = (str(target), int(stat.st_mtime), stat.st_size)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    value = _read(target, stat.st_size)
    _cache[key] = value
    return value


# FFmpegが報告するストリーム行から、行に出す事実だけを取り出す。
_STREAM = re.compile(
    r"Stream #\d+:\d+.*?: Audio: .*?, (?P<rate>\d+) Hz, (?P<layout>[^,]+)")
_DURATION = re.compile(r"Duration: (\d+):(\d\d):(\d\d)\.(\d+)")
_LAYOUT_CHANNELS = {
    "mono": 1, "stereo": 2, "2.1": 3, "3.0": 3, "quad": 4, "4.0": 4,
    "5.0": 5, "5.1": 6, "6.1": 7, "7.1": 8,
}
_PROBE_TIMEOUT_SECONDS = 20


def _probe_channels(layout: str) -> int:
    layout = layout.strip()
    known = _LAYOUT_CHANNELS.get(layout.split("(")[0].strip())
    if known:
        return known
    matched = re.match(r"(\d+) channels", layout)
    return int(matched.group(1)) if matched else 0


def _probe(target: Path) -> tuple[str, int, int]:
    """同梱FFmpegに素性だけ聞く（長さ・サンプルレート・チャンネル数）。

    復号も書き出しもしない。読めなければ空の事実を返し、値を作らない。
    """
    from lss.paths import resource_root

    ffmpeg = resource_root() / "bin" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        return ("", 0, 0)
    try:
        result = subprocess.run(
            [str(ffmpeg), "-nostdin", "-hide_banner", "-i", str(target)],
            capture_output=True, timeout=_PROBE_TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ("", 0, 0)
    report = result.stderr.decode("utf-8", errors="replace")
    stream = _STREAM.search(report)
    if stream is None:
        return ("", 0, 0)
    duration = ""
    spans = _DURATION.search(report)
    if spans is not None:
        hours, minutes, seconds, _fraction = spans.groups()
        duration = clock(int(hours) * 3600 + int(minutes) * 60 + int(seconds))
    return (duration, int(stream.group("rate")),
            _probe_channels(stream.group("layout")))


# 調査は「1音源につき1回」で、同時に走る数も絞る。行の更新は毎回起きるので、
# 依頼のたびにスレッドとFFmpegを起こすと、読めない音源が並ぶだけで資源が尽きる。
PROBE_WORKERS = 2
_probed: set[tuple] = set()
_probe_lock = threading.Lock()
_probe_queue: "queue.Queue[tuple]" = queue.Queue()
_probe_workers: list[threading.Thread] = []


def _probe_worker() -> None:
    while True:
        key, target, size, done = _probe_queue.get()
        try:
            base = _cache.get(key) or _read(target, size, probe=False)
            duration, rate, channels = _probe(target)
            if rate:
                _cache[key] = MediaFacts(
                    duration=duration or base.duration,
                    media_format=base.media_format,
                    sample_rate=f"{rate / 1000.0:g}",
                    bit_depth=base.bit_depth,
                    megabytes=base.megabytes,
                    channels=channels or base.channels,
                )
            else:
                # 調べても読めなかった事実も覚える。毎回調べ直さない。
                _cache[key] = base
        except Exception:
            # 素性が読めないだけで、行も処理も止めない。
            pass
        finally:
            _probe_queue.task_done()
        if done is not None:
            try:
                done()
            except Exception:
                # 表示側の通知失敗で調査結果を捨てない。
                pass


def _ensure_workers() -> None:
    with _probe_lock:
        while len(_probe_workers) < PROBE_WORKERS:
            worker = threading.Thread(
                target=_probe_worker, name="lss-media-probe", daemon=True)
            _probe_workers.append(worker)
            worker.start()


def request(path: str | Path, done=None) -> None:
    """ヘッダを読めない音源の素性を、GUIスレッドの外で確かめる。

    同じ音源は一度しか調べない（読めなかったことも覚える）。結果は`facts()`の
    キャッシュへ入り、`done`で通知する。呼び出し側は待たない。
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return
    key = (str(target), int(stat.st_mtime), stat.st_size)
    with _probe_lock:
        if key in _probed:
            return
        _probed.add(key)
    _ensure_workers()
    _probe_queue.put((key, target, stat.st_size, done))


def _read(target: Path, size: int, *, probe: bool = True) -> MediaFacts:
    megabytes = f"{size / 1_000_000:.1f}" if size else ""
    container = target.suffix.lstrip(".").upper()
    try:
        import soundfile as sf
    except ImportError:
        return MediaFacts(media_format=container, megabytes=megabytes)
    try:
        info = sf.info(str(target))
    except (OSError, RuntimeError, ValueError):
        return MediaFacts(media_format=container, megabytes=megabytes)

    duration = clock(info.frames / info.samplerate) if info.samplerate else ""
    rate = info.samplerate / 1000.0
    sample_rate = f"{rate:g}" if info.samplerate else ""
    return MediaFacts(
        duration=duration,
        media_format=container or (info.format or ""),
        sample_rate=sample_rate,
        bit_depth=_BIT_DEPTH.get(info.subtype or "", ""),
        megabytes=megabytes,
        channels=int(info.channels or 0),
    )
