"""外部サービスへの礼儀（LSSが自分で行う発見・取得の通信の調整）。

- 取得先（ホストの系統）ごとに一度に一つだけ通信する。
- 通信の間に揺らぎのある間隔を空ける。混雑の合図の後は間隔を広げ、成功が続けば少しずつ戻す。
- 混雑の合図（HTTP 429・Too Many Requests）を受けたら、その取得先への通信を休む。サーバの指示
  （`Retry-After`）が分かればそれより早く戻らない。休みは再起動後も守る（状態領域に残す）。
- 長い休みの間は待ち続けず、呼び出し側へ「後で」と返す（作業は中断ではなく保留。次の機会に続きから）。

固定の一律の遅延ではなく、取得先の反応で間隔を変える。時計・待ち・乱数は試験で差し替える。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from urllib.parse import urlsplit

SCHEMA = 1
STATE_FILE = "acquisition-courtesy.json"

RATE_LIMITED = "rate_limited"
RECENT_LIMIT = 200
TRANSIENT = "transient"


@dataclass(frozen=True)
class Policy:
    interval: tuple[float, float] = (3.0, 8.0)  # 通常の間隔（一様な揺らぎ）
    factor_limit: float = 8.0                   # 混雑の後に広げる間隔の倍率の上限
    cooldown_start: float = 600.0               # 混雑の合図の後の休み（初回）
    cooldown_limit: float = 6 * 3600.0          # 休みの上限
    transient_cooldown: float = 60.0            # 一時的な失敗（接続の切断など）の後の短い休み
    recover_after: int = 5                      # 成功がこれだけ続けば倍率を半分へ戻す
    max_wait: float = 120.0                     # これより長い休みは待たずに「後で」を返す


class Deferred(RuntimeError):
    """取得先が休みの間。`seconds` 後まで通信しない（作業は保留して後で続ける）。"""

    def __init__(self, host: str, seconds: float):
        super().__init__(f"{host} への通信を {seconds:.0f} 秒休みます")
        self.host = host
        self.seconds = float(seconds)


_YOUTUBE = re.compile(r"(^|\.)(youtube\.com|youtu\.be|googlevideo\.com|ytimg\.com)$")


def host_key(url: str) -> str:
    """同じ相手として扱う取得先の名前（YouTubeとYouTube Musicは一つ）。"""
    host = (urlsplit(str(url)).hostname or "").lower()
    if _YOUTUBE.search(host):
        return "youtube"
    return host or "unknown"


def retry_after_seconds(value, now: float | None = None) -> float | None:
    """`Retry-After` の値（秒数またはHTTPの日時）を秒へ。文中の `Retry-After: N` も読む。読めなければNone。"""
    if value is None:
        return None
    text = str(value).strip()
    found = re.search(r"retry-after\s*[:=]\s*([^\r\n;]+)", text, re.IGNORECASE)
    if found:
        text = found.group(1).strip()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return max(0.0, float(text))
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = time.time() if now is None else now
    return max(0.0, when.timestamp() - now)


class Courtesy:
    """取得先ごとの順番・間隔・休みを持つ。一つのプロセスで共有し、休みは状態領域で他の起動と共有する。"""

    def __init__(self, state_root: Path | None = None, policy: Policy = Policy(), *, clock=time.time,
                 sleep=time.sleep, jitter=None):
        self.state_root = Path(state_root) if state_root is not None else None
        self.policy = policy
        self.clock, self.sleep = clock, sleep
        self.jitter = jitter or (lambda low, high: random.uniform(low, high))
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._memory: dict[str, dict] = {}
        self.log: list[dict] = []

    # --- 状態 ------------------------------------------------------------------

    def _path(self) -> Path | None:
        return self.state_root / STATE_FILE if self.state_root is not None else None

    def _load(self) -> dict:
        path = self._path()
        if path is None:
            return self._memory
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return dict(self._memory)
        hosts = payload.get("hosts") if isinstance(payload, dict) and payload.get("schema") == SCHEMA else None
        return hosts if isinstance(hosts, dict) else dict(self._memory)

    def _save(self, hosts: dict) -> None:
        self._memory = hosts
        path = self._path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps({"schema": SCHEMA, "hosts": hosts}, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass

    def _host(self, hosts: dict, host: str) -> dict:
        return dict(hosts.get(host) or {"factor": 1.0, "streak": 0, "successes": 0,
                                        "cooldown_until": 0.0, "last_request": 0.0})

    def _lock(self, host: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(host, threading.Lock())

    def cooldown_remaining(self, url_or_host: str) -> float:
        host = url_or_host if "/" not in url_or_host else host_key(url_or_host)
        entry = self._load().get(host) or {}
        return max(0.0, float(entry.get("cooldown_until") or 0.0) - self.clock())

    # --- 通信の枠 ----------------------------------------------------------------

    @contextmanager
    def slot(self, url: str, *, purpose: str = "request", should_stop=lambda: False):
        """一つの通信（または一つの取得）の枠。間隔・休みを守ってから入る。長い休みなら `Deferred`。"""
        host = host_key(url)
        with self._lock(host):
            low, high = self.policy.interval
            spacing = self.jitter(low, high)   # 揺らぎは枠ごとに一度だけ引く
            while True:
                hosts = self._load()
                entry = self._host(hosts, host)
                now = self.clock()
                cooldown = float(entry["cooldown_until"]) - now
                factor = min(max(float(entry["factor"]), 1.0), self.policy.factor_limit)
                gap = float(entry["last_request"]) + spacing * factor - now
                wait = max(cooldown, gap, 0.0)
                if cooldown > self.policy.max_wait:
                    self.log.append({"host": host, "purpose": purpose, "deferred_seconds": round(cooldown, 1)})
                    raise Deferred(host, cooldown)
                if wait <= 0:
                    break
                if should_stop():
                    raise Deferred(host, wait)
                self.sleep(min(wait, 5.0))
            entry["last_request"] = self.clock()
            event = {"host": host, "purpose": purpose, "at": round(entry["last_request"], 3),
                     "factor": round(factor, 2)}
            # 直近の通信の記録（間隔・倍率）。状態領域に残して、礼儀を守ったかを後から確かめられるようにする。
            entry["recent"] = (list(entry.get("recent") or []) + [event])[-RECENT_LIMIT:]
            hosts[host] = entry
            self._save(hosts)
            self.log.append(event)
            yield

    def report(self, url: str, outcome: str | None = None, *, retry_after=None) -> None:
        """通信の結果を返す。None＝成功、`RATE_LIMITED`＝混雑の合図、`TRANSIENT`＝一時的な失敗。"""
        host = host_key(url)
        with self._guard:
            hosts = self._load()
            entry = self._host(hosts, host)
            now = self.clock()
            if outcome == RATE_LIMITED:
                entry["streak"] = int(entry["streak"]) + 1
                entry["successes"] = 0
                entry["factor"] = min(float(entry["factor"]) * 2.0, self.policy.factor_limit)
                pause = min(self.policy.cooldown_start * 2 ** (entry["streak"] - 1), self.policy.cooldown_limit)
                told = retry_after_seconds(retry_after, now)
                if told is not None:
                    pause = max(pause, min(told, 24 * 3600.0))
                entry["cooldown_until"] = max(float(entry["cooldown_until"]), now + pause)
                entry["last_signal"] = {"kind": RATE_LIMITED, "at": datetime.now(timezone.utc).isoformat(),
                                        "pause_seconds": round(pause, 1), "retry_after": told}
            elif outcome == TRANSIENT:
                entry["successes"] = 0
                entry["factor"] = min(float(entry["factor"]) * 1.5, self.policy.factor_limit)
                entry["cooldown_until"] = max(float(entry["cooldown_until"]),
                                              now + self.policy.transient_cooldown)
                entry["last_signal"] = {"kind": TRANSIENT, "at": datetime.now(timezone.utc).isoformat()}
            else:
                entry["successes"] = int(entry["successes"]) + 1
                if entry["successes"] >= self.policy.recover_after:
                    entry["successes"] = 0
                    entry["factor"] = max(1.0, float(entry["factor"]) / 2.0)
                    if entry["factor"] == 1.0:
                        entry["streak"] = 0
            hosts[host] = entry
            self._save(hosts)


_SHARED: dict[str, Courtesy] = {}
_SHARED_LOCK = threading.Lock()


def shared(state_root: Path | None) -> Courtesy:
    """状態領域ごとに一つの調整役（同じプロセスの発見・取り直しが同じ順番と間隔を使う）。"""
    key = str(Path(state_root).resolve()) if state_root is not None else ""
    with _SHARED_LOCK:
        found = _SHARED.get(key)
        if found is None:
            found = _SHARED[key] = Courtesy(state_root)
        return found
