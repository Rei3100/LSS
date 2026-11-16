"""LSSが持つ専用のログインプロファイル（WebView2）のCookieを、取得の認証に使う。

普段使いのブラウザのCookieは、使用中はデータベースが排他ロックされ、App-Bound暗号化で
取得ツールから復号できないことがある。LSS専用のWebView2プロファイルは暗号鍵がDPAPIだけで、
LSSがログイン用の窓を閉じていれば取得ツールがそのまま読める。

ここが持つのは場所・ログインの有無の判定・取得に渡す認証の種類だけ。Cookieの値は読まない。
ログインの窓（サービスのログインページを開く面）は画面側が持つ。
"""
from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from lss.acquisition.store import AuthMethod
from lss.core import session

PROFILE_DIRNAME = "login-profile"
# WebView2のユーザーデータフォルダの中で、Chromiumの既定プロファイルが置かれる場所。
_CHROMIUM_PROFILE = Path("EBWebView") / "Default"
# Chromiumの時刻は1601-01-01からのマイクロ秒。
_CHROMIUM_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)

# サービスごとの、ログインしている間だけ存在するセッションCookieの名前と置き場所。
SESSION_COOKIES: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {
    "niconico": (("nicovideo.jp",), frozenset({"user_session"})),
    "bilibili": (("bilibili.com",), frozenset({"SESSDATA"})),
    "youtube": (("youtube.com",), frozenset({"__Secure-3PSID", "SID", "LOGIN_INFO"})),
}


class LoginProfileBusy(RuntimeError):
    """ログインの窓がプロファイルを使用中で、Cookieのデータベースを読めない。"""


@dataclass(frozen=True)
class SessionState:
    service: str
    logged_in: bool
    expires_at: str | None = None  # 期限の最も遅いセッションCookie。ブラウザを閉じると消えるものはNone


def profile_root(state_root: Path | None = None) -> Path:
    """ログインの窓に渡すユーザーデータフォルダ。状態領域の中（配備の更新でも保持）。"""
    return (Path(state_root) if state_root is not None else session.state_dir()) / PROFILE_DIRNAME


def cookie_profile(state_root: Path | None = None) -> Path:
    return profile_root(state_root) / _CHROMIUM_PROFILE


def service_of(url: str) -> str | None:
    host = (urlsplit(url).hostname or "").lower()
    for service, (domains, _names) in SESSION_COOKIES.items():
        if any(host == domain or host.endswith("." + domain) for domain in domains):
            return service
    if host == "youtu.be":
        return "youtube"
    return None


def _cookie_rows(state_root: Path | None) -> list[tuple[str, str, int, int]]:
    database = cookie_profile(state_root) / "Network" / "Cookies"
    if not database.is_file():
        return []
    with TemporaryDirectory(prefix="lss-login-cookies-") as folder:
        copy = Path(folder) / "Cookies"
        try:
            shutil.copyfile(database, copy)
        except PermissionError as exc:
            raise LoginProfileBusy("ログインの窓が開いている間はログインの状態を読めません") from exc
        connection = sqlite3.connect(f"file:{copy.as_posix()}?mode=ro", uri=True)
        try:
            # 値（value / encrypted_value）は選ばない。名前・所在・期限だけを見る。
            return connection.execute(
                "SELECT host_key, name, expires_utc, has_expires FROM cookies").fetchall()
        finally:
            connection.close()


def sessions(state_root: Path | None = None, *, now: datetime | None = None) -> dict[str, SessionState]:
    """サービスごとのログインの有無。期限切れのセッションCookieはログインとみなさない。"""
    now = now or datetime.now(timezone.utc)
    rows = _cookie_rows(state_root)
    result: dict[str, SessionState] = {}
    for service, (domains, names) in SESSION_COOKIES.items():
        latest: datetime | None = None
        session_only = False
        for host, name, expires_utc, has_expires in rows:
            host = host.lstrip(".").lower()
            if name not in names or not any(host == d or host.endswith("." + d) for d in domains):
                continue
            if not has_expires or not expires_utc:
                session_only = True
                continue
            expires = _CHROMIUM_EPOCH + timedelta(microseconds=int(expires_utc))
            if expires > now and (latest is None or expires > latest):
                latest = expires
        logged_in = latest is not None or session_only
        result[service] = SessionState(service, logged_in, latest.isoformat() if latest else None)
    return result


COOKIE_FILES = "cookie-files.json"
COOKIE_FILE_ENV = "LSS_COOKIES_FILE_{service}"
# 利用者のyt-dlp GUIがサービスごとに置くCookieファイル（`<service>.txt`）のフォルダ。場所は機械ごとに違うので
# コードには持たず、環境変数 `LSS_GUI_COOKIE_DIR` で指定されたときだけ使う。
GUI_COOKIE_DIR_ENV = "LSS_GUI_COOKIE_DIR"
GUI_COOKIE_DIR: Path | None = None


def cookie_file_for(service: str, state_root: Path | None = None) -> Path | None:
    """そのサービスのCookieファイル。環境変数、状態領域の `cookie-files.json`、`LSS_GUI_COOKIE_DIR` のフォルダの `<service>.txt` の順。
    無ければNone。

    中身は読まない（yt-dlpへ複製を渡すだけ）。所在が無いファイルは使わない。
    """
    import json
    import os

    value = os.environ.get(COOKIE_FILE_ENV.format(service=service.upper()))
    if not value:
        root = Path(state_root) if state_root is not None else session.state_dir()
        try:
            value = (json.loads((root / COOKIE_FILES).read_text(encoding="utf-8")) or {}).get(service)
        except (OSError, ValueError, AttributeError):
            value = None
    if value:
        path = Path(value)
    else:
        folder = GUI_COOKIE_DIR or (Path(os.environ[GUI_COOKIE_DIR_ENV]) if os.environ.get(GUI_COOKIE_DIR_ENV) else None)
        if folder is None:
            return None
        path = folder / f"{service}.txt"
    return path if path.is_file() else None


def auth_for(url: str, state_root: Path | None = None) -> AuthMethod:
    """そのURLのサービスへLSSのプロファイルでログイン済みなら、そのCookieを使う認証。無ければ利用者が指定した
    Cookieファイル（会員向けの音源など）。どちらも無ければ認証なし。"""
    service = service_of(url)
    if service is None:
        return AuthMethod()
    configured = cookie_file_for(service, state_root)
    try:
        state = sessions(state_root).get(service)
    except LoginProfileBusy:
        # 窓を開いたままの取得は、Cookieを使う取得として試み、読めなければ失敗の種類で知らせる。
        return AuthMethod.app_login(cookie_profile(state_root))
    if state is None or not state.logged_in:
        return AuthMethod.cookies_file(configured) if configured is not None else AuthMethod()
    return AuthMethod.app_login(cookie_profile(state_root))
