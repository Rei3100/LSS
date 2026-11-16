"""LSS専用のログインプロファイルのCookieを、取得の認証に使う。値は読まず・記録しない。"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.acquisition import login_profile, service, store
from lss.acquisition.store import AuthMethod, FailureKind
from lss.acquisition.ytdlp import YtDlpBackend


def _chromium_time(moment: datetime) -> int:
    return int((moment - datetime(1601, 1, 1, tzinfo=timezone.utc)).total_seconds() * 1_000_000)


def _cookies(state: Path, rows: list[tuple[str, str, datetime | None]]) -> None:
    database = login_profile.cookie_profile(state) / "Network" / "Cookies"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, value TEXT, encrypted_value BLOB, "
                       "expires_utc INTEGER, has_expires INTEGER)")
    for host, name, expires in rows:
        connection.execute("INSERT INTO cookies VALUES (?, ?, '', ?, ?, ?)",
                           (host, name, b"encrypted-secret", _chromium_time(expires) if expires else 0,
                            1 if expires else 0))
    connection.commit()
    connection.close()


def test_logged_in_services_come_from_unexpired_session_cookie_names(tmp_path):
    now = datetime(2026, 9, 13, tzinfo=timezone.utc)
    _cookies(tmp_path, [
        (".nicovideo.jp", "user_session", now + timedelta(days=30)),
        (".bilibili.com", "SESSDATA", now - timedelta(days=1)),
        (".bilibili.com", "buvid3", now + timedelta(days=300)),
        (".youtube.com", "PREF", now + timedelta(days=300)),
    ])
    states = login_profile.sessions(tmp_path, now=now)
    assert states["niconico"].logged_in and states["niconico"].expires_at.startswith("2026-10-13")
    assert not states["bilibili"].logged_in  # 期限切れのセッションと、ログインと無関係なCookie
    assert not states["youtube"].logged_in


def test_auth_is_chosen_per_service_and_provenance_never_names_the_profile(tmp_path):
    _cookies(tmp_path, [(".nicovideo.jp", "user_session", datetime.now(timezone.utc) + timedelta(days=1))])
    auth = login_profile.auth_for("https://www.nicovideo.jp/watch/sm9", tmp_path)
    assert auth.method == "app_login" and Path(auth.profile) == login_profile.cookie_profile(tmp_path)
    assert auth.provenance() == {"method": "app_login"}
    assert login_profile.auth_for("https://www.bilibili.com/video/BV1xx411c7mD", tmp_path) == AuthMethod()
    assert login_profile.auth_for("https://example.com/a.flac", tmp_path) == AuthMethod()
    assert login_profile.service_of("https://youtu.be/aqz-KE-bpKQ") == "youtube"


def test_no_profile_means_no_login(tmp_path):
    assert not any(state.logged_in for state in login_profile.sessions(tmp_path).values())
    assert login_profile.auth_for("https://www.nicovideo.jp/watch/sm9", tmp_path) == AuthMethod()


def test_a_profile_held_open_by_the_login_window_is_reported_busy(tmp_path, monkeypatch):
    _cookies(tmp_path, [(".nicovideo.jp", "user_session", datetime.now(timezone.utc) + timedelta(days=1))])

    def locked(*_args, **_kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(login_profile.shutil, "copyfile", locked)
    with pytest.raises(login_profile.LoginProfileBusy):
        login_profile.sessions(tmp_path)
    # 取得はCookieを使う取得として試み、読めなければ取得手段の失敗の種類で知らせる。
    assert login_profile.auth_for("https://www.nicovideo.jp/watch/sm9", tmp_path).method == "app_login"


# --- 実際のWebView2プロファイルと実際のyt-dlp -------------------------------

_MAKE_PROFILE = r'''
import sys, threading, time, http.server
import webview
url, root = sys.argv[1], sys.argv[2]
window = webview.create_window("lss-login-probe", url, hidden=True)
def finish():
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            if window.evaluate_js("document.cookie.length >= 0 && document.readyState") == "complete":
                break
        except Exception:
            pass
        time.sleep(0.2)
    time.sleep(1.5)  # Cookieのデータベースへ書き出されるのを待つ
    window.destroy()
webview.start(finish, gui="edgechromium", private_mode=False, storage_path=root)
'''


class _Site(BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path.startswith("/login"):
            self.send_response(200)
            self.send_header("Set-Cookie", "lss_session=value-never-recorded; Max-Age=86400; Path=/; HttpOnly")
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>logged in</body></html>")
            return
        if "lss_session=value-never-recorded" not in (self.headers.get("Cookie") or ""):
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/flac")
        self.send_header("Content-Length", str(len(type(self).payload)))
        self.end_headers()
        self.wfile.write(type(self).payload)


def test_cookies_from_the_lss_webview2_profile_authenticate_the_real_acquisition_tool(tmp_path, monkeypatch):
    pytest.importorskip("webview")
    state = tmp_path / "state"
    monkeypatch.setenv("LSS_STATE_DIR", str(state))
    monkeypatch.setenv(store.LIBRARY_ENV, str(tmp_path / "library"))
    media = tmp_path / "song.flac"
    sf.write(media, (np.random.default_rng(3).standard_normal((44100 * 2, 2)) * 0.1).astype(np.float32),
             44100, subtype="PCM_16")
    handler = type("Site", (_Site,), {"payload": media.read_bytes()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        subprocess.run([sys.executable, "-c", _MAKE_PROFILE, f"{base}/login",
                        str(login_profile.profile_root(state))], check=True, timeout=90)
        assert (login_profile.cookie_profile(state) / "Network" / "Cookies").is_file()
        backend = YtDlpBackend((sys.executable, "-m", "yt_dlp"))
        pacing = service.Pacing(between_items=(0.0, 0.0), backoff_start=0.0, attempts=1)

        [refused] = service.acquire(f"{base}/member/song.flac", backend=backend, pacing=pacing)
        assert refused.failure is not None and refused.failure.kind is not FailureKind.AUTH_UNAVAILABLE

        auth = AuthMethod.app_login(login_profile.cookie_profile(state))
        [outcome] = service.acquire(f"{base}/member/song.flac", backend=backend, auth=auth, pacing=pacing)
        assert outcome.ok, outcome.failure
        assert outcome.source.path.read_bytes() == media.read_bytes()
        recorded = json.dumps(outcome.source.record, ensure_ascii=False)
        assert outcome.source.record["auth"] == {"method": "app_login"}
        assert "value-never-recorded" not in recorded and str(login_profile.profile_root(state)) not in recorded
    finally:
        server.shutdown()
        server.server_close()


def test_a_user_specified_cookie_file_is_used_when_lss_is_not_logged_in(tmp_path, monkeypatch):
    """利用者が指定したCookieファイル（会員向けの音源）。LSSのログインが無い時だけ使い、中身・所在は記録しない。"""
    import json as _json

    cookies = tmp_path / "youtube.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(login_profile, "GUI_COOKIE_DIR", None)
    monkeypatch.setenv("LSS_GUI_COOKIE_DIR", str(tmp_path / "gui"))
    assert login_profile.auth_for("https://music.youtube.com/watch?v=x", state) == AuthMethod()
    (state / login_profile.COOKIE_FILES).write_text(_json.dumps({"youtube": str(cookies)}), encoding="utf-8")
    auth = login_profile.auth_for("https://music.youtube.com/watch?v=x", state)
    assert auth.method == "cookies_file" and auth.provenance() == {"method": "cookies_file"}
    assert login_profile.auth_for("https://www.nicovideo.jp/watch/sm9", state) == AuthMethod()
    monkeypatch.setenv("LSS_COOKIES_FILE_YOUTUBE", str(tmp_path / "missing.txt"))
    assert login_profile.auth_for("https://youtu.be/x", state) == AuthMethod()   # 無いファイルは使わない


def test_the_yt_dlp_gui_cookie_file_is_used_from_the_configured_directory(tmp_path, monkeypatch):
    """GUIと同じCookieフォルダーを `LSS_GUI_COOKIE_DIR` で指定して使う。"""
    gui = tmp_path / "gui"
    gui.mkdir()
    (gui / "youtube.txt").write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setattr(login_profile, "GUI_COOKIE_DIR", None)
    monkeypatch.setenv("LSS_GUI_COOKIE_DIR", str(gui))
    monkeypatch.delenv("LSS_COOKIES_FILE_YOUTUBE", raising=False)
    state = tmp_path / "state"
    state.mkdir()
    auth = login_profile.auth_for("https://music.youtube.com/watch?v=x", state)
    assert auth == AuthMethod.cookies_file(gui / "youtube.txt")
    assert login_profile.auth_for("https://www.nicovideo.jp/watch/sm9", state) == AuthMethod()


def test_the_cookie_file_is_passed_as_a_copy_and_the_original_is_never_rewritten(tmp_path):
    """yt-dlpは `--cookies` のファイルへ書き戻す。取得ごとの複製を渡し、終われば消す。"""
    from lss.acquisition.ytdlp import YtDlpBackend

    cookies = tmp_path / "youtube.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    script = tmp_path / "fake_ytdlp.py"
    script.write_text(
        "import sys\n"
        "args = sys.argv[1:]\n"
        "path = args[args.index('--cookies') + 1]\n"
        "open(path, 'a').write('# rewritten by tool')\n"
        "print(path)\n", encoding="utf-8")
    backend = YtDlpBackend([sys.executable, str(script)])
    run, _ = backend._run([], AuthMethod.cookies_file(cookies))
    copy = Path(run.stdout[0])
    assert copy != cookies and not copy.exists() and not copy.parent.exists()
    assert cookies.read_text(encoding="utf-8") == "# Netscape HTTP Cookie File\n"
