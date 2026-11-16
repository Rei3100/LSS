"""yt-dlpの実行物の選択と、検証付きの更新。

更新は `docs/RELIABILITY_AND_TRUST.md` の手順（発見→取得→検証→配置→動作確認→切替）に従う。

- 発見: GitHubの最新リリース。
- 検証: リポジトリに固定した公開鍵で `SHA2-256SUMS` の署名を確かめ、実行物のSHA-256を照合する。
- 動作確認: 版の表示と、ローカルHTTPだけを使う取得の試行。外部サービスへは接続しない。
- 切替: 確認に通った版だけを有効にし、直前の版を戻し先として残す。失敗した候補は有効にしない。

アプリ本体は更新しない。有効な実行物が壊れていれば直前の版へ戻り、開発環境では仮想環境の
モジュールへ戻る。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from lss.acquisition import openpgp
from lss.acquisition.store import AcquisitionFailure, AuthMethod, FailureKind
from lss.core import session
from lss.paths import is_frozen

logger = logging.getLogger("lss.acquisition")

RELEASE_API = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"
ASSET = "yt-dlp.exe"
SUMS = "SHA2-256SUMS"
SIGNATURE = "SHA2-256SUMS.sig"
# yt-dlpの配布署名鍵（リポジトリの public.key）。鍵を替えるときはコードの変更として行う。
PINNED_FINGERPRINTS = frozenset({"AC0CBBE6848D6A873464AF4E57CF65933B5A7581"})
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
# 取得の失敗をきっかけにした確認は、これより頻繁には行わない。
FAILURE_CHECK_INTERVAL_SECONDS = 60 * 60
DOWNLOAD_TIMEOUT_SECONDS = 60
VERSION_TIMEOUT_SECONDS = 60
USER_AGENT = "LSS-provider-update"

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Provider:
    command: tuple[str, ...]
    kind: str
    version: str | None = None
    sha256: str | None = None

    def identity(self) -> dict:
        return {"kind": self.kind, "version": self.version, "sha256": self.sha256}


def root() -> Path:
    return session.state_dir() / "providers" / "yt-dlp"


def _state_path() -> Path:
    return root() / "state.json"


def load_state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    from lss.acquisition.store import _write_json_atomic

    _write_json_atomic(_state_path(), state)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed(entry: dict | None) -> Provider | None:
    if not entry:
        return None
    path = root() / "versions" / str(entry.get("version")) / ASSET
    try:
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            return None
    except OSError:
        return None
    return Provider((str(path),), "release", entry["version"], entry["sha256"])


def active() -> Provider:
    """使う実行物。検証済みの有効版、壊れていれば直前の版、開発環境ではモジュール。"""
    state = load_state()
    for key in ("active", "previous"):
        found = _installed(state.get(key))
        if found is not None:
            if key == "previous":
                logger.warning("有効なyt-dlpが壊れているため直前の版を使います: %s", found.version)
            return found
    if not is_frozen():
        try:
            import yt_dlp.version as version

            return Provider((sys.executable, "-m", "yt_dlp"), "module", version.__version__)
        except ImportError:
            pass
    raise AcquisitionFailure(FailureKind.TOOL_UNAVAILABLE, "検証済みのyt-dlpが導入されていません")


# --- 発見と取得 ---------------------------------------------------------------

def http_get(url: str) -> bytes:
    import ssl
    import urllib.request

    try:
        import certifi

        # この環境のPython既定の証明書ストアではGitHubの発行者を検証できないことを実測した。
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS, context=context) as response:  # noqa: S310
        return response.read()


def pinned_keys() -> list[openpgp.RsaKey]:
    armored = (Path(__file__).with_name("keys") / "yt-dlp.asc").read_text(encoding="ascii")
    keys = [key for key in openpgp.public_keys(armored) if key.fingerprint in PINNED_FINGERPRINTS]
    if not keys:
        raise openpgp.SignatureError("固定した指紋の公開鍵が見つかりません")
    return keys


def parse_sums(text: str) -> dict[str, str]:
    sums = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def validate_candidate(command: tuple[str, ...], expected_version: str) -> None:
    """候補を有効にしてよいか。版の表示と、ローカルHTTPでの無変換取得を確かめる。"""
    completed = subprocess.run([*command, "--version"], capture_output=True, timeout=VERSION_TIMEOUT_SECONDS,
                               stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    shown = completed.stdout.decode("utf-8", errors="replace").strip()
    if completed.returncode or shown != expected_version:
        raise RuntimeError(f"版の表示が一致しません: {shown!r} != {expected_version!r}")
    _smoke_download(command)


def _smoke_download(command: tuple[str, ...]) -> None:
    import numpy as np
    import soundfile as sf

    from lss.acquisition.ytdlp import YtDlpBackend

    with TemporaryDirectory(prefix="lss-ytdlp-validate-") as folder:
        folder = Path(folder)
        sample = folder / "sample.flac"
        rng = np.random.default_rng(0)
        sf.write(sample, (rng.standard_normal((44100, 2)) * 0.1).astype(np.float32), 44100, subtype="PCM_16")
        payload = sample.read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_HEAD(self):
                self._send(False)

            def do_GET(self):
                self._send(True)

            def _send(self, body: bool):
                self.send_response(200)
                self.send_header("Content-Type", "audio/flac")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if body:
                    self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            backend = YtDlpBackend(command, provider={"kind": "candidate"})
            auth = AuthMethod()
            [entry] = backend.resolve(f"http://127.0.0.1:{server.server_address[1]}/sample.flac", auth)
            description = backend.describe(entry, auth)
            saved = backend.download(description, folder / "work", auth)
        finally:
            server.shutdown()
            server.server_close()
        if saved.read_bytes() != payload:
            raise RuntimeError("取得の試行で原本と異なるバイト列になりました")


def check_and_install(*, fetch: Fetch = http_get,
                      validate: Callable[[tuple[str, ...], str], None] = validate_candidate,
                      now: float | None = None) -> dict:
    """最新リリースを確かめ、新しければ検証して有効にする。結果を状態へ記録して返す。"""
    now = time.time() if now is None else now
    state = load_state()
    state["last_check"] = now
    result: dict = {"checked_at": datetime.fromtimestamp(now, timezone.utc).isoformat()}
    try:
        release = json.loads(fetch(RELEASE_API))
        version = str(release["tag_name"])
        result["latest"] = version
        current = _installed(state.get("active"))
        if current is not None and current.version == version:
            result["status"] = "current"
            return result
        assets = {item["name"]: item for item in release.get("assets", [])}
        missing = [name for name in (ASSET, SUMS, SIGNATURE) if name not in assets]
        if missing:
            raise RuntimeError(f"リリースに必要な配布物がありません: {missing}")

        sums_bytes = fetch(assets[SUMS]["browser_download_url"])
        signature = fetch(assets[SIGNATURE]["browser_download_url"])
        signer = openpgp.verify_detached(sums_bytes, signature, pinned_keys())
        expected = parse_sums(sums_bytes.decode("utf-8", errors="replace")).get(ASSET)
        if expected is None:
            raise RuntimeError("署名済みのチェックサムに実行物がありません")

        staging = root() / "staging" / version
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        candidate = staging / ASSET
        candidate.write_bytes(fetch(assets[ASSET]["browser_download_url"]))
        actual = _sha256(candidate)
        digest = str(assets[ASSET].get("digest") or "")
        if actual != expected or (digest.startswith("sha256:") and digest[7:] != actual):
            raise RuntimeError("実行物のSHA-256が署名済みのチェックサムと一致しません")

        validate((str(candidate),), version)

        destination = root() / "versions" / version
        shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
        entry = {"version": version, "sha256": actual, "signed_by": signer.fingerprint,
                 "validated_at": datetime.fromtimestamp(now, timezone.utc).isoformat()}
        if current is not None:
            state["previous"] = state.get("active")
        state["active"] = entry
        state.pop("held", None)
        result.update(status="installed", version=version)
        _prune(state)
        return result
    except Exception as exc:  # 候補は有効にしない。現在の版のまま使い続ける。
        message = f"{type(exc).__name__}: {exc}"
        result.update(status="held", error=message[:500])
        state["held"] = result
        logger.warning("yt-dlpの更新を保留しました: %s", message)
        return result
    finally:
        shutil.rmtree(root() / "staging", ignore_errors=True)
        _save_state(state)


def _prune(state: dict) -> None:
    keep = {str((state.get(key) or {}).get("version")) for key in ("active", "previous")}
    versions = root() / "versions"
    for entry in versions.iterdir() if versions.is_dir() else []:
        if entry.name not in keep:
            shutil.rmtree(entry, ignore_errors=True)


_lock = threading.Lock()


def ensure(*, after_failure: bool = False, fetch: Fetch = http_get, now: float | None = None,
           validate: Callable[[tuple[str, ...], str], None] = validate_candidate) -> dict | None:
    """必要なときだけ更新を確かめる。導入済みの版が無い配備版では、最初の取得の前に導入する。

    通常は1日に1回。取得が失敗したときは、1時間に1回まで前倒しで確かめる。
    """
    now = time.time() if now is None else now
    with _lock:
        state = load_state()
        installed = _installed(state.get("active")) or _installed(state.get("previous"))
        # 配備版で使える版が無ければ取得できないので、失敗時と同じ間隔で導入を試みる。
        missing = installed is None and is_frozen()
        interval = FAILURE_CHECK_INTERVAL_SECONDS if after_failure or missing else CHECK_INTERVAL_SECONDS
        last = state.get("last_check")
        if last is not None and now - float(last) < interval:
            return None
        return check_and_install(fetch=fetch, validate=validate, now=now)


def environment_allows_updates() -> bool:
    """検証などで外部への更新確認を止めたいときは `LSS_YTDLP_UPDATES=0`。"""
    return os.environ.get("LSS_YTDLP_UPDATES", "1") != "0"
