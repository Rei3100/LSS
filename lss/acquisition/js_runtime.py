"""YouTubeの取得に必要なJavaScript実行環境（deno）を、検証してアプリのフォルダ内に用意する。

- 使える deno がPATHにあればそれを使い、無いときや古いときだけ導入する。
- 導入の検証: GitHubの公式リリースのzipを、公開チェックサムとGitHubの記録するダイジェストで照合し、
  取り出した `deno.exe` のWindowsコード署名が有効で署名者が Deno Land Inc. であることを確かめる。
  さらに版の表示と簡単な計算の実行を確かめてから有効にする。通らない候補は有効にしない。
- 導入した版は状態領域の `providers/deno/` に置き、直前の版を戻し先として残す。
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lss.acquisition.provider import http_get
from lss.core import session

logger = logging.getLogger("lss.acquisition")

RELEASE_API = "https://api.github.com/repos/denoland/deno/releases/latest"
ASSET = "deno-x86_64-pc-windows-msvc.zip"
SUMS = ASSET + ".sha256sum"
EXECUTABLE = "deno.exe"
SIGNER = "CN=Deno Land Inc., O=Deno Land Inc."
MIN_VERSION = (2, 3, 0)  # yt-dlpが対応するdenoの下限
CHECK_INTERVAL_SECONDS = 7 * 24 * 60 * 60
FAILURE_CHECK_INTERVAL_SECONDS = 60 * 60
RUN_TIMEOUT_SECONDS = 60

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Runtime:
    path: str
    version: str
    kind: str  # "managed" | "path"


def root() -> Path:
    return session.state_dir() / "providers" / "deno"


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_of(executable: str | Path) -> tuple[int, ...] | None:
    try:
        completed = subprocess.run([str(executable), "--version"], capture_output=True, timeout=RUN_TIMEOUT_SECONDS,
                                   stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"deno (\d+)\.(\d+)\.(\d+)", completed.stdout.decode("utf-8", errors="replace"))
    return tuple(int(part) for part in match.groups()) if completed.returncode == 0 and match else None


def _installed(entry: dict | None) -> Runtime | None:
    if not entry:
        return None
    path = root() / "versions" / str(entry.get("version")) / EXECUTABLE
    try:
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            return None
    except OSError:
        return None
    return Runtime(str(path), str(entry["version"]), "managed")


def path_runtime() -> Runtime | None:
    found = shutil.which("deno")
    if not found:
        return None
    version = version_of(found)
    if version is None or version < MIN_VERSION:
        return None
    return Runtime(found, ".".join(map(str, version)), "path")


def active() -> Runtime | None:
    """使う実行環境。導入済みの検証済み版（壊れていれば直前の版）、なければPATHの対応版。"""
    state = load_state()
    for key in ("active", "previous"):
        found = _installed(state.get(key))
        if found is not None:
            return found
    return path_runtime()


def authenticode_signer(path: Path) -> tuple[str, str]:
    """Windowsのコード署名の状態と署名者。検証はOSが行う。"""
    import os

    # パスは命令文へ埋め込まず、環境変数で渡す（引用符や記号を含むパスでも解釈が変わらない）。
    command = ("Get-AuthenticodeSignature -LiteralPath $env:LSS_SIGNED_PATH | Select-Object "
               "@{n='Status';e={$_.Status.ToString()}},@{n='Subject';e={$_.SignerCertificate.Subject}} "
               "| ConvertTo-Json -Compress")
    completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                               capture_output=True, timeout=RUN_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
                               env={**os.environ, "LSS_SIGNED_PATH": str(path)},
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        value = json.loads(completed.stdout.decode("utf-8", errors="replace"))
    except ValueError:
        return "Unknown", ""
    return str(value.get("Status") or "Unknown"), str(value.get("Subject") or "")


def validate_candidate(executable: Path, expected_version: str) -> None:
    version = version_of(executable)
    if version is None or ".".join(map(str, version)) != expected_version:
        raise RuntimeError(f"版の表示が一致しません: {version} != {expected_version}")
    completed = subprocess.run([str(executable), "eval", "console.log(6*7)"], capture_output=True,
                               timeout=RUN_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if completed.returncode or completed.stdout.decode("utf-8", errors="replace").strip() != "42":
        raise RuntimeError("JavaScriptの実行を確かめられません")


def check_and_install(*, fetch: Fetch = http_get,
                      signer: Callable[[Path], tuple[str, str]] = authenticode_signer,
                      validate: Callable[[Path, str], None] = validate_candidate,
                      now: float | None = None) -> dict:
    now = time.time() if now is None else now
    state = load_state()
    state["last_check"] = now
    result: dict = {"checked_at": datetime.fromtimestamp(now, timezone.utc).isoformat()}
    staging = root() / "staging"
    try:
        release = json.loads(fetch(RELEASE_API))
        version = str(release["tag_name"]).lstrip("v")
        result["latest"] = version
        current = _installed(state.get("active"))
        if current is not None and current.version == version:
            result["status"] = "current"
            return result
        assets = {item["name"]: item for item in release.get("assets", [])}
        if ASSET not in assets or SUMS not in assets:
            raise RuntimeError("リリースに必要な配布物がありません")
        archive = fetch(assets[ASSET]["browser_download_url"])
        published = re.search(r"\b([0-9a-fA-F]{64})\b", fetch(assets[SUMS]["browser_download_url"]).decode(
            "utf-8", errors="replace"))
        actual = _sha256_bytes(archive)
        digest = str(assets[ASSET].get("digest") or "")
        if published is None or published.group(1).lower() != actual or (
                digest.startswith("sha256:") and digest[7:] != actual):
            raise RuntimeError("配布物のSHA-256が公開チェックサムと一致しません")
        shutil.rmtree(staging, ignore_errors=True)
        candidate_dir = staging / version
        candidate_dir.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            if EXECUTABLE not in bundle.namelist():
                raise RuntimeError("配布物に実行ファイルがありません")
            (candidate_dir / EXECUTABLE).write_bytes(bundle.read(EXECUTABLE))
        candidate = candidate_dir / EXECUTABLE
        status, subject = signer(candidate)
        if status != "Valid" or not subject.startswith(SIGNER):
            raise RuntimeError(f"コード署名を確かめられません: {status} {subject}")
        validate(candidate, version)
        destination = root() / "versions" / version
        shutil.rmtree(destination, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate_dir.replace(destination)
        entry = {"version": version, "sha256": _sha256(destination / EXECUTABLE), "archive_sha256": actual,
                 "signer": subject, "validated_at": result["checked_at"]}
        if current is not None:
            state["previous"] = state.get("active")
        state["active"] = entry
        state.pop("held", None)
        keep = {str((state.get(key) or {}).get("version")) for key in ("active", "previous")}
        for old in (root() / "versions").iterdir():
            if old.name not in keep:
                shutil.rmtree(old, ignore_errors=True)
        result.update(status="installed", version=version)
        return result
    except Exception as exc:
        result.update(status="held", error=f"{type(exc).__name__}: {exc}"[:500])
        state["held"] = result
        logger.warning("denoの導入を保留しました: %s", result["error"])
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        _save_state(state)


_lock = threading.Lock()


def ensure(*, after_failure: bool = False, fetch: Fetch = http_get, now: float | None = None, **options) -> dict | None:
    """PATHに対応版が無いときだけ導入し、導入済みなら週1回（失敗時は1時間に1回まで）更新を確かめる。"""
    now = time.time() if now is None else now
    with _lock:
        state = load_state()
        managed = _installed(state.get("active")) or _installed(state.get("previous"))
        if managed is None and path_runtime() is not None:
            return None
        interval = FAILURE_CHECK_INTERVAL_SECONDS if after_failure or managed is None else CHECK_INTERVAL_SECONDS
        last = state.get("last_check")
        if last is not None and now - float(last) < interval:
            return None
        return check_and_install(fetch=fetch, now=now, **options)
