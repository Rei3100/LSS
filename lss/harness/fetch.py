"""モデル資産（重み・config）の取得（D-4キャッシュ／T325・T326）。

ダウンロードはユーザー許可を要さない（本人指示2026-08-14＝AGENTS.md運用原則・srcのidea制約欄）。
台帳`ledger/models.tsv`の`distributor_url`/`config_url`に記録済みの配布元のみを対象にする。

キャッシュ先はD-4に従い環境変数`LSS_MODEL_DIR`（未設定なら開発ツリー配下`.models/`）。
実行ツリー配備時は配備先の`models\\`を指すようにこの環境変数を設定する。
"""
from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlsplit

import requests

from lss.paths import app_root, is_frozen

_CHUNK = 1 << 20  # 1MiB
MAX_CONFIG_BYTES = 4 * 1024**2
_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class InvalidModelSourceError(ValueError):
    """モデル取得元またはキャッシュ名が安全な取得契約に合わない。"""


class DownloadSizeError(ValueError):
    """取得物が台帳の予定サイズまたは資産別の上限に合わない。"""


class DownloadTooLargeError(DownloadSizeError):
    """取得物が台帳の予定サイズまたは資産別の上限を超えた。"""


class DownloadIntegrityError(DownloadSizeError):
    """取得物のサイズまたはSHA-256が期待した実体と一致しない。"""


class ModelDownloadError(RuntimeError):
    """HTTPS通信または配布元の応答によりモデル資産を取得できない。"""


def _https_url(url: str):
    if not isinstance(url, str):
        raise InvalidModelSourceError("モデル取得元はHTTPS URLである必要があります")
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise InvalidModelSourceError(f"モデル取得元は認証情報を含まないHTTPS URLである必要があります: {url}")
    return parsed


def _safe_component(value: str, label: str) -> str:
    decoded = unquote(value)
    if (
        not decoded or decoded in {".", ".."} or decoded[-1] in {" ", "."}
        or any(character in _WINDOWS_FORBIDDEN or ord(character) < 32 for character in decoded)
        or decoded.split(".", 1)[0].upper() in _WINDOWS_RESERVED
    ):
        raise InvalidModelSourceError(f"{label}を安全なWindowsファイル名として扱えません: {value}")
    return decoded


def validate_model_id(model_id: str) -> str:
    return _safe_component(model_id, "model_id")


def validate_source_url(url: str) -> None:
    _https_url(url)


def _asset_name(url: str) -> str:
    parsed = _https_url(url)
    return _safe_component(parsed.path.rsplit("/", 1)[-1], "モデル資産名")


def model_cache_dir() -> Path:
    override = os.environ.get("LSS_MODEL_DIR")
    if override:
        return Path(override)
    return app_root() / ("models" if is_frozen() else ".models")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_download(part: Path, dest: Path) -> None:
    """Windowsの同時renameや一時的な共有違反だけを、短い上限内で再試行する。"""
    for attempt in range(6):
        try:
            part.replace(dest)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def download(
    url: str, dest: Path, timeout: float = 60.0, *,
    expected_size: int | None = None, expected_sha256: str | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    max_bytes: int | None = None,
) -> Path:
    """destが既に存在すればそのまま返す（再取得しない）。無ければストリーミング取得する。

    途中で失敗した場合は部分ファイルを残さない（.part→renameの2段）。
    """
    _https_url(url)
    if expected_sha256:
        normalized_sha256 = expected_sha256.strip().lower()
        if len(normalized_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_sha256
        ):
            raise InvalidModelSourceError("モデル資産のSHA-256形式が不正です")
    if expected_size is not None and expected_size < 0:
        raise DownloadSizeError("取得予定サイズは0以上である必要があります")
    if max_bytes is not None and max_bytes < 0:
        raise DownloadSizeError("取得上限は0以上である必要があります")
    byte_limit = expected_size if expected_size is not None else max_bytes
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = is_cached(dest, expected_size, expected_sha256)
    if cached and (max_bytes is None or dest.stat().st_size <= max_bytes):
        return dest

    with NamedTemporaryFile(
        dir=dest.parent, prefix=f".{dest.name}-", suffix=".part", delete=False,
    ) as temporary:
        part = Path(temporary.name)
    try:
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for hop in [*getattr(resp, "history", []), resp]:
                    _https_url(getattr(hop, "url", url))
                header_size = getattr(resp, "headers", {}).get("content-length")
                total_size = expected_size
                if total_size is None and header_size:
                    try:
                        total_size = int(header_size)
                    except (TypeError, ValueError):
                        total_size = None
                if byte_limit is not None and total_size is not None and total_size > byte_limit:
                    raise DownloadTooLargeError(
                        f"取得ファイルが上限を超えます: {dest.name} ({total_size}>{byte_limit})"
                    )
                downloaded = 0
                if progress is not None:
                    progress(downloaded, total_size)
                with open(part, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if chunk:
                            downloaded += len(chunk)
                            if byte_limit is not None and downloaded > byte_limit:
                                raise DownloadTooLargeError(
                                    f"取得ファイルが上限を超えます: {dest.name} ({downloaded}>{byte_limit})"
                                )
                            f.write(chunk)
                            if progress is not None:
                                progress(downloaded, total_size)
        except requests.RequestException as exc:
            raise ModelDownloadError(f"モデル資産を取得できません: {url} ({exc})") from exc
        if not is_cached(part, expected_size):
            raise DownloadIntegrityError(f"取得ファイルのサイズが不正です: {dest.name}")
        if expected_sha256 and file_sha256(part) != expected_sha256.strip().lower():
            raise DownloadIntegrityError(f"取得ファイルのSHA-256が不正です: {dest.name}")
        _publish_download(part, dest)
    finally:
        if part.exists():
            part.unlink()
    return dest


def is_cached(
    path: Path, expected_size: int | None = None, expected_sha256: str | None = None,
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return not expected_sha256 or file_sha256(path) == expected_sha256.strip().lower()


def model_asset_path(model_id: str, url: str) -> Path:
    root = model_cache_dir().resolve()
    safe_model_id = validate_model_id(model_id)
    base = (root / safe_model_id).resolve()
    if base.parent != root:
        raise InvalidModelSourceError(f"モデルキャッシュの直下ではありません: {base}")
    return base / _asset_name(url)


def model_paths(model_id: str, ckpt_url: str, config_url: str) -> tuple[Path, Path]:
    ckpt_path = model_asset_path(model_id, ckpt_url)
    config_path = model_asset_path(model_id, config_url)
    ckpt_name = ckpt_path.name
    config_name = config_path.name
    if ckpt_name == config_name:
        raise InvalidModelSourceError("モデル重みと設定ファイルの保存名が重複しています")
    return ckpt_path, config_path


def ensure_model_files(
    model_id: str, ckpt_url: str, config_url: str, *,
    expected_size: int | None = None, expected_sha256: str | None = None,
    weight_progress: Callable[[int, int | None], None] | None = None,
    config_progress: Callable[[int, int | None], None] | None = None,
    expected_config_sha256: str | None = None,
) -> tuple[Path, Path]:
    """model_id専用のキャッシュ配下へ重みとconfigを取得し、(ckpt_path, config_path)を返す。

    配布元のファイル名をそのまま保持する（同一リポ内の別バリアントと衝突させないため
    model_idごとのサブディレクトリへ置く）。
    """
    ckpt_path, config_path = model_paths(model_id, ckpt_url, config_url)
    ckpt = download(
        ckpt_url, ckpt_path, expected_size=expected_size, expected_sha256=expected_sha256,
        progress=weight_progress,
    )
    config = download(
        config_url, config_path, progress=config_progress, max_bytes=MAX_CONFIG_BYTES,
        expected_sha256=expected_config_sha256,
    )
    return ckpt, config
