"""取得した素材の公開・来歴台帳・再利用。取得手段に依存しない。

- 取得物は作業領域で完成させ、内容ハッシュを確定してから取得ライブラリへ置く。
- 取得ライブラリの既存ファイルは上書きしない（ソースは不変）。
- 来歴は論理ソースごとのJSONに、取得の版（revision）を追記して残す。
- 資格情報そのものは来歴・診断へ書かない。認証の種類だけを残す。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile

from lss.core import session
from lss.paths import app_root

RECORD_SCHEMA = 1
LIBRARY_ENV = "LSS_ACQUISITION_DIR"
STALE_WORK_SECONDS = 7 * 24 * 60 * 60
WORK_PREFIX = "acquire-"
_MAX_TITLE_CHARS = 120
_CHUNK = 1 << 20


class FailureKind(str, Enum):
    UNSUPPORTED_URL = "unsupported_url"
    AUTH_REQUIRED = "auth_required"
    # 認証情報を使ったのに拒否された（ブラウザでのログインが切れている等）。
    AUTH_EXPIRED = "auth_expired"
    # 指定したブラウザのCookieを読めない（ブラウザが使用中・暗号化の変更等）。
    AUTH_UNAVAILABLE = "auth_unavailable"
    UNAVAILABLE = "unavailable"
    REGION_RESTRICTED = "region_restricted"
    RATE_LIMITED = "rate_limited"
    NETWORK = "network"
    INTERRUPTED = "interrupted"
    INVALID_MEDIA = "invalid_media"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INSUFFICIENT_SPACE = "insufficient_space"
    # 利用者の停止。部分ファイルは残り、次回の取得が続きから再開する。
    CANCELLED = "cancelled"
    BUSY = "busy"
    EXTRACTOR_FAILED = "extractor_failed"


# 同じ要求をそのまま再試行して成功し得る失敗。認証・対象の存在・URLの問題は含めない。
RETRYABLE = frozenset({
    FailureKind.RATE_LIMITED, FailureKind.NETWORK, FailureKind.INTERRUPTED, FailureKind.BUSY,
})


class AcquisitionFailure(RuntimeError):
    """取得の失敗。メッセージは秘匿情報を除去済みの診断文。"""

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(sanitize_text(message))
        self.kind = kind

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE


@dataclass(frozen=True)
class AuthMethod:
    """認証の種類。ブラウザのCookieを使うときも、値は記録しない。"""

    method: str = "none"
    browser: str | None = None
    profile: str | None = None

    @classmethod
    def browser_cookies(cls, browser: str, profile: str | None = None) -> "AuthMethod":
        if not browser:
            raise ValueError("Cookieを読むブラウザを指定してください")
        return cls("browser_cookies", browser.lower(), profile)

    @classmethod
    def cookies_file(cls, path: Path) -> "AuthMethod":
        """利用者が指定したCookieファイル（Netscape形式。yt-dlpの `--cookies`）。中身も所在も記録しない。"""
        return cls("cookies_file", None, str(path))

    @classmethod
    def app_login(cls, profile: Path) -> "AuthMethod":
        """LSS専用のログインプロファイル（WebView2＝Chromium系）のCookie。"""
        return cls("app_login", "edge", str(profile))

    def provenance(self) -> dict:
        # プロファイル名はアカウントを特定し得るので、指定の有無だけを残す。
        if self.method == "none":
            return {"method": "none"}
        if self.method in ("app_login", "cookies_file"):
            return {"method": self.method}
        return {"method": self.method, "browser": self.browser,
                "profile_specified": self.profile is not None}


@dataclass
class AcquiredSource:
    path: Path
    sha256: str
    logical_source: str
    reused: bool
    record: dict = field(repr=False)


# --- 秘匿情報の除去 ----------------------------------------------------------

_URL = re.compile(r"(https?://[^\s'\"<>?#]+)(\?[^\s'\"<>#]*)?(#[^\s'\"<>]*)?", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(cookie|set-cookie|authorization|password|passwd|token|access_token|session(?:id)?)"
    r"(\s*[:=]\s*)([^\s;,&]+)")


# yt-dlp writes an unreadable Netscape cookie-file line verbatim (name and value) into this warning.
_COOKIE_FILE_ENTRY = re.compile(r"(?i)(skipping cookie file entry due to [^:\r\n]*:)[^\r\n]*")
# A Cookie header carries several "name=value" pairs separated by ";": drop the rest of the line.
_COOKIE_HEADER = re.compile(r"(?i)\b((?:set-)?cookie\s*:\s*)[^\r\n]*")


def sanitize_text(text: str) -> str:
    """診断文からURLのクエリ（署名・トークンを含み得る）と資格情報らしき値を落とす。"""
    text = _URL.sub(lambda m: sanitize_url(m.group(1)) + ("?…" if m.group(2) else ""), str(text))
    text = _COOKIE_FILE_ENTRY.sub(lambda m: m.group(1) + " …", text)
    text = _COOKIE_HEADER.sub(lambda m: m.group(1) + "…", text)
    return _SECRET_ASSIGNMENT.sub(lambda m: m.group(1) + m.group(2) + "…", text)


_SENSITIVE_QUERY_KEYS = frozenset({
    "token", "access_token", "auth", "authorization", "key", "api_key", "sig", "signature",
    "session", "sessionid", "password", "cookie", "expire", "expires", "policy",
})


def sanitize_url(url: str) -> str:
    """ユーザーが渡したURLのうち、秘匿値になり得るクエリだけを落とす。所在は保つ。"""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        # 不正なportやIPv6構文を診断文へ戻すと、userinfo等の秘匿値も戻ってしまう。
        return "[不正なURL]"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower() not in _SENSITIVE_QUERY_KEYS]
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), ""))


# --- 置き場所 ----------------------------------------------------------------

LIBRARY_DIR_NAME = "Downloads"


def library_root() -> Path:
    """取得した素材の置き場所。配備版ではアプリのフォルダ内の `Downloads`。

    状態領域（.state）の外に置く。分離結果は各サービスのフォルダの中の `LSS` に並ぶ。
    """
    override = os.environ.get(LIBRARY_ENV)
    if override:
        return Path(override)
    return app_root() / LIBRARY_DIR_NAME


def ledger_dir(state_root: Path | None = None) -> Path:
    return (Path(state_root) if state_root is not None else session.state_dir()) / "acquisitions"


def _key(logical_source: str) -> str:
    return hashlib.sha256(logical_source.encode("utf-8")).hexdigest()[:32]


def work_dir_for(logical_source: str) -> Path:
    """論理ソースごとに決まる作業領域。中断した部分ファイルを次回の取得が引き継ぐ。"""
    return session.work_dir() / (WORK_PREFIX + _key(logical_source))


def cleanup_stale_work(*, older_than_seconds: float = STALE_WORK_SECONDS,
                       now: float | None = None) -> list[Path]:
    cutoff = (time.time() if now is None else now) - older_than_seconds
    removed = []
    for entry in session.work_dir().iterdir():
        if not entry.name.startswith(WORK_PREFIX) or not entry.is_dir():
            continue
        try:
            newest = max([entry.stat().st_mtime] + [p.stat().st_mtime for p in entry.rglob("*")])
        except OSError:
            continue
        if newest < cutoff:
            shutil.rmtree(entry, ignore_errors=True)
            if not entry.exists():
                removed.append(entry)
    return removed


class WorkLock:
    """同じ論理ソースを同時に取得しない（部分ファイルの取り合いを防ぐ）。"""

    def __init__(self, directory: Path) -> None:
        self._path = directory / ".lock"
        self._stream = None

    def __enter__(self) -> "WorkLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(self._path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - 製品はWindowsのみ
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise AcquisitionFailure(FailureKind.BUSY, "同じソースを別の取得が処理中です") from exc
        self._stream = stream
        return self

    def __exit__(self, *_exc) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        stream.close()


# --- 台帳 --------------------------------------------------------------------

def _ledger_path(logical_source: str, state_root: Path | None = None) -> Path:
    return ledger_dir(state_root) / f"{_key(logical_source)}.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
            encoding="utf-8", newline="\n", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_revisions(logical_source: str, state_root: Path | None = None) -> list[dict]:
    path = _ledger_path(logical_source, state_root)
    try:
        return _read_ledger(path, logical_source)
    except FileNotFoundError:
        return []


def _read_ledger(path: Path, logical_source: str | None = None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or payload.get("schema") != RECORD_SCHEMA
            or not isinstance(payload.get("logical_source"), str) or not payload["logical_source"]
            or (logical_source is not None and payload["logical_source"] != logical_source)
            or path != _ledger_path(payload["logical_source"], path.parent.parent)
            or not isinstance(payload.get("revisions"), list)
            or any(not isinstance(record, dict) or record.get("logical_source") != payload["logical_source"]
                   for record in payload["revisions"])):
        raise ValueError(f"取得台帳のソース・版記録が不正です: {path}")
    return list(payload["revisions"])


def _append_revision(record: dict, state_root: Path | None = None) -> None:
    logical_source = record["logical_source"]
    revisions = load_revisions(logical_source, state_root)
    revisions.append(record)
    _write_json_atomic(_ledger_path(logical_source, state_root),
                       {"schema": RECORD_SCHEMA, "logical_source": logical_source,
                        "revisions": revisions})


def iter_records(state_root: Path | None = None, *, strict: bool = True):
    """保存済みの取得記録を読む。読取失敗は記録不在として扱わない。

    strict=Falseは任意の曲名等のタグ補完用。目録・再取得・録音同一性の判断には使わない。
    """
    directory = ledger_dir(state_root)
    try:
        paths = sorted(path for path in directory.iterdir() if path.suffix.lower() == ".json")
    except FileNotFoundError:
        return
    except OSError:
        if strict:
            raise
        return
    for path in paths:
        try:
            revisions = _read_ledger(path)
        except (OSError, ValueError):
            if strict:
                raise
            continue
        yield from revisions


def records_for_sha256(sha256: str, state_root: Path | None = None) -> list[dict]:
    """分離の完了台帳（input_sha256）から、取得元の来歴へたどる。"""
    return [r for r in iter_records(state_root) if r.get("content", {}).get("sha256") == sha256]


def file_sha256(path: Path) -> str:
    return session.input_sha256_of(path)


def _still_valid(record: dict) -> Path | None:
    content = record.get("content") or {}
    path = Path(content.get("path") or "")
    try:
        if not path.is_file() or path.stat().st_size != content.get("size_bytes"):
            return None
    except OSError:
        return None
    return path if file_sha256(path) == content.get("sha256") else None


def find_valid(logical_source: str, state_root: Path | None = None) -> tuple[Path, dict] | None:
    """最新の取得物が、記録どおりの内容で残っていれば返す。"""
    for record in reversed(load_revisions(logical_source, state_root)):
        path = _still_valid(record)
        if path is not None:
            return path, record
    return None


def find_valid_by_url(url: str, state_root: Path | None = None) -> tuple[Path, dict] | None:
    """所在（URL）からの再利用。通信できないときだけ使う補助で、同一性の根拠にはしない。"""
    wanted = sanitize_url(url)
    matches = [r for r in iter_records(state_root)
               if wanted in {(r.get("location") or {}).get("requested_url"),
                             (r.get("location") or {}).get("webpage_url")}]
    matches.sort(key=lambda r: r.get("acquired_at") or "")
    for record in reversed(matches):
        path = _still_valid(record)
        if path is not None:
            return path, record
    return None


# --- 公開 --------------------------------------------------------------------

_UNSAFE_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                       *(f"LPT{i}" for i in range(1, 10))})


def safe_name(text: str, fallback: str = "source") -> str:
    name = _UNSAFE_NAME.sub("_", str(text or "")).strip().rstrip(". ")
    name = re.sub(r"\s+", " ", name)[:_MAX_TITLE_CHARS].rstrip(". ")
    if not name or name.upper() in _RESERVED:
        name = fallback
    return name


def _copy_verified(staged: Path, directory: Path, sha256: str) -> Path:
    """配置先と同じボリューム上に一時ファイルを完成させてから名前を与える。"""
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix=".lss-acquire-", suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        try:
            os.replace(staged, temporary)
        except OSError:
            shutil.copyfile(staged, temporary)
        if file_sha256(temporary) != sha256:
            raise AcquisitionFailure(FailureKind.INTERRUPTED, "取得物の配置中に内容が変化しました")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def publish(staged: Path, *, record: dict, folder: str, stem: str,
            state_root: Path | None = None, library: Path | None = None) -> AcquiredSource:
    """作業領域の完成品をライブラリへ置き、台帳へ版を追記する。既存ファイルは上書きしない。"""
    staged = Path(staged)
    sha256 = file_sha256(staged)
    size = staged.stat().st_size
    if size <= 0:
        raise AcquisitionFailure(FailureKind.INVALID_MEDIA, "取得物が空です")
    directory = (library if library is not None else library_root()) / safe_name(folder, "other")
    suffix = staged.suffix.lower()
    base = safe_name(stem)

    def same_content(candidate: Path) -> bool:
        try:
            return candidate.stat().st_size == size and file_sha256(candidate) == sha256
        except OSError:
            return False

    names = [directory / (f"{base}{suffix}" if n == 1 else f"{base} ({n}){suffix}")
             for n in range(1, 1000)]
    destination = next((c for c in names if c.exists() and same_content(c)), None)
    if destination is not None:
        staged.unlink(missing_ok=True)
    else:
        temporary = _copy_verified(staged, directory, sha256)
        try:
            for candidate in names:
                try:
                    # 既存名では失敗するので、間に作られた同名ファイルも上書きしない。
                    os.link(temporary, candidate)
                except FileExistsError:
                    if same_content(candidate):
                        destination = candidate
                        break
                    continue
                except OSError:
                    # ハードリンクを持たないファイルシステム。排他作成してから中身を写す。
                    try:
                        with open(candidate, "xb") as target, open(temporary, "rb") as origin:
                            shutil.copyfileobj(origin, target, _CHUNK)
                    except FileExistsError:
                        continue
                    if not same_content(candidate):
                        candidate.unlink(missing_ok=True)
                        raise AcquisitionFailure(FailureKind.INTERRUPTED, "取得物の配置に失敗しました")
                destination = candidate
                break
            else:  # pragma: no cover
                raise AcquisitionFailure(FailureKind.EXTRACTOR_FAILED, "保存名を決められません")
        finally:
            temporary.unlink(missing_ok=True)

    completed = dict(record)
    completed["schema"] = RECORD_SCHEMA
    completed["acquired_at"] = completed.get("acquired_at") or datetime.now(timezone.utc).isoformat()
    completed["content"] = {"sha256": sha256, "size_bytes": size, "path": str(destination.resolve())}
    _append_revision(completed, state_root)
    return AcquiredSource(destination.resolve(), sha256, completed["logical_source"], False, completed)


def reuse(path: Path, record: dict) -> AcquiredSource:
    return AcquiredSource(Path(path), record["content"]["sha256"], record["logical_source"], True, record)


def inherited_tags_for(source_path: Path, state_root: Path | None = None) -> dict[str, str | list[str]]:
    """取得物に元のタグが無いとき、取得時に記録したソースの事実を継承タグとして返す。

    記録の内容SHA-256と一致するファイルだけを対象にする。投稿者名は曲のアーティストとは限らないため使わない。
    """
    path = Path(source_path)
    try:
        resolved = str(path.resolve())
        size = path.stat().st_size
    except OSError:
        return {}
    records = [r for r in iter_records(state_root, strict=False)
               if (r.get("content") or {}).get("path") == resolved
               and (r.get("content") or {}).get("size_bytes") == size]
    if not records:
        return {}
    digest = file_sha256(path)
    matching = [r for r in records if r["content"].get("sha256") == digest]
    if not matching:
        return {}
    metadata = max(matching, key=lambda r: r.get("acquired_at") or "").get("metadata") or {}
    tags: dict[str, str | list[str]] = {}
    title = metadata.get("track") or metadata.get("title")
    if isinstance(title, str) and title.strip():
        tags["TITLE"] = title.strip()
    artists = metadata.get("artists") or metadata.get("creators")
    if isinstance(artists, list) and artists:
        values = [str(item) for item in artists if str(item).strip()]
        if values:
            tags["ARTIST"] = values if len(values) > 1 else values[0]
    if isinstance(metadata.get("album"), str) and metadata["album"].strip():
        tags["ALBUM"] = metadata["album"].strip()
    release = metadata.get("release_date")
    if isinstance(release, str) and re.fullmatch(r"\d{8}", release):
        tags["DATE"] = f"{release[:4]}-{release[4:6]}-{release[6:]}"
    elif metadata.get("release_year"):
        tags["DATE"] = str(metadata["release_year"])
    return tags
