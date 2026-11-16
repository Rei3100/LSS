"""参照録音の知識（軽く、ずっと残す）と、重い音源（取得・保持・退役・取り直し）を分けて持つ。

契約は `docs/ACQUISITION.md`「Referenceの取得」、`docs/ASSETS_AND_PROVENANCE.md`、`docs/PRODUCT_SCRATCH_LIFECYCLE.md`。

- **参照の知識**（状態領域 `reference-intelligence/records/<内容SHA-256>.json`）：録音の同一性（内容SHA-256）、
  作品との結び付き、タグ・取得元の事実、長さ、取り直しの所在、指紋（`identity` の指紋cacheのファイル名と
  そのSHA-256）、照合した番組との関係の履歴。数KB〜。指紋（1曲0.1〜1MB）と合わせて、候補の取り出しは
  重い音源なしで行える。
- **重い音源**の寿命は種類で決まる。
  - 利用者が置いた音源（`References`）：利用者のもの。LSSは消さない（`user_media`）。
  - LSSが取得した音源で、取り直しの所在があるもの：再生成できる（`regenerable_reference_media`）。
    知識と指紋を保存し終え、使用中でなく、一定期間使われていなければ退役（削除）できる。波形の確認・
    再構成で要る時は、取得の台帳の所在から取り直し、内容SHA-256が同じことを確かめてから使う。
  - 取り直しの所在が無いもの：保持する（`retained_reference_media`）。
- 退役は「退役中」を書いてから消し、「退役済み」を書く。途中で止まっても次回に続きから終える。
  所有記録と一致しない・リンク・置き場所の外・使用中（OSのバイトロック）・内容が違うものは消さない。
- 重い音源が無い・取り直せない参照は、波形の確認が要る判定（EXACT）の根拠にしない（関係はUNKNOWNのまま）。
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Iterable

import numpy as np

from lss.bgm import identity

SCHEMA = 1
RECORD_TYPE = "lss_reference_intelligence"
STORE_DIR = "reference-intelligence"
MEDIA_DIR = "reference-media"
LEASE_DIR = ".reference-media-leases"

ORIGIN_ACQUIRED = "acquired"
ORIGIN_USER = "user_library"
CLASS_REGENERABLE = "regenerable_reference_media"
CLASS_RETAINED = "retained_reference_media"
CLASS_USER = "user_media"

STATE_PRESENT = "present"
STATE_RETIRING = "retiring"
STATE_RETIRED = "retired"

IDLE_GRACE_SECONDS = 7 * 24 * 60 * 60
HISTORY_LIMIT = 200


class ReferenceStoreError(ValueError):
    """参照の知識の記録が契約に合わない。"""


class ReferenceMediaUnavailable(RuntimeError):
    """波形の確認に要る重い音源が無く、取り直しもできない（判定の根拠にしない）。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state(state_root: Path | None) -> Path:
    from lss.core import session

    return Path(state_root) if state_root is not None else session.state_dir()


def store_root(state_root: Path | None = None) -> Path:
    return _state(state_root) / STORE_DIR


def media_root(state_root: Path | None = None) -> Path:
    """LSSが取得した参照音源の置き場所（Productの所有。利用者の `References` とは別）。"""
    return _state(state_root) / MEDIA_DIR


def _record_path(sha256: str, state_root: Path | None) -> Path:
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ReferenceStoreError(f"内容SHA-256が不正です: {sha256!r}")
    return store_root(state_root) / "records" / f"{sha256}.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(sha256: str, state_root: Path | None = None) -> dict | None:
    path = _record_path(sha256, state_root)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if (not isinstance(record, dict) or record.get("type") != RECORD_TYPE or record.get("schema") != SCHEMA
            or record.get("sha256") != sha256):
        raise ReferenceStoreError(f"参照の知識の記録が不正です: {path}")
    return record


def records(state_root: Path | None = None, *, work_key: str | None = None) -> list[dict]:
    """保存済みの参照の知識。`work_key` を与えると、その作品に結び付いたものと利用者の音源だけ。"""
    folder = store_root(state_root) / "records"
    found = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            record = load(path.stem, state_root)
        except (OSError, ValueError, ReferenceStoreError):
            continue
        if record is None or record.get("active") is False:
            continue
        if work_key is not None and record["origin"] != ORIGIN_USER and work_key not in record.get("work_keys", []):
            continue
        found.append(record)
    return found


def _event(record: dict, event: str, **details) -> None:
    history = record.setdefault("history", [])
    history.append({"at": _now(), "event": event, **details})
    del history[:-HISTORY_LIMIT]


def _media_class(origin: str, reacquire: dict | None) -> str:
    if origin == ORIGIN_USER:
        return CLASS_USER
    return CLASS_REGENERABLE if (reacquire or {}).get("url") else CLASS_RETAINED


def ingest(path: Path, *, cache_dir: Path, state_root: Path | None = None, origin: str = ORIGIN_ACQUIRED,
           work_keys: Iterable[str] = (), acquisition: dict | None = None, discovery: dict | None = None,
           loader=identity.decode_mono, sha256: str | None = None) -> dict:
    """手元にある参照音源から、参照の知識（指紋・長さ・タグ・取得元・取り直しの所在）を作って保存する。

    同じ内容がすでにあれば、作品の結び付きと所在を足すだけ。指紋は `identity` の指紋cacheへ保存する。
    """
    from lss.bgm import reference_relations

    if origin not in (ORIGIN_ACQUIRED, ORIGIN_USER):
        raise ReferenceStoreError(f"未対応の出所です: {origin}")
    path = Path(path).resolve()
    sha256 = sha256 or _file_sha256(path)
    cached = identity._load_fingerprint(cache_dir, sha256)
    if cached is None:
        index = identity.ReferenceIndex.build([path], loader=loader, cache_dir=cache_dir)
    else:
        index = identity.ReferenceIndex._from_fingerprints([(path.stem, str(path), sha256, cached)], loader)
    [reference] = index.references
    fingerprint = identity._fingerprint_path(cache_dir, sha256)
    record = load(sha256, state_root) or {
        "schema": SCHEMA, "type": RECORD_TYPE, "sha256": sha256, "origin": origin, "work_keys": [],
        "active": True, "created_at": _now(), "history": []}
    if record["origin"] == ORIGIN_ACQUIRED and origin == ORIGIN_USER:
        # 利用者も同じ内容を置いた：利用者の音源として扱う（LSSは消さない）。
        record["origin"] = ORIGIN_USER
    tags = reference_relations.reference_metadata(path)
    acquisition = acquisition or {}
    metadata = dict(acquisition.get("metadata") or {})
    record["metadata"] = {
        "display_title": tags.get("display_title") or metadata.get("track") or metadata.get("title"),
        "title_source": tags.get("title_source") or ("acquisition_metadata" if metadata else None),
        "tags": tags.get("tags") or {}, "file_name": path.name,
        "album": (tags.get("tags") or {}).get("album") or metadata.get("album"),
        "artists": metadata.get("artists") or metadata.get("creators"),
        "track_number": metadata.get("track_number"), "release_date": metadata.get("release_date"),
        "lss_output": bool(tags.get("lss_output")),
    }
    record["audio"] = {"duration_seconds": round(reference.frames * identity.FRAME_SECONDS, 3),
                       "bytes": path.stat().st_size, "container": path.suffix.lower().lstrip(".")}
    record["fingerprint"] = {"settings_id": identity.fingerprint_settings_id(), "file": fingerprint.name,
                             "sha256": _file_sha256(fingerprint), "bytes": fingerprint.stat().st_size}
    record["chroma"] = _ensure_chroma(sha256, cache_dir, lambda: index.samples(0))
    if acquisition:
        record["provenance"] = {
            "logical_source": acquisition.get("logical_source"), "authority": acquisition.get("authority"),
            "location": acquisition.get("location"), "tool": {k: (acquisition.get("tool") or {}).get(k)
                                                              for k in ("name", "version")},
            "format": acquisition.get("format"), "acquired_at": acquisition.get("acquired_at"),
            "auth": acquisition.get("auth")}
        record["reacquire"] = dict(acquisition.get("reacquire") or {})
        record["reacquire"]["logical_source"] = acquisition.get("logical_source")
    if discovery:
        record["discovery"] = discovery
    record["work_keys"] = sorted(set(record.get("work_keys", [])) | {k for k in work_keys if k})
    media = record.setdefault("media", {})
    media_class = _media_class(record["origin"], record.get("reacquire"))
    if media_class == CLASS_REGENERABLE and not _inside(path, media_root(state_root)):
        # 取得は再利用で、Productの参照音源の置き場所の外（取得ライブラリ等）にある：LSSの退役の対象にしない。
        media_class = CLASS_RETAINED
    media.update({"class": media_class, "state": STATE_PRESENT, "path": str(path)})
    media.setdefault("last_used_at", None)
    _event(record, "ingested", path=str(path), origin=origin)
    _write(_record_path(sha256, state_root), record)
    return record


def _ensure_chroma(sha256: str, cache_dir: Path, samples) -> dict:
    """和声の並び（別版の候補の判定）を、`ReferenceLibrary.chroma` と同じ名前で保存する。"""
    from lss.bgm import versions

    path = Path(cache_dir) / f"{sha256}-chroma-v{versions.METHOD_VERSION}.npy"
    try:
        np.load(path)
    except (OSError, ValueError):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.npy")
        np.save(temporary, versions.chroma(samples()))
        temporary.replace(path)
    return {"method_version": versions.METHOD_VERSION, "file": path.name, "sha256": _file_sha256(path)}


def link_work(sha256: str, work_key: str, state_root: Path | None = None, *, release: dict | None = None) -> None:
    """作品へ結び付ける。`release` は確かめた収録盤（盤の所在・題）。知識の記録へ残す。"""
    record = load(sha256, state_root)
    if record is None or not work_key:
        return
    changed = False
    if work_key not in record.get("work_keys", []):
        record["work_keys"] = sorted(set(record["work_keys"]) | {work_key})
        _event(record, "linked_work", work_key=work_key)
        changed = True
    if release and (record.get("discovery") or {}).get("release") != release:
        record.setdefault("discovery", {})["release"] = dict(release)
        changed = True
    if changed:
        _write(_record_path(sha256, state_root), record)


# --- 重い音源 --------------------------------------------------------------------

def _plain_file(path: Path) -> os.stat_result:
    value = path.lstat()
    if (not stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode)
            or getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or value.st_nlink != 1):
        raise ReferenceStoreError(f"通常のファイルではない参照音源は扱いません: {path}")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
        return True
    except ValueError:
        return False


def present_media(record: dict) -> Path | None:
    """記録どおりの内容の重い音源があれば、その所在。"""
    media = record.get("media") or {}
    if media.get("state") != STATE_PRESENT or not media.get("path"):
        return None
    path = Path(media["path"])
    try:
        if not path.is_file() or path.stat().st_size != (record.get("audio") or {}).get("bytes"):
            return None
        return path if _file_sha256(path) == record["sha256"] else None
    except OSError:
        return None


@dataclass
class MediaResolver:
    """参照の内容SHA-256から、波形の確認に使う重い音源を返す（無ければ取り直す）。

    `strict=False` では、重い音源が無く取り直せない参照を、長さだけ合わせた無音として返し、`unavailable` に
    記録する（照合は指紋の候補までで止まり、波形の確認が成り立たないので `known_exact` にも EXACT にもならない）。
    """

    state_root: Path | None = None
    reacquire: bool = True
    acquire_options: dict | None = None
    fetched: list | None = None
    strict: bool = True
    frames: dict = field(default_factory=dict)
    unavailable: dict = field(default_factory=dict)

    def path(self, sha256: str) -> Path:
        return ensure_media(sha256, state_root=self.state_root, reacquire=self.reacquire,
                            fetched=self.fetched, **(self.acquire_options or {}))

    def __call__(self, location) -> np.ndarray:
        sha256 = str(location)
        if sha256 in self.unavailable and not self.strict:
            return np.zeros(int(self.frames.get(sha256, 0)) * identity.HOP, np.float32)
        try:
            return identity.decode_mono(self.path(sha256))
        except (ReferenceMediaUnavailable, identity.DecodeError, OSError) as exc:
            if self.strict:
                raise
            self.unavailable[sha256] = f"{type(exc).__name__}: {exc}"
            return np.zeros(int(self.frames.get(sha256, 0)) * identity.HOP, np.float32)


def ensure_media(sha256: str, *, state_root: Path | None = None, reacquire: bool = True,
                 fetched: list | None = None, **acquire_options) -> Path:
    """重い音源の所在。退役済みで取り直しの所在があれば、取得の台帳から取り直して同じ内容か確かめる。

    取り直した内容が違えば（配信側の再符号化など）、新しい内容は別の参照として知識へ入れ、
    この内容SHA-256の音源としては使わない。
    """
    from lss.acquisition import service

    record = load(sha256, state_root)
    if record is None:
        raise ReferenceMediaUnavailable(f"参照の知識がありません: {sha256}")
    found = present_media(record)
    if found is not None:
        return found
    if record["origin"] == ORIGIN_USER:
        raise ReferenceMediaUnavailable("利用者の参照音源が置き場所にありません")
    plan = record.get("reacquire") or {}
    if not reacquire or not plan.get("url"):
        raise ReferenceMediaUnavailable("参照音源は退役済みで、取り直しの所在がありません")
    started = time.perf_counter()
    from lss.acquisition import courtesy
    from lss.bgm import reference_acquisition

    # 取り直しも取得先への礼儀（順番・間隔・混雑の後の休み）を守る。休みの間は取り直さない（波形の確認が
    # 要る判定は根拠なしのまま＝何も消さない）。
    pacer = acquire_options.pop("courtesy", None) or courtesy.shared(state_root)
    acquire_options.setdefault("pacing", reference_acquisition.reference_pacing())
    try:
        with pacer.slot(plan["url"], purpose="reacquire"):
            outcome = service.reacquire(sha256, state_root=state_root, library=media_root(state_root),
                                        **acquire_options)
    except courtesy.Deferred as exc:
        raise ReferenceMediaUnavailable(f"取得先が休みの間です（{exc.seconds:.0f}秒）") from exc
    except (KeyError, ValueError, OSError) as exc:
        raise ReferenceMediaUnavailable(f"参照音源を取り直せません: {type(exc).__name__}") from exc
    failure = getattr(getattr(outcome.outcome, "failure", None), "kind", None)
    pacer.report(plan["url"], reference_acquisition._signal(failure))
    source = outcome.outcome.source
    record = load(sha256, state_root)
    if source is None:
        kind = getattr(outcome.outcome.failure, "kind", None)
        _event(record, "reacquire_failed", failure=getattr(kind, "value", str(kind)))
        _write(_record_path(sha256, state_root), record)
        raise ReferenceMediaUnavailable("参照音源の取り直しに失敗しました")
    if not outcome.identical:
        _event(record, "reacquired_different_content", new_sha256=source.sha256)
        _write(_record_path(sha256, state_root), record)
        raise ReferenceMediaUnavailable("取り直した参照音源の内容が以前と違います")
    record["media"].update(state=STATE_PRESENT, path=str(Path(source.path).resolve()))
    _event(record, "reacquired", seconds=round(time.perf_counter() - started, 2), reused=bool(source.reused))
    _write(_record_path(sha256, state_root), record)
    if fetched is not None:
        fetched.append(sha256)
    return Path(source.path)


def _lease_path(sha256: str, state_root: Path | None) -> Path:
    return _state(state_root) / LEASE_DIR / f"{sha256}.lock"


@contextmanager
def _byte_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - 製品はWindows
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def lease(sha256s: Iterable[str], state_root: Path | None = None, *, wait_seconds: float = 30.0):
    """使用中の参照音源を退役させない（OSのバイトロック。異常終了で自動に外れる）。終わりに最終使用を記録する。"""
    wanted = sorted(set(sha256s))
    with ExitStack() as stack:
        for sha256 in wanted:
            deadline = time.monotonic() + wait_seconds
            while True:
                try:
                    stack.enter_context(_byte_lock(_lease_path(sha256, state_root)))
                    break
                except OSError:
                    # 別processの退役の確認中。少し待つ（退役側は使用中を見ると消さない）。
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
        try:
            yield
        finally:
            stamp = _now()
            for sha256 in wanted:
                try:
                    record = load(sha256, state_root)
                    if record is not None:
                        record.setdefault("media", {})["last_used_at"] = stamp
                        _write(_record_path(sha256, state_root), record)
                except (OSError, ValueError, ReferenceStoreError):
                    pass


def _regenerable(record: dict, cache_dir: Path, state_root: Path | None) -> str | None:
    """退役してよいか。よくなければその理由。"""
    from lss.acquisition import store as acquisition_store

    media = record.get("media") or {}
    if record.get("origin") != ORIGIN_ACQUIRED:
        return "user_media"
    if media.get("class") != CLASS_REGENERABLE:
        return "no_reacquisition_path"
    if media.get("pinned"):
        return "pinned"
    plan = record.get("reacquire") or {}
    if not plan.get("url") or not plan.get("logical_source"):
        return "no_reacquisition_path"
    if not any((r.get("content") or {}).get("sha256") == record["sha256"]
               for r in acquisition_store.load_revisions(plan["logical_source"], state_root)):
        return "acquisition_ledger_missing"
    fingerprint = record.get("fingerprint") or {}
    path = Path(cache_dir) / str(fingerprint.get("file") or "")
    try:
        if (fingerprint.get("settings_id") != identity.fingerprint_settings_id()
                or _file_sha256(path) != fingerprint.get("sha256")):
            return "compact_intelligence_not_current"
    except OSError:
        return "compact_intelligence_missing"
    chroma = record.get("chroma") or {}
    try:
        if _file_sha256(Path(cache_dir) / str(chroma.get("file") or "")) != chroma.get("sha256"):
            return "compact_intelligence_not_current"
    except OSError:
        return "compact_intelligence_missing"
    if not _inside(Path(media.get("path") or ""), media_root(state_root)):
        return "outside_product_media_root"
    return None


def retire_idle(*, cache_dir: Path, state_root: Path | None = None, grace_seconds: float = IDLE_GRACE_SECONDS,
                now: float | None = None, dry_run: bool = False, only: set | None = None) -> dict:
    """再生成できる重い参照音源のうち、知識を保存済みで、使用中でなく、一定期間使われていないものを退役させる。

    返り値は機械可読の在庫（退役・保持とその理由・回収したバイト数）。
    """
    now = time.time() if now is None else now
    root = media_root(state_root)
    report = {"schema": SCHEMA, "type": "lss_reference_media_retirement", "dry_run": dry_run,
              "media_root": str(root), "grace_seconds": grace_seconds, "items": [],
              "retired": [], "reclaimed_bytes": 0}
    if True:
        for record in records(state_root):
            if only is not None and record["sha256"] not in only:
                continue
            media = record.get("media") or {}
            row = {"sha256": record["sha256"], "class": media.get("class"), "state": media.get("state"),
                   "path": media.get("path")}
            report["items"].append(row)
            if media.get("state") == STATE_RETIRED:
                row["decision"] = "already_retired"
                continue
            reason = _regenerable(record, cache_dir, state_root)
            if reason is not None:
                row["decision"] = f"keep_{reason}"
                continue
            last = media.get("last_used_at") or record.get("created_at")
            try:
                idle = now - datetime.fromisoformat(last).timestamp()
            except (TypeError, ValueError):
                idle = 0.0
            if media.get("state") != STATE_RETIRING and idle < grace_seconds:
                row["decision"] = "keep_recently_used"
                continue
            path = Path(media.get("path") or "")
            try:
                with _byte_lock(_lease_path(record["sha256"], state_root)):
                    latest = load(record["sha256"], state_root)
                    if (latest or {}).get("media", {}).get("path") != media.get("path"):
                        raise ReferenceStoreError("確認中に所在が変わりました")
                    if path.exists() or path.is_symlink():
                        if not _inside(path, root):
                            raise ReferenceStoreError("Productの参照音源の置き場所の外です")
                        _plain_file(path)
                        size = path.stat().st_size
                        if _file_sha256(path) != record["sha256"]:
                            raise ReferenceStoreError("内容が知識の記録と違います")
                    else:
                        size = 0
                    row.update(size_bytes=size)
                    if dry_run:
                        row["decision"] = "would_retire"
                        continue
                    latest["media"]["state"] = STATE_RETIRING
                    _write(_record_path(record["sha256"], state_root), latest)
                    if size:
                        path.unlink()
                    latest["media"].update(state=STATE_RETIRED, path=None, retired_at=_now(),
                                           reclaimed_bytes=size)
                    _event(latest, "retired", bytes=size, idle_seconds=round(idle, 1))
                    _write(_record_path(record["sha256"], state_root), latest)
                    row["decision"] = "retired"
                    report["retired"].append(record["sha256"])
                    report["reclaimed_bytes"] += size
            except OSError:
                row["decision"] = "keep_in_use_or_io_blocked"
            except ReferenceStoreError as exc:
                row["decision"] = "keep_unsafe"
                row["detail"] = str(exc)
    return report


def retire_now(sha256s: Iterable[str], *, cache_dir: Path, state_root: Path | None = None,
               reason: str) -> list[str]:
    """作品に結び付かない（使わない）取得済みの参照音源を、猶予を待たずに退役させる。安全の確認は同じ。"""
    targets = []
    for sha256 in sorted(set(sha256s)):
        record = load(sha256, state_root)
        if record is None or record.get("work_keys") or record.get("origin") != ORIGIN_ACQUIRED:
            continue
        record["media"]["last_used_at"] = "1970-01-01T00:00:00+00:00"
        _event(record, "retire_requested", reason=reason)
        _write(_record_path(sha256, state_root), record)
        targets.append(sha256)
    if not targets:
        return []
    report = retire_idle(cache_dir=cache_dir, state_root=state_root, only=set(targets))
    return report["retired"]


def pin(sha256: str, reason: str, state_root: Path | None = None) -> None:
    """重い音源を退役させない理由を付ける（例：再構成した資産の作り直しに要る）。"""
    record = load(sha256, state_root)
    if record is None:
        raise ReferenceStoreError(f"参照の知識がありません: {sha256}")
    pins = set(record["media"].get("pinned") or [])
    if reason not in pins:
        record["media"]["pinned"] = sorted(pins | {reason})
        _event(record, "pinned", reason=reason)
        _write(_record_path(sha256, state_root), record)


def note_relations(sha256s: Iterable[str], *, source_sha256: str, work_key: str | None, summary: dict,
                   state_root: Path | None = None) -> None:
    """照合した番組と、その参照との関係の要約を履歴へ残す（照合の結果の記憶）。"""
    for sha256 in sorted(set(sha256s)):
        record = load(sha256, state_root)
        if record is None:
            continue
        found = summary.get(sha256) or {}
        verification = record.setdefault("verification", {"matched_sources": [], "checked_sources": []})
        if source_sha256 not in verification["checked_sources"]:
            verification["checked_sources"].append(source_sha256)
        if found and source_sha256 not in verification["matched_sources"]:
            verification["matched_sources"].append(source_sha256)
        verification["state"] = "matched" if verification["matched_sources"] else "not_matched_yet"
        if found:
            _event(record, "related", source_sha256=source_sha256, work_key=work_key, relations=found)
        _write(_record_path(sha256, state_root), record)


# --- 照合の索引 ------------------------------------------------------------------

def build_library(entries: list[dict], *, cache_dir: Path, state_root: Path | None = None,
                  resolver: MediaResolver | None = None):
    """参照の知識から、照合の索引を作る（指紋は保存物から読み、重い音源は波形の確認の時だけ）。

    指紋が今の設定で無い参照は、重い音源（無ければ取り直し）から作り直す。作り直せない参照は外し、
    外した理由を返す。返り値は（`reference_relations.ReferenceLibrary`, 外した参照）。
    """
    from lss.bgm import reference_relations, release_availability

    resolver = resolver or MediaResolver(state_root=state_root)
    items, metadata, skipped = [], {}, []
    for record in sorted(entries, key=lambda r: r["sha256"]):
        sha256 = record["sha256"]
        cached = identity._load_fingerprint(cache_dir, sha256)
        if cached is None:
            try:
                samples = identity.decode_mono(resolver.path(sha256))
            except (ReferenceMediaUnavailable, identity.DecodeError, OSError) as exc:
                skipped.append({"sha256": sha256, "reason": "fingerprint_unavailable", "detail": str(exc)})
                continue
            spec = identity.spectrogram(samples)
            hashes, times = identity.landmarks(*identity.peaks(spec))
            cached = (hashes, times, identity.band_profile(spec), spec.shape[0])
            identity._save_fingerprint(cache_dir, sha256, cached)
        name = (record.get("metadata") or {}).get("file_name") or sha256
        items.append((Path(name).stem, sha256, sha256, cached))
        resolver.frames[sha256] = int(cached[3])
        meta = record.get("metadata") or {}
        metadata[sha256] = {"tags": meta.get("tags") or {}, "display_title": meta.get("display_title"),
                            "title_source": meta.get("title_source"), "file_name": meta.get("file_name"),
                            "independent_distribution": (meta.get("independent_distribution")
                                                         or release_availability.from_record(record)
                                                         or release_availability.user_copy(record))}
    index = identity.ReferenceIndex._from_fingerprints(items, resolver)
    durations = {reference.sha256: round(reference.frames * identity.FRAME_SECONDS, 3)
                 for reference in index.references}
    return reference_relations.ReferenceLibrary(index, metadata, durations, Path(cache_dir)), skipped


def sync_user_media(paths_and_sha: Iterable[tuple[Path, str]], *, cache_dir: Path,
                    state_root: Path | None = None) -> list[dict]:
    """利用者の参照音源（置き場所・登録済みの集まり）の知識を揃える。記録と所在・大きさが同じなら読み直さない。"""
    found = []
    for path, sha256 in paths_and_sha:
        path = Path(path).resolve()
        record = None
        try:
            record = load(sha256, state_root)
        except ReferenceStoreError:
            record = None
        current = (record is not None and record.get("origin") == ORIGIN_USER
                   and (record.get("media") or {}).get("path") == str(path)
                   and (record.get("audio") or {}).get("bytes") == path.stat().st_size
                   and identity._fingerprint_path(cache_dir, sha256).is_file()
                   and (record.get("chroma") or {}).get("file")
                   and (Path(cache_dir) / record["chroma"]["file"]).is_file())
        if not current:
            record = ingest(path, cache_dir=cache_dir, state_root=state_root, origin=ORIGIN_USER, sha256=sha256)
        found.append(record)
    return found
