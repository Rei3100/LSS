"""作品ごとの参照録音の集まり（リリース音源・OP/EDなど）を、出所とともに持ち、一致の照合へ渡す。

参照録音は外部の証拠。手元のファイルでも、URLから取得したもの（取得の来歴つき）でもよい。

- 集まりは状態領域の `reference_sets/<識別子>.json`。入れた録音の内容SHA-256・場所・出所を持つ。
- 使うときは内容SHA-256で確かめる。場所から消えていれば取得の来歴から同じ内容の取得物を探し、
  それも無ければ見つからない録音として返す（取り直し先があれば添える）。黙って外さない。
- 何も削除しない。集まりから外すのは明示の操作だけ。

通常の使い方は参照録音の置き場所（`library_root`、配備版ではアプリのフォルダ内の `References`）。利用者は
手元の既発売の音源（フォルダごとでよい）を一度置くだけで、BGMの処理のたびにLSSが中を見て使う。話ごとの
登録・cueごとの分類は要らない。置き場所の中身の同一性は内容SHA-256だけで決め、名前・タグ・フォルダの構成は
同一性に使わない。状態領域の `reference_library.json` は所在・大きさ・更新時刻から内容SHA-256を引く
cache（消えても読み直して作り直せる）。参照録音が無くても単話のBGMは従来どおり処理する。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lss.core import session
from lss.paths import app_root

DIRECTORY = "reference_sets"
SCHEMA = 1
_SET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
LIBRARY_DIR_NAME = "References"
LIBRARY_ENV = "LSS_REFERENCES_DIR"
LIBRARY_INDEX = "reference_library.json"
LIBRARY_INDEX_TYPE = "lss_reference_library_index"
LIBRARY_POLICY = "reference-library-folder-v1"


class ReferenceSetError(ValueError):
    """参照録音の集まりを扱えない。"""


def _path(set_id: str, state_root: Path | None) -> Path:
    if not _SET_ID.match(set_id or ""):
        raise ReferenceSetError(f"集まりの識別子が不正です（英数字と_.-）: {set_id!r}")
    return (Path(state_root) if state_root is not None else session.state_dir()) / DIRECTORY / f"{set_id}.json"


def load(set_id: str, state_root: Path | None = None) -> dict:
    path = _path(set_id, state_root)
    if not path.is_file():
        return {"schema": SCHEMA, "set_id": set_id, "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("set_id") != set_id:
        raise ReferenceSetError(f"参照録音の集まりの記録が不正です: {path}")
    return payload


def _save(payload: dict, state_root: Path | None) -> None:
    path = _path(payload["set_id"], state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _add(set_id: str, items: list[tuple[Path, str, dict]], state_root: Path | None) -> list[dict]:
    payload = load(set_id, state_root)
    known = {entry["sha256"] for entry in payload["entries"]}
    added = []
    now = datetime.now(timezone.utc).isoformat()
    for path, sha256, origin in items:
        if sha256 in known:
            continue
        entry = {"sha256": sha256, "path": str(Path(path).resolve()), "origin": origin, "added_at": now}
        payload["entries"].append(entry)
        known.add(sha256)
        added.append(entry)
    if added:
        _save(payload, state_root)
    return added


def add_local(set_id: str, paths: list[Path], state_root: Path | None = None) -> list[dict]:
    """手元のファイル・フォルダの音声を集まりへ入れる。同じ内容は重ねて入れない。"""
    files = session.iter_input_files([Path(p) for p in paths])
    return _add(set_id, [(f, session.input_sha256_of(f), {"kind": "local"}) for f in files], state_root)


def add_acquired(set_id: str, url: str, *, state_root: Path | None = None, **acquire_options) -> tuple[list[dict], list]:
    """URLから取得した録音を集まりへ入れる。取得できなかった項目は失敗として返す（集まりには入れない）。"""
    from lss.acquisition import service

    outcomes = service.acquire(url, state_root=state_root, **acquire_options)
    items = []
    for outcome in outcomes:
        if not outcome.ok:
            continue
        record = outcome.source.record
        items.append((outcome.source.path, outcome.source.sha256, {
            "kind": "acquisition", "logical_source": outcome.source.logical_source,
            "reacquire_url": (record.get("reacquire") or {}).get("url"),
        }))
    return _add(set_id, items, state_root), [outcome for outcome in outcomes if not outcome.ok]


@dataclass
class Resolution:
    paths: list[Path] = field(default_factory=list)
    origins: dict[str, dict] = field(default_factory=dict)  # 内容SHA-256 → 出所
    missing: list[dict] = field(default_factory=list)


def resolve(set_id: str, state_root: Path | None = None) -> Resolution:
    """集まりの録音を、内容SHA-256が一致するファイルとして返す。"""
    from lss.acquisition import store

    result = Resolution()
    for entry in load(set_id, state_root)["entries"]:
        sha256 = entry["sha256"]
        found = None
        path = Path(entry["path"])
        try:
            if path.is_file() and session.input_sha256_of(path) == sha256:
                found = path
        except OSError:
            found = None
        if found is None:
            for record in store.records_for_sha256(sha256, state_root):
                candidate = store._still_valid(record)
                if candidate is not None:
                    found = candidate
                    break
        if found is None:
            result.missing.append({"sha256": sha256, "path": entry["path"],
                                   "reacquire_url": (entry.get("origin") or {}).get("reacquire_url")})
            continue
        result.paths.append(found)
        result.origins[sha256] = entry.get("origin") or {}
    return result


def library_root() -> Path:
    """参照録音の置き場所。配備版ではアプリのフォルダ内の `References`（配備の入れ替えでも残る利用者のデータ）。"""
    override = os.environ.get(LIBRARY_ENV)
    if override:
        return Path(override)
    return app_root() / LIBRARY_DIR_NAME


def ensure_library_root() -> Path | None:
    """置き場所が無ければ空のフォルダを作る（利用者が場所を見つけられるように）。作れなくても処理は止めない。"""
    root = library_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return root


@dataclass
class LibraryScan:
    """置き場所の中の参照録音。``paths`` は内容ごとに1つ（同じ内容の複製は最初の所在だけ）、内容SHA-256の順。"""

    root: Path
    paths: list[Path] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)      # 所在 → 内容SHA-256
    duplicates: list[dict] = field(default_factory=list)       # 同じ内容の別の所在（使わない）
    unreadable: list[str] = field(default_factory=list)       # 読めなかった所在（使わない。次回また見る）
    index_rebuilt: bool = False

    @property
    def available(self) -> bool:
        return bool(self.paths)

    def summary(self) -> dict:
        return {"policy": LIBRARY_POLICY, "root": str(self.root), "count": len(self.paths),
                "duplicates": len(self.duplicates), "unreadable": len(self.unreadable)}


def _index_path(state_root: Path | None) -> Path:
    return (Path(state_root) if state_root is not None else session.state_dir()) / LIBRARY_INDEX


def _load_index(path: Path, root: Path) -> dict[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("type") != LIBRARY_INDEX_TYPE or payload.get("schema") != SCHEMA
                or payload.get("root") != str(root)):
            return {}
        return {str(entry["path"]): entry for entry in payload.get("entries") or []
                if isinstance(entry, dict) and isinstance(entry.get("sha256"), str)}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def scan_library(state_root: Path | None = None, root: Path | None = None) -> LibraryScan:
    """置き場所の音声を見て、内容SHA-256で確かめた参照録音を返す。置き場所が無い・空なら何も無い。

    所在・大きさ・更新時刻が前回と同じファイルは、状態領域の索引から内容SHA-256を引く（読み直さない）。
    索引が無い・壊れている・置き場所が変わった時は全部読み直して作り直す。結果は所在の名前に依らず
    内容SHA-256の順に並べる（同じ中身なら同じ順・同じ識別子）。
    """
    root = (Path(root) if root is not None else library_root()).resolve()
    scan = LibraryScan(root=root)
    if not root.is_dir():
        return scan
    index_path = _index_path(state_root)
    known = _load_index(index_path, root)
    scan.index_rebuilt = not known
    entries, changed = [], False
    for path in session.iter_input_files([root]):
        try:
            stat = path.stat()
            entry = known.get(str(path))
            if entry is None or entry.get("bytes") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns:
                entry = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                         "sha256": session.input_sha256_of(path)}
                changed = True
        except OSError:
            scan.unreadable.append(str(path))
            continue
        entries.append(entry)
    if changed or len(entries) != len(known):
        payload = {"type": LIBRARY_INDEX_TYPE, "schema": SCHEMA, "policy": LIBRARY_POLICY, "root": str(root),
                   "entries": sorted(entries, key=lambda item: item["path"])}
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            # 画面の照会とワーカーが同時に書いても互いの途中の内容を置かない（process毎の一時名）。
            temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            temporary.replace(index_path)
        except OSError:
            pass  # 索引はcache。書けなくても次回読み直す
    first: dict[str, str] = {}
    for entry in sorted(entries, key=lambda item: (item["sha256"], item["path"])):
        scan.sha256[entry["path"]] = entry["sha256"]
        if entry["sha256"] in first:
            scan.duplicates.append({"path": entry["path"], "sha256": entry["sha256"], "same_as": first[entry["sha256"]]})
            continue
        first[entry["sha256"]] = entry["path"]
        scan.paths.append(Path(entry["path"]))
    return scan
