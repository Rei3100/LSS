"""保存状態（台帳・キュー・公開の途中記録・取得の来歴）のバックアップと復元。

- 対象は小さな記録だけ。分離結果・原本・モデル・取得ツールの実行物・作業領域・キャッシュは含めない
  （それぞれ内容のSHA-256で確かめて取り直せる、または作り直せる）。
- バックアップはアプリのフォルダ内の `Backups\\` に、各ファイルのSHA-256を持つ目録付きのzipで作る。
  作成中にファイルが置き換わったら読み直し、目録と中身が一致したものだけを完成とする。
- 配備版は常駐ワーカーの起動時に自動で取る（`ensure_recent_backup`）。前回から一定時間が
  経っていて、記録が前回から変わっているときだけ作り、自動で作ったものだけを新しい順に
  一定数残す。手動のバックアップと復元前の退避は消さない。
- 復元はLSSが動いていないときだけ行う。目録を検証してから、今の記録を `Backups\\before-restore-*` へ
  退避し、置き換える。途中で失敗したら退避した記録へ戻す。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from lss.core import session
from lss.paths import app_root

MANIFEST = "lss-backup-manifest.json"
SCHEMA = 1
BACKUP_DIR_NAME = "Backups"
# アプリのフォルダからの相対パス。
STATE_FILES = ("completed_recipes.tsv", "queue.json", "relocations.json")
# ログインのプロファイル（Cookie）は資格情報に当たるので含めない。
STATE_TREES = ("publications", "acquisitions", "reference_sets", "recipes")
PROVIDER_STATES = ("providers/yt-dlp/state.json", "providers/deno/state.json")
APP_FILES = ("ledger/models.tsv", "ui_state.json")
# 自動で作るバックアップの名前。手動のもの（lss-state-）と区別し、自動分だけを整理する。
AUTOMATIC_PREFIX = "lss-state-auto-"
AUTOMATIC_INTERVAL_SECONDS = 24 * 3600
AUTOMATIC_KEEP = 14


class BackupError(RuntimeError):
    """バックアップまたは復元を安全に完了できない。"""


def backup_root() -> Path:
    return app_root() / BACKUP_DIR_NAME


def _members(state_root: Path, root: Path) -> list[tuple[str, Path]]:
    """（zip内の名前, 実ファイル）。状態領域は `state/`、アプリのフォルダ直下は `app/` に置く。"""
    members = []
    for name in (*STATE_FILES, *PROVIDER_STATES):
        path = state_root / name
        if path.is_file():
            members.append((f"state/{name}", path))
    for tree in STATE_TREES:
        folder = state_root / tree
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and not path.name.endswith(".tmp"):
                    members.append((f"state/{path.relative_to(state_root).as_posix()}", path))
    for name in APP_FILES:
        path = root / name
        if path.is_file():
            members.append((f"app/{name}", path))
    return members


def _read_stable(path: Path, attempts: int = 5) -> bytes:
    """読んでいる間に置き換わっていないことを、前後の状態で確かめて読む。"""
    for _ in range(attempts):
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size) == (after.st_mtime_ns, len(data)):
            return data
    raise BackupError(f"記録が更新され続けていて読めません: {path}")


def create_backup(*, state_root: Path | None = None, root: Path | None = None, destination: Path | None = None,
                  now: datetime | None = None, prefix: str = "lss-state-") -> Path:
    state_root = Path(state_root) if state_root is not None else session.state_dir()
    root = Path(root) if root is not None else app_root()
    destination = Path(destination) if destination is not None else root / BACKUP_DIR_NAME
    moment = now or datetime.now(timezone.utc)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{prefix}{moment.strftime('%Y%m%dT%H%M%SZ')}.zip"
    if target.exists():
        raise BackupError(f"同じ時刻のバックアップが既にあります: {target}")
    manifest = {"schema": SCHEMA, "created_at": moment.isoformat(), "files": []}
    with NamedTemporaryFile(dir=destination, prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, path in _members(state_root, root):
                data = _read_stable(path)
                bundle.writestr(name, data)
                manifest["files"].append({"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            bundle.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        verify_backup(temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        try:
            # Windowsのrenameは既存の完成バックアップを上書きしない。
            temporary.rename(target)
        except FileExistsError as exc:
            raise BackupError(f"同じ時刻のバックアップが既にあります: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _automatic_backups(destination: Path) -> list[tuple[datetime, Path]]:
    """自動で作ったバックアップを古い順に。名前の時刻が読めないものは数えない。"""
    found = []
    for path in destination.glob(f"{AUTOMATIC_PREFIX}*.zip") if destination.is_dir() else ():
        stamp = path.stem[len(AUTOMATIC_PREFIX):]
        try:
            moment = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        found.append((moment, path))
    return sorted(found)


def ensure_recent_backup(*, state_root: Path | None = None, root: Path | None = None,
                         now: datetime | None = None,
                         interval_seconds: float = AUTOMATIC_INTERVAL_SECONDS,
                         keep: int = AUTOMATIC_KEEP) -> Path | None:
    """前回の自動バックアップから時間が経ち、記録が変わっていれば新しく作る。

    作ったバックアップのパスを返す。作る必要が無ければNone。自動で作ったものだけを
    新しい順に `keep` 件残し、手動のバックアップと復元前の退避には触れない。
    """
    if keep < 1:
        raise ValueError("残す件数は1以上である必要があります")
    state_root = Path(state_root) if state_root is not None else session.state_dir()
    root = Path(root) if root is not None else app_root()
    destination = root / BACKUP_DIR_NAME
    moment = now or datetime.now(timezone.utc)
    members = _members(state_root, root)
    if not any(name.startswith("state/") for name, _ in members):
        return None  # まだ何も処理していない。守る記録が無い
    existing = _automatic_backups(destination)
    created = None
    latest_manifest = None
    if existing:
        try:
            latest_manifest = verify_backup(existing[-1][1])
        except BackupError:
            latest_manifest = None  # 壊れた最新分は比較に使わず、新しく作る
    recent = (latest_manifest is not None
              and (moment - existing[-1][0]).total_seconds() < interval_seconds)
    if not recent:
        current = {name: hashlib.sha256(_read_stable(path)).hexdigest() for name, path in members}
        previous = ({item["name"]: item["sha256"] for item in latest_manifest["files"]}
                    if latest_manifest is not None else None)
        if current != previous:
            created = create_backup(state_root=state_root, root=root, now=moment,
                                    prefix=AUTOMATIC_PREFIX)
    for _, path in _automatic_backups(destination)[:-keep]:
        path.unlink(missing_ok=True)
    return created


def _member_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    parts = name.split("/")
    if len(parts) < 2 or parts[0] not in {"state", "app"}:
        return False
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if any(not part or part in {".", ".."} or part.endswith((".", " "))
           or any(ord(char) < 32 or char in '<>:"\\|?*' for char in part)
           or part.split(".", 1)[0].casefold() in reserved for part in parts):
        return False
    relative = "/".join(parts[1:])
    if parts[0] == "app":
        return relative in APP_FILES
    return relative in (*STATE_FILES, *PROVIDER_STATES) or (
        len(parts) > 2 and parts[1] in STATE_TREES and not parts[-1].endswith(".tmp"))


def _read_verified_backup(path: Path) -> tuple[dict, dict[str, bytes]]:
    """検証した同じバイト列を返す。復元時にZIPを開き直さない。"""
    try:
        with zipfile.ZipFile(path) as bundle:
            entries = bundle.namelist()
            if len(entries) != len(set(entries)) or entries.count(MANIFEST) != 1:
                raise BackupError("バックアップに重複した記録があります")
            manifest = json.loads(bundle.read(MANIFEST))
            if not isinstance(manifest, dict) or type(manifest.get("schema")) is not int or manifest["schema"] != SCHEMA:
                raise BackupError("未対応のバックアップ形式です")
            files = manifest.get("files")
            if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
                raise BackupError("バックアップの目録が不正です")
            names = set(entries) - {MANIFEST}
            listed = [item["name"] for item in files]
            if any(not _member_name(name) for name in listed):
                raise BackupError("バックアップに管理範囲外のパスがあります")
            if len(listed) != len({name.casefold() for name in listed}):
                raise BackupError("バックアップに重複した保存先があります")
            listed = set(listed)
            if names != listed:
                raise BackupError("バックアップの中身が目録と一致しません")
            data = {}
            for item in files:
                content = bundle.read(item["name"])
                if (type(item.get("bytes")) is not int or item["bytes"] != len(content)
                        or hashlib.sha256(content).hexdigest() != item["sha256"]):
                    raise BackupError(f"バックアップの内容が壊れています: {item['name']}")
                data[item["name"]] = content
    except (OSError, zipfile.BadZipFile, KeyError, ValueError, TypeError) as exc:
        raise BackupError(f"バックアップを読めません: {exc}") from exc
    return manifest, data


def verify_backup(path: Path) -> dict:
    """目録どおりの保存状態の中身かを確かめ、目録を返す。"""
    return _read_verified_backup(path)[0]


def lss_running() -> bool:
    """LSSの画面または常駐ワーカーが動いているか。"""
    try:
        from lss.core import client

        if client.is_worker_running():
            return True
    except Exception:
        pass
    try:
        listed = subprocess.run(["tasklist", "/FI", "IMAGENAME eq LSS.exe", "/FO", "CSV", "/NH"], capture_output=True,
                                timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return bool(_other_lss_processes(listed.stdout))
    except (OSError, subprocess.TimeoutExpired):
        return True  # 確かめられないときは復元しない


def _other_lss_processes(listing: bytes, own_pid: int | None = None) -> list[int]:
    """tasklistのCSVから、自分以外のLSS.exeのPIDを返す。

    バックアップのzipをLSS.exeへドラッグして復元するとき、復元している自分自身も
    LSS.exeとして一覧に出る。それを「起動中」と数えると復元が常に拒否される。
    読めない行は安全側（他のLSSが動いている）として数える。
    """
    import csv
    import io
    import os

    own = os.getpid() if own_pid is None else own_pid
    found = []
    for row in csv.reader(io.StringIO(listing.decode("utf-8", errors="replace"))):
        if not row or row[0].casefold() != "lss.exe":
            continue
        try:
            pid = int(row[1])
        except (IndexError, ValueError):
            pid = -1
        if pid != own:
            found.append(pid)
    return found


def restore_backup(path: Path, *, state_root: Path | None = None, root: Path | None = None,
                   is_running: Callable[[], bool] = lss_running, now: datetime | None = None) -> Path:
    """バックアップの記録で置き換える。退避した今の記録のフォルダを返す。"""
    state_root = Path(state_root) if state_root is not None else session.state_dir()
    root = Path(root) if root is not None else app_root()
    if is_running():
        raise BackupError("LSSが動いている間は復元できません。終了してから実行してください")
    _, data = _read_verified_backup(path)
    moment = now or datetime.now(timezone.utc)
    saved = root / BACKUP_DIR_NAME / f"before-restore-{moment.strftime('%Y%m%dT%H%M%SZ')}"
    if saved.exists():
        raise BackupError(f"退避先が既にあります: {saved}")

    def target_of(name: str) -> Path:
        kind, _, relative = name.partition("/")
        base = state_root if kind == "state" else root
        target = base / Path(relative)
        if not _member_name(name) or not target.resolve().is_relative_to(base.resolve()):
            raise BackupError(f"復元先が管理範囲外へ移動しています: {name}")
        return target

    # 今の記録（バックアップに含まれる種類すべて）を退避する。
    current = _members(state_root, root)
    # 退避や削除を始める前に、junction等で管理範囲外へ向かう保存先も拒否する。
    for name in {*data, *(name for name, _ in current)}:
        target_of(name)
    restored: list[Path] = []
    saved.mkdir(parents=True)
    try:
        for name, file_path in current:
            destination = saved / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, destination)
        restored_names = set(data)
        for name, file_path in current:
            if name not in restored_names:
                target_of(name)
                file_path.unlink()  # バックアップの時点に無かった記録は、退避したうえで外す
        for name, content in data.items():
            target = target_of(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.restore.tmp")
            try:
                with open(temporary, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            restored.append(target)
    except BaseException:
        # 途中で止まったら、退避した記録へ戻す。
        for name, _ in current:
            backup_copy = saved / name
            if backup_copy.is_file():
                target = target_of(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_copy, target)
        for target in restored:
            if not any(target == target_of(name) for name, _ in current):
                target.unlink(missing_ok=True)
        raise
    return saved
