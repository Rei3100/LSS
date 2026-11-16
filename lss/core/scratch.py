"""LSSが明示的に所有する使い捨て作業物だけを回収する。"""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import shutil
import stat
from tempfile import NamedTemporaryFile
import time
from uuid import UUID, uuid4

AUTHORITY = "lss.core.scratch/v1"
PURPOSES = frozenset({"input_decode", "separation_inference"})
ORPHAN_GRACE_SECONDS = 24 * 60 * 60


def _is_link(value: os.stat_result) -> bool:
    return bool(stat.S_ISLNK(value.st_mode)
                or getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _plain(path: Path) -> os.stat_result:
    value = path.lstat()
    if _is_link(value) or (stat.S_ISREG(value.st_mode) and value.st_nlink != 1):
        raise ValueError(f"linkを含む作業物は回収しません: {path}")
    return value


def _ancestors(path: Path) -> None:
    """作業領域とその上位がディレクトリであることを確かめる。

    上位に置かれたディレクトリlink（保存先の配置で ``.state/work`` などを別driveへ移したjunction）は一度だけ辿り、
    行き先とその全上位が link を含まない実ディレクトリであることを確かめる。作業物そのもの（scratch・receipt・中身）は
    ``_plain`` のまま link を拒否する。
    """
    for item in (path, *path.parents):
        value = item.lstat()
        if _is_link(value):
            target = Path(os.path.realpath(item))
            for real in (target, *target.parents):
                if not stat.S_ISDIR(_plain(real).st_mode):
                    raise ValueError(f"作業領域がディレクトリではありません: {real}")
            continue
        if not stat.S_ISDIR(_plain(item).st_mode):
            raise ValueError(f"作業領域がディレクトリではありません: {item}")


def _registry(root: Path) -> Path:
    # Payloadと別に保持し、途中まで削除されたPayloadの所有権を失わない。
    return root.parent / f".{root.name}-scratch-owners"


def _write(path: Path, record: dict) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=".receipt-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _lease(path: Path, *, create: bool = False):
    """OSのbyte lock。process crash後は自動解放、使用中なら回収側はskip。"""
    if not create:
        if not stat.S_ISREG(_plain(path).st_mode):
            raise ValueError("作業物のlockが通常fileではありません")
    with path.open("x+b" if create else "r+b") as stream:
        if create:
            stream.write(b"0")
            stream.flush()
            os.fsync(stream.fileno())
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


def _read(receipt: Path, root: Path) -> dict:
    if not stat.S_ISREG(_plain(receipt).st_mode) or receipt.stat().st_size > 16384:
        raise ValueError("作業物の所有記録が不正です")
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or record.get("authority") != AUTHORITY
            or record.get("class") != "disposable_scratch"
            or record.get("purpose") not in PURPOSES
            or record.get("work_root") != str(root)
            or record.get("artifact_id") != receipt.stem
            or UUID(receipt.stem).hex != receipt.stem):
        raise ValueError("作業物の所有権を確認できません")
    name = record.get("directory")
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or Path(name).name != name or "/" in name or "\\" in name or ":" in name):
        raise ValueError("作業物が管理範囲外です")
    for key in ("created_at", "released_at"):
        value = record.get(key)
        if key == "released_at" and value is None:
            continue
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError("作業物の時刻が不正です")
    identity = record.get("directory_identity")
    if identity is not None and (not isinstance(identity, list) or len(identity) != 2
                                or any(type(value) is not int for value in identity)):
        raise ValueError("作業物の実体識別が不正です")
    return record


def _tree(path: Path, record: dict) -> tuple[int, int]:
    """削除直前に境界・実体・全descendantを確認する。リンクへは降りない。"""
    root = Path(record["work_root"])
    _ancestors(root)
    _ancestors(_registry(root))
    if path.parent != root or path.resolve().parent != root.resolve():
        raise ValueError("作業物が管理範囲外です")
    value = _plain(path)
    if (not stat.S_ISDIR(value.st_mode)
            or record.get("directory_identity") != [value.st_dev, value.st_ino]):
        raise ValueError("作業物のdirectoryが差し替えられています")
    pending, size, count = [path], 0, 0
    while pending:
        for child in pending.pop().iterdir():
            info = _plain(child)
            if stat.S_ISDIR(info.st_mode):
                pending.append(child)
            elif stat.S_ISREG(info.st_mode):
                size += info.st_size
                count += 1
            else:
                raise ValueError("未知のfile種別を含む作業物は回収しません")
    return size, count


def _delete(receipt: Path, record: dict) -> tuple[int, int]:
    target = Path(record["work_root"]) / record["directory"]
    if not target.exists() and not target.is_symlink():
        receipt.unlink()
        return 0, 0
    measured = _tree(target, record)
    shutil.rmtree(target)
    # rmtree失敗時はreceiptが残るため、部分削除を次回も安全に再試行できる。
    receipt.unlink()
    return measured


class ScratchDirectory:
    """Productのdecode/inference専用。source/cache/publicationには使わない。"""

    def __init__(self, *, dir: Path, purpose: str, prefix: str):
        if purpose not in PURPOSES or not prefix or any(c in prefix for c in "/\\:"):
            raise ValueError("作業物の用途またはprefixが不正です")
        self.root = Path(os.path.abspath(dir))
        self.purpose, self.prefix = purpose, prefix

    def __enter__(self) -> str:
        # 未作成のrootについても、既存の全ancestorを確認してから作る。
        ancestor = self.root
        while not ancestor.exists() and not ancestor.is_symlink():
            ancestor = ancestor.parent
        _ancestors(ancestor)
        self.root.mkdir(parents=True, exist_ok=True)
        _ancestors(self.root)
        registry = _registry(self.root)
        if registry.exists() or registry.is_symlink():
            _ancestors(registry)
        registry.mkdir(exist_ok=True)
        _ancestors(registry)
        key = uuid4().hex
        self.receipt = registry / f"{key}.json"
        self.lock = registry / f"{key}.lock"
        # 旧runtimeのdecode-/hash-prefix GCにも拾われないnamespaceにする。
        self.path = self.root / f"scratch-{self.prefix}{key}"
        self.record = {"schema_version": 1, "authority": AUTHORITY,
                       "class": "disposable_scratch", "artifact_id": key,
                       "work_root": str(self.root), "directory": self.path.name,
                       "purpose": self.purpose, "created_at": time.time(),
                       "released_at": None, "directory_identity": None}
        self.owner = _lease(self.lock, create=True)
        self.owner.__enter__()
        try:
            _write(self.receipt, self.record)
            self.path.mkdir()
            value = _plain(self.path)
            self.record["directory_identity"] = [value.st_dev, value.st_ino]
            _write(self.receipt, self.record)
        except BaseException:
            self.owner.__exit__(None, None, None)
            raise
        return str(self.path)

    def __exit__(self, *_exc) -> None:
        try:
            self.record["released_at"] = time.time()
            _write(self.receipt, self.record)
            _delete(self.receipt, self.record)
        except (OSError, ValueError):
            # 保存成功をcleanup失敗で失敗扱いにしない。残ったreceiptをstartupで再試行。
            logging.getLogger(__name__).warning("作業物の回収を次回へ延期しました: %s", self.path)
        finally:
            self.owner.__exit__(None, None, None)
        if not self.receipt.exists():
            try:
                self.lock.unlink(missing_ok=True)
            except OSError:
                pass


def reconcile(root: Path, *, older_than_seconds: float = ORPHAN_GRACE_SECONDS,
              now: float | None = None, dry_run: bool = False) -> dict:
    """Machine-readable inventoryと回収。同じreceiptを実行中も再読して判断する。"""
    now = time.time() if now is None else now
    if (not math.isfinite(older_than_seconds) or older_than_seconds < 0
            or not math.isfinite(now) or now < 0):
        raise ValueError("作業物の保持時間と現在時刻は有限な0以上の値が必要です")
    root = Path(os.path.abspath(root))
    report = {"schema_version": 1, "authority": AUTHORITY, "dry_run": dry_run,
              "work_root": str(root), "artifacts": [], "removed": [],
              "reclaimed_bytes": 0, "reclaimed_files": 0}
    if not root.exists() and not root.is_symlink():
        return report
    _ancestors(root)
    registry = _registry(root)
    receipts = []
    if registry.exists() or registry.is_symlink():
        _ancestors(registry)
        receipts = sorted(registry.glob("*.json"))
    records, claims = [], {}
    for receipt in receipts:
        try:
            record = _read(receipt, root)
            claims.setdefault(record["directory"], []).append(receipt)
            records.append((receipt, record))
        except (OSError, ValueError, TypeError, KeyError):
            report["artifacts"].append({"receipt": str(receipt), "class": "UNKNOWN",
                                        "decision": "keep_invalid_receipt"})
    for receipt, record in records:
        target = root / record["directory"]
        row = {"path": str(target), "receipt": str(receipt),
               "class": "disposable_scratch", "purpose": record["purpose"]}
        report["artifacts"].append(row)
        lock = receipt.with_suffix(".lock")
        if len(claims[record["directory"]]) != 1:
            row["decision"] = "keep_conflicting_ownership"
            continue
        try:
            with _lease(lock):
                # list時とlock取得時の間にownerが更新したreceiptを使い直す。
                latest = _read(receipt, root)
                if any(latest.get(key) != record.get(key) for key in (
                        "directory", "directory_identity", "created_at", "purpose")):
                    raise ValueError("lock取得前に作業物の所有権が変更されました")
                record = latest
                if not target.exists() and not target.is_symlink():
                    row["decision"] = "missing_payload"
                    if not dry_run:
                        _delete(receipt, record)
                else:
                    size, count = _tree(target, record)
                    row.update(size_bytes=size, file_count=count)
                    if record["released_at"] is None and now - record["created_at"] < older_than_seconds:
                        row["decision"] = "keep_orphan_grace"
                    else:
                        row["decision"] = "would_remove" if dry_run else "removed"
                        if not dry_run:
                            size, count = _delete(receipt, record)
                            report["removed"].append(str(target))
                            report["reclaimed_bytes"] += size
                            report["reclaimed_files"] += count
        except FileNotFoundError:
            row["decision"] = "keep_missing_ownership"
        except OSError:
            row["decision"] = "keep_active_or_io_blocked"
        except (ValueError, TypeError, KeyError):
            row["decision"] = "keep_unsafe_or_changed"
        if not dry_run and not receipt.exists():
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
    for entry in sorted(root.iterdir()):
        if entry.name not in claims:
            report["artifacts"].append({"path": str(entry), "class": "UNKNOWN",
                                        "decision": "keep_no_ownership"})
    return report
