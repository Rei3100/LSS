"""モデルキャッシュの容量管理(D-4)。lss/harness/fetch.pyのdownload本体は再利用し、
容量キャップ・LRU退避・空き容量下限チェックのみをここに足す(A10)。
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from lss.core import model_config
from lss.harness import fetch
from lss.harness.adopt import PEAK_VRAM_MB_LIMIT, RTF_LIMIT
from lss.harness.tsv_io import load_tsv, save_tsv
from lss.paths import app_root, guard_state_root, is_frozen, resource_root

_RUNTIME_LEDGER_FIELDS = ("last_used_at", "rtf", "peak_vram_mb", "first_measured_at")
_MODEL_STATES = frozenset({"既定", "候補", "除外", "保留"})


class InvalidModelLedgerError(ValueError):
    """モデル台帳の必須識別子、状態、または一意性が壊れている。"""


def load_model_ledger(path: Path) -> tuple[list[str], list[dict]]:
    header, rows = load_tsv(path)
    if "model_id" not in header:
        raise InvalidModelLedgerError(f"モデル台帳にmodel_id列がありません: {path}")
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        model_id = row.get("model_id", "").strip()
        if not model_id:
            raise InvalidModelLedgerError(f"モデル台帳の{index}行目にmodel_idがありません: {path}")
        if model_id in seen:
            raise InvalidModelLedgerError(f"モデル台帳のmodel_idが重複しています: {model_id}")
        seen.add(model_id)
        try:
            fetch.validate_model_id(model_id)
        except fetch.InvalidModelSourceError as exc:
            raise InvalidModelLedgerError(f"モデル台帳のmodel_idが不正です: {model_id}") from exc
        state = row.get("状態", "").strip()
        if "状態" in header and state not in _MODEL_STATES:
            raise InvalidModelLedgerError(f"モデル台帳の状態が不正です: {model_id} ({state or '未指定'})")
        size = row.get("size_bytes", "").strip()
        if size:
            try:
                if int(size) <= 0:
                    raise ValueError
            except ValueError as exc:
                raise InvalidModelLedgerError(
                    f"モデル台帳のsize_bytesが不正です: {model_id} ({size})"
                ) from exc
        sha256 = row.get("sha256", "").strip().lower()
        if sha256 and (
            len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise InvalidModelLedgerError(f"モデル台帳のSHA-256が不正です: {model_id}")
        config_sha256 = row.get("config_sha256", "").strip().lower()
        if config_sha256 and (
            len(config_sha256) != 64 or any(character not in "0123456789abcdef" for character in config_sha256)
        ):
            raise InvalidModelLedgerError(f"モデル台帳の設定SHA-256が不正です: {model_id}")
        if state in {"既定", "候補"}:
            for field in ("distributor_url", "config_url"):
                if field not in header:
                    continue
                try:
                    fetch.validate_source_url(row.get(field, ""))
                except fetch.InvalidModelSourceError as exc:
                    raise InvalidModelLedgerError(
                        f"モデル台帳の{field}が不正です: {model_id}"
                    ) from exc
    return header, rows


def _merge_runtime_ledger(bundled: Path, target: Path) -> None:
    bundled_header, bundled_rows = load_model_ledger(bundled)
    runtime_header, runtime_rows = load_model_ledger(target)
    header = list(dict.fromkeys(bundled_header + runtime_header))
    runtime_by_id = {
        row.get("model_id"): row for row in runtime_rows if row.get("model_id")
    }
    bundled_ids: set[str] = set()
    merged_rows = []
    for bundled_row in bundled_rows:
        model_id = bundled_row.get("model_id")
        if not model_id:
            raise ValueError(f"同梱モデル台帳にmodel_id未指定の行があります: {bundled}")
        bundled_ids.add(model_id)
        runtime_row = runtime_by_id.get(model_id, {})
        merged = dict(bundled_row)
        for field in _RUNTIME_LEDGER_FIELDS:
            if runtime_row.get(field):
                merged[field] = runtime_row[field]
        if not merged.get("sha256") and runtime_row.get("sha256"):
            merged["sha256"] = runtime_row["sha256"]
        merged_rows.append({field: merged.get(field, "") for field in header})

    # 配布台帳から外れたモデルもキャッシュLRUの対象として残すが、通常利用には出さない。
    for runtime_row in runtime_rows:
        if runtime_row.get("model_id") in bundled_ids:
            continue
        retired = dict(runtime_row)
        retired["状態"] = "保留"
        merged_rows.append({field: retired.get(field, "") for field in header})

    if header != runtime_header or merged_rows != runtime_rows:
        save_tsv(target, header, merged_rows)


def models_tsv_path() -> Path:
    """同梱台帳を初回だけ可変領域へコピーし、更新時にも実測値を残す。

    配布版はアプリのフォルダの `ledger/`、開発ツリーは状態領域（`LSS_STATE_DIR`、既定は
    `.state/`）に置く。開発ツリーの同梱台帳はGitが追跡する正本なので、処理のたびに
    最終使用時刻や実測値を書き込まない。
    """
    bundled = resource_root() / "ledger" / "models.tsv"
    if is_frozen():
        target = app_root() / "ledger" / "models.tsv"
    else:
        state = os.environ.get("LSS_STATE_DIR")
        target = guard_state_root(Path(state) if state else app_root() / ".state") / "models.tsv"
    if target.exists():
        _merge_runtime_ledger(bundled, target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with bundled.open("rb") as source, NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=".models.tsv-", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(source, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            temporary_path.rename(target)
        except OSError:
            if not target.exists():
                raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


MODELS_TSV = models_tsv_path()
CACHE_CAP_BYTES = 20 * 1024**3  # D-4: 上限20GB。環境変数LSS_MODEL_CACHE_CAP_BYTESで上書き可(AC6の縮小試験用)
MIN_FREE_BYTES = 25 * 1024**3  # D-4: C:空き下限25GB。環境変数LSS_MIN_FREE_BYTESで上書き可

MIN_RUNTIME_SAMPLE_AUDIO_SECONDS = 30.0


class InsufficientCacheSpaceError(RuntimeError):
    """退避しきってもキャップ/下限を満たせない場合。メッセージに実測バイト数を含める(C7)。"""


def cache_cap_bytes() -> int:
    override = os.environ.get("LSS_MODEL_CACHE_CAP_BYTES")
    return int(override) if override else CACHE_CAP_BYTES


def min_free_bytes() -> int:
    override = os.environ.get("LSS_MIN_FREE_BYTES")
    return int(override) if override else MIN_FREE_BYTES


def cache_dir() -> Path:
    return fetch.model_cache_dir()


def cache_size_bytes() -> int:
    return sum(f.stat().st_size for f in cache_dir().rglob("*") if f.is_file())


def free_disk_bytes() -> int:
    cache_dir().mkdir(parents=True, exist_ok=True)  # 初回起動でキャッシュ未作成でもdisk_usageが失敗しないように
    return shutil.disk_usage(cache_dir()).free


def evict_lru_once(
    header: list[str], rows: list[dict], *, protected_model_ids: frozenset[str] = frozenset(),
) -> dict | None:
    candidates = [
        row
        for row in rows
        if row["model_id"] not in protected_model_ids and (cache_dir() / row["model_id"]).exists()
    ]
    if not candidates:
        return None

    oldest = min(candidates, key=lambda row: row.get("last_used_at") or "")
    target = (cache_dir() / oldest["model_id"]).resolve()
    if target.parent != cache_dir().resolve():
        raise ValueError(f"モデルキャッシュの直下ではありません: {target}")
    shutil.rmtree(target)
    return oldest


def _evictable_bytes(rows: list[dict], protected_model_ids: frozenset[str]) -> int:
    """evict_lru_onceが消せるモデル(台帳にあり、保護外で、キャッシュ直下に実在)の合計バイト数。"""
    root = cache_dir().resolve()
    total = 0
    for row in rows:
        target = (cache_dir() / row["model_id"]).resolve()
        if row["model_id"] in protected_model_ids or target.parent != root or not target.is_dir():
            continue
        total += sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    return total


def ensure_cache_headroom(
    needed_bytes: int, models_tsv_path: Path = MODELS_TSV, *,
    protected_model_ids: frozenset[str] = frozenset(),
) -> None:
    if needed_bytes < 0 or needed_bytes > cache_cap_bytes():
        raise InsufficientCacheSpaceError(f"取得予定サイズ{needed_bytes}がキャッシュ上限{cache_cap_bytes()}を超えます")
    # 退避できる全量を消しても上限・下限に届かないなら、1件も消さずに止める。
    # 届かない退避は再取得コストだけを残し、取得自体はどのみち失敗する。
    if cache_size_bytes() + needed_bytes > cache_cap_bytes() or (
        needed_bytes and free_disk_bytes() - needed_bytes < min_free_bytes()
    ):
        _, rows = load_model_ledger(models_tsv_path)
        evictable = _evictable_bytes(rows, protected_model_ids)
        size_after = cache_size_bytes() - evictable + needed_bytes
        free_after = free_disk_bytes() + evictable - needed_bytes
        if size_after > cache_cap_bytes() or (needed_bytes and free_after < min_free_bytes()):
            raise InsufficientCacheSpaceError(
                f"退避しても容量が足りない: cache_size_bytes={cache_size_bytes()}, "
                f"free_disk_bytes={free_disk_bytes()}, evictable_bytes={evictable}, needed_bytes={needed_bytes}, "
                f"cache_cap_bytes={cache_cap_bytes()}, min_free_bytes={min_free_bytes()}"
            )
    while True:
        cap_ok = cache_size_bytes() + needed_bytes <= cache_cap_bytes()
        # 空き下限は新規cache取得が他の処理領域を食い潰さないためのreserve。
        # cache hitまで全diskの固定下限で拒否すると、取得量0でも既存modelを使えず、
        # 別途実容量を検査するoutput writerとも責務が重複する。
        free_ok = needed_bytes == 0 or free_disk_bytes() - needed_bytes >= min_free_bytes()
        if cap_ok and free_ok:
            return

        header, rows = load_model_ledger(models_tsv_path)
        evicted = evict_lru_once(header, rows, protected_model_ids=protected_model_ids)
        if evicted is None:
            raise InsufficientCacheSpaceError(
                f"退避対象が尽きた: cache_size_bytes={cache_size_bytes()}, "
                f"free_disk_bytes={free_disk_bytes()}, needed_bytes={needed_bytes}, "
                f"cache_cap_bytes={cache_cap_bytes()}, min_free_bytes={min_free_bytes()}"
            )


def touch_last_used(model_id: str, when: str | None = None, models_tsv_path: Path = MODELS_TSV) -> None:
    header, rows = load_model_ledger(models_tsv_path)
    for row in rows:
        if row["model_id"] == model_id:
            # 日付だけでは同日に利用した複数モデルのLRU順を判定できない。
            # ISO 8601のUTC時刻なら、既存の日付形式とも文字列順で比較できる。
            row["last_used_at"] = when or datetime.now(timezone.utc).isoformat()
            save_tsv(models_tsv_path, header, rows)
            return


def model_sha256(
    model_id: str, ckpt_path: Path, models_tsv_path: Path = MODELS_TSV,
) -> str:
    """台帳の既知ハッシュを返し、未記録ならモデル実体から算出して台帳へ保存する。"""
    header, rows = load_model_ledger(models_tsv_path)
    for row in rows:
        if row["model_id"] != model_id:
            continue
        recorded = row.get("sha256", "").strip().lower()
        if recorded:
            return recorded
        if not ckpt_path.is_file():
            return "不明"
        actual = fetch.file_sha256(ckpt_path)
        if "sha256" in header:
            row["sha256"] = actual
            save_tsv(models_tsv_path, header, rows)
        return actual
    raise ValueError(f"model_idが見つからない: {model_id}")


def ensure_model_cached(
    model_id: str,
    ckpt_url: str,
    config_url: str,
    size_bytes: int | None,
    models_tsv_path: Path = MODELS_TSV,
    progress: Callable[[dict], None] | None = None,
) -> tuple[Path, Path]:
    if size_bytes is None or size_bytes <= 0:
        raise fetch.InvalidModelSourceError(f"モデル取得予定サイズが未登録です: {model_id}")
    ckpt_path, config_path = fetch.model_paths(model_id, ckpt_url, config_url)
    _, rows = load_model_ledger(models_tsv_path)
    row = next((item for item in rows if item["model_id"] == model_id), None)
    if row is None:
        raise ValueError(f"model_idが見つからない: {model_id}")
    expected_sha256 = row.get("sha256", "").strip().lower() or None
    expected_config_sha256 = row.get("config_sha256", "").strip().lower() or None
    # 既存の正常な重みを追加容量へ二重計上しない。通常の再利用時は大容量ファイルを
    # 二度ハッシュせず、ここでの検証結果をそのまま採用する。
    cached = fetch.is_cached(ckpt_path, size_bytes, expected_sha256)
    if not cached and ckpt_path.is_file():
        # サイズまたはSHA-256が一致しない実体は再利用不能。これを残したまま新しい重みの
        # 全容量を要求すると、キャッシュ上限が十分でも旧実体との二重計上で自己修復できない。
        # 正常なキャッシュはcached=Trueでこの経路へ入らず、取得中の.partはdownload()が
        # 原子的に扱うため、削除対象は検証済みの不正実体だけに限定される。
        ckpt_path.unlink()
    needed_bytes = 0 if cached else (size_bytes or 0)
    protected = frozenset({model_id})
    ensure_cache_headroom(needed_bytes, models_tsv_path=models_tsv_path, protected_model_ids=protected)

    def report(asset: str):
        if progress is None:
            return None
        return lambda completed, total: progress({
            "phase": "downloading_model",
            "model_id": model_id,
            "asset": asset,
            "completed_bytes": completed,
            "total_bytes": total,
        })

    config_download = lambda: fetch.download(
        config_url, config_path, progress=report("config"), max_bytes=fetch.MAX_CONFIG_BYTES,
        expected_sha256=expected_config_sha256,
    )
    if cached:
        model_config.load_cached_config(config_path, config_download)
    else:
        ckpt_path, config_path = fetch.ensure_model_files(
            model_id, ckpt_url, config_url,
            expected_size=size_bytes, expected_sha256=expected_sha256,
            expected_config_sha256=expected_config_sha256,
            weight_progress=report("weight"), config_progress=report("config"),
        )
        model_config.load_cached_config(config_path, config_download)
    ensure_cache_headroom(0, models_tsv_path=models_tsv_path, protected_model_ids=protected)
    model_sha256(model_id, ckpt_path, models_tsv_path)
    touch_last_used(model_id, models_tsv_path=models_tsv_path)
    return ckpt_path, config_path


def record_recipe_metrics(
    model_id: str,
    rtf: float,
    peak_vram_mb: float,
    models_tsv_path: Path = MODELS_TSV,
    *, audio_seconds: float | None = None,
) -> bool:
    """実行資源値を記録する。資源値だけを理由に品質候補を除外しない。"""
    header, rows = load_model_ledger(models_tsv_path)
    for row in rows:
        if row["model_id"] != model_id:
            continue
        row["rtf"] = str(rtf)
        row["peak_vram_mb"] = str(peak_vram_mb)
        break
    save_tsv(models_tsv_path, header, rows)
    return False


def exceeds_runtime_limits(rtf: float, peak_vram_mb: float) -> bool:
    return rtf > RTF_LIMIT or peak_vram_mb > PEAK_VRAM_MB_LIMIT
