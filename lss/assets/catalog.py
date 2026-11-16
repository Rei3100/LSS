"""保存された記録から、入力・処理結果・派生成果・録音・登場区間を区別して引ける目録を作る。

目録は派生物で、正本ではない。完了台帳・取得の来歴・一致の記録（identity.json・series_identity.json）・移動の記録から
いつでも作り直せる（SQLite。既定は状態領域の `asset-catalog.sqlite`）。記録を書き換えない。

- sources: 入力の内容SHA-256ごと。記録された場所、取得元（あれば）。
- results: 完了記録ごと（入力・Recipe・処理・時刻・成果物の有無・ゼロ件の理由）。
- derivatives: 成果物の内容SHA-256ごと（どの結果の成果物か、記録された場所、移動先）。
- recordings: 参照録音（内容SHA-256）と、参照は無いが番組をまたいで同じ録音と裏付けられた録音（`recurring:`）。
- occurrences: 入力（番組）の中で、ある録音が同じ録音として現れる区間と根拠。
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

from lss.core import session
from lss.harness.tsv_io import load_tsv

SCHEMA = """
CREATE TABLE sources (sha256 TEXT PRIMARY KEY, recorded_paths TEXT NOT NULL, logical_source TEXT,
                      reacquire_url TEXT);
CREATE TABLE results (id INTEGER PRIMARY KEY, input_sha256 TEXT NOT NULL, recipe_id TEXT, completed_at TEXT,
                      source_path TEXT, outcome TEXT, zero_reason TEXT, model_id TEXT, model_sha256 TEXT,
                      config_sha256 TEXT);
CREATE TABLE derivatives (sha256 TEXT NOT NULL, result_id INTEGER NOT NULL, recorded_path TEXT NOT NULL,
                          relocated_paths TEXT NOT NULL, PRIMARY KEY (sha256, result_id));
CREATE TABLE recordings (sha256 TEXT PRIMARY KEY, label TEXT, path TEXT);
CREATE TABLE occurrences (id INTEGER PRIMARY KEY, recording_sha256 TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                          start_seconds REAL NOT NULL, end_seconds REAL NOT NULL,
                          reference_start_seconds REAL, waveform_explained REAL, evidence_record TEXT NOT NULL);
CREATE INDEX occurrences_by_recording ON occurrences (recording_sha256);
CREATE INDEX occurrences_by_source ON occurrences (source_sha256);
CREATE INDEX results_by_input ON results (input_sha256);
"""


def default_path(state_root: Path | None = None) -> Path:
    return (Path(state_root) if state_root is not None else session.state_dir()) / "asset-catalog.sqlite"


class CatalogIdentityError(ValueError):
    """指定された録音照合記録を欠けた目録へ読み替えない。"""


def _hash_value(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value, *, maximum: float | None = None) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0
            and (maximum is None or value <= maximum))


def _span(item: dict, start: str, end: str) -> bool:
    return _number(item.get(start)) and _number(item.get(end)) and item[end] > item[start]


def _validate_identity_record(record: object) -> None:
    """保存済みの観測を取り込む。現在のRecipeによる品質の再認定はしない。"""
    if not isinstance(record, dict):
        raise CatalogIdentityError("録音照合記録がobjectではありません")
    if "schema_version" in record and (type(record["schema_version"]) is not int or record["schema_version"] != 1):
        raise CatalogIdentityError("録音照合記録のschemaが未対応です")
    if "payload_sha256" in record:
        body = {key: value for key, value in record.items() if key != "payload_sha256"}
        digest = hashlib.sha256(json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if record["payload_sha256"] != digest:
            raise CatalogIdentityError("録音照合記録の内容hashが一致しません")
    if record.get("type") == "series_identity_annotation":
        episodes, pairs = record.get("episodes"), record.get("shared_recordings")
        if not isinstance(episodes, list) or not isinstance(pairs, list):
            raise CatalogIdentityError("共通録音の番組・区間集合が不正です")
        keys = set()
        for episode in episodes:
            if (not isinstance(episode, dict) or not isinstance(episode.get("key"), str)
                    or not episode["key"] or episode["key"] in keys
                    or not _hash_value(episode.get("source_sha256"))):
                raise CatalogIdentityError("共通録音の番組同一性が不正です")
            keys.add(episode["key"])
        for pair in pairs:
            if (not isinstance(pair, dict) or not isinstance(pair.get("first"), str)
                    or not isinstance(pair.get("second"), str)
                    or pair["first"] not in keys or pair["second"] not in keys
                    or pair["first"] == pair["second"] or not isinstance(pair.get("occurrences"), list)):
                raise CatalogIdentityError("共通録音の組が番組集合へ対応しません")
            for occurrence in pair["occurrences"]:
                if (not isinstance(occurrence, dict)
                        or not _span(occurrence, "first_start_seconds", "first_end_seconds")
                        or not _span(occurrence, "second_start_seconds", "second_end_seconds")):
                    raise CatalogIdentityError("共通録音の時刻範囲が不正です")
        return
    # typeが無かった初期の記録も、取り込む全フィールドを検証して保持する。
    if record.get("type") not in (None, "episode_identity_annotation"):
        raise CatalogIdentityError("録音照合記録の種類が未対応です")
    run, references, matches = record.get("run"), record.get("external_evidence_used"), record.get("matches")
    if (not isinstance(run, dict) or not _hash_value(run.get("source_sha256"))
            or not isinstance(references, list) or not isinstance(matches, list)):
        raise CatalogIdentityError("録音照合の原本・参照・一致集合が不正です")
    for reference in references:
        if not isinstance(reference, dict) or not _hash_value(reference.get("sha256")):
            raise CatalogIdentityError("録音照合の参照同一性が不正です")
    for match in matches:
        if not isinstance(match, dict) or match.get("identity_state") not in ("known_exact", "unresolved"):
            raise CatalogIdentityError("録音一致の判定が不正です")
        if match["identity_state"] != "known_exact":
            continue
        if (not _hash_value(match.get("reference_sha256"))
                or not _span(match, "query_start_seconds", "query_end_seconds")
                or not _number(match.get("reference_start_seconds"))
                or (match.get("waveform_explained") is not None and not _number(match["waveform_explained"], maximum=1))):
            raise CatalogIdentityError("同一録音の内容同一性または時刻範囲が不正です")


def _identity_snapshots(paths) -> list[tuple[Path, str, dict]]:
    snapshots = []
    for path in dict.fromkeys(Path(path).resolve() for path in paths):
        try:
            content = path.read_bytes()
            record = json.loads(content)
            _validate_identity_record(record)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            raise CatalogIdentityError(f"録音照合記録を取り込めません: {path}: {exc}") from exc
        snapshots.append((path, hashlib.sha256(content).hexdigest(), record))
    return snapshots


def build(*, state_root: Path | None = None, identity_records: list[Path] = (),
          outcome_records: list[Path] = (), database: Path | None = None) -> Path:
    """記録から目録を作り直す。作り終えるまで古い目録を置き換えない。"""
    from lss.assets.relocate import current_locations
    from lss.assets.reprocess import _identity_of
    from lss.assets import held_catalog

    snapshots = _identity_snapshots(identity_records)
    held_snapshots = held_catalog.snapshots(outcome_records, state_root=state_root)
    database = Path(database) if database is not None else default_path(state_root)
    database.parent.mkdir(parents=True, exist_ok=True)
    # 同じ目録の再構築が重なっても、相手の作業中DBを削除・共有しない。
    with NamedTemporaryFile(dir=database.parent, prefix=f".{database.name}.", suffix=".building", delete=False) as stream:
        temporary = Path(stream.name)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(SCHEMA)
        connection.executescript(held_catalog.SCHEMA)
        try:
            _, rows = load_tsv(session.completion_ledger_path(state_root))
        except FileNotFoundError:
            # 新規状態の台帳不在と、既存台帳の読取・解析失敗を区別する。
            # 後者で空の目録を公開すると、既知の結果と来歴を見失う。
            rows = []
        sources: dict[str, set[str]] = {}
        for row in rows:
            sha = row.get("input_sha256")
            if not sha:
                continue
            sources.setdefault(sha, set()).add(row.get("source_path") or "")
            try:
                outputs = json.loads(row.get("outputs") or "[]")
            except ValueError:
                outputs = []
            outputs = [item for item in outputs if isinstance(item, dict)] if isinstance(outputs, list) else []
            identity = _identity_of(row, outputs, state_root=state_root)
            cursor = connection.execute(
                "INSERT INTO results (input_sha256, recipe_id, completed_at, source_path, outcome, zero_reason,"
                " model_id, model_sha256, config_sha256) VALUES (?,?,?,?,?,?,?,?,?)",
                (sha, row.get("recipe_id"), row.get("completed_at"), row.get("source_path"), row.get("outcome"),
                 row.get("zero_reason") or None, identity.model_id, identity.model_sha256, identity.config_sha256))
            for item in outputs:
                if not item.get("sha256"):
                    continue
                relocated = [str(path) for path in current_locations(item["sha256"], state_root=state_root)]
                connection.execute(
                    "INSERT OR IGNORE INTO derivatives VALUES (?,?,?,?)",
                    (item["sha256"], cursor.lastrowid, str(item.get("path") or ""), json.dumps(relocated)))

        acquisitions: dict[str, dict] = {}
        from lss.acquisition.store import iter_records

        for record in iter_records(state_root):
            content = record.get("content") or {}
            if content.get("sha256"):
                acquisitions[content["sha256"]] = record
                sources.setdefault(content["sha256"], set()).add(content.get("path") or "")

        for path, _, record in snapshots:
            if record.get("type") == "series_identity_annotation":
                _insert_recurring(connection, record, str(Path(path)), sources)
                continue
            source_sha = (record.get("run") or {}).get("source_sha256")
            if not source_sha:
                continue
            sources.setdefault(source_sha, set())
            references = {item["sha256"]: item for item in record.get("external_evidence_used") or []}
            for match in record.get("matches") or []:
                if match.get("identity_state") != "known_exact":
                    continue
                reference = references.get(match["reference_sha256"], {})
                connection.execute("INSERT OR IGNORE INTO recordings VALUES (?,?,?)",
                                   (match["reference_sha256"], match.get("reference_id"),
                                    reference.get("path") or match.get("reference_path")))
                connection.execute(
                    "INSERT INTO occurrences (recording_sha256, source_sha256, start_seconds, end_seconds,"
                    " reference_start_seconds, waveform_explained, evidence_record) VALUES (?,?,?,?,?,?,?)",
                    (match["reference_sha256"], source_sha, match["query_start_seconds"], match["query_end_seconds"],
                     match.get("reference_start_seconds"), match.get("waveform_explained"), str(Path(path))))

        held_catalog.insert(connection, held_snapshots, sources)
        for sha, paths in sources.items():
            record = acquisitions.get(sha) or {}
            connection.execute("INSERT INTO sources VALUES (?,?,?,?)",
                               (sha, json.dumps(sorted(p for p in paths if p), ensure_ascii=False),
                                record.get("logical_source"), (record.get("reacquire") or {}).get("url")))
        connection.commit()
        connection.close()
        connection = None
        for path, digest, _ in snapshots:
            try:
                current = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise CatalogIdentityError(f"目録確定前に録音照合記録を再確認できません: {path}") from exc
            if current != digest:
                raise CatalogIdentityError(f"目録構築中に録音照合記録が変更されています: {path}")
        held_catalog.verify(held_snapshots)
        temporary.replace(database)
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)
    return database


RECURRING_PREFIX = "recurring:"


def _insert_recurring(connection, record: dict, evidence: str, sources: dict[str, set[str]]) -> None:
    """番組をまたいで同じ録音と裏付けられた区間を、参照録音の無い1つの録音としてまとめて登録する。

    区間どうしは、組ごとの共通録音の証拠でつながり、同じ番組の中で半分以上重なる区間は同じ出現とみなす。
    録音の識別子は出現の集まりから決まり、内容のSHA-256ではない（元の音声ファイルが無いため）。
    """
    source_of = {item["key"]: item.get("source_sha256") for item in record.get("episodes") or []}
    spans: list[tuple[str, float, float]] = []
    parent: list[int] = []
    exact: dict[tuple[str, float, float], int] = {}

    def node(episode: str, start: float, end: float) -> int:
        span = (episode, start, end)
        if span in exact:
            return exact[span]
        index = len(spans)
        spans.append(span)
        parent.append(index)
        exact[span] = index
        for other, (e, s, t) in enumerate(spans[:-1]):
            overlap = min(end, t) - max(start, s)
            if e == episode and overlap > 0 and overlap / max(min(end - start, t - s), 1e-9) >= 0.5:
                # 区間を置き換えず全観測を残す。後から両区間をつなぐ観測が来た場合も、両方へつなぐ。
                parent[find(index)] = find(other)
        return index

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for pair in record.get("shared_recordings") or []:
        for item in pair.get("occurrences") or []:
            a = node(pair["first"], item["first_start_seconds"], item["first_end_seconds"])
            b = node(pair["second"], item["second_start_seconds"], item["second_end_seconds"])
            parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for index in range(len(spans)):
        groups.setdefault(find(index), []).append(index)
    for members in groups.values():
        places: list[tuple[str, float, float]] = []
        for episode, start, end in sorted(spans[i] for i in members):
            if places and episode == places[-1][0] and start <= places[-1][2]:
                places[-1] = (episode, places[-1][1], max(end, places[-1][2]))
            else:
                places.append((episode, start, end))
        recording = RECURRING_PREFIX + hashlib.sha256(json.dumps(places, ensure_ascii=False).encode("utf-8")).hexdigest()
        connection.execute("INSERT OR IGNORE INTO recordings VALUES (?,?,?)", (recording, "recurring_without_reference", None))
        for episode, start, end in places:
            source_sha = source_of.get(episode)
            if not source_sha:
                continue
            sources.setdefault(source_sha, set())
            connection.execute(
                "INSERT INTO occurrences (recording_sha256, source_sha256, start_seconds, end_seconds,"
                " reference_start_seconds, waveform_explained, evidence_record) VALUES (?,?,?,?,?,?,?)",
                (recording, source_sha, start, end, None, None, evidence))


def occurrences_of(recording_sha256: str, *, database: Path) -> list[dict]:
    """ある録音が、どの入力のどこに現れるか。"""
    # sqlite3の接続のwithは閉じないため、Windowsで作り直しを妨げないよう明示的に閉じる。
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT o.source_sha256, o.start_seconds, o.end_seconds, o.reference_start_seconds, o.waveform_explained,"
            " s.recorded_paths, s.logical_source FROM occurrences o JOIN sources s ON s.sha256 = o.source_sha256"
            " WHERE o.recording_sha256 = ? ORDER BY o.source_sha256, o.start_seconds", (recording_sha256,)).fetchall()
    return [dict(row) for row in rows]


def summary(*, database: Path) -> dict:
    # sqlite3の接続のwithは閉じないため、Windowsで作り直しを妨げないよう明示的に閉じる。
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in ("sources", "results", "derivatives", "recordings", "occurrences")}
        counts["recordings_in_multiple_sources"] = connection.execute(
            "SELECT COUNT(*) FROM (SELECT recording_sha256 FROM occurrences GROUP BY recording_sha256"
            " HAVING COUNT(DISTINCT source_sha256) > 1)").fetchone()[0]
    return counts
