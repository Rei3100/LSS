"""BGMの保留結果を、正式成果物・録音同一性と混同せず派生目録へ取り込む。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing

from mutagen import MutagenError
from mutagen.flac import FLAC
import soundfile as sf

from lss.bgm import candidate_store
from lss.harness.fetch import file_sha256

SCHEMA = """
CREATE TABLE held_results (record_path TEXT PRIMARY KEY, record_sha256 TEXT NOT NULL,
                          source_sha256 TEXT NOT NULL, source_locations TEXT NOT NULL,
                          recipe_id TEXT NOT NULL, evidence_path TEXT NOT NULL,
                          disposition TEXT NOT NULL CHECK (disposition = 'HELD'), reason TEXT NOT NULL,
                          declared_branches TEXT NOT NULL, unavailable_branches TEXT NOT NULL);
CREATE TABLE held_candidates (candidate_id TEXT NOT NULL, result_path TEXT NOT NULL,
                             source_sha256 TEXT NOT NULL, content_sha256 TEXT NOT NULL,
                             role TEXT NOT NULL CHECK (role IN ('primary', 'music_alternative', 'reference_residual')),
                             primary_candidate_id TEXT, recorded_path TEXT NOT NULL,
                             current_paths TEXT NOT NULL, start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
                             sample_rate INTEGER NOT NULL, offset_numerator INTEGER NOT NULL,
                             offset_denominator INTEGER NOT NULL, origin TEXT NOT NULL,
                             semantic_role TEXT NOT NULL CHECK (semantic_role = 'unresolved_music_like'),
                             publication_state TEXT NOT NULL CHECK (publication_state = 'not_published'),
                             PRIMARY KEY (candidate_id, result_path));
CREATE INDEX held_candidates_by_source ON held_candidates (source_sha256);
CREATE INDEX held_candidates_by_primary ON held_candidates (primary_candidate_id);
"""


class CatalogHeldError(ValueError):
    """指定した保留記録を欠けた目録へ読み替えない。"""


@dataclass(frozen=True)
class Snapshot:
    path: Path
    digest: str
    record: dict
    source_locations: tuple[Path, ...]
    candidates: tuple[dict, ...]
    pinned: dict[Path, str]
    unavailable_branches: tuple[str, ...]
    unavailable_evidence: tuple[str, ...] = ()


def snapshots(paths, *, state_root=None) -> list[Snapshot]:
    from lss import application

    result = []
    for path in dict.fromkeys(Path(path).resolve() for path in paths):
        try:
            content = path.read_bytes()
            record = json.loads(content)
            outcome = application.read_bgm_development_outcome(path, state_root=state_root)
            if outcome.disposition != application.ResultDisposition.HELD:
                raise ValueError(outcome.message or "保留結果ではありません")
            pinned = {path: hashlib.sha256(content).hexdigest()}

            def read_pinned(item):
                location = Path(item["path"]).resolve()
                body = location.read_bytes()
                if hashlib.sha256(body).hexdigest() != item["sha256"]:
                    raise ValueError("保留結果の根拠が読取り中に変更されています")
                pinned[location] = item["sha256"]
                return json.loads(body)

            run = read_pinned(record["evidence"])
            # 初期形式を履歴から読んだ場合も、検証済みの元runにある計画を使う。
            plan_record = record.get("plan") or {"path": str(outcome.evidence_path.parent / "plan.json"),
                                                "sha256": run["plan_sha256"]}
            read_pinned(plan_record)
            if outcome.identity_evidence_path is not None:
                read_pinned(record["recording_identity"])
            for location in outcome.source_locations:
                pinned[location] = record["source"]["sha256"]
            parents = {candidate["asset"]["candidate_asset_id"]: candidate for candidate in run["candidates"]}
            if len(parents) != len(run["candidates"]):
                raise ValueError("保留結果の元cue IDが重複しています")
            selected = [(candidate["asset"], candidate["origin"], "primary", None) for candidate in parents.values()]
            unavailable = []
            for key, role, evidence_path in (
                    ("held_alternatives", "music_alternative", outcome.alternative_evidence_path),
                    ("held_reference_residuals", "reference_residual", outcome.reference_residual_evidence_path)):
                if key not in record:
                    continue
                if evidence_path is None:
                    unavailable.append(role)
                    continue
                branch = read_pinned(record[key])
                read_pinned(record[key]["plan"])
                origins = {candidate["asset"]["candidate_asset_id"]: candidate["origin"] for candidate in branch["candidates"]}
                for asset in record[key]["artifacts"]:
                    primary_id = asset["primary_candidate_asset_id"]
                    if primary_id not in parents:
                        raise ValueError("保留分岐が元cueへ対応しません")
                    selected.append((asset, origins[asset["candidate_asset_id"]], role, primary_id))
            candidates = []
            for asset, origin_record, role, primary_id in selected:
                recorded_path = Path(asset["path"]).resolve()
                locations = outcome.held_artifact_locations[recorded_path]
                origin = candidate_store.CandidateOrigin(**origin_record)
                origin.validate()
                if origin.source_sha256 != record["source"]["sha256"] or origin.candidate_id != asset["candidate_asset_id"]:
                    raise ValueError("保留候補の原本・候補IDが来歴へ対応しません")
                for location in locations:
                    if file_sha256(location) != asset["sha256"]:
                        raise ValueError("保留候補の内容が読取り中に変更されています")
                    pinned[location] = asset["sha256"]
                    tags = FLAC(location)
                    metadata = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
                    expected = {"schema_version": origin.format_revision, "candidate_id": origin.candidate_id,
                                "origin": origin.to_dict(), "start_seconds": origin.start_frame / origin.sample_rate
                                + origin.time_offset_numerator / origin.time_offset_denominator,
                                "end_seconds": origin.end_frame / origin.sample_rate
                                + origin.time_offset_numerator / origin.time_offset_denominator,
                                "timebase": "source_media_time_from_decoded_audio_origin",
                                "semantic_role": "unresolved_music_like", "identity_state": "unresolved",
                                "publication_state": "not_published"}
                    candidate_store._verify_provenance(tags, metadata, expected, origin)
                    info = sf.info(location)
                    if info.samplerate != origin.sample_rate or info.channels != 2 or info.frames != origin.end_frame - origin.start_frame:
                        raise ValueError("保留候補の時間軸・左右が来歴へ対応しません")
                candidates.append({"candidate_id": origin.candidate_id, "source_sha256": origin.source_sha256,
                                   "content_sha256": asset["sha256"], "role": role, "primary_candidate_id": primary_id,
                                   "recorded_path": str(recorded_path), "current_paths": [str(p) for p in locations],
                                   "origin": origin.to_dict()})
            snapshot = Snapshot(path, pinned[path], record, outcome.source_locations,
                                tuple(candidates), pinned, tuple(unavailable), outcome.unavailable_evidence)
            verify([snapshot])
            result.append(snapshot)
        except (OSError, TypeError, ValueError, KeyError, AttributeError, IndexError, MutagenError, sf.LibsndfileError) as exc:
            raise CatalogHeldError(f"BGM保留結果を目録へ取り込めません: {path}: {exc}") from exc
    return result


def verify(records):
    for record in records:
        for path, digest in record.pinned.items():
            try:
                unchanged = file_sha256(path) == digest
            except OSError as exc:
                raise CatalogHeldError(f"目録確定前にBGM保留記録・音声を読めません: {path}") from exc
            if not unchanged:
                raise CatalogHeldError(f"目録構築中にBGM保留記録・音声が変更されています: {path}")


def insert(connection, records, sources):
    for snapshot in records:
        record = snapshot.record
        source_sha = record["source"]["sha256"]
        sources.setdefault(source_sha, set()).add(record["source"]["path"])
        declared = {key: record[key] for key in ("held_alternatives", "held_reference_residuals") if key in record}
        connection.execute("INSERT INTO held_results VALUES (?,?,?,?,?,?,?,?,?,?)", (
            str(snapshot.path), snapshot.digest, source_sha, json.dumps([str(p) for p in snapshot.source_locations]),
            record["recipe_id"], record["evidence"]["path"], "HELD", record["reason"],
            json.dumps(declared, ensure_ascii=False), json.dumps(snapshot.unavailable_branches)))
        for candidate in snapshot.candidates:
            origin = candidate["origin"]
            connection.execute("INSERT INTO held_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                candidate["candidate_id"], str(snapshot.path), source_sha, candidate["content_sha256"],
                candidate["role"], candidate["primary_candidate_id"], candidate["recorded_path"],
                json.dumps(candidate["current_paths"], ensure_ascii=False), origin["start_frame"], origin["end_frame"],
                origin["sample_rate"], origin.get("time_offset_numerator", 0), origin.get("time_offset_denominator", 1),
                json.dumps(origin, ensure_ascii=False), "unresolved_music_like", "not_published"))


def candidates_of(source_sha256: str, *, database: Path) -> list[dict]:
    """出典・元cueとの対応・全所在を返す。代替を別の登場区間や正式成果へ数えない。"""
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM held_candidates WHERE source_sha256 = ?"
                                  " ORDER BY start_frame, end_frame, role, candidate_id", (source_sha256,)).fetchall()
    return [{**dict(row), "current_paths": json.loads(row["current_paths"]), "origin": json.loads(row["origin"])} for row in rows]


def summary(*, database: Path) -> dict:
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        counts = dict(connection.execute("SELECT role, COUNT(*) FROM held_candidates GROUP BY role"))
        results = connection.execute("SELECT COUNT(*) FROM held_results").fetchone()[0]
        occurrences = connection.execute("SELECT COUNT(*) FROM (SELECT source_sha256,start_frame,end_frame,sample_rate,"
                                         "offset_numerator,offset_denominator FROM held_candidates"
                                         " WHERE role='primary' GROUP BY source_sha256,start_frame,end_frame,sample_rate,"
                                         "offset_numerator,offset_denominator)").fetchone()[0]
    return {"held_results": results, "primary_cues": counts.get("primary", 0),
            "music_alternatives": counts.get("music_alternative", 0), "reference_residuals": counts.get("reference_residual", 0),
            "logical_occurrences": occurrences, "published_outputs": 0}
