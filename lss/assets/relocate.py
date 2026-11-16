"""移動された原本・成果物を内容のSHA-256で見つけ、来歴をたどれるようにする。

- 完了台帳に記録された場所から消えた原本（入力SHA-256）と成果物（出力SHA-256）を、利用者が指定した
  フォルダの中から、同じ内容のファイルとして探す。名前や場所は手がかりにしない。
- 見つけた移動先は、台帳を書き換えずに追記の記録（状態領域の `relocations.json`）へ残す。
- 同じ内容のファイルが複数あれば、すべてを候補として並べ、1つに決めつけない。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from lss.core import audio_identity, session
from lss.harness.tsv_io import load_tsv

RECORD_NAME = "relocations.json"
# tagの書き戻しが増やし得る容器metadataの上限。復号する候補をこの範囲へ絞る。
MAX_CONTAINER_METADATA_BYTES = 1 << 20
SEARCH_EXTENSIONS = frozenset({".flac", ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aif", ".aiff",
                               ".webm", ".ts", ".mp4", ".mkv", ".mka"})


class RelocationError(RuntimeError):
    """移動の来歴を確認できないため、既存記録へ追記しない。"""


def _validate_record(payload) -> None:
    if (not isinstance(payload, dict) or type(payload.get("schema")) is not int or payload["schema"] != 1
            or not isinstance(payload.get("entries"), list)):
        raise RelocationError("移動記録の形式を確認できません")
    for entry in payload["entries"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or not isinstance(entry.get("kind"), str) or entry["kind"] not in {"source", "output"}
                or not isinstance(entry.get("path"), str) or not entry["path"].strip()):
            raise RelocationError("移動記録の出典・所在を確認できません")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RelocationError("移動記録をJSONへ保存できません") from exc


def _read_record(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": 1, "entries": []}
    except (OSError, ValueError) as exc:
        raise RelocationError("既存の移動記録を読めません。上書きせず保持します") from exc
    _validate_record(payload)
    return payload


def _missing(state_root: Path | None, held_outcomes: list[dict] = ()) -> dict[str, dict]:
    """内容SHA-256 → 記録された場所と種類（記録どおりの場所に残っていないものだけ）。"""
    try:
        _, rows = load_tsv(session.completion_ledger_path(state_root))
    except FileNotFoundError:
        rows = []
    except (OSError, ValueError) as exc:
        raise RelocationError("完了台帳を読めないため移動対象を確認できません") from exc
    wanted: dict[str, dict] = {}

    def add(sha256: str, path: str, kind: str, audio: object = None) -> None:
        if not sha256 or not path:
            return
        recorded = Path(path)
        try:
            # tagだけが書き戻された成果物は、その場所に残っている。失われたものとして探さない。
            if recorded.is_file() and audio_identity.diagnose(
                    recorded, recorded_sha256=sha256, recorded_audio=audio,
                    observed_sha256=session.input_sha256_of(recorded)) in audio_identity.REUSABLE_STATES:
                return
        except OSError:
            pass
        entry = wanted.setdefault(sha256, {"kind": kind, "recorded_paths": set(), "audio": None})
        if entry.get("audio") is None:
            entry["audio"] = audio
        entry["recorded_paths"].add(path)

    for row in rows:
        add(row.get("input_sha256") or "", row.get("source_path") or "", "source")
        try:
            outputs = json.loads(row.get("outputs") or "[]")
        except ValueError:
            outputs = []
        for item in outputs if isinstance(outputs, list) else []:
            if isinstance(item, dict):
                add(str(item.get("sha256") or ""), str(item.get("path") or ""), "output",
                    item.get("audio"))
    for outcome in held_outcomes:
        source = outcome["source"]
        add(source["sha256"], source["path"], "source")
        for item in outcome["held_artifacts"]:
            add(item["sha256"], item["path"], "output", item.get("audio"))
        for item in (outcome.get("held_alternatives") or {}).get("artifacts", []):
            add(item["sha256"], item["path"], "output", item.get("audio"))
        for item in (outcome.get("held_reference_residuals") or {}).get("artifacts", []):
            add(item["sha256"], item["path"], "output", item.get("audio"))
    return wanted


def _held_outcomes(paths) -> list[tuple[Path, bytes, dict]]:
    """保留記録の所在だけを読む。発見後の再利用には製品境界での全検証が必要。"""
    records = []
    for path in dict.fromkeys(Path(path).resolve() for path in paths):
        try:
            content = path.read_bytes()
            record = json.loads(content)
            if (not isinstance(record, dict) or record.get("type") != "processing_outcome"
                    or type(record.get("schema_version")) is not int or record["schema_version"] != 1
                    or record.get("intent") != "extract_bgm_cues" or record.get("disposition") != "HELD"
                    or record.get("publication_allowed") is not False or record.get("outputs") != []
                    or not isinstance(record.get("source"), dict) or not isinstance(record.get("held_artifacts"), list)):
                raise ValueError("非公開BGM保留結果の記録が必要です")
            alternatives = record.get("held_alternatives")
            if alternatives is not None and (not isinstance(alternatives, dict)
                                               or not isinstance(alternatives.get("artifacts"), list)):
                raise ValueError("BGM代替候補の所在記録が不正です")
            reference = record.get("held_reference_residuals")
            if reference is not None and (not isinstance(reference, dict) or not isinstance(reference.get("artifacts"), list)):
                raise ValueError("BGM参照残差分岐の所在記録が不正です")
            for item in [record["source"], *record["held_artifacts"], *(alternatives or {}).get("artifacts", []),
                         *(reference or {}).get("artifacts", [])]:
                if (not isinstance(item, dict) or not isinstance(item.get("sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                        or not isinstance(item.get("path"), str) or not item["path"].strip()):
                    raise ValueError("BGM保留結果の内容同一性・所在が不正です")
        except (OSError, ValueError, TypeError) as exc:
            raise RelocationError(f"移動対象のBGM保留結果を読めません: {path}") from exc
        records.append((path, content, record))
    return records


def find_moved(search_roots: list[Path], *, state_root: Path | None = None,
               outcome_records: list[Path] = ()) -> list[dict]:
    """完了台帳と指定したBGM保留結果から、移動した素材を内容で探す。"""
    snapshots = _held_outcomes(outcome_records)
    wanted = _missing(state_root, [record for _, _, record in snapshots])
    if not wanted:
        return []
    found: dict[str, set[str]] = {}
    for root in search_roots:
        for folder, _, names in os.walk(root):
            for name in names:
                path = Path(folder) / name
                if path.suffix.lower() not in SEARCH_EXTENSIONS:
                    continue
                try:
                    digest = session.input_sha256_of(path)
                    size = path.stat().st_size
                except OSError:
                    continue
                if digest in wanted:
                    found.setdefault(digest, set()).add(str(path.resolve()))
                    continue
                # 移動先でtagが書き戻された成果物だけ、復号したsampleで照合する。
                plausible = [sha256 for sha256, entry in wanted.items()
                             if _metadata_sized(entry.get("audio"), size)
                             and audio_identity.header_matches(path, entry["audio"])]
                if not plausible:
                    continue
                try:
                    observed = audio_identity.audio_identity(path)
                except (audio_identity.AudioIdentityError, OSError):
                    continue
                for sha256 in plausible:
                    if (audio_identity.same_audio(wanted[sha256]["audio"], observed)
                            and audio_identity.stream_complete(path, observed)):
                        found.setdefault(sha256, set()).add(str(path.resolve()))
    for path, content, _ in snapshots:
        try:
            if path.read_bytes() != content:
                raise RelocationError(f"所在の検索中にBGM保留結果が変更されています: {path}")
        except OSError as exc:
            raise RelocationError(f"所在の検索後にBGM保留結果を確認できません: {path}") from exc
    return [{"sha256": sha256, "kind": wanted[sha256]["kind"],
             "recorded_paths": sorted(wanted[sha256]["recorded_paths"]), "found_at": sorted(paths),
             **({"audio": wanted[sha256]["audio"]} if wanted[sha256].get("audio") else {})}
            for sha256, paths in sorted(found.items())]


def _metadata_sized(audio: object, size: int) -> bool:
    """容器metadataの増減だけで説明できる大きさか。無関係なファイルを復号しない。"""
    if not isinstance(audio, dict) or type(audio.get("audio_bytes")) is not int:
        return False
    return audio["audio_bytes"] <= size <= audio["audio_bytes"] + MAX_CONTAINER_METADATA_BYTES


def record_relocations(moves: list[dict], *, state_root: Path | None = None) -> Path:
    """見つけた移動先を追記する。同じ内容・同じ移動先は重複させない。"""
    from lss.acquisition.store import _write_json_atomic

    path = (Path(state_root) if state_root is not None else session.state_dir()) / RECORD_NAME
    payload = _read_record(path)
    if not isinstance(moves, list):
        raise RelocationError("追記する移動候補の形式が不正です")
    known = {(entry["sha256"], entry["path"]) for entry in payload["entries"]}
    changed = False
    moment = datetime.now(timezone.utc).isoformat()
    for move in moves:
        if (not isinstance(move, dict) or not isinstance(move.get("found_at"), list)
                or not isinstance(move.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", move["sha256"])
                or not isinstance(move.get("kind"), str) or move["kind"] not in {"source", "output"}
                or any(not isinstance(location, str) or not location.strip() for location in move["found_at"])
                or not isinstance(move.get("recorded_paths"), list)
                or any(not isinstance(location, str) for location in move["recorded_paths"])):
            raise RelocationError("追記する移動候補の所在が不正です")
        for location in move["found_at"]:
            if (move.get("sha256"), location) in known:
                continue
            payload["entries"].append({"sha256": move.get("sha256"), "kind": move.get("kind"), "path": location,
                                       "recorded_paths": move["recorded_paths"], "found_at": moment,
                                       **({"audio": move["audio"]} if move.get("audio") else {})})
            known.add((move.get("sha256"), location))
            changed = True
    _validate_record(payload)
    if changed:
        _write_json_atomic(path, payload)
    return path


def current_locations(sha256: str, *, state_root: Path | None = None) -> list[Path]:
    """記録した移動先のうち、今も同じ内容で残っている場所。"""
    path = (Path(state_root) if state_root is not None else session.state_dir()) / RECORD_NAME
    try:
        entries = _read_record(path)["entries"]
    except RelocationError:
        return []
    locations = []
    for entry in entries:
        if entry.get("sha256") != sha256:
            continue
        candidate = Path(entry["path"])
        try:
            if candidate.is_file() and audio_identity.diagnose(
                    candidate, recorded_sha256=sha256, recorded_audio=entry.get("audio"),
                    observed_sha256=session.input_sha256_of(candidate)) in audio_identity.REUSABLE_STATES:
                locations.append(candidate)
        except OSError:
            continue
    return locations
