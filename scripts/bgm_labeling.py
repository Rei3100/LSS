"""意味的roleの正解を先に固定するための、人手ラベルの作業票と取り込み。

保存済みのBGM保留結果から、候補ごとの区間・長さ・音声の所在・録音照合の根拠を並べた
作業票を作る（`draft`）。`semantic_role` と `wanted` は空のままで、推測して埋めない。
人が試聴して埋めた作業票を、検証コーパスの形式へ取り込む（`compile`）。

ラベルは評価の正解であり、自動選択の条件ではない。ここでは音声を作らず、保留結果・
候補音声・既存の記録を一切変更しない。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf

from lss.bgm import identity_candidates
from lss.harness.fetch import file_sha256
from lss.validation.bgm_corpus import EXPECTED_OUTCOMES, SEMANTIC_ROLES, load_bgm_corpus

SCHEMA_VERSION = 1
WORKLIST_TYPE = "bgm_labeling_worklist"
AWAITING = "AWAITING_HUMAN_LABELS"
LABELLED = "HUMAN_LABELS_COMPLETE"
# 欲しい劇伴かどうかまでは人が決める。ここでは選べる値だけを示す。
CHOICES = sorted(SEMANTIC_ROLES - {"unresolved_music_like"})


class LabelingError(ValueError):
    """作業票、またはそこへ書かれたラベルが契約に合わない。"""


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise LabelingError(f"JSONキーが重複しています: {key}")
        value[key] = item
    return value


def _decode_json(content: bytes, *, context: str) -> dict:
    try:
        value = json.loads(content, object_pairs_hook=_object_without_duplicate_keys)
    except LabelingError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LabelingError(f"{context}のJSONが不正です") from exc
    if not isinstance(value, dict):
        raise LabelingError(f"{context}はJSONオブジェクトである必要があります")
    return value


def read_json(path: Path) -> dict:
    path = Path(path)
    try:
        return _decode_json(path.read_bytes(), context=str(path))
    except OSError as exc:
        raise LabelingError(f"JSONを読めません: {path}") from exc


def _write_new_file_atomic(output: Path, payload: bytes, *, validate=None) -> None:
    """新規成果物だけを、検証後に原子的に作る。既存記録は上書きしない。"""
    output = Path(output)
    if output.exists():
        raise LabelingError(f"保存先が既にあります: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
                dir=output.parent, prefix=f".{output.name}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if validate is not None:
            validate(temporary)
        # 入力検証中に別の成果物が現れた場合も、既存記録を置き換えない。
        if output.exists():
            raise LabelingError(f"保存先が検証中に作られました: {output}")
        try:
            temporary.rename(output)
        except FileExistsError as exc:
            raise LabelingError(f"保存先が検証後に競合したため、上書きしません: {output}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _matching_location(paths, expected_sha256: str, context: str) -> Path:
    """移動後の所在も許すが、内容SHAが一致する実体だけを返す。"""
    seen = set()
    for raw_path in paths:
        try:
            path = Path(raw_path).resolve()
        except (OSError, TypeError, ValueError):
            continue
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            if file_sha256(path) == expected_sha256:
                return path
        except OSError:
            continue
    raise LabelingError(f"{context}が記録された内容と一致する所在にありません")


def _required_mapping(value, context: str) -> dict:
    if not isinstance(value, dict):
        raise LabelingError(f"{context}の来歴が不正です")
    return value


def _required_text(value, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LabelingError(f"{context}が未指定です")
    return value


def _required_positive_int(value, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LabelingError(f"{context}は正の整数である必要があります")
    return value


def _verified_worklist_inputs(worklist: dict) -> dict:
    """作業票のラベル以外が、現在の保留結果と同じ実体を指すことを確認する。"""
    from lss import application

    held = _required_mapping(worklist.get("held_result"), "held_result")
    evidence = _required_mapping(worklist.get("evidence"), "evidence")
    source = _required_mapping(worklist.get("source"), "source")
    origin_media = _required_mapping(worklist.get("origin_media"), "origin_media")
    held_path = Path(_required_text(held.get("path"), "held_result.path")).resolve()
    evidence_path = Path(_required_text(evidence.get("path"), "evidence.path")).resolve()
    source_recorded_path = Path(_required_text(source.get("path"), "source.path")).resolve()
    held_sha256 = _required_text(held.get("sha256"), "held_result.sha256")
    evidence_sha256 = _required_text(evidence.get("sha256"), "evidence.sha256")
    source_sha256 = _required_text(source.get("sha256"), "source.sha256")

    try:
        outcome = application.read_bgm_development_outcome(held_path)
    except Exception as exc:
        raise LabelingError(f"現在読み戻せる保留結果が必要です: {exc}") from exc
    if outcome.disposition is not application.ResultDisposition.HELD:
        raise LabelingError("現在読み戻せる保留結果が必要です")
    if outcome.evidence_path is None or outcome.record_path is None:
        raise LabelingError("保留結果に根拠または記録の所在がありません")
    if outcome.record_path.resolve() != held_path:
        raise LabelingError("作業票の保留結果の所在が一致しません")
    if outcome.evidence_path.resolve() != evidence_path:
        raise LabelingError("作業票の根拠の所在が保留結果と一致しません")
    try:
        if file_sha256(held_path) != held_sha256:
            raise LabelingError("作業票を作ったときの保留結果から内容が変わっています")
        if file_sha256(evidence_path) != evidence_sha256:
            raise LabelingError("作業票を作ったときの根拠から内容が変わっています")
    except OSError as exc:
        raise LabelingError("作業票の保留結果または根拠を読めません") from exc

    report = read_json(evidence_path)
    signal = _required_mapping(_required_mapping(report.get("rendering"), "rendering").get("signal"),
                               "rendering.signal")
    if (source.get("path") != signal.get("path") or source_sha256 != signal.get("sha256")):
        raise LabelingError("作業票の解析信号がBGM根拠と一致しません")
    source_path = _matching_location(
        [source_recorded_path, *outcome.source_locations], source_sha256, "原本の解析信号")
    try:
        source_info = sf.info(source_path)
    except (OSError, RuntimeError) as exc:
        raise LabelingError("作業票の解析信号を読めません") from exc
    sample_rate = _required_positive_int(source.get("sample_rate"), "source.sample_rate")
    frames = _required_positive_int(source.get("frames"), "source.frames")
    if (source_info.samplerate != sample_rate or source_info.frames != frames
            or source_info.channels != 2):
        raise LabelingError("作業票の解析信号のstereoまたは時間軸が変わっています")

    report_source = _required_mapping(report.get("source"), "根拠source")
    if (origin_media.get("path") != report_source.get("path")
            or origin_media.get("sha256") != report_source.get("sha256")):
        raise LabelingError("作業票の原本来歴がBGM根拠と一致しません")

    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise LabelingError("BGM根拠の候補一覧が不正です")
    if len(candidates) != len(outcome.held_artifacts):
        raise LabelingError("BGM根拠と保留候補の件数が一致しません")

    identity_states = {}
    if outcome.identity_evidence_path is not None:
        try:
            identity_states = identity_candidates.candidate_identity_states(
                identity_candidates.annotation_view(evidence_path))
        except Exception as exc:
            raise LabelingError("作業票に結び付いた録音照合を読み戻せません") from exc

    candidate_rows = []
    candidate_ids = []
    for index, (candidate, held_artifact) in enumerate(zip(candidates, outcome.held_artifacts)):
        candidate = _required_mapping(candidate, f"候補[{index}]")
        asset = _required_mapping(candidate.get("asset"), f"候補[{index}].asset")
        origin = _required_mapping(candidate.get("origin"), f"候補[{index}].origin")
        candidate_id = _required_text(asset.get("candidate_asset_id"), f"候補[{index}].candidate_asset_id")
        if candidate_id in candidate_ids:
            raise LabelingError(f"BGM候補IDが重複しています: {candidate_id}")
        candidate_ids.append(candidate_id)
        asset_sha256 = _required_text(asset.get("sha256"), f"{candidate_id}.sha256")
        sample_rate_origin = _required_positive_int(origin.get("sample_rate"),
                                                     f"{candidate_id}.sample_rate")
        start_frame = origin.get("start_frame")
        end_frame = origin.get("end_frame")
        if (not isinstance(start_frame, int) or isinstance(start_frame, bool) or start_frame < 0
                or not isinstance(end_frame, int) or isinstance(end_frame, bool)
                or end_frame <= start_frame or end_frame > source_info.frames
                or sample_rate_origin != source_info.samplerate):
            raise LabelingError(f"{candidate_id}の候補区間が解析信号の時間軸と一致しません")
        asset_path = Path(_required_text(asset.get("path"), f"{candidate_id}.path")).resolve()
        locations = outcome.held_artifact_locations.get(asset_path, ())
        artifact_path = _matching_location(
            [held_artifact, *locations], asset_sha256, f"{candidate_id}候補音声")
        candidate_rows.append({
            "candidate": candidate, "asset": asset, "origin": origin,
            "artifact_path": artifact_path,
            "recording_identity": identity_states.get(candidate_id) or None,
        })

    raw_occurrences = worklist.get("occurrences")
    if not isinstance(raw_occurrences, list) or len(raw_occurrences) != len(candidate_rows):
        raise LabelingError("作業票のoccurrence件数が保留候補と一致しません")
    by_id = {row["asset"]["candidate_asset_id"]: row for row in candidate_rows}
    seen_ids = set()
    for index, raw in enumerate(raw_occurrences):
        raw = _required_mapping(raw, f"occurrence[{index}]")
        occurrence_id = _required_text(raw.get("occurrence_id"), f"occurrence[{index}].occurrence_id")
        if occurrence_id in seen_ids or occurrence_id not in by_id:
            raise LabelingError(f"作業票のoccurrenceが保留候補と一致しません: {occurrence_id}")
        seen_ids.add(occurrence_id)
        row = by_id[occurrence_id]
        origin = row["origin"]
        if (raw.get("start_frame") != origin["start_frame"]
                or raw.get("end_frame") != origin["end_frame"]):
            raise LabelingError(f"{occurrence_id}の区間が保留候補から変わっています")
        seconds = raw.get("seconds")
        expected_seconds = (origin["end_frame"] - origin["start_frame"]) / origin["sample_rate"]
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(float(seconds)) or abs(float(seconds) - expected_seconds) > 0.001):
            raise LabelingError(f"{occurrence_id}の長さが保留候補から変わっています")
        if raw.get("audio_sha256") != row["asset"]["sha256"]:
            raise LabelingError(f"{occurrence_id}の候補音声のSHA-256が変わっています")
        audio_path = _required_text(raw.get("audio_path"), f"{occurrence_id}.audio_path")
        _matching_location(
            [audio_path, row["artifact_path"]], row["asset"]["sha256"],
            f"{occurrence_id}候補音声")
        if "recording_identity" not in raw:
            raise LabelingError(f"{occurrence_id}の録音照合の記録が欠落しています")
        if raw.get("recording_identity") != row["recording_identity"]:
            raise LabelingError(f"{occurrence_id}の録音照合の記録が変わっています")
    if seen_ids != set(by_id):
        raise LabelingError("作業票のoccurrenceに保留候補の欠落があります")
    return {"outcome": outcome, "report": report, "source_path": source_path,
            "source_info": source_info, "candidate_rows": candidate_rows}


def draft(record_path: Path, output: Path) -> Path:
    """保留結果から作業票を作る。roleとwantedは空にしておく。"""
    from lss import application

    record_path, output = Path(record_path).resolve(), Path(output).resolve()
    outcome = application.read_bgm_development_outcome(record_path)
    if outcome.disposition is not application.ResultDisposition.HELD:
        raise LabelingError(f"読み戻せる保留結果が必要です: {outcome.message}")
    report_path = Path(outcome.evidence_path)
    record_bytes = record_path.read_bytes()
    report_bytes = report_path.read_bytes()
    report = _decode_json(report_bytes, context=str(report_path))
    states = identity_candidates.candidate_identity_states(
        identity_candidates.annotation_view(report_path))
    origin_media = report["source"]
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(outcome.held_artifacts):
        raise LabelingError("BGM根拠と保留候補の件数が一致しません")
    # 候補の時刻は解析信号のframeで記録されている。ラベルも同じ時間軸へ置く。
    signal = report["rendering"]["signal"]
    if file_sha256(Path(signal["path"])) != signal["sha256"]:
        raise LabelingError("解析信号が記録と一致しません")
    info = sf.info(signal["path"])
    occurrences = []
    for candidate, place in zip(candidates, outcome.held_artifacts):
        asset, origin = candidate["asset"], candidate["origin"]
        identity = states.get(asset["candidate_asset_id"], {})
        occurrences.append({
            "occurrence_id": asset["candidate_asset_id"],
            "start_frame": origin["start_frame"], "end_frame": origin["end_frame"],
            "seconds": round((origin["end_frame"] - origin["start_frame"]) / origin["sample_rate"], 3),
            "audio_path": str(Path(place).resolve()), "audio_sha256": asset["sha256"],
            "recording_identity": identity or None,
            # 人が試聴して埋める欄。推測で埋めない。
            "semantic_role": None, "wanted": None, "novel": None})
    worklist = {
        "schema_version": SCHEMA_VERSION, "type": WORKLIST_TYPE, "status": AWAITING,
        "held_result": {"path": str(record_path), "sha256": file_sha256(record_path)},
        "evidence": {"path": str(report_path), "sha256": file_sha256(report_path)},
        "source": {"path": signal["path"], "sha256": signal["sha256"],
                   "sample_rate": int(info.samplerate), "frames": int(info.frames)},
        "origin_media": {"path": origin_media["path"], "sha256": origin_media["sha256"]},
        "semantic_role_choices": CHOICES,
        "instructions": ("各occurrenceを試聴し、semantic_roleを選択肢から選び、wantedを真偽で書く。"
                         "既に公開した同じcueの再出現だけnovelをfalseにする。"
                         "判断できない区間はsemantic_roleを空のまま残す（取り込みで拒否される）。"),
        "occurrences": occurrences}
    if output.exists():
        raise LabelingError(f"作業票の保存先が既にあります: {output}")
    encoded = (json.dumps(worklist, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    def validate_inputs(_temporary: Path) -> None:
        try:
            if record_path.read_bytes() != record_bytes or report_path.read_bytes() != report_bytes:
                raise LabelingError("作業票の作成中に保留結果または根拠が変更されました")
            if file_sha256(Path(signal["path"])) != signal["sha256"]:
                raise LabelingError("作業票の作成中に解析信号が変更されました")
            for place, candidate in zip(outcome.held_artifacts, candidates):
                if file_sha256(place) != candidate["asset"]["sha256"]:
                    raise LabelingError("作業票の作成中に候補音声が変更されました")
        except OSError as exc:
            raise LabelingError("作業票の作成中に入力を読めません") from exc

    _write_new_file_atomic(output, encoded, validate=validate_inputs)
    return output


def _labelled_occurrence(raw: dict) -> dict:
    role, wanted, novel = raw.get("semantic_role"), raw.get("wanted"), raw.get("novel")
    name = raw.get("occurrence_id")
    if role not in SEMANTIC_ROLES or role == "unresolved_music_like":
        raise LabelingError(f"{name}: semantic_roleを{CHOICES}から選んでください（現在: {role!r}）")
    if not isinstance(wanted, bool):
        raise LabelingError(f"{name}: wantedを真偽で書いてください（現在: {wanted!r}）")
    if novel is not None and not isinstance(novel, bool):
        raise LabelingError(f"{name}: novelは真偽またはnullです（現在: {novel!r}）")
    if not wanted and novel is not None:
        raise LabelingError(f"{name}: 欲しくない区間へnovelを設定できません")
    identity = raw.get("recording_identity") or {}
    state = identity.get("identity_state", "unresolved")
    if state not in {"unresolved", "unknown"}:
        raise LabelingError(f"{name}: 確定した同一性の取り込みは未対応です（{state}）")
    return {"occurrence_id": name, "start_frame": raw["start_frame"], "end_frame": raw["end_frame"],
            "semantic_role": role, "wanted": wanted, "novel": novel if wanted else None,
            "identity_state": state, "cue_identity_id": None, "audio_version_id": None}


def _expected_outcome(occurrences: list[dict]) -> str:
    wanted = [item for item in occurrences if item["wanted"]]
    if not occurrences:
        return "no_wanted_score"
    if not wanted:
        return "music_not_wanted"
    if all(item["novel"] is False for item in wanted):
        return "no_novel_asset"
    return "wanted_present"


def compile_manifest(worklist_path: Path, output: Path, *, corpus_id: str, revision: str,
                     case_id: str, description: str) -> Path:
    """埋めた作業票を検証コーパスへ取り込む。空欄が一つでもあれば拒否する。"""
    worklist_path, output = Path(worklist_path).resolve(), Path(output).resolve()
    try:
        worklist_bytes = worklist_path.read_bytes()
    except OSError as exc:
        raise LabelingError(f"作業票を読めません: {worklist_path}") from exc
    worklist = _decode_json(worklist_bytes, context=str(worklist_path))
    if (worklist.get("type") != WORKLIST_TYPE or worklist.get("schema_version") != SCHEMA_VERSION
            or worklist.get("status") not in {AWAITING, LABELLED}
            or not isinstance(worklist.get("origin_media"), dict)):
        raise LabelingError("この道具が作った作業票が必要です")
    verified = _verified_worklist_inputs(worklist)
    source = worklist["source"]
    occurrences = [_labelled_occurrence(raw) for raw in worklist.get("occurrences", [])]
    outcome = _expected_outcome(occurrences)
    if outcome not in EXPECTED_OUTCOMES:
        raise LabelingError(f"導けない期待結果です: {outcome}")
    lines = [
        {"type": "manifest", "schema_version": 1, "corpus_id": corpus_id, "revision": revision,
         "description": description},
        {"type": "case", "case_id": case_id, "description": description,
         "source": {"path": str(verified["source_path"]), "sha256": source["sha256"],
                    "sample_rate": verified["source_info"].samplerate,
                    "frames": verified["source_info"].frames},
         "expected_outcome": outcome,
         "attributes": ["human_labelled", "real_episode", "routed_analysis_signal"],
         "occurrences": occurrences},
    ]
    encoded = ("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n").encode("utf-8")

    def validate_written(temporary: Path) -> None:
        try:
            load_bgm_corpus(temporary, verify_sources=True)
        except Exception as exc:
            raise LabelingError(f"生成したBGM corpusを検証できません: {exc}") from exc
        # tempを検証した後、入力一式をもう一度読む。遅い変更時はtempも公開しない。
        try:
            if worklist_path.read_bytes() != worklist_bytes:
                raise LabelingError("コーパス取り込み中に作業票が変更されました")
        except OSError as exc:
            raise LabelingError("コーパス取り込み中に作業票を確認できません") from exc
        refreshed = _decode_json(worklist_bytes, context=str(worklist_path))
        refreshed_inputs = _verified_worklist_inputs(refreshed)
        refreshed_occurrences = [
            _labelled_occurrence(raw) for raw in refreshed.get("occurrences", [])
        ]
        if (_expected_outcome(refreshed_occurrences) != outcome
                or refreshed_inputs["source_path"] != verified["source_path"]):
            raise LabelingError("コーパス取り込み中に作業票の判定入力が変わりました")

    _write_new_file_atomic(output, encoded, validate=validate_written)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    drafting = commands.add_parser("draft", help="保留結果から人手ラベルの作業票を作る")
    drafting.add_argument("--record", type=Path, required=True)
    drafting.add_argument("--out", type=Path, required=True)
    compiling = commands.add_parser("compile", help="埋めた作業票を検証コーパスへ取り込む")
    compiling.add_argument("--worklist", type=Path, required=True)
    compiling.add_argument("--out", type=Path, required=True)
    compiling.add_argument("--corpus-id", required=True)
    compiling.add_argument("--revision", required=True)
    compiling.add_argument("--case-id", required=True)
    compiling.add_argument("--description", required=True)
    args = parser.parse_args()
    if args.command == "draft":
        print(draft(args.record, args.out), flush=True)
    else:
        print(compile_manifest(args.worklist, args.out, corpus_id=args.corpus_id,
                               revision=args.revision, case_id=args.case_id,
                               description=args.description), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
