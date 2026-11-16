"""録音照合で分けた未解決区間を、出典付きの未公開Development音声として保存する。"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import soundfile as sf

from lss.bgm import candidate_store, identity, identity_run
from lss.output import writer

RECORD_NAME = "identity_candidates.json"
SCHEMA = 1
BOUNDARY_POLICY = "media_seconds_to_nearest_signal_frame_half_even_v1"


class IdentityCandidatesError(RuntimeError):
    """同一性を確認できない区間を、切り出した候補として確定しない。"""


def _source_hashes() -> dict:
    return {"implementation_sha256": identity_run._sha256(Path(__file__)),
            "candidate_store_sha256": identity_run._sha256(Path(candidate_store.__file__)),
            "writer_sha256": identity_run._sha256(Path(writer.__file__)),
            "identity_sha256": identity_run._sha256(Path(identity.__file__)),
            "identity_run_sha256": identity_run._sha256(Path(identity_run.__file__))}


_SOURCES_AT_IMPORT = _source_hashes()


def _revision() -> dict:
    # 読み込み後・呼び出し前のファイル編集も、実行済みコードの版として記録しない。
    return {"schema_version": SCHEMA, "boundary_policy": BOUNDARY_POLICY, **_SOURCES_AT_IMPORT}


def _load(run_path: Path) -> tuple[dict, dict, dict, Path | None]:
    run_bytes = run_path.read_bytes()
    run = json.loads(run_bytes)
    if not isinstance(run, dict) or not identity_run.is_annotatable(run):
        raise IdentityCandidatesError("完了した番組候補runが必要です")
    annotation_path = run_path.parent / identity_run.RECORD_NAME
    annotation_bytes = annotation_path.read_bytes()
    annotation = json.loads(annotation_bytes)
    if not isinstance(annotation, dict) or not isinstance(annotation.get("intent"), dict):
        raise IdentityCandidatesError("録音照合の記録が不正です")
    raw = identity_run.candidate_sets(run)
    references = annotation["intent"].get("references")
    if not isinstance(references, list):
        raise IdentityCandidatesError("照合の参照録音の記録が不正です")
    signal = kind = signal_sha = None
    if references or any(raw.values()):
        signal, kind, signal_sha = identity_run._query_signal(run)
    if kind == "source" and any(raw.values()):
        raise IdentityCandidatesError("切り出しには記録どおりの分離または描画信号が必要です")
    run_sha = hashlib.sha256(run_bytes).hexdigest()
    if not identity_run.is_current_annotation(
            annotation, run_sha256=run_sha,
            query_signal={"kind": kind, "sha256": signal_sha} if references else None):
        raise IdentityCandidatesError("現在のrun・信号・判定条件に対応する録音照合が必要です")
    intent = {**_revision(), "run_sha256": run_sha,
              "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
              "signal_sha256": signal_sha, "annotation_settings": identity_run.evidence_settings()}
    return run, annotation, intent, signal


def _verify(run_path: Path, annotation: dict, intent: dict, signal: Path | None) -> None:
    if (identity_run._sha256(run_path) != intent["run_sha256"]
            or identity_run._sha256(run_path.parent / identity_run.RECORD_NAME) != intent["annotation_sha256"]
            or (signal is not None and identity_run._sha256(signal) != intent["signal_sha256"])):
        raise IdentityCandidatesError("切り出し中にrun・照合記録・信号が変更されています")
    revision = _revision()
    if (_source_hashes() != _SOURCES_AT_IMPORT or revision != {k: intent[k] for k in revision}
            or identity_run.evidence_settings() != intent["annotation_settings"]):
        raise IdentityCandidatesError("切り出し中に実装または判定条件が変更されています")
    for reference in annotation["intent"]["references"]:
        if identity_run._sha256(Path(reference["path"])) != reference["sha256"]:
            raise IdentityCandidatesError("照合に使った参照録音が変更されています")
    identity_run.candidate_sets(json.loads(run_path.read_bytes()))


def _activities(run: dict, annotation: dict) -> tuple[list[dict], str]:
    """照合がなくても未知の原候補を保持し、照合済みの分割は欠落がないことを確かめる。"""
    raw = identity_run.candidate_sets(run)
    if not annotation["intent"]["references"]:
        return [{"recipe_id": recipe, "candidates": candidates} for recipe, candidates in raw.items()], "raw_unmatched_candidates"
    activities = annotation["activity_runs"]
    if len(activities) != len(raw) or {a["recipe_id"] for a in activities} != set(raw):
        raise IdentityCandidatesError("照合記録が元の候補集合と対応しません")
    for activity in activities:
        parents = {c["candidate_id"]: c for c in raw[activity["recipe_id"]]}
        children = {key: [] for key in parents}
        for candidate in activity["candidates"]:
            key = candidate.get("derived_from", candidate["candidate_id"])
            if key not in children:
                raise IdentityCandidatesError("分割候補に対応する元の候補がありません")
            children[key].append(candidate)
        for key, pieces in children.items():
            cursor = float(parents[key]["start_seconds"])
            for piece in sorted(pieces, key=lambda p: p["start_seconds"]):
                start, end = float(piece["start_seconds"]), float(piece["end_seconds"])
                if abs(start - cursor) > 1e-8 or not end > start:
                    raise IdentityCandidatesError("分割候補の時間範囲に欠落または重複があります")
                cursor = end
            if abs(cursor - float(parents[key]["end_seconds"])) > 1e-8:
                raise IdentityCandidatesError("分割候補が元候補の時間範囲を全て保持していません")
    return activities, "identity_split_candidates"


def annotation_view(run_path: Path) -> dict:
    """現在の入力に対応し、原候補の全区間を保持した録音照合を返す。

    音声の追加保存や既知曲の除外をせず、製品側の保留結果にも同じ照合根拠を渡す。
    参照録音・信号・判定条件が変わった記録は使用しない。
    """
    run_path = Path(run_path).resolve()
    try:
        run, annotation, intent, signal = _load(run_path)
        activities, _ = _activities(run, annotation)
        _verify(run_path, annotation, intent, signal)
        return {
            "path": str(run_path.parent / identity_run.RECORD_NAME),
            "sha256": intent["annotation_sha256"],
            "query_signal": annotation["query_signal"],
            "semantic_role": "unresolved",
            "activity_runs": activities,
        }
    except (OSError, TypeError, ValueError, KeyError, identity_run.IdentityRunError) as exc:
        raise IdentityCandidatesError("現在の候補へ録音照合の根拠を結び付けられません") from exc


def _union_seconds(spans: list[tuple[float, float]]) -> float:
    total, cursor_start, cursor_end = 0.0, None, None
    for start, end in sorted(spans):
        if cursor_end is None or start > cursor_end:
            if cursor_end is not None:
                total += cursor_end - cursor_start
            cursor_start, cursor_end = start, end
        else:
            cursor_end = max(cursor_end, end)
    return total + (cursor_end - cursor_start if cursor_end is not None else 0.0)


def candidate_identity_states(view: dict, *, tolerance_seconds: float = 0.05) -> dict[str, dict]:
    """元候補ID → 子候補から決めた同一性の状態と、その根拠。

    分けた子のうち `known_exact` が元候補の区間をすべて覆うときだけ `known_exact` とする。
    一部だけ一致した候補は `unresolved` のまま保ち、覆っている割合と版IDを返す。
    既知曲だからという理由で候補を捨てたり、まとめたりはしない。
    """
    parents: dict[str, list[dict]] = {}
    for activity in view["activity_runs"]:
        for item in activity["candidates"]:
            parents.setdefault(item.get("derived_from", item["candidate_id"]), []).append(item)
    states = {}
    for parent, children in parents.items():
        spans = [(float(child["start_seconds"]), float(child["end_seconds"])) for child in children]
        known = [(float(child["start_seconds"]), float(child["end_seconds"])) for child in children
                 if child.get("identity_state") == "known_exact"]
        total = _union_seconds(spans)
        covered = _union_seconds(known)
        fraction = covered / total if total > 0 else 0.0
        complete = total > 0 and covered >= total - tolerance_seconds
        states[parent] = {
            "identity_state": "known_exact" if complete else "unresolved",
            "known_exact_seconds": round(covered, 3), "candidate_seconds": round(total, 3),
            "known_exact_fraction": round(fraction, 4),
            "audio_version_ids": sorted({child["audio_version_id"] for child in children
                                         if child.get("audio_version_id")}),
            "children": sorted(child["candidate_id"] for child in children)}
    return states


def _check_destination(path: Path, source_sha256: str) -> None:
    if not path.exists():
        return
    try:
        previous = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise IdentityCandidatesError("保存先に読めない既存記録があります") from exc
    if (not isinstance(previous, dict) or previous.get("type") != "identity_development_candidates"
            or type(previous.get("schema_version")) is not int or previous.get("schema_version") != SCHEMA
            or not isinstance(previous.get("source"), dict)
            or previous["source"].get("sha256") != source_sha256):
        raise IdentityCandidatesError("保存先に別の出典または種類の記録があります")


def _origin(run: dict, activity: dict, signal_sha: str, sample_rate: int) -> tuple[str, str, Fraction, str]:
    if run.get("status") == identity_run.ROUTED_STATUS:
        origins = [c["origin"] for c in run.get("candidates") or []
                   if f"{c['origin']['separation_recipe_id']}+{c['origin']['activity_recipe_id']}" == activity["recipe_id"]]
        if not origins:
            raise IdentityCandidatesError("子候補の親となる描画記録がありません")
        first = origins[0]
        for origin in origins:
            candidate_store.CandidateOrigin(**origin).validate()
        fields = ("source_sha256", "sample_rate", "separation_recipe_id", "runtime_id", "signal_sha256", "time_offset_numerator",
                  "time_offset_denominator", "serialization_policy")
        if any(any(c.get(k) != first.get(k) for k in fields) for c in origins):
            raise IdentityCandidatesError("同じ候補集合の親信号・処理・時刻の来歴が一致しません")
        offset = Fraction(first.get("time_offset_numerator", 0), first.get("time_offset_denominator", 1))
        if (first["signal_sha256"] != signal_sha or first["source_sha256"] != run["source"]["sha256"]
                or first["sample_rate"] != sample_rate
                or abs(float(offset) - identity_run.timeline_offset(run)) > 1e-8):
            raise IdentityCandidatesError("照合信号が子候補の親信号と一致しません")
        return (first["separation_recipe_id"], first["runtime_id"],
                offset,
                first.get("serialization_policy", "native_range"))
    separation = run["separation"]
    timeline = run["source"].get("audio_timeline_origin") or {}
    offset = (Fraction(timeline["offset_numerator"], timeline["offset_denominator"])
              if "offset_numerator" in timeline else Fraction(str(identity_run.timeline_offset(run))))
    return separation["recipe_id"], separation["runtime_id"], offset, "native_range"


def materialize(run_path: Path, output_dir: Path) -> Path:
    """最新のidentity.jsonの子候補を保存する。正式公開・元runの変更・意味的昇格はしない。"""
    run_path, output_dir = Path(run_path).resolve(), Path(output_dir).resolve()
    try:
        run, annotation, intent, signal = _load(run_path)
        _verify(run_path, annotation, intent, signal)
        activities, basis = _activities(run, annotation)
        info = sf.info(signal) if any(a["candidates"] for a in activities) else None
        if info is not None and info.channels != 2:
            raise IdentityCandidatesError("子候補の元信号は完成ステレオである必要があります")
        source = run["source"]
        record_path = output_dir / RECORD_NAME
        _check_destination(record_path, source["sha256"])
        source_path = Path(source["path"])
        source_id = str(source.get("source_id") or run.get("source_id") or source["sha256"])
        records = []
        for activity in activities:
            if not activity["candidates"]:
                continue
            separation_id, runtime_id, offset, policy = _origin(run, activity, intent["signal_sha256"], info.samplerate)
            # 描画runのrecipe_idは分離+activity。元のactivityだけを保存来歴へ使う。
            activity_id = (activity["recipe_id"].removeprefix(separation_id + "+")
                           if run.get("status") == identity_run.ROUTED_STATUS else activity["recipe_id"])
            for candidate, decision in zip(activity["candidates"], identity.novelty_decisions(activity["candidates"])):
                row = {"annotation_candidate_id": candidate["candidate_id"], "candidate": candidate,
                       "activity_recipe_id": activity_id, "decision": decision["decision"]}
                if decision["decision"] == "known_recording_exists":
                    row["status"] = "known_recording_not_materialized"
                else:
                    start, end = [round((Fraction(str(candidate[key])) - offset) * info.samplerate)
                                  for key in ("start_seconds", "end_seconds")]
                    if not 0 <= start <= end <= info.frames:
                        raise IdentityCandidatesError("子候補の区間が親信号の時間軸を超えています")
                    row["frame_boundary"] = {"start_frame": start, "end_frame": end,
                                             "policy": BOUNDARY_POLICY}
                    if start == end:
                        row["status"] = "pending_subsample_interval_not_materializable"
                    else:
                        origin = candidate_store.CandidateOrigin(
                            source["sha256"], intent["signal_sha256"], separation_id, runtime_id, activity_id,
                            start, end, info.samplerate, offset.numerator, offset.denominator, policy)
                        row["asset"] = candidate_store.materialize_candidate(
                            signal, output_dir / "candidates" / f"{origin.candidate_id}.flac", origin=origin,
                            source_id=source_id, source_path=source_path,
                            evidence={"annotation_sha256": intent["annotation_sha256"],
                                      "annotation_candidate_id": candidate["candidate_id"],
                                      "boundary_policy": BOUNDARY_POLICY,
                                      "short_unresolved_remainder": candidate.get("short_unresolved_remainder", False)})
                        row["status"] = "materialized_unresolved_development_candidate"
                records.append(row)
        payload = {"type": "identity_development_candidates", "schema_version": SCHEMA, "intent": intent,
                   "status": "completed_development_candidates", "publication_state": "not_published",
                   "source": source, "signal": None if signal is None else {"path": str(signal), "sha256": intent["signal_sha256"]},
                   "candidate_basis": basis,
                   "semantic_role": "unresolved_music_like", "promotion_allowed": False, "candidates": records,
                   "zero_reason": identity.episode_zero_reason([r for a in activities
                                                               for r in identity.novelty_decisions(a["candidates"])])}
        output_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with NamedTemporaryFile(dir=output_dir, prefix=".identity-candidates-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            _verify(run_path, annotation, intent, signal)
            _check_destination(record_path, source["sha256"])
            # 同じ入力の再実行でも各音声を検証する。完成記録は同じ内容なら書き直さない。
            if not record_path.is_file() or record_path.read_text("utf-8") != encoded:
                # Windowsでは競合相手を上書きしない。相手が同一payloadならその完成記録を使う。
                try:
                    temporary.rename(record_path)
                except FileExistsError:
                    if (not record_path.is_file()
                            or record_path.read_text("utf-8") != encoded):
                        raise IdentityCandidatesError("保存先が別の候補記録で競合しています")
        finally:
            temporary.unlink(missing_ok=True)
        return record_path
    except IdentityCandidatesError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IdentityCandidatesError(f"未解決候補の切り出しを確定できません: {exc}") from exc
