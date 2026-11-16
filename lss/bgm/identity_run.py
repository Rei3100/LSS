"""完了した番組の候補runへ、既知録音との一致の証拠を別記録として付ける。

M3の候補run（`run.json` と activity 記録）は「番組だけから得た観測」として固定されている。
ここはそれを変更せず、同じrunのdirectoryに `identity.json` を追加する。

- 照合する信号は、runが記録した分離成果物（内容SHA-256が一致するもの）。無ければ原本。
- 候補の時刻は元メディアの時刻なので、一致の時刻にも音声の開始位置を足す。
- 参照録音は外部の証拠として、内容SHA-256とともに記録する。
- 一致しないことは不在の証拠にしない。候補の分割・判断材料は `identity.py` の規則に従う。
- 同じ番組の中で同じ録音が二度以上鳴る区間も探す。参照録音が無くても曲の範囲の根拠になる。
- 同じrun・同じ参照・同じ手法の記録が既にあれば、再計算せずに返す。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from lss.bgm import identity, self_recurrence
from lss.bgm.cue_extent import CueExtentRecipe
from lss.bgm.music_gap import MusicGapRecipe, extent_from_gaps, find_interior_gaps
from lss.bgm import occurrence_quality
from lss.bgm.recurring_extent import read_extents_for_candidates

RECORD_NAME = "identity.json"
RECORD_SCHEMA = 1
# 候補が一曲かどうかの読み取りを、結果を見る前に固定する。既知録音の一致も、同じ番組の
# 中の再演も、同じ強さ・覆い・ずれの広がりで読む。
CUE_EXTENT_RECIPE = CueExtentRecipe(
    minimum_similarity=identity.MIN_BLOCK_EXPLAINED,
    minimum_single_recording_coverage=0.8,
    maximum_offset_spread_seconds=0.5,
)
# 候補の内側で音楽が止まる箇所。止まってまた始まるなら、その前後は別のcueである。
# 実番組の固定読み取りで宣言した値をそのまま使う。
MUSIC_GAP_RECIPE = MusicGapRecipe(window_seconds=0.10, hop_seconds=0.025, floor_dbfs=-60.0,
                                  minimum_gap_seconds=1.5, edge_margin_seconds=0.5)


class IdentityRunError(RuntimeError):
    """候補runに一致の証拠を付けられない。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_signal(run: dict) -> tuple[Path, str, str]:
    """（照合する音声, 種別, SHA-256）。記録どおりの内容で残っている信号を使う。

    方向付き分岐のrunは、候補を切り出した描画済みの信号を優先する。元の番組runは分離成果物を優先する。
    どちらも無ければ原本。
    """
    signals = []
    rendering = (run.get("rendering") or {}).get("signal")
    if rendering:
        signals.append((rendering, "routed_signal"))
    stem_target = str(run.get("separation", {}).get("stem_target") or "").casefold()
    for artifact in run.get("separation", {}).get("artifacts") or []:
        if artifact.get("stem", "").casefold() == stem_target:
            signals.append((artifact, "separation_artifact"))
    signals.append((run.get("source") or {}, "source"))
    for item, kind in signals:
        path = Path(item.get("path") or "")
        if path.is_file() and item.get("sha256") and _sha256(path) == item["sha256"]:
            return path, kind, item["sha256"]
    raise IdentityRunError("照合に使う信号も原本も、記録どおりの内容で見つかりません")


ROUTED_STATUS = "completed_development_routed_episode_candidates"
EPISODE_STATUS = "completed_unannotated_episode_only_observation"


def is_annotatable(run: dict) -> bool:
    return (run.get("type") == "episode_bgm_discovery_run" and run.get("status") == EPISODE_STATUS)         or run.get("status") == ROUTED_STATUS


def timeline_offset(run: dict) -> float:
    """照合する信号の0秒が、元メディアの何秒か。"""
    source = run.get("source") or {}
    origin = source.get("audio_timeline_origin") or {}
    if origin.get("offset_seconds") is not None:
        return float(origin["offset_seconds"])
    for candidate in run.get("candidates") or []:
        item = candidate.get("origin") or {}
        if item.get("time_offset_denominator"):
            return item["time_offset_numerator"] / item["time_offset_denominator"]
    return 0.0


def candidate_sets(run: dict) -> dict[str, list[dict]]:
    """Recipeごとの候補（元メディアの時刻）。記録の内容が変わっていれば止める。"""
    if run.get("status") == ROUTED_STATUS:
        offset = timeline_offset(run)
        sets: dict[str, list[dict]] = {}
        for candidate in run.get("candidates") or []:
            origin = candidate["origin"]
            recipe = f"{origin['separation_recipe_id']}+{origin['activity_recipe_id']}"
            rate = float(origin["sample_rate"])
            sets.setdefault(recipe, []).append({
                "candidate_id": candidate["asset"]["candidate_asset_id"],
                "start_seconds": round(origin["start_frame"] / rate + offset, 6),
                "end_seconds": round(origin["end_frame"] / rate + offset, 6),
                "identity_state": "unresolved", "audio_version_id": None,
            })
        return sets
    sets = {}
    for activity_run in run.get("activity_runs") or []:
        record_file = Path(activity_run["record_path"])
        if _sha256(record_file) != activity_run.get("record_sha256"):
            raise IdentityRunError(f"activity記録が変更されています: {record_file}")
        record = json.loads(record_file.read_text(encoding="utf-8"))
        sets[activity_run["recipe_id"]] = [
            {key: item[key] for key in ("candidate_id", "start_seconds", "end_seconds", "identity_state", "audio_version_id")}
            for item in record.get("candidates") or []]
    return sets


OCCURRENCE_QUALITY_RECIPE = occurrence_quality.OccurrenceQualityRecipe(minimum_explained=0.5)


def occurrence_quality_view(matches, source_id: str, theme_reference_sha256) -> dict:
    """同じ番組の中の録音一致から、主題歌を外し最も重なりの少ない使用箇所を選ぶ。

    実番組の測定で、欲しい劇伴はほぼ常に台詞・効果音と重なり、同じ録音でも回によって
    重なりが大きく違うことが分かっている（`lv-bgm-region-routing-r1`）。分離の前に
    抽出元を選べるようにする。順位付けは候補を統合・削除しない。
    """
    occurrences = []
    for match in matches:
        if match.identity_state != "known_exact" or match.waveform_explained is None:
            continue
        if match.query_end_seconds <= match.query_start_seconds:
            continue
        occurrences.append({
            "occurrence_id": f"{source_id}@{match.query_start_seconds:.3f}",
            "recording_key": match.reference_sha256,
            "source_id": source_id,
            "start_seconds": match.query_start_seconds,
            "end_seconds": match.query_end_seconds,
            "waveform_explained": match.waveform_explained,
            "evidence_source": occurrence_quality.EVIDENCE_RELEASED,
        })
    if not occurrences:
        return {"recipe_id": OCCURRENCE_QUALITY_RECIPE.recipe_id, "recording_count": 0,
                "theme_excluded_count": 0, "selected_count": 0, "selected_above_minimum": 0,
                "groups": [], "claim_boundary": ["録音一致が無いので順位付けの対象がない"]}
    return occurrence_quality.select_best_occurrences(
        occurrences, OCCURRENCE_QUALITY_RECIPE,
        theme_recording_keys=theme_reference_sha256 or ())


def evidence_settings() -> dict:
    """録音一致の記録を作る側と利用する側で共有する、現在の証拠の条件。"""
    return {
        "schema_version": RECORD_SCHEMA,
        "fingerprint_settings_id": identity.fingerprint_settings_id(),
        "matching_settings_id": identity.matching_settings_id(),
        "decision_thresholds": {
            "min_block_explained": identity.MIN_BLOCK_EXPLAINED, "min_run_blocks": identity.MIN_RUN_BLOCKS,
            "min_explained_margin": identity.MIN_EXPLAINED_MARGIN, "min_seconds": identity.MIN_SECONDS,
            "min_candidate_coverage": identity.MIN_CANDIDATE_COVERAGE,
        },
        "implementation_sha256": _sha256(Path(identity.__file__)),
        "stage_sha256": _sha256(Path(__file__)),
        "repeat_method": self_recurrence.METHOD,
        "repeat_stage_sha256": _sha256(Path(self_recurrence.__file__)),
        "repeat_minimum_separation_seconds": self_recurrence.MINIMUM_SEPARATION_SECONDS,
        "cue_extent_recipe_id": CUE_EXTENT_RECIPE.recipe_id,
        "music_gap_recipe_id": MUSIC_GAP_RECIPE.recipe_id,
        "occurrence_quality_recipe_id": OCCURRENCE_QUALITY_RECIPE.recipe_id,
    }


def _interior_gaps(signal_path: Path, candidates: list[dict], offset: float) -> dict:
    """候補の区間を信号から切り出し、内側で音楽が止まる箇所を読む。

    読めない候補には読み取りを付けない。読めないことを「止まりが無い」にしない。
    """
    import soundfile as sf

    readings = {}
    try:
        with sf.SoundFile(str(signal_path)) as stream:
            rate, frames = int(stream.samplerate), len(stream)
            for candidate in candidates:
                start = int(round((float(candidate["start_seconds"]) - offset) * rate))
                end = int(round((float(candidate["end_seconds"]) - offset) * rate))
                start, end = max(0, start), min(frames, end)
                if end - start < int(round(MUSIC_GAP_RECIPE.window_seconds * rate)) + 1:
                    continue
                stream.seek(start)
                block = stream.read(end - start, dtype="float64", always_2d=True)
                if block.shape[0] <= 0:
                    continue
                readings[candidate["candidate_id"]] = extent_from_gaps(
                    find_interior_gaps(block.T, rate, MUSIC_GAP_RECIPE))
    except (OSError, ValueError, RuntimeError):
        # 信号を読めないことを、止まりの有無の結論にしない。
        return {}
    return readings


def _repeat_intent() -> dict:
    """再演の探索を決める条件。実装か条件が変われば保存物を使わない。"""
    settings = evidence_settings()
    return {"schema_version": RECORD_SCHEMA,
            **{key: settings[key] for key in (
                "fingerprint_settings_id", "matching_settings_id", "repeat_method",
                "repeat_stage_sha256", "repeat_minimum_separation_seconds",
                "implementation_sha256")},
            "chunk_phase_ratios": list(self_recurrence.CHUNK_PHASE_RATIOS)}


def _repeat_cache_path(cache_dir: Path, signal_sha256: str) -> Path:
    name = hashlib.sha256(json.dumps(
        {"signal": signal_sha256, **_repeat_intent()},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"repeats-{name}.json"


def _repeats(signal, signal_sha256: str | None, cache_dir: Path | None):
    """同じ信号・同じ条件なら、保存した再演を読み直す。壊れた保存物は使わない。"""
    path = (None if cache_dir is None or signal_sha256 is None
            else _repeat_cache_path(cache_dir, signal_sha256))
    if path is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("type") == "within_source_repeat_cache"
                    and payload.get("intent") == _repeat_intent()
                    and payload.get("signal_sha256") == signal_sha256):
                return [identity.SharedOccurrence(**item) for item in payload["repeats"]]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    found = self_recurrence.repeated_recordings(signal)
    if path is not None:
        payload = {"type": "within_source_repeat_cache", "signal_sha256": signal_sha256,
                   "intent": _repeat_intent(),
                   "repeats": [item.to_dict() for item in found]}
        _write_annotation(path, payload, lambda: None)
    return found


def is_current_annotation(record: dict, *, run_sha256: str, query_signal: dict | None) -> bool:
    """現在のrun・照合信号・判定条件に対応する一致だけを引き継ぐ。"""
    if not isinstance(record, dict) or record.get("type") != "episode_identity_annotation":
        return False
    intent = record.get("intent")
    return (record.get("schema_version") == RECORD_SCHEMA
            and isinstance(intent, dict)
            and intent.get("run_sha256") == run_sha256
            and all(intent.get(key) == value for key, value in evidence_settings().items())
            and record.get("query_signal") == query_signal)


def verify_annotation_inputs(run_path: Path, run: dict, intent: dict,
                             query_path: Path | None, query_sha256: str | None) -> None:
    """長い照合中に変わった入力から、一致の記録を確定しない。"""
    try:
        if _sha256(run_path) != intent["run_sha256"]:
            raise IdentityRunError("照合中に番組runが変更されています")
        if any(intent.get(key) != value for key, value in evidence_settings().items()):
            raise IdentityRunError("照合中に実装または判定条件が変更されています")
        if query_path is not None:
            if _sha256(query_path) != query_sha256:
                raise IdentityRunError("照合中に信号が変更されています")
            candidate_sets(run)
        for item in intent["references"]:
            if _sha256(Path(item["path"])) != item["sha256"]:
                raise IdentityRunError("照合中に参照録音が変更されています")
    except OSError as exc:
        raise IdentityRunError("照合中に入力または候補記録が失われています") from exc


def annotate_run(run_path: Path, reference_paths: list[Path], *, cache_dir: Path | None = None,
                 loader=identity.decode_mono, reference_origins: dict[str, dict] | None = None,
                 theme_reference_sha256: frozenset[str] | set[str] | None = None) -> Path:
    """`run.json` の隣に `identity.json` を作り、そのパスを返す。"""
    run_path = Path(run_path).resolve()
    run_bytes = run_path.read_bytes()
    run = json.loads(run_bytes)
    if run.get("type") != "episode_bgm_discovery_run" and run.get("status") != ROUTED_STATUS:
        raise IdentityRunError("番組の候補runではありません")
    references = sorted({Path(path).resolve() for path in reference_paths})
    reference_records = [{"path": str(path), "sha256": _sha256(path)} for path in references]
    intent = {
        **evidence_settings(),
        "run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "references": sorted(reference_records, key=lambda item: item["sha256"]),
    }
    # 出所の変更は波形照合を変えない。呼出元の可変dictから切り離した事実を保存する。
    origins = json.loads(json.dumps(reference_origins or {}, ensure_ascii=False, allow_nan=False))
    external_evidence = [
        {"kind": "released_reference", **item,
         **({"origin": origins[item["sha256"]]} if item["sha256"] in origins else {})}
        for item in reference_records
    ]
    query_path = query_kind = query_sha256 = None
    sets = {}
    if is_annotatable(run):
        # 保存物を返す前にも、現在使える信号とactivity記録を確認する。
        query_path, query_kind, query_sha256 = _query_signal(run)
        sets = candidate_sets(run)
    query_signal = None if query_kind is None else {"kind": query_kind, "sha256": query_sha256}
    record_path = run_path.parent / RECORD_NAME
    if record_path.is_file():
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
        except ValueError:
            existing = None
        if (is_current_annotation(existing, run_sha256=intent["run_sha256"], query_signal=query_signal)
                and existing.get("intent") == intent):
            if reference_origins is None:
                # 指定の省略は、既知の出所を消す指示ではない。
                prior_origins = {item.get("sha256"): item["origin"]
                                 for item in existing.get("external_evidence_used") or []
                                 if isinstance(item, dict) and "origin" in item}
                external_evidence = [
                    {**item, **({"origin": prior_origins[item["sha256"]]}
                               if item["sha256"] in prior_origins else {})}
                    for item in external_evidence
                ]
            if existing.get("external_evidence_used") == external_evidence:
                verify_annotation_inputs(run_path, run, intent, query_path, query_sha256)
                return record_path
            # 同じ照合の出所だけを更新し、既知録音・未解決区間の再計算を避ける。
            payload = {**existing, "external_evidence_used": external_evidence,
                       "metadata_updated_at": datetime.now(timezone.utc).isoformat()}
            _write_annotation(record_path, payload, lambda: verify_annotation_inputs(
                run_path, run, intent, query_path, query_sha256))
            return record_path

    offset = timeline_offset(run)
    activity = []
    matches: list[identity.RecordingMatch] = []
    repeats: list[identity.SharedOccurrence] = []
    recurring: list[tuple[float, float, dict]] = []
    quality_view: dict = {"recipe_id": OCCURRENCE_QUALITY_RECIPE.recipe_id, "recording_count": 0,
                          "theme_excluded_count": 0, "selected_count": 0,
                          "selected_above_minimum": 0, "groups": [],
                          "claim_boundary": ["照合する信号が無いので順位付けの対象がない"]}
    if query_path is not None:
        signal = loader(query_path)
        if references:
            index = identity.ReferenceIndex.build(references, loader=loader, cache_dir=cache_dir)
            matches = identity.match(signal, index, query_offset_seconds=offset,
                                     query_content_sha256=query_sha256)
        repeats = _repeats(signal, query_sha256, cache_dir)
        source_key = str((run.get("source") or {}).get("source_id")
                         or (run.get("source") or {}).get("sha256") or "this_source")
        for span in self_recurrence.spans_for_source(repeats, source_key=source_key):
            # 候補の時刻は元メディアの時刻なので、再演の時刻にも音声の開始位置を足す。
            recurring.append((span["start_seconds"] + offset, span["end_seconds"] + offset,
                              {"other_episode": span["other_episode"],
                               "other_start_seconds": span["other_start_seconds"] + offset,
                               "other_end_seconds": span["other_end_seconds"] + offset,
                               "waveform_explained": span["waveform_explained"],
                               "same_timeline": True}))
        quality_view = occurrence_quality_view(matches, source_key, theme_reference_sha256)
        for recipe_id, candidates in sets.items():
            pieces = identity.split_candidates_at_matches(candidates, matches,
                                                          recurring=recurring)
            read_extents_for_candidates(pieces, recurring, CUE_EXTENT_RECIPE,
                                        gap_readings=_interior_gaps(query_path, pieces, offset))
            decisions = identity.novelty_decisions(pieces)
            activity.append({
                "recipe_id": recipe_id,
                "candidates": pieces,
                "novelty_decisions": decisions,
                "zero_reason": identity.episode_zero_reason(decisions),
                # 同じ録音が二度鳴る候補どうしをつなぐ。統合も削除もしない。
                "occurrence_groups": identity.link_occurrences(
                    {source_key: pieces}, {(source_key, source_key): repeats}),
            })

    payload = {
        "type": "episode_identity_annotation",
        "schema_version": RECORD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intent": intent,
        "run": {"path": str(run_path), "status": run.get("status"),
                "source_sha256": (run.get("source") or {}).get("sha256")},
        "query_signal": query_signal,
        "external_evidence_used": external_evidence,
        "evidence_limits": [
            "known_exact means the same recording is audible in the span; semantic role is unresolved",
            "no match is not evidence that wanted music is absent",
            "a repeat inside this source shows the same recording; the semantic role is unresolved",
            "no repeat is not evidence that a cue appears only once",
            "occurrence_quality ranks how little other sound is layered on each occurrence; it is not a wanted-score decision",
        ],
        "matches": [m.to_dict() for m in matches],
        "repeated_recordings": [item.to_dict() for item in repeats],
        # 分離の前に抽出元を選べるようにする。順位付けであって公開可否の判断ではない。
        "occurrence_quality": quality_view,
        "activity_runs": activity,
    }
    _write_annotation(record_path, payload, lambda: verify_annotation_inputs(
        run_path, run, intent, query_path, query_sha256))
    return record_path


def _write_annotation(record_path: Path, payload: dict, validate) -> None:
    """確定前に入力を再確認し、重なる保存にも別の一時ファイルを使う。"""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=record_path.parent, prefix=f".{record_path.name}-",
                                suffix=".tmp", encoding="utf-8", newline="\n", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        validate()
        temporary.replace(record_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
