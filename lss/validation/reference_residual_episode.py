"""既知録音区間を参照残差で補い、元cueと別のDevelopment保留分岐へ保存する。"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

from mutagen import MutagenError
from mutagen.flac import FLAC
import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

from lss.bgm import candidate_store, identity, identity_run, recording_recovery, reference_alignment, reference_residual
from lss.core.session import _write_json_atomic
from lss.core import audio_identity
from lss.harness.fetch import file_sha256
from lss.output import writer

ROOT = Path(__file__).resolve().parents[2]
METHOD = "retain_music_with_known_recording_reference_residual"
BOUNDARY_POLICY = "media_seconds_minus_decoded_audio_origin_half_even_v1"
REFERENCE_CONVERSION = {"method": "scipy_resample_poly_same_stereo_timeline_v1", "sample_rate": 44100,
                        "window": ["kaiser", 5.0], "padtype": "constant"}


def artifact(path):
    # 試聴でtagが書き戻されても、同じ音の素材として確認できるようにする。
    return audio_identity.material_record(path)


def _recipe(payload):
    payload = dict(payload)
    if payload.get("alignment_recipe") is not None:
        payload["alignment_recipe"] = reference_alignment.ReferenceAlignmentRecipe(**payload["alignment_recipe"])
    return reference_residual.ReferenceResidualRecipe(
        **{**payload, "fit_recipe": recording_recovery.RecordingRecoveryRecipe(**payload["fit_recipe"])})


def _frame(seconds, timeline, rate):
    if type(seconds) not in (int, float) or not np.isfinite(seconds):
        raise ValueError("録音一致の時刻が不正です")
    offset = Fraction(timeline["offset_numerator"], timeline["offset_denominator"])
    return round((Fraction(str(seconds)) - offset) * rate)


def _cases(upstream, timeline, annotation, rate, frames):
    result, known, ambiguous, seen = [], [], [], set()
    references = {item["path"]: item["sha256"] for item in annotation["intent"]["references"]}
    for payload in sorted(annotation["matches"], key=lambda m: m["query_start_seconds"]):
        match = identity.RecordingMatch(**payload)
        if match.identity_state != "known_exact":
            continue
        key = json.dumps(payload, sort_keys=True, allow_nan=False)
        if key in seen:
            raise ValueError("録音一致の同じ根拠が重複しています")
        seen.add(key)
        if references.get(match.reference_path) != match.reference_sha256:
            raise ValueError("固定した正の録音一致と参照同一性が必要です")
        start = _frame(match.query_start_seconds, timeline, rate)
        end = _frame(match.query_end_seconds, timeline, rate)
        if not 0 <= start < end <= frames:
            raise ValueError("録音一致区間が原本の時間軸から外れています")
        known.append((start, end, payload))
    boundaries = sorted({value for start, end, _ in known for value in (start, end)})
    for start, end in zip(boundaries, boundaries[1:]):
        active = [item for item in known if item[0] <= start < end <= item[1]]
        if len(active) > 1:
            ambiguous.append({"start_frame": start, "end_frame": end,
                              "reason": "overlapping_recording_alignments_retain_music",
                              "matches": [item[2] for item in active]})
            continue
        if not active:
            continue
        match_start, _, payload = active[0]
        match = identity.RecordingMatch(**payload)
        parents = [item for item in upstream["candidates"]
                   if item["origin"]["start_frame"] <= start < end <= item["origin"]["end_frame"]]
        if len(parents) != 1:
            raise ValueError("録音一致区間が元cueへ一意に対応しません")
        result.append({"match": payload, "start_frame": start, "end_frame": end,
                       "effective_reference_start_seconds": match.reference_start_seconds + (start - match_start) / rate,
                       "primary_candidate_asset_id": parents[0]["asset"]["candidate_asset_id"]})
    return {"cases": result, "retained_ambiguous_spans": ambiguous,
            "unresolved_match_count": sum(m["identity_state"] != "known_exact" for m in annotation["matches"])}


def _implementation():
    return {str(Path(module.__file__).resolve().relative_to(ROOT)): file_sha256(Path(module.__file__))
            for module in (identity, identity_run, recording_recovery, reference_alignment, reference_residual, candidate_store, writer)} | {
                str(Path(__file__).resolve().relative_to(ROOT)): file_sha256(Path(__file__))}


def _read_reference(path, expected_format):
    reference, rate = sf.read(path, dtype="float64", always_2d=True)
    if (rate != expected_format["sample_rate"] or len(reference) != expected_format["frames"]
            or reference.shape[1] != 2 or not len(reference) or not np.isfinite(reference).all()):
        raise ValueError("参照録音のrate・左右・有限音声が固定記録へ対応しません")
    if rate != 44100:
        divisor = math.gcd(rate, 44100)
        reference = resample_poly(reference, 44100 // divisor, rate // divisor, axis=0,
                                  window=("kaiser", 5.0), padtype="constant")
    return reference.T


def freeze(run_path: Path, preparation_run_path: Path, output: Path,
           *, recipe=reference_residual.ReferenceResidualRecipe()) -> Path:
    from lss.application import _read_bgm_plan, _read_bgm_result

    run_path, preparation_run_path = Path(run_path).resolve(), Path(preparation_run_path).resolve()
    upstream = json.loads(run_path.read_bytes())
    parent_plan_path = run_path.parent / "plan.json"
    parent, parent_hash = _read_bgm_plan(parent_plan_path, Path(upstream["source"]["path"]).resolve())
    _read_bgm_result(parent_plan_path, parent, parent_hash, run_path)
    if parent["source"].get("corpus_role") != "development" or recipe.fit_recipe.sample_rate != 44100:
        raise ValueError("44.1kHzのDevelopment原本だけを処理します")
    for candidate in upstream["candidates"]:
        origin = candidate_store.CandidateOrigin(**candidate["origin"])
        origin.validate()
        if (origin.candidate_id != candidate["asset"]["candidate_asset_id"]
                or origin.source_sha256 != parent["source"]["sha256"]
                or origin.signal_sha256 != upstream["rendering"]["signal"]["sha256"]
                or origin.sample_rate != 44100
                or origin.time_offset_numerator != parent["audio_timeline_origin"]["offset_numerator"]
                or origin.time_offset_denominator != parent["audio_timeline_origin"]["offset_denominator"]):
            raise ValueError("元cueの単独来歴と原本の時間基準が一致しません")
        candidate_store._verify_crop(Path(upstream["rendering"]["signal"]["path"]),
                                     Path(candidate["asset"]["path"]), origin)
    prepared = json.loads(preparation_run_path.read_bytes())
    if (prepared.get("status") != "PASSED_REPRESENTATIVE_HARD_SCREEN"
            or prepared.get("hard_failure_count") != 0 or prepared.get("source") != parent["source"]
            or (prepared.get("outputs") or {}).get("music") != parent["stems"]["music"]):
        raise ValueError("原本・Musicと同じ完了した準備記録が必要です")
    source_audio = prepared["prepared_input"]
    music = parent["stems"]["music"]
    for item in (source_audio, music):
        info = sf.info(item["path"])
        if info.samplerate != 44100 or info.channels != 2 or info.frames != parent["frames"]:
            raise ValueError("準備音声とMusicの時間軸が一致しません")
    music_info = sf.info(music["path"])
    if music_info.subtype not in {"FLOAT", "DOUBLE"}:
        raise ValueError("参照残差分岐には元Musicの浮動小数点cacheが必要です")
    annotation_path = run_path.parent / identity_run.RECORD_NAME
    annotation = json.loads(annotation_path.read_bytes())
    query_path, kind, query_hash = identity_run._query_signal(upstream)
    query_signal = {"kind": kind, "sha256": query_hash}
    if not identity_run.is_current_annotation(annotation, run_sha256=file_sha256(run_path), query_signal=query_signal):
        raise ValueError("現在の元run・信号・判定条件に対応する録音一致が必要です")
    timeline = parent["audio_timeline_origin"]
    coverage = _cases(upstream, timeline, annotation, 44100, parent["frames"])
    cases = coverage["cases"]
    reference_formats = {}
    for case in cases:
        path = case["match"]["reference_path"]
        info = sf.info(path)
        if info.channels != 2 or not 8000 <= info.samplerate <= 192000 or not info.frames:
            raise ValueError("参照録音は対応rateの完成ステレオである必要があります")
        reference_formats[path] = {"sample_rate": info.samplerate, "frames": info.frames, "channels": info.channels}
    parents = {item["asset"]["candidate_asset_id"]: item for item in upstream["candidates"]}
    parent_ids = list(dict.fromkeys(case["primary_candidate_asset_id"] for case in cases))
    inputs = [artifact(run_path), artifact(parent_plan_path), artifact(preparation_run_path), artifact(annotation_path),
              parent["source"], source_audio, music, artifact(query_path), *annotation["intent"]["references"],
              *(item["asset"] for item in upstream["candidates"])]
    implementation = _implementation()
    revision = {"schema_version": 1, "method": METHOD, "boundary_policy": BOUNDARY_POLICY,
                "overlap_policy": "retain_parent_music_without_selecting_alignment",
                "reference_conversion": REFERENCE_CONVERSION,
                "recipe": recipe.to_dict(), "upstream_renderer_id": upstream["renderer_id"],
                "source_music_revision": parent["renderer_revision"], "implementation": implementation,
                "signal_serialization": music_info.subtype, "candidate_serialization": "declared_power_of_two_headroom_v1"}
    renderer_id = "bgm-reference-residual-held-v1-" + hashlib.sha256(json.dumps(
        revision, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
               "source_music_runtime": parent["renderer_revision"]["runtime_revision"]["music"]}
    runtime_id = "bgm-reference-residual-runtime-v1-" + hashlib.sha256(json.dumps(
        runtime, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan = {"schema_version": 1, "status": "FROZEN_BEFORE_REFERENCE_RESIDUAL_HELD_BRANCH",
            "promotion_allowed": False, "holdout_consumed": False, "source": parent["source"],
            "source_id": parent["source_id"], "source_audio": source_audio, "music": music,
            "audio_timeline_origin": timeline, "frames": parent["frames"], **coverage,
            "parents": [parents[key] for key in parent_ids], "inputs": inputs,
            "upstream_run": artifact(run_path), "upstream_plan": artifact(parent_plan_path),
            "preparation_run": artifact(preparation_run_path),
            "annotation": artifact(annotation_path), "query_signal": query_signal,
            "matching_settings": identity_run.evidence_settings(), "recipe": recipe.to_dict(),
            "reference_conversion": REFERENCE_CONVERSION, "reference_formats": reference_formats,
            "renderer_id": renderer_id, "renderer_revision": revision, "implementation": implementation,
            "runtime_id": runtime_id, "runtime_revision": runtime,
            "activity_recipe_id": parent["activity_recipe"]["recipe_id"],
            "signal_subtype": music_info.subtype, "upstream_alignment_status": parent["alignment_status"],
            "limits": ["Developmentの録音一致は意味的BGMや同一masterの証明ではない。",
                       "元Musicの保持と数値的な残差回収。台詞・効果音の漏れは未評価。",
                       "元cueと未一致区間を保持し、勝者・正式公開・既定変更・Holdout消費を行わない。",
                       "source uploadと追加推論を行わない。"]}
    _verify(plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "plan.json"
    _write_json_atomic(path, plan)
    return path


def _verify(plan, *, locator=None, executing=True):
    if (plan.get("status") != "FROZEN_BEFORE_REFERENCE_RESIDUAL_HELD_BRANCH"
            or plan.get("promotion_allowed") is not False or plan.get("holdout_consumed") is not False
            or plan["matching_settings"] != identity_run.evidence_settings()):
        raise ValueError("固定したDevelopment分岐と現在の録音一致条件が必要です")
    _recipe(plan["recipe"]).to_dict()
    if plan.get("reference_conversion") != REFERENCE_CONVERSION:
        raise ValueError("参照のrate変換が固定したRecipeへ対応しません")
    for item in plan["inputs"]:
        paths = locator(item) if locator is not None else (Path(item["path"]),)
        if not paths or any(not audio_identity.matches_record(path, item) for path in paths):
            raise ValueError(f"参照残差分岐の入力内容が変更されています: {item['path']}")
    upstream = json.loads(Path(plan["upstream_run"]["path"]).read_bytes())
    upstream_plan = json.loads(Path(plan["upstream_plan"]["path"]).read_bytes())
    prepared = json.loads(Path(plan["preparation_run"]["path"]).read_bytes())
    if (upstream.get("plan_sha256") != plan["upstream_plan"]["sha256"]
            or upstream.get("source") != plan["source"] or upstream_plan.get("source") != plan["source"]
            or upstream_plan.get("stems", {}).get("music") != plan["music"]
            or upstream_plan.get("frames") != plan["frames"]
            or upstream_plan.get("audio_timeline_origin") != plan["audio_timeline_origin"]
            or prepared.get("source") != plan["source"] or prepared.get("prepared_input") != plan["source_audio"]
            or prepared.get("outputs", {}).get("music") != plan["music"]):
        raise ValueError("保留分岐の原本・準備音声・Music・時間軸が元記録へ対応しません")
    required = [plan["upstream_run"], plan["upstream_plan"], plan["preparation_run"], plan["annotation"],
                plan["source"], plan["source_audio"], plan["music"]]
    pinned = {(item["path"], item["sha256"]) for item in plan["inputs"]}
    if any((item["path"], item["sha256"]) not in pinned for item in required):
        raise ValueError("保留分岐の必須入力が固定hash集合へ対応しません")
    annotation = json.loads(Path(plan["annotation"]["path"]).read_bytes())
    if (not identity_run.is_current_annotation(annotation, run_sha256=plan["upstream_run"]["sha256"],
                                               query_signal=plan["query_signal"])
            or _cases(upstream, plan["audio_timeline_origin"], annotation, 44100, plan["frames"])
            != {key: plan[key] for key in ("cases", "retained_ambiguous_spans", "unresolved_match_count")}):
        raise ValueError("参照残差分岐の時刻・録音一致・元cueの対応が変更されています")
    parent_ids = list(dict.fromkeys(case["primary_candidate_asset_id"] for case in plan["cases"]))
    parents = {item["asset"]["candidate_asset_id"]: item for item in upstream["candidates"]}
    if [parents[key] for key in parent_ids] != plan["parents"]:
        raise ValueError("参照残差分岐の元候補の対応が変更されています")
    if executing and _implementation() != plan["implementation"]:
        raise ValueError("参照残差分岐の固定実装が変更されています")


def _write_record(path, payload, validate):
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, prefix=".residual-record-", suffix=".tmp",
                                encoding="utf-8", newline="\n", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        validate()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _render_fresh(plan, output, validate):
    recipe = _recipe(plan["recipe"])
    observations = []
    temporary = None
    try:
        with NamedTemporaryFile(dir=output, prefix=".residual-signal-", suffix=".wav", delete=False) as stream:
            temporary = Path(stream.name)
        with sf.SoundFile(plan["music"]["path"]) as music, sf.SoundFile(plan["source_audio"]["path"]) as source:
            with sf.SoundFile(temporary, mode="w", format="WAV", subtype=plan["signal_subtype"],
                              samplerate=44100, channels=2) as signal:
                def copy_until(end):
                    while music.tell() < end:
                        block = music.read(min(candidate_store.COPY_BLOCK_FRAMES, end - music.tell()),
                                           dtype="float64", always_2d=True)
                        if not len(block) or not np.isfinite(block).all():
                            raise ValueError("元Musicの全区間を有限の音声として保持できません")
                        signal.write(block)
                for case in plan["cases"]:
                    start, end = case["start_frame"], case["end_frame"]
                    copy_until(start)
                    source.seek(start)
                    raw = source.read(end - start, dtype="float64", always_2d=True).T
                    base = music.read(end - start, dtype="float64", always_2d=True).T
                    match = identity.RecordingMatch(**case["match"])
                    reference = _read_reference(match.reference_path, plan["reference_formats"][match.reference_path])
                    local_match = replace(match, query_start_seconds=0, query_end_seconds=(end - start) / 44100,
                                          reference_start_seconds=case["effective_reference_start_seconds"])
                    recovered, measured = reference_residual.recover_reference_residual(raw, base, reference,
                                                                                         local_match, recipe)
                    signal.write(recovered.T)
                    observations.append({**case, "observations": measured,
                                         "original_media_start_seconds": start / 44100 + float(Fraction(
                                             plan["audio_timeline_origin"]["offset_numerator"],
                                             plan["audio_timeline_origin"]["offset_denominator"]))})
                copy_until(plan["frames"])
        sha256 = file_sha256(temporary)
        info = sf.info(temporary)
        if info.frames != plan["frames"] or info.channels != 2 or info.samplerate != 44100:
            raise ValueError("参照残差の全体信号の時間軸が一致しません")
        validate()
        destination = output / f"signal-{sha256}.wav"
        if destination.exists():
            if file_sha256(destination) != sha256:
                raise ValueError("保存済み分岐信号の内容hashが一致しません")
        else:
            temporary.replace(destination)
        return {"path": str(destination), "sha256": sha256, "frames": info.frames,
                "sample_rate": info.samplerate, "channels": info.channels, "subtype": info.subtype}, observations
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _render(plan, output, validate, plan_hash):
    """FLOAT WAVのencoded identityを保つため、完了した信号をhashで検証して再利用する。"""
    receipt_path = output / "rendering.json"
    if receipt_path.exists():
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if (not isinstance(receipt, dict) or receipt.get("status") != "completed_reference_residual_signal"
                or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
                or receipt.get("plan_sha256") != plan_hash or receipt.get("source") != plan["source"]
                or receipt.get("renderer_id") != plan["renderer_id"]
                or receipt.get("publication_allowed") is not False
                or not isinstance(receipt.get("matches"), list) or len(receipt["matches"]) != len(plan["cases"])):
            raise ValueError("保存済み分岐信号の同一性・完了契約が不正です")
        signal = receipt["signal"]
        signal_path = Path(signal["path"]).resolve()
        if not signal_path.is_relative_to(output) or file_sha256(signal_path) != signal["sha256"]:
            raise ValueError("保存済み分岐信号の内容hashが一致しません")
        info = sf.info(signal_path)
        if (info.frames != plan["frames"] or info.samplerate != 44100 or info.channels != 2
                or info.subtype != plan["signal_subtype"]):
            raise ValueError("保存済み分岐信号の時間軸・左右・保存精度が一致しません")
        for case, row in zip(plan["cases"], receipt["matches"]):
            if any(row.get(key) != value for key, value in case.items()):
                raise ValueError("保存済み分岐信号の録音区間の集合が一致しません")
        validate()
        if receipt_path.read_bytes() != receipt_bytes or file_sha256(signal_path) != signal["sha256"]:
            raise ValueError("再利用の確認中に保存済み分岐信号が変更されています")
        return signal, receipt["matches"]
    signal, observations = _render_fresh(plan, output, validate)
    receipt = {"schema_version": 1, "status": "completed_reference_residual_signal",
               "plan_sha256": plan_hash, "source": plan["source"], "renderer_id": plan["renderer_id"],
               "publication_allowed": False, "signal": signal, "matches": observations}
    def verify_signal():
        validate()
        if file_sha256(Path(signal["path"])) != signal["sha256"]:
            raise ValueError("分岐信号の保存中に内容が変更されています")
    _write_record(receipt_path, receipt, verify_signal)
    return signal, observations


def run(plan_path: Path, *, on_candidate=None) -> Path:
    plan_path = Path(plan_path).resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()

    def verify():
        _verify(plan)
        if plan_path.read_bytes() != plan_bytes:
            raise ValueError("保存中に参照残差の固定構成が変更されています")

    verify()
    signal, observations = _render(plan, plan_path.parent, verify, plan_hash)
    candidates = []
    for parent in plan["parents"]:
        primary_id = parent["asset"]["candidate_asset_id"]
        origin = candidate_store.CandidateOrigin(**{**parent["origin"], "signal_sha256": signal["sha256"],
            "separation_recipe_id": plan["renderer_id"], "runtime_id": plan["runtime_id"],
            "activity_recipe_id": plan["activity_recipe_id"], "serialization_policy": "declared_power_of_two_headroom_v1"})
        evidence = {"candidate_role": "held_reference_residual", "primary_candidate_asset_id": primary_id,
                    "plan_sha256": plan_hash, "recipe_id": _recipe(plan["recipe"]).recipe_id,
                    "matches": [row for row in observations if row["primary_candidate_asset_id"] == primary_id]}
        saved = candidate_store.materialize_candidate(Path(signal["path"]),
            plan_path.parent / "candidates" / f"{origin.candidate_id}.flac", origin=origin,
            source_id=plan["source_id"], source_path=Path(plan["source"]["path"]), evidence=evidence)
        candidates.append({"primary_candidate_asset_id": primary_id, "origin": origin.to_dict(), "asset": saved,
                           "candidate_role": "held_reference_residual", "selection_state": "unresolved"})
        if on_candidate is not None:
            on_candidate(saved)
    report_path = plan_path.parent / "residuals.json"
    payload = {"schema_version": 1, "status": "completed_development_reference_residual_held_branch",
               "plan_sha256": plan_hash, "source": plan["source"], "signal": signal,
               "renderer_id": plan["renderer_id"], "candidates": candidates, "matches": observations,
               "retained_ambiguous_spans": plan["retained_ambiguous_spans"],
               "unresolved_match_count": plan["unresolved_match_count"],
               "promotion_allowed": False, "holdout_consumed": False, "publication_state": "not_published",
               "source_hashes_after_verified": True, "original_candidates_unchanged": True,
               "semantic_ground_truth": None, "selection_state": "unresolved", "new_inference_performed": False,
               "upstream_alignment_status": plan["upstream_alignment_status"],
               "outcome": "held_reference_residual_candidates" if candidates else "no_positive_recording_spans",
               "limits": plan["limits"]}

    def final_verify():
        verify()
        _verify_assets(plan, payload)
    _write_record(report_path, payload, final_verify)
    return report_path


def _verify_assets(plan, report, *, locator=None):
    def paths(item):
        found = locator(item) if locator is not None else (Path(item["path"]),)
        if not found or any(not audio_identity.matches_record(path, item) for path in found):
            raise ValueError("保留分岐の音声内容が記録と一致しません")
        return found
    signal = report["signal"]
    signal_paths = paths(signal)
    for path in signal_paths:
        info = sf.info(path)
        if (info.frames != plan["frames"] or info.channels != 2 or info.samplerate != 44100
                or info.subtype != plan["signal_subtype"]):
            raise ValueError("保留分岐の信号の時間軸・左右・精度が一致しません")
    if len(report["candidates"]) != len(plan["parents"]):
        raise ValueError("保留分岐と元cueの対応件数が一致しません")
    artifacts = []
    for parent, candidate in zip(plan["parents"], report["candidates"]):
        origin = candidate_store.CandidateOrigin(**{**parent["origin"], "signal_sha256": signal["sha256"],
            "separation_recipe_id": plan["renderer_id"], "runtime_id": plan["runtime_id"],
            "activity_recipe_id": plan["activity_recipe_id"], "serialization_policy": "declared_power_of_two_headroom_v1"})
        asset = candidate["asset"]
        primary_id = parent["asset"]["candidate_asset_id"]
        if (candidate["origin"] != origin.to_dict() or asset["candidate_asset_id"] != origin.candidate_id
                or candidate["primary_candidate_asset_id"] != primary_id
                or candidate.get("candidate_role") != "held_reference_residual"
                or candidate.get("selection_state") != "unresolved"
                or asset.get("representation") != "noncanonical_staged_candidate"
                or asset.get("publication_state") != "not_published"):
            raise ValueError("保留分岐の来歴・親cue・未確定状態が一致しません")
        for path in paths(asset):
            metadata = json.loads(FLAC(path)["LSS_CANDIDATE_PROVENANCE"][0])
            evidence = metadata["evidence"]
            if (metadata.get("origin") != origin.to_dict() or metadata.get("candidate_id") != origin.candidate_id
                    or metadata.get("semantic_role") != "unresolved_music_like"
                    or metadata.get("identity_state") != "unresolved" or metadata.get("publication_state") != "not_published"
                    or evidence.get("primary_candidate_asset_id") != primary_id
                    or evidence.get("candidate_role") != "held_reference_residual"
                    or evidence.get("plan_sha256") != report["plan_sha256"]
                    or evidence.get("recipe_id") != _recipe(plan["recipe"]).recipe_id
                    or evidence.get("matches") != [row for row in report["matches"] if row["primary_candidate_asset_id"] == primary_id]):
                raise ValueError("保留分岐の単独音声の来歴が記録へ対応しません")
            candidate_store._verify_crop(signal_paths[0], path, origin)
        artifacts.append({"path": asset["path"], "sha256": asset["sha256"],
                          **({"audio": asset["audio"]} if isinstance(asset.get("audio"), dict) else {}),
                          "candidate_asset_id": origin.candidate_id, "primary_candidate_asset_id": primary_id})
    return artifacts


def _read_result(run_path: Path, report_path: Path, *, locator=None) -> dict:
    """推論・係数推定・記録更新なしに、現在の元runへ対応する保留分岐を読む。"""
    run_path, report_path = Path(run_path).resolve(), Path(report_path).resolve()
    if report_path.name != "residuals.json":
        raise ValueError("完了した参照残差分岐記録が必要です")
    plan_path = report_path.parent / "plan.json"
    plan_bytes, report_bytes = plan_path.read_bytes(), report_path.read_bytes()
    plan, report = json.loads(plan_bytes), json.loads(report_bytes)
    if (not isinstance(plan, dict) or not isinstance(report, dict)
            or plan["upstream_run"] != artifact(run_path)
            or type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "completed_development_reference_residual_held_branch"
            or report.get("plan_sha256") != hashlib.sha256(plan_bytes).hexdigest()
            or report.get("source") != plan["source"] or report.get("renderer_id") != plan["renderer_id"]
            or report.get("promotion_allowed") is not False or report.get("holdout_consumed") is not False
            or report.get("publication_state") != "not_published"
            or report.get("source_hashes_after_verified") is not True or report.get("original_candidates_unchanged") is not True
            or report.get("semantic_ground_truth") is not None or report.get("selection_state") != "unresolved"
            or report.get("new_inference_performed") is not False):
        raise ValueError("参照残差分岐の同一性・非公開・完了契約が不正です")
    expected_outcome = "held_reference_residual_candidates" if plan["parents"] else "no_positive_recording_spans"
    if (report.get("outcome") != expected_outcome or len(report["matches"]) != len(plan["cases"])
            or report.get("retained_ambiguous_spans") != plan["retained_ambiguous_spans"]
            or report.get("unresolved_match_count") != plan["unresolved_match_count"]):
        raise ValueError("参照残差分岐の固定区間の完了集合が不正です")
    for case, row in zip(plan["cases"], report["matches"]):
        if any(row.get(key) != value for key, value in case.items()):
            raise ValueError("参照残差分岐の録音一致区間と保存結果が対応しません")
    _verify(plan, locator=locator, executing=False)
    artifacts = _verify_assets(plan, report, locator=locator)
    _verify(plan, locator=locator, executing=False)
    # 長い入力再確認の間に移動先や保存済み音声が変わっても、古い所在を返さない。
    if _verify_assets(plan, report, locator=locator) != artifacts:
        raise ValueError("読戻し中に保留分岐の音声集合が変更されています")
    if plan_path.read_bytes() != plan_bytes or report_path.read_bytes() != report_bytes:
        raise ValueError("読戻し中に参照残差分岐記録が変更されています")
    return {"path": str(report_path), "sha256": hashlib.sha256(report_bytes).hexdigest(),
            "plan": {"path": str(plan_path), "sha256": hashlib.sha256(plan_bytes).hexdigest()},
            "artifacts": artifacts, "selection_state": "unresolved", "semantic_role": "unresolved",
            "publication_allowed": False, "new_inference_performed": False}


class ReferenceResidualResultError(ValueError):
    """現在の元cueへ参照残差の保留分岐を安全に接続できない。"""


def read_result(run_path: Path, report_path: Path, *, locator=None) -> dict:
    """破損・読取失敗は明示する。呼出側は任意の分岐だけを除外して元cueを保持できる。"""
    try:
        return _read_result(run_path, report_path, locator=locator)
    except (OSError, TypeError, ValueError, KeyError, AttributeError, IndexError, OverflowError,
            MutagenError, sf.LibsndfileError) as exc:
        raise ReferenceResidualResultError(f"参照残差の保留分岐を確認できません: {exc}") from exc
