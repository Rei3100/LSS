"""外部参照や正解labelを使わず、episode-only BGM候補を凍結する。"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

import numpy as np
import soundfile as sf

from lss.bgm.activity import (
    ActivityRecipe,
    RelativeHysteresisRecipe,
    StemEnergyShareRecipe,
    discover_activity_file,
    discover_stem_energy_share_files,
    load_discovery_recipe,
)
from lss.core import audio_input, session, worker
from lss.core.audio_timeline import probe_audio_timeline
from lss.bgm.candidate_store import CandidateOrigin, materialize_candidate
from lss.output import writer
from lss.validation.activity_evidence import activity_implementation_revision


class EpisodeDiscoveryError(RuntimeError):
    """episode-only runを再現可能な記録として確定できない。"""


class EpisodeSeparationError(EpisodeDiscoveryError):
    """workerの構造化失敗を、製品境界で意味を失わず分類できるよう保持する。"""

    def __init__(self, worker_result: dict) -> None:
        self.worker_result = dict(worker_result)
        super().__init__(f"episode分離に失敗しました: {worker_result}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_IMPLEMENTATION_ROOT = Path(__file__).parents[1]
_IMPLEMENTATION_PATHS = ("validation/episode_discovery.py", "validation/activity_evidence.py",
                         "bgm/activity.py", "bgm/candidate_store.py", "output/writer.py",
                         "core/audio_input.py", "core/audio_timeline.py", "core/session.py")
_LOADED_IMPLEMENTATION = {name: _sha256(_IMPLEMENTATION_ROOT / name) for name in _IMPLEMENTATION_PATHS}


def _verify_loaded_implementation() -> None:
    if {name: _sha256(_IMPLEMENTATION_ROOT / name) for name in _IMPLEMENTATION_PATHS} != _LOADED_IMPLEMENTATION:
        raise EpisodeDiscoveryError("読み込み済みの番組候補実装と現在のファイルが一致しません")


def _write_json(path: Path, value: object) -> str:
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}-", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256(temporary_path)
        temporary_path.replace(path)
        return digest
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _scalar_tag(tags: dict, key: str) -> str | None:
    value = tags.get(key)
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value if isinstance(value, str) else None


def _artifact_records(
    paths: list[str], destination: Path, *, reusable_root: Path | None = None,
) -> list[dict]:
    records = []
    destination = destination.resolve()
    reusable_root = reusable_root.resolve() if reusable_root is not None else None
    for raw_path in paths:
        path = Path(raw_path).resolve()
        managed = path.parent == destination or (
            reusable_root is not None and path.is_relative_to(reusable_root)
        )
        if not managed or not path.is_file():
            raise EpisodeDiscoveryError(f"分離成果物が検証runの管理範囲外です: {path}")
        tags = writer.read_source_tags(path)
        stem = _scalar_tag(tags, "LSS_STEM")
        if not stem:
            raise EpisodeDiscoveryError(f"分離成果物のstem identityがありません: {path}")
        info = sf.info(path)
        records.append({
            "path": str(path),
            "sha256": _sha256(path),
            "stem": stem,
            "sample_rate": info.samplerate,
            "frames": info.frames,
            "channels": info.channels,
            "subtype": info.subtype,
        })
    if not records:
        raise EpisodeDiscoveryError("分離成果物がありません")
    return records


def run_episode_discovery(
    source_path: Path,
    activity_recipe_paths: list[Path],
    output_dir: Path,
    *,
    source_id: str,
    model_id: str,
    stem_target: str = "Instrumental",
    corpus_role: str = "development",
    preserve_original_mixture: bool = False,
    separate: Callable[..., dict] = worker.handle_job,
    resume: bool = False,
) -> Path:
    """未注釈sourceの分離とactivityを実行し、metricを捏造せず結果を凍結する。"""
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    _verify_loaded_implementation()
    if not source_path.is_file():
        raise FileNotFoundError(f"episode sourceが見つかりません: {source_path}")
    if not source_id.strip() or corpus_role not in {"development", "holdout", "product"}:
        raise ValueError("source_idまたはcorpus_roleが不正です")
    if not isinstance(preserve_original_mixture, bool):
        raise ValueError("原本mix保全指定が不正です")
    if not activity_recipe_paths:
        raise EpisodeDiscoveryError("activity Recipeがありません")
    recipes = []
    seen_recipe_ids = set()
    for raw_path in activity_recipe_paths:
        recipe_path = Path(raw_path).resolve()
        recipe_sha256 = _sha256(recipe_path)
        recipe = load_discovery_recipe(recipe_path)
        if _sha256(recipe_path) != recipe_sha256:
            raise EpisodeDiscoveryError("読み込み中にactivity Recipeが変更されました")
        if recipe.recipe_id in seen_recipe_ids:
            raise EpisodeDiscoveryError(f"activity Recipeが重複しています: {recipe.recipe_id}")
        seen_recipe_ids.add(recipe.recipe_id)
        recipes.append((recipe_path, recipe, recipe_sha256))
    if output_dir.exists() and not resume:
        raise FileExistsError(f"出力先は新規directoryである必要があります: {output_dir}")

    source_sha256 = _sha256(source_path)
    source_size = source_path.stat().st_size
    timeline_origin = probe_audio_timeline(source_path).to_dict()
    intent = {
        "schema_version": 1,
        "source_id": source_id, "corpus_role": corpus_role,
        "source_sha256": source_sha256, "source_bytes": source_size,
        "audio_timeline_origin": timeline_origin,
        "model_id": model_id, "stem_target": stem_target.casefold(),
        "preserve_original_mixture": preserve_original_mixture,
        "activity_recipes": [
            {"recipe_id": item.recipe_id, "sha256": digest}
            for _, item, digest in recipes
        ],
        "implementation_sha256": _LOADED_IMPLEMENTATION["validation/episode_discovery.py"],
        "implementation_sources": _LOADED_IMPLEMENTATION,
        "activity_implementation": activity_implementation_revision(),
        "candidate_store_sha256": _sha256(Path(__file__).parents[1] / "bgm" / "candidate_store.py"),
    }
    intent_path = output_dir / "intent.json"
    if output_dir.exists():
        if not intent_path.is_file():
            raise EpisodeDiscoveryError("再開先に処理の同一性を確認する記録がありません")
        if json.loads(intent_path.read_text(encoding="utf-8")) != intent:
            raise EpisodeDiscoveryError("再開先の入力・処理の版が今回の要求と一致しません")
    else:
        output_dir.mkdir(parents=True)
        _write_json(intent_path, intent)

    artifacts_dir = output_dir / "artifacts"
    activity_dir = output_dir / "activity"
    artifacts_dir.mkdir(exist_ok=True)
    activity_dir.mkdir(exist_ok=True)
    separation_path = output_dir / "separation.json"
    saved_separation = None
    if resume and separation_path.exists():
        separation_bytes = separation_path.read_bytes()
        saved_separation = json.loads(separation_bytes)
        separation_sha256 = hashlib.sha256(separation_bytes).hexdigest()
        result = saved_separation["result"]
    else:
        result = separate(
            {
                "cmd": "separate", "mix_path": str(source_path),
                "model_id": model_id, "stem_target": stem_target,
            },
            destination_dir=artifacts_dir,
        )
    if result.get("status") not in {"ok", "skipped"}:
        raise EpisodeSeparationError(result)
    if result.get("input_sha256") != source_sha256:
        raise EpisodeDiscoveryError("workerが処理したepisode identityが一致しません")

    outputs = result.get("outputs", [])
    zero_result = result.get("zero_result")
    if zero_result is not None:
        if result.get("status") not in {"ok", "skipped"} or outputs:
            raise EpisodeDiscoveryError("zero resultと分離成果物を同時に受理できません")
        try:
            zero_result = session.validate_zero_result(zero_result)
        except ValueError as exc:
            raise EpisodeDiscoveryError(f"zero resultの検証証拠が不正です: {exc}") from exc
        artifacts = []
    else:
        artifacts = _artifact_records(
            outputs, artifacts_dir,
            reusable_root=output_dir.parent if result.get("status") == "skipped" else None,
        )
    if saved_separation is not None and saved_separation.get("artifacts") != artifacts:
        raise EpisodeDiscoveryError("再開に使う分離信号が削除・変更されています")
    if saved_separation is None:
        separation_sha256 = _write_json(separation_path, {"result": result, "artifacts": artifacts})

    run_path = output_dir / "run.json"

    def verify_completion(activity_runs=()) -> None:
        """検証中・切り出し後の変更も、完成記録を確定する前に拒否する。"""
        try:
            _verify_loaded_implementation()
            if json.loads(intent_path.read_bytes()) != intent or _sha256(separation_path) != separation_sha256:
                raise EpisodeDiscoveryError("完成確認中に入力構成または分離記録が変更されました")
            if _sha256(source_path) != source_sha256 or source_path.stat().st_size != source_size:
                raise EpisodeDiscoveryError("episode sourceが処理中に変更されました")
            for recipe_path, _, digest in recipes:
                if _sha256(recipe_path) != digest:
                    raise EpisodeDiscoveryError("activity Recipeが処理中に変更されました")
            for artifact in artifacts:
                if _sha256(Path(artifact["path"])) != artifact["sha256"]:
                    raise EpisodeDiscoveryError("完成確認中に分離成果物が変更されました")
            for activity in activity_runs:
                path = Path(activity["record_path"])
                if _sha256(path) != activity["record_sha256"]:
                    raise EpisodeDiscoveryError("完成確認中に候補記録が変更されました")
                for candidate in json.loads(path.read_bytes())["candidates"]:
                    for field in ("asset", "preservation_asset"):
                        asset = candidate.get(field)
                        if asset is not None and _sha256(Path(asset["path"])) != asset["sha256"]:
                            raise EpisodeDiscoveryError("完成確認中に候補音声が変更されました")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EpisodeDiscoveryError("完成確認に必要な入力・成果物が読めません") from exc

    if resume and run_path.exists():
        completed = json.loads(run_path.read_text(encoding="utf-8"))
        if completed.get("source", {}).get("sha256") != source_sha256:
            raise EpisodeDiscoveryError("完了済みrunの入力identityが一致しません")
        if completed.get("separation", {}).get("artifacts") != artifacts:
            raise EpisodeDiscoveryError("完了済みrunの分離信号が一致しません")
        expected_ids = [] if zero_result is not None else [recipe.recipe_id for _, recipe, _ in recipes]
        records_valid = (
            [item.get("recipe_id") for item in completed["activity_runs"]] == expected_ids
            and completed.get("status") == (
                "completed_validated_zero_result" if zero_result is not None
                else "completed_unannotated_episode_only_observation"
            )
        )
        if zero_result is not None:
            records_valid = records_valid and completed.get("outcome") == {"kind": "zero", **zero_result}
        else:
            counts = [item.get("candidate_count") for item in completed["activity_runs"]]
            if any(not isinstance(count, int) or count < 0 for count in counts):
                records_valid = False
            else:
                count = sum(counts)
                expected_outcome = {
                    "kind": "unresolved_candidates" if count else "unresolved_no_activity_candidates",
                    "candidate_count": count, "publication_state": "not_published",
                }
                if preserve_original_mixture:
                    expected_outcome["artifact_count"] = sum(
                        item.get("artifact_count", -1) for item in completed["activity_runs"]
                    )
                records_valid = records_valid and completed.get("outcome") == expected_outcome
        for activity_run in completed["activity_runs"]:
            record_path = Path(activity_run["record_path"])
            if not record_path.is_relative_to(activity_dir) or not record_path.is_file():
                records_valid = False
                break
            if _sha256(record_path) != activity_run["record_sha256"]:
                records_valid = False
                break
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                record.get("source_id") != source_id
                or record.get("separation_recipe_id") != result.get("recipe_id")
                or len(record["candidates"]) != activity_run.get("candidate_count")
            ):
                records_valid = False
                break
            actual_artifacts = 0
            for candidate in record["candidates"]:
                for field in ("asset", "preservation_asset"):
                    asset = candidate.get(field)
                    if asset is None:
                        continue
                    actual_artifacts += 1
                    candidate_path = Path(asset["path"])
                    if (
                        not candidate_path.is_relative_to(output_dir / "candidates")
                        or not candidate_path.is_file()
                        or _sha256(candidate_path) != asset["sha256"]
                    ):
                        records_valid = False
                        break
            expected_artifacts = activity_run.get(
                "artifact_count", activity_run.get("candidate_count"),
            )
            if actual_artifacts != expected_artifacts:
                records_valid = False
        if records_valid:
            verify_completion(completed["activity_runs"])
            return run_path
    if zero_result is not None:
        verify_completion()
        run_path = output_dir / "run.json"
        _write_json(run_path, {
            "type": "episode_bgm_discovery_run",
            "schema_version": 2,
            "status": "completed_validated_zero_result",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "source_id": source_id,
                "corpus_role": corpus_role,
                "path": str(source_path),
                "sha256": source_sha256,
                "bytes": source_size,
                "container_suffix": source_path.suffix.lower(),
                "audio_timeline_origin": timeline_origin,
            },
            "evidence_scope": "validated_zero_result",
            "external_evidence_used": [],
            "semantic_ground_truth": None,
            "outcome": {"kind": "zero", **zero_result},
            "separation": {
                "model_id": model_id,
                "stem_target": stem_target,
                "status": result["status"],
                "recipe_id": result.get("recipe_id"),
                "recipe_revision": result.get("recipe_revision"),
                "runtime_id": result.get("runtime_id"),
                "runtime_revision": result.get("runtime_revision"),
                "config_sha256": result.get("config_sha256"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "rtf": result.get("rtf"),
                "peak_vram_mb": result.get("peak_vram_mb"),
                "artifacts": artifacts,
            },
            "activity_runs": [],
            "observation_limits": [
                "zero outcome is accepted only with a passed validation hard gate",
                "no candidate asset is fabricated for a validated zero outcome",
            ],
            "evaluation_runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "soundfile": sf.__version__,
                "implementation_sha256": _LOADED_IMPLEMENTATION["validation/episode_discovery.py"],
            },
        })
        return run_path

    artifacts_by_stem: dict[str, dict] = {}
    for artifact in artifacts:
        key = artifact["stem"].casefold()
        if key in artifacts_by_stem:
            raise EpisodeDiscoveryError(f"stem identityが重複しています: {artifact['stem']}")
        artifacts_by_stem[key] = artifact

    activity_runs = []
    original_context = None
    original_signal_path = None
    original_signal_sha256 = None
    original_info = None

    def prepared_original():
        nonlocal original_context, original_signal_path, original_signal_sha256, original_info
        if original_signal_path is None:
            context = audio_input.prepare_input(
                source_path, work_root=output_dir / "decode-work",
            )
            original_signal_path = Path(context.__enter__()).resolve()
            original_context = context
            session.validate_input(original_signal_path)
            original_info = sf.info(original_signal_path)
            original_signal_sha256 = _sha256(original_signal_path)
        return original_signal_path, original_signal_sha256, original_info

    try:
        for recipe_path, recipe, recipe_sha256 in recipes:
            if _sha256(recipe_path) != recipe_sha256:
                raise EpisodeDiscoveryError("activity Recipeが処理中に変更されました")
            if isinstance(recipe, (ActivityRecipe, RelativeHysteresisRecipe)):
                signal = artifacts_by_stem.get(stem_target.casefold())
                if signal is None:
                    raise EpisodeDiscoveryError(f"解析対象stemがありません: {stem_target}")
                supporting_signals: list[dict] = []
                signal_paths = {signal["stem"]: Path(signal["path"])}
                discovery = discover_activity_file(next(iter(signal_paths.values())), recipe)
            elif isinstance(recipe, StemEnergyShareRecipe):
                missing = [
                    stem for stem in recipe.required_stems
                    if stem.casefold() not in artifacts_by_stem
                ]
                if missing:
                    raise EpisodeDiscoveryError(f"activity解析に必要なstemがありません: {missing}")
                signal = artifacts_by_stem[recipe.ANALYSIS_STEM.casefold()]
                supporting_signals = [
                    artifacts_by_stem[stem.casefold()] for stem in recipe.COMPARISON_STEMS
                ]
                signal_paths = {
                    item["stem"]: Path(item["path"])
                    for item in (signal, *supporting_signals)
                }
                before = {stem: _sha256(path) for stem, path in signal_paths.items()}
                discovery = discover_stem_energy_share_files(signal_paths, recipe)
                if any(_sha256(path) != before[stem] for stem, path in signal_paths.items()):
                    raise EpisodeDiscoveryError("activity解析中に分離信号が変更されました")
            else:  # pragma: no cover - loaderが未対応型を返した場合の防御
                raise EpisodeDiscoveryError(f"未対応のactivity Recipeです: {type(recipe)}")
            signal_path = Path(signal["path"])
            signal_sha256 = _sha256(signal_path)
            if signal_sha256 != signal["sha256"]:
                raise EpisodeDiscoveryError("activity解析後の信号identityが成果物記録と一致しません")
            detection_path = activity_dir / f"{recipe.recipe_id}.json"
            candidates = []
            for item in discovery.candidates:
                asset_path = (
                    output_dir / "candidates" / recipe.recipe_id / f"{item.candidate_id}.flac"
                )
                candidate = {
                    "candidate_id": item.candidate_id,
                    "start_frame": item.start_frame,
                    "end_frame": item.end_frame,
                    "start_seconds": (
                        item.start_frame / discovery.sample_rate
                        + timeline_origin["offset_seconds"]
                    ),
                    "end_seconds": (
                        item.end_frame / discovery.sample_rate
                        + timeline_origin["offset_seconds"]
                    ),
                    "peak_rms": item.peak_rms,
                    "semantic_role": "unresolved_music_like",
                    "identity_state": "unresolved",
                    "cue_identity_id": None,
                    "audio_version_id": None,
                    "asset": materialize_candidate(
                        signal_path, asset_path,
                        origin=CandidateOrigin(
                            source_sha256, signal_sha256,
                            result["recipe_id"], result["runtime_id"], recipe.recipe_id,
                            item.start_frame, item.end_frame, discovery.sample_rate,
                            timeline_origin["offset_numerator"],
                            timeline_origin["offset_denominator"],
                        ),
                        source_id=source_id, source_path=source_path,
                        evidence={"peak_rms": item.peak_rms, "method": recipe.method},
                    ),
                }
                if preserve_original_mixture:
                    raw_path, raw_sha256, raw_info = prepared_original()
                    start_seconds = item.start_frame / discovery.sample_rate
                    end_seconds = item.end_frame / discovery.sample_rate
                    raw_start = max(0, math.floor(start_seconds * raw_info.samplerate))
                    raw_end = min(raw_info.frames, math.ceil(end_seconds * raw_info.samplerate))
                    if raw_end <= raw_start:
                        raise EpisodeDiscoveryError("原本mix保全候補の時間範囲が空です")
                    preservation_path = asset_path.with_name(
                        f"{item.candidate_id}-original-mixture.flac"
                    )
                    preservation_asset = materialize_candidate(
                        raw_path, preservation_path,
                        origin=CandidateOrigin(
                            source_sha256, raw_sha256,
                            result["recipe_id"], result["runtime_id"], recipe.recipe_id,
                            raw_start, raw_end, raw_info.samplerate,
                            timeline_origin["offset_numerator"], timeline_origin["offset_denominator"],
                        ),
                        source_id=source_id, source_path=source_path,
                        evidence={
                            "method": recipe.method,
                            "representation": "original_mixture_no_op",
                            "detected_from_signal_sha256": signal_sha256,
                        },
                    )
                    preservation_asset["representation"] = "original_mixture_no_op_candidate"
                    candidate["preservation_asset"] = preservation_asset
                candidates.append(candidate)
            artifact_count = sum(
                1 + int("preservation_asset" in candidate) for candidate in candidates
            )
            record = {
                "type": "episode_activity_discovery",
                "schema_version": 2,
                "source_id": source_id,
                "corpus_role": corpus_role,
                "evidence_scope": "episode_only",
                "external_evidence_used": [],
                "analysis_signal": signal,
                "supporting_analysis_signals": supporting_signals,
                "separation_recipe_id": result.get("recipe_id"),
                "activity_recipe": {
                    "path": str(recipe_path),
                    "sha256": recipe_sha256,
                    "recipe_id": recipe.recipe_id,
                    "revision": recipe.to_dict(),
                },
                "activity_implementation": activity_implementation_revision(recipe.method),
                "semantic_status": "unresolved_unannotated_source",
                "metric_status": "not_scored_without_ground_truth",
                "candidates": candidates,
            }
            record_sha256 = _write_json(detection_path, record)
            activity_run = {
                "recipe_id": recipe.recipe_id,
                "method": recipe.method,
                "analysis_stems": [
                    signal["stem"], *(item["stem"] for item in supporting_signals)
                ],
                "record_path": str(detection_path),
                "record_sha256": record_sha256,
                "candidate_count": len(discovery.candidates),
                "candidate_frames": sum(
                    item.end_frame - item.start_frame for item in discovery.candidates
                ),
                "signal_frames": discovery.source_frames,
            }
            if preserve_original_mixture:
                activity_run["artifact_count"] = artifact_count
            activity_runs.append(activity_run)
    finally:
        if original_context is not None:
            original_context.__exit__(None, None, None)

    verify_completion(activity_runs)
    run_path = output_dir / "run.json"
    _write_json(run_path, {
        "type": "episode_bgm_discovery_run",
        "schema_version": 2,
        "status": "completed_unannotated_episode_only_observation",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "source_id": source_id,
            "corpus_role": corpus_role,
            "path": str(source_path),
            "sha256": source_sha256,
            "bytes": source_size,
            "container_suffix": source_path.suffix.lower(),
            "audio_timeline_origin": timeline_origin,
        },
        "evidence_scope": "episode_only",
        "external_evidence_used": [],
        "semantic_ground_truth": None,
        "outcome": {
            "kind": (
                "unresolved_candidates"
                if any(item["candidate_count"] for item in activity_runs)
                else "unresolved_no_activity_candidates"
            ),
            "candidate_count": sum(item["candidate_count"] for item in activity_runs),
            "publication_state": "not_published",
            **({
                "artifact_count": sum(item["artifact_count"] for item in activity_runs),
            } if preserve_original_mixture else {}),
        },
        "separation": {
            "model_id": model_id,
            "stem_target": stem_target,
            "status": result["status"],
            "recipe_id": result.get("recipe_id"),
            "recipe_revision": result.get("recipe_revision"),
            "runtime_id": result.get("runtime_id"),
            "runtime_revision": result.get("runtime_revision"),
            "config_sha256": result.get("config_sha256"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "rtf": result.get("rtf"),
            "peak_vram_mb": result.get("peak_vram_mb"),
            "artifacts": artifacts,
        },
        "activity_runs": activity_runs,
        "observation_limits": [
            "activity candidates are not semantic BGM labels",
            "source music, songs, dialogue residuals, and musical SFX remain unresolved",
            "no quantitative accuracy claim is available without independent evidence",
        ],
        "evaluation_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "implementation_sha256": _LOADED_IMPLEMENTATION["validation/episode_discovery.py"],
        },
    })
    return run_path
