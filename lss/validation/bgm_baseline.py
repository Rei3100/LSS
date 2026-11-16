"""BGM corpusを現行分離backendとactivity候補群で一貫して実測する。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from typing import Callable

import soundfile as sf
import numpy as np

from lss.bgm.activity import ActivityRecipe, discover_activity, load_activity_recipe
from lss.core import worker
from lss.output import writer
from lss.validation.activity_evidence import build_activity_detection_record
from lss.validation.bgm_corpus import load_bgm_corpus
from lss.validation.occurrence_metrics import (
    evaluate_activity_discovery,
    summarize_occurrence_results,
)


class BgmBaselineError(RuntimeError):
    """baseline runを完全な検証記録として確定できない。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, records) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True, allow_nan=False,
            ) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _tag_scalar(tags: dict, key: str) -> str | None:
    value = tags.get(key)
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value if isinstance(value, str) else None


def _choose_output(outputs: list[str], stem_target: str, expected_frames: int) -> tuple[Path, dict]:
    matches = []
    for raw_path in outputs:
        path = Path(raw_path).resolve()
        tags = writer.read_source_tags(path)
        if (_tag_scalar(tags, "LSS_STEM") or "").casefold() == stem_target.casefold():
            matches.append((path, tags))
    if len(matches) != 1:
        raise BgmBaselineError(
            f"stem={stem_target}の出力を一意に選べません: matches={len(matches)}"
        )
    path, tags = matches[0]
    info = sf.info(path)
    if info.frames != expected_frames:
        raise BgmBaselineError(
            f"分離出力がsource timelineを保持していません: {path}, {info.frames}/{expected_frames}"
        )
    return path, tags


def _rank_key(record: dict) -> tuple:
    summary = record["summary"]
    boundary = summary["worst_boundary_error_frames"]
    return (
        summary["hard_failure_count"],
        -summary["macro_activity_coverage"],
        -summary["macro_candidate_precision"],
        summary["total_missed_activity_frames"],
        summary["total_outside_activity_frames"],
        boundary if boundary is not None else 0,
        record["recipe_id"],
    )


def run_bgm_baseline(
    manifest_path: Path,
    recipe_paths: list[Path],
    output_dir: Path,
    *,
    model_id: str | None,
    stem_target: str,
    separate: Callable[[dict], dict] = worker.handle_job,
) -> Path:
    """分離を1回ずつ行い、複数activity Recipeを比較する未昇格の実測runを公開する。"""
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    corpus = load_bgm_corpus(manifest_path, verify_sources=True)
    if not recipe_paths:
        raise BgmBaselineError("activity Recipeがありません")
    recipes: list[tuple[Path, ActivityRecipe]] = []
    recipe_ids = set()
    for raw_path in recipe_paths:
        path = Path(raw_path).resolve()
        recipe = load_activity_recipe(path)
        if recipe.recipe_id in recipe_ids:
            raise BgmBaselineError(f"同一activity Recipeが重複しています: {recipe.recipe_id}")
        recipe_ids.add(recipe.recipe_id)
        recipes.append((path, recipe))
    if output_dir.exists():
        raise FileExistsError(f"出力先は新規directoryである必要があります: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}-", suffix=".tmp"))
    try:
        separated = []
        recipe_runs = {
            recipe.recipe_id: {
                "path": recipe_path,
                "recipe": recipe,
                "detection_records": [],
                "results": [],
            }
            for recipe_path, recipe in recipes
        }
        for case in corpus.cases:
            if _sha256(case.source_path) != case.source_sha256:
                raise BgmBaselineError(f"case {case.case_id}のsourceが処理前に変更されています")
            if model_id is None:
                result = {
                    "status": "source_passthrough", "outputs": [str(case.source_path)],
                    "input_sha256": case.source_sha256,
                }
                selected = case.source_path
                tags = {}
            else:
                result = separate({
                    "cmd": "separate",
                    "mix_path": str(case.source_path),
                    "model_id": model_id,
                    "stem_target": stem_target,
                })
                if result.get("status") not in {"ok", "skipped"} or not result.get("outputs"):
                    raise BgmBaselineError(f"case {case.case_id}の分離に失敗しました: {result}")
                if result.get("input_sha256") != case.source_sha256:
                    raise BgmBaselineError(f"case {case.case_id}のworker input identityが一致しません")
                selected, tags = _choose_output(result["outputs"], stem_target, case.frames)
            selected_sha256 = _sha256(selected)
            samples, sample_rate = sf.read(selected, dtype="float32", always_2d=True)
            if _sha256(selected) != selected_sha256:
                raise BgmBaselineError(f"case {case.case_id}の分離信号が読込中に変更されました")
            if sample_rate != case.sample_rate or len(samples) != case.frames:
                raise BgmBaselineError(f"case {case.case_id}の分離信号layoutが不正です")
            runtime_revision = None
            if model_id is not None:
                runtime_text = _tag_scalar(tags, "LSS_RUNTIME_REVISION")
                try:
                    runtime_revision = json.loads(runtime_text) if runtime_text else None
                except json.JSONDecodeError as exc:
                    raise BgmBaselineError(f"case {case.case_id}のRuntimeRevisionが不正です") from exc
                if not isinstance(runtime_revision, dict):
                    raise BgmBaselineError(f"case {case.case_id}にRuntimeRevisionがありません")
            separated.append({
                "case_id": case.case_id,
                "stage_status": result["status"],
                "selected_output": str(selected),
                "selected_output_sha256": selected_sha256,
                "model_id": _tag_scalar(tags, "LSS_MODEL"),
                "model_sha256": _tag_scalar(tags, "LSS_MODEL_SHA256"),
                "config_sha256": _tag_scalar(tags, "LSS_CONFIG_SHA256"),
                "recipe_id": _tag_scalar(tags, "LSS_RECIPE_ID"),
                "runtime_revision": runtime_revision,
            })
            for recipe_path, recipe in recipes:
                discovery = discover_activity(samples.T, case.sample_rate, recipe)
                detections = discovery.as_detections()
                recipe_run = recipe_runs[recipe.recipe_id]
                recipe_run["detection_records"].append(build_activity_detection_record(
                    case, selected, recipe, discovery,
                    analysis_signal_sha256=selected_sha256,
                ))
                recipe_run["results"].append(evaluate_activity_discovery(case, detections))
            del samples
            if _sha256(case.source_path) != case.source_sha256:
                raise BgmBaselineError(f"case {case.case_id}のsourceが処理中に変更されました")

        scoreboard = []
        for recipe_path, recipe in recipes:
            recipe_run = recipe_runs[recipe.recipe_id]
            recipe_dir = stage / recipe.recipe_id
            recipe_dir.mkdir()
            results = recipe_run["results"]
            summary = summarize_occurrence_results(corpus, results)
            _write_jsonl(recipe_dir / "detections.jsonl", recipe_run["detection_records"])
            _write_jsonl(recipe_dir / "evaluation.jsonl", [
                summary, *results,
            ])
            scoreboard.append({
                "recipe_id": recipe.recipe_id,
                "recipe_path": str(recipe_path),
                "recipe_sha256": _sha256(recipe_path),
                "parameters": recipe.to_dict(),
                "summary": summary,
            })
        scoreboard.sort(key=_rank_key)
        _write_json(stage / "scoreboard.json", {
            "schema_version": 2,
            "evaluation_layer": "activity_discovery",
            "selection_status": "best_observed_not_validated",
            "selection_rule": [
                "hard_failure_count asc", "macro_activity_coverage desc",
                "macro_candidate_precision desc", "total_missed_activity_frames asc",
                "total_outside_activity_frames asc", "worst_boundary_error_frames asc",
                "recipe_id asc",
            ],
            "best_observed_recipe_id": scoreboard[0]["recipe_id"],
            "recipes": scoreboard,
        })
        _write_json(stage / "run.json", {
            "schema_version": 2,
            "status": "completed_unvalidated_baseline",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "stage_graph": "source_activity" if model_id is None else "separation_activity",
            "corpus_manifest": str(manifest_path),
            "corpus_manifest_sha256": _sha256(manifest_path),
            "evaluation_runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "soundfile": sf.__version__,
                "implementation_artifacts": [
                    {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
                    {
                        "path": str((Path(__file__).parents[1] / "bgm" / "activity.py").resolve()),
                        "sha256": _sha256(Path(__file__).parents[1] / "bgm" / "activity.py"),
                    },
                    {
                        "path": str((Path(__file__).parent / "occurrence_metrics.py").resolve()),
                        "sha256": _sha256(Path(__file__).parent / "occurrence_metrics.py"),
                    },
                ],
            },
            "model_id": model_id,
            "stem_target": stem_target,
            "separations": separated,
        })
        stage.rename(output_dir)
        return output_dir / "scoreboard.json"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
