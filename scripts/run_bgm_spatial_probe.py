"""BGM空間refinementを凍結Developmentケースで比較する（本番昇格は行わない）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

from lss.bgm.refinement import SpatialRefinementRecipe, refine_stereo
from lss.bgm.blend import StereoBlendRecipe, blend_stereo
from lss.bgm.activity import discover_stem_energy_share_files, load_stem_energy_share_recipe
from lss.validation.stereo_quality import db_ratio, energy, stereo_quality


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def artifact(path: Path) -> dict:
    info = sf.info(path)
    if info.samplerate != 44100 or info.channels != 2 or info.frames == 0:
        raise ValueError(f"44100Hz完成stereoのfixtureが必要です: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "frames": info.frames}


def evaluation_interval(occurrences: list[dict], frames: int) -> list[int]:
    # targetなしは無音padding/開始終了を含む全timelineを検証する。
    if not occurrences:
        return [0, frames]
    return [min(item["start_frame"] for item in occurrences), max(item["end_frame"] for item in occurrences)]


def freeze(args) -> None:
    root = args.corpus.resolve().parent
    rows = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 固定の既存stereo screenは再推論せず、記録済みhashと照合する。
    cached_run = args.provider / "qr-b1-stereo-r1" / "run.json"
    cached = json.loads(cached_run.read_text(encoding="utf-8"))
    if cached["status"] != "PASSED_B1" or cached["hard_failure_count"]:
        raise ValueError("再利用するBandIt stereo B1 runが成功していません")
    cached_cases = {item["case_id"]: item for item in cached["cases"]}
    cases = []
    for row in rows:
        if row["type"] != "case":
            continue
        case_id = row["case_id"]
        source = artifact(root / row["source"]["path"])
        if source["sha256"] != row["source"]["sha256"] or source["frames"] != row["source"]["frames"]:
            raise ValueError(f"corpus sourceが固定manifestと一致しません: {case_id}")
        stems = {}
        if case_id in cached_cases:
            prior = cached_cases[case_id]
            if prior["source"]["sha256"] != source["sha256"]:
                raise ValueError("cached BandIt入力identityが一致しません")
            for role, item in prior["outputs"].items():
                path = Path(item["path"])
                if sha256(path) != item["sha256"]:
                    raise ValueError(f"cached BandIt stemが記録と一致しません: {path}")
                stems[role] = {"path": str(path.resolve()), "sha256": item["sha256"]}
        absent = not row["occurrences"]
        cases.append({
            "case_id": case_id, "evidence_role": "controlled-clean-component",
            "singing_guard": "singing_music" in row["attributes"],
            "target_absent": absent, "source": source, "stems": stems,
            "reference": artifact(root / "references" / case_id / "music.flac"),
            "signal_frames": [0, source["frames"]],
            "evaluation_frames": evaluation_interval(row["occurrences"], source["frames"]),
        })
    provider_artifacts = {}
    for name, relative in {
        "checkpoint": "downloads/dnr-3s-mus64-l1snr.ckpt",
        "experiment": "expt/dnr-3s-mus64-l1snr.yaml",
        "model": "configs/model/bsrnn-small-dnr-multi-mus64.yaml",
        "inference": "configs/inference/default8.yaml",
        "fader": "core/utils/audio.py",
    }.items():
        provider_artifacts[name] = {"path": relative, "sha256": sha256(args.provider / relative)}
    # モデルを選び直すprobeではない。既存comparatorのidentityを固定する。
    prior_identity = cached["candidate"]
    for name, prior_key in (("checkpoint", "checkpoint_sha256"), ("model", "model_config_sha256"),
                            ("inference", "inference_config_sha256")):
        if provider_artifacts[name]["sha256"] != prior_identity[prior_key]:
            raise ValueError("BandIt artifactが既存stereo comparatorと異なります")
    revision = subprocess.run(
        ["git", "-C", str(args.provider), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != prior_identity["provider_source_revision"]:
        raise ValueError("BandIt source revisionが既存comparatorと異なります")

    if args.episode_frontier:
        frontier = json.loads(args.episode_frontier.read_text(encoding="utf-8"))
        refs = {item["reference_id"]: item for item in frontier["references"]}
        episode_source = artifact(Path(frontier["sources"]["episode_raw"]["path"]))
        if episode_source["sha256"] != frontier["sources"]["episode_raw"]["sha256"]:
            raise ValueError("episode sourceが既存frontierと異なります")
        episode_stems = {}
        for role in ("music", "speech", "effects"):
            item = frontier["sources"][f"bandit_{role}"]
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("episode stemが固定frontierと異なります")
            episode_stems[role] = {"path": item["path"], "sha256": item["sha256"]}
        for item in frontier["target_cases"]:
            start = round(item["episode_start_seconds"] * 44100)
            frames = round(item["duration_seconds"] * 44100)
            margin = 6 * 44100
            reference = refs[item["reference_id"]]
            if sha256(Path(reference["path"])) != reference["sha256"]:
                raise ValueError("released referenceが固定frontierと異なります")
            cases.append({
                "case_id": item["case_id"], "evidence_role": "identity-reference-applicability-uncertain",
                "singing_guard": False, "target_absent": False,
                "source": episode_source, "stems": episode_stems,
                "signal_frames": [max(0, start - margin), start + frames + margin],
                "evaluation_frames": [start, start + frames],
                "reference": {"path": reference["path"], "sha256": reference["sha256"],
                              "start_seconds": item["reference_start_seconds"]},
            })
    base = SpatialRefinementRecipe()
    reuse = None
    if args.reuse_receipt:
        if not args.reuse_plan:
            raise ValueError("再利用CUDA receiptの親planが必要です")
        receipt = json.loads(args.reuse_receipt.read_text(encoding="utf-8"))
        if receipt["status"] != "COMPLETED_CUDA_WITHOUT_FALLBACK" or receipt["plan_sha256"] != sha256(args.reuse_plan):
            raise ValueError("再利用CUDA receiptの親planが一致しません")
        if receipt["provider"]["artifacts"] != provider_artifacts or receipt["provider"]["revision"] != revision:
            raise ValueError("再利用CUDA receiptのprovider identityが一致しません")
        by_id = {item["case_id"]: item for item in receipt["cases"]}
        for case in cases:
            if case["stems"]:
                continue
            item = by_id[case["case_id"]]
            if item["source_sha256"] != case["source"]["sha256"]:
                raise ValueError("再利用CUDA sourceが新planと異なります")
            for value in item["stems"].values():
                checked = artifact(Path(value["path"]))
                if checked["sha256"] != value["sha256"] or checked["frames"] != case["source"]["frames"]:
                    raise ValueError("再利用CUDA stemのidentity/timelineが異なります")
            case["stems"] = item["stems"]
        reuse = {"receipt_path": str(args.reuse_receipt.resolve()), "receipt_sha256": sha256(args.reuse_receipt),
                 "plan_path": str(args.reuse_plan.resolve()), "plan_sha256": sha256(args.reuse_plan)}
    candidates = [replace(base, iterations=value).to_dict() for value in (1, 2)]
    if args.partial_refinement:
        candidates = [{"spatial": base.to_dict(),
                       "blend": StereoBlendRecipe(value).to_dict()} for value in (0.25, 0.5)]
    if args.absence_only:
        cases = [case for case in cases if case["target_absent"]]
    activity_path = root.parent / (prior_identity["activity_recipe_id"] + ".json")
    if sha256(activity_path) != prior_identity["activity_recipe_sha256"]:
        raise ValueError("target-absenceの既存activity Recipeが変わっています")
    plan = {
        "schema_version": 1, "status": "FROZEN_BEFORE_RESULTS", "corpus_role": "Development",
        "objective": "ステレオ空間情報による残留音・fidelity改善とwanted Music損失を比較する",
        "promotion_allowed": False, "holdout_consumed": False, "l4_oracle": None,
        "sample_rate": 44100, "cases": cases,
        "provider": {"root": str(args.provider.resolve()), "revision": revision, "artifacts": provider_artifacts},
        "parent_receipt": {"path": str(cached_run.resolve()), "sha256": sha256(cached_run)},
        "candidates": candidates, "reuse_inference": reuse,
        "absence_only": args.absence_only,
        "activity_recipe": {"path": str(activity_path.resolve()), "sha256": sha256(activity_path)},
        "decision_rule": {
            "maximum_any_controlled_regression_db": 1.0,
            "minimum_material_controlled_improvement_db": 1.0,
            "maximum_absent_output_to_input_db": -30.0,
            "maximum_absent_activity_candidate_count": 0,
            "identity_reference_failure_blocks_controlled_development": False,
            "require_improvement_without_singing_or_stereo_regression": True,
        },
        "transforms": {
            "audio": "no gain/EQ/resampling/channel/timeline transformation",
            "identity_reference": "resample_poly to 44100; one positive joint LS gain for evaluation only",
            "zero_quality_dimensions": "null, never a fabricated perfect score",
            "residual": "unexplained source remains separate",
        },
        "code": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256(path)
            for path in (Path(__file__), REPO_ROOT / "scripts/infer_bgm_bandit.py",
                         REPO_ROOT / "lss/bgm/refinement.py", REPO_ROOT / "lss/bgm/blend.py",
                         REPO_ROOT / "lss/bgm/activity.py",
                         REPO_ROOT / "lss/validation/stereo_quality.py")
        },
    }
    write_json(args.plan, plan)
    print(f"Development比較を凍結: {len(cases)}ケース、2候補。昇格不可。", flush=True)


def read_audio(item: dict, start: int, end: int) -> np.ndarray:
    values, rate = sf.read(item["path"], start=start, frames=end - start, dtype="float64", always_2d=True)
    if rate != 44100 or values.shape != (end - start, 2) or not np.isfinite(values).all():
        raise ValueError(f"stereo/timelineが不正です: {item['path']}")
    return values.T


def quality_vector(measured: dict) -> dict[str, float]:
    result = {name: value for name, value in measured["si_sdr_db"].items() if value is not None}
    if measured["scale_dependent_snr_db"] is not None:
        result["unscaled_snr"] = measured["scale_dependent_snr_db"]
    return result


def graph_parts(definition: dict) -> tuple[SpatialRefinementRecipe, StereoBlendRecipe | None, str]:
    if "spatial" not in definition:
        spatial = SpatialRefinementRecipe(**definition)
        return spatial, None, spatial.recipe_id
    spatial = SpatialRefinementRecipe(**definition["spatial"])
    blend = StereoBlendRecipe(**definition["blend"])
    spatial.validate()
    blend.validate()
    canonical = json.dumps({"order": ["spatial", "blend-with-parent"], **definition},
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    return spatial, blend, "bgm-spatial-blend-v1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def evaluate(args) -> None:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_RESULTS" or plan["promotion_allowed"]:
        raise ValueError("凍結Development planが必要です")
    for relative, expected in plan["code"].items():
        if sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"plan固定後に評価／処理codeが変わっています: {relative}")
    activity_recipe = None
    if plan.get("activity_recipe"):
        item = plan["activity_recipe"]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("target-absenceのactivity Recipeが変わっています")
        activity_recipe = load_stem_energy_share_recipe(Path(item["path"]))
    inferred = {}
    if args.receipt:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        if receipt["status"] != "COMPLETED_CUDA_WITHOUT_FALLBACK" or receipt["plan_sha256"] != sha256(args.plan):
            raise ValueError("CUDA receiptと凍結planが一致しません")
        inferred = {item["case_id"]: item for item in receipt["cases"]}
    paths = {}
    for case in plan["cases"]:
        if not case["stems"]:
            inferred_case = inferred[case["case_id"]]
            if inferred_case["source_sha256"] != case["source"]["sha256"]:
                raise ValueError("CUDA receipt source identityが異なります")
            case["stems"] = inferred_case["stems"]
        if set(case["stems"]) != {"music", "speech", "effects"}:
            raise ValueError("3-stem comparatorが必要です")
        for item in [case["source"], case["reference"], *case["stems"].values()]:
            prior = paths.setdefault(item["path"], item["sha256"])
            if prior != item["sha256"]:
                raise ValueError("同一pathのidentityが重複しています")
    for path, expected in paths.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"比較source/stem/referenceが固定identityと異なります: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    all_results = []
    for case in plan["cases"]:
        left, right = case["signal_frames"]
        eval_start, eval_end = case["evaluation_frames"]
        crop = slice(eval_start - left, eval_end - left)
        raw = read_audio(case["source"], left, right)
        stems = {role: read_audio(item, left, right) for role, item in case["stems"].items()}
        reference = None
        if not case["target_absent"]:
            if "start_seconds" in case["reference"]:
                values, rate = sf.read(case["reference"]["path"], dtype="float64", always_2d=True)
                divisor = gcd(rate, 44100)
                values = resample_poly(values, 44100 // divisor, rate // divisor, axis=0)
                ref_start = round(case["reference"]["start_seconds"] * 44100)
                reference = values[ref_start:ref_start + eval_end - eval_start].T
            else:
                reference = read_audio(case["reference"], eval_start, eval_end)
        base_music = stems["music"][:, crop]
        reference_eval = reference

        def measured(music):
            if reference_eval is None:
                return {"output_to_input_db": db_ratio(energy(music), energy(raw[:, crop]))}
            estimate = music
            fitted_gain = 1.0
            if case["evidence_role"] == "identity-reference-applicability-uncertain":
                denominator = energy(estimate)
                fitted_gain = max(0.0, float(np.sum(estimate * reference_eval) / denominator)) if denominator else 0.0
                estimate = estimate * fitted_gain
            quality = stereo_quality(estimate, reference_eval)
            quality["evaluation_joint_gain"] = fitted_gain
            return quality

        baseline = measured(base_music)
        results = []
        rendered = {}
        for definition in plan["candidates"]:
            recipe, blend, recipe_id = graph_parts(definition)
            if recipe.recipe_id not in rendered:
                rendered[recipe.recipe_id] = refine_stereo(raw, stems, recipe)
            result = rendered[recipe.recipe_id]
            full_music = result.stems["music"]
            if blend:
                full_music = blend_stereo(stems["music"], full_music, blend)
            music = full_music[:, crop]
            quality = measured(music)
            failures = []
            peak = float(np.max(np.abs(full_music)))
            if peak > 1:
                failures.append("music_outside_serialization_range")
            deltas = {}
            if case["target_absent"]:
                if quality["output_to_input_db"] > plan["decision_rule"]["maximum_absent_output_to_input_db"]:
                    failures.append("target_absent_leakage")
                if activity_recipe:
                    # 判定用Speech/Effectsは親comparator。候補Musicだけを差し替える。
                    with TemporaryDirectory(prefix="absence-", dir=args.output) as temporary:
                        stem_paths = {}
                        for role, values in (("Music", full_music), ("Speech", stems["speech"]),
                                             ("Effects", stems["effects"])):
                            path = Path(temporary) / f"{role}.wav"
                            sf.write(path, values.T, 44100, subtype="FLOAT")
                            stem_paths[role] = path
                        discovery = discover_stem_energy_share_files(stem_paths, activity_recipe)
                    quality["activity_candidate_count"] = len(discovery.candidates)
                    if len(discovery.candidates) > plan["decision_rule"]["maximum_absent_activity_candidate_count"]:
                        failures.append("target_absent_activity_false_positive")
            else:
                before, after = quality_vector(baseline), quality_vector(quality)
                deltas = {key: after[key] - value for key, value in before.items() if key in after}
                if case["evidence_role"] == "controlled-clean-component":
                    if any(value < -plan["decision_rule"]["maximum_any_controlled_regression_db"] for value in deltas.values()):
                        failures.append("controlled_target_or_stereo_regression")
                    for key, limit in (("relative_channel_gain_error_db", 1.0),
                                       ("side_to_mid_balance_error_db", 2.0),
                                       ("interchannel_correlation_error", 0.2)):
                        if quality[key] is not None and abs(quality[key]) > limit:
                            failures.append(key)
            results.append({
                "recipe_id": recipe_id, "quality": quality, "paired_delta_db": deltas,
                "failures": failures, "music_peak": peak,
                "unexplained_rms": float(np.sqrt(np.mean(result.unexplained ** 2))),
            })
        del rendered, result, full_music
        all_results.append({
            "case_id": case["case_id"], "evidence_role": case["evidence_role"],
            "singing_guard": case["singing_guard"], "target_absent": case["target_absent"],
            "baseline": baseline, "candidates": results,
        })
        print(json.dumps({"case_id": case["case_id"], "results": results}, ensure_ascii=False), flush=True)
    summaries = []
    for definition in plan["candidates"]:
        recipe_id = graph_parts(definition)[2]
        paired = [(case, next(value for value in case["candidates"] if value["recipe_id"] == recipe_id)) for case in all_results]
        failures = [{"case_id": case["case_id"], "failures": value["failures"]} for case, value in paired if value["failures"]]
        improvements = [case["case_id"] for case, value in paired
                        if case["evidence_role"] == "controlled-clean-component" and not case["target_absent"]
                        and value["paired_delta_db"].get("joint", -300) >= plan["decision_rule"]["minimum_material_controlled_improvement_db"]]
        status = "DEVELOPMENT_SURVIVOR" if not failures and improvements else "REJECTED_EXACT_REFINEMENT"
        if plan.get("absence_only"):
            status = "PASSED_ABSENCE_SCREEN" if not failures else "REJECTED_ABSENCE_SCREEN"
        summaries.append({
            "recipe_id": recipe_id,
            "status": status,
            "failures": failures, "material_controlled_improvements": improvements,
        })
    after = {path: sha256(Path(path)) for path in paths}
    if after != paths:
        raise ValueError("比較中にsource/stem/referenceが変わりました")
    write_json(args.output / "report.json", {
        "schema_version": 1, "status": "COMPLETED_DEVELOPMENT_COMPARISON",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "plan": {"path": str(args.plan.resolve()), "sha256": sha256(args.plan)},
        "receipt": {"path": str(args.receipt.resolve()), "sha256": sha256(args.receipt)} if args.receipt else None,
        "promotion_allowed": False, "holdout_consumed": False, "l4_oracle": None,
        "source_identity_preserved": True,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "soundfile": sf.__version__},
        "summaries": summaries, "cases": all_results,
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("corpus", type=Path)
    prepare.add_argument("provider", type=Path)
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("--episode-frontier", type=Path)
    prepare.add_argument("--reuse-receipt", type=Path)
    prepare.add_argument("--reuse-plan", type=Path)
    prepare.add_argument("--partial-refinement", action="store_true")
    prepare.add_argument("--absence-only", action="store_true")
    execute = commands.add_parser("evaluate")
    execute.add_argument("plan", type=Path)
    execute.add_argument("output", type=Path)
    execute.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    freeze(args) if args.command == "freeze" else evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
