"""凍結した局所routingを同じCUDA branchの全Developmentケースで比較する。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from math import gcd
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from lss.bgm.activity import discover_stem_energy_share_files, load_stem_energy_share_recipe
from lss.bgm.dominance_routing import DominanceRoutingRecipe, route_stereo
from lss.bgm.inference_variants import InferenceVariantRecipe, average_stereo
from lss.bgm.parent_preservation import POLICY, preserve_parent
from lss.validation.stereo_quality import db_ratio, energy, stereo_quality
from lss.validation.song_component_comparison import component_regressions, wanted_components
from scripts.run_bgm_inference_probe import checked_plan, checked_receipt
from scripts.run_bgm_spatial_probe import artifact, quality_vector, read_audio, sha256, write_json


def definition(label, dominant, mixed, preserve=False):
    result = {"label": label, "routing": DominanceRoutingRecipe().to_dict(),
              "dominant_branch": dominant, "mixed_branch": mixed, "schema_version": 1,
              "order": ["fixed-bandit-variants", "restore-gain-and-timeline", "equal-branch-average", "parent-energy-routing"]}
    if preserve:
        result["parent_preservation_guard"] = copy.deepcopy(POLICY)
        result["order"].append("all-music-components-parent-preservation")
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["recipe_id"] = "bgm-routed-inference-v1-" + hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return result


def freeze(args):
    parent = checked_plan(args.parent)
    gain = [InferenceVariantRecipe(-6.0).to_dict()]
    phase = [InferenceVariantRecipe().to_dict(), InferenceVariantRecipe(0.0, 11025).to_dict()]
    preserve = getattr(args, "preserve_parent", False)
    suffix = "-parent-preserved" if preserve else ""
    write_json(args.plan, {"status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False, "holdout_consumed": False,
               "parent_plan": {"path": str(args.parent.resolve()), "sha256": sha256(args.parent)},
               "candidates": [definition("dominant-gain" + suffix, gain, None, preserve),
                              definition("mixed-phase" + suffix, None, phase, preserve),
                              definition("dominant-gain-mixed-phase" + suffix, gain, phase, preserve)],
               "evidence_scope": "Development thresholds are provisional engineering choices after the first two gain/phase cases; frozen before all routing results. Parent model's Music share is not independent semantic truth.",
               "decision_rule": parent["decision_rule"],
               "code": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                        for path in (Path(__file__), ROOT / "lss/bgm/dominance_routing.py",
                                     ROOT / "lss/bgm/parent_preservation.py")}})
    print("局所routingの3候補を比較前に固定しました。", flush=True)


def evaluate(args):
    plan_hash = sha256(args.plan)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["status"] != "FROZEN_BEFORE_RESULTS" or plan["promotion_allowed"] or plan["holdout_consumed"]:
        raise ValueError("Development専用の凍結routing planが必要です")
    for relative, expected in plan["code"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("凍結routing codeが変わっています")
    parent_path = Path(plan["parent_plan"]["path"])
    if sha256(parent_path) != plan["parent_plan"]["sha256"]:
        raise ValueError("parent planが変わっています")
    parent = checked_plan(parent_path)
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    checked_receipt(parent_path, parent, receipt)
    records = {(item["case_id"], item["variant_id"]): item for item in receipt["cases"]}
    expected = {(case["case_id"], InferenceVariantRecipe(**value).recipe_id)
                for case in parent["cases"] for value in parent["variants"]}
    if len(records) != len(receipt["cases"]) or set(records) != expected:
        raise ValueError("CUDA branchに重複・不足・余分があります")
    for record in records.values():
        for item in record["stems"].values():
            if artifact(Path(item["path"]))["sha256"] != item["sha256"]:
                raise ValueError("CUDA branchのstem identityが変わっています")
    activity_item = parent["activity_recipe"]
    if sha256(Path(activity_item["path"])) != activity_item["sha256"]:
        raise ValueError("activity Recipeが変わっています")
    activity = load_stem_energy_share_recipe(Path(activity_item["path"]))
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in parent["cases"]:
        frames = case["source"]["frames"]
        raw = read_audio(case["source"], 0, frames)
        outputs = {}
        for value in parent["variants"]:
            key = InferenceVariantRecipe(**value).recipe_id
            record = records[case["case_id"], key]
            if record["source_sha256"] != case["source"]["sha256"] or set(record["stems"]) != {"music", "speech", "effects"}:
                raise ValueError("CUDA branchのsource/stem setが異なります")
            outputs[key] = {role: read_audio(item, 0, frames) for role, item in record["stems"].items()}
        stems = outputs[InferenceVariantRecipe().recipe_id]
        start, end = case["evaluation_frames"]
        crop = slice(start, end)
        base_music = stems["music"][:, crop]
        reference = None
        if case.get("reference") and not case["target_absent"]:
            item = case["reference"]
            if "start_seconds" in item:
                values, rate = sf.read(item["path"], dtype="float64", always_2d=True)
                divisor = gcd(rate, 44100)
                values = resample_poly(values, 44100 // divisor, rate // divisor, axis=0)
                offset = round(item["start_seconds"] * 44100)
                reference = values[offset:offset + end - start].T
            else:
                reference = read_audio(item, start, end)

        components = {role: read_audio(item, start, end)
                      for role, item in case.get("wanted_components", {}).items()}

        def measured(music):
            if case["target_absent"]:
                return {"output_to_input_db": db_ratio(energy(music), energy(raw[:, crop]))}
            if reference is None:
                return {"difference_to_baseline_db": db_ratio(energy(music - base_music), energy(base_music)),
                        "comparison_to_baseline": stereo_quality(music, base_music), "reference_available": False}
            gain = 1.0
            if case["evidence_role"] == "identity-reference-applicability-uncertain":
                gain = max(0, float(np.sum(music * reference)) / max(energy(music), 1e-30))
            result = {**stereo_quality(music * gain, reference), "evaluation_joint_gain": gain}
            if components:
                result["wanted_components"] = wanted_components(music, components["accompaniment"], components["singing"])
            return result

        baseline = measured(base_music)
        candidates = []
        for value in plan["candidates"]:
            def branch(definitions):
                if definitions is None:
                    return None
                keys = [InferenceVariantRecipe(**item).recipe_id for item in definitions]
                return {role: average_stereo([outputs[key][role] for key in keys]) for role in stems}
            routed = route_stereo(stems, branch(value["dominant_branch"]), branch(value["mixed_branch"]),
                                  DominanceRoutingRecipe(**value["routing"]))
            preservation = None
            if value.get("parent_preservation_guard") is not None:
                if value["parent_preservation_guard"] != POLICY:
                    raise ValueError("固定した親保護の規則が一致しません")
                preserved, preservation = preserve_parent(stems, routed.stems)
                routed = replace(routed, stems=preserved)
                if not preservation["enabled"]:
                    routed = replace(routed, parent_weight=np.ones(frames),
                                     dominant_weight=np.zeros(frames), mixed_weight=np.zeros(frames))
            music = routed.stems["music"][:, crop]
            quality = measured(music)
            peak = float(np.max(np.abs(routed.stems["music"])))
            failures = ["music_outside_serialization_range"] if peak > 1 else []
            deltas = {}
            if reference is not None:
                before, after = quality_vector(baseline), quality_vector(quality)
                deltas = {key: after[key] - result for key, result in before.items() if key in after}
                if case["evidence_role"] == "controlled-clean-component":
                    if any(result < -plan["decision_rule"]["maximum_any_controlled_regression_db"] for result in deltas.values()):
                        failures.append("controlled_target_or_stereo_regression")
                    for key, limit in (("relative_channel_gain_error_db", 1), ("side_to_mid_balance_error_db", 2),
                                       ("interchannel_correlation_error", 0.2)):
                        if quality[key] is not None and abs(quality[key]) > limit:
                            failures.append(key)
                    if "spatial_gain_max_extra_error_db" in plan["decision_rule"]:
                        from lss.validation.stereo_comparison import spatial_regressions
                        spatial = spatial_regressions(baseline, quality,
                            maximum_extra_gain_error_db=plan["decision_rule"]["spatial_gain_max_extra_error_db"],
                            maximum_component_si_regression_db=plan["decision_rule"]["spatial_component_si_max_regression_db"])
                        if spatial:
                            failures.append("individual_stereo_component_regression")
                        quality["spatial_regressions"] = spatial
                    if components and "maximum_extra_wanted_component_gain_error_db" in plan["decision_rule"]:
                        damages = component_regressions(baseline["wanted_components"], quality["wanted_components"],
                            maximum_extra_gain_error_db=plan["decision_rule"]["maximum_extra_wanted_component_gain_error_db"])
                        quality["wanted_component_regressions"] = damages
                        if damages:
                            failures.append("individual_wanted_component_regression")
            if case["target_absent"]:
                if quality["output_to_input_db"] > plan["decision_rule"]["maximum_absent_output_to_input_db"]:
                    failures.append("target_absent_leakage")
                with TemporaryDirectory(prefix="absence-", dir=args.output) as temporary:
                    paths = {}
                    for role, values in routed.stems.items():
                        path = Path(temporary) / f"{role}.wav"
                        sf.write(path, values.T, 44100, subtype="FLOAT")
                        paths[role.title()] = path
                    count = len(discover_stem_energy_share_files(paths, activity).candidates)
                quality["activity_candidate_count"] = count
                if count > plan["decision_rule"]["maximum_absent_activity_candidate_count"]:
                    failures.append("target_absent_activity_false_positive")
            result = {"label": value["label"], "recipe_id": value["recipe_id"], "quality": quality,
                      "parent_preservation": preservation,
                      "paired_delta_db": deltas, "failures": failures, "music_peak": peak,
                      "mean_weights": {"parent": float(routed.parent_weight[crop].mean()),
                                       "dominant": float(routed.dominant_weight[crop].mean()),
                                       "mixed": float(routed.mixed_weight[crop].mean())},
                      "reconstruction_error_to_input_db": db_ratio(energy(sum(routed.stems.values()) - raw), energy(raw))}
            if case["evidence_role"] != "controlled-clean-component":
                folder = args.output / case["case_id"]
                folder.mkdir(exist_ok=True)
                path = folder / (value["label"] + "-music.wav")
                sf.write(path, routed.stems["music"].T, 44100, subtype="FLOAT")
                result["music_artifact"] = artifact(path)
            candidates.append(result)
        results.append({"case_id": case["case_id"], "evidence_role": case["evidence_role"],
                        "singing_guard": case["singing_guard"], "target_absent": case["target_absent"],
                        "baseline": baseline, "candidates": candidates})
        print(json.dumps({"case_id": case["case_id"], "joint_deltas": {item["label"]: item["paired_delta_db"].get("joint")
                          for item in candidates}}), flush=True)
    summaries = []
    for value in plan["candidates"]:
        paired = [(case, next(item for item in case["candidates"] if item["label"] == value["label"])) for case in results]
        failures = [{"case_id": case["case_id"], "failures": item["failures"]} for case, item in paired if item["failures"]]
        improvements = [case["case_id"] for case, item in paired if case["evidence_role"] == "controlled-clean-component"
                        and item["paired_delta_db"].get("joint", -300) >= plan["decision_rule"]["minimum_material_controlled_improvement_db"]]
        summaries.append({"label": value["label"], "recipe_id": value["recipe_id"],
                          "status": "DEVELOPMENT_SURVIVOR" if not failures and improvements else "REJECTED_EXACT_CONFIGURATION",
                          "failures": failures, "material_controlled_improvements": improvements})
    checked_plan(parent_path)
    checked_receipt(parent_path, parent, receipt)
    if sha256(args.plan) != plan_hash:
        raise ValueError("比較中に固定routing planが変更されています")
    for relative, expected in plan["code"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("比較中に固定routing codeが変更されています")
    write_json(args.output / "report.json", {"status": "COMPLETED_DEVELOPMENT_COMPARISON",
               "plan": {"path": str(args.plan.resolve()), "sha256": sha256(args.plan)},
               "receipt": {"path": str(args.receipt.resolve()), "sha256": sha256(args.receipt)},
               "promotion_allowed": False, "holdout_consumed": False, "l4_oracle": None,
               "source_identity_preserved": True, "evidence_scope": plan["evidence_scope"],
               "summaries": summaries, "cases": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("parent", type=Path)
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("--preserve-parent", action="store_true")
    execute = commands.add_parser("evaluate")
    execute.add_argument("plan", type=Path)
    execute.add_argument("receipt", type=Path)
    execute.add_argument("output", type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
