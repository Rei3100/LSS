"""既存Musicを保持する4stem原本残差回収を、固定33ケースで比較する。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
from lss.bgm.four_stem_residual import FourStemResidualRecipe, recover_four_stem_residual
from lss.harness.fetch import file_sha256
from lss.validation import reference_residual_episode as guarded, stereo_quality
from scripts.probe_bgm_reference_alignment import GATES, failures

RECIPES = (
    FourStemResidualRecipe("music_vocals"),
    FourStemResidualRecipe("combined_music"),
)


def read(path):
    return json.loads(Path(path).read_bytes())


def art(path):
    path = Path(path).resolve()
    return {"path":str(path), "sha256":file_sha256(path)}


def verify(plan):
    if (plan.get("status") != "FROZEN_BEFORE_RESULTS" or plan.get("promotion_allowed") is not False
            or plan.get("holdout_consumed") is not False
            or plan.get("recipes") != [recipe.to_dict() for recipe in RECIPES]):
        raise ValueError("固定した未昇格4stem残差計画が必要です")
    for path,digest in plan["inputs"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"固定入力が変更されています: {path}")


def freeze(parent_dir: Path, output: Path):
    parent_dir, output = parent_dir.resolve(), output.resolve()
    plan_path, receipt_path, report_path = (parent_dir/name for name in
                                             ("plan.json", "receipt-after-external-metadata-recovery.json", "report.json"))
    parent, receipt, report = map(read, (plan_path, receipt_path, report_path))
    if (report.get("status") != "COMPLETED_FIXED_FOUR_STEM_CUDA_DEVELOPMENT_COMPARISON"
            or report.get("promotion_allowed") is not False or report.get("holdout_consumed") is not False
            or report["plan"]["sha256"] != file_sha256(plan_path)
            or receipt.get("status") != "COMPLETED_CUDA_WITHOUT_FALLBACK"
            or receipt["plan_sha256"] != file_sha256(plan_path)
            or len(parent["cases"]) != len(receipt["cases"]) != 0):
        raise ValueError("完了した固定4stem比較が必要です")
    decisions = {row["renderer"]:row["decision"] for row in report["summary"]}
    if decisions != {"instrumental":"REJECTED_EXACT_CONFIGURATION",
                     "instrumental_and_singing":"REJECTED_EXACT_CONFIGURATION"}:
        raise ValueError("直接置換を棄却した比較から残差回収へ進みます")
    inputs = {str(path.resolve()):file_sha256(path) for path in (
        plan_path, receipt_path, report_path, Path(__file__),
        ROOT/"lss/bgm/four_stem_residual.py", ROOT/"lss/validation/stereo_quality.py",
        ROOT/"lss/validation/stereo_comparison.py", ROOT/"scripts/probe_bgm_reference_alignment.py")}
    rows = {row["case_id"]:row for row in receipt["cases"]}
    cases = []
    for case in parent["cases"]:
        row = rows[case["case_id"]]
        current = {"case_id":case["case_id"], "family":case["family"],
            "wanted_role_policy":case["wanted_role_policy"], "source":case["source"],
            "reference":case["reference"], "baseline_music":case["baseline_music"],
            "four_stems":copy.deepcopy(row["stems"])}
        for item in (current["source"], current["reference"], current["baseline_music"],
                     *current["four_stems"].values()):
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("固定caseの入力が変更されています")
            inputs[item["path"]] = item["sha256"]
        cases.append(current)
    plan = {"schema_version":1, "status":"FROZEN_BEFORE_RESULTS",
        "promotion_allowed":False, "holdout_consumed":False, "sample_rate":44100,
        "parent":{name:art(path) for name,path in (("plan",plan_path),("receipt",receipt_path),("report",report_path))},
        "cases":cases, "recipes":[recipe.to_dict() for recipe in RECIPES], "gates":GATES,
        "inputs":inputs, "runtime":{"python":platform.python_version(), "numpy":np.__version__,
                                    "soundfile":sf.__version__, "libsndfile":sf.__libsndfile_version__},
        "decision_inputs":["original source", "existing Music", "named 4stem estimates"],
        "quality_only_inputs":["constructed/clean reference", "target-absent label"],
        "limits":["同じ二つの固定Recipeを全33ケースへ適用。case正解から基底・係数・閾値を選ばない。",
            "原本との局所整合は意味的BGMや台詞不在を証明しない。hard gateは別に評価する。",
            "既存Music・原本・4stem cacheを不変に保ち、候補を公開しない。",
            "Holdout・外部音声送信・新規GPU推論・正式対応・既定変更なし。"]}
    output.mkdir(parents=True, exist_ok=False)
    target = output/"plan.json"
    guarded._write_record(target, plan, lambda:verify(plan))
    return target


def evaluate(plan_path: Path):
    plan = read(plan_path)
    verify(plan)
    plan_sha = file_sha256(plan_path)
    records = []
    for case in plan["cases"]:
        def audio(item):
            value, rate = sf.read(item["path"], dtype="float64", always_2d=True)
            if rate != 44100 or value.shape != (case["source"]["frames"], 2) or not np.isfinite(value).all():
                raise ValueError("固定音声のrate・時間軸・有限値が不正です")
            return value.T
        source, base, target = (audio(case[key]) for key in ("source","baseline_music","reference"))
        estimates = {role:audio(item) for role,item in case["four_stems"].items()}
        absent = not np.any(target)
        before = None if absent else stereo_quality.stereo_quality(base, target)
        for recipe in RECIPES:
            candidate, observations = recover_four_stem_residual(source, base, estimates, recipe)
            measured = None if absent else stereo_quality.stereo_quality(candidate, target)
            bad = failures(before, measured, absent, float(np.sum(base ** 2)), float(np.sum(candidate ** 2)))
            gain = None if absent else measured["scale_dependent_snr_db"]-before["scale_dependent_snr_db"]
            records.append({"case_id":case["case_id"], "family":case["family"],
                "recipe_id":recipe.recipe_id, "recipe":recipe.to_dict(), "target_absent":absent,
                "wanted_role_policy":case["wanted_role_policy"], "before":before, "after":measured,
                "failures":bad, "snr_gain_db":gain,
                "material_improvement":gain is not None and gain >= GATES["meaningful_improvement_db"],
                "candidate_float64_sha256":hashlib.sha256(candidate.tobytes()).hexdigest(),
                "observations":observations})
    summary = []
    for recipe in RECIPES:
        selected = [row for row in records if row["recipe_id"] == recipe.recipe_id]
        bad = [row["case_id"] for row in selected if row["failures"]]
        improved = [row["case_id"] for row in selected if row["material_improvement"]]
        summary.append({"recipe_id":recipe.recipe_id, "basis":recipe.basis,
            "regression_cases":bad, "materially_improved_cases":improved,
            "decision":"REJECTED_EXACT_CONFIGURATION" if bad else
                       "DEVELOPMENT_SURVIVOR_REQUIRES_REAL_DOMAIN_EVIDENCE"})
    report = {"schema_version":1,
        "status":"COMPLETED_FIXED_SOURCE_FITTED_FOUR_STEM_RESIDUAL_COMPARISON",
        "plan":art(plan_path), "case_variant_count":len(records), "cases":records, "summary":summary,
        "inputs_after_verified":True, "promotion_allowed":False, "holdout_consumed":False,
        "winner_selected":False, "source_upload_used":False, "new_gpu_inference_used":False,
        "limits":plan["limits"]}
    def final():
        verify(plan)
        assert file_sha256(plan_path) == plan_sha
    target = plan_path.with_name("report.json")
    guarded._write_record(target, report, final)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--parent-dir", type=Path,
        default=dev_state_root(ROOT) / "validation/bgm-facing-music-controls-r2")
    freeze_parser.add_argument("--out-dir", type=Path, required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.parent_dir,args.out_dir) if args.command == "freeze" else evaluate(args.plan)
    print(result, flush=True)


if __name__ == "__main__":
    main()
