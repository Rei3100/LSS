"""前比較で選んだvocal基底0.25 blend一案を、同じ固定33ケースで確認する。"""
from __future__ import annotations

import argparse
import copy
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
from lss.bgm.four_stem_residual import FourStemResidualRecipe
from lss.harness.fetch import file_sha256
from lss.validation import reference_residual_episode as guarded
from scripts import probe_bgm_four_stem_residual as base

RECIPE = FourStemResidualRecipe("music_vocals", blend=0.25)


def freeze(parent_dir: Path, output: Path):
    parent_dir, output = parent_dir.resolve(), output.resolve()
    parent_plan_path, parent_report_path, parent_summary_path = (
        parent_dir/name for name in ("plan.json","report.json","summary.json"))
    parent, report, summary = map(base.read, (parent_plan_path,parent_report_path,parent_summary_path))
    old_recipes = base.RECIPES
    try:
        base.RECIPES = tuple(FourStemResidualRecipe(**item) for item in parent["recipes"])
        base.verify(parent)
    finally:
        base.RECIPES = old_recipes
    if (report.get("status") != "COMPLETED_FIXED_SOURCE_FITTED_FOUR_STEM_RESIDUAL_COMPARISON"
            or any(row["decision"] != "REJECTED_EXACT_CONFIGURATION" for row in report["summary"])
            or summary.get("selected_next_route",{}).get("route") != "VOCAL_BASIS_QUARTER_BLEND_DEVELOPMENT_PROBE"
            or report["plan"]["sha256"] != file_sha256(parent_plan_path)):
        raise ValueError("前比較で固定したquarter vocal probeが必要です")
    paths = [parent_plan_path,parent_report_path,parent_summary_path,Path(__file__),
             ROOT/"scripts/probe_bgm_four_stem_residual.py", ROOT/"lss/bgm/four_stem_residual.py",
             ROOT/"lss/validation/stereo_quality.py", ROOT/"lss/validation/stereo_comparison.py",
             ROOT/"scripts/probe_bgm_reference_alignment.py"]
    inputs = {str(path.resolve()):file_sha256(path) for path in paths}
    for case in parent["cases"]:
        for item in (case["source"],case["reference"],case["baseline_music"],*case["four_stems"].values()):
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("固定音声が変更されています")
            inputs[item["path"]] = item["sha256"]
    plan = {"schema_version":1, "status":"FROZEN_BEFORE_RESULTS",
        "promotion_allowed":False, "holdout_consumed":False, "sample_rate":44100,
        "parent":{name:base.art(path) for name,path in
                  (("plan",parent_plan_path),("report",parent_report_path),("summary",parent_summary_path))},
        "cases":copy.deepcopy(parent["cases"]), "recipes":[RECIPE.to_dict()],
        "gates":copy.deepcopy(parent["gates"]), "inputs":inputs,
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,
                   "soundfile":sf.__version__,"libsndfile":sf.__libsndfile_version__},
        "decision_inputs":parent["decision_inputs"], "quality_only_inputs":parent["quality_only_inputs"],
        "limits":["前比較の結果からvocal基底・0.25 blend一案を固定し、全33ケースへ同じに適用。",
                  *parent["limits"][1:]]}
    old_recipes = base.RECIPES
    try:
        base.RECIPES = (RECIPE,)
        output.mkdir(parents=True, exist_ok=False)
        target = output/"plan.json"
        guarded._write_record(target,plan,lambda:base.verify(plan))
    finally:
        base.RECIPES = old_recipes
    return target


def evaluate(plan_path: Path):
    old_recipes = base.RECIPES
    try:
        base.RECIPES = (RECIPE,)
        return base.evaluate(plan_path)
    finally:
        base.RECIPES = old_recipes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    freezing=commands.add_parser("freeze")
    freezing.add_argument("--parent-dir",type=Path,
        default=dev_state_root(ROOT) / "validation/bgm-four-stem-residual-controls-r1")
    freezing.add_argument("--out-dir",type=Path,required=True)
    evaluating=commands.add_parser("evaluate")
    evaluating.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    result=freeze(args.parent_dir,args.out_dir) if args.command=="freeze" else evaluate(args.plan)
    print(result,flush=True)


if __name__ == "__main__":
    main()
