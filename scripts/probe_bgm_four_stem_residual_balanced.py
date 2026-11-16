"""左右の両方で原本残差を説明するquarter vocal回収を固定比較する。"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from lss.bgm.four_stem_residual import FourStemResidualRecipe
from lss.harness.fetch import file_sha256
from lss.validation import reference_residual_episode as guarded
from scripts import probe_bgm_four_stem_residual as base

RECIPE=FourStemResidualRecipe("music_vocals",blend=0.25,
                              minimum_channel_explained_fraction=0.1)
PARENT_COMMIT="70d4b14"


def verify_parent(plan):
    historical=[]
    for text,expected in plan["inputs"].items():
        path=Path(text)
        if file_sha256(path)==expected:
            continue
        object_name=PARENT_COMMIT+":"+path.relative_to(ROOT).as_posix()
        original=subprocess.check_output(["git","show",object_name])
        if hashlib.sha256(original).hexdigest()!=expected:
            raise ValueError(f"前比較の固定入力が復元できません: {text}")
        historical.append(object_name)
    if historical!=[PARENT_COMMIT+":lss/bgm/four_stem_residual.py"]:
        raise ValueError(f"前比較から変更された実装範囲が不正です: {historical}")
    return historical


def freeze(parent_dir:Path,output:Path):
    parent_dir,output=parent_dir.resolve(),output.resolve()
    plan_path,report_path,summary_path=(parent_dir/name for name in
        ("plan.json","report.json","summary.json"))
    parent,report,summary=map(base.read,(plan_path,report_path,summary_path))
    historical=verify_parent(parent)
    if (report.get("status")!="COMPLETED_FIXED_SOURCE_FITTED_FOUR_STEM_RESIDUAL_COMPARISON"
            or report["summary"]!=[summary["decision"]]
            or report["summary"][0]["decision"]!="REJECTED_EXACT_CONFIGURATION"
            or summary.get("selected_next_route",{}).get("route")!=
                "CHANNEL_BALANCED_QUARTER_VOCAL_RESIDUAL_PROBE"
            or report["plan"]["sha256"]!=file_sha256(plan_path)):
        raise ValueError("片側回帰を特定した固定比較が必要です")
    paths=[plan_path,report_path,summary_path,Path(__file__),
           ROOT/"scripts/probe_bgm_four_stem_residual.py",ROOT/"lss/bgm/four_stem_residual.py",
           ROOT/"lss/validation/stereo_quality.py",ROOT/"lss/validation/stereo_comparison.py",
           ROOT/"scripts/probe_bgm_reference_alignment.py"]
    inputs={str(path.resolve()):file_sha256(path) for path in paths}
    for case in parent["cases"]:
        for item in (case["source"],case["reference"],case["baseline_music"],*case["four_stems"].values()):
            if file_sha256(Path(item["path"]))!=item["sha256"]:
                raise ValueError("固定音声が変更されています")
            inputs[item["path"]]=item["sha256"]
    plan={"schema_version":1,"status":"FROZEN_BEFORE_RESULTS",
        "promotion_allowed":False,"holdout_consumed":False,"sample_rate":44100,
        "parent":{name:base.art(path) for name,path in
                  (("plan",plan_path),("report",report_path),("summary",summary_path))},
        "parent_historical_implementation":historical,
        "cases":copy.deepcopy(parent["cases"]),"recipes":[RECIPE.to_dict()],
        "gates":copy.deepcopy(parent["gates"]),"inputs":inputs,
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,
                   "soundfile":sf.__version__,"libsndfile":sf.__libsndfile_version__},
        "decision_inputs":parent["decision_inputs"],"quality_only_inputs":parent["quality_only_inputs"],
        "limits":["前比較で特定した片側回帰へ、両channel 10%説明gateだけを追加。全33ケースで条件を固定。",
                  *parent["limits"][1:]]}
    old=base.RECIPES
    try:
        base.RECIPES=(RECIPE,)
        output.mkdir(parents=True,exist_ok=False)
        target=output/"plan.json"
        guarded._write_record(target,plan,lambda:base.verify(plan))
    finally:
        base.RECIPES=old
    return target


def evaluate(plan_path:Path):
    old=base.RECIPES
    try:
        base.RECIPES=(RECIPE,)
        return base.evaluate(plan_path)
    finally:
        base.RECIPES=old


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    freezing=commands.add_parser("freeze")
    freezing.add_argument("--parent-dir",type=Path,
        default=dev_state_root(ROOT) / "validation/bgm-four-stem-residual-controls-r2")
    freezing.add_argument("--out-dir",type=Path,required=True)
    evaluating=commands.add_parser("evaluate")
    evaluating.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    result=freeze(args.parent_dir,args.out_dir) if args.command=="freeze" else evaluate(args.plan)
    print(result,flush=True)


if __name__=="__main__":main()
