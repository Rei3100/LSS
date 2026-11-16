"""実番組の固定stemへ区間規則をかけ、候補区間の違いだけを記述する。

正解labelがないので良し悪しは判定しない。どの候補が統合され、どこが広がり、
どれだけ長さが変わるかを記録するだけである。分離は行わず、固定済みの推論成果を
読み直す。公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from lss.bgm.activity import (  # noqa: E402
    STEM_SHARE_RECIPES,
    discover_stem_energy_share_files,
    load_discovery_recipe,
)
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
STEM_FILES = (("Music", "music.wav"), ("Speech", "speech.wav"), ("Effects", "effects.wav"))


class EpisodeBoundaryProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def build(episode: Path, stems_root: Path, recipes: list[Path], output: Path) -> Path:
    episode_plan_path = episode / "plan.json"
    episode_plan = json.loads(episode_plan_path.read_text(encoding="utf-8"))
    episode_run_path = episode / "run.json"
    episode_run = json.loads(episode_run_path.read_text(encoding="utf-8"))
    if episode_plan.get("promotion_allowed") is not False:
        raise EpisodeBoundaryProbeError("固定Development記録だけを読み直します")
    if episode_run.get("publication_state") != "not_published":
        raise EpisodeBoundaryProbeError("公開済みの結果は読み直しません")

    output.mkdir(parents=True, exist_ok=False)
    frozen_recipes = []
    for index, recipe_path in enumerate(recipes, start=1):
        recipe = load_discovery_recipe(recipe_path)
        if not isinstance(recipe, STEM_SHARE_RECIPES):
            raise EpisodeBoundaryProbeError(f"3 stemを見るRecipeが必要です: {recipe.method}")
        frozen = output / f"recipe-{index:02d}-{recipe.method}.json"
        shutil.copyfile(recipe_path, frozen)
        frozen_recipes.append({
            "role": "baseline" if index == 1 else "variant", "method": recipe.method,
            "recipe_id": recipe.recipe_id, "path": str(frozen),
            "sha256": fetch.file_sha256(frozen), "revision": recipe.to_dict(),
        })
    if len({item["recipe_id"] for item in frozen_recipes}) != len(frozen_recipes):
        raise EpisodeBoundaryProbeError("比較するRecipeが重複しています")

    stems = {}
    layouts = set()
    for stem, name in STEM_FILES:
        path = (stems_root / name).resolve()
        if not path.is_file():
            raise EpisodeBoundaryProbeError(f"固定記録にstemがありません: {path}")
        info = sf.info(path)
        layouts.add((info.samplerate, info.frames, info.channels))
        stems[stem] = {"path": str(path), "sha256": fetch.file_sha256(path),
                       "sample_rate": info.samplerate, "frames": info.frames}
    if len(layouts) != 1:
        raise EpisodeBoundaryProbeError(f"stem layoutが一致しません: {sorted(layouts)}")

    # 記録された候補は区間へcontextを足したcueなので、区間そのものと比べる。
    existing = sorted(
        ({"start_frame": start, "end_frame": end}
         for cue in episode_run["rendering"]["cues"]
         for start, end in cue["activity_intervals"]),
        key=lambda item: (item["start_frame"], item["end_frame"]),
    )
    existing_cues = sorted(
        ({"start_frame": item["origin"]["start_frame"], "end_frame": item["origin"]["end_frame"]}
         for item in episode_run["candidates"]),
        key=lambda item: (item["start_frame"], item["end_frame"]),
    )
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_episode_boundary_description_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "scores_quality": False,
        "episode": {"plan": {"path": str(episode_plan_path),
                             "sha256": fetch.file_sha256(episode_plan_path)},
                    "run": {"path": str(episode_run_path),
                            "sha256": fetch.file_sha256(episode_run_path)},
                    "source": episode_run["source"], "source_id": episode_run["source"]["source_id"]},
        "stems": stems, "recipes": frozen_recipes,
        "existing_activity_intervals": existing,
        "existing_routed_cues": existing_cues,
        "existing_activity_recipe_id": episode_plan["activity_recipe"]["recipe_id"],
        "limits": [
            "reuses_frozen_separation_outputs_no_new_inference",
            "no_semantic_ground_truth_for_this_episode",
            "difference_is_described_not_scored",
            "no_publication_default_change_or_holdout_use",
        ],
    })
    return plan


def _verify_inputs(plan: dict) -> None:
    for artifact in list(plan["stems"].values()) + plan["recipes"]:
        if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise EpisodeBoundaryProbeError(f"固定入力が変更されています: {artifact['path']}")


def _spans(discovery) -> list[dict]:
    return [{"start_frame": item.start_frame, "end_frame": item.end_frame}
            for item in discovery.candidates]


def _describe(baseline: list[dict], variant: list[dict], *, rate: int) -> dict:
    """変更後の各区間が、変更前のどの区間を含むかで統合と延長を説明する。"""
    groups = []
    for target in variant:
        contained = [index for index, item in enumerate(baseline)
                     if item["start_frame"] < target["end_frame"]
                     and target["start_frame"] < item["end_frame"]]
        covered_start = min((baseline[index]["start_frame"] for index in contained), default=None)
        covered_end = max((baseline[index]["end_frame"] for index in contained), default=None)
        groups.append({
            "start_seconds": round(target["start_frame"] / rate, 2),
            "end_seconds": round(target["end_frame"] / rate, 2),
            "seconds": round((target["end_frame"] - target["start_frame"]) / rate, 2),
            "merged_previous_candidates": len(contained),
            "previous_indices": contained,
            "added_head_seconds": None if covered_start is None else round(
                (covered_start - target["start_frame"]) / rate, 2),
            "added_tail_seconds": None if covered_end is None else round(
                (target["end_frame"] - covered_end) / rate, 2),
        })
    matched = {index for group in groups for index in group["previous_indices"]}
    def total(spans):
        return sum(item["end_frame"] - item["start_frame"] for item in spans) / rate
    return {
        "previous_count": len(baseline), "current_count": len(variant),
        "previous_total_seconds": round(total(baseline), 2),
        "current_total_seconds": round(total(variant), 2),
        "previous_candidates_not_covered": sorted(set(range(len(baseline))) - matched),
        "groups": groups,
    }


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_episode_boundary_description_plan"):
        raise EpisodeBoundaryProbeError("固定した区間記述の計画だけを実行します")
    _verify_inputs(plan)
    output = plan_path.parent
    stem_paths = {stem: Path(item["path"]) for stem, item in plan["stems"].items()}
    rate = next(iter(plan["stems"].values()))["sample_rate"]

    results = {}
    for frozen in plan["recipes"]:
        recipe = load_discovery_recipe(Path(frozen["path"]))
        discovery = discover_stem_energy_share_files(stem_paths, recipe)
        if discovery.recipe_id != frozen["recipe_id"]:
            raise EpisodeBoundaryProbeError("Recipe identityが固定記録と一致しません")
        results[frozen["recipe_id"]] = _spans(discovery)
        print(f"実番組の区間を求めました: {frozen['method']} / {len(discovery.candidates)}件",
              flush=True)

    baseline_id = plan["recipes"][0]["recipe_id"]
    _verify_inputs(plan)
    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_episode_boundary_description_report",
        "status": "complete_real_episode_description",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False, "scores_quality": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "source_id": plan["episode"]["source_id"],
        "candidates": results,
        "reproduces_existing_activity_intervals":
            results[baseline_id] == plan["existing_activity_intervals"],
        "differences": {
            frozen["recipe_id"]: _describe(results[baseline_id], results[frozen["recipe_id"]],
                                           rate=rate)
            for frozen in plan["recipes"][1:]
        },
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/activity.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--episode", type=Path, help="固定済みの実番組候補生成のdirectory")
    parser.add_argument("--stems-root", type=Path, help="music/speech/effects.wavを持つdirectory")
    parser.add_argument("--recipe", type=Path, action="append", default=[],
                        help="比較するactivity Recipe。最初が現行")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.episode is None or args.stems_root is None or len(args.recipe) < 2:
            parser.error("--episode、--stems-root、2つ以上の--recipeが必要です")
        plan = build(args.episode.resolve(), args.stems_root.resolve(),
                     [path.resolve() for path in args.recipe], output)
    print(f"実番組の区間記述を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
