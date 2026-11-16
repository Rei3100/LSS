"""既知録音との一致から、候補が何曲を含むかと曲の境目を読み取る。

新しい推論も照合も行わない。凍結済みの候補と一致記録を読み直し、一つの候補の中に
別々の録音が並んでいるかを数える。別々の録音が並ぶ位置は、信号の外側からの根拠で
示された曲の境目である。

その位置を、信号だけを見る区切り候補が見つけられるかも併せて測る。意味的role・
公開可否は扱わない。公開・既定変更・Holdout消費は行わない。
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

from lss.bgm.cue_segmentation import CueSegmentationRecipe, segment_cues  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する条件。一致の読み取りは記述、区切り候補の側だけを判定する。
DECISION_RULE = {
    "identity_reading_is_described_not_judged": True,
    "implied_boundary_must_be_found_within_tolerance": True,
    "implied_boundary_must_be_the_strongest_in_the_candidate": False,
}


class IdentityBoundaryProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def build(episode: Path, recipe_path: Path, output: Path, *,
          tolerance_seconds: float, minimum_similarity: float) -> Path:
    run_path, identity_path = episode / "run.json", episode / "identity.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if run.get("promotion_allowed") is not False or run.get("publication_state") != "not_published":
        raise IdentityBoundaryProbeError("固定Development記録だけを読み直します")
    if identity.get("type") != "episode_identity_annotation":
        raise IdentityBoundaryProbeError("この番組の一致記録ではありません")

    recipe = CueSegmentationRecipe(**json.loads(recipe_path.read_text(encoding="utf-8")))
    recipe.validate()
    output.mkdir(parents=True, exist_ok=False)
    frozen_recipe = output / "recipe.json"
    shutil.copyfile(recipe_path, frozen_recipe)

    candidates = []
    for item in sorted(run["candidates"], key=lambda row: row["origin"]["start_frame"]):
        asset, origin = item["asset"], item["origin"]
        if asset.get("publication_state") != "not_published":
            raise IdentityBoundaryProbeError("公開済みの候補は読み直しません")
        candidates.append({
            "candidate_asset_id": asset["candidate_asset_id"],
            "path": asset["path"], "sha256": asset["sha256"],
            "sample_rate": asset["sample_rate"], "frames": asset["frames"],
            "headroom_shift": asset["serialization"]["headroom_shift"],
            "start_seconds": origin["start_frame"] / asset["sample_rate"],
            "end_seconds": origin["end_frame"] / asset["sample_rate"],
        })

    matches = [{
        "reference_id": row["reference_id"], "reference_sha256": row["reference_sha256"],
        "query_start_seconds": row["query_start_seconds"],
        "query_end_seconds": row["query_end_seconds"],
        "similarity": row["similarity"], "identity_state": row["identity_state"],
        "evidence_kind": row["evidence_kind"],
    } for row in sorted(identity["matches"], key=lambda row: row["query_start_seconds"])]

    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_identity_boundary_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "episode": {"run": {"path": str(run_path), "sha256": fetch.file_sha256(run_path)},
                    "identity": {"path": str(identity_path),
                                 "sha256": fetch.file_sha256(identity_path)},
                    "source_id": run["source"]["source_id"]},
        "tolerance_seconds": tolerance_seconds,
        "minimum_similarity": minimum_similarity,
        "recipe": {"path": str(frozen_recipe), "sha256": fetch.file_sha256(frozen_recipe),
                   "recipe_id": recipe.recipe_id, "revision": recipe.to_dict()},
        "candidates": candidates, "matches": matches,
        "decision_rule": DECISION_RULE,
        "limits": [
            "reads_frozen_candidates_and_matches_only_no_new_inference",
            "identity_covers_released_music_only_unknown_cues_stay_invisible",
            "a_boundary_between_two_matched_references_is_evidence_not_a_complete_cue_list",
            "novelty_is_an_observation_not_a_semantic_cue_boundary",
            "no_publication_default_change_or_holdout_use",
        ],
    })
    return plan


def _verify_inputs(plan: dict) -> None:
    artifacts = [plan["episode"]["run"], plan["episode"]["identity"], plan["recipe"]]
    artifacts.extend(plan["candidates"])
    for artifact in artifacts:
        if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise IdentityBoundaryProbeError(f"固定入力が変更されています: {artifact['path']}")


def _implied_boundaries(rows: list[dict]) -> list[dict]:
    """並んだ二つの一致が別の録音なら、その間を曲の境目として読む。"""
    boundaries = []
    for first, second in zip(rows, rows[1:]):
        if first["reference_id"] == second["reference_id"]:
            continue
        boundaries.append({
            "before_reference_id": first["reference_id"],
            "after_reference_id": second["reference_id"],
            "earliest_seconds": first["query_end_seconds"],
            "latest_seconds": second["query_start_seconds"],
            "seconds": (first["query_end_seconds"] + second["query_start_seconds"]) / 2.0,
        })
    return boundaries


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_identity_boundary_plan"):
        raise IdentityBoundaryProbeError("固定した一致読み取りの計画だけを実行します")
    _verify_inputs(plan)
    output = plan_path.parent
    recipe = CueSegmentationRecipe(**plan["recipe"]["revision"])
    tolerance, minimum = plan["tolerance_seconds"], plan["minimum_similarity"]
    usable = [row for row in plan["matches"] if row["similarity"] >= minimum]

    records, failures = [], []
    for candidate in plan["candidates"]:
        start, end = candidate["start_seconds"], candidate["end_seconds"]
        inside = [row for row in usable
                  if row["query_start_seconds"] < end and start < row["query_end_seconds"]]
        references = sorted({row["reference_id"] for row in inside})
        covered = sum(min(end, row["query_end_seconds"]) - max(start, row["query_start_seconds"])
                      for row in inside)
        implied = _implied_boundaries(inside)
        row = {
            "candidate_asset_id": candidate["candidate_asset_id"],
            "start_seconds": round(start, 2), "end_seconds": round(end, 2),
            "seconds": round(end - start, 2),
            "matched_reference_count": len(references),
            "matched_references": references,
            "matched_seconds": round(covered, 2),
            "matched_fraction": round(covered / (end - start), 4) if end > start else None,
            "implied_boundaries": implied,
            "contains_more_than_one_released_recording": len(references) > 1,
        }
        if implied:
            values, rate = sf.read(candidate["path"], dtype="float64", always_2d=True)
            samples = (values * 2.0 ** candidate["headroom_shift"]).T
            result = segment_cues(samples, int(rate), recipe)
            if result.recipe_id != plan["recipe"]["recipe_id"]:
                raise IdentityBoundaryProbeError("Recipe identityが固定記録と一致しません")
            found = []
            for boundary in implied:
                target = boundary["seconds"] - start
                near = [item for item in result.boundaries
                        if abs(item.seconds - target) <= tolerance]
                best = max(near, key=lambda item: item.novelty, default=None)
                found.append({
                    "boundary_seconds_in_episode": round(boundary["seconds"], 2),
                    "boundary_seconds_in_candidate": round(target, 2),
                    "found": best is not None,
                    "novelty": None if best is None else best.novelty,
                    "offset_seconds": None if best is None else round(best.seconds - target, 2),
                })
                if best is None:
                    failures.append(
                        f"{candidate['candidate_asset_id'][:28]}:"
                        f"{round(boundary['seconds'], 2)}秒の境目に区切り候補が出ない")
            row["segmentation"] = {
                "recipe_id": result.recipe_id,
                "boundary_count": len(result.boundaries),
                "maximum_novelty": max(result.novelty),
                "boundaries_in_candidate_seconds": [round(item.seconds, 2)
                                                    for item in result.boundaries],
                "implied_boundary_checks": found,
            }
        records.append(row)
        print(f"一致の読み取り完了: {candidate['candidate_asset_id'][:28]} / "
              f"録音{row['matched_reference_count']}件", flush=True)

    _verify_inputs(plan)
    multi = [row for row in records if row["contains_more_than_one_released_recording"]]
    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_identity_boundary_report",
        "status": "complete_real_episode_description",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "source_id": plan["episode"]["source_id"],
        "decision_rule": plan["decision_rule"],
        "summary": {
            "candidate_count": len(records),
            "candidates_with_any_match": sum(1 for row in records
                                             if row["matched_reference_count"] > 0),
            "candidates_with_more_than_one_recording": len(multi),
            "implied_boundary_count": sum(len(row["implied_boundaries"]) for row in records),
            "segmentation_found_every_implied_boundary": not failures,
            "segmentation_failures": failures,
        },
        "candidates": records,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/cue_segmentation.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--episode", type=Path, help="固定済みの実番組候補生成のdirectory")
    parser.add_argument("--recipe", type=Path, help="先に宣言した区切り候補Recipe")
    parser.add_argument("--tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-similarity", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.episode is None or args.recipe is None:
            parser.error("--episodeと--recipeが必要です")
        plan = build(args.episode.resolve(), args.recipe.resolve(), output,
                     tolerance_seconds=args.tolerance_seconds,
                     minimum_similarity=args.minimum_similarity)
    print(f"一致からの境目の記述を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
