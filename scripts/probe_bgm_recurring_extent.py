"""別の回に同じ録音が出ることで、参照の無い候補の範囲がどれだけ読めるかを測る。

新しい推論も照合も行わない。固定済みのシリーズ照合記録を読み直し、候補ごとに
「一曲だと言える」「複数曲を含む」「まだ言えない」を数える。既発売の参照録音だけで
読んだ場合と並べ、参照の無い候補がどれだけ増えるかを見る。

意味的role・公開可否は扱わない。公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm.cue_extent import (  # noqa: E402
    CueExtentRecipe,
    combine_readings,
    matches_from_identity_evidence,
    read_cue_extent,
)
from lss.bgm.recurring_extent import (  # noqa: E402
    read_recurring_extent,
    recurring_matches,
    spans_for_episode,
)
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 候補の端が共通録音の端と同じなら、その候補はこの証拠から切り出されている。
# 範囲が読めたことを、切り出しの独立した裏付けとして数えないために記録する。
BOUNDS_SECONDS = 0.05
# 結果を見る前に固定する。読み取りの強さと覆いは既発売の参照録音と同じ宣言を使う。
DECISION_RULE = {
    "recurring_evidence_is_read_the_same_way_as_released_evidence": True,
    "no_shared_recording_is_not_evidence_of_a_single_or_multiple_cue": True,
    "identity_names_stay_unresolved": True,
}
LIMITS = [
    "reads_frozen_series_identity_records_only_no_new_inference_or_matching",
    "recurring_evidence_needs_another_episode_single_occurrence_cues_stay_invisible",
    "a_recurring_recording_bounds_the_extent_it_does_not_name_the_cue",
    "candidates_already_cut_at_the_same_shared_spans_are_not_independent_confirmation",
    "semantic_role_and_publication_are_untouched",
    "no_publication_default_change_or_holdout_use",
]


class RecurringExtentProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def build(record: Path, output: Path, *, minimum_similarity: float,
          minimum_recurring_similarity: float, minimum_coverage: float,
          maximum_offset_spread_seconds: float) -> Path:
    annotation = json.loads(record.read_text(encoding="utf-8"))
    if annotation.get("type") != "series_identity_annotation":
        raise RecurringExtentProbeError(f"このシリーズ照合記録は読めません: {record}")
    released = CueExtentRecipe(
        minimum_similarity=minimum_similarity,
        minimum_single_recording_coverage=minimum_coverage,
        maximum_offset_spread_seconds=maximum_offset_spread_seconds)
    recurring = CueExtentRecipe(
        minimum_similarity=minimum_recurring_similarity,
        minimum_single_recording_coverage=minimum_coverage,
        maximum_offset_spread_seconds=maximum_offset_spread_seconds)
    released.validate()
    recurring.validate()
    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_recurring_extent_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "record": {"path": str(record), "sha256": fetch.file_sha256(record)},
        "released_recipe": {"recipe_id": released.recipe_id, "revision": released.to_dict()},
        "recurring_recipe": {"recipe_id": recurring.recipe_id, "revision": recurring.to_dict()},
        "decision_rule": DECISION_RULE, "limits": LIMITS,
    })
    return plan


def _verify_inputs(plan: dict) -> None:
    artifact = plan["record"]
    if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
        raise RecurringExtentProbeError(f"固定入力が変更されています: {artifact['path']}")


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_recurring_extent_plan"):
        raise RecurringExtentProbeError("固定した範囲読み取りの計画だけを実行します")
    _verify_inputs(plan)
    annotation = json.loads(Path(plan["record"]["path"]).read_text(encoding="utf-8"))
    released_recipe = CueExtentRecipe(**plan["released_recipe"]["revision"])
    recurring_recipe = CueExtentRecipe(**plan["recurring_recipe"]["revision"])

    spans: dict[str, list[dict]] = {}
    for pair in annotation["shared_recordings"]:
        for key, rows in spans_for_episode(
                pair["occurrences"], first=pair["first"], second=pair["second"]).items():
            spans.setdefault(key, []).extend(rows)

    episodes = []
    for recipe_entry in annotation["activity_recipes"]:
        for key, candidates in recipe_entry["candidates"].items():
            matches = recurring_matches(spans.get(key, []))
            rows: list[dict] = []
            counts: dict[str, dict[str, int]] = {}
            for candidate in candidates:
                start, end = candidate["start_seconds"], candidate["end_seconds"]
                inside = [row for row in matches
                          if row["query_start_seconds"] < end and start < row["query_end_seconds"]]
                released = read_cue_extent(start, end,
                                           matches_from_identity_evidence(candidate),
                                           released_recipe)
                recurring = read_recurring_extent(start, end, inside, recurring_recipe)
                combined = combine_readings([released, recurring])
                for name, reading in (("released", released), ("recurring", recurring),
                                      ("combined", combined)):
                    counts.setdefault(name, {})
                    counts[name][reading["state"]] = counts[name].get(reading["state"], 0) + 1
                bounds = [row for row in inside
                          if abs(row["query_start_seconds"] - start) <= BOUNDS_SECONDS
                          and abs(row["query_end_seconds"] - end) <= BOUNDS_SECONDS]
                rows.append({
                    "candidate_id": candidate["candidate_id"],
                    "bounds_equal_a_recurring_span": bool(bounds),
                    "start_seconds": round(start, 3), "end_seconds": round(end, 3),
                    "released_state": released["state"],
                    "released_reason": released["evidence"].get("reason"),
                    "recurring_state": recurring["state"],
                    "recurring_reason": recurring["evidence"].get("reason"),
                    "recurring_match_count": len(inside),
                    "recurring_coverage": (
                        round(recurring["evidence"]["single_recording_coverage"], 4)
                        if "single_recording_coverage" in recurring["evidence"] else None),
                    "combined_state": combined["state"],
                })
            gained = [row for row in rows if row["released_state"] == "unresolved"
                      and row["recurring_state"] != "unresolved"]
            independent = [row for row in gained if not row["bounds_equal_a_recurring_span"]]
            episodes.append({
                "episode": key, "recipe_id": recipe_entry["recipe_id"],
                "candidate_count": len(rows),
                "recurring_span_count": len(spans.get(key, [])),
                "recurring_recording_count": len({row["reference_id"] for row in matches}),
                "states": counts,
                "decided_only_by_recurring_evidence": len(gained),
                "decided_without_being_cut_at_the_same_span": len(independent),
                "decided_only_by_recurring_evidence_rows": gained,
                "candidates": rows,
            })
            print("読み取り完了: {} / 候補{}件 / 再登場だけで決まった{}件".format(
                key, len(rows), len(gained)), flush=True)

    _verify_inputs(plan)
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_recurring_extent_report",
        "status": "complete_reading_of_frozen_records",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False, "anything_published": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "released_recipe": plan["released_recipe"], "recurring_recipe": plan["recurring_recipe"],
        "decision_rule": plan["decision_rule"],
        "summary": {
            "episode_count": len(episodes),
            "candidate_count": sum(row["candidate_count"] for row in episodes),
            "decided_only_by_recurring_evidence": sum(
                row["decided_only_by_recurring_evidence"] for row in episodes),
            "decided_without_being_cut_at_the_same_span": sum(
                row["decided_without_being_cut_at_the_same_span"] for row in episodes),
            "single_cue_released": sum(row["states"]["released"].get("single_cue", 0)
                                       for row in episodes),
            "single_cue_combined": sum(row["states"]["combined"].get("single_cue", 0)
                                       for row in episodes),
            "multiple_cues_combined": sum(row["states"]["combined"].get("multiple_cues", 0)
                                          for row in episodes),
        },
        "episodes": episodes,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/recurring_extent.py", ROOT / "lss/bgm/cue_extent.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--record", type=Path, help="固定済みのシリーズ照合記録")
    parser.add_argument("--minimum-similarity", type=float, default=0.15)
    parser.add_argument("--minimum-recurring-similarity", type=float, default=0.15)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--maximum-offset-spread-seconds", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.record is None:
            parser.error("--recordが必要です")
        plan = build(args.record.resolve(), output,
                     minimum_similarity=args.minimum_similarity,
                     minimum_recurring_similarity=args.minimum_recurring_similarity,
                     minimum_coverage=args.minimum_coverage,
                     maximum_offset_spread_seconds=args.maximum_offset_spread_seconds)
    print(f"再登場からの範囲の記述を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
