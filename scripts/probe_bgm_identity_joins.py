"""固定した一致記録から、裏付けのある曲の境目が何件数えられるかを読む。

新しい推論も照合も行わない。番組全体の一致記録を読み直し、別々の録音が続けて鳴る位置
だけを境目として数える。信号の内側だけを見る観測では境目を決められなかったので、
その観測を確かめられる件数が実際に集まるのかを先に測る。

意味的role・公開可否は扱わない。公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm.identity_joins import IdentityJoinRecipe, read_joins  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する。一致の読み取りは記述であり、良し悪しの判定はしない。
DECISION_RULE = {
    "joins_are_counted_not_judged": True,
    "same_recording_adjacency_is_not_a_boundary": True,
    "unmatched_regions_stay_invisible": True,
}
LIMITS = [
    "reads_frozen_identity_records_only_no_new_inference_or_matching",
    "identity_covers_released_recordings_only_unknown_cues_stay_invisible",
    "a_join_between_two_matched_recordings_is_evidence_not_a_complete_cue_list",
    "no_publication_default_change_or_holdout_use",
]


class IdentityJoinProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def build(records: list[Path], output: Path, *, minimum_similarity: float,
          minimum_match_seconds: float, maximum_join_seconds: float) -> Path:
    recipe = IdentityJoinRecipe(
        minimum_similarity=minimum_similarity,
        minimum_match_seconds=minimum_match_seconds,
        maximum_join_seconds=maximum_join_seconds)
    recipe.validate()
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("kind") != "bgm_identity_development_probe":
            raise IdentityJoinProbeError(f"この一致記録は読めません: {path}")
        if record.get("split") != "Development":
            raise IdentityJoinProbeError("Developmentの記録だけを読み直します")
    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_identity_join_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "recipe": {"recipe_id": recipe.recipe_id, "revision": recipe.to_dict()},
        "records": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in records],
        "decision_rule": DECISION_RULE,
        "limits": LIMITS,
    })
    return plan


def _verify_inputs(plan: dict) -> None:
    for artifact in plan["records"]:
        if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise IdentityJoinProbeError(f"固定入力が変更されています: {artifact['path']}")


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_identity_join_plan"):
        raise IdentityJoinProbeError("固定した境目読み取りの計画だけを実行します")
    _verify_inputs(plan)
    recipe = IdentityJoinRecipe(**plan["recipe"]["revision"])

    records = []
    for artifact in plan["records"]:
        path = Path(artifact["path"])
        record = json.loads(path.read_text(encoding="utf-8"))
        episodes = []
        for episode in record["episodes"]:
            read = read_joins([{
                "reference_id": row["reference_id"],
                "query_start_seconds": row["query_start_seconds"],
                "query_end_seconds": row["query_end_seconds"],
                "similarity": row["similarity"],
            } for row in episode["own"]], recipe)
            if read["recipe_id"] != plan["recipe"]["recipe_id"]:
                raise IdentityJoinProbeError("Recipe identityが固定記録と一致しません")
            episodes.append({
                "series": episode["series"],
                "signal": episode.get("query", episode["source"]),
                "seconds": episode["seconds"],
                "match_count": read["match_count"],
                "usable_count": read["usable_count"],
                "accepted_count": read["accepted_count"],
                "span_count": len(read["spans"]),
                "matched_seconds": round(sum(span["end_seconds"] - span["start_seconds"]
                                             for span in read["spans"]), 2),
                "join_count": len(read["joins"]),
                "separated_change_count": len(read["separated_changes"]),
                "joins": [{key: (round(value, 3) if isinstance(value, float) else value)
                           for key, value in join.items()} for join in read["joins"]],
                "separated_changes": [{"seconds": round(row["seconds"], 2),
                                       "gap_seconds": round(row["gap_seconds"], 2)}
                                      for row in read["separated_changes"]],
            })
            print(f"読み取り完了: {episode['series']} / 境目{len(read['joins'])}件", flush=True)
        records.append({"path": artifact["path"], "signal": record.get("note"),
                        "episodes": episodes,
                        "join_count": sum(row["join_count"] for row in episodes)})

    _verify_inputs(plan)
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_identity_join_report",
        "status": "complete_reading_of_frozen_records",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False, "anything_published": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "recipe": plan["recipe"], "decision_rule": plan["decision_rule"],
        "summary": {
            "episode_count": sum(len(row["episodes"]) for row in records),
            "join_count": sum(row["join_count"] for row in records),
            "separated_change_count": sum(episode["separated_change_count"]
                                          for row in records for episode in row["episodes"]),
        },
        "records": records,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/identity_joins.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--record", type=Path, action="append", default=None,
                        help="固定済みの一致記録。複数指定できる")
    parser.add_argument("--minimum-similarity", type=float, default=0.7)
    parser.add_argument("--minimum-match-seconds", type=float, default=6.0)
    parser.add_argument("--maximum-join-seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if not args.record:
            parser.error("--recordが必要です")
        plan = build([path.resolve() for path in args.record], output,
                     minimum_similarity=args.minimum_similarity,
                     minimum_match_seconds=args.minimum_match_seconds,
                     maximum_join_seconds=args.maximum_join_seconds)
    print(f"一致からの境目の記述を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
