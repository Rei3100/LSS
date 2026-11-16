"""番組の中の再演を、既発売録音の一致から裏付けられる範囲でどれだけ拾えているか測る。

新しい推論も照合も行わない。固定した記録だけを読み直す。同じ既発売録音の同じ部分が
離れた二つの時刻で鳴っているなら、それは信号の外側から裏付けられた再演である。
その組を、参照を使わない再演探しが拾えているかを数える。

既発売録音の一致は疎なので、ここで数えられるのは再演の一部だけである。拾えなかった組は
「その再演が無い」ことではなく「この探し方では見つからない」ことを示す。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する。
DECISION_RULE = {
    "an_evidenced_repeat_needs_the_same_part_of_the_same_released_recording": True,
    "the_two_times_must_be_separated_by_the_declared_gap": True,
    "a_missed_repeat_is_not_evidence_that_the_repeat_is_absent": True,
}
LIMITS = [
    "reads_frozen_records_only_no_new_inference_or_matching",
    "released_matches_are_sparse_so_this_covers_only_part_of_the_repeats",
    "unreleased_cues_cannot_appear_in_this_count_at_all",
    "no_publication_default_change_or_holdout_use",
]


class RepeatRecallProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _overlap_share(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = min(first[1], second[1]) - max(first[0], second[0])
    shorter = min(first[1] - first[0], second[1] - second[0])
    return overlap / shorter if shorter > 0 else 0.0


def _evidenced(matches: list[dict], *, minimum_gap_seconds: float,
               minimum_reference_overlap: float) -> list[dict]:
    """同じ録音の同じ部分が、離れた二つの時刻で鳴っている組。"""
    rows = sorted(matches, key=lambda row: row["query_start_seconds"])
    found = []
    for index, first in enumerate(rows):
        for second in rows[index + 1:]:
            if first["reference_id"] != second["reference_id"]:
                continue
            if second["query_start_seconds"] - first["query_end_seconds"] < minimum_gap_seconds:
                continue
            spans = []
            for row in (first, second):
                start = row["reference_start_seconds"]
                spans.append((start, start + row["query_end_seconds"] - row["query_start_seconds"]))
            share = _overlap_share(spans[0], spans[1])
            if share < minimum_reference_overlap:
                continue
            found.append({
                "reference_id": first["reference_id"],
                "first_start_seconds": first["query_start_seconds"],
                "first_end_seconds": first["query_end_seconds"],
                "second_start_seconds": second["query_start_seconds"],
                "second_end_seconds": second["query_end_seconds"],
                "reference_overlap_share": round(share, 4)})
    return found


def build(repeats: Path, known: Path, output: Path, *, minimum_gap_seconds: float,
          minimum_reference_overlap: float, minimum_found_overlap: float) -> Path:
    report = json.loads(repeats.read_text(encoding="utf-8"))
    if report.get("type") != "bgm_self_recurrence_report":
        raise RepeatRecallProbeError(f"この再演の記述は読めません: {repeats}")
    record = json.loads(known.read_text(encoding="utf-8"))
    if record.get("kind") != "bgm_identity_development_probe":
        raise RepeatRecallProbeError(f"この一致記録は読めません: {known}")
    signal = Path(json.loads(
        (repeats.parent / "plan.json").read_text(encoding="utf-8"))["signal"]["path"]).resolve()
    matches = None
    for item in record["episodes"]:
        path = item.get("query") or item.get("source")
        if path is not None and Path(path).resolve() == signal:
            matches = [{"reference_id": row["reference_id"],
                        "query_start_seconds": row["query_start_seconds"],
                        "query_end_seconds": row["query_end_seconds"],
                        "reference_start_seconds": row["reference_start_seconds"]}
                       for row in item["own"]]
            break
    if matches is None:
        raise RepeatRecallProbeError(f"この信号の一致記録がありません: {signal}")

    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_repeat_recall_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "signal": str(signal),
        "repeats": {"path": str(repeats), "sha256": fetch.file_sha256(repeats),
                    "source_id": report["source_id"], "pairs": report["repeats"]},
        "known_matches": {"path": str(known), "sha256": fetch.file_sha256(known),
                          "matches": matches},
        "thresholds": {"minimum_gap_seconds": minimum_gap_seconds,
                       "minimum_reference_overlap": minimum_reference_overlap,
                       "minimum_found_overlap": minimum_found_overlap},
        "decision_rule": DECISION_RULE, "limits": LIMITS,
    })
    return plan


def _verify_inputs(plan: dict) -> None:
    for artifact in (plan["repeats"], plan["known_matches"]):
        if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise RepeatRecallProbeError(f"固定入力が変更されています: {artifact['path']}")


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_repeat_recall_plan"):
        raise RepeatRecallProbeError("固定した再現率の計画だけを実行します")
    _verify_inputs(plan)
    thresholds = plan["thresholds"]
    evidenced = _evidenced(plan["known_matches"]["matches"],
                           minimum_gap_seconds=thresholds["minimum_gap_seconds"],
                           minimum_reference_overlap=thresholds["minimum_reference_overlap"])
    pairs = plan["repeats"]["pairs"]

    rows = []
    for item in evidenced:
        hit = None
        for pair in pairs:
            forward = min(
                _overlap_share((item["first_start_seconds"], item["first_end_seconds"]),
                               (pair["first_start_seconds"], pair["first_end_seconds"])),
                _overlap_share((item["second_start_seconds"], item["second_end_seconds"]),
                               (pair["second_start_seconds"], pair["second_end_seconds"])))
            backward = min(
                _overlap_share((item["first_start_seconds"], item["first_end_seconds"]),
                               (pair["second_start_seconds"], pair["second_end_seconds"])),
                _overlap_share((item["second_start_seconds"], item["second_end_seconds"]),
                               (pair["first_start_seconds"], pair["first_end_seconds"])))
            share = max(forward, backward)
            if share >= thresholds["minimum_found_overlap"] and (hit is None or share > hit[1]):
                hit = (pair, share)
        rows.append({**item, "found": hit is not None,
                     "found_overlap_share": None if hit is None else round(hit[1], 4),
                     "found_pair": None if hit is None else
                     {"first_start_seconds": hit[0]["first_start_seconds"],
                      "second_start_seconds": hit[0]["second_start_seconds"]}})

    _verify_inputs(plan)
    found = sum(1 for row in rows if row["found"])
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_repeat_recall_report",
        "status": "complete_real_episode_description",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False, "anything_published": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "source_id": plan["repeats"]["source_id"],
        "thresholds": thresholds, "decision_rule": plan["decision_rule"],
        "summary": {
            "evidenced_repeat_count": len(rows),
            "found_by_the_reference_free_search": found,
            "missed": len(rows) - found,
            "reference_free_pair_count": len(pairs)},
        "evidenced_repeats": rows,
        "limits": plan["limits"],
        "implementation": [{"path": str(Path(__file__)),
                            "sha256": fetch.file_sha256(Path(__file__))}],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--repeats", type=Path, help="保存済みの再演の記述")
    parser.add_argument("--known-matches", type=Path, help="既発売録音の一致の開発記録")
    parser.add_argument("--minimum-gap-seconds", type=float, default=20.0)
    parser.add_argument("--minimum-reference-overlap", type=float, default=0.5)
    parser.add_argument("--minimum-found-overlap", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.repeats is None or args.known_matches is None:
            parser.error("--repeatsと--known-matchesが必要です")
        plan = build(args.repeats.resolve(), args.known_matches.resolve(), output,
                     minimum_gap_seconds=args.minimum_gap_seconds,
                     minimum_reference_overlap=args.minimum_reference_overlap,
                     minimum_found_overlap=args.minimum_found_overlap)
    print(f"再演の取りこぼしの記述を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
