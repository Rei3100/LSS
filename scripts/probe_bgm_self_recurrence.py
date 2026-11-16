"""番組一本の中の再演から、参照の無い候補の範囲がどれだけ読めるかを測る。

新しい分離推論は行わない。固定済みの番組信号を読み直し、同じ録音が二度以上鳴る区間を
探して、候補ごとに「一曲だと言える」「まだ言えない」を数える。既知録音の一致だけで
読んだ場合と並べる。

意味的role・公開可否は扱わない。公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm import identity, self_recurrence  # noqa: E402
from lss.bgm.cue_extent import CueExtentRecipe, combine_readings, read_cue_extent  # noqa: E402
from lss.bgm.recurring_extent import read_recurring_extent, recurring_matches  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する。再演の探し方も範囲の読み方も、既存のStageの宣言をそのまま使う。
DECISION_RULE = {
    "repeats_are_found_by_fingerprint_and_waveform_not_by_novelty": True,
    "a_repeat_bounds_the_extent_it_does_not_name_the_cue": True,
    "no_repeat_is_not_evidence_of_a_single_occurrence": True,
}
LIMITS = [
    "reads_a_frozen_signal_and_frozen_candidates_no_new_separation_inference",
    "repeats_inside_one_index_chunk_collapse_so_immediate_repeats_can_be_missed",
    "a_repeat_shows_the_same_recording_it_does_not_show_the_semantic_role",
    "no_publication_default_change_or_holdout_use",
]


class SelfRecurrenceProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _known_matches_for_signal(record_path: Path, signal: Path) -> tuple[list[dict], dict]:
    """開発用の一致記録から、この信号に付いた既発売録音の一致を取り出す。"""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("kind") != "bgm_identity_development_probe":
        raise SelfRecurrenceProbeError(f"この一致記録は読めません: {record_path}")
    target = signal.resolve()
    for item in record["episodes"]:
        known = item.get("query") or item.get("source")
        if known is not None and Path(known).resolve() == target:
            return ([{"reference_id": row["reference_id"],
                      "query_start_seconds": row["query_start_seconds"],
                      "query_end_seconds": row["query_end_seconds"],
                      "offset_seconds": row["offset_seconds"],
                      "similarity": row["waveform_explained"] or 0.0}
                     for row in sorted(item["own"], key=lambda row: row["query_start_seconds"])],
                    {"path": str(record_path), "sha256": fetch.file_sha256(record_path)})
    raise SelfRecurrenceProbeError(f"この信号の一致記録がありません: {signal}")


def build(signal: Path, episode: Path | None, output: Path, *, source_id: str | None,
          known_matches: Path | None, chunk_seconds: float, minimum_separation_seconds: float,
          minimum_similarity: float, minimum_coverage: float,
          maximum_offset_spread_seconds: float) -> Path:
    recipe = CueExtentRecipe(minimum_similarity=minimum_similarity,
                             minimum_single_recording_coverage=minimum_coverage,
                             maximum_offset_spread_seconds=maximum_offset_spread_seconds)
    recipe.validate()

    candidates: list[dict] = []
    matches: list[dict] = []
    known: list[dict] = []
    known_record = None
    if episode is None:
        # 候補の記録が無い番組でも、再演があるかどうかだけは測れる。
        if not (isinstance(source_id, str) and source_id.strip()):
            raise SelfRecurrenceProbeError("候補を読まないときは--source-idが必要です")
        frozen = {"source_id": source_id.strip()}
        if known_matches is not None:
            known, known_record = _known_matches_for_signal(known_matches, signal)
    else:
        run_path, identity_path = episode / "run.json", episode / "identity.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        annotation = json.loads(identity_path.read_text(encoding="utf-8"))
        if (run.get("promotion_allowed") is not False
                or run.get("publication_state") != "not_published"):
            raise SelfRecurrenceProbeError("固定Development記録だけを読み直します")
        if annotation.get("type") != "episode_identity_annotation":
            raise SelfRecurrenceProbeError("この番組の一致記録ではありません")
        for item in sorted(run["candidates"], key=lambda row: row["origin"]["start_frame"]):
            asset, origin = item["asset"], item["origin"]
            if asset.get("publication_state") != "not_published":
                raise SelfRecurrenceProbeError("公開済みの候補は読み直しません")
            candidates.append({
                "candidate_asset_id": asset["candidate_asset_id"],
                "start_seconds": origin["start_frame"] / origin["sample_rate"],
                "end_seconds": origin["end_frame"] / origin["sample_rate"]})
        matches = [{"reference_id": row["reference_id"],
                    "query_start_seconds": row["query_start_seconds"],
                    "query_end_seconds": row["query_end_seconds"],
                    "offset_seconds": row["offset_seconds"],
                    "similarity": row["waveform_explained"] or 0.0}
                   for row in sorted(annotation["matches"],
                                     key=lambda row: row["query_start_seconds"])]
        known = matches
        known_record = {"path": str(identity_path), "sha256": fetch.file_sha256(identity_path)}
        frozen = {"run": {"path": str(run_path), "sha256": fetch.file_sha256(run_path)},
                  "identity": {"path": str(identity_path),
                               "sha256": fetch.file_sha256(identity_path)},
                  "source_id": run["source"]["source_id"]}
    output.mkdir(parents=True, exist_ok=False)

    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_self_recurrence_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "signal": {"path": str(signal), "sha256": fetch.file_sha256(signal)},
        "episode": frozen,
        "repeat_recipe": {"method": self_recurrence.METHOD,
                          "schema_version": self_recurrence.SCHEMA_VERSION,
                          "chunk_seconds": chunk_seconds,
                          "minimum_separation_seconds": minimum_separation_seconds},
        "extent_recipe": {"recipe_id": recipe.recipe_id, "revision": recipe.to_dict()},
        "candidates": candidates, "matches": matches,
        "known_matches": known, "known_matches_record": known_record,
        "decision_rule": DECISION_RULE, "limits": LIMITS,
    })
    return plan


def _references_at(matches: list[dict], start: float, end: float) -> set[str]:
    return {row["reference_id"] for row in matches
            if row["query_start_seconds"] < end and start < row["query_end_seconds"]}


def _confirmed_by_released_matches(occurrences: list, matches: list[dict]) -> list[dict]:
    """見つけた再演の両側が、同じ既発売録音と一致しているかを数える。

    一致が付く再演だけが正しいという意味ではない。既知の一致は疎なので、
    確かめられるのは一部だけである。
    """
    rows = []
    for item in occurrences:
        first = _references_at(matches, item.first_start_seconds, item.first_end_seconds)
        second = _references_at(matches, item.second_start_seconds, item.second_end_seconds)
        shared = sorted(first & second)
        rows.append({"first_start_seconds": item.first_start_seconds,
                     "second_start_seconds": item.second_start_seconds,
                     "shared_reference_ids": shared,
                     "confirmed": bool(shared),
                     "contradicted": bool(first and second and not shared)})
    return rows


def _verify_inputs(plan: dict) -> None:
    artifacts = [plan["signal"]]
    artifacts.extend(plan["episode"][key] for key in ("run", "identity")
                     if key in plan["episode"])
    for artifact in artifacts:
        if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise SelfRecurrenceProbeError(f"固定入力が変更されています: {artifact['path']}")


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_self_recurrence_plan"):
        raise SelfRecurrenceProbeError("固定した再演読み取りの計画だけを実行します")
    _verify_inputs(plan)
    recipe = CueExtentRecipe(**plan["extent_recipe"]["revision"])
    repeat = plan["repeat_recipe"]

    signal = identity.decode_mono(Path(plan["signal"]["path"]))
    print(f"信号を読み込みました: {signal.size / identity.SAMPLE_RATE:.1f}秒", flush=True)
    occurrences = self_recurrence.repeated_recordings(
        signal, chunk_seconds=repeat["chunk_seconds"],
        minimum_separation_seconds=repeat["minimum_separation_seconds"])
    print(f"再演の組: {len(occurrences)}件", flush=True)
    confirmed = _confirmed_by_released_matches(occurrences, plan["known_matches"])
    spans = self_recurrence.spans_for_source(occurrences, source_key=plan["episode"]["source_id"])
    matches = recurring_matches(spans)

    rows, counts = [], {}
    for candidate in plan["candidates"]:
        start, end = candidate["start_seconds"], candidate["end_seconds"]
        inside = [row for row in matches
                  if row["query_start_seconds"] < end and start < row["query_end_seconds"]]
        released = read_cue_extent(start, end, [
            row for row in plan["matches"]
            if row["query_start_seconds"] < end and start < row["query_end_seconds"]], recipe)
        repeated = read_recurring_extent(start, end, inside, recipe)
        combined = combine_readings([released, repeated])
        for name, reading in (("released", released), ("repeated", repeated),
                              ("combined", combined)):
            counts.setdefault(name, {})
            counts[name][reading["state"]] = counts[name].get(reading["state"], 0) + 1
        rows.append({
            "candidate_asset_id": candidate["candidate_asset_id"],
            "start_seconds": round(start, 3), "end_seconds": round(end, 3),
            "released_state": released["state"],
            "released_reason": released["evidence"].get("reason"),
            "repeated_state": repeated["state"],
            "repeated_reason": repeated["evidence"].get("reason"),
            "repeat_match_count": len(inside),
            "repeat_coverage": (round(repeated["evidence"]["single_recording_coverage"], 4)
                                if "single_recording_coverage" in repeated["evidence"] else None),
            "combined_state": combined["state"]})

    _verify_inputs(plan)
    gained = [row for row in rows if row["released_state"] == "unresolved"
              and row["repeated_state"] != "unresolved"]
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_self_recurrence_report",
        "status": "complete_real_episode_description",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False, "anything_published": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "source_id": plan["episode"]["source_id"],
        "repeat_recipe": plan["repeat_recipe"], "extent_recipe": plan["extent_recipe"],
        "decision_rule": plan["decision_rule"],
        "summary": {
            "candidate_count": len(rows),
            "repeat_pair_count": len(occurrences),
            "repeated_recording_count": len({row["reference_id"] for row in matches}),
            "decided_only_by_repeats": len(gained),
            "known_match_count": len(plan["known_matches"]),
            "repeats_confirmed_by_a_shared_released_recording":
                sum(1 for row in confirmed if row["confirmed"]),
            "repeats_contradicted_by_released_recordings":
                sum(1 for row in confirmed if row["contradicted"]),
            "states": counts},
        "released_confirmation": confirmed,
        "known_matches_record": plan["known_matches_record"],
        "repeats": [{"first_start_seconds": item.first_start_seconds,
                     "first_end_seconds": item.first_end_seconds,
                     "second_start_seconds": item.second_start_seconds,
                     "second_end_seconds": item.second_end_seconds,
                     "waveform_explained": item.waveform_explained,
                     "blocks": item.blocks} for item in occurrences],
        "candidates": rows,
        "decided_only_by_repeats": gained,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/self_recurrence.py",
            ROOT / "lss/bgm/recurring_extent.py", ROOT / "lss/bgm/identity.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--signal", type=Path, help="固定済みの番組信号")
    parser.add_argument("--episode", type=Path, help="固定済みの実番組候補生成のdirectory")
    parser.add_argument("--source-id", help="候補を読まないときの番組の識別子")
    parser.add_argument("--known-matches", type=Path,
                        help="既発売録音の一致を照合に使う開発記録")
    parser.add_argument("--chunk-seconds", type=float,
                        default=identity.SHARED_CHUNK_SECONDS)
    parser.add_argument("--minimum-separation-seconds", type=float,
                        default=self_recurrence.MINIMUM_SEPARATION_SECONDS)
    parser.add_argument("--minimum-similarity", type=float, default=0.15)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--maximum-offset-spread-seconds", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.signal is None:
            parser.error("--signalが必要です")
        plan = build(args.signal.resolve(),
                     args.episode.resolve() if args.episode is not None else None, output,
                     source_id=args.source_id,
                     known_matches=(args.known_matches.resolve()
                                    if args.known_matches is not None else None),
                     chunk_seconds=args.chunk_seconds,
                     minimum_separation_seconds=args.minimum_separation_seconds,
                     minimum_similarity=args.minimum_similarity,
                     minimum_coverage=args.minimum_coverage,
                     maximum_offset_spread_seconds=args.maximum_offset_spread_seconds)
    print(f"番組の中の再演の記述を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
