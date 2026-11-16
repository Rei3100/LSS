"""実番組の再演どうしを突き合わせ、読み取りが独立した観測と一致するかを測る。

新しい分離推論も照合も行わない。固定した番組信号と、保存済みの再演・既発売録音の一致を
読み直す。両方の登場が同じ既発売録音と一致している再演だけを取り出し、
「余計な音が少ない方」の読み取りが、記録済みの波形説明率の大小と一致するかを数える。

波形説明率は参照録音がある区間にしか付かない独立した観測であり、正解ではない。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm import identity  # noqa: E402
from lss.bgm.occurrence_contrast import OccurrenceContrastRecipe, compare  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する。読み取りは記述であり、良し悪しの判定はしない。
DECISION_RULE = {
    "only_repeats_whose_both_sides_match_the_same_released_recording_are_checked": True,
    "waveform_explained_is_an_independent_observation_not_a_ground_truth": True,
    "a_small_difference_is_left_unranked": True,
}
LIMITS = [
    "reads_a_frozen_signal_and_frozen_records_only_no_new_inference_or_matching",
    "the_comparison_runs_on_the_11025hz_mono_matching_signal_not_the_published_audio",
    "production_ducking_and_edits_also_look_like_excess",
    "released_matches_are_sparse_so_only_a_few_repeats_can_be_checked",
    "no_publication_default_change_or_holdout_use",
]


class OccurrenceContrastProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _covering(matches: list[dict], start: float, end: float) -> dict[str, float]:
    """区間を覆う既発売録音ごとの、記録済みの波形説明率。"""
    found: dict[str, float] = {}
    for row in matches:
        if row["query_start_seconds"] <= start + 1.0 and row["query_end_seconds"] >= end - 1.0:
            value = row.get("waveform_explained")
            if value is None:
                continue
            found[row["reference_id"]] = max(found.get(row["reference_id"], 0.0), float(value))
    return found


def build(signal: Path, repeats: Path, known: Path, output: Path, *,
          window_frames: int, hop_frames: int, minimum_seconds: float,
          minimum_excess_share_margin: float) -> Path:
    report = json.loads(repeats.read_text(encoding="utf-8"))
    if report.get("type") != "bgm_self_recurrence_report":
        raise OccurrenceContrastProbeError(f"この再演の記述は読めません: {repeats}")
    record = json.loads(known.read_text(encoding="utf-8"))
    if record.get("kind") != "bgm_identity_development_probe":
        raise OccurrenceContrastProbeError(f"この一致記録は読めません: {known}")
    target = signal.resolve()
    matches = None
    for item in record["episodes"]:
        path = item.get("query") or item.get("source")
        if path is not None and Path(path).resolve() == target:
            matches = [{"reference_id": row["reference_id"],
                        "query_start_seconds": row["query_start_seconds"],
                        "query_end_seconds": row["query_end_seconds"],
                        "waveform_explained": row.get("waveform_explained")}
                       for row in item["own"]]
            break
    if matches is None:
        raise OccurrenceContrastProbeError(f"この信号の一致記録がありません: {signal}")

    recipe = OccurrenceContrastRecipe(
        window_frames=window_frames, hop_frames=hop_frames,
        minimum_seconds=minimum_seconds,
        minimum_excess_share_margin=minimum_excess_share_margin)
    recipe.validate()
    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_occurrence_contrast_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "signal": {"path": str(signal), "sha256": fetch.file_sha256(signal)},
        "repeats": {"path": str(repeats), "sha256": fetch.file_sha256(repeats),
                    "source_id": report["source_id"], "pairs": report["repeats"]},
        "known_matches": {"path": str(known), "sha256": fetch.file_sha256(known),
                          "matches": matches},
        "recipe": {"recipe_id": recipe.recipe_id, "revision": recipe.to_dict()},
        "decision_rule": DECISION_RULE, "limits": LIMITS,
    })
    return plan


def _verify_inputs(plan: dict) -> None:
    for artifact in (plan["signal"], plan["repeats"], plan["known_matches"]):
        if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise OccurrenceContrastProbeError(f"固定入力が変更されています: {artifact['path']}")


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_occurrence_contrast_plan"):
        raise OccurrenceContrastProbeError("固定した突き合わせの計画だけを実行します")
    _verify_inputs(plan)
    recipe = OccurrenceContrastRecipe(**plan["recipe"]["revision"])
    matches = plan["known_matches"]["matches"]
    signal = identity.decode_mono(Path(plan["signal"]["path"]))
    rate = identity.SAMPLE_RATE

    rows, checked, agreed, unranked, tied = [], 0, 0, 0, 0
    for pair in plan["repeats"]["pairs"]:
        first = _covering(matches, pair["first_start_seconds"], pair["first_end_seconds"])
        second = _covering(matches, pair["second_start_seconds"], pair["second_end_seconds"])
        common = sorted(set(first) & set(second))
        row = {"first_start_seconds": pair["first_start_seconds"],
               "second_start_seconds": pair["second_start_seconds"],
               "seconds": round(pair["first_end_seconds"] - pair["first_start_seconds"], 3),
               "shared_reference_ids": common}
        if common:
            reference = max(common, key=lambda name: min(first[name], second[name]))
            row["reference_id"] = reference
            row["waveform_explained"] = [first[reference], second[reference]]
        cut = []
        for start, end in ((pair["first_start_seconds"], pair["first_end_seconds"]),
                           (pair["second_start_seconds"], pair["second_end_seconds"])):
            cut.append(signal[int(round(start * rate)):int(round(end * rate))])
        try:
            read = compare(cut[0], cut[1], rate, recipe)
        except ValueError as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        row.update({key: read[key] for key in (
            "state", "reason", "compared_seconds", "level_gain", "excess_energy_dbfs",
            "excess_share", "common_energy_dbfs", "excess_share_difference")})
        if common:
            checked += 1
            explained = row["waveform_explained"]
            if read["state"] == "unresolved":
                unranked += 1
                row["agrees"] = None
            elif explained[0] == explained[1]:
                # 独立した観測が同じ値なら、どちらが良いかを言っていない。
                tied += 1
                row["agrees"] = None
            else:
                better = 0 if read["state"] == "first_has_less_excess" else 1
                row["agrees"] = explained[better] > explained[1 - better]
                agreed += int(row["agrees"])
        rows.append(row)
        print(f"{pair['first_start_seconds']:.1f}秒 <-> {pair['second_start_seconds']:.1f}秒: "
              f"{row.get('state')}", flush=True)

    _verify_inputs(plan)
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_occurrence_contrast_report",
        "status": "complete_real_episode_description",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False, "anything_published": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "source_id": plan["repeats"]["source_id"],
        "recipe": plan["recipe"], "decision_rule": plan["decision_rule"],
        "summary": {
            "pair_count": len(rows),
            "pairs_with_a_shared_released_recording": checked,
            "ranked_and_agreed": agreed,
            "left_unranked": unranked,
            "tied_on_the_independent_observation": tied,
            "ranked_and_disagreed": checked - agreed - unranked - tied},
        "pairs": rows,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/occurrence_contrast.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--signal", type=Path, help="固定済みの番組信号")
    parser.add_argument("--repeats", type=Path, help="保存済みの再演の記述")
    parser.add_argument("--known-matches", type=Path, help="既発売録音の一致の開発記録")
    parser.add_argument("--window-frames", type=int, default=1024)
    parser.add_argument("--hop-frames", type=int, default=256)
    parser.add_argument("--minimum-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-excess-share-margin", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.signal is None or args.repeats is None or args.known_matches is None:
            parser.error("--signalと--repeatsと--known-matchesが必要です")
        plan = build(args.signal.resolve(), args.repeats.resolve(),
                     args.known_matches.resolve(), output,
                     window_frames=args.window_frames, hop_frames=args.hop_frames,
                     minimum_seconds=args.minimum_seconds,
                     minimum_excess_share_margin=args.minimum_excess_share_margin)
    print(f"使用箇所の突き合わせを保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
