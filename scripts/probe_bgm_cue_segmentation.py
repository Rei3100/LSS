"""曲の変わり目を信号だけで見つけられるかを、正解のある連結で固定比較する。

既存の固定componentから、継ぎ目の位置が作り方から分かる信号を組み立てる。分離は
行わない。測るのは「別の曲が並んだ位置を、同じ曲が続く場合と区別できるか」だけで、
意味的role・同一性・公開可否は扱わない。

同じ録音の離れた箇所をつないだ場合は、正解が一つに定まらないので判定へ入れず記述だけ
残す。公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from lss.bgm.cue_segmentation import CueSegmentationRecipe, segment_cues  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する合否条件。
DECISION_RULE = {
    "every_join_is_the_strongest_boundary_within_tolerance": True,
    "control_maximum_novelty_below_every_join_novelty": True,
    "control_boundaries_above_threshold_at_most": 1,
    "same_recording_join_is_described_not_judged": True,
}


class CueSegmentationProbeError(RuntimeError):
    """比較の入力または組み立てが契約に合わない。"""


def _active_span(path: Path) -> tuple[int, int, int]:
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    active = np.any(values != 0.0, axis=1).nonzero()[0]
    if active.size == 0:
        raise CueSegmentationProbeError(f"音のある区間がありません: {path}")
    return int(active[0]), int(active[-1]) + 1, int(rate)


def build(recordings: list[str], recipe_path: Path, output: Path, *,
          segment_seconds: float, margin_seconds: float, tolerance_seconds: float) -> Path:
    parsed = []
    for item in recordings:
        label, _, raw = item.partition("=")
        if not label.strip() or not raw.strip():
            raise CueSegmentationProbeError(f"録音の指定が不正です: {item}")
        path = Path(raw).resolve()
        if not path.is_file():
            raise CueSegmentationProbeError(f"録音がありません: {path}")
        start, end, rate = _active_span(path)
        parsed.append({"label": label.strip(), "path": str(path),
                       "sha256": fetch.file_sha256(path), "sample_rate": rate,
                       "active_start_frame": start, "active_end_frame": end})
    if len(parsed) < 2:
        raise CueSegmentationProbeError("比較には2つ以上の録音が必要です")
    rates = {item["sample_rate"] for item in parsed}
    if len(rates) != 1:
        raise CueSegmentationProbeError(f"sample rateが一致しません: {sorted(rates)}")
    rate = rates.pop()
    segment = int(round(segment_seconds * rate))
    margin = int(round(margin_seconds * rate))
    for item in parsed:
        usable = item["active_end_frame"] - item["active_start_frame"] - 2 * margin
        if usable < 2 * segment:
            raise CueSegmentationProbeError(
                f"{item['label']}は{2 * segment_seconds}秒の連結に足りません")

    recipe = CueSegmentationRecipe(**json.loads(recipe_path.read_text(encoding="utf-8")))
    recipe.validate()
    output.mkdir(parents=True, exist_ok=False)
    frozen_recipe = output / "recipe.json"
    shutil.copyfile(recipe_path, frozen_recipe)

    cases = []
    for first in parsed:
        for second in parsed:
            if first["label"] == second["label"]:
                continue
            cases.append({
                "case_id": f"join-{first['label']}-to-{second['label']}",
                "family": "different_recording_join",
                "judged": True,
                "join_seconds": segment_seconds,
                "parts": [
                    {"label": first["label"],
                     "start_frame": first["active_end_frame"] - margin - segment,
                     "end_frame": first["active_end_frame"] - margin},
                    {"label": second["label"],
                     "start_frame": second["active_start_frame"] + margin,
                     "end_frame": second["active_start_frame"] + margin + segment},
                ],
            })
    for item in parsed:
        start = item["active_start_frame"] + margin
        cases.append({
            "case_id": f"control-{item['label']}",
            "family": "same_recording_continuous",
            "judged": True, "join_seconds": None,
            "parts": [{"label": item["label"], "start_frame": start,
                       "end_frame": start + 2 * segment}],
        })
    for item in parsed:
        start = item["active_start_frame"] + margin
        usable_end = item["active_end_frame"] - margin
        cases.append({
            "case_id": f"same-recording-jump-{item['label']}",
            "family": "same_recording_discontinuous_join",
            # 同じ曲の離れた箇所をつないだ位置は、切るべきか続けるべきか一つに決まらない。
            "judged": False, "join_seconds": segment_seconds,
            "parts": [
                {"label": item["label"], "start_frame": start, "end_frame": start + segment},
                {"label": item["label"], "start_frame": usable_end - segment,
                 "end_frame": usable_end},
            ],
        })

    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_cue_segmentation_comparison_plan",
        "role": "development_controlled_diagnostic_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "sample_rate": rate,
        "segment_seconds": segment_seconds, "margin_seconds": margin_seconds,
        "tolerance_seconds": tolerance_seconds,
        "join_style": "hard_cut_without_crossfade",
        "recordings": parsed,
        "recipe": {"path": str(frozen_recipe), "sha256": fetch.file_sha256(frozen_recipe),
                   "recipe_id": recipe.recipe_id, "revision": recipe.to_dict()},
        "cases": cases,
        "decision_rule": DECISION_RULE,
        "limits": [
            "three_or_few_distinct_recordings_not_a_corpus",
            "constructed_hard_cut_not_a_real_programme_transition",
            "clean_music_components_without_dialogue_or_separation_error",
            "novelty_is_an_observation_not_a_semantic_cue_boundary",
            "no_publication_default_change_or_holdout_use",
        ],
    })
    return plan


def _verify_inputs(plan: dict) -> None:
    for item in plan["recordings"] + [plan["recipe"]]:
        if fetch.file_sha256(Path(item["path"])) != item["sha256"]:
            raise CueSegmentationProbeError(f"固定入力が変更されています: {item['path']}")


def _signal(case: dict, sources: dict[str, Path]) -> np.ndarray:
    blocks = []
    for part in case["parts"]:
        values, _ = sf.read(sources[part["label"]], dtype="float64", always_2d=True,
                            start=part["start_frame"], stop=part["end_frame"])
        blocks.append(values)
    if len({block.shape[1] for block in blocks}) != 1:
        raise CueSegmentationProbeError(f"連結するchannel数が一致しません: {case['case_id']}")
    return np.concatenate(blocks, axis=0).T


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_cue_segmentation_comparison_plan"):
        raise CueSegmentationProbeError("固定した区切り比較の計画だけを実行します")
    _verify_inputs(plan)
    output = plan_path.parent
    sources = {item["label"]: Path(item["path"]) for item in plan["recordings"]}
    recipe = CueSegmentationRecipe(**plan["recipe"]["revision"])
    rate, tolerance = plan["sample_rate"], plan["tolerance_seconds"]

    records = []
    for case in plan["cases"]:
        result = segment_cues(_signal(case, sources), rate, recipe)
        if result.recipe_id != plan["recipe"]["recipe_id"]:
            raise CueSegmentationProbeError("Recipe identityが固定記録と一致しません")
        boundaries = [{"seconds": round(item.seconds, 3), "novelty": item.novelty}
                      for item in result.boundaries]
        strongest = max(boundaries, key=lambda item: item["novelty"], default=None)
        join = case["join_seconds"]
        near = [item for item in boundaries
                if join is not None and abs(item["seconds"] - join) <= tolerance]
        records.append({
            "case_id": case["case_id"], "family": case["family"], "judged": case["judged"],
            "join_seconds": join, "boundary_count": len(boundaries), "boundaries": boundaries,
            "maximum_novelty": max(result.novelty),
            "strongest_boundary": strongest,
            "strongest_is_at_the_join": bool(
                strongest is not None and join is not None
                and abs(strongest["seconds"] - join) <= tolerance),
            "novelty_at_the_join": max((item["novelty"] for item in near), default=None),
        })
        print(f"区切り比較完了: {case['case_id']} / 候補{len(boundaries)}件", flush=True)

    _verify_inputs(plan)
    joins = [row for row in records if row["family"] == "different_recording_join"]
    controls = [row for row in records if row["family"] == "same_recording_continuous"]
    failures = []
    for row in joins:
        if not row["strongest_is_at_the_join"]:
            failures.append(f"{row['case_id']}:最も強い候補が継ぎ目にない")
    join_novelties = [row["novelty_at_the_join"] for row in joins
                      if row["novelty_at_the_join"] is not None]
    control_maximum = max((row["maximum_novelty"] for row in controls), default=0.0)
    if len(join_novelties) != len(joins):
        failures.append("継ぎ目に候補が出ないcaseがある")
    elif join_novelties and control_maximum >= min(join_novelties):
        failures.append(
            f"同じ曲が続くcaseの最大novelty {control_maximum:.4f}が、"
            f"継ぎ目の最小novelty {min(join_novelties):.4f}以上になっている")
    for row in controls:
        if row["boundary_count"] > plan["decision_rule"]["control_boundaries_above_threshold_at_most"]:
            failures.append(f"{row['case_id']}:同じ曲が続くのに候補が{row['boundary_count']}件出た")

    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_cue_segmentation_comparison_report",
        "status": "complete_development_diagnostic",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "recipe_id": plan["recipe"]["recipe_id"],
        "decision_rule": plan["decision_rule"],
        "decision": {"passed_declared_rule": not failures, "failures": failures,
                     "join_case_count": len(joins),
                     "minimum_join_novelty": min(join_novelties) if join_novelties else None,
                     "maximum_control_novelty": control_maximum},
        "cases": records,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/cue_segmentation.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="比較記録を作る新規directory")
    parser.add_argument("--recording", action="append", default=[],
                        help="label=path の形で、別々の録音を渡す")
    parser.add_argument("--recipe", type=Path, help="先に宣言した区切り候補Recipe")
    parser.add_argument("--segment-seconds", type=float, default=12.0)
    parser.add_argument("--margin-seconds", type=float, default=1.0)
    parser.add_argument("--tolerance-seconds", type=float, default=1.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.recipe is None or len(args.recording) < 2:
            parser.error("--recipeと2つ以上の--recordingが必要です")
        plan = build(args.recording, args.recipe.resolve(), output,
                     segment_seconds=args.segment_seconds, margin_seconds=args.margin_seconds,
                     tolerance_seconds=args.tolerance_seconds)
    print(f"区切り比較を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
