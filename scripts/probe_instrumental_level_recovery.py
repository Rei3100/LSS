"""取りこぼした大きさを戻す処理が、固定した伴奏ケースで利益と損傷をどう動かすか測る。

新しい分離推論は行わない。固定診断の入力と出力をそのまま読み直し、原本と推定だけから
戻し量を読んで適用し、同じ指標で測り直す。参照は測定にしか使わず、戻し量の決定には
使わない。

規則は結果を見る前に固定する。改善は「正解との距離が縮み、取りこぼしたレベルが0へ寄る」
こと、損傷は「正解との距離が広がる」こと。平均で最悪例を隠さず、ケースごとに並べる。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402
from lss.separation.level_recovery import (  # noqa: E402
    LevelRecoveryRecipe,
    apply_reading,
    read_missing_level,
)
from lss.validation.instrumental_quality import instrumental_quality  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する。上限は固定診断で測られたレベル低下（最大1.54dB）より広く取り、
# 戻しすぎが起きるかどうかも見えるようにする。
RECIPE = LevelRecoveryRecipe(minimum_estimate_dbfs=-60.0, maximum_gain_db=6.0,
                             minimum_gain_db=0.05)
DECISION_RULE = {
    "the_recovery_uses_only_the_mixture_and_the_estimate": True,
    "the_reference_is_used_for_measurement_only": True,
    "improvement_is_a_smaller_distance_to_the_truth_and_a_gain_error_closer_to_zero": True,
    "damage_is_a_larger_distance_to_the_truth_in_any_case": True,
}
LIMITS = [
    "reads_frozen_inputs_and_frozen_outputs_no_new_separation_inference",
    "public_training_material_and_constructed_linear_mixtures_only",
    "twelve_cases_are_not_a_promotion_corpus",
    "singing_and_accompaniment_are_not_orthogonal_so_the_share_can_overshoot",
    "no_publication_default_change_or_holdout_use",
]


class LevelRecoveryProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _read(path: Path) -> tuple[np.ndarray, int]:
    values, rate = sf.read(str(path), dtype="float64", always_2d=True)
    return values.T, int(rate)


def run_probe(plan_path: Path, report_path: Path, output: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if plan.get("promotion_allowed") is not False or report.get("promotion_allowed") is not False:
        raise LevelRecoveryProbeError("固定したDevelopment診断だけを読み直します")
    if report.get("plan_sha256") != fetch.file_sha256(plan_path):
        raise LevelRecoveryProbeError("この固定計画の結果ではありません")
    before = {path: fetch.file_sha256(path) for path in (plan_path, report_path)}
    cases = {case["case_id"]: case for case in plan["cases"]}
    output.mkdir(parents=True, exist_ok=False)

    rows, improved, damaged, untouched = [], 0, 0, 0
    for item in report["cases"]:
        case = cases[item["case_id"]]
        artifacts = case["artifacts"]
        for name in ("input", "target", "vocals"):
            path = Path(artifacts[name]["path"])
            if fetch.file_sha256(path) != artifacts[name]["sha256"]:
                raise LevelRecoveryProbeError(f"固定入力が変更されています: {path}")
        [produced] = item["result"]["outputs"]
        if fetch.file_sha256(Path(produced)) != item["output_sha256"]:
            raise LevelRecoveryProbeError(f"固定出力が変更されています: {produced}")

        mixture, rate = _read(Path(artifacts["input"]["path"]))
        target, _ = _read(Path(artifacts["target"]["path"]))
        vocals, _ = _read(Path(artifacts["vocals"]["path"]))
        stored, _ = _read(Path(produced))
        estimate = stored * 2.0 ** item["serialization_shift"]
        frames = min(mixture.shape[1], estimate.shape[1], target.shape[1], vocals.shape[1])
        mixture, estimate = mixture[:, :frames], estimate[:, :frames]
        target, vocals = target[:, :frames], vocals[:, :frames]

        reading = read_missing_level(mixture, estimate, RECIPE)
        recovered = apply_reading(estimate, reading)
        quality = {"before": instrumental_quality(estimate, target, vocals),
                   "after": instrumental_quality(recovered, target, vocals)}

        def distance(value: np.ndarray) -> float:
            return float(np.sum((value - target) ** 2))

        row = {"case_id": item["case_id"], "variant": item["variant"],
               "reading": reading,
               "distance_before": distance(estimate), "distance_after": distance(recovered),
               "si_sdr_before": (quality["before"]["fidelity"].get("si_sdr_db") or {}).get("joint"),
               "si_sdr_after": (quality["after"]["fidelity"].get("si_sdr_db") or {}).get("joint"),
               "gain_error_before":
                   (quality["before"]["fidelity"].get("projection_gain_db") or {}).get("joint"),
               "gain_error_after":
                   (quality["after"]["fidelity"].get("projection_gain_db") or {}).get("joint"),
               "target_absent_before": quality["before"].get("target_absent_output_to_vocals_db"),
               "target_absent_after": quality["after"].get("target_absent_output_to_vocals_db")}
        if reading["state"] == "unchanged":
            untouched += 1
        elif row["distance_after"] < row["distance_before"]:
            improved += 1
        else:
            damaged += 1
        rows.append(row)
        print(f"{item['case_id']}: {reading['state']} "
              f"gain={reading['gain_db']:.2f}dB "
              f"si {row['si_sdr_before']} -> {row['si_sdr_after']}", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise LevelRecoveryProbeError(f"固定入力が変更されています: {path}")

    result = output / "report.json"
    session._write_json_atomic(result, {
        "schema_version": SCHEMA_VERSION,
        "type": "instrumental_level_recovery_report",
        "status": "complete_development_comparison",
        "role": "development_comparison_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "recipe": RECIPE.to_dict(), "recipe_id": RECIPE.recipe_id,
        "decision_rule": DECISION_RULE,
        "case_count": len(rows),
        "improved": improved, "damaged": damaged, "untouched": untouched,
        "cases": rows,
        "limits": LIMITS,
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/separation/level_recovery.py",
            ROOT / "lss/validation/instrumental_quality.py",
        )],
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--plan", type=Path, required=True, help="固定診断のplan")
    parser.add_argument("--report", type=Path, required=True, help="固定診断のreport")
    args = parser.parse_args(argv)
    result = run_probe(args.plan.resolve(), args.report.resolve(), args.output.resolve())
    print(f"取りこぼしを戻した比較を保存しました: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
