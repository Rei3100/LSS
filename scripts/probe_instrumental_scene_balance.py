"""場面の種類の観測が、固定ケースの品質の弱い側と一致するかを測る。

新しい推論は行わない。固定診断の伴奏出力と、同じ入力で保存した歌唱出力を読み直し、
釣り合いの観測を当てる。参照は品質の測定にだけ使い、観測の決定には使わない。

境目は結果を見る前に固定する。0dBは「どちらのstemが大きいか」という意味の境目で、
±3dBはその近くを判断しないための幅、−40dBは既に測られている対象不在の残留（−47〜−49dB）
より上の値である。素材へ合わせて選んだ値ではない。
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
from lss.separation.scene_balance import SceneBalanceRecipe, read_balance  # noqa: E402

SCHEMA_VERSION = 1
RECIPE = SceneBalanceRecipe(absent_below_db=-40.0, quiet_below_db=-3.0, dominant_above_db=3.0)
DECISION_RULE = {
    "the_boundaries_are_declared_before_the_results": True,
    "the_reference_is_used_for_measurement_only": True,
    "a_balance_state_is_not_a_quality_verdict": True,
}
LIMITS = [
    "reads_frozen_stems_no_new_inference",
    "public_training_material_and_constructed_linear_mixtures_only",
    "twelve_cases_are_not_a_promotion_corpus",
    "the_balance_uses_the_two_stems_so_it_inherits_their_errors",
    "no_publication_default_change_or_holdout_use",
]


class SceneBalanceProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _read(path: Path) -> np.ndarray:
    values, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return values.T


def run_probe(report_path: Path, stems_path: Path, output: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stems = json.loads(stems_path.read_text(encoding="utf-8"))
    if any(item.get("promotion_allowed") is not False for item in (report, stems)):
        raise SceneBalanceProbeError("固定したDevelopment記録だけを読み直します")
    if stems.get("type") != "instrumental_stem_complement_report":
        raise SceneBalanceProbeError(f"この歌唱側の記録は読めません: {stems_path}")
    produced = {row["case_id"]: row for row in stems["cases"]}
    before = {path: fetch.file_sha256(path) for path in (report_path, stems_path)}
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for item in report["cases"]:
        stem = produced[item["case_id"]]
        folder = stems_path.parent / "outputs" / item["case_id"]
        [found] = sorted(folder.glob("*.flac"))
        if fetch.file_sha256(found) != stem["vocals_output_sha256"]:
            raise SceneBalanceProbeError(f"固定した歌唱出力が変更されています: {found}")
        instrumental = (_read(Path(item["result"]["outputs"][0]))
                        * 2.0 ** item["serialization_shift"])
        vocals = _read(found) * 2.0 ** stem["vocals_serialization_shift"]
        frames = min(instrumental.shape[1], vocals.shape[1])
        read = read_balance(instrumental[:, :frames], vocals[:, :frames], RECIPE)
        fidelity = item["quality"]["fidelity"]
        rows.append({"case_id": item["case_id"], "variant": item["variant"],
                     "balance_state": read["state"], "balance_db": read["balance_db"],
                     "si_sdr_db": (fidelity.get("si_sdr_db") or {}).get("joint"),
                     "target_absent_output_to_vocals_db":
                         item["quality"].get("target_absent_output_to_vocals_db")})
        print(f"{item['case_id']}: {read['state']} ({read['balance_db']})", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise SceneBalanceProbeError(f"固定入力が変更されています: {path}")

    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})
        state = by_variant[row["variant"]]
        state[row["balance_state"]] = state.get(row["balance_state"], 0) + 1
    quiet = [row["si_sdr_db"] for row in rows if row["balance_state"] == "target_quiet"]
    rest = [row["si_sdr_db"] for row in rows
            if row["balance_state"] not in ("target_quiet", "target_absent")
            and row["si_sdr_db"] is not None]

    result = output / "report.json"
    session._write_json_atomic(result, {
        "schema_version": SCHEMA_VERSION,
        "type": "instrumental_scene_balance_report",
        "status": "complete_development_measurement",
        "role": "development_measurement_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "recipe": RECIPE.to_dict(), "recipe_id": RECIPE.recipe_id,
        "decision_rule": DECISION_RULE,
        "case_count": len(rows),
        "states_by_variant": by_variant,
        "worst_si_sdr_outside_the_quiet_band": min(rest) if rest else None,
        "best_si_sdr_inside_the_quiet_band": max(quiet) if quiet else None,
        "cases": rows,
        "limits": LIMITS,
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/separation/scene_balance.py",
        )],
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--report", type=Path, required=True, help="固定診断のreport")
    parser.add_argument("--stems", type=Path, required=True, help="歌唱側の測定record")
    args = parser.parse_args(argv)
    result = run_probe(args.report.resolve(), args.stems.resolve(), args.output.resolve())
    print(f"場面の種類の観測を保存しました: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
