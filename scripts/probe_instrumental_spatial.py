"""固定12ケースの伴奏へ空間再推定を比較する。GPU・通常処理・既定は変更しない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm.refinement import SpatialRefinementRecipe, refine_stereo
from lss.core import session
from lss.harness.fetch import file_sha256
from lss.validation.instrumental_quality import instrumental_quality
from scripts.probe_instrumental_repeat import DECISION_RULE, comparison


BLENDS = (0.25, 0.5, 1.0)


def run(baseline_path: Path, output: Path) -> Path:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    plan_path = baseline_path.parent / "plan.json"
    if (baseline.get("role") != "development_diagnostic_only"
            or baseline.get("promotion_allowed") is not False
            or file_sha256(plan_path) != baseline["plan_sha256"]):
        raise ValueError("完了した固定Development診断が必要です")
    definitions = json.loads(plan_path.read_text(encoding="utf-8"))["cases"]
    ids = [case["case_id"] for case in definitions]
    measured_ids = [case["case_id"] for case in baseline["cases"]]
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(measured_ids):
        raise ValueError("固定ケースの比較元が完全ではありません")
    definitions = {case["case_id"]: case for case in definitions}
    artifacts = []
    for prior in baseline["cases"]:
        if len(prior["result"]["outputs"]) != 1:
            raise ValueError("伴奏1件の完了結果だけを使います")
        artifacts.append({"path": prior["result"]["outputs"][0], "sha256": prior["output_sha256"]})
        artifacts.extend(definitions[prior["case_id"]]["artifacts"].values())
    for artifact in artifacts:
        if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"固定音声が変更されています: {artifact['path']}")
    recipe = SpatialRefinementRecipe()
    revisions = {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "scripts/probe_instrumental_repeat.py",
        ROOT / "lss/bgm/refinement.py", ROOT / "lss/validation/instrumental_quality.py",
        ROOT / "lss/validation/stereo_quality.py",
        ROOT / "lss/validation/stereo_comparison.py",
    )}
    output.mkdir(parents=True, exist_ok=False)
    frozen = {
        "schema_version": 1, "status": "FROZEN_BEFORE_RESULTS",
        "promotion_allowed": False, "holdout_consumed": False,
        "baseline": {"path": str(baseline_path), "sha256": file_sha256(baseline_path)},
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "cases": ids, "artifacts": artifacts, "spatial_recipe": recipe.to_dict(),
        "spatial_recipe_id": recipe.recipe_id, "blend_weights": list(BLENDS),
        "decision_rule": DECISION_RULE, "implementation": revisions,
        "stage_graph": ["cached_instrumental", "mixture_minus_instrumental_as_unwanted_estimate",
                        "local_stereo_wiener_with_separate_unexplained", "fixed_parent_refiner_blend"],
        "limits": ["public_training_components_not_independent_holdout",
                   "unwanted_estimate_is_not_clean_vocals", "no_semantic_presence_validation",
                   "cached_renderer_runtime_not_current_runtime_equivalence"],
    }
    frozen_path = output / "comparison-plan.json"
    session._write_json_atomic(frozen_path, frozen)
    cases = []
    for prior in baseline["cases"]:
        definition = definitions[prior["case_id"]]
        signals = {}
        for role, artifact in definition["artifacts"].items():
            values, rate = sf.read(artifact["path"], always_2d=True, dtype="float64")
            if rate != definition["sample_rate"] or values.shape != (definition["frames"], 2):
                raise ValueError("固定音声の時間軸が一致しません")
            signals[role] = values.T
        first = Path(prior["result"]["outputs"][0])
        estimate, rate = sf.read(first, always_2d=True, dtype="float64")
        estimate = estimate.T * 2.0 ** int(FLAC(first)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        if rate != definition["sample_rate"] or estimate.shape != signals["input"].shape:
            raise ValueError("比較元の時間軸が一致しません")
        refined = refine_stereo(signals["input"], {
            "Instrumental": estimate, "Unwanted": signals["input"] - estimate,
        }, recipe).stems["Instrumental"]
        variants = []
        for weight in BLENDS:
            candidate = (1 - weight) * estimate + weight * refined
            if candidate.shape != estimate.shape or not np.isfinite(candidate).all():
                raise ValueError("候補の形状・有限値が不正です")
            quality = instrumental_quality(candidate, signals["target"], signals["vocals"])
            saved = output / f"{prior['case_id']}-blend-{weight}.wav"
            sf.write(saved, candidate.T, rate, subtype="DOUBLE")
            variants.append({"weight": weight, "quality": quality,
                             "comparison": comparison(prior["quality"], quality),
                             "artifact": {"path": str(saved), "sha256": file_sha256(saved)}})
        cases.append({"case_id": prior["case_id"], "variant": prior["variant"], "variants": variants})
        print(f"空間比較完了: {prior['case_id']}", flush=True)
    # 共有作業ツリーでも比較中の入力・評価実装の変更を結果へ混入させない。
    for artifact in artifacts + [frozen["baseline"], frozen["plan"]]:
        if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("比較中に固定入力・参照・比較元が変更されています")
    for relative, expected in revisions.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("比較中に評価実装が変更されています")
    decisions = []
    for weight in BLENDS:
        rows = [(case, next(v for v in case["variants"] if v["weight"] == weight)) for case in cases]
        regressions = [case["case_id"] for case, row in rows if row["comparison"]["regression"]]
        quiet_gain = any(case["variant"] == "quiet-accompaniment"
                         and row["comparison"]["uncorrected_snr_delta_db"] >= DECISION_RULE["quiet_case_required_improvement_db"]
                         for case, row in rows)
        decisions.append({"weight": weight, "regression_cases": regressions,
                          "disposition": "candidate_survivor" if not regressions and quiet_gain
                          else "reject_as_general_cleanup"})
    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": 1, "status": "complete_development_comparison",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "plan_sha256": file_sha256(frozen_path), "source_hashes_after_verified": True,
        "cases": cases, "decisions": decisions,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"空間比較を保存しました: {run(args.baseline.resolve(), args.output.resolve())}")


if __name__ == "__main__":
    main()
