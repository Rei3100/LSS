"""固定12ケースで伴奏の2回目の分離を比較する。通常処理や既定は変更しない。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402
from mutagen.flac import FLAC  # noqa: E402

from lss.core import models_store, session, worker  # noqa: E402
from lss.harness import fetch  # noqa: E402
from lss.validation.instrumental_quality import instrumental_quality  # noqa: E402
from lss.validation.stereo_comparison import COMPONENT_SI_CEILING_DB, spatial_regressions  # noqa: E402


DECISION_RULE = {
    "policy_revision": 2,
    "scope": "development_survivor_only_no_promotion",
    "active_snr_max_regression_db": 0.1,
    "target_gain_max_extra_error_db": 0.1,
    "absence_max_regression_db": 0.1,
    "quiet_case_required_improvement_db": 1.0,
    "spatial_gain_max_extra_error_db": 0.1,
    "spatial_component_si_max_regression_db": 0.1,
    "spatial_component_si_ceiling_db": COMPONENT_SI_CEILING_DB,
}


def runtime_transition(before: dict, after: dict) -> dict:
    """共有環境への依存追加は交絡として残す。既存rendererの変更は比較を止める。"""
    for key in before.keys() | after.keys():
        if key not in {"runtime_id", "dependency_versions"} and before.get(key) != after.get(key):
            raise ValueError(f"比較途中に実行環境の{key}が変更されています")
    prior = {item["name"]: item["version"] for item in before["dependency_versions"]}
    current = {item["name"]: item["version"] for item in after["dependency_versions"]}
    if any(current.get(name) != version for name, version in prior.items()):
        raise ValueError("比較途中に既存依存が変更されています")
    return {
        "baseline_runtime_id": before["runtime_id"], "candidate_runtime_id": after["runtime_id"],
        "identity_equal": before["runtime_id"] == after["runtime_id"],
        "equivalence": "identity_equal" if before["runtime_id"] == after["runtime_id"] else "not_asserted",
        "added_dependencies": [{"name": name, "version": current[name]} for name in sorted(current.keys() - prior.keys())],
    }


def comparison(before: dict, after: dict) -> dict:
    active = before["target_present"]
    if active != after["target_present"]:
        raise ValueError("同じ対象の比較ではありません")
    if not active:
        prior, current = before["target_absent_output_to_vocals_db"], after["target_absent_output_to_vocals_db"]
        return {"absence_delta_db": current - prior,
                "regression": current - prior > DECISION_RULE["absence_max_regression_db"]}
    prior, current = before["fidelity"], after["fidelity"]
    delta = current["scale_dependent_snr_db"] - prior["scale_dependent_snr_db"]
    prior_gain, current_gain = prior["projection_gain_db"]["joint"], current["projection_gain_db"]["joint"]
    gain_error_delta = abs(current_gain) - abs(prior_gain) if prior_gain is not None and current_gain is not None else None
    spatial_loss = any(current["spatial_component_lost"].values())
    spatial_failures = spatial_regressions(prior, current,
        maximum_extra_gain_error_db=DECISION_RULE["spatial_gain_max_extra_error_db"],
        maximum_component_si_regression_db=DECISION_RULE["spatial_component_si_max_regression_db"])
    return {
        "uncorrected_snr_delta_db": delta, "absolute_gain_error_delta_db": gain_error_delta,
        "regression": delta < -DECISION_RULE["active_snr_max_regression_db"]
        or gain_error_delta is None or gain_error_delta > DECISION_RULE["target_gain_max_extra_error_db"]
        or spatial_loss or bool(spatial_failures),
        "spatial_regressions": spatial_failures,
    }


def run(baseline_path: Path, output: Path) -> Path:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("role") != "development_diagnostic_only" or baseline.get("promotion_allowed") is not False:
        raise ValueError("固定済みのDevelopment診断だけを比較します")
    plan_path = baseline_path.parent / "plan.json"
    if fetch.file_sha256(plan_path) != baseline["plan_sha256"]:
        raise ValueError("固定構成が変更されています")
    cases = json.loads(plan_path.read_text(encoding="utf-8"))["cases"]
    definitions = {case["case_id"]: case for case in cases}
    for case in baseline["cases"]:
        saved = Path(case["result"]["outputs"][0])
        if fetch.file_sha256(saved) != case["output_sha256"]:
            raise ValueError(f"比較元の成果が変更されています: {saved}")
        for artifact in definitions[case["case_id"]]["artifacts"].values():
            if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError(f"固定参照が変更されています: {artifact['path']}")
    output.mkdir(parents=True, exist_ok=False)
    private_ledger = output / "models.tsv"
    shutil.copyfile(models_store.MODELS_TSV, private_ledger)
    models_store.MODELS_TSV = private_ledger
    os.environ["LSS_STATE_DIR"] = str(output / "state")
    os.environ["LSS_MIN_FREE_BYTES"] = "0"
    # 判定条件と比較元の同一性を、2回目の推論より前に固定する。
    session._write_json_atomic(output / "comparison-plan.json", {
        "role": "development_diagnostic_only", "promotion_allowed": False,
        "baseline_path": str(baseline_path), "baseline_sha256": fetch.file_sha256(baseline_path),
        "decision_rule": DECISION_RULE, "stage_graph": ["fixed_first_separation", "same_model_second_separation"],
        "cases": [case["case_id"] for case in baseline["cases"]],
    })
    results = []
    for prior in baseline["cases"]:
        definition = definitions[prior["case_id"]]
        first = Path(prior["result"]["outputs"][0])
        data, rate = sf.read(first, always_2d=True, dtype="float64")
        shift = int(FLAC(first)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        second_input = output / f"{prior['case_id']}-first-estimate.wav"
        sf.write(second_input, data * 2.0 ** shift, rate, subtype="FLOAT")
        stage = prior["result"]["recipe_revision"]["stage_graph"][0]
        result = worker.handle_job(
            {"mix_path": str(second_input), "model_id": stage["model_id"], "stem_target": "Instrumental"},
            destination_dir=output / "outputs" / prior["case_id"],
        )
        if result["status"] not in {"ok", "skipped"} or len(result["outputs"]) != 1:
            raise RuntimeError(f"2回目の分離が失敗: {prior['case_id']}: {result}")
        candidate_stage = result["recipe_revision"]["stage_graph"][0]
        for key in ("model_sha256", "config_sha256", "arch"):
            if stage[key] != candidate_stage[key]:
                raise ValueError(f"2回目の処理で{key}が変更されています")
        transition = runtime_transition(prior["result"]["runtime_revision"], result["runtime_revision"])
        if results and results[0]["result"]["runtime_id"] != result["runtime_id"]:
            raise ValueError("候補の比較中に実行環境が変更されています")
        saved = Path(result["outputs"][0])
        estimate, saved_rate = sf.read(saved, always_2d=True, dtype="float64")
        target, reference_rate = sf.read(definition["artifacts"]["target"]["path"], always_2d=True, dtype="float64")
        vocals, vocals_rate = sf.read(definition["artifacts"]["vocals"]["path"], always_2d=True, dtype="float64")
        if saved_rate != reference_rate or vocals_rate != reference_rate:
            raise ValueError("2回目の処理で時間軸の形式が変更されています")
        output_shift = int(FLAC(saved)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        quality = instrumental_quality((estimate * 2.0 ** output_shift).T, target.T, vocals.T)
        results.append({"case_id": prior["case_id"], "variant": prior["variant"],
                        "quality": quality, "comparison": comparison(prior["quality"], quality),
                        "runtime_transition": transition,
                        "result": result, "output_sha256": fetch.file_sha256(saved)})
        print(f"2回目の固定比較完了: {prior['case_id']}", flush=True)
    regressions = [case["case_id"] for case in results if case["comparison"]["regression"]]
    quiet_gain = any(case["variant"] == "quiet-accompaniment"
                     and case["comparison"]["uncorrected_snr_delta_db"] >= DECISION_RULE["quiet_case_required_improvement_db"]
                     for case in results)
    report = output / "report.json"
    session._write_json_atomic(report, {
        "role": "development_diagnostic_only", "promotion_allowed": False,
        "default_changed": False, "holdout_consumed": False,
        "decision_rule": DECISION_RULE, "cases": results, "regression_cases": regressions,
        "disposition": "candidate_survivor" if not regressions and quiet_gain else "reject_as_general_cleanup",
        "implementation_sha256": fetch.file_sha256(Path(__file__)),
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"2回目の分離比較を保存しました: {run(args.baseline.resolve(), args.output.resolve())}")


if __name__ == "__main__":
    main()
