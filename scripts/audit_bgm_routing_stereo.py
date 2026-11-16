"""保存済みBGM分岐の固定ベクトルを、左右・Mid/Sideの追加損傷で再評価する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.validation.stereo_comparison import COMPONENT_SI_CEILING_DB, spatial_regressions


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(artifacts: list[dict]) -> None:
    for item in artifacts:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"固定した入力・成果物が変更されています: {item['path']}")


def implementation() -> dict:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "lss/validation/stereo_comparison.py")}


def freeze(reports: list[Path], output: Path) -> Path:
    if not reports or len({path.resolve() for path in reports}) != len(reports):
        raise ValueError("重複のない比較記録が必要です")
    artifacts, comparisons = [], []
    for path in reports:
        plan_path = path.parent / "plan.json"
        report, original = read(path), read(plan_path)
        if (report.get("status") != "complete_development_routing_comparison"
                or report.get("promotion_allowed") is not False
                or report.get("holdout_consumed") is not False
                or report.get("source_hashes_after_verified") is not True
                or original.get("promotion_allowed") is not False
                or original.get("holdout_consumed") is not False
                or file_sha256(plan_path) != report.get("plan_sha256")):
            raise ValueError("完了した未昇格Developmentの固定比較が必要です")
        ids = [case["case_id"] for case in report["cases"]]
        source_ids = [case["case_id"] for case in original["cases"]]
        if (not ids or len(set(ids)) != len(ids) or len(set(source_ids)) != len(source_ids)
                or set(ids) != set(source_ids)):
            raise ValueError("全caseの一意な比較記録が必要です")
        artifacts.extend({"path": str(p.resolve()), "sha256": file_sha256(p)} for p in (path, plan_path))
        artifacts.extend(original["artifacts"])
        artifacts.extend(case["artifact"] for case in report["cases"])
        comparisons.append({"report": str(path.resolve()), "original_plan": str(plan_path.resolve())})
    verify(artifacts)
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    _write_json_atomic(plan_path, {
        "schema_version": 1, "status": "FROZEN_BEFORE_STEREO_REEVALUATION",
        "promotion_allowed": False, "holdout_consumed": False,
        "artifacts": artifacts, "comparisons": comparisons, "implementation": implementation(),
        "rule": {"maximum_extra_gain_error_db": 0.1, "maximum_component_si_regression_db": 0.1,
                 "component_si_ceiling_db": COMPONENT_SI_CEILING_DB},
        "scope": "stored_metric_reevaluation_not_fresh_inference_or_new_ground_truth",
    })
    return plan_path


def evaluate(plan_path: Path) -> Path:
    plan = read(plan_path)
    if (plan.get("status") != "FROZEN_BEFORE_STEREO_REEVALUATION"
            or plan.get("promotion_allowed") is not False or plan.get("holdout_consumed") is not False
            or plan.get("implementation") != implementation()
            or plan["rule"]["component_si_ceiling_db"] != COMPONENT_SI_CEILING_DB):
        raise ValueError("固定したDevelopment評価構成が必要です")
    verify(plan["artifacts"])
    groups = []
    for comparison in plan["comparisons"]:
        report = read(Path(comparison["report"]))
        original = read(Path(comparison["original_plan"]))
        source_cases = {case["case_id"]: case for case in original["cases"]}
        cases = []
        for case in report["cases"]:
            spatial = []
            if case["quality"]["target_present"]:
                before = case["music_quality"] if case["singing"] else source_cases[case["case_id"]]["baseline_quality"]
                spatial = spatial_regressions(
                    before["separation_and_activity_fidelity"], case["quality"]["separation_and_activity_fidelity"],
                    maximum_extra_gain_error_db=plan["rule"]["maximum_extra_gain_error_db"],
                    maximum_component_si_regression_db=plan["rule"]["maximum_component_si_regression_db"])
            failures = list(case["failures"])
            if spatial and "individual_stereo_component_regression" not in failures:
                failures.append("individual_stereo_component_regression")
            cases.append({"case_id": case["case_id"], "singing": case["singing"],
                          "prior_failures": case["failures"], "failures": failures,
                          "spatial_regressions": spatial,
                          "newly_detected_spatial_regression": bool(spatial) and not case["failures"]})
        failed = [case["case_id"] for case in cases if case["failures"]]
        groups.append({"source": comparison["report"], "prior_disposition": report["disposition"],
            "disposition": "reject_general_routing" if failed else report["disposition"],
            "regression_cases": failed, "cases": cases,
            "newly_detected_spatial_regression_cases": [case["case_id"] for case in cases
                                                        if case["newly_detected_spatial_regression"]]})
    verify(plan["artifacts"])
    if plan["implementation"] != implementation():
        raise ValueError("比較中に評価実装が変更されています")
    output = plan_path.parent / "report.json"
    if output.exists():
        raise FileExistsError("完了した再評価を上書きしません")
    _write_json_atomic(output, {
        "schema_version": 1, "status": "complete_development_stereo_reevaluation",
        "plan_sha256": file_sha256(plan_path), "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "source_hashes_after_verified": True, "groups": groups,
        "limits": ["stored_metric_vectors_not_new_ground_truth", "original_runtime_uncertainty_retained",
                   "prior_reports_unchanged", "development_only_no_final_quality_claim"],
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("reports", type=Path, nargs="+")
    measure = commands.add_parser("evaluate")
    measure.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(freeze(args.reports, args.output) if args.command == "freeze" else evaluate(args.plan))


if __name__ == "__main__":
    main()
