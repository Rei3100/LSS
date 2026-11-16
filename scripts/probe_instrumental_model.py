"""静かな伴奏の不足を、台帳内の別候補と固定Developmentケースで比較する。"""
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
from lss.core import gpu_lease, models_store, session, worker  # noqa: E402
from lss.harness import infer  # noqa: E402
from lss.harness.fetch import file_sha256  # noqa: E402
from lss.harness.tsv_io import save_tsv  # noqa: E402
from lss.validation.instrumental_quality import instrumental_quality  # noqa: E402
from scripts.probe_instrumental_repeat import DECISION_RULE, comparison, runtime_transition  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(artifacts: list[dict]) -> None:
    for item in artifacts:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"固定比較の内容が変更されています: {item['path']}")


def revisions() -> dict:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "scripts/probe_instrumental_repeat.py",
        ROOT / "lss/validation/instrumental_quality.py", ROOT / "lss/validation/stereo_quality.py",
        ROOT / "lss/validation/stereo_comparison.py",
    )}


def physical_target(stems: list[str]) -> tuple[str, str]:
    named = [stem for stem in stems if stem.casefold() == "instrumental"]
    if len(named) == 1:
        return named[0], "provider_named_instrumental_not_final_quality_validation"
    if len(stems) == 1 and stems[0].casefold() == "other":
        return stems[0], "provider_other_tested_as_instrumental_development_hypothesis_only"
    raise ValueError("伴奏仮説を比較できる単独出力ではありません")


def baseline_inputs(path: Path) -> tuple[dict, list[dict]]:
    baseline = read_json(path)
    plan_path = path.parent / "plan.json"
    plan = read_json(plan_path)
    if (baseline.get("role") != "development_diagnostic_only" or baseline.get("promotion_allowed") is not False
            or plan.get("role") != "development_diagnostic_only" or plan.get("promotion_allowed") is not False
            or file_sha256(plan_path) != baseline["plan_sha256"]):
        raise ValueError("固定した完了Development診断が必要です")
    ids = [case["case_id"] for case in plan["cases"]]
    measured = [case["case_id"] for case in baseline["cases"]]
    if not ids or len(set(ids)) != len(ids) or len(set(measured)) != len(measured) or set(ids) != set(measured):
        raise ValueError("全固定ケースの一意な比較元が必要です")
    artifacts = [{"path": str(path.resolve()), "sha256": file_sha256(path)},
                 {"path": str(plan_path.resolve()), "sha256": file_sha256(plan_path)}]
    for case in plan["cases"]:
        artifacts.extend(case["artifacts"].values())
    for case in baseline["cases"]:
        if case["result"]["status"] not in {"ok", "skipped"} or len(case["result"]["outputs"]) != 1:
            raise ValueError("比較元は伴奏1件の検証済み完了結果である必要があります")
        artifacts.append({"path": case["result"]["outputs"][0], "sha256": case["output_sha256"]})
    verify(artifacts)
    return baseline, artifacts


def freeze(baseline_path: Path, model_id: str, output: Path) -> Path:
    baseline, artifacts = baseline_inputs(baseline_path)
    header, rows = models_store.load_model_ledger(models_store.MODELS_TSV)
    selected = [row for row in rows if row["model_id"] == model_id and row.get("状態") in {"既定", "候補"}]
    if len(selected) != 1 or selected[0].get("infer_tool") != "msst":
        raise ValueError("台帳にある利用可能なMSST候補だけを比較します")
    if any(model_id == case["result"]["recipe_revision"]["stage_graph"][0]["model_id"] for case in baseline["cases"]):
        raise ValueError("同じモデルの再分離は別の既存比較を使います")
    output.mkdir(parents=True, exist_ok=False)
    ledger = output / "models.tsv"
    shutil.copyfile(models_store.MODELS_TSV, ledger)
    row = selected[0]
    weights, config = models_store.ensure_model_cached(
        model_id, row["distributor_url"], row["config_url"], int(row["size_bytes"]), models_tsv_path=ledger)
    header, rows = models_store.load_model_ledger(ledger)
    row = next(row for row in rows if row["model_id"] == model_id)
    row["sha256"], row["config_sha256"] = file_sha256(weights), file_sha256(config)
    if "config_sha256" not in header:
        header.append("config_sha256")
    save_tsv(ledger, header, rows)
    stem, role_hypothesis = physical_target(infer.expected_output_layout(config, 44100)[0])
    with gpu_lease.acquire():
        runtime = infer.capture_msst_runtime().to_dict()
    artifacts += [{"path": str(weights.resolve()), "sha256": row["sha256"]},
                  {"path": str(config.resolve()), "sha256": row["config_sha256"]}]
    verify(artifacts)
    plan = output / "comparison-plan.json"
    session._write_json_atomic(plan, {
        "schema_version": 1, "status": "FROZEN_BEFORE_MODEL_RESULTS", "promotion_allowed": False,
        "holdout_consumed": False, "baseline": artifacts[0], "baseline_plan": artifacts[1],
        "cases": [case["case_id"] for case in baseline["cases"]], "artifacts": artifacts,
        "candidate": {"model_id": row["model_id"], "arch": row["arch"],
                      "model_sha256": row["sha256"], "config_sha256": row["config_sha256"]},
        "private_ledger": str(ledger.resolve()), "runtime_revision": runtime,
        "physical_stem": stem, "role_hypothesis": role_hypothesis,
        "decision_rule": DECISION_RULE, "implementation": revisions(),
        "stage_graph": ["same_fixed_linear_mixture", "alternate_registered_instrumental_model"],
        "limits": ["public_training_AAC_components_not_final_evaluation", "no_default_or_capability_promotion",
                   "observed_artifact_hashes_not_publisher_attestation", "runtime_difference_reported_not_hidden"],
    })
    return plan


def summarize(cases: list[dict]) -> dict:
    regressions = [case["case_id"] for case in cases if case["comparison"]["regression"]]
    quiet = [case["case_id"] for case in cases if case["variant"] == "quiet-accompaniment"
             and case["comparison"]["uncorrected_snr_delta_db"] >= DECISION_RULE["quiet_case_required_improvement_db"]]
    return {"regression_cases": regressions, "quiet_improvement_cases": quiet,
            "disposition": "development_candidate_survivor" if not regressions and quiet else "reject_as_general_replacement"}


def run(plan_path: Path) -> Path:
    plan = read_json(plan_path)
    if (plan.get("status") != "FROZEN_BEFORE_MODEL_RESULTS" or plan.get("promotion_allowed") is not False
            or plan.get("holdout_consumed") is not False or plan["decision_rule"] != DECISION_RULE):
        raise ValueError("推論前に固定した未昇格Development比較だけを実行します")
    expected_plan_hash = file_sha256(plan_path)
    verify(plan["artifacts"])
    if revisions() != plan["implementation"]:
        raise ValueError("比較・評価実装が固定構成と一致しません")
    baseline = read_json(Path(plan["baseline"]["path"]))
    definitions = {case["case_id"]: case for case in read_json(Path(plan["baseline_plan"]["path"]))["cases"]}
    models_store.MODELS_TSV = Path(plan["private_ledger"])
    os.environ["LSS_STATE_DIR"] = str(plan_path.parent / "state")
    os.environ["LSS_MIN_FREE_BYTES"] = "0"
    results = []
    for prior in baseline["cases"]:
        case = definitions[prior["case_id"]]
        result = worker.handle_job({"mix_path": case["artifacts"]["input"]["path"],
                                   "model_id": plan["candidate"]["model_id"], "stem_target": plan["physical_stem"]},
                                  destination_dir=plan_path.parent / "outputs" / prior["case_id"])
        if result["status"] not in {"ok", "skipped"} or len(result["outputs"]) != 1:
            raise RuntimeError(f"別候補の分離に失敗しました: {prior['case_id']}: {result}")
        stage = result["recipe_revision"]["stage_graph"][0]
        if any(stage[key] != value for key, value in plan["candidate"].items()):
            raise ValueError("候補の推論が固定モデル・設定と一致しません")
        runtime = result.get("runtime_revision")
        if runtime is None:
            runtime = infer.capture_msst_runtime().to_dict()
        if runtime != plan["runtime_revision"] or runtime["device"] != "cuda:0":
            raise ValueError("候補の推論runtimeが固定条件と一致しません")
        transition = runtime_transition(prior["result"]["runtime_revision"], runtime)
        target, rate = sf.read(case["artifacts"]["target"]["path"], dtype="float64", always_2d=True)
        vocals, vocals_rate = sf.read(case["artifacts"]["vocals"]["path"], dtype="float64", always_2d=True)
        saved = Path(result["outputs"][0])
        estimate, saved_rate = sf.read(saved, dtype="float64", always_2d=True)
        if rate != saved_rate or rate != vocals_rate or estimate.shape != target.shape:
            raise ValueError("候補のstereo/timelineが比較元と一致しません")
        shift = int(FLAC(saved)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        quality = instrumental_quality((estimate * 2.0 ** shift).T, target.T, vocals.T)
        record = {"case_id": prior["case_id"], "variant": prior["variant"], "quality": quality,
                  "comparison": comparison(prior["quality"], quality), "runtime_transition": transition,
                  "result": result, "output_sha256": file_sha256(saved), "serialization_shift": shift}
        session._write_json_atomic(plan_path.parent / f"{prior['case_id']}-result.json", record)
        results.append(record)
        print(json.dumps({"case_id": prior["case_id"], "comparison": record["comparison"]}), flush=True)
    if [case["case_id"] for case in results] != plan["cases"]:
        raise ValueError("全固定ケースの比較がそろっていません")
    verify(plan["artifacts"])
    if revisions() != plan["implementation"] or file_sha256(plan_path) != expected_plan_hash:
        raise ValueError("比較中に実装・構成が変更されています")
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": 1, "status": "complete_development_model_comparison", "promotion_allowed": False,
        "holdout_consumed": False, "default_changed": False, "plan_sha256": expected_plan_hash,
        "source_hashes_after_verified": True, "cases": results, "decision": summarize(results),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freezing = sub.add_parser("freeze")
    freezing.add_argument("baseline", type=Path)
    freezing.add_argument("model_id")
    freezing.add_argument("output", type=Path)
    running = sub.add_parser("run")
    running.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(freeze(args.baseline, args.model_id, args.output) if args.command == "freeze" else run(args.plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
