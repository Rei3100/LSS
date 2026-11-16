"""再開CUDA推論から、旧cacheを使わないBGMの固定比較と来歴付きreceiptを作る。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.validation.variant_receipt import METHOD, normalized_records
from scripts.run_bgm_inference_probe import checked_plan


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def freeze(original_path: Path, native_plan_path: Path, native_receipt_path: Path, output: Path) -> Path:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("promotion_allowed") is not False or original.get("holdout_consumed") is not False:
        raise ValueError("未昇格Developmentの元比較構成が必要です")
    native_plan, native_receipt, records = normalized_records(native_plan_path, native_receipt_path)
    original_link = native_plan["original_comparison_plan"]
    if original_link != artifact(original_path) or native_plan["original_decision_rule"] != original["decision_rule"]:
        raise ValueError("推論前に固定した元構成と採否規則が必要です")
    plan = copy.deepcopy(original)
    for case in plan["cases"]:
        case["stems"] = {}
        if case.get("context_stems"):
            plan.setdefault("unavailable_context_comparators", []).append({
                "case_id": case["case_id"], "reason": "old_cache_not_reused_after_identity_mismatch"})
            case["context_stems"] = {}
    plan["native_variant_evidence"] = {
        "plan": artifact(native_plan_path), "receipt": artifact(native_receipt_path),
        "identity": artifact(native_receipt_path.parent / "identity.json"),
    }
    plan["receipt_method"] = METHOD
    plan["original_comparison_plan"] = artifact(original_path)
    plan["original_decision_rule"] = copy.deepcopy(original["decision_rule"])
    plan["decision_rule_scope"] = "original rule preserved; stricter stereo gates fixed in evaluator before this comparison; no retuning against case results"
    plan["inference_adapter"] = artifact(ROOT / "scripts/infer_bgm_bandit_resumable.py")
    plan["decision_rule"]["spatial_gain_max_extra_error_db"] = 0.1
    plan["decision_rule"]["spatial_component_si_max_regression_db"] = 0.1
    if any(case.get("wanted_components") for case in plan["cases"]):
        plan["decision_rule"]["maximum_extra_wanted_component_gain_error_db"] = 0.1
    plan["code"] = {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "scripts/run_bgm_inference_probe.py", ROOT / "scripts/run_bgm_spatial_probe.py",
        ROOT / "lss/bgm/inference_variants.py", ROOT / "lss/bgm/activity.py",
        ROOT / "lss/validation/stereo_quality.py", ROOT / "lss/validation/stereo_comparison.py",
        ROOT / "lss/validation/variant_receipt.py", ROOT / "scripts/infer_bgm_bandit_resumable.py",
        ROOT / "lss/validation/song_component_comparison.py", ROOT / "lss/validation/instrumental_quality.py")}
    plan["claim_boundary"] += "; fresh uniform runtime; original stale caches not used; Development only"
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    _write_json_atomic(plan_path, plan)
    checked_plan(plan_path)
    receipt = {"status": "COMPLETED_CUDA_WITHOUT_FALLBACK", "plan_sha256": file_sha256(plan_path),
        "provider": native_plan["provider"], "adapter_sha256": plan["inference_adapter"]["sha256"],
        "receipt_method": METHOD, "runtime": native_receipt["runtime"], "cases": records}
    from lss.validation.variant_receipt import verify_normalized
    verify_normalized(plan, receipt)
    _write_json_atomic(output / "receipt.json", receipt)
    return plan_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "native_plan", "native_receipt", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(freeze(args.original, args.native_plan, args.native_receipt, args.output))


if __name__ == "__main__":
    main()
