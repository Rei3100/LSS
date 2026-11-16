"""再開可能なcase単位の完了記録を、元case×variantの比較へ厳密に対応付ける。"""
from __future__ import annotations

import json
from pathlib import Path

from lss.bgm.inference_variants import InferenceVariantRecipe
from lss.harness.fetch import file_sha256
from lss.validation.bandit_inference_cache import InferenceCache, validate_plan

METHOD = "resumable-per-case-variant-receipt-v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_records(plan_path: Path, receipt_path: Path) -> tuple[dict, dict, list[dict]]:
    from scripts.infer_bgm_bandit_resumable import verify_fixed

    plan, receipt = read(plan_path), read(receipt_path)
    validate_plan(plan)
    plan_hash = file_sha256(plan_path)
    verify_fixed(plan_path, plan, plan_hash)
    identity = read(receipt_path.parent / "identity.json")
    if (receipt.get("status") != "COMPLETED_CUDA_WITHOUT_FALLBACK"
            or identity.get("plan_sha256") != plan_hash
            or receipt.get("execution_device") != "cuda:0" or receipt.get("cpu_fallback") is not False
            or any(receipt.get(key) != value for key, value in identity.items())):
        raise ValueError("固定したCUDA専用の再開完了記録が必要です")
    expected = {case["case_id"]: case for case in plan["cases"]}
    mapping = {item["case_id"]: item for item in plan["variant_mapping"]}
    completed = {case["case_id"]: case for case in receipt["cases"]}
    if (len(mapping) != len(plan["variant_mapping"]) or set(mapping) != set(expected)
            or len(completed) != len(receipt["cases"]) or set(completed) != set(expected)):
        raise ValueError("元case×variantへの一意な全件対応が必要です")
    cache = InferenceCache(receipt_path.parent, plan, identity, resume=True)
    records, keys = [], set()
    for case_id, case in expected.items():
        native = cache.load(case)
        if native is None or {key: value for key, value in native.items() if key != "identity"} != completed[case_id]:
            raise ValueError("ケース単独の完了記録と最終receiptが一致しません")
        item = mapping[case_id]
        variant_id = InferenceVariantRecipe(**case["inference_variant"]).recipe_id
        key = item["original_case_id"], item["variant_id"]
        if item["variant_id"] != variant_id or key in keys:
            raise ValueError("元case×variantの対応が重複または不一致です")
        keys.add(key)
        records.append({"case_id": item["original_case_id"], "variant_id": variant_id,
            "native_case_id": case_id, "source_sha256": native["source_sha256"], "stems": native["stems"],
            "seconds": native["seconds"], "peak_cuda_bytes": native["peak_cuda_bytes"],
            "source_hash_after_verified": True, "reused_prior_comparison_cache": False,
            "native_case_recovery_policy": "validated_complete_case_only_v1"})
    return plan, receipt, records


def verify_normalized(plan: dict, receipt: dict) -> None:
    native = plan["native_variant_evidence"]
    for item in native.values():
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("固定した再開推論の証拠が変更されています")
    source_plan, source_receipt, records = normalized_records(Path(native["plan"]["path"]), Path(native["receipt"]["path"]))
    original_link = source_plan["original_comparison_plan"]
    if file_sha256(Path(original_link["path"])) != original_link["sha256"]:
        raise ValueError("推論前に固定した元の比較構成が変更されています")
    if (receipt.get("cases") != records or plan["provider"] != source_plan["provider"]
            or receipt.get("provider") != source_plan["provider"]
            or receipt.get("runtime") != source_receipt["runtime"]):
        raise ValueError("比較用receiptが実際の再開推論と一致しません")
    variants = {InferenceVariantRecipe(**value).recipe_id for value in plan["variants"]}
    sources = {case["case_id"]: case["source"]["sha256"] for case in plan["cases"]}
    if (not sources or not variants or len(sources) != len(plan["cases"])
            or len(variants) != len(plan["variants"])):
        raise ValueError("比較には一意なcaseとvariantが必要です")
    expected = {(case_id, variant_id) for case_id in sources for variant_id in variants}
    if {(row["case_id"], row["variant_id"]) for row in records} != expected:
        raise ValueError("比較の全case×variantがそろっていません")
    if any(row["source_sha256"] != sources[row["case_id"]] for row in records):
        raise ValueError("元の比較入力と再開推論のsourceが一致しません")
