"""case単独の再開記録と、元case×variantの全件対応を検証する。"""
import copy
import json

import numpy as np
import pytest
import soundfile as sf

from lss.bgm.inference_variants import InferenceVariantRecipe
from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.validation.bandit_inference_cache import InferenceCache, exclusive_output
from lss.validation.variant_receipt import normalized_records, verify_normalized


@pytest.fixture
def native(tmp_path, monkeypatch):
    # GPUそのものは別の実GPU受入で確認する。ここはmanifestと音声の対応境界。
    monkeypatch.setattr("scripts.infer_bgm_bandit_resumable.verify_fixed", lambda *args: None)
    source = tmp_path / "source.wav"
    sf.write(source, np.ones((20, 2)) * 0.1, 44100, subtype="FLOAT")
    source_record = {"path": str(source), "sha256": file_sha256(source), "frames": 20}
    definitions = [InferenceVariantRecipe(), InferenceVariantRecipe(-6, 11)]
    cases = [{"case_id": f"native-{i}", "source": source_record, "stems": {},
              "inference_variant": variant.to_dict()} for i, variant in enumerate(definitions)]
    plan = {"status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False, "holdout_consumed": False,
        "sample_rate": 44100, "cases": cases, "provider": {"name": "test"}, "variant_mapping": [
            {"case_id": case["case_id"], "original_case_id": "original", "variant_id": variant.recipe_id}
            for case, variant in zip(cases, definitions)]}
    activity_path = tmp_path / "activity.json"
    _write_json_atomic(activity_path, {"schema_version": 1, "method": "stem_energy_share_hysteresis",
        "onset_share": 0.2, "offset_share": 0.1, "onset_music_rms": 0.001, "offset_music_rms": 0.0005,
        "window_frames": 16, "hop_frames": 8, "bridge_gap_frames": 0,
        "minimum_active_frames": 1, "padding_frames": 0})
    rule = {"maximum_any_controlled_regression_db": 1, "minimum_material_controlled_improvement_db": 1,
            "maximum_absent_output_to_input_db": -30, "maximum_absent_activity_candidate_count": 0}
    original_path = tmp_path / "original.json"
    _write_json_atomic(original_path, {
        "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False, "holdout_consumed": False,
        "provider": plan["provider"], "code": {}, "variants": [variant.to_dict() for variant in definitions],
        "candidates": [{"label": "gain-phase", "variants": [definitions[1].to_dict()]}],
        "cases": [{"case_id": "original", "source": source_record, "reference": source_record,
            "stems": {"music": {"path": str(tmp_path / "unusable-old-cache.wav"), "sha256": "0" * 64}},
            "target_absent": False, "singing_guard": False, "evidence_role": "controlled-clean-component",
            "evaluation_frames": [0, 20]}], "decision_rule": rule, "claim_boundary": "Development only",
        "activity_recipe": {"path": str(activity_path), "sha256": file_sha256(activity_path)},
    })
    plan["original_comparison_plan"] = {"path": str(original_path.resolve()), "sha256": file_sha256(original_path)}
    plan["original_decision_rule"] = rule
    plan_path = tmp_path / "plan.json"
    _write_json_atomic(plan_path, plan)
    identity = {"plan_sha256": file_sha256(plan_path), "execution_device": "cuda:0", "cpu_fallback": False,
                "provider": plan["provider"], "runtime": {"torch": "test-CUDA-only"}}
    output = tmp_path / "output"
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        for case in cases:
            cache.save(case, {role: np.ones((20, 2)) * 0.1 for role in ("music", "speech", "effects")},
                       seconds=0.1, peak_cuda_bytes=1234)
        receipt_path = cache.complete(verify_identity=lambda: None)
    return plan_path, receipt_path


def test_native_mapping_preserves_real_adapter_evidence_and_physical_outputs(native):
    plan, receipt, records = normalized_records(*native)
    assert {row["case_id"] for row in records} == {"original"}
    assert len(records) == 2 and len({row["variant_id"] for row in records}) == 2
    assert all(row["source_hash_after_verified"] for row in records)
    assert all(row["reused_prior_comparison_cache"] is False for row in records)
    assert {row["native_case_id"] for row in records} == {"native-0", "native-1"}
    assert receipt["provider"] == plan["provider"]


@pytest.mark.parametrize("mutation", ["duplicate_mapping", "duplicate_receipt", "missing", "cpu", "case_variant"])
def test_ambiguous_incomplete_or_fallback_evidence_cannot_be_normalized(native, mutation):
    plan_path, receipt_path = native
    if mutation in ("duplicate_mapping", "case_variant"):
        plan = json.loads(plan_path.read_text())
        if mutation == "duplicate_mapping":
            plan["variant_mapping"].append(plan["variant_mapping"][0])
        else:
            # 最終receipt側のvariant対応だけを変えてもcase単独の記録とは一致しない。
            metadata = receipt_path.parent / "native-0" / "case.json"
            case = json.loads(metadata.read_text())
            case["inference_variant_id"] = "another-variant"
            _write_json_atomic(metadata, case)
        if mutation == "duplicate_mapping":
            _write_json_atomic(plan_path, plan)
            identity_path = receipt_path.parent / "identity.json"
            identity = json.loads(identity_path.read_text())
            identity["plan_sha256"] = file_sha256(plan_path)
            _write_json_atomic(identity_path, identity)
            receipt = json.loads(receipt_path.read_text())
            receipt["plan_sha256"] = file_sha256(plan_path)
            _write_json_atomic(receipt_path, receipt)
    else:
        receipt = json.loads(receipt_path.read_text())
        if mutation == "duplicate_receipt":
            receipt["cases"].append(receipt["cases"][0])
        elif mutation == "missing":
            receipt["cases"].pop()
        else:
            receipt["cpu_fallback"] = True
        _write_json_atomic(receipt_path, receipt)
    with pytest.raises(ValueError):
        normalized_records(plan_path, receipt_path)


def test_normalized_matrix_must_match_the_original_sources_and_every_variant(native):
    native_plan, native_receipt, records = normalized_records(*native)
    plan_path, receipt_path = native
    target = {"native_variant_evidence": {"plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "receipt": {"path": str(receipt_path), "sha256": file_sha256(receipt_path)}},
        "cases": [{"case_id": "original", "source": native_plan["cases"][0]["source"]}],
        "variants": [case["inference_variant"] for case in native_plan["cases"]]}
    receipt = {"cases": records, "provider": native_plan["provider"], "runtime": native_receipt["runtime"]}
    target["provider"] = native_plan["provider"]
    verify_normalized(target, receipt)
    missing = copy.deepcopy(target)
    missing["variants"].pop()
    with pytest.raises(ValueError, match="全case"):
        verify_normalized(missing, receipt)
    changed = copy.deepcopy(target)
    changed["cases"][0]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        verify_normalized(changed, receipt)
    corrupt = copy.deepcopy(receipt)
    corrupt["cases"][0]["stems"]["music"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="実際"):
        verify_normalized(target, corrupt)
    duplicate = copy.deepcopy(target)
    duplicate["cases"].append(duplicate["cases"][0])
    with pytest.raises(ValueError, match="一意"):
        verify_normalized(duplicate, receipt)


def test_freeze_and_evaluation_use_native_adapter_without_relabelling_it_as_legacy(native, tmp_path):
    from types import SimpleNamespace
    from lss.validation.variant_receipt import METHOD
    from scripts.prepare_resumable_bgm_variant_comparison import freeze
    from scripts.run_bgm_inference_probe import evaluate

    plan_path, receipt_path = native
    comparison = freeze(plan_path.parent / "original.json", plan_path, receipt_path, tmp_path / "comparison")
    frozen = json.loads(comparison.read_text())
    receipt = json.loads((comparison.parent / "receipt.json").read_text())
    assert frozen["receipt_method"] == METHOD
    assert receipt["adapter_sha256"] == frozen["inference_adapter"]["sha256"]
    assert frozen["cases"][0]["stems"] == {}
    evaluate(SimpleNamespace(plan=comparison, receipt=comparison.parent / "receipt.json", output=tmp_path / "result"))
    report = json.loads((tmp_path / "result" / "report.json").read_text())
    assert report["source_identity_preserved"]
    assert not report["promotion_allowed"] and not report["holdout_consumed"]


def test_changed_native_final_receipt_is_rejected_before_evaluation_outputs(native, tmp_path):
    from types import SimpleNamespace
    from scripts.prepare_resumable_bgm_variant_comparison import freeze
    from scripts.run_bgm_inference_probe import evaluate

    plan_path, receipt_path = native
    comparison = freeze(plan_path.parent / "original.json", plan_path, receipt_path, tmp_path / "comparison")
    corrupt = json.loads(receipt_path.read_text())
    corrupt["cases"].pop()
    _write_json_atomic(receipt_path, corrupt)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="証拠"):
        evaluate(SimpleNamespace(plan=comparison, receipt=comparison.parent / "receipt.json", output=output))
    assert not output.exists()


def test_changed_original_comparison_definition_cannot_be_hidden_by_native_receipt(native, tmp_path):
    from scripts.prepare_resumable_bgm_variant_comparison import freeze

    plan_path, receipt_path = native
    comparison = freeze(plan_path.parent / "original.json", plan_path, receipt_path, tmp_path / "comparison")
    target = json.loads(comparison.read_text())
    receipt = json.loads((comparison.parent / "receipt.json").read_text())
    original_path = plan_path.parent / "original.json"
    with original_path.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="元の比較構成"):
        verify_normalized(target, receipt)
