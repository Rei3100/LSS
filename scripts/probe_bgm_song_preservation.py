"""歌唱を含む曲の候補を、固定BandIt Music rendererと現行伴奏rendererで比較する。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from lss.bgm.activity import discover_activity_file, load_activity_recipe  # noqa: E402
from lss.bgm.candidate_store import CandidateOrigin, materialize_candidate  # noqa: E402
from lss.bgm.provider_padding import padding_revision  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402
from lss.validation.activity_quality import activity_quality  # noqa: E402


def freeze(baseline_path: Path, provider_plan_path: Path, output: Path) -> Path:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    plan_path = baseline_path.parent / "plan.json"
    if (
        baseline.get("status") != "complete_development_diagnostic"
        or baseline.get("promotion_allowed") is not False
        or fetch.file_sha256(plan_path) != baseline["plan_sha256"]
    ):
        raise ValueError("完了した固定Development診断が必要です")
    parent = json.loads(provider_plan_path.read_text(encoding="utf-8"))
    if parent["promotion_allowed"] or parent["holdout_consumed"] or parent["sample_rate"] != 44100:
        raise ValueError("44.1kHz stereoの固定Development providerだけを使います")
    provider = parent["provider"]
    for artifact in provider["artifacts"].values():
        if fetch.file_sha256(Path(provider["root"]) / artifact["path"]) != artifact["sha256"]:
            raise ValueError("既存BandIt providerの固定内容が変わっています")
    baseline_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if {item["case_id"] for item in baseline["cases"]} != {item["case_id"] for item in baseline_plan["cases"]}:
        raise ValueError("baselineのcase coverageが完全ではありません")
    cases = []
    for item in baseline_plan["cases"]:
        source = item["input"]
        for artifact in (source, item["reference"]):
            if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("固定入力・参照が変わっています")
        info = sf.info(source["path"])
        if info.samplerate != 44100 or info.channels != 2:
            raise ValueError("BandItへは同じ44.1kHz stereo入力だけを渡します")
        cases.append({"case_id": item["case_id"], "family": item["family"],
                      "wanted_role_policy": item["wanted_role_policy"],
                      "source": {**source, "frames": info.frames}, "reference": item["reference"], "stems": {}})
    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": 1, "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False,
        "holdout_consumed": False, "sample_rate": 44100, "provider": provider, "cases": cases,
        "baseline": {"path": str(baseline_path), "sha256": fetch.file_sha256(baseline_path)},
        "activity_recipe": baseline_plan["activity_recipe"],
        "decision_rule": {"scope": "development_only", "maximum_wanted_snr_regression_db": 0.5,
                          "maximum_absent_candidate_frame_increase": 0,
                          "singing_required_snr_improvement_db": 1.0},
        "adapter_sha256": fetch.file_sha256(ROOT / "scripts/infer_bgm_bandit.py"),
        "short_input_padding": {
            "adapter_sha256": fetch.file_sha256(ROOT / "scripts/infer_bgm_bandit_short.py"),
            "implementation_sha256": fetch.file_sha256(ROOT / "lss/bgm/provider_padding.py"),
            "policy": padding_revision(),
        },
        "evaluation_revision": {
            str(path.relative_to(ROOT)): fetch.file_sha256(path) for path in (
                Path(__file__), ROOT / "lss/validation/activity_quality.py",
                ROOT / "lss/validation/stereo_quality.py", ROOT / "lss/bgm/candidate_store.py",
            )
        },
        "limits": baseline_plan["limits"] + ["different_renderer_and_runtime_no_equivalence_assertion",
                                             "no_source_role_classifier_or_automatic_routing_validated"],
    })
    return plan


def compare(before: dict, after: dict, rule: dict) -> dict:
    if before["target_present"] != after["target_present"]:
        raise ValueError("異なるtargetの診断は比較できません")
    if not before["target_present"]:
        delta = after["candidate_union_frames"] - before["candidate_union_frames"]
        return {"absent_candidate_frame_delta": delta,
                "regression": delta > rule["maximum_absent_candidate_frame_increase"]}
    delta = (after["separation_and_activity_fidelity"]["scale_dependent_snr_db"]
             - before["separation_and_activity_fidelity"]["scale_dependent_snr_db"])
    frame_delta = after["reference_frame_retained_fraction"] - before["reference_frame_retained_fraction"]
    return {"uncorrected_snr_delta_db": delta, "reference_frame_recall_delta": frame_delta,
            "regression": delta < -rule["maximum_wanted_snr_regression_db"] or frame_delta < -1e-12,
            "material_singing_improvement": delta >= rule["singing_required_snr_improvement_db"]}


def evaluate(plan_path: Path, receipt_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for relative, expected in plan["evaluation_revision"].items():
        if fetch.file_sha256(ROOT / relative) != expected:
            raise ValueError("固定した評価・候補保存処理が変更されています")
    if (
        plan["promotion_allowed"] or plan["holdout_consumed"]
        or receipt["status"] != "COMPLETED_CUDA_WITHOUT_FALLBACK"
        or receipt["plan_sha256"] != fetch.file_sha256(plan_path)
        or receipt["provider"] != plan["provider"]
        or receipt["adapter_sha256"] != plan["adapter_sha256"]
        or receipt.get("short_input_padding") != plan["short_input_padding"]
    ):
        raise ValueError("固定providerの完全なCUDA完了記録が必要です")
    if len(receipt["cases"]) != len(plan["cases"]) or {c["case_id"] for c in receipt["cases"]} != {c["case_id"] for c in plan["cases"]}:
        raise ValueError("全caseの一意な完了記録が必要です")
    baseline_artifact = plan["baseline"]
    if fetch.file_sha256(Path(baseline_artifact["path"])) != baseline_artifact["sha256"]:
        raise ValueError("baseline診断が変更されています")
    baseline = json.loads(Path(baseline_artifact["path"]).read_text(encoding="utf-8"))
    prior = {item["case_id"]: item for item in baseline["cases"]}
    outputs = {item["case_id"]: item for item in receipt["cases"]}
    recipe_artifact = plan["activity_recipe"]
    if fetch.file_sha256(Path(recipe_artifact["path"])) != recipe_artifact["sha256"]:
        raise ValueError("activity Recipeが変更されています")
    recipe = load_activity_recipe(Path(recipe_artifact["path"]))
    revision = {"provider": receipt["provider"], "adapter_sha256": receipt["adapter_sha256"],
                "runtime": receipt["runtime"], "sample_rate": plan["sample_rate"],
                "short_input_padding": receipt["short_input_padding"]}
    digest = hashlib.sha256(json.dumps(revision, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    renderer_id = f"bandit-development-v1-{digest}"
    records = []
    for case in plan["cases"]:
        output = outputs[case["case_id"]]
        if output["source_sha256"] != case["source"]["sha256"]:
            raise ValueError("推論したsourceが固定入力と一致しません")
        for artifact in (case["source"], case["reference"], *output["stems"].values()):
            if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("入力・参照・推論stemが変更されています")
        music = output["stems"]["music"]
        signal = Path(music["path"])
        estimate, rate = sf.read(signal, dtype="float64", always_2d=True)
        reference, reference_rate = sf.read(case["reference"]["path"], dtype="float64", always_2d=True)
        if rate != 44100 or reference_rate != rate or len(estimate) != case["source"]["frames"]:
            raise ValueError("評価する信号の時間軸が一致しません")
        discovery = discover_activity_file(signal, recipe)
        intervals = [(item.start_frame, item.end_frame) for item in discovery.candidates]
        quality = activity_quality(estimate.T, reference.T, intervals)
        assets = []
        for candidate in discovery.candidates:
            assets.append(materialize_candidate(
                signal, plan_path.parent / "candidates" / case["case_id"] / f"{candidate.candidate_id}.flac",
                origin=CandidateOrigin(case["source"]["sha256"], music["sha256"], renderer_id, renderer_id,
                                       recipe.recipe_id, candidate.start_frame, candidate.end_frame, rate),
                source_id=case["case_id"], source_path=Path(case["source"]["path"]),
                evidence={"method": recipe.method, "peak_rms": candidate.peak_rms,
                          "renderer_scope": "development_only_not_product_recipe",
                          "receipt_sha256": fetch.file_sha256(receipt_path)},
            ))
        records.append({"case_id": case["case_id"], "wanted_role_policy": case["wanted_role_policy"],
                        "candidate_count": len(intervals), "assets": assets, "quality": quality,
                        "comparison": compare(prior[case["case_id"]]["quality"], quality, plan["decision_rule"])})
    failures = [item["case_id"] for item in records if item["comparison"]["regression"]]
    singing_improved = [item["case_id"] for item in records
                        if item["wanted_role_policy"] == "entire_constructed_song_including_singing"
                        and item["case_id"].endswith(("singing-mix", "vocals-only", "quiet-accompaniment"))
                        and item["comparison"].get("material_singing_improvement")]
    for case in plan["cases"]:
        for artifact in (case["source"], case["reference"], *outputs[case["case_id"]]["stems"].values()):
            if fetch.file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("評価中に入力・参照・推論stemが変更されています")
    for relative, expected in plan["evaluation_revision"].items():
        if fetch.file_sha256(ROOT / relative) != expected:
            raise ValueError("評価中に固定処理が変更されています")
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": 1, "status": "complete_development_comparison",
        "promotion_allowed": False, "holdout_consumed": False, "default_changed": False,
        "disposition": "not_a_general_replacement" if failures else "development_survivor_not_promoted",
        "regression_cases": failures, "song_preservation_improved_cases": singing_improved,
        "plan_sha256": fetch.file_sha256(plan_path), "receipt_sha256": fetch.file_sha256(receipt_path),
        "renderer_id": renderer_id, "renderer_revision": revision, "cases": records,
        "evaluation_revision": plan["evaluation_revision"], "source_identity_preserved": True,
        "automatic_role_routing_validated": False, "implementation_sha256": fetch.file_sha256(Path(__file__)),
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("freeze")
    build.add_argument("baseline", type=Path)
    build.add_argument("provider_plan", type=Path)
    build.add_argument("output", type=Path)
    measure = commands.add_parser("evaluate")
    measure.add_argument("plan", type=Path)
    measure.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.baseline.resolve(), args.provider_plan.resolve(), args.output.resolve()))
    else:
        print(evaluate(args.plan.resolve(), args.receipt.resolve()))
