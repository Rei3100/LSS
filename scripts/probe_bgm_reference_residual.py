"""固定済みの33制御ケースと録音一致を再利用し、元Musicを保持する参照残差回収を比較する。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
import soundfile as sf

from lss.bgm import identity, identity_run, recording_recovery, reference_residual
from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.validation import stereo_comparison, stereo_quality


def read_audio(path):
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != 44100 or samples.shape[1] != 2 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"制御ケースは有限の44.1kHz stereoである必要があります: {path}")
    return samples.T


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matching-dir", type=Path,
                        default=dev_state_root(ROOT) / "validation/bgm-recording-recovery-controls-r3")
    args = parser.parse_args(argv)
    matching_plan_path, matching_report_path = args.matching_dir / "plan.json", args.matching_dir / "report.json"
    matching_plan = json.loads(matching_plan_path.read_bytes())
    cached = json.loads(matching_report_path.read_bytes())
    if (cached.get("status") != "COMPLETED_WITH_FROZEN_INPUTS_AND_POST_RUN_HASH_VERIFICATION"
            or cached.get("promotion_allowed") is not False or cached.get("holdout_consumed") is not False
            or cached.get("inputs_unchanged") is not True
            or cached.get("plan_sha256") != file_sha256(matching_plan_path)
            or matching_plan.get("matching_settings") != identity_run.evidence_settings()):
        raise ValueError("現在の録音一致条件に対応する完了した固定制御比較が必要です")
    cases = matching_plan["cases"]
    matches = {row["case_id"]: row for row in cached["matching"]}
    if len(cases) != 33 or len(matches) != 33 or {c["case_id"] for c in cases} != set(matches):
        raise ValueError("33制御ケースと固定録音一致の対応が不正です")
    recipes = [replace(reference_residual.ReferenceResidualRecipe(), blend=blend) for blend in (0.5, 1.0)]
    modules = (identity, identity_run, recording_recovery, reference_residual, stereo_quality, stereo_comparison)
    artifacts = {str(p.resolve()): file_sha256(p) for p in (
        matching_plan_path, matching_report_path, Path(__file__), *(Path(m.__file__) for m in modules),
    )}
    for case in cases:
        for item in [case["source"], case["reference"], case["stems"]["music"]]:
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"固定制御ケースの内容が変更されています: {item['path']}")
            artifacts[item["path"]] = item["sha256"]
        selected = matches[case["case_id"]]["selected"]
        if selected is not None:
            if selected["identity_state"] != "known_exact":
                raise ValueError("正の録音一致以外を回収へ使えません")
            if file_sha256(Path(selected["reference_path"])) != selected["reference_sha256"]:
                raise ValueError("固定録音一致の参照同一性が変更されています")
            artifacts[selected["reference_path"]] = selected["reference_sha256"]
    limits = [
        "Development制御ケース。正しい参照の選択と同一masterは制御作者のoracle。",
        "既存の33ケースと固定した録音一致をそのまま使う。未一致は元Musicを保持し不在にしない。",
        "元Musicと参照残差の候補を比較する。現行の製品cue rendererを置き換えない。",
        "参照と原本からの係数推定へ台詞・効果音が漏れる可能性がある。",
        "実番組のmaster違い、意味的劇伴、独立した実番組品質は未評価。",
        "追加の推論・source upload・Holdout・正式公開・既定変更は行わない。",
    ]
    gates = {"maximum_extra_gain_error_db": 0.1, "maximum_component_si_regression_db": 0.1,
             "maximum_joint_si_regression_db": 0.1, "maximum_snr_regression_db": 0.1,
             "maximum_extra_absent_energy_db": 0.1, "meaningful_improvement_db": 0.5}
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    _write_json_atomic(plan_path, {
        "status": "FROZEN_BEFORE_REFERENCE_RESIDUAL_COMPARISON", "promotion_allowed": False,
        "holdout_consumed": False, "cases": cases, "recipes": [r.to_dict() for r in recipes],
        "matching_report": {"path": str(matching_report_path.resolve()), "sha256": file_sha256(matching_report_path)},
        "inputs": artifacts, "gates": gates, "sample_rate": 44100,
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__}, "limits": limits,
    })
    plan_sha256 = file_sha256(plan_path)
    rows = []
    for case in cases:
        source, reference, music = map(read_audio, (case["source"]["path"], case["reference"]["path"],
                                                   case["stems"]["music"]["path"]))
        if source.shape != reference.shape or source.shape != music.shape:
            raise ValueError("制御ケースの原本・参照・Musicの時間軸が一致しません")
        target_absent = not np.any(reference)
        selected = matches[case["case_id"]]["selected"]
        match = identity.RecordingMatch(**selected) if selected is not None else None
        matched_reference = read_audio(match.reference_path) if match is not None else None
        baseline = stereo_quality.stereo_quality(music, reference)
        for recipe in recipes:
            observations = {"recovery_used": False, "reason": "no_positive_recording_match"}
            candidate = music.copy()
            if match is not None:
                candidate, observations = reference_residual.recover_reference_residual(
                    source, music, matched_reference, match, recipe,
                )
                observations["recovery_used"] = True
            measured = stereo_quality.stereo_quality(candidate, reference)
            failures, improvement, extra_absent = [], None, None
            if target_absent:
                extra_absent = stereo_quality.db_ratio(stereo_quality.energy(candidate), stereo_quality.energy(music))
                if extra_absent > gates["maximum_extra_absent_energy_db"]:
                    failures.append({"metric": "target_absent_energy", "extra_error_db": extra_absent})
            else:
                failures.extend(stereo_comparison.spatial_regressions(
                    baseline, measured, maximum_extra_gain_error_db=gates["maximum_extra_gain_error_db"],
                    maximum_component_si_regression_db=gates["maximum_component_si_regression_db"],
                ))
                improvement = measured["si_sdr_db"]["joint"] - baseline["si_sdr_db"]["joint"]
                if improvement < -gates["maximum_joint_si_regression_db"]:
                    failures.append({"metric": "joint_si_loss", "extra_error_db": -improvement})
                snr_loss = baseline["scale_dependent_snr_db"] - measured["scale_dependent_snr_db"]
                if snr_loss > gates["maximum_snr_regression_db"]:
                    failures.append({"metric": "scale_dependent_snr_loss", "extra_error_db": snr_loss})
            rows.append({"case_id": case["case_id"], "recipe_id": recipe.recipe_id, "target_absent": target_absent,
                         "wanted_role_policy": case["wanted_role_policy"], "joint_si_improvement_db": improvement,
                         "extra_absent_energy_db": extra_absent, "baseline": baseline, "candidate": measured,
                         "failures": failures, "observations": observations})
            print(json.dumps({"case_id": case["case_id"], "blend": recipe.blend,
                              "improvement_db": improvement, "failure_metrics": [f["metric"] for f in failures]}), flush=True)
    if len(rows) != len(cases) * len(recipes) or file_sha256(plan_path) != plan_sha256:
        raise ValueError("固定ケース×構成の完了または固定plan同一性が不正です")
    if any(file_sha256(Path(path)) != sha for path, sha in artifacts.items()):
        raise ValueError("比較中に制御素材・固定録音一致・実装が変更されています")
    summary = []
    for recipe in recipes:
        results = [row for row in rows if row["recipe_id"] == recipe.recipe_id]
        failed = [row["case_id"] for row in results if row["failures"]]
        improved = [row["case_id"] for row in results if (row["joint_si_improvement_db"] or 0) >= gates["meaningful_improvement_db"]]
        summary.append({"recipe_id": recipe.recipe_id, "regression_cases": failed,
                        "materially_improved_cases": improved, "decision": "REJECTED_EXACT_CONFIGURATION" if failed
                        else "DEVELOPMENT_SURVIVOR_REQUIRES_REAL_DOMAIN_EVIDENCE" if improved else "NO_MATERIAL_GAIN"})
    _write_json_atomic(output / "report.json", {
        "status": "COMPLETED_WITH_FROZEN_INPUTS_AND_POST_RUN_HASH_VERIFICATION", "plan_sha256": plan_sha256,
        "promotion_allowed": False, "holdout_consumed": False, "inputs_unchanged": True,
        "source_upload_used": False, "new_cuda_inference_performed": False, "winner_selected": False,
        "completed_at": datetime.now(timezone.utc).isoformat(), "case_variant_count": len(rows),
        "summary": summary, "cases": rows, "limits": limits,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
