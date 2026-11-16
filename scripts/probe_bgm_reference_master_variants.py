"""参照残差の48kHz変換・stereo master差・速度差を固定33制御ケースで比較する。"""
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
from scipy.signal import resample_poly
import soundfile as sf

from lss.bgm import identity, identity_run, reference_residual, recording_recovery
from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.validation import reference_residual_episode as episode, stereo_comparison, stereo_quality
from scripts.probe_bgm_reference_residual import read_audio

SCENARIOS = (
    {"id": "48k-rate-conversion", "crossfeed": 0.0, "speed_numerator": 1, "speed_denominator": 1},
    {"id": "48k-stereo-crossfeed-15pct", "crossfeed": 0.15, "speed_numerator": 1, "speed_denominator": 1},
    {"id": "48k-reference-speed-slower-0.1pct", "crossfeed": 0.0, "speed_numerator": 1001, "speed_denominator": 1000},
)


def construct_reference(reference, scenario):
    """明示した制御変換。原本・元Music・採点用正解を変更しない。"""
    crossfeed = scenario["crossfeed"]
    transformed = (1 - crossfeed) * reference + crossfeed * reference[::-1]
    numerator, denominator = scenario["speed_numerator"], scenario["speed_denominator"]
    if numerator != denominator:
        transformed = resample_poly(transformed, numerator, denominator, axis=1,
                                    window=("kaiser", 5.0), padtype="constant")
    return resample_poly(transformed, 160, 147, axis=1, window=("kaiser", 5.0), padtype="constant")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matching-dir", type=Path, default=dev_state_root(ROOT) / "validation/bgm-recording-recovery-controls-r3")
    args = parser.parse_args(argv)
    parent_plan_path, parent_report_path = args.matching_dir / "plan.json", args.matching_dir / "report.json"
    parent, matching = json.loads(parent_plan_path.read_bytes()), json.loads(parent_report_path.read_bytes())
    if (matching.get("status") != "COMPLETED_WITH_FROZEN_INPUTS_AND_POST_RUN_HASH_VERIFICATION"
            or matching.get("inputs_unchanged") is not True or matching.get("promotion_allowed") is not False
            or matching.get("holdout_consumed") is not False or matching.get("plan_sha256") != file_sha256(parent_plan_path)
            or parent.get("matching_settings") != identity_run.evidence_settings()):
        raise ValueError("現在の一致条件と対応する完了した固定制御比較が必要です")
    cases = parent["cases"]
    matches = {row["case_id"]: row["selected"] for row in matching["matching"]}
    if len(cases) != 33 or len(matches) != 33 or {case["case_id"] for case in cases} != set(matches):
        raise ValueError("固定33ケースと録音一致の対応が不正です")
    artifacts = {str(path.resolve()): file_sha256(path) for path in (parent_plan_path, parent_report_path, Path(__file__),
        ROOT / "scripts/probe_bgm_reference_residual.py", *(Path(module.__file__) for module in (
            identity, identity_run, reference_residual, recording_recovery, episode, stereo_quality, stereo_comparison)))}
    for case in cases:
        for item in [case["source"], case["reference"], case["stems"]["music"]]:
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("固定制御素材の内容が変更されています")
            artifacts[item["path"]] = item["sha256"]
        selected = matches[case["case_id"]]
        if selected is not None:
            if selected["identity_state"] != "known_exact" or file_sha256(Path(selected["reference_path"])) != selected["reference_sha256"]:
                raise ValueError("固定した正の録音一致と参照同一性が必要です")
            artifacts[selected["reference_path"]] = selected["reference_sha256"]
    output = args.out_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    constructed = {}
    for index, case in enumerate(cases):
        selected = matches[case["case_id"]]
        if selected is None:
            continue
        raw = read_audio(selected["reference_path"])
        for scenario in SCENARIOS:
            path = output / "references" / scenario["id"] / f"case-{index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, construct_reference(raw, scenario).T, 48000, subtype="FLOAT")
            info = sf.info(path); item = episode.artifact(path)
            artifacts[item["path"]] = item["sha256"]
            factor = scenario["speed_numerator"] / scenario["speed_denominator"]
            match = {**selected, "reference_path": item["path"], "reference_sha256": item["sha256"],
                     "reference_start_seconds": selected["reference_start_seconds"] * factor}
            constructed[f"{case['case_id']}:{scenario['id']}"] = {"artifact": item, "match": match,
                "format": {"sample_rate": info.samplerate, "frames": info.frames, "channels": info.channels},
                "original_match": selected, "correspondence_evidence": "controlled_author_oracle_not_new_matcher_evidence"}
    recipes = [replace(reference_residual.ReferenceResidualRecipe(), blend=blend) for blend in (0.5, 1.0)]
    gates = {"maximum_extra_gain_error_db": 0.1, "maximum_component_si_regression_db": 0.1,
             "maximum_joint_si_regression_db": 0.1, "maximum_snr_regression_db": 0.1,
             "maximum_extra_absent_energy_db": 0.1, "meaningful_improvement_db": 0.5}
    limits = ["Developmentの固定33制御ケース。構成した参照の対応付けは作者oracle。",
              "参照変換後のmatcherを再実行していない。実製品の一致生産経路の成功を意味しない。",
              "原本・元Music・採点用正解を変更せず、参照だけを明示した規則で変える。",
              "実番組の独立した意味的正解・知覚品質・同一masterは証明しない。",
              "新しい推論・source upload・Holdout・正式公開・既定変更・勝者選択を行わない。"]
    plan_path = output / "plan.json"
    _write_json_atomic(plan_path, {"status": "FROZEN_BEFORE_REFERENCE_MASTER_VARIANTS", "promotion_allowed": False,
        "holdout_consumed": False, "cases": cases, "scenarios": SCENARIOS, "recipes": [r.to_dict() for r in recipes],
        "constructed_references": constructed, "inputs": artifacts, "gates": gates,
        "reference_conversion": episode.REFERENCE_CONVERSION,
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__}, "limits": limits})
    plan_hash = file_sha256(plan_path)
    rows = []
    for case in cases:
        source, reference, music = map(read_audio, (case["source"]["path"], case["reference"]["path"], case["stems"]["music"]["path"]))
        if source.shape != reference.shape or source.shape != music.shape:
            raise ValueError("制御ケースの原本・正解・Musicの時間軸が一致しません")
        absent = not np.any(reference)
        baseline = stereo_quality.stereo_quality(music, reference)
        for scenario in SCENARIOS:
            item = constructed.get(f"{case['case_id']}:{scenario['id']}")
            converted = episode._read_reference(item["artifact"]["path"], item["format"]) if item else None
            for recipe in recipes:
                candidate = music.copy(); observations = {"recovery_used": False, "reason": "no_positive_recording_match"}
                if item is not None:
                    candidate, observations = reference_residual.recover_reference_residual(
                        source, music, converted, identity.RecordingMatch(**item["match"]), recipe)
                    observations["recovery_used"] = True
                measured = stereo_quality.stereo_quality(candidate, reference)
                failures, improvement, extra_absent = [], None, None
                if absent:
                    extra_absent = stereo_quality.db_ratio(stereo_quality.energy(candidate), stereo_quality.energy(music))
                    if extra_absent > gates["maximum_extra_absent_energy_db"]:
                        failures.append({"metric": "extra_absent_energy", "extra_error_db": extra_absent})
                else:
                    failures.extend(stereo_comparison.spatial_regressions(baseline, measured,
                        maximum_extra_gain_error_db=gates["maximum_extra_gain_error_db"],
                        maximum_component_si_regression_db=gates["maximum_component_si_regression_db"]))
                    improvement = measured["si_sdr_db"]["joint"] - baseline["si_sdr_db"]["joint"]
                    if improvement < -gates["maximum_joint_si_regression_db"]:
                        failures.append({"metric": "joint_si_loss", "extra_error_db": -improvement})
                    snr_loss = baseline["scale_dependent_snr_db"] - measured["scale_dependent_snr_db"]
                    if snr_loss > gates["maximum_snr_regression_db"]:
                        failures.append({"metric": "scale_dependent_snr_loss", "extra_error_db": snr_loss})
                rows.append({"case_id": case["case_id"], "scenario": scenario["id"], "recipe_id": recipe.recipe_id,
                    "target_absent": absent, "baseline": baseline, "candidate": measured, "failures": failures,
                    "joint_si_improvement_db": improvement, "extra_absent_energy_db": extra_absent, "observations": observations})
            print(json.dumps({"case": case["case_id"], "scenario": scenario["id"],
                              "gate_failures": [len(row["failures"]) for row in rows[-2:]]}), flush=True)
    expected = len(cases) * len(SCENARIOS) * len(recipes)
    if len(rows) != expected or file_sha256(plan_path) != plan_hash or any(file_sha256(Path(p)) != sha for p,sha in artifacts.items()):
        raise ValueError("固定case×参照変換×構成の完了または事後hash同一性が不正です")
    summary = []
    for scenario in SCENARIOS:
        for recipe in recipes:
            subset = [r for r in rows if r["scenario"] == scenario["id"] and r["recipe_id"] == recipe.recipe_id]
            failed = [r["case_id"] for r in subset if r["failures"]]
            improved = [r["case_id"] for r in subset if (r["joint_si_improvement_db"] or 0) >= gates["meaningful_improvement_db"]]
            summary.append({"scenario":scenario["id"], "recipe_id":recipe.recipe_id,"regression_cases":failed,
                "materially_improved_cases":improved, "decision":"REJECTED_FOR_EXACT_REFERENCE_VARIANT" if failed
                else "DEVELOPMENT_SURVIVOR_REQUIRES_REAL_DOMAIN_EVIDENCE" if improved else "NO_MATERIAL_GAIN"})
    _write_json_atomic(output / "report.json", {"status":"COMPLETED_WITH_FROZEN_INPUTS_AND_POST_RUN_HASH_VERIFICATION",
        "plan_sha256":plan_hash, "case_variant_count":expected,"summary":summary,"cases":rows,
        "inputs_unchanged":True,"promotion_allowed":False,"holdout_consumed":False,"winner_selected":False,
        "new_inference_performed":False,"source_upload_used":False,"reference_correspondence_oracle_used":True,
        "new_reference_matching_performed":False,"completed_at":datetime.now(timezone.utc).isoformat(),"limits":limits})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
