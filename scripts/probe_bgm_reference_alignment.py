"""固定33ケースで原本だけから参照速度を選び、元Musicと既存回収への劣化を比較する。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from fractions import Fraction
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

from lss.bgm import identity, identity_run, reference_alignment, reference_residual, recording_recovery
from lss.harness.fetch import file_sha256
from lss.validation import reference_residual_episode as episode, stereo_comparison, stereo_quality
from scripts.probe_bgm_reference_master_variants import construct_reference
from scripts.probe_bgm_reference_residual import read_audio

SCENARIOS = (
    {"id": "48k-rate-conversion", "crossfeed": 0.0, "speed_numerator": 1, "speed_denominator": 1},
    {"id": "48k-stereo-crossfeed-15pct", "crossfeed": 0.15, "speed_numerator": 1, "speed_denominator": 1},
    {"id": "48k-reference-speed-slower-0.1pct", "crossfeed": 0.0, "speed_numerator": 1001, "speed_denominator": 1000},
    {"id": "48k-reference-speed-faster-0.1pct", "crossfeed": 0.0, "speed_numerator": 999, "speed_denominator": 1000},
    {"id": "48k-reference-speed-slower-0.05pct", "crossfeed": 0.0, "speed_numerator": 2001, "speed_denominator": 2000},
)
GATES = {"maximum_extra_gain_error_db": 0.1, "maximum_component_si_regression_db": 0.1,
         "maximum_joint_si_regression_db": 0.1, "maximum_snr_regression_db": 0.1,
         "maximum_extra_absent_energy_db": 0.1, "meaningful_improvement_db": 0.5}


def failures(baseline, measured, absent, baseline_energy, candidate_energy):
    if absent:
        extra = stereo_quality.db_ratio(candidate_energy, baseline_energy)
        return [{"metric": "extra_absent_energy", "extra_error_db": extra}] if extra > GATES["maximum_extra_absent_energy_db"] else []
    failed = stereo_comparison.spatial_regressions(baseline, measured,
        maximum_extra_gain_error_db=GATES["maximum_extra_gain_error_db"],
        maximum_component_si_regression_db=GATES["maximum_component_si_regression_db"])
    for name, loss, limit in (
        ("joint_si_loss", baseline["si_sdr_db"]["joint"] - measured["si_sdr_db"]["joint"], GATES["maximum_joint_si_regression_db"]),
        ("scale_dependent_snr_loss", baseline["scale_dependent_snr_db"] - measured["scale_dependent_snr_db"], GATES["maximum_snr_regression_db"]),
    ):
        if loss > limit:
            failed.append({"metric": name, "extra_error_db": loss})
    return failed


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
        raise ValueError("現在の一致条件と対応する完了した固定33ケースが必要です")
    cases = parent["cases"]
    matches = {row["case_id"]: row["selected"] for row in matching["matching"]}
    if len(cases) != 33 or len(matches) != 33 or {c["case_id"] for c in cases} != set(matches):
        raise ValueError("固定33ケースと録音一致の対応が不正です")
    paths = [parent_plan_path, parent_report_path, Path(__file__), ROOT / "scripts/probe_bgm_reference_master_variants.py",
             ROOT / "scripts/probe_bgm_reference_residual.py", *(Path(m.__file__) for m in (
                 identity, identity_run, reference_alignment, reference_residual, recording_recovery,
                 episode, stereo_comparison, stereo_quality))]
    artifacts = {str(p.resolve()): file_sha256(p) for p in paths}
    for case in cases:
        for item in [case["source"], case["reference"], case["stems"]["music"]]:
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("固定した制御ケースが変更されています")
            artifacts[item["path"]] = item["sha256"]
        selected = matches[case["case_id"]]
        if selected is not None:
            if selected["identity_state"] != "known_exact" or file_sha256(Path(selected["reference_path"])) != selected["reference_sha256"]:
                raise ValueError("固定した正の録音一致と参照同一性が必要です")
            artifacts[selected["reference_path"]] = selected["reference_sha256"]
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
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
            item = episode.artifact(path); info = sf.info(path)
            artifacts[item["path"]] = item["sha256"]
            factor = scenario["speed_numerator"] / scenario["speed_denominator"]
            effective_start = selected["reference_start_seconds"] * factor
            duration = float(Fraction(info.frames, info.samplerate)) - effective_start
            constructed[f"{case['case_id']}:{scenario['id']}"] = {"artifact": item,
                "format": {"sample_rate": info.samplerate, "frames": info.frames, "channels": info.channels},
                "overlap_policy": "retain_parent_outside_actual_authored_reference_overlap",
                "match": {**selected, "reference_path": item["path"], "reference_sha256": item["sha256"],
                          "reference_start_seconds": effective_start,
                          "query_end_seconds": min(selected["query_end_seconds"], selected["query_start_seconds"] + duration)}}
    legacy_recipe = replace(reference_residual.ReferenceResidualRecipe(), blend=0.5)
    recipe = replace(legacy_recipe, alignment_recipe=reference_alignment.ReferenceAlignmentRecipe())
    limits = ["Development固定33ケース。参照の録音対応は制御作者oracle。実製品のmatcherを再評価しない。",
              "速度の選択は原本と参照のみ。採点用正解・Music・構成した速度の正解を選択器へ渡さない。",
              "既知の左右差と速度差を含む5参照構成。元Musicと整列なしの回収への劣化を別々に集計する。",
              "実番組の意味的劇伴・独立知覚品質・同一masterは未証明。正式公開・既定変更・Holdoutなし。"]
    plan_path = output / "plan.json"
    episode._write_record(plan_path, {"status": "FROZEN_BEFORE_SOURCE_ONLY_REFERENCE_SPEED_COMPARISON",
        "cases": cases, "scenarios": SCENARIOS, "constructed_references": constructed,
        "recipe": recipe.to_dict(), "legacy_recipe": legacy_recipe.to_dict(), "gates": GATES,
        "inputs": artifacts, "promotion_allowed": False, "holdout_consumed": False,
        "reference_conversion": episode.REFERENCE_CONVERSION,
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__}, "limits": limits},
        lambda: verify_inputs(artifacts))
    plan_hash = file_sha256(plan_path)
    rows = []
    for case in cases:
        source, reference, music = map(read_audio, (case["source"]["path"], case["reference"]["path"], case["stems"]["music"]["path"]))
        if source.shape != reference.shape or source.shape != music.shape:
            raise ValueError("原本・正解・Musicの時間軸が一致しません")
        absent = not np.any(reference)
        baseline = stereo_quality.stereo_quality(music, reference)
        for scenario in SCENARIOS:
            item = constructed.get(f"{case['case_id']}:{scenario['id']}")
            legacy, candidate = music.copy(), music.copy()
            observations = {"recovery_used": False, "reason": "no_positive_recording_match"}
            if item is not None:
                converted = episode._read_reference(item["artifact"]["path"], item["format"])
                match = identity.RecordingMatch(**item["match"])
                legacy, _ = reference_residual.recover_reference_residual(source, music, converted, match, legacy_recipe)
                candidate, observations = reference_residual.recover_reference_residual(source, music, converted, match, recipe)
                observations["recovery_used"] = True
            legacy_score, measured = [stereo_quality.stereo_quality(audio, reference) for audio in (legacy, candidate)]
            failure = failures(baseline, measured, absent, stereo_quality.energy(music), stereo_quality.energy(candidate))
            legacy_failure = failures(legacy_score, measured, absent, stereo_quality.energy(legacy), stereo_quality.energy(candidate))
            row = {"case_id": case["case_id"], "scenario": scenario["id"], "recipe_id": recipe.recipe_id,
                "target_absent": absent, "wanted_role_policy": case["wanted_role_policy"],
                "baseline": baseline, "legacy": legacy_score, "candidate": measured,
                "failures_vs_music": failure, "failures_vs_legacy": legacy_failure, "observations": observations,
                "joint_si_improvement_vs_music_db": measured["si_sdr_db"]["joint"] - baseline["si_sdr_db"]["joint"] if not absent else None,
                "joint_si_improvement_vs_legacy_db": measured["si_sdr_db"]["joint"] - legacy_score["si_sdr_db"]["joint"] if not absent else None}
            rows.append(row)
            print(json.dumps({"case": case["case_id"], "scenario": scenario["id"],
                "speed_used": observations.get("speed_alignment", {}).get("speed_correction_used", False),
                "failures": len(failure) + len(legacy_failure), "gain_vs_legacy_db": row["joint_si_improvement_vs_legacy_db"]}), flush=True)
    summary = []
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario["id"]]
        failed = [r["case_id"] for r in subset if r["failures_vs_music"] or r["failures_vs_legacy"]]
        improved = [r["case_id"] for r in subset if (r["joint_si_improvement_vs_legacy_db"] or 0) >= GATES["meaningful_improvement_db"]]
        summary.append({"scenario": scenario["id"], "regression_cases": failed, "materially_improved_vs_legacy": improved,
            "speed_correction_case_count": sum(r["observations"].get("speed_alignment", {}).get("speed_correction_used", False) for r in subset),
            "decision": "REJECTED_EXACT_CONFIGURATION" if failed else "DEVELOPMENT_SURVIVOR_REQUIRES_REAL_DOMAIN_EVIDENCE" if improved else "NO_MATERIAL_GAIN"})
    def verify():
        verify_inputs(artifacts)
        if len(rows) != len(cases) * len(SCENARIOS) or file_sha256(plan_path) != plan_hash:
            raise ValueError("固定case×参照構成の完了またはplan同一性が不正です")
    episode._write_record(output / "report.json", {"status": "COMPLETED_WITH_FROZEN_INPUTS_AND_POST_RUN_HASH_VERIFICATION",
        "plan_sha256": plan_hash, "case_variant_count": len(rows), "recipe_id": recipe.recipe_id,
        "cases": rows, "summary": summary, "inputs_unchanged": True, "promotion_allowed": False,
        "holdout_consumed": False, "winner_selected": False, "new_inference_performed": False,
        "source_upload_used": False, "completed_at": datetime.now(timezone.utc).isoformat(), "limits": limits}, verify)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def verify_inputs(artifacts):
    if any(file_sha256(Path(path)) != digest for path, digest in artifacts.items()):
        raise ValueError("固定した入力・参照・実装が変更されています")


if __name__ == "__main__":
    raise SystemExit(main())
