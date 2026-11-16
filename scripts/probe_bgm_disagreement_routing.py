"""歌唱と伴奏を守る自動分岐を固定ケースで比較する。通常処理・既定は変更しない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm.activity import discover_stem_energy_share_files, load_stem_energy_share_recipe
from lss.bgm.cue_routing import CueRoutingRecipe, route_cues_stereo
from lss.bgm.routing import DisagreementRoutingRecipe, route_stereo
from lss.core.session import _write_json_atomic
from lss.harness.fetch import file_sha256
from lss.validation.activity_quality import activity_quality
from lss.validation.stereo_comparison import COMPONENT_SI_CEILING_DB, spatial_regressions


def checked_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(artifacts: list[dict]) -> None:
    for artifact in artifacts:
        if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError(f"固定入力・推定・参照が変更されています: {artifact['path']}")


def freeze(baseline_path: Path, song_plan_path: Path, receipt_path: Path,
           activity_path: Path, output: Path, *, cue_persistent: bool = False,
           cue_context: bool = False, directional: bool = False) -> Path:
    if cue_context and not cue_persistent:
        raise ValueError("候補端のcontextは区間単位の分岐でだけ利用できます")
    if directional and not cue_persistent:
        raise ValueError("候補全体のエネルギー条件は区間単位の分岐でだけ利用できます")
    baseline = checked_json(baseline_path)
    original_plan = baseline_path.parent / "plan.json"
    if (baseline.get("status") != "complete_development_diagnostic"
            or baseline.get("promotion_allowed") is not False
            or file_sha256(original_plan) != baseline["plan_sha256"]):
        raise ValueError("完了した固定Development診断が必要です")
    song_plan, receipt = checked_json(song_plan_path), checked_json(receipt_path)
    if (song_plan.get("promotion_allowed") is not False or song_plan.get("holdout_consumed") is not False
            or receipt.get("status") != "COMPLETED_CUDA_WITHOUT_FALLBACK"
            or receipt.get("plan_sha256") != file_sha256(song_plan_path)):
        raise ValueError("固定Development構成の全CUDA完了記録が必要です")
    prior_ids = [case["case_id"] for case in baseline["cases"]]
    receipt_ids = [case["case_id"] for case in receipt["cases"]]
    source_ids = [case["case_id"] for case in song_plan["cases"]]
    if (len(prior_ids) != len(set(prior_ids)) or len(receipt_ids) != len(set(receipt_ids))
            or len(source_ids) != len(set(source_ids)) or sorted(prior_ids) != sorted(receipt_ids)
            or sorted(prior_ids) != sorted(source_ids)):
        raise ValueError("全caseの一意な完了記録が必要です")
    receipt_cases = {case["case_id"]: case for case in receipt["cases"]}
    baseline_cases = {case["case_id"]: case for case in baseline["cases"]}
    artifacts = [{"path": str(path), "sha256": file_sha256(path)} for path in (
        baseline_path, original_plan, song_plan_path, receipt_path, activity_path,
    )]
    cases = []
    for case in song_plan["cases"]:
        prior = baseline_cases[case["case_id"]]
        recorded_run = {"path": prior["run_path"], "sha256": prior["run_sha256"]}
        verify([recorded_run])
        run = checked_json(Path(prior["run_path"]))
        if run["source"]["sha256"] != case["source"]["sha256"]:
            raise ValueError("両rendererの入力が一致しません")
        instrumental = run["separation"]["artifacts"]
        if len(instrumental) != 1 or instrumental[0]["stem"] != "Instrumental":
            raise ValueError("伴奏1件の比較元だけを使います")
        inferred = receipt_cases[case["case_id"]]
        if inferred["source_sha256"] != case["source"]["sha256"]:
            raise ValueError("Music rendererの入力が一致しません")
        artifacts.extend([recorded_run, case["source"], case["reference"], instrumental[0],
                          *inferred["stems"].values()])
        cases.append({**case, "instrumental": instrumental[0], "stems": inferred["stems"],
                      "baseline_quality": prior["quality"]})
    verify(artifacts)
    route = DisagreementRoutingRecipe()
    cue = CueRoutingRecipe(window_recipe=route, context_frames=route.window_frames if cue_context else 0,
                           require_energy_increase=directional)
    activity = load_stem_energy_share_recipe(activity_path)
    revisions = {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "lss/bgm/routing.py", ROOT / "lss/bgm/cue_routing.py", ROOT / "lss/bgm/activity.py",
        ROOT / "lss/validation/activity_quality.py", ROOT / "lss/validation/stereo_quality.py",
        ROOT / "lss/validation/stereo_comparison.py",
    )}
    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    _write_json_atomic(plan, {
        "schema_version": 1, "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False,
        "holdout_consumed": False, "cases": cases, "artifacts": artifacts,
        "route_recipe": route.to_dict(), "route_recipe_id": route.recipe_id,
        "route_kind": "cue_persistent" if cue_persistent else "window_interpolated",
        "cue_recipe": cue.to_dict() if cue_persistent else None,
        "cue_recipe_id": cue.recipe_id if cue_persistent else None,
        "activity_recipe": {"path": str(activity_path), "sha256": file_sha256(activity_path),
                            "recipe_id": activity.recipe_id},
        "rule": {"policy_revision": 2, "non_singing_maximum_snr_regression_db": 0.5,
                 "singing_maximum_snr_regression_against_music_db": 0.5,
                 "singing_minimum_improvement_against_instrumental_db": 1.0,
                 "maximum_absent_candidate_frames": 0,
                 "maximum_reference_energy_recall_regression": 0.001,
                 "spatial_gain_max_extra_error_db": 0.1,
                 "spatial_component_si_max_regression_db": 0.1,
                 "spatial_component_si_ceiling_db": COMPONENT_SI_CEILING_DB},
        "implementation": revisions,
        "renderer_evidence": {"instrumental_runtime_id": baseline["runtime_id"],
                              "music_runtime": receipt["runtime"], "runtime_equivalence": "not_asserted"},
        "limits": ["development_components_not_independent_final_validation",
                   "renderer_disagreement_not_semantic_singing_truth",
                   "same_provider_stem_shares_not_independent_role_validation",
                   "activity_stage_differs_from_original_rms_baseline"],
    })
    return plan


def evaluate(plan_path: Path) -> Path:
    plan = checked_json(plan_path)
    if plan.get("status") != "FROZEN_BEFORE_RESULTS" or plan.get("promotion_allowed") is not False:
        raise ValueError("推論前に固定したDevelopment構成が必要です")
    verify(plan["artifacts"])
    for relative, expected in plan["implementation"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("固定した評価実装が変更されています")
    route = DisagreementRoutingRecipe(**plan["route_recipe"])
    activity = load_stem_energy_share_recipe(Path(plan["activity_recipe"]["path"]))
    if route.recipe_id != plan["route_recipe_id"] or activity.recipe_id != plan["activity_recipe"]["recipe_id"]:
        raise ValueError("固定Recipeの同一性が一致しません")
    records = []
    for case in plan["cases"]:
        signals = {}
        for role, artifact in {"Instrumental": case["instrumental"], "Reference": case["reference"],
                               **{role.title(): art for role, art in case["stems"].items()}}.items():
            path = Path(artifact["path"])
            values, rate = sf.read(path, dtype="float64", always_2d=True)
            if rate != 44100 or values.shape != (case["source"]["frames"], 2):
                raise ValueError("両rendererと参照の時間軸・stereo形式が一致しません")
            shift = int(FLAC(path)["LSS_OUTPUT_HEADROOM_SHIFT"][0]) if role == "Instrumental" else 0
            signals[role] = values.T * 2.0 ** shift
        stems = {role: signals[role] for role in ("Music", "Speech", "Effects")}
        directory = plan_path.parent / "cases" / case["case_id"]
        directory.mkdir(parents=True, exist_ok=False)
        paths = {}
        for role, values in stems.items():
            path = directory / f"{role}.wav"
            sf.write(path, values.T, 44100, subtype="DOUBLE")
            paths[role] = path
        def discover(music_path):
            return discover_stem_energy_share_files(
                {"Music": music_path, "Speech": paths["Speech"], "Effects": paths["Effects"]}, activity)
        music_discovery = discover(paths["Music"])
        music_intervals = [(c.start_frame, c.end_frame) for c in music_discovery.candidates]
        cue_persistent = plan.get("route_kind") == "cue_persistent"
        if cue_persistent:
            cue_definition = dict(plan["cue_recipe"])
            cue_definition["window_recipe"] = DisagreementRoutingRecipe(**cue_definition["window_recipe"])
            cue_recipe = CueRoutingRecipe(**cue_definition)
            if cue_recipe.recipe_id != plan["cue_recipe_id"]:
                raise ValueError("固定cue Recipeの同一性が一致しません")
            routed = route_cues_stereo(signals["Instrumental"], stems, music_intervals, cue_recipe)
            routed_intervals = [(c["start_frame"], c["end_frame"]) for c in routed.cues]
            evidence = {"cues": routed.cues,
                        "music_weight_mean": sum(c["end_frame"] - c["start_frame"] for c in routed.cues
                                                 if c["use_music"]) / case["source"]["frames"]}
        else:
            routed = route_stereo(signals["Instrumental"], stems, route)
            evidence = {"windows": routed.windows, "music_weight_mean": float(routed.music_weights.mean())}
        paths["Routed"] = directory / "Routed.wav"
        sf.write(paths["Routed"], routed.audio.T, 44100, subtype="DOUBLE")
        def quality(music_path, intervals=None):
            if intervals is None:
                intervals = [(c.start_frame, c.end_frame) for c in discover(music_path).candidates]
            values = sf.read(music_path, dtype="float64", always_2d=True)[0].T
            return activity_quality(values, signals["Reference"], intervals)
        measured = quality(paths["Routed"], routed_intervals if cue_persistent else None)
        music_quality = quality(paths["Music"], music_intervals)
        singing = (case["wanted_role_policy"] == "entire_constructed_song_including_singing"
                   and case["case_id"].endswith(("singing-mix", "vocals-only", "quiet-accompaniment")))
        failures, deltas, spatial_failures = [], {}, []
        rule = plan["rule"]
        if not measured["target_present"]:
            if measured["candidate_union_frames"] > rule["maximum_absent_candidate_frames"]:
                failures.append("absent_activity_false_positive")
        else:
            base = case["baseline_quality"]
            for name, other in (("instrumental", base), ("music", music_quality)):
                deltas[f"snr_against_{name}_db"] = (measured["separation_and_activity_fidelity"]["scale_dependent_snr_db"]
                    - other["separation_and_activity_fidelity"]["scale_dependent_snr_db"])
            comparator = music_quality if singing else base
            maximum_regression = rule["singing_maximum_snr_regression_against_music_db" if singing
                                      else "non_singing_maximum_snr_regression_db"]
            if deltas["snr_against_music_db" if singing else "snr_against_instrumental_db"] < -maximum_regression:
                failures.append("wanted_signal_regression")
            if measured["reference_energy_retained_fraction"] < comparator["reference_energy_retained_fraction"] - rule["maximum_reference_energy_recall_regression"]:
                failures.append("wanted_activity_energy_regression")
            if any(measured["separation_and_activity_fidelity"]["spatial_component_lost"].values()):
                failures.append("spatial_component_lost")
            # routing自体の左右/Mid/Side損傷は全信号で比較する。activityでマスクすると、
            # 親側の候補区間が空のケースで比較不能となり、チャンネル劣化を見逃す。
            spatial_failures = spatial_regressions(comparator["separation_fidelity"],
                measured["separation_fidelity"],
                maximum_extra_gain_error_db=rule["spatial_gain_max_extra_error_db"],
                maximum_component_si_regression_db=rule["spatial_component_si_max_regression_db"])
            if spatial_failures:
                failures.append("individual_stereo_component_regression")
        records.append({"case_id": case["case_id"], "singing": singing, "quality": measured,
                        "music_quality": music_quality, "comparison": deltas, "failures": failures,
                        "spatial_regressions": spatial_failures,
                        **evidence,
                        "artifact": {"path": str(paths["Routed"]), "sha256": file_sha256(paths["Routed"])}})
        print(f"分岐比較完了: {case['case_id']} ({','.join(failures) or '規則内'})", flush=True)
    verify(plan["artifacts"])
    for relative, expected in plan["implementation"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("比較中に評価実装が変更されています")
    failures = [r["case_id"] for r in records if r["failures"]]
    improved = [r["case_id"] for r in records if r["singing"] and
                r["comparison"].get("snr_against_instrumental_db", -300) >= plan["rule"]["singing_minimum_improvement_against_instrumental_db"]]
    report = plan_path.parent / "report.json"
    _write_json_atomic(report, {
        "schema_version": 1, "status": "complete_development_routing_comparison",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "plan_sha256": file_sha256(plan_path), "source_hashes_after_verified": True,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "soundfile": sf.__version__},
        "disposition": "development_survivor" if not failures and improved else "reject_general_routing",
        "regression_cases": failures, "singing_improved_cases": improved, "cases": records,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    for name in ("baseline", "song_plan", "receipt", "activity", "output"):
        prepare.add_argument(name, type=Path)
    prepare.add_argument("--cue-persistent", action="store_true")
    prepare.add_argument("--cue-context", action="store_true")
    prepare.add_argument("--directional", action="store_true")
    measure = commands.add_parser("evaluate")
    measure.add_argument("plan", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.baseline.resolve(), args.song_plan.resolve(), args.receipt.resolve(),
                     args.activity.resolve(), args.output.resolve(), cue_persistent=args.cue_persistent,
                     cue_context=args.cue_context, directional=args.directional))
    else:
        print(evaluate(args.plan.resolve()))


if __name__ == "__main__":
    main()
