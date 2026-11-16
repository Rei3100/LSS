"""同じCUDA runtimeの推定を左右交換・極性反転で平均し、静かな伴奏を比較する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from mutagen.flac import FLAC  # noqa: E402
from lss.core import gpu_lease, models_store, session, worker  # noqa: E402
from lss.harness import infer  # noqa: E402
from lss.harness.fetch import file_sha256  # noqa: E402
from lss.validation.instrumental_quality import instrumental_quality  # noqa: E402
from scripts.probe_instrumental_model import baseline_inputs, read_json, summarize, verify  # noqa: E402
from scripts.probe_instrumental_repeat import DECISION_RULE, comparison  # noqa: E402

TRANSFORMS = ("identity", "swap", "negative", "swap_negative")
VARIANTS = {"swap_mean": ("identity", "swap"), "polarity_mean": ("identity", "negative"),
            "full_group_mean": TRANSFORMS}


def transform(audio: np.ndarray, name: str) -> np.ndarray:
    if (name not in TRANSFORMS or audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0
            or np.iscomplexobj(audio) or not np.isfinite(audio).all()):
        raise ValueError("有限ステレオと固定した変換が必要です")
    values = audio[::-1] if name in {"swap", "swap_negative"} else audio
    return (-values if name in {"negative", "swap_negative"} else values).copy()


def combine(estimates: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    if not names or len(set(names)) != len(names) or any(name not in estimates for name in names):
        raise ValueError("一意な固定推定の平均が必要です")
    restored = [transform(estimates[name], name) for name in names]
    if len({values.shape for values in restored}) != 1:
        raise ValueError("推定の共通時間軸が一致しません")
    return np.mean(restored, axis=0, dtype=np.float64)


def implementation() -> dict:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "scripts/probe_instrumental_model.py", ROOT / "scripts/probe_instrumental_repeat.py",
        ROOT / "lss/validation/instrumental_quality.py", ROOT / "lss/validation/stereo_quality.py",
        ROOT / "lss/validation/stereo_comparison.py",
    )}


def freeze(baseline_path: Path, current_path: Path, output: Path) -> Path:
    baseline, artifacts = baseline_inputs(baseline_path)
    current, current_plan = read_json(current_path), read_json(current_path.parent / "plan.json")
    if (current.get("status") != "complete_development_diagnostic" or current.get("promotion_allowed") is not False
            or current.get("holdout_consumed") is not False
            or file_sha256(current_path.parent / "plan.json") != current["plan_sha256"]):
        raise ValueError("現行runtimeの完了Development診断が必要です")
    by_id = {case["case_id"]: case for case in current["cases"]}
    definitions = {case["case_id"]: case for case in read_json(baseline_path.parent / "plan.json")["cases"]}
    if len(by_id) != len(current["cases"]):
        raise ValueError("現行cacheのケースが重複しています")
    artifacts += [{"path": str(current_path.resolve()), "sha256": file_sha256(current_path)},
                  {"path": str((current_path.parent / "plan.json").resolve()),
                   "sha256": file_sha256(current_path.parent / "plan.json")}]
    with gpu_lease.acquire():
        runtime = infer.capture_msst_runtime().to_dict()
    cases = []
    for prior in baseline["cases"]:
        definition = definitions[prior["case_id"]]
        cached = by_id[prior["case_id"]]
        run_path = Path(cached["run_path"])
        if file_sha256(run_path) != cached["run_sha256"]:
            raise ValueError("現行cacheの完了runが変更されています")
        run = read_json(run_path)
        separation = run["separation"]
        source = definition["artifacts"]["input"]
        if (run["status"] != "completed_unannotated_episode_only_observation" or run["source"]["sha256"] != source["sha256"]
                or run["source"]["corpus_role"] != "development" or len(separation["artifacts"]) != 1
                or separation["runtime_revision"] != runtime or runtime["device"] != "cuda:0"):
            raise ValueError("同じ入力・現行CUDA runtimeの完了cacheが必要です")
        stage = separation["recipe_revision"]["stage_graph"][0]
        prior_stage = prior["result"]["recipe_revision"]["stage_graph"][0]
        if any(stage[key] != prior_stage[key] for key in ("model_id", "arch", "model_sha256", "config_sha256")):
            raise ValueError("比較元と同じモデル・設定だけを平均します")
        native = separation["artifacts"][0]
        tags = FLAC(native["path"])
        if (native["stem"] != "Instrumental" or tags["LSS_INPUT_SHA256"][0] != source["sha256"]
                or tags["LSS_RECIPE_ID"][0] != separation["recipe_id"]):
            raise ValueError("現行cacheの単独来歴が一致しません")
        artifacts += [native, {"path": str(run_path.resolve()), "sha256": cached["run_sha256"]}]
        cases.append({**definition, "native": native, "native_separation": separation})
    verify(artifacts)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(models_store.MODELS_TSV, output / "models.tsv")
    frozen = {"schema_version": 1, "status": "FROZEN_BEFORE_SYMMETRY_RESULTS", "promotion_allowed": False,
              "holdout_consumed": False, "cases": cases, "artifacts": artifacts, "runtime_revision": runtime,
              "transforms": TRANSFORMS, "variants": VARIANTS, "decision_rule": DECISION_RULE,
              "implementation": implementation(), "baseline_runtime_relation": "identical_current_cached_runtime",
              "limits": ["public_training_AAC_components_not_final_evaluation", "no_default_or_capability_promotion",
                         "reversible_transforms_are_not_a_semantic_presence_classifier"]}
    frozen["renderer_id"] = "instrumental-symmetry-development-v1-" + hashlib.sha256(json.dumps(
        {"implementation": frozen["implementation"], "transforms": TRANSFORMS, "variants": VARIANTS,
         "runtime_revision": runtime, "native_recipe_ids": [case["native_separation"]["recipe_id"] for case in cases]},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    plan = output / "comparison-plan.json"
    session._write_json_atomic(plan, frozen)
    return plan


def run(plan_path: Path) -> Path:
    plan = read_json(plan_path)
    if (plan["status"] != "FROZEN_BEFORE_SYMMETRY_RESULTS" or plan["promotion_allowed"] is not False
            or plan["holdout_consumed"] is not False or plan["implementation"] != implementation()
            or plan["decision_rule"] != DECISION_RULE
            or plan["variants"] != {key: list(value) for key, value in VARIANTS.items()}):
        raise ValueError("固定したDevelopment対称変換の構成が必要です")
    plan_hash = file_sha256(plan_path)
    verify(plan["artifacts"])
    models_store.MODELS_TSV = plan_path.parent / "models.tsv"
    os.environ["LSS_STATE_DIR"] = str(plan_path.parent / "state")
    cases, generated = [], []
    for case in plan["cases"]:
        artifacts = case["artifacts"]
        raw, rate = sf.read(artifacts["input"]["path"], dtype="float64", always_2d=True)
        target, target_rate = sf.read(artifacts["target"]["path"], dtype="float64", always_2d=True)
        vocals, vocals_rate = sf.read(artifacts["vocals"]["path"], dtype="float64", always_2d=True)
        native, native_rate = sf.read(case["native"]["path"], dtype="float64", always_2d=True)
        if {rate, target_rate, vocals_rate, native_rate} != {44100} or len(native) != case["frames"]:
            raise ValueError("比較元の時間軸が一致しません")
        native = native.T * 2.0 ** int(FLAC(case["native"]["path"])["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        estimates = {"identity": native}
        before = instrumental_quality(native, target.T, vocals.T)
        stage = case["native_separation"]["recipe_revision"]["stage_graph"][0]
        records = []
        for name in TRANSFORMS[1:]:
            input_path = plan_path.parent / "inputs" / f"{case['case_id']}-{name}.wav"
            input_path.parent.mkdir(exist_ok=True)
            sf.write(input_path, transform(raw.T, name).T, rate, subtype="FLOAT")
            result = worker.handle_job({"mix_path": str(input_path), "model_id": stage["model_id"], "stem_target": "Instrumental"},
                                      destination_dir=plan_path.parent / "outputs" / case["case_id"] / name)
            if (result["status"] not in {"ok", "skipped"} or len(result["outputs"]) != 1
                    or result.get("runtime_id") != plan["runtime_revision"]["runtime_id"]):
                raise RuntimeError(f"対称変換のCUDA推論が固定条件と一致しません: {case['case_id']}/{name}")
            observed = result["recipe_revision"]["stage_graph"][0]
            if any(observed[key] != stage[key] for key in ("model_id", "arch", "model_sha256", "config_sha256")):
                raise ValueError("対称変換のモデル・設定が変更されています")
            path = Path(result["outputs"][0])
            values, observed_rate = sf.read(path, dtype="float64", always_2d=True)
            if observed_rate != rate or values.shape != raw.shape:
                raise ValueError("対称変換の共通時間軸が変更されています")
            estimates[name] = values.T * 2.0 ** int(FLAC(path)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
            record = {"transform": name, "result": result, "output_sha256": file_sha256(path)}
            records.append(record)
            generated.append({"path": str(path), "sha256": record["output_sha256"]})
        variants = []
        for name, members in VARIANTS.items():
            candidate = combine(estimates, members)
            quality = instrumental_quality(candidate, target.T, vocals.T)
            path = plan_path.parent / "combined" / f"{case['case_id']}-{name}.wav"
            path.parent.mkdir(exist_ok=True)
            sf.write(path, candidate.T, rate, subtype="DOUBLE")
            artifact = {"path": str(path.resolve()), "sha256": file_sha256(path)}
            generated.append(artifact)
            variants.append({"name": name, "quality": quality, "comparison": comparison(before, quality), "artifact": artifact})
        cases.append({"case_id": case["case_id"], "variant": case["variant"], "native_quality": before,
                      "transformations": records, "variants": variants})
        session._write_json_atomic(plan_path.parent / f"{case['case_id']}-result.json", cases[-1])
        print(f"対称変換比較完了: {case['case_id']}", flush=True)
    verify(plan["artifacts"] + generated)
    if file_sha256(plan_path) != plan_hash or implementation() != plan["implementation"]:
        raise ValueError("対称変換の比較中に構成・実装が変更されています")
    decisions = {name: summarize([{**case, "comparison": next(row["comparison"] for row in case["variants"] if row["name"] == name)}
                                 for case in cases]) for name in VARIANTS}
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {"schema_version": 1, "status": "complete_development_symmetry_comparison",
        "promotion_allowed": False, "holdout_consumed": False, "default_changed": False,
        "plan_sha256": plan_hash, "source_hashes_after_verified": True, "cases": cases, "decisions": decisions,
        "baseline_runtime_relation": plan["baseline_runtime_relation"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezer = commands.add_parser("freeze")
    freezer.add_argument("baseline", type=Path)
    freezer.add_argument("current", type=Path)
    freezer.add_argument("output", type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(freeze(args.baseline, args.current, args.output) if args.command == "freeze" else run(args.plan))


if __name__ == "__main__":
    main()
