"""声のみの混入を増やさない親推定保持の平均を、固定Developmentケースで比較する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from mutagen.flac import FLAC  # noqa: E402
from lss.core.session import _write_json_atomic  # noqa: E402
from lss.harness.fetch import file_sha256  # noqa: E402
from lss.validation.instrumental_quality import instrumental_quality  # noqa: E402
from scripts.probe_instrumental_model import read_json, verify, summarize  # noqa: E402
from scripts.probe_instrumental_repeat import DECISION_RULE, comparison  # noqa: E402

WEIGHTS = (0.25, 0.5, 0.75)
MAXIMUM_PARENT_PROJECTION_ERROR_DB = 0.1


def guarded_mean(parent: np.ndarray, alternate: np.ndarray, weight: float) -> tuple[np.ndarray, dict]:
    """energyと親への投影は選択の診断。正規化・利得補正や対象不在の断定をしない。"""
    if np.iscomplexobj(parent) or np.iscomplexobj(alternate):
        raise ValueError("実数のステレオ音声が必要です")
    parent, alternate = np.asarray(parent, dtype=np.float64), np.asarray(alternate, dtype=np.float64)
    if (parent.ndim != 2 or parent.shape[0] != 2 or parent.shape[1] == 0 or parent.shape != alternate.shape
            or not np.isfinite(parent).all() or not np.isfinite(alternate).all()
            or isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight <= 1):
        raise ValueError("同じ有限ステレオと0より大きく1以下の固定weightが必要です")
    candidate = (1 - weight) * parent + weight * alternate
    parent_energy, candidate_energy = float(np.sum(parent ** 2)), float(np.sum(candidate ** 2))
    gain = float(np.sum(candidate * parent)) / parent_energy if parent_energy else None
    gain_db = float(20 * np.log10(gain)) if gain is not None and gain > 0 else None
    if not np.isfinite(parent_energy) or not np.isfinite(candidate_energy) or not np.isfinite(candidate).all():
        raise ValueError("平均の計算が有限範囲を超えています")
    energy_ok = candidate_energy <= parent_energy
    projection_ok = gain_db is not None and abs(gain_db) <= MAXIMUM_PARENT_PROJECTION_ERROR_DB
    enabled = energy_ok and projection_ok
    return (candidate if enabled else parent.copy()), {
        "weight": weight, "parent_energy": parent_energy, "candidate_energy": candidate_energy,
        "candidate_projection_to_parent_db": gain_db, "energy_increase_veto": not energy_ok,
        "parent_projection_veto": not projection_ok, "enabled": enabled,
        "semantic_presence": "unresolved",
    }


def implementation() -> dict:
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT / "scripts/probe_instrumental_model.py", ROOT / "scripts/probe_instrumental_repeat.py",
        ROOT / "lss/validation/instrumental_quality.py", ROOT / "lss/validation/stereo_quality.py",
        ROOT / "lss/validation/stereo_comparison.py",
    )}


def freeze(native_path: Path, alternate_paths: list[Path], output: Path) -> Path:
    native = read_json(native_path)
    native_plan_path = native_path.parent / "comparison-plan.json"
    native_plan = read_json(native_plan_path)
    if (native["status"] != "complete_development_symmetry_comparison" or native["promotion_allowed"] is not False
            or native["holdout_consumed"] is not False or file_sha256(native_plan_path) != native["plan_sha256"]):
        raise ValueError("現行CUDAの同一入力cacheを固定したDevelopment記録が必要です")
    if not alternate_paths:
        raise ValueError("固定した別候補の比較記録が必要です")
    native_ids = [case["case_id"] for case in native_plan["cases"]]
    if not native_ids or len(set(native_ids)) != len(native_ids):
        raise ValueError("一意なnative cacheが必要です")
    artifacts = [{"path": str(path.resolve()), "sha256": file_sha256(path)}
                 for path in (native_path, native_plan_path)] + native_plan["artifacts"]
    alternates = []
    for path in alternate_paths:
        report, plan_path = read_json(path), path.parent / "comparison-plan.json"
        plan = read_json(plan_path)
        ids = [case["case_id"] for case in report["cases"]]
        if (report["status"] != "complete_development_model_comparison" or report["promotion_allowed"] is not False
                or report["holdout_consumed"] is not False or file_sha256(plan_path) != report["plan_sha256"]
                or len(ids) != len(native_ids) or set(ids) != set(native_ids)
                or plan["runtime_revision"] != native_plan["runtime_revision"]):
            raise ValueError("同じ全入力・CUDA runtimeの別候補だけを平均します")
        artifacts += [{"path": str(p.resolve()), "sha256": file_sha256(p)} for p in (path, plan_path)] + plan["artifacts"]
        records = []
        for case in report["cases"]:
            artifact = {"path": case["result"]["outputs"][0], "sha256": case["output_sha256"]}
            source = next(record for record in native_plan["cases"] if record["case_id"] == case["case_id"])["artifacts"]["input"]
            if case["result"]["input_sha256"] != source["sha256"]:
                raise ValueError("別候補の実入力がnative cacheと一致しません")
            records.append({"case_id": case["case_id"], "artifact": artifact})
            artifacts.append(artifact)
        alternates.append({"model_id": plan["candidate"]["model_id"], "cases": records})
    if len({item["model_id"] for item in alternates}) != len(alternates):
        raise ValueError("別候補のモデルが重複しています")
    verify(artifacts)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "plan.json"
    _write_json_atomic(path, {"status": "FROZEN_BEFORE_GUARDED_ENSEMBLE_RESULTS", "promotion_allowed": False,
        "holdout_consumed": False, "cases": native_plan["cases"], "alternates": alternates, "artifacts": artifacts,
        "weights": WEIGHTS, "maximum_parent_projection_error_db": MAXIMUM_PARENT_PROJECTION_ERROR_DB,
        "decision_rule": DECISION_RULE, "implementation": implementation(),
        "limits": ["public_training_AAC_components_not_final_evaluation", "same_runtime_cached_comparison",
                   "energy_and_parent_projection_not_semantic_presence_or_target_fidelity_proof",
                   "no_normalization_gain_fit_or_default_promotion"]})
    return path


def audio(artifact: dict, frames: int) -> np.ndarray:
    path = Path(artifact["path"])
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    if rate != 44100 or values.shape != (frames, 2):
        raise ValueError("平均するcacheのstereo/timelineが一致しません")
    if path.suffix.casefold() == ".flac":
        values *= 2.0 ** int(FLAC(path)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
    return values.T


def run(plan_path: Path) -> Path:
    plan = read_json(plan_path)
    if (plan["status"] != "FROZEN_BEFORE_GUARDED_ENSEMBLE_RESULTS" or plan["promotion_allowed"] is not False
            or plan["holdout_consumed"] is not False or plan["weights"] != list(WEIGHTS)
            or plan["maximum_parent_projection_error_db"] != MAXIMUM_PARENT_PROJECTION_ERROR_DB
            or plan["decision_rule"] != DECISION_RULE or plan["implementation"] != implementation()):
        raise ValueError("固定したDevelopment平均の構成が必要です")
    plan_hash = file_sha256(plan_path)
    verify(plan["artifacts"])
    candidates, generated = [], []
    for definition in plan["cases"]:
        frames = definition["frames"]
        parent = audio(definition["native"], frames)
        target, vocals = (audio(definition["artifacts"][role], frames) for role in ("target", "vocals"))
        before = instrumental_quality(parent, target, vocals)
        variants = []
        for model in plan["alternates"]:
            artifact = next(case["artifact"] for case in model["cases"] if case["case_id"] == definition["case_id"])
            alternate = audio(artifact, frames)
            for weight in WEIGHTS:
                candidate, trace = guarded_mean(parent, alternate, weight)
                quality = instrumental_quality(candidate, target, vocals)
                path = plan_path.parent / "outputs" / model["model_id"] / f"{definition['case_id']}-{weight}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, candidate.T, 44100, subtype="DOUBLE")
                record = {"path": str(path.resolve()), "sha256": file_sha256(path)}
                generated.append(record)
                variants.append({"model_id": model["model_id"], "weight": weight, "trace": trace, "quality": quality,
                                 "comparison": comparison(before, quality), "artifact": record})
        candidates.append({"case_id": definition["case_id"], "variant": definition["variant"], "variants": variants})
        print(f"親保持の平均比較完了: {definition['case_id']}", flush=True)
    verify(plan["artifacts"] + generated)
    if file_sha256(plan_path) != plan_hash or implementation() != plan["implementation"]:
        raise ValueError("平均の比較中に構成・実装が変更されています")
    decisions = []
    for model in plan["alternates"]:
        for weight in WEIGHTS:
            rows = [{**case, **next(row for row in case["variants"] if row["model_id"] == model["model_id"] and row["weight"] == weight)}
                    for case in candidates]
            decisions.append({"model_id": model["model_id"], "weight": weight,
                              "enabled_cases": [case["case_id"] for case in rows if case["trace"]["enabled"]],
                              **summarize(rows)})
    report = plan_path.parent / "report.json"
    _write_json_atomic(report, {"status": "complete_development_guarded_ensemble_comparison", "promotion_allowed": False,
        "holdout_consumed": False, "default_changed": False, "source_hashes_after_verified": True,
        "plan_sha256": plan_hash, "cases": candidates, "decisions": decisions, "limits": plan["limits"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezer = commands.add_parser("freeze")
    freezer.add_argument("native", type=Path)
    freezer.add_argument("output", type=Path)
    freezer.add_argument("--alternate", action="append", type=Path, required=True)
    runner = commands.add_parser("run")
    runner.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(freeze(args.native, args.alternate, args.output) if args.command == "freeze" else run(args.plan))


if __name__ == "__main__":
    main()
