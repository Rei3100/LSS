"""実番組の保留候補へ4stem推論をかけ、cueごとの歌唱の有無を観測する。

固定ケースで矛盾0件だった歌唱取り分の観測を、実際の番組のcueへ適用する。
観測であって意味的roleではない。ここでは何も公開せず、既存の候補音声・保留結果・
既定を変更しない。対応領域の判断は、観測を渡した読み直しとして別に記録する。

    python scripts/probe_bgm_episode_singing.py freeze --record <application-outcome.json> --out-dir DIR
    python scripts/probe_bgm_episode_singing.py run --plan DIR/plan.json --out-dir INFER
    python scripts/probe_bgm_episode_singing.py evaluate --plan DIR/plan.json --inference-dir INFER
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf

from lss.bgm.singing_share import SingingShareRecipe, classify, measure_files
from lss.harness.fetch import file_sha256
from lss.validation import facing_music, reference_residual_episode as guarded
from lss.validation.bandit_inference_cache import validate_plan
from scripts import probe_bgm_facing_music as facing

# 固定27ケースで矛盾0件だった帯をそのまま使う。実番組を見てから決め直さない。
RECIPE = SingingShareRecipe(maximum_absent_share=0.01, minimum_present_share=0.05,
                            minimum_music_dbfs=-70.0)


def freeze(record_path: Path, output: Path) -> Path:
    """保留結果の候補音声を、そのまま4stem推論の固定ケースにする。"""
    from lss import application

    record_path, output = Path(record_path).resolve(), Path(output).resolve()
    outcome = application.read_bgm_development_outcome(record_path)
    if outcome.disposition is not application.ResultDisposition.HELD:
        raise ValueError(f"読み戻せる保留結果が必要です: {outcome.message}")
    report_path = Path(outcome.evidence_path)
    report = facing.read(report_path)
    provider = ROOT / ".providers/facing-the-music/query-bandit"
    checkpoint = ROOT / ".providers/facing-the-music/bandit-split.ckpt"
    if subprocess.check_output(["git", "-C", str(provider), "rev-parse", "HEAD"],
                               text=True).strip() != facing.QUERY_REVISION:
        raise ValueError("固定する公開実装revisionが一致しません")
    if file_sha256(checkpoint) != facing.CHECKPOINT_SHA256:
        raise ValueError("公式checksum確認済みの4stem重みが必要です")
    fader_path = ROOT / ".providers/bandit/core/utils/audio.py"
    files = {str(p.resolve()): file_sha256(p) for p in (provider / "core").rglob("*.py")}
    files.update({str(p.resolve()): file_sha256(p) for p in (provider / "LICENSE", fader_path, checkpoint)})
    files.update(facing.implementation())
    files.update({str(path): file_sha256(path) for path in
                  (record_path, report_path, Path(__file__).resolve(),
                   ROOT / "lss/bgm/singing_share.py")})
    cases = []
    for candidate, place in zip(report["candidates"], outcome.held_artifacts):
        asset = candidate["asset"]
        path = Path(place).resolve()
        info = sf.info(str(path))
        if info.channels != 2 or info.samplerate != 44100:
            raise ValueError(f"44.1kHz stereoの候補音声が必要です: {path}")
        files[str(path)] = file_sha256(path)
        cases.append({"case_id": asset["candidate_asset_id"], "stems": {},
                      "source": {"path": str(path), "sha256": file_sha256(path),
                                 "frames": int(info.frames)}})
    if not cases:
        raise ValueError("保留候補がありません")
    plan = {"schema_version": 1, "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False,
            "holdout_consumed": False, "sample_rate": 44100,
            "output_roles": list(facing_music.ROLES), "cases": cases,
            "provider_root": str(provider), "provider_revision": facing.QUERY_REVISION,
            "checkpoint": facing.art(checkpoint), "provider_and_inputs": files,
            "model_config": facing_music.MODEL_CONFIG, "fader": facing.FADER,
            "fader_source": facing.art(fader_path),
            "held_result": facing.art(record_path), "run": facing.art(report_path),
            "singing_recipe": RECIPE.to_dict(), "singing_recipe_id": RECIPE.recipe_id,
            "limits": [
                "歌唱stemの取り分は観測であって意味的roleではない。",
                "帯は作り方が分かる固定27ケースで決めたもの。実番組を見てから決め直さない。",
                "実番組には正解が無い。ここで測るのは観測であり、正しさの証明ではない。",
                "候補音声・保留結果・既定を変更しない。公開もしない。",
                "重みは非商用Development研究の範囲。製品への採用を判断しない。",
                "Holdout・外部音声送信・CPU推論fallbackなし。"]}
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "plan.json"
    guarded._write_record(target, plan, lambda: facing.verify(plan))
    return target


def evaluate(plan_path: Path, inference_dir: Path) -> Path:
    """推論結果から歌唱の取り分を測り、対応領域の読み直しまで記録する。"""
    from lss import application
    from lss.bgm.cue_readiness import CueReadinessRecipe
    from lss.bgm.supported_scope import SupportedScope

    plan_path, inference_dir = Path(plan_path).resolve(), Path(inference_dir).resolve()
    plan = facing.read(plan_path)
    facing.verify(plan)
    receipt = facing.read(inference_dir / "receipt.json")
    if receipt.get("status") != "COMPLETED_CUDA_WITHOUT_FALLBACK":
        raise ValueError("完了したCUDA推論が必要です")
    if receipt.get("plan_sha256") != file_sha256(plan_path):
        raise ValueError("この固定計画の推論結果が必要です")
    observations, rows = {}, []
    for case in receipt["cases"]:
        stems = case["stems"]
        for name in ("music_vocals", "music_instrumental"):
            if file_sha256(Path(stems[name]["path"])) != stems[name]["sha256"]:
                raise ValueError(f"推論結果が記録と一致しません: {stems[name]['path']}")
        measured = measure_files(Path(stems["music_vocals"]["path"]),
                                 Path(stems["music_instrumental"]["path"]))
        decided = classify(measured, RECIPE)
        observations[case["case_id"]] = {"singing": decided["singing"]}
        rows.append({"case_id": case["case_id"], "singing": decided["singing"],
                     "reason": decided["reason"], "singing_share": measured["singing_share"],
                     "music_rms_dbfs": measured["music_rms_dbfs"]})
    scope = SupportedScope(
        source_domains=("finished_stereo_episode",), published_semantic_roles=("dramatic_score",),
        covered_semantic_roles=("dramatic_score", "singing_score"),
        minimum_cue_seconds=20.0, maximum_cue_seconds=600.0,
        required_evidence=("source_provenance", "occurrence_span", "recording_identity"),
        quality_gates={"readiness_passed": 1.0},
        singing_observation_policy="exclude_from_publication",
        derived_role_policy="accept_declared_ambiguity",
        cue_extent_policy="require_single_cue_evidence", publication_allowed=False)
    readiness = CueReadinessRecipe(required_channels=2, minimum_loudness_lufs=-45.0,
                                   maximum_true_peak_dbtp=-1.0, maximum_clipped_samples=0,
                                   minimum_spectral_edge_hz=15000.0, spectral_edge_drop_db=60.0,
                                   minimum_side_to_mid_db=-40.0)
    review = application.review_bgm_scope_coverage(
        Path(plan["held_result"]["path"]), scope, source_domain="finished_stereo_episode",
        readiness=readiness, observations=observations)
    counts = {}
    for row in rows:
        counts[row["singing"]] = counts.get(row["singing"], 0) + 1
    report = {"schema_version": 1, "status": "COMPLETED_EPISODE_SINGING_OBSERVATION",
              "promotion_allowed": False, "holdout_consumed": False, "anything_published": False,
              "plan": facing.art(plan_path), "inference_receipt": facing.art(inference_dir / "receipt.json"),
              "singing_recipe": RECIPE.to_dict(), "singing_recipe_id": RECIPE.recipe_id,
              "cue_count": len(rows), "singing_counts": counts, "cues": rows,
              "scope_review": {key: review[key] for key in
                               ("status", "disposition", "reason", "counts", "scope_id",
                                "cue_readiness_recipe_id", "observed_candidate_count")},
              "scope_reasons": {row["reason"]: sum(1 for other in review["candidates"]
                                                   if other["reason"] == row["reason"])
                                for row in review["candidates"]},
              "limits": plan["limits"]}
    target = plan_path.parent / "report.json"
    guarded._write_record(target, report, lambda: facing.verify(plan))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--record", type=Path, required=True)
    freezing.add_argument("--out-dir", type=Path, required=True)
    running = commands.add_parser("run")
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--out-dir", type=Path, required=True)
    running.add_argument("--resume", action="store_true")
    running.add_argument("--stop-after", type=int)
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--plan", type=Path, required=True)
    evaluating.add_argument("--inference-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.record, args.out_dir)
    elif args.command == "run":
        result = facing.run(args.plan, args.out_dir, args.resume, args.stop_after)
    else:
        result = evaluate(args.plan, args.inference_dir)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
