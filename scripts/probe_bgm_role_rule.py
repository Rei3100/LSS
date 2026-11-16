"""作り方が分かるcontrolled corpusで、最初の意味role規則を固定比較する。

区間はコーパスが与える（L1は前提）。ここで試すのは区間に対する役割の当て方だけである。
規則は結果を見る前に固定する。

    歌唱ありと観測 → `singing_score`
    歌唱なしと観測 → `dramatic_score`
    どちらとも言えない → 判断しない（L4の選別へ出さない）

`source_music` と `musical_sfx` は、この規則では区別できない。区別できないことを
隠さず、誤りとして測る。ここでは何も公開せず、既定も変えない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from lss.bgm.role_policy import RolePolicy, decide_role
from lss.bgm.singing_share import SingingShareRecipe, classify, measure
from lss.harness.fetch import file_sha256
from lss.validation import facing_music, reference_residual_episode as guarded
from lss.validation.bandit_inference_cache import validate_plan
from lss.validation.bgm_corpus import load_bgm_corpus
from lss.validation.occurrence_metrics import (
    detection_from_dict, evaluate_occurrences, summarize_occurrence_results,
)
from scripts import probe_bgm_facing_music as facing

# 固定27ケースで矛盾0件だった帯。corpusを見てから決め直さない。
RECIPE = SingingShareRecipe(maximum_absent_share=0.01, minimum_present_share=0.05,
                            minimum_music_dbfs=-70.0)
# 通常実行と同じ採用policy本体を使う。Development専用の別実装を置かない。
ROLE_POLICY = RolePolicy(singing_present_role="singing_score",
                         singing_absent_role="dramatic_score",
                         unresolved_role="unresolved_music_like",
                         indistinguishable_roles=("source_music", "musical_sfx"))
ROLE_RULE = {"present": "singing_score", "absent": "dramatic_score", "unresolved": None}
DEFAULT_MANIFEST = dev_state_root(ROOT) / "validation/lv-bgm-002-controlled-r2/manifest.jsonl"


def freeze(manifest: Path, output: Path) -> Path:
    """corpusの音源を4stem推論の固定ケースにし、規則と区間を先に固定する。"""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    corpus = load_bgm_corpus(manifest, verify_sources=True)
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
                  (manifest, Path(__file__).resolve(), ROOT / "lss/bgm/singing_share.py",
                   ROOT / "lss/bgm/role_policy.py")})
    cases, references = [], []
    for case in corpus.cases:
        if case.sample_rate != 44100:
            raise ValueError(f"44.1kHzのケースが必要です: {case.case_id}")
        files[str(case.source_path)] = case.source_sha256
        cases.append({"case_id": case.case_id, "stems": {},
                      "source": {"path": str(case.source_path), "sha256": case.source_sha256,
                                 "frames": case.frames}})
        references.append({"case_id": case.case_id, "expected_outcome": case.expected_outcome,
                           "occurrences": [{"occurrence_id": item.occurrence_id,
                                            "start_frame": item.start_frame,
                                            "end_frame": item.end_frame,
                                            "semantic_role": item.semantic_role,
                                            "wanted": item.wanted} for item in case.occurrences]})
    plan = {"schema_version": 1, "status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False,
            "holdout_consumed": False, "sample_rate": 44100,
            "output_roles": list(facing_music.ROLES), "cases": cases,
            "provider_root": str(provider), "provider_revision": facing.QUERY_REVISION,
            "checkpoint": facing.art(checkpoint), "provider_and_inputs": files,
            "model_config": facing_music.MODEL_CONFIG, "fader": facing.FADER,
            "fader_source": facing.art(fader_path),
            "corpus": {"path": str(manifest), "sha256": file_sha256(manifest),
                       "corpus_id": corpus.corpus_id, "revision": corpus.revision},
            "singing_recipe": RECIPE.to_dict(), "singing_recipe_id": RECIPE.recipe_id,
            "role_rule": {name: value for name, value in ROLE_RULE.items()},
            "role_policy_id": ROLE_POLICY.policy_id, "role_policy": ROLE_POLICY.to_dict(),
            "references": references,
            "limits": [
                "区間はcorpusが与える。ここで測るのは区間に対する役割の当て方だけである。",
                "規則は結果を見る前に固定した。source_musicとmusical_sfxは区別できない。",
                "作り方が分かるcontrolled mixtureであり、実番組・独立Holdoutへ一般化しない。",
                "歌唱stemはモデルの出力であり、意味的roleの正解ではない。",
                "公開・既定変更・Holdout消費なし。"]}
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "plan.json"
    guarded._write_record(target, plan, lambda: facing.verify(plan))
    return target


def _span_share(stems: dict, start: int, end: int) -> dict:
    def crop(name: str) -> np.ndarray:
        with sf.SoundFile(stems[name]["path"]) as stream:
            stream.seek(start)
            block = stream.read(end - start, dtype="float64", always_2d=True)
        return block.T
    return measure(crop("music_vocals"), crop("music_instrumental"))


def evaluate(plan_path: Path, inference_dir: Path) -> Path:
    """区間ごとに歌唱を観測し、固定した規則で役割を当てて選別を測る。"""
    plan_path, inference_dir = Path(plan_path).resolve(), Path(inference_dir).resolve()
    plan = facing.read(plan_path)
    facing.verify(plan)
    receipt = facing.read(inference_dir / "receipt.json")
    if (receipt.get("status") != "COMPLETED_CUDA_WITHOUT_FALLBACK"
            or receipt.get("plan_sha256") != file_sha256(plan_path)):
        raise ValueError("この固定計画の完了したCUDA推論が必要です")
    corpus = load_bgm_corpus(Path(plan["corpus"]["path"]), verify_sources=True)
    stems = {case["case_id"]: case["stems"] for case in receipt["cases"]}
    for case in receipt["cases"]:
        for name, item in case["stems"].items():
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"推論結果が記録と一致しません: {item['path']}")
    rows, detections, undecided = [], {}, []
    for reference in plan["references"]:
        selected = []
        for occurrence in reference["occurrences"]:
            measured = _span_share(stems[reference["case_id"]], occurrence["start_frame"],
                                   occurrence["end_frame"])
            observed = classify(measured, RECIPE)
            derived = decide_role({"singing": observed["singing"]}, ROLE_POLICY)
            predicted = (None if derived["semantic_role"] == ROLE_POLICY.unresolved_role
                         else derived["semantic_role"])
            if predicted != ROLE_RULE[observed["singing"]]:
                raise ValueError("採用policyが固定した規則と一致しません")
            rows.append({"case_id": reference["case_id"],
                         "occurrence_id": occurrence["occurrence_id"],
                         "reference_role": occurrence["semantic_role"],
                         "reference_wanted": occurrence["wanted"],
                         "singing": observed["singing"], "reason": observed["reason"],
                         "singing_share": measured["singing_share"],
                         "music_rms_dbfs": measured["music_rms_dbfs"],
                         "predicted_role": predicted,
                         "agrees": predicted == occurrence["semantic_role"]})
            if predicted is None:
                undecided.append(occurrence["occurrence_id"])
                continue
            selected.append({"candidate_id": occurrence["occurrence_id"],
                             "start_frame": occurrence["start_frame"],
                             "end_frame": occurrence["end_frame"],
                             "semantic_role": predicted, "identity_state": "unresolved",
                             "cue_identity_id": None, "audio_version_id": None})
        detections[reference["case_id"]] = selected
    results = []
    for case in corpus.cases:
        found = [detection_from_dict(item, source_frames=case.frames)
                 for item in detections.get(case.case_id, [])]
        results.append(evaluate_occurrences(case, found))
    summary = summarize_occurrence_results(corpus, results)
    confusion = {}
    for row in rows:
        key = f"{row['reference_role']}->{row['predicted_role'] or 'undecided'}"
        confusion[key] = confusion.get(key, 0) + 1
    report = {"schema_version": 1, "status": "COMPLETED_FIXED_ROLE_RULE_COMPARISON",
              "promotion_allowed": False, "holdout_consumed": False, "anything_published": False,
              "plan": facing.art(plan_path),
              "inference_receipt": facing.art(inference_dir / "receipt.json"),
              "role_rule": plan["role_rule"], "singing_recipe_id": RECIPE.recipe_id,
              "occurrence_count": len(rows),
              "agreed": sum(1 for row in rows if row["agrees"]),
              "undecided": len(undecided), "undecided_occurrences": undecided,
              "confusion": dict(sorted(confusion.items())), "occurrences": rows,
              "selection_summary": summary, "case_results": results,
              "limits": plan["limits"]}
    target = plan_path.parent / "report.json"
    guarded._write_record(target, report, lambda: facing.verify(plan))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    freezing.add_argument("--out-dir", type=Path, required=True)
    running = commands.add_parser("run")
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--out-dir", type=Path, required=True)
    running.add_argument("--resume", action="store_true")
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--plan", type=Path, required=True)
    evaluating.add_argument("--inference-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.manifest, args.out_dir)
    elif args.command == "run":
        result = facing.run(args.plan, args.out_dir, args.resume, None)
    else:
        result = evaluate(args.plan, args.inference_dir)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
