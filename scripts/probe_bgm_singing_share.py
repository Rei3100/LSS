"""構成が分かっている固定ケースで、歌唱stemの取り分が歌唱の有無を分けるか比較する。

正解は人手ラベルではなく作り方から決まる。`validate_instrumental_controls.py` の
variant定義（伴奏gain, 歌gain）がそのまま歌唱の有無である。

    accompaniment-only (1.0, 0.0) 歌唱なし / singing-mix (1.0, 1.0) 歌唱あり
    vocals-only        (0.0, 1.0) 歌唱あり / quiet-accompaniment (0.1, 1.0) 歌唱あり

帯は原作品3families（nightowl, easy-tiger, velvet-curtain）の分布から決め、
拡張3families（devils-words, child, milk-cow-blues）を確認へ回す。既に全ケースの
値を見た後で帯を決めているため、これはDevelopment Evidenceであって昇格の証拠ではない。
"""
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

from lss.bgm.singing_share import SingingShareRecipe, classify, measure_files
from lss.harness.fetch import file_sha256
from lss.validation import reference_residual_episode as guarded

# 作り方から決まる歌唱の有無。variant名の末尾で一意に決まる。
CONSTRUCTED_SINGING = {
    "accompaniment-only": False, "singing-mix": True,
    "vocals-only": True, "quiet-accompaniment": True,
}
# 作り方から音楽が無いと分かるケース。歌唱の有無を言ってはいけない。
CONSTRUCTED_NO_MUSIC = ("dialogue-only-target-absent", "dialogue-sfx-target-absent",
                        "sfx-only-target-absent")
DEVELOPMENT_FAMILIES = ("nightowl", "easy-tiger", "velvet-curtain")
CONFIRMATION_FAMILIES = ("devils-words", "child", "milk-cow-blues")
RECIPE = SingingShareRecipe(maximum_absent_share=0.01, minimum_present_share=0.05,
                            minimum_music_dbfs=-70.0)
IMPLEMENTATION = ("lss/bgm/singing_share.py", "scripts/probe_bgm_singing_share.py")


def read(path: Path) -> dict:
    return json.loads(Path(path).read_bytes())


def art(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _labelled(case_id: str) -> tuple[str, str, bool] | None:
    for variant, singing in CONSTRUCTED_SINGING.items():
        if case_id.endswith("-" + variant):
            family = case_id[: -len(variant) - 1]
            if family in DEVELOPMENT_FAMILIES + CONFIRMATION_FAMILIES:
                return family, variant, singing
    return None


def freeze(inference_dir: Path, output: Path) -> Path:
    """cacheされた4stem出力と、作り方から決まる正解を固定する。"""
    inference_dir, output = inference_dir.resolve(), output.resolve()
    receipt = read(inference_dir / "receipt.json")
    if receipt.get("status") != "COMPLETED_CUDA_WITHOUT_FALLBACK":
        raise ValueError("完了したCUDA推論のcacheが必要です")
    cases, unlabelled = [], []
    for row in receipt["cases"]:
        stems = row["stems"]
        vocals, instrumental = Path(stems["music_vocals"]["path"]), Path(stems["music_instrumental"]["path"])
        for item, name in ((vocals, "music_vocals"), (instrumental, "music_instrumental")):
            if file_sha256(item) != stems[name]["sha256"]:
                raise ValueError(f"推論cacheの内容が記録と一致しません: {item}")
        entry = {"case_id": row["case_id"],
                 "music_vocals": art(vocals), "music_instrumental": art(instrumental)}
        if row["case_id"] in CONSTRUCTED_NO_MUSIC:
            cases.append({**entry, "family": "target-absent", "variant": row["case_id"],
                          "constructed_singing": None, "split": "no-music"})
            continue
        labelled = _labelled(row["case_id"])
        if labelled is None:
            unlabelled.append(entry)
            continue
        family, variant, singing = labelled
        cases.append({**entry, "family": family, "variant": variant,
                      "constructed_singing": singing,
                      "split": "development" if family in DEVELOPMENT_FAMILIES else "confirmation"})
    if len(cases) != 27:
        raise ValueError(f"構成が分かる27ケースが必要です: {len(cases)}")
    plan = {"schema_version": 1, "status": "FROZEN_BEFORE_RESULTS",
            "promotion_allowed": False, "holdout_consumed": False,
            "inference_receipt": art(inference_dir / "receipt.json"),
            "recipe": RECIPE.to_dict(), "recipe_id": RECIPE.recipe_id,
            "implementation": {name: file_sha256(ROOT / name) for name in IMPLEMENTATION},
            "cases": cases, "unlabelled_cases": unlabelled,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
            "limits": [
                "正解は作り方（伴奏gainと歌gain）から決まる。人手の試聴ラベルではない。",
                "音楽が無い3ケースは、取り分の比だけでは歌唱ありと誤読された。音楽の絶対量の"
                "下限を加えた後の挙動をここで固定する。",
                "帯は原作品3familiesの分布を見た後に決めた。確認3familiesも既に観測済みで、"
                "独立したholdoutではない。Development Evidenceとして扱う。",
                "歌唱stemはモデルの出力であり、意味的roleの正解ではない。",
                "実番組へ使うには同じ4stem推論が要る。ここでは既存cacheだけを読む。",
                "Holdout・外部送信・正式対応・既定変更なし。"]}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "plan.json"
    guarded._write_record(target, plan, lambda: verify(plan))
    return target


def verify(plan: dict) -> None:
    if (plan.get("status") != "FROZEN_BEFORE_RESULTS" or plan.get("promotion_allowed") is not False
            or plan.get("holdout_consumed") is not False
            or plan.get("recipe") != RECIPE.to_dict()):
        raise ValueError("固定した未昇格の歌唱取り分計画が必要です")
    for name, digest in plan["implementation"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"固定した実装が変更されています: {name}")
    for case in plan["cases"]:
        for key in ("music_vocals", "music_instrumental"):
            if file_sha256(Path(case[key]["path"])) != case[key]["sha256"]:
                raise ValueError(f"固定した推論cacheが変更されています: {case[key]['path']}")


def evaluate(plan_path: Path) -> Path:
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    verify(plan)
    rows = []
    for case in plan["cases"]:
        measured = measure_files(Path(case["music_vocals"]["path"]),
                                 Path(case["music_instrumental"]["path"]))
        decided = classify(measured, RECIPE)
        expected = ("unresolved" if case["constructed_singing"] is None
                    else "present" if case["constructed_singing"] else "absent")
        rows.append({**{key: case[key] for key in ("case_id", "family", "variant", "split")},
                     "constructed_singing": case["constructed_singing"],
                     "singing_share": measured["singing_share"],
                     "music_rms_dbfs": measured["music_rms_dbfs"],
                     "observed": decided["singing"], "reason": decided["reason"],
                     "expected": expected,
                     "agrees": decided["singing"] == expected,
                     "contradicts": decided["singing"] not in (expected, "unresolved")})
    unlabelled = []
    for case in plan["unlabelled_cases"]:
        measured = measure_files(Path(case["music_vocals"]["path"]),
                                 Path(case["music_instrumental"]["path"]))
        unlabelled.append({"case_id": case["case_id"], "singing_share": measured["singing_share"],
                           "observed": classify(measured, RECIPE)["singing"]})

    def summarise(split: str) -> dict:
        selected = [row for row in rows if row["split"] == split]
        absent = [row["singing_share"] for row in selected if not row["constructed_singing"]]
        present = [row["singing_share"] for row in selected if row["constructed_singing"]]
        return {"case_count": len(selected),
                "expected_unresolved": sum(1 for row in selected
                                           if row["constructed_singing"] is None),
                "agreed": sum(1 for row in selected if row["agrees"]),
                "unresolved": sum(1 for row in selected if row["observed"] == "unresolved"),
                "contradicted": sum(1 for row in selected if row["contradicts"]),
                "maximum_share_without_singing": max(absent, default=None),
                "minimum_share_with_singing": min(present, default=None)}

    report = {"schema_version": 1, "status": "COMPLETED_CONSTRUCTED_SINGING_SHARE_COMPARISON",
              "promotion_allowed": False, "holdout_consumed": False,
              "plan": art(plan_path), "recipe_id": RECIPE.recipe_id, "recipe": RECIPE.to_dict(),
              "cases": rows, "unlabelled_cases": unlabelled,
                    "development": summarise("development"), "confirmation": summarise("confirmation"),
              "no_music": summarise("no-music"),
              "contradicted_case_ids": [row["case_id"] for row in rows if row["contradicts"]],
              "limits": plan["limits"]}
    target = plan_path.parent / "report.json"
    guarded._write_record(target, report, lambda: verify(plan))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--inference-dir", type=Path,
                          default=ROOT / ".providers/facing-the-music/controls-r2")
    freezing.add_argument("--out-dir", type=Path, required=True)
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.inference_dir, args.out_dir) if args.command == "freeze"
          else evaluate(args.plan), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
