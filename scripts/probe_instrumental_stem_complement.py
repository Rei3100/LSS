"""既定の分離が、伴奏と歌唱で原本をどれだけ覆っているかを測る。

取りこぼしたレベルの読み取りは、原本から伴奏を引いた残りを使う。もし歌唱の推定と伴奏の
推定を足すと原本に戻るなら、その残りは歌唱の推定そのものであり、「歌唱を先に引く」案は
同じ計算になる。まずそこを確かめる。

同じ固定ケースへ歌唱側の分離を実行する。CUDAのみで、fallbackは失敗にする。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
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

from lss.core import catalog, session, worker  # noqa: E402
from lss.core import models_store  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
DECISION_RULE = {
    "this_measures_coverage_not_quality": True,
    "the_reference_is_not_used_here_at_all": True,
    "a_complementary_pair_means_the_residual_is_the_other_stem": True,
}
LIMITS = [
    "runs_the_declared_default_separation_again_for_the_other_stem",
    "public_training_material_and_constructed_linear_mixtures_only",
    "coverage_is_not_separation_quality",
    "no_publication_default_change_or_holdout_use",
]


class StemComplementProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _read(path: Path) -> np.ndarray:
    values, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return values.T


def run_probe(plan_path: Path, report_path: Path, output: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if plan.get("promotion_allowed") is not False or report.get("promotion_allowed") is not False:
        raise StemComplementProbeError("固定したDevelopment診断だけを読み直します")
    if report.get("plan_sha256") != fetch.file_sha256(plan_path):
        raise StemComplementProbeError("この固定計画の結果ではありません")
    cases = {case["case_id"]: case for case in plan["cases"]}
    before = {path: fetch.file_sha256(path) for path in (plan_path, report_path)}
    output.mkdir(parents=True, exist_ok=False)

    ledger = output / "models.tsv"
    shutil.copyfile(models_store.MODELS_TSV, ledger)
    models_store.MODELS_TSV = ledger
    os.environ["LSS_STATE_DIR"] = str(output / "state")
    os.environ["LSS_MIN_FREE_BYTES"] = "0"
    default = catalog.default_model()

    rows = []
    for item in report["cases"]:
        case = cases[item["case_id"]]
        source = Path(case["artifacts"]["input"]["path"])
        if fetch.file_sha256(source) != case["artifacts"]["input"]["sha256"]:
            raise StemComplementProbeError(f"固定入力が変更されています: {source}")
        [produced] = item["result"]["outputs"]
        if fetch.file_sha256(Path(produced)) != item["output_sha256"]:
            raise StemComplementProbeError(f"固定出力が変更されています: {produced}")

        result = worker.handle_job(
            {"mix_path": str(source), "model_id": default["model_id"], "stem_target": "Vocals"},
            destination_dir=output / "outputs" / item["case_id"])
        if result["status"] not in {"ok", "skipped"} or len(result["outputs"]) != 1:
            raise StemComplementProbeError(f"歌唱側の分離に失敗しました: {item['case_id']}")
        saved = Path(result["outputs"][0])
        shift = int(FLAC(saved)["LSS_OUTPUT_HEADROOM_SHIFT"][0])

        mixture = _read(source)
        instrumental = _read(Path(produced)) * 2.0 ** item["serialization_shift"]
        vocals = _read(saved) * 2.0 ** shift
        frames = min(mixture.shape[1], instrumental.shape[1], vocals.shape[1])
        mixture = mixture[:, :frames]
        instrumental, vocals = instrumental[:, :frames], vocals[:, :frames]
        residual = mixture - instrumental - vocals

        def energy(value: np.ndarray) -> float:
            return float(np.sum(value ** 2))

        total = energy(mixture)
        share = energy(residual) / total if total > 0 else None
        rows.append({
            "case_id": item["case_id"], "variant": item["variant"],
            "vocals_output_sha256": fetch.file_sha256(saved),
            "vocals_serialization_shift": shift,
            "mixture_energy": total,
            "uncovered_share": share,
            "uncovered_db": (None if not share or share <= 0
                             else 10.0 * float(np.log10(share))),
            "residual_to_vocals_share":
                (energy(residual) / energy(vocals)) if energy(vocals) > 0 else None,
        })
        print(f"{item['case_id']}: 覆えていない割合 {share}", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise StemComplementProbeError(f"固定入力が変更されています: {path}")

    shares = [row["uncovered_share"] for row in rows if row["uncovered_share"] is not None]
    result_path = output / "report.json"
    session._write_json_atomic(result_path, {
        "schema_version": SCHEMA_VERSION,
        "type": "instrumental_stem_complement_report",
        "status": "complete_development_measurement",
        "role": "development_measurement_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "model_id": default["model_id"],
        "decision_rule": DECISION_RULE,
        "case_count": len(rows),
        "maximum_uncovered_share": max(shares) if shares else None,
        "cases": rows,
        "limits": LIMITS,
        "implementation": [{"path": str(Path(__file__)),
                            "sha256": fetch.file_sha256(Path(__file__))}],
    })
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--plan", type=Path, required=True, help="固定診断のplan")
    parser.add_argument("--report", type=Path, required=True, help="固定診断のreport")
    args = parser.parse_args(argv)
    result = run_probe(args.plan.resolve(), args.report.resolve(), args.output.resolve())
    print(f"覆いの測定を保存しました: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
