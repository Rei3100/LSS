"""歌唱の推定の中に、どれだけ伴奏が入っているかをMid/Sideに分けて測る。

覆いの測定から、取りこぼした伴奏は歌唱の推定の中にあることが分かっている。ここでは
その中身を、正解の伴奏と正解の歌唱への射影で分けて見る。参照は測定にだけ使い、
処理の決定には使わない。

狙いは、参照なしで使える手掛かりがあるかを先に見ることである。左右の広がり（Side）に
伴奏が偏っているなら、そこから回収できる見込みがある。偏っていないなら、この道は無い。

あわせて、伴奏の推定の誤差そのものを「足りない伴奏」と「混ざった歌唱」へ分ける。
どちらでも説明できない分が大きければ、レベルや引き算では直せない。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
DECISION_RULE = {
    "the_reference_is_used_for_measurement_only": True,
    "no_audio_is_written_and_nothing_is_added_back": True,
    "a_share_is_not_a_recovery_rule": True,
}
LIMITS = [
    "reads_frozen_inputs_and_frozen_stems_no_new_inference",
    "public_training_material_and_constructed_linear_mixtures_only",
    "projection_shares_overlap_when_the_components_are_correlated",
    "no_publication_default_change_or_holdout_use",
]


class WrongStemShareProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _read(path: Path) -> np.ndarray:
    values, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return values.T


def _mid_side(value: np.ndarray) -> dict:
    mid = (value[0] + value[1]) / 2.0
    side = (value[0] - value[1]) / 2.0
    return {"mid": mid, "side": side}


def _projected(part: np.ndarray, basis: np.ndarray) -> float:
    """partのうち、basisと同じ形で説明できるエネルギー。"""
    energy = float(np.sum(basis ** 2))
    if energy <= 0:
        return 0.0
    scale = float(np.sum(part * basis)) / energy
    return float(np.sum((scale * basis) ** 2))


def _share(part: np.ndarray, target: np.ndarray) -> float | None:
    """partのうち、targetで説明できるエネルギーの割合。"""
    energy = float(np.sum(target ** 2))
    if energy <= 0:
        return None
    projection = float(np.dot(part, target)) / energy * target
    total = float(np.sum(part ** 2))
    return (float(np.sum(projection ** 2)) / total) if total > 0 else None


def run_probe(plan_path: Path, report_path: Path, stems_path: Path, output: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stems = json.loads(stems_path.read_text(encoding="utf-8"))
    if any(item.get("promotion_allowed") is not False for item in (plan, report, stems)):
        raise WrongStemShareProbeError("固定したDevelopment記録だけを読み直します")
    if stems.get("type") != "instrumental_stem_complement_report":
        raise WrongStemShareProbeError(f"この歌唱側の記録は読めません: {stems_path}")
    cases = {case["case_id"]: case for case in plan["cases"]}
    produced = {row["case_id"]: row for row in stems["cases"]}
    before = {path: fetch.file_sha256(path) for path in (plan_path, report_path, stems_path)}
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for item in report["cases"]:
        case, stem = cases[item["case_id"]], produced[item["case_id"]]
        artifacts = case["artifacts"]
        vocals_path = (stems_path.parent / "outputs" / item["case_id"])
        [found] = sorted(vocals_path.glob("*.flac"))
        if fetch.file_sha256(found) != stem["vocals_output_sha256"]:
            raise WrongStemShareProbeError(f"固定した歌唱出力が変更されています: {found}")
        estimate = _read(found) * 2.0 ** stem["vocals_serialization_shift"]
        target = _read(Path(artifacts["target"]["path"]))
        voice = _read(Path(artifacts["vocals"]["path"]))
        frames = min(estimate.shape[1], target.shape[1], voice.shape[1])
        estimate, target, voice = (estimate[:, :frames], target[:, :frames], voice[:, :frames])

        row = {"case_id": item["case_id"], "variant": item["variant"],
               "vocals_estimate_energy": float(np.sum(estimate ** 2))}
        parts = _mid_side(estimate)
        truths = {"accompaniment": _mid_side(target), "vocals": _mid_side(voice)}
        for channel in ("mid", "side"):
            row[f"{channel}_energy"] = float(np.sum(parts[channel] ** 2))
            for name, truth in truths.items():
                row[f"{channel}_share_of_{name}"] = _share(parts[channel], truth[channel])
        instrumental = (_read(Path(item["result"]["outputs"][0]))
                        * 2.0 ** item["serialization_shift"])[:, :frames]
        error = instrumental - target
        error_energy = float(np.sum(error ** 2))
        row["instrumental_error_energy"] = error_energy
        row["instrumental_error_to_target"] = (
            error_energy / float(np.sum(target ** 2)) if float(np.sum(target ** 2)) > 0 else None)
        for name, truth in (("missing_accompaniment", target), ("added_vocals", voice)):
            row[f"error_share_{name}"] = (
                None if error_energy <= 0 else _projected(error, truth) / error_energy)
        if error_energy > 0:
            explained = sum(row[f"error_share_{name}"] or 0.0
                            for name in ("missing_accompaniment", "added_vocals"))
            row["error_share_unexplained"] = max(0.0, 1.0 - explained)
        else:
            row["error_share_unexplained"] = None
        rows.append(row)
        print(f"{item['case_id']}: side伴奏 {row['side_share_of_accompaniment']} / "
              f"誤差の説明できない割合 {row['error_share_unexplained']}", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise WrongStemShareProbeError(f"固定入力が変更されています: {path}")

    result = output / "report.json"
    session._write_json_atomic(result, {
        "schema_version": SCHEMA_VERSION,
        "type": "instrumental_wrong_stem_share_report",
        "status": "complete_development_measurement",
        "role": "development_measurement_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "decision_rule": DECISION_RULE,
        "case_count": len(rows),
        "cases": rows,
        "limits": LIMITS,
        "implementation": [{"path": str(Path(__file__)),
                            "sha256": fetch.file_sha256(Path(__file__))}],
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--plan", type=Path, required=True, help="固定診断のplan")
    parser.add_argument("--report", type=Path, required=True, help="固定診断のreport")
    parser.add_argument("--stems", type=Path, required=True, help="歌唱側の測定record")
    args = parser.parse_args(argv)
    result = run_probe(args.plan.resolve(), args.report.resolve(), args.stems.resolve(),
                       args.output.resolve())
    print(f"誤配分の中身を保存しました: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
