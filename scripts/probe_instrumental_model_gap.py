"""静かな伴奏の失敗が、模型の選び方で直るのかを固定記録から読む。

新しい推論は行わない。同じ12ケースで既に測り終えている模型ごとの記録を読み直し、
場面の種類ごとに並べる。どの模型を採るかはここでは決めない。

比べる前に何を見るかを決める。静かな伴奏の改善、歌唱混合の改善、そして歌唱が無いときの
伴奏の保持である。三つを同時に見て、平均で最悪例を隠さない。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
DECISION_RULE = {
    "no_new_inference_only_frozen_reports_are_read": True,
    "three_scene_kinds_are_read_together_not_averaged": True,
    "no_model_is_selected_here": True,
}
LIMITS = [
    "reads_frozen_reports_only",
    "public_training_material_and_constructed_linear_mixtures_only",
    "twelve_cases_and_three_models_are_not_a_promotion_corpus",
    "the_runs_were_made_at_different_times_so_runtime_differences_stay_a_confound",
    "no_publication_default_change_or_holdout_use",
]


class ModelGapProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def run_probe(entries: list[tuple[str, Path]], output: Path) -> Path:
    before, measured = {}, {}
    for name, path in entries:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("promotion_allowed") is not False:
            raise ModelGapProbeError(f"固定したDevelopment記録だけを読み直します: {path}")
        before[path] = fetch.file_sha256(path)
        for case in report["cases"]:
            fidelity = case["quality"]["fidelity"]
            measured.setdefault(case["case_id"], {"variant": case["variant"], "models": {}})
            measured[case["case_id"]]["models"][name] = {
                "si_sdr_db": (fidelity.get("si_sdr_db") or {}).get("joint"),
                "projection_gain_db": (fidelity.get("projection_gain_db") or {}).get("joint"),
                "target_absent_output_to_vocals_db":
                    case["quality"].get("target_absent_output_to_vocals_db"),
            }

    names = [name for name, _ in entries]
    by_variant = {}
    for case_id, row in measured.items():
        if set(row["models"]) != set(names):
            raise ModelGapProbeError(f"全模型の記録が揃っていません: {case_id}")
        by_variant.setdefault(row["variant"], []).append(case_id)

    summary = {}
    for variant, case_ids in sorted(by_variant.items()):
        entry = {}
        for name in names:
            values = [measured[case_id]["models"][name]["si_sdr_db"] for case_id in case_ids]
            if any(value is None for value in values):
                entry[name] = {"si_sdr_db": None, "case_count": len(values)}
                continue
            entry[name] = {"si_sdr_db": sorted(values), "worst_si_sdr_db": min(values),
                           "case_count": len(values)}
        summary[variant] = entry
        print(f"{variant}: " + " / ".join(
            f"{name}={entry[name]['si_sdr_db']}" for name in names), flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise ModelGapProbeError(f"固定入力が変更されています: {path}")

    output.mkdir(parents=True, exist_ok=False)
    result = output / "report.json"
    session._write_json_atomic(result, {
        "schema_version": SCHEMA_VERSION,
        "type": "instrumental_model_gap_report",
        "status": "complete_development_reading",
        "role": "development_reading_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"name": name, "path": str(path),
                           "sha256": before[path]} for name, path in entries],
        "decision_rule": DECISION_RULE,
        "models": names,
        "case_count": len(measured),
        "by_variant": summary,
        "cases": measured,
        "limits": LIMITS,
        "implementation": [{"path": str(Path(__file__)),
                            "sha256": fetch.file_sha256(Path(__file__))}],
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--model", action="append", required=True,
                        help="名前=固定reportのパス。複数指定する")
    args = parser.parse_args(argv)
    entries = []
    for item in args.model:
        name, _, path = item.partition("=")
        if not name or not path:
            parser.error("--modelは 名前=パス の形で指定します")
        entries.append((name, Path(path).resolve()))
    result = run_probe(entries, args.output.resolve())
    print(f"模型ごとの読み取りを保存しました: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
