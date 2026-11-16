"""無音を挟まない継ぎ目を、落ち込みの深さを決めて作り、当たり外れを数える。

曲Aの終わりを宣言した深さまで下げ、続けて曲Bを鳴らす。継ぎ目の位置は作り方から分かる。
作った落ち込みを見つけられるかは、観測の効き方であって「実際の番組の継ぎ目がその深さを
持つ」ことの証明ではない。

一方、同じ録音が続くだけの対照で何件出るかは、作り物ではない誤りの数である。両方を
並べて示す。公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from lss.bgm.music_gap import MusicDipRecipe, find_relative_dips  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1


class JoinDipProbeError(RuntimeError):
    """比較の入力または組み立てが契約に合わない。"""


def _active_span(path: Path) -> tuple[int, int, int]:
    values, rate = sf.read(path, dtype="float64", always_2d=True)
    active = np.any(values != 0.0, axis=1).nonzero()[0]
    if active.size == 0:
        raise JoinDipProbeError(f"音のある区間がありません: {path}")
    return int(active[0]), int(active[-1]) + 1, int(rate)


def build(recordings: list[str], output: Path, *, segment_seconds: float,
          margin_seconds: float, fade_seconds: float, depths_db: list[float],
          tolerance_seconds: float) -> Path:
    parsed = []
    for item in recordings:
        label, _, raw = item.partition("=")
        path = Path(raw).resolve()
        if not label.strip() or not path.is_file():
            raise JoinDipProbeError(f"録音の指定が不正です: {item}")
        start, end, rate = _active_span(path)
        parsed.append({"label": label.strip(), "path": str(path),
                       "sha256": fetch.file_sha256(path), "sample_rate": rate,
                       "active_start_frame": start, "active_end_frame": end})
    if len(parsed) < 2:
        raise JoinDipProbeError("比較には2つ以上の録音が必要です")
    rates = {item["sample_rate"] for item in parsed}
    if len(rates) != 1:
        raise JoinDipProbeError(f"sample rateが一致しません: {sorted(rates)}")
    rate = rates.pop()
    segment = int(round(segment_seconds * rate))
    margin = int(round(margin_seconds * rate))
    for item in parsed:
        if item["active_end_frame"] - item["active_start_frame"] - 2 * margin < 2 * segment:
            raise JoinDipProbeError(f"{item['label']}は連結に足りません")

    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_join_dip_plan",
        "role": "development_controlled_diagnostic_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "sample_rate": rate,
        "segment_seconds": segment_seconds, "margin_seconds": margin_seconds,
        "fade_seconds": fade_seconds, "construction_depths_db": list(depths_db),
        "detector_depths_db": list(depths_db),
        "tolerance_seconds": tolerance_seconds,
        "fixed_detector_fields": {"window_seconds": 0.10, "hop_seconds": 0.025, "rise_db": 10.0,
                                  "rise_window_seconds": 3.0,
                                  "minimum_separation_seconds": 5.0, "edge_margin_seconds": 1.0},
        "recordings": parsed,
        "limits": [
            "the_fade_at_the_join_is_constructed_here_so_detection_is_sensitivity_not_realism",
            "the_continuous_controls_are_real_music_so_their_count_is_a_real_error_count",
            "three_distinct_recordings_are_not_a_corpus",
            "no_threshold_is_selected_here_and_no_default_is_changed",
        ],
    })
    return plan


def _segment(item: dict, *, at_end: bool, segment: int, margin: int) -> np.ndarray:
    if at_end:
        stop = item["active_end_frame"] - margin
        start = stop - segment
    else:
        start = item["active_start_frame"] + margin
        stop = start + segment
    values, _ = sf.read(item["path"], dtype="float64", always_2d=True, start=start, stop=stop)
    return values.T


def _faded(values: np.ndarray, *, fade: int, depth_db: float) -> np.ndarray:
    """終わりの fade frames を、宣言した深さまでdB直線で下げる。"""
    faded = values.copy()
    ramp = np.linspace(0.0, -depth_db, fade)
    faded[:, -fade:] *= 10.0 ** (ramp / 20.0)
    return faded


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("promotion_allowed") is not False or plan.get("type") != "bgm_join_dip_plan":
        raise JoinDipProbeError("固定した継ぎ目比較の計画だけを実行します")
    for item in plan["recordings"]:
        if fetch.file_sha256(Path(item["path"])) != item["sha256"]:
            raise JoinDipProbeError(f"録音が記録と一致しません: {item['path']}")

    rate = plan["sample_rate"]
    segment = int(round(plan["segment_seconds"] * rate))
    margin = int(round(plan["margin_seconds"] * rate))
    fade = int(round(plan["fade_seconds"] * rate))
    tolerance = plan["tolerance_seconds"]
    fixed = plan["fixed_detector_fields"]

    joins = []
    for first in plan["recordings"]:
        for second in plan["recordings"]:
            if first["label"] == second["label"]:
                continue
            joins.append((first, second))
    controls = plan["recordings"]

    rows = []
    for construction_depth in plan["construction_depths_db"]:
        built = []
        for first, second in joins:
            head = _faded(_segment(first, at_end=True, segment=segment, margin=margin),
                          fade=fade, depth_db=construction_depth)
            tail = _segment(second, at_end=False, segment=segment, margin=margin)
            built.append((f"{first['label']}-to-{second['label']}",
                          np.concatenate([head, tail], axis=1)))
        for detector_depth in plan["detector_depths_db"]:
            recipe = MusicDipRecipe(depth_db=detector_depth, **fixed)
            hits = 0
            for _, signal in built:
                dips = find_relative_dips(signal, rate, recipe)["dips"]
                if any(abs(dip["seconds"] - plan["segment_seconds"]) <= tolerance
                       for dip in dips):
                    hits += 1
            rows.append({"construction_depth_db": construction_depth,
                         "detector_depth_db": detector_depth,
                         "join_count": len(built), "join_hits": hits})

    control_rows = []
    for detector_depth in plan["detector_depths_db"]:
        recipe = MusicDipRecipe(depth_db=detector_depth, **fixed)
        found = 0
        seconds = 0.0
        for item in controls:
            start = item["active_start_frame"] + margin
            values, _ = sf.read(item["path"], dtype="float64", always_2d=True,
                                start=start, stop=start + 2 * segment)
            seconds += values.shape[0] / rate
            found += len(find_relative_dips(values.T, rate, recipe)["dips"])
        control_rows.append({"detector_depth_db": detector_depth,
                             "control_count": len(controls),
                             "control_seconds": round(seconds, 2),
                             "dips_in_continuous_music": found})

    for item in plan["recordings"]:
        if fetch.file_sha256(Path(item["path"])) != item["sha256"]:
            raise JoinDipProbeError("比較中に録音が変わりました")
    report = plan_path.parent / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_join_dip_report",
        "status": "complete_development_diagnostic",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "sensitivity": rows, "controls": control_rows,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/music_gap.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--segment-seconds", type=float, default=12.0)
    parser.add_argument("--margin-seconds", type=float, default=1.0)
    parser.add_argument("--fade-seconds", type=float, default=1.0)
    parser.add_argument("--depth-db", type=float, action="append", default=[])
    parser.add_argument("--tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if len(args.recording) < 2 or not args.depth_db:
            parser.error("2つ以上の--recordingと1つ以上の--depth-dbが必要です")
        plan = build(args.recording, output, segment_seconds=args.segment_seconds,
                     margin_seconds=args.margin_seconds, fade_seconds=args.fade_seconds,
                     depths_db=args.depth_db, tolerance_seconds=args.tolerance_seconds)
    print(f"継ぎ目の落ち込みの記録を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
