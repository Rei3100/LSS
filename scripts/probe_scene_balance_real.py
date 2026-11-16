"""場面の種類の観測を、実際の番組の分離成果物で確かめる。

新しい分離推論は行わない。既に保存されている実番組の伴奏成果物と、その元の原本を
読み直し、同じ宣言で観測を当てる。作られた12ケースの外でも読み取りが成立するか、
どの帯に入るのかを見る。

実番組には正解が無いので、ここで品質は測らない。返すのは観測と、その値だけである。
公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core import audio_input, session  # noqa: E402
from lss.harness import fetch  # noqa: E402
from lss.separation import observation_audio  # noqa: E402
from lss.separation.scene_balance import (  # noqa: E402
    SceneBalanceRecipe,
    TimedSceneBalanceRecipe,
    read_balance_from_segments,
)

SCHEMA_VERSION = 1
# 固定ケースで使ったものと同じ宣言。実番組を見てから変えない。
RECIPE = SceneBalanceRecipe(absent_below_db=-40.0, quiet_below_db=-3.0, dominant_above_db=3.0)
# 区間の長さは、cueとして扱う下限（20秒）より短い10秒にし、半分ずつ進める。
TIMED = TimedSceneBalanceRecipe(window_seconds=10.0, hop_seconds=5.0, absent_below_db=-40.0,
                                quiet_below_db=-3.0, dominant_above_db=3.0)
DECISION_RULE = {
    "the_boundaries_are_the_same_as_the_controlled_comparison": True,
    "no_quality_is_measured_because_there_is_no_ground_truth": True,
    "the_other_stem_is_the_mixture_minus_the_estimate": True,
}
LIMITS = [
    "reads_frozen_sources_and_frozen_separation_artifacts_no_new_inference",
    "the_mixture_is_matched_to_the_estimate_rate_for_the_reading_and_that_is_recorded",
    "a_whole_episode_mixes_many_scenes_so_one_value_describes_the_average",
    "no_ground_truth_so_nothing_here_says_the_output_is_good_or_bad",
    "no_publication_default_change_or_holdout_use",
]


class SceneBalanceRealProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def run_probe(pairs: list[tuple[str, Path, Path]], output: Path) -> Path:
    before = {}
    for _, source, estimate in pairs:
        for path in (source, estimate):
            if not path.is_file():
                raise SceneBalanceRealProbeError(f"入力がありません: {path}")
            before[path] = fetch.file_sha256(path)
    output.mkdir(parents=True, exist_ok=False)

    rows = []
    for name, source, estimate in pairs:
        # 通常経路と同じ読み取り本体を使う。別実装で測らない。
        with audio_input.prepare_input(source) as prepared:
            # 通常経路と同じく、推論へ渡す形になった信号と比べる。
            read = observation_audio.segment_energies(
                prepared, estimate, segment_seconds=TIMED.hop_seconds)
        observed = read_balance_from_segments(
            read["segments"], TIMED, other_energy_of=observation_audio.other_energy)
        timed = observed.get("over_time") or {}
        rows.append({"name": name, "source": str(source), "estimate": str(estimate),
                     "seconds": sum(row["frames"] for row in read["segments"])
                     / read["revision"]["sample_rate"],
                     "state": observed["state"], "reason": observed["reason"],
                     "balance_db": observed["balance_db"],
                     "read": read["revision"],
                     "window_count": timed.get("window_count"),
                     "state_shares": timed.get("state_shares"),
                     "median_window_balance_db": timed.get("median_balance_db")})
        print(f"{name}: {observed['state']} ({observed['balance_db']:.2f} dB) / "
              f"区間 {timed.get('state_shares')}", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise SceneBalanceRealProbeError(f"固定入力が変更されています: {path}")

    states = {}
    for row in rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    result = output / "report.json"
    session._write_json_atomic(result, {
        "schema_version": SCHEMA_VERSION,
        "type": "scene_balance_real_report",
        "status": "complete_real_material_description",
        "role": "development_real_material_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "recipe": RECIPE.to_dict(), "recipe_id": RECIPE.recipe_id,
        "timed_recipe": TIMED.to_dict(), "timed_recipe_id": TIMED.recipe_id,
        "decision_rule": DECISION_RULE,
        "source_count": len(rows), "states": states,
        "sources": rows,
        "limits": LIMITS,
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/separation/scene_balance.py",
        )],
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--pair", action="append", required=True,
                        help="名前=原本のパス=分離成果物のパス。複数指定する")
    args = parser.parse_args(argv)
    pairs = []
    for item in args.pair:
        name, _, rest = item.partition("=")
        source, _, estimate = rest.partition("=")
        if not name or not source or not estimate:
            parser.error("--pairは 名前=原本=成果物 の形で指定します")
        pairs.append((name, Path(source).resolve(), Path(estimate).resolve()))
    result = run_probe(pairs, args.output.resolve())
    print(f"実素材での観測を保存しました: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
