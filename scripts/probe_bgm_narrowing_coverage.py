"""公開の対象を狭める案ごとに、実番組で何件が残るかを数える。

新しい推論も照合も行わない。固定した観測（歌唱の取り分、既発売録音の一致、番組の中の
再演）を読み直し、候補ごとに観測の組み合わせを並べる。

ここでは狭め方の正しさを主張しない。どの狭め方に材料があるのかを先に見て、
材料の無い案へ時間を使わないためのものである。数えるだけで、何も決めない。
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
    "this_counts_material_it_does_not_justify_any_narrowing": True,
    "every_observation_is_read_from_a_frozen_record": True,
    "a_candidate_without_an_observation_is_counted_as_unobserved_not_as_negative": True,
}
LIMITS = [
    "reads_frozen_records_only_no_new_inference_or_matching",
    "coverage_is_not_correctness_a_narrowing_with_material_can_still_be_wrong",
    "one_real_episode_is_not_a_corpus",
    "no_publication_default_change_or_holdout_use",
]


class NarrowingCoverageProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def run_probe(singing: Path, repeats: Path, output: Path) -> Path:
    observed = json.loads(singing.read_text(encoding="utf-8"))
    if observed.get("status") != "COMPLETED_EPISODE_SINGING_OBSERVATION":
        raise NarrowingCoverageProbeError(f"この歌唱の観測は読めません: {singing}")
    recurrence = json.loads(repeats.read_text(encoding="utf-8"))
    if recurrence.get("type") != "bgm_self_recurrence_report":
        raise NarrowingCoverageProbeError(f"この再演の記述は読めません: {repeats}")
    before = {path: fetch.file_sha256(path) for path in (singing, repeats)}
    output.mkdir(parents=True, exist_ok=False)

    by_id = {row["candidate_asset_id"]: row for row in recurrence["candidates"]}
    rows = []
    for cue in observed["cues"]:
        found = by_id.get(cue["case_id"])
        if found is None:
            raise NarrowingCoverageProbeError(f"再演の記述にこの候補がありません: {cue['case_id']}")
        rows.append({
            "candidate_id": cue["case_id"],
            "seconds": round(found["end_seconds"] - found["start_seconds"], 3),
            "singing": cue["singing"],
            "matches_a_released_recording": found["released_state"] != "unresolved",
            "repeats_inside_this_source": found["repeat_match_count"] > 0,
        })

    def count(**conditions) -> dict:
        kept = [row for row in rows
                if all(row[key] == value for key, value in conditions.items())]
        return {"conditions": conditions, "candidate_count": len(kept),
                "candidate_seconds": round(sum(row["seconds"] for row in kept), 3),
                "candidate_ids": [row["candidate_id"] for row in kept]}

    narrowings = [
        count(),
        count(singing="absent"),
        count(singing="absent", matches_a_released_recording=True),
        count(singing="absent", matches_a_released_recording=False),
        count(singing="absent", repeats_inside_this_source=True),
        count(singing="absent", matches_a_released_recording=False,
              repeats_inside_this_source=True),
    ]

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise NarrowingCoverageProbeError(f"固定入力が変更されています: {path}")

    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_narrowing_coverage_report",
        "status": "complete_real_episode_description",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "source_id": recurrence["source_id"],
        "decision_rule": DECISION_RULE,
        "candidate_count": len(rows),
        "candidates": rows,
        "narrowings": narrowings,
        "limits": LIMITS,
        "implementation": [{"path": str(Path(__file__)),
                            "sha256": fetch.file_sha256(Path(__file__))}],
    })
    for row in narrowings:
        print(f"{row['conditions']}: {row['candidate_count']}件 "
              f"/ {row['candidate_seconds']:.0f}秒", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--singing", type=Path, required=True, help="固定した歌唱の観測")
    parser.add_argument("--repeats", type=Path, required=True, help="固定した再演の記述")
    args = parser.parse_args(argv)
    report = run_probe(args.singing.resolve(), args.repeats.resolve(), args.output.resolve())
    print(f"狭め方ごとの材料を保存しました: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
