"""歌唱ありと言う帯を振って、対応領域の内訳がどれだけ動くかを測る。

新しい推論は行わない。固定した観測値（歌唱stemの取り分と音楽の大きさ）を読み直し、
帯だけを振って歌唱の判定と対応領域の読み直しを並べる。帯は選ばない。ここで返すのは、
現在の内訳がどれだけ帯に依存しているかの記述である。

公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss import application  # noqa: E402
from lss.bgm.singing_share import METHOD, SingingShareRecipe, classify  # noqa: E402
from lss.bgm.supported_scope import SupportedScope  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
PRESENT_SHARES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
SCOPE_FIELDS = {
    "source_domains": ("finished_stereo_episode",),
    "published_semantic_roles": ("dramatic_score",),
    "covered_semantic_roles": ("dramatic_score", "singing_score"),
    "minimum_cue_seconds": 20.0,
    "maximum_cue_seconds": 600.0,
    "required_evidence": ("source_provenance", "occurrence_span", "recording_identity"),
    "quality_gates": {"reference_snr_db": 6.0},
    "singing_observation_policy": "exclude_from_publication",
    "derived_role_policy": "accept_declared_ambiguity",
    "cue_extent_policy": "allow_unresolved_cue_extent",
    "publication_allowed": False,
}
DECISION_RULE = {
    "no_band_is_selected_here": True,
    "the_measured_values_are_frozen_and_only_the_band_moves": True,
    "singing_is_an_observation_not_a_role": True,
}
LIMITS = [
    "reads_frozen_observations_only_no_new_inference",
    "one_real_episode_is_not_a_corpus",
    "a_band_that_changes_fewer_candidates_is_not_thereby_more_correct",
    "no_publication_default_change_or_holdout_use",
]


class SingingBandProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def run_probe(singing: Path, episode: Path, output: Path) -> Path:
    record = json.loads(singing.read_text(encoding="utf-8"))
    if record.get("status") != "COMPLETED_EPISODE_SINGING_OBSERVATION":
        raise SingingBandProbeError(f"この歌唱の観測は読めません: {singing}")
    held = episode / "application-outcome.json"
    before = {path: fetch.file_sha256(path) for path in (singing, held)}
    declared = record["singing_recipe"]
    output.mkdir(parents=True, exist_ok=False)

    arms = []
    for present in PRESENT_SHARES:
        recipe = SingingShareRecipe(
            maximum_absent_share=min(declared["maximum_absent_share"], present),
            minimum_present_share=present,
            minimum_music_dbfs=declared["minimum_music_dbfs"])
        observations, counts = {}, {}
        for cue in record["cues"]:
            decided = classify({"method": METHOD, "singing_share": cue["singing_share"],
                                "music_rms_dbfs": cue["music_rms_dbfs"],
                                "music_present": cue["singing_share"] is not None},
                               recipe)
            observations[cue["case_id"]] = {"singing": decided["singing"]}
            counts[decided["singing"]] = counts.get(decided["singing"], 0) + 1
        scope = SupportedScope(**SCOPE_FIELDS)
        review = application.review_bgm_scope_coverage(
            held, scope, source_domain="finished_stereo_episode", observations=observations)
        if review["status"] != "REVIEWED_WITHOUT_PUBLISHING" or review["publication_performed"]:
            raise SingingBandProbeError("読み直しだけの結果ではありません")
        reasons = {}
        for row in review["candidates"]:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        arms.append({"minimum_present_share": present,
                     "singing_recipe_id": recipe.recipe_id,
                     "singing_counts": counts, "counts": review["counts"],
                     "reasons": reasons, "disposition": review["disposition"]})
        print(f"{present}: {counts} -> {review['counts']}", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise SingingBandProbeError(f"固定入力が変更されています: {path}")

    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_singing_band_report",
        "status": "complete_real_episode_description",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "declared_recipe": declared,
        "scope": SupportedScope(**SCOPE_FIELDS).to_dict(),
        "decision_rule": DECISION_RULE,
        "cue_count": len(record["cues"]),
        "observed_shares": sorted(cue["singing_share"] for cue in record["cues"]
                                  if cue["singing_share"] is not None),
        "arms": arms,
        "limits": LIMITS,
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/singing_share.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--singing", type=Path, required=True, help="固定した歌唱の観測")
    parser.add_argument("--episode", type=Path, required=True, help="固定した実番組の保留結果")
    args = parser.parse_args(argv)
    report = run_probe(args.singing.resolve(), args.episode.resolve(), args.output.resolve())
    print(f"歌唱の帯の効き方を保存しました: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
