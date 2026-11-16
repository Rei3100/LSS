"""参照位置の飛びが、参照録音の中の繰り返しで説明できるかを測る。

凍結済みの一致記録から飛びの位置を取り出し、参照録音そのものの中で「飛ぶ前の位置」と
「飛ぶ後の位置」が似ているかを、その録音自身の無関係な組と比べる。参照は読むだけで、
分離も照合もやり直さない。公開・既定変更・Holdout消費は行わない。

似ていれば、その飛びは編集ではなく繰り返しへの当たり損ねとして説明できる。似ていなければ
説明できない。どちらも観測であって、曲の範囲の判定そのものではない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from lss.bgm.reference_repetition import (  # noqa: E402
    ReferenceRepetitionRecipe,
    compare_positions,
)
from lss.bgm.spectral_features import SpectralFeatureRecipe  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1


class ReferenceRepetitionProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _jump_pairs(matches: list[dict], start: float, end: float) -> tuple[str, list[dict]]:
    """同じ録音への一致が並ぶ区間から、参照位置の飛びを取り出す。"""
    inside = sorted((row for row in matches
                     if row["query_start_seconds"] < end and start < row["query_end_seconds"]),
                    key=lambda row: row["query_start_seconds"])
    references = {row["reference_id"] for row in inside}
    if len(references) != 1:
        raise ReferenceRepetitionProbeError(
            f"一つの録音だけが当たる区間が必要です: {sorted(references)}")
    pairs = []
    for first, second in zip(inside, inside[1:]):
        position = second["query_start_seconds"]
        pairs.append({
            "label": f"jump-at-{position:.2f}",
            "query_seconds": position,
            "offset_before_seconds": first["offset_seconds"],
            "offset_after_seconds": second["offset_seconds"],
            "first_seconds": position + first["offset_seconds"],
            "second_seconds": position + second["offset_seconds"],
        })
    return references.pop(), pairs


def build(episode: Path, output: Path, *, start_seconds: float, end_seconds: float) -> Path:
    identity_path = episode / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("type") != "episode_identity_annotation":
        raise ReferenceRepetitionProbeError("この番組の一致記録ではありません")
    reference_id, pairs = _jump_pairs(identity["matches"], start_seconds, end_seconds)
    if not pairs:
        raise ReferenceRepetitionProbeError("参照位置の飛びがありません")
    reference = next(row for row in identity["matches"] if row["reference_id"] == reference_id)
    reference_path = Path(reference["reference_path"])
    if fetch.file_sha256(reference_path) != reference["reference_sha256"]:
        raise ReferenceRepetitionProbeError(f"参照録音が記録と一致しません: {reference_path}")
    info = sf.info(reference_path)

    feature = SpectralFeatureRecipe(
        n_fft=4096, stft_hop_frames=1024, feature_seconds=1.0, feature_hop_seconds=0.5,
        band_count=24, low_hz=55.0, high_hz=8000.0)
    recipe = ReferenceRepetitionRecipe(
        feature=feature, chroma_weight=0.5, comparison_seconds=4.0,
        null_pair_count=500, null_percentile=95.0, random_seed=20260917)
    recipe.validate()

    output.mkdir(parents=True, exist_ok=False)
    plan = output / "plan.json"
    session._write_json_atomic(plan, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_reference_repetition_plan",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "episode": {"identity": {"path": str(identity_path),
                                 "sha256": fetch.file_sha256(identity_path)}},
        "candidate_span_seconds": [start_seconds, end_seconds],
        "reference": {"reference_id": reference_id, "path": str(reference_path),
                      "sha256": reference["reference_sha256"], "sample_rate": info.samplerate,
                      "frames": info.frames, "seconds": info.frames / info.samplerate},
        "recipe": {"recipe_id": recipe.recipe_id, "revision": recipe.to_dict()},
        "pairs": pairs,
        "decision_rule": {
            "a_jump_is_explained_when_similarity_exceeds_the_null_percentile": True,
            "declared_before_results": True,
        },
        "limits": [
            "reads_frozen_matches_and_the_released_reference_only",
            "one_candidate_in_one_episode_not_a_corpus",
            "similarity_does_not_prove_the_same_repeat_and_its_absence_does_not_prove_an_edit",
            "the_released_reference_is_not_assumed_identical_to_the_programme_mix",
            "no_publication_default_change_or_holdout_use",
        ],
    })
    return plan


def run(plan_path: Path) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("promotion_allowed") is not False
            or plan.get("type") != "bgm_reference_repetition_plan"):
        raise ReferenceRepetitionProbeError("固定した繰り返し比較の計画だけを実行します")
    reference = plan["reference"]
    if fetch.file_sha256(Path(reference["path"])) != reference["sha256"]:
        raise ReferenceRepetitionProbeError("参照録音が記録と一致しません")

    revision = dict(plan["recipe"]["revision"])
    recipe = ReferenceRepetitionRecipe(**{
        **revision, "feature": SpectralFeatureRecipe(**revision["feature"])})
    if recipe.recipe_id != plan["recipe"]["recipe_id"]:
        raise ReferenceRepetitionProbeError("Recipe identityが固定記録と一致しません")

    values, rate = sf.read(reference["path"], dtype="float64", always_2d=True)
    read = compare_positions(values.T, int(rate), plan["pairs"], recipe)
    explained = sum(1 for row in read["pairs"] if row["above_null_threshold"])

    rows = []
    for pair, measured in zip(plan["pairs"], read["pairs"]):
        rows.append({**{key: pair[key] for key in
                        ("label", "query_seconds", "offset_before_seconds",
                         "offset_after_seconds")},
                     "reference_seconds_before": measured["first_seconds"],
                     "reference_seconds_after": measured["second_seconds"],
                     "similarity": measured["similarity"],
                     "explained_by_repetition": measured["above_null_threshold"]})

    output = plan_path.parent
    if fetch.file_sha256(Path(reference["path"])) != reference["sha256"]:
        raise ReferenceRepetitionProbeError("比較中に参照録音が変わりました")
    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_reference_repetition_report",
        "status": "complete_real_episode_description",
        "role": plan["role"], "promotion_allowed": False, "default_changed": False,
        "holdout_consumed": False, "winner_selected": False,
        "plan_sha256": fetch.file_sha256(plan_path),
        "reference_id": reference["reference_id"],
        "recipe_id": read["recipe_id"],
        "decision_rule": plan["decision_rule"],
        "null": read["null"],
        "jump_count": len(rows), "explained_jump_count": explained,
        "jumps": rows,
        "limits": plan["limits"],
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/reference_repetition.py",
            ROOT / "lss/bgm/spectral_features.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--episode", type=Path, help="固定済みの実番組候補生成のdirectory")
    parser.add_argument("--start-seconds", type=float, help="対象とする候補の開始")
    parser.add_argument("--end-seconds", type=float, help="対象とする候補の終わり")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.resume:
        plan = output / "plan.json"
    else:
        if args.episode is None or args.start_seconds is None or args.end_seconds is None:
            parser.error("--episodeと--start-seconds、--end-secondsが必要です")
        plan = build(args.episode.resolve(), output,
                     start_seconds=args.start_seconds, end_seconds=args.end_seconds)
    print(f"参照録音の繰り返しの記述を保存しました: {run(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
