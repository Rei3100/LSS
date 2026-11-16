"""実番組の保留結果へ採用policyを当て、役割を自分で決めたときの内訳を測る。

新しい推論は行わない。固定した観測（歌唱stemの取り分）と保留結果を読み直し、
役割を記録から受け取る場合と、宣言した採用policyで観測から導く場合を並べる。

役割は人手のラベルから受け取らない。導いた役割は採用policyの結論であり、annotationの
事実ではない。観測が決めないものは未解決のまま残す。
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
from lss.bgm.cue_readiness import CueReadinessRecipe  # noqa: E402
from lss.bgm.role_policy import RolePolicy  # noqa: E402
from lss.bgm.supported_scope import SupportedScope  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する。制御corpusで比較した規則と同じ本体・同じ値を使う。
ROLE_POLICY = RolePolicy(singing_present_role="singing_score",
                         singing_absent_role="dramatic_score",
                         unresolved_role="unresolved_music_like",
                         indistinguishable_roles=("source_music", "musical_sfx"))
# 対応領域は二通り宣言する。どちらも結果を見る前に固定し、値は契約文から取る。
# `docs/BGM.md`「Single occurrence、unknown identity、unreleasedであることだけを理由に、
# 他はvalidなcueを除外しない」に従い、既知録音の一致を必須にしない宣言も並べる。
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
# 参照なしで測れる検査。既存の実番組観測で固定した値をそのまま使う。
READINESS = CueReadinessRecipe(required_channels=2, minimum_loudness_lufs=-45.0,
                               maximum_true_peak_dbtp=-1.0, maximum_clipped_samples=0,
                               minimum_spectral_edge_hz=15000.0, spectral_edge_drop_db=60.0,
                               minimum_side_to_mid_db=-40.0)
DECISION_RULE = {
    "the_policy_and_its_values_are_declared_before_results": True,
    "no_human_label_is_used_anywhere": True,
    "an_observation_that_does_not_decide_leaves_the_role_unresolved": True,
    "source_music_and_musical_sfx_are_declared_indistinguishable": True,
    "a_scope_may_refuse_a_role_whose_declared_ambiguity_reaches_unpublished_roles": True,
}
LIMITS = [
    "reads_frozen_observations_and_a_frozen_held_result_no_new_inference",
    "the_derived_role_is_a_policy_conclusion_not_an_annotation_fact",
    "the_singing_observation_comes_from_a_model_so_it_is_not_semantic_ground_truth",
    "source_music_read_as_dramatic_score_is_a_known_and_declared_error_mode",
    "one_real_episode_is_not_a_corpus",
    "no_publication_default_change_or_holdout_use",
]


class RolePolicyProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def run_probe(singing: Path, episode: Path, output: Path) -> Path:
    record = json.loads(singing.read_text(encoding="utf-8"))
    if record.get("status") != "COMPLETED_EPISODE_SINGING_OBSERVATION":
        raise RolePolicyProbeError(f"この歌唱の観測は読めません: {singing}")
    held = episode / "application-outcome.json"
    before = {path: fetch.file_sha256(path) for path in (singing, held)}
    observations = {row["case_id"]: {"singing": row["singing"]} for row in record["cues"]}
    scope = SupportedScope(**SCOPE_FIELDS)
    output.mkdir(parents=True, exist_ok=False)

    unknown_ok = SupportedScope(**{**SCOPE_FIELDS,
                                   "required_evidence": ("source_provenance", "occurrence_span")})
    # 参照が無い曲でも測れる品質で読む宣言。既知録音の一致も参照SNRも要求しない。
    measurable = SupportedScope(**{**SCOPE_FIELDS,
                                   "required_evidence": ("source_provenance", "occurrence_span"),
                                   "quality_gates": {"readiness_passed": 1.0}})
    # 採用policyが区別できないと宣言した役割を、公開の場では許さない宣言。
    unambiguous = SupportedScope(**{**SCOPE_FIELDS,
                                    "required_evidence": ("source_provenance", "occurrence_span"),
                                    "quality_gates": {"readiness_passed": 1.0},
                                    "derived_role_policy": "require_unambiguous_role"})
    arms = []
    for name, policy, used, recipe in (
            ("role_from_the_record", None, scope, None),
            ("role_from_the_policy", ROLE_POLICY, scope, None),
            ("role_from_the_policy_without_requiring_identity", ROLE_POLICY, unknown_ok, None),
            ("role_from_the_policy_with_measurable_quality", ROLE_POLICY, measurable, READINESS),
            ("role_from_the_policy_requiring_an_unambiguous_role",
             ROLE_POLICY, unambiguous, READINESS)):
        review = application.review_bgm_scope_coverage(
            held, used, source_domain="finished_stereo_episode", readiness=recipe,
            observations=observations, role_policy=policy)
        if review["status"] != "REVIEWED_WITHOUT_PUBLISHING" or review["publication_performed"]:
            raise RolePolicyProbeError("読み直しだけの結果ではありません")
        reasons, roles = {}, {}
        for row in review["candidates"]:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            roles[row["semantic_role"]] = roles.get(row["semantic_role"], 0) + 1
        arms.append({"arm": name, "scope_id": used.scope_id,
                     "required_evidence": list(used.required_evidence),
                     "quality_gates": dict(used.quality_gates),
                     "derived_role_policy": used.derived_role_policy,
                     "cue_readiness_recipe_id": review["cue_readiness_recipe_id"],
                     "role_policy_id": review["role_policy_id"],
                     "derived_role_candidate_count": review["derived_role_candidate_count"],
                     "semantic_roles": roles, "counts": review["counts"], "reasons": reasons,
                     "disposition": review["disposition"], "reason": review["reason"],
                     "candidates": [{"candidate_id": row["candidate_id"],
                                     "semantic_role": row["semantic_role"],
                                     "disposition": row["disposition"],
                                     "reason": row["reason"],
                                     "missing_evidence": row.get("missing_evidence"),
                                     "cue_seconds": row["cue_seconds"]}
                                    for row in review["candidates"]]})
        print(f"{name}: {review['counts']} / {reasons}", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise RolePolicyProbeError(f"固定入力が変更されています: {path}")

    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_role_policy_episode_report",
        "status": "complete_real_episode_description",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "role_policy": ROLE_POLICY.to_dict(), "role_policy_id": ROLE_POLICY.policy_id,
        "singing_recipe_id": record["singing_recipe_id"],
        "scopes": [{"scope_id": row.scope_id, "revision": row.to_dict()}
                   for row in (scope, unknown_ok, measurable, unambiguous)],
        "cue_readiness_recipe": READINESS.to_dict(),
        "decision_rule": DECISION_RULE,
        "candidate_count": len(record["cues"]),
        "arms": arms,
        "limits": LIMITS,
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/role_policy.py", ROOT / "lss/bgm/supported_scope.py",
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
    print(f"採用policyを当てた内訳を保存しました: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
