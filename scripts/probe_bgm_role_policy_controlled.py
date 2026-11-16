"""作り方が分かるcorpusで、採用policyと対応領域が欲しくない音楽を止められるか測る。

新しい推論は行わない。固定した歌唱の観測（4stem推論の記録）とcorpusが宣言した役割を
読み直し、採用policyで導いた役割を対応領域へ通す。corpusは役割と欲しさを作り方として
宣言しているので、人手のラベルは使わない。

止まった理由の名前だけを見る。品質は測っていないので、役割の関門を通った候補は
`quality_not_measured` まで進む。そこまで進むことは「公開できる」ではなく、
「役割の仕組みでは止まらなかった」という意味である。

公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm.role_policy import RolePolicy, decide_role  # noqa: E402
from lss.bgm.supported_scope import SupportedScope, classify_candidate  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402
from lss.validation.bgm_corpus import load_bgm_corpus  # noqa: E402

SCHEMA_VERSION = 1
# 実番組で使ったものと同じ採用policy。ここで値を変えない。
ROLE_POLICY = RolePolicy(singing_present_role="singing_score",
                         singing_absent_role="dramatic_score",
                         unresolved_role="unresolved_music_like",
                         indistinguishable_roles=("source_music", "musical_sfx"))
# 公開する役割の宣言を三通り並べる。採用policyが区別できないと言う役割を、
# 対応領域が公開対象に含めるかどうかで、曖昧さが害になるかが変わる。
PUBLISHED_ROLE_SETS = (
    ("dramatic_score",),
    ("dramatic_score", "source_music"),
    ("dramatic_score", "source_music", "musical_sfx"),
)
SCOPE_FIELDS = {
    "source_domains": ("finished_stereo_episode",),
    "covered_semantic_roles": ("dramatic_score", "singing_score"),
    "minimum_cue_seconds": 20.0,
    "maximum_cue_seconds": 600.0,
    "required_evidence": ("source_provenance", "occurrence_span"),
    "quality_gates": {"readiness_passed": 1.0},
    "singing_observation_policy": "exclude_from_publication",
    "cue_extent_policy": "allow_unresolved_cue_extent",
    "publication_allowed": True,
}
# 役割の仕組みで止まったと言える理由。ここに入らずに進んだものは止まらなかったと数える。
ROLE_STOPS = ("observed_singing_out_of_scope", "semantic_role_unresolved",
              "semantic_role_not_wanted_bgm", "semantic_role_out_of_scope",
              "semantic_role_covered_but_not_published",
              "derived_role_cannot_exclude_other_roles")
DECISION_RULE = {
    "roles_and_wantedness_come_from_the_corpus_construction_not_from_a_human_label": True,
    "the_policy_and_the_scopes_are_declared_before_results": True,
    "widening_the_published_roles_is_compared_not_chosen": True,
    "quality_is_not_measured_here_so_passing_the_role_gates_is_not_publishability": True,
}
LIMITS = [
    "reads_a_frozen_corpus_and_frozen_observations_no_new_inference",
    "the_corpus_has_only_two_source_music_cases_so_no_rate_can_be_calibrated",
    "one_of_them_reuses_the_score_recording_so_no_acoustic_observation_can_separate_it",
    "constructed_cases_are_not_the_real_target_domain",
    "no_publication_default_change_or_holdout_use",
]


class ControlledRoleProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def run_probe(observed: Path, manifest: Path, output: Path) -> Path:
    record = json.loads(observed.read_text(encoding="utf-8"))
    if record.get("status") != "COMPLETED_FIXED_ROLE_RULE_COMPARISON":
        raise ControlledRoleProbeError(f"この固定比較の記録は読めません: {observed}")
    corpus = load_bgm_corpus(manifest, verify_sources=True)
    before = {path: fetch.file_sha256(path) for path in (observed, manifest)}
    cases = {case.case_id: case for case in corpus.cases}
    output.mkdir(parents=True, exist_ok=False)

    scopes = []
    for published in PUBLISHED_ROLE_SETS:
        covered = tuple(sorted(set(SCOPE_FIELDS["covered_semantic_roles"]) | set(published)))
        for policy in ("accept_declared_ambiguity", "require_unambiguous_role"):
            scopes.append(SupportedScope(**{**SCOPE_FIELDS,
                                            "covered_semantic_roles": covered,
                                            "published_semantic_roles": published},
                                         derived_role_policy=policy))

    arms = []
    for scope in scopes:
        rows, unwanted_not_stopped, wanted_stopped = [], 0, 0
        for item in record["occurrences"]:
            case = cases[item["case_id"]]
            [occurrence] = [row for row in case.occurrences
                            if row.occurrence_id == item["occurrence_id"]]
            derived = decide_role({"singing": item["singing"]}, ROLE_POLICY)
            decided = classify_candidate({
                "candidate_id": item["occurrence_id"],
                "sample_rate": case.sample_rate,
                "start_frame": occurrence.start_frame, "end_frame": occurrence.end_frame,
                "source_domain": "finished_stereo_episode",
                "semantic_role": derived["semantic_role"],
                "semantic_role_derivation": derived,
                "identity_state": occurrence.identity_state,
                "observations": {"singing": item["singing"]},
                "evidence": {"source_provenance": {"path": str(case.source_path),
                                                   "sha256": case.source_sha256},
                             "occurrence_span": [occurrence.start_frame, occurrence.end_frame]},
            }, scope)
            stopped = decided["reason"] in ROLE_STOPS
            if not occurrence.wanted and not stopped:
                unwanted_not_stopped += 1
            if occurrence.wanted and stopped:
                wanted_stopped += 1
            rows.append({"case_id": item["case_id"], "occurrence_id": item["occurrence_id"],
                         "declared_role": item["reference_role"],
                         "declared_wanted": occurrence.wanted,
                         "observed_singing": item["singing"],
                         "derived_role": derived["semantic_role"],
                         "disposition": decided["disposition"], "reason": decided["reason"],
                         "stopped_by_the_role_machinery": stopped})
        arms.append({"derived_role_policy": scope.derived_role_policy,
                     "published_semantic_roles": list(scope.published_semantic_roles),
                     "scope_id": scope.scope_id,
                     "unwanted_not_stopped": unwanted_not_stopped,
                     "wanted_stopped": wanted_stopped,
                     "published": sum(1 for row in rows if row["disposition"] == "PUBLISHED"),
                     "occurrences": rows})
        print(f"{'+'.join(scope.published_semantic_roles)} / {scope.derived_role_policy}: "
              f"欲しくないのに止まらない{unwanted_not_stopped}件 / "
              f"欲しいのに止まる{wanted_stopped}件", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise ControlledRoleProbeError(f"固定入力が変更されています: {path}")

    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_role_policy_controlled_report",
        "status": "complete_controlled_comparison",
        "role": "development_controlled_comparison_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "corpus_id": corpus.corpus_id,
        "role_policy": ROLE_POLICY.to_dict(), "role_policy_id": ROLE_POLICY.policy_id,
        "scopes": [{"scope_id": row.scope_id, "revision": row.to_dict()} for row in scopes],
        "role_stop_reasons": list(ROLE_STOPS),
        "decision_rule": DECISION_RULE,
        "occurrence_count": len(record["occurrences"]),
        "declared_roles": sorted({row["reference_role"] for row in record["occurrences"]}),
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
    parser.add_argument("--observed", type=Path, required=True,
                        help="固定した役割規則の比較記録")
    parser.add_argument("--manifest", type=Path, required=True, help="制御corpusのmanifest")
    args = parser.parse_args(argv)
    report = run_probe(args.observed.resolve(), args.manifest.resolve(), args.output.resolve())
    print(f"制御corpusでの役割の効き方を保存しました: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
