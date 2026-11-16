"""曲の範囲の読み取りが、対応領域の判定を実際にどれだけ変えるかを測る。

固定した番組runを別のdirectoryへ写して通常の注釈を作り直し、子候補の読み取りを親の
候補へまとめてから、保留結果を対応領域で読み直す。範囲の読み取りを渡さない場合と、
範囲を必須にする方針・しない方針を並べる。

元のrunと固定記録は変更しない。公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss import application  # noqa: E402
from lss.bgm import identity_run  # noqa: E402
from lss.bgm.supported_scope import SupportedScope  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
# 結果を見る前に固定する。対応領域は既存のDevelopment用と同じで、範囲の方針だけを振る。
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
    "publication_allowed": False,
}
ARMS = (
    ("no_extent_reading", "allow_unresolved_cue_extent", False),
    ("with_extent_reading", "allow_unresolved_cue_extent", True),
    ("extent_required", "require_single_cue_evidence", True),
)
LIMITS = [
    "the_original_run_and_frozen_records_are_not_modified",
    "one_real_episode_is_not_a_corpus",
    "the_scope_is_a_development_scope_with_publication_allowed_false",
    "semantic_role_is_taken_from_the_record_and_never_guessed",
    "no_publication_default_change_or_holdout_use",
]


class CueExtentProductProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _scope(policy: str) -> SupportedScope:
    return SupportedScope(**SCOPE_FIELDS, cue_extent_policy=policy)


def _observations(path: Path | None) -> tuple[dict, dict | None]:
    """記録済みの歌唱の観測を、対応領域の読み直しが受け取る形へ直す。"""
    if path is None:
        return {}, None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "COMPLETED_EPISODE_SINGING_OBSERVATION":
        raise CueExtentProductProbeError(f"この歌唱の観測は読めません: {path}")
    return ({row["case_id"]: {"singing": row["singing"]} for row in record["cues"]},
            {"path": str(path), "sha256": fetch.file_sha256(path),
             "singing_recipe_id": record["singing_recipe_id"]})


def run_probe(episode: Path, output: Path, *, cache_dir: Path | None,
              observations_path: Path | None) -> Path:
    run_path = episode / "run.json"
    record_path = episode / "application-outcome.json"
    frozen_identity = episode / "identity.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    annotation = json.loads(frozen_identity.read_text(encoding="utf-8"))
    if run.get("promotion_allowed") is not False or run.get("publication_state") != "not_published":
        raise CueExtentProductProbeError("固定Development記録だけを読み直します")
    before = {path: fetch.file_sha256(path)
              for path in (run_path, record_path, frozen_identity)}

    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "episode"
    workspace.mkdir()
    shutil.copyfile(run_path, workspace / "run.json")
    references = [Path(item["path"]) for item in annotation["external_evidence_used"]
                  if item.get("kind") == "released_reference"]
    print(f"参照録音 {len(references)}件で注釈を作り直します", flush=True)
    fresh = identity_run.annotate_run(workspace / "run.json", references, cache_dir=cache_dir)
    made = json.loads(fresh.read_text(encoding="utf-8"))
    # 製品経路と同じ引き方を使う。別実装で数えない。
    extents = application._cue_extents_from_annotation(run, fresh) or {}
    observations, observation_record = _observations(observations_path)
    print(f"範囲が読めた候補: {len(extents)}件 / {len(run['candidates'])}件", flush=True)

    arms = []
    for name, policy, use_extents in ARMS:
        scope = _scope(policy)
        review = application.review_bgm_scope_coverage(
            record_path, scope, source_domain="finished_stereo_episode",
            observations=observations or None,
            cue_extents=extents if use_extents else None)
        if review["status"] != "REVIEWED_WITHOUT_PUBLISHING" or review["publication_performed"]:
            raise CueExtentProductProbeError("読み直しだけの結果ではありません")
        reasons = {}
        for row in review["candidates"]:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        arms.append({"arm": name, "cue_extent_policy": policy,
                     "cue_extents_passed": use_extents,
                     "scope_id": scope.scope_id,
                     "counts": review["counts"], "reasons": reasons,
                     "disposition": review["disposition"], "reason": review["reason"],
                     "cue_extent_candidate_count": review["cue_extent_candidate_count"]})
        print(f"{name}: {review['counts']}", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise CueExtentProductProbeError(f"固定入力が変更されています: {path}")

    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_cue_extent_product_report",
        "status": "complete_real_episode_description",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "source_id": run["source"]["source_id"],
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "evidence_settings": identity_run.evidence_settings(),
        "singing_observations": observation_record,
        "observed_candidate_count": len(observations),
        "reference_count": len(references),
        "candidate_count": len(run["candidates"]),
        "repeat_pair_count": len(made.get("repeated_recordings") or []),
        "match_count": len(made.get("matches") or []),
        "candidates_with_an_extent_reading": len(extents),
        "extents": extents,
        "arms": arms,
        "limits": LIMITS,
        "implementation": [{"path": str(path), "sha256": fetch.file_sha256(path)} for path in (
            Path(__file__), ROOT / "lss/bgm/cue_extent.py", ROOT / "lss/bgm/identity_run.py",
            ROOT / "lss/bgm/supported_scope.py",
        )],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--episode", type=Path, required=True,
                        help="固定済みの実番組候補生成のdirectory")
    parser.add_argument("--cache-dir", type=Path, help="指紋のcache")
    parser.add_argument("--observations", type=Path, help="記録済みの歌唱の観測")
    args = parser.parse_args(argv)
    report = run_probe(
        args.episode.resolve(), args.output.resolve(),
        cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
        observations_path=args.observations.resolve() if args.observations else None)
    print(f"範囲の読み取りが判定へ与える影響を保存しました: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
