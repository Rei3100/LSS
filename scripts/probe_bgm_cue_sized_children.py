"""切り分けた子候補のうち、曲一つとして扱える大きさと根拠を持つものを数える。

新しい推論も照合も行わない。runとその注釈を読み直し、候補を根拠の境目で分けた子候補が
いくつ「一曲だと言える」かを数える。現在描画されている候補が粗いだけで、曲単位の材料が
既にあるのかを見る。

公開・既定変更・Holdout消費は行わない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.bgm import identity_run  # noqa: E402
from lss.core import session  # noqa: E402
from lss.harness import fetch  # noqa: E402

SCHEMA_VERSION = 1
DECISION_RULE = {
    "children_are_not_rendered_here": True,
    "cue_length_window_is_declared_before_counting": True,
    "an_extent_reading_is_not_a_semantic_role": True,
}
LIMITS = [
    "reads_a_frozen_run_and_its_annotation_no_new_inference_or_rendering",
    "a_child_with_a_decided_extent_is_not_thereby_publishable",
    "the_semantic_role_of_every_child_is_still_unresolved",
    "no_publication_default_change_or_holdout_use",
]


class CueSizedChildrenProbeError(RuntimeError):
    """比較の入力が固定記録と一致しない。"""


def _kinds(reading: dict) -> list[str]:
    found = []
    for row in (reading.get("evidence") or {}).get("readings") or []:
        if row.get("state") == "single_cue":
            kind = (row.get("evidence") or {}).get("evidence_kind")
            if isinstance(kind, str):
                found.append(kind)
    return sorted(set(found))


def run_probe(run_path: Path, annotation_path: Path, output: Path, *,
              minimum_seconds: float, maximum_seconds: float) -> Path:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation.get("type") != "episode_identity_annotation":
        raise CueSizedChildrenProbeError(f"この注釈は読めません: {annotation_path}")
    if run.get("promotion_allowed") is not False:
        raise CueSizedChildrenProbeError("固定Development記録だけを読み直します")
    before = {path: fetch.file_sha256(path) for path in (run_path, annotation_path)}
    output.mkdir(parents=True, exist_ok=False)

    recipes = []
    for activity in annotation.get("activity_runs") or []:
        children = activity.get("candidates") or []
        seconds = [float(row["end_seconds"]) - float(row["start_seconds"]) for row in children]
        states, kinds, usable = {}, {}, []
        for row, span in zip(children, seconds):
            reading = row.get("cue_extent") or {}
            state = reading.get("state", "no_reading")
            states[state] = states.get(state, 0) + 1
            if state != "single_cue":
                continue
            for kind in _kinds(reading) or ["unnamed"]:
                kinds[kind] = kinds.get(kind, 0) + 1
            if minimum_seconds <= span <= maximum_seconds:
                usable.append({"start_seconds": round(float(row["start_seconds"]), 3),
                               "end_seconds": round(float(row["end_seconds"]), 3),
                               "seconds": round(span, 3),
                               "evidence_kinds": _kinds(reading),
                               "identity_state": row.get("identity_state", "unresolved")})
        recipes.append({
            "recipe_id": activity.get("recipe_id"),
            "child_count": len(children),
            "extent_states": states,
            "single_cue_evidence_kinds": kinds,
            "seconds": {"minimum": round(min(seconds), 3) if seconds else None,
                        "median": round(statistics.median(seconds), 3) if seconds else None,
                        "maximum": round(max(seconds), 3) if seconds else None,
                        "total": round(sum(seconds), 3)},
            "cue_sized_single_cue_count": len(usable),
            "cue_sized_single_cue": usable,
        })
        print(f"{activity.get('recipe_id')}: 子候補{len(children)}件 / "
              f"曲の大きさで一曲と言えるもの{len(usable)}件", flush=True)

    for path, digest in before.items():
        if fetch.file_sha256(path) != digest:
            raise CueSizedChildrenProbeError(f"固定入力が変更されています: {path}")

    report = output / "report.json"
    session._write_json_atomic(report, {
        "schema_version": SCHEMA_VERSION,
        "type": "bgm_cue_sized_children_report",
        "status": "complete_real_episode_description",
        "role": "development_real_episode_description_only",
        "promotion_allowed": False, "default_changed": False, "holdout_consumed": False,
        "winner_selected": False, "anything_published": False,
        "source_id": (run.get("source") or {}).get("source_id"),
        "frozen_inputs": [{"path": str(path), "sha256": digest}
                          for path, digest in before.items()],
        "evidence_settings": identity_run.evidence_settings(),
        "rendered_candidate_count": len(run.get("candidates") or []),
        "cue_length_window_seconds": [minimum_seconds, maximum_seconds],
        "decision_rule": DECISION_RULE,
        "activity_recipes": recipes,
        "limits": LIMITS,
        "implementation": [{"path": str(Path(__file__)),
                            "sha256": fetch.file_sha256(Path(__file__))}],
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="記述を作る新規directory")
    parser.add_argument("--run", type=Path, required=True, help="固定した番組run")
    parser.add_argument("--annotation", type=Path, required=True, help="そのrunの注釈")
    parser.add_argument("--minimum-seconds", type=float, default=20.0)
    parser.add_argument("--maximum-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    report = run_probe(args.run.resolve(), args.annotation.resolve(), args.output.resolve(),
                       minimum_seconds=args.minimum_seconds,
                       maximum_seconds=args.maximum_seconds)
    print(f"曲の大きさの子候補を保存しました: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
