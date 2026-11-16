"""BGM activity discoveryの区間候補を版付きcorpusに対して評価する。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.validation.bgm_corpus import BgmCorpus, load_bgm_corpus  # noqa: E402
from lss.bgm.activity import ActivityRecipe, InvalidActivityRecipeError  # noqa: E402
from lss.validation.occurrence_metrics import (  # noqa: E402
    InvalidDetectionError,
    detection_from_dict,
    evaluate_activity_discovery,
    evaluate_occurrences,
    summarize_occurrence_results,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = {
    "type", "schema_version", "case_id", "corpus_source", "analysis_signal",
    "activity_recipe", "activity_implementation", "candidates",
}


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"JSON object keyが重複しています: {key}")
        value[key] = item
    return value


def load_detections(path: Path, corpus: BgmCorpus) -> dict[str, list]:
    cases = {case.case_id: case for case in corpus.cases}
    loaded: dict[str, list] = {}
    identities: set[tuple[str, str]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise InvalidDetectionError(f"検出結果を読めません: {path}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            raise InvalidDetectionError(f"検出結果{line_number}行目のJSONが不正です") from exc
        if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
            raise InvalidDetectionError(
                f"検出結果{line_number}行目のschemaが不正です"
            )
        if value["type"] != "activity_detection" or value["schema_version"] != 1:
            raise InvalidDetectionError(f"検出結果{line_number}行目のversionが不正です")
        case_id = value["case_id"]
        if case_id not in cases:
            raise InvalidDetectionError(f"corpusにないcase_idです: {case_id}")
        if case_id in loaded:
            raise InvalidDetectionError(f"検出結果のcase_idが重複しています: {case_id}")
        case = cases[case_id]
        expected_source = {
            "sha256": case.source_sha256, "sample_rate": case.sample_rate, "frames": case.frames,
        }
        if value["corpus_source"] != expected_source:
            raise InvalidDetectionError(f"case {case_id}のcorpus source identityが一致しません")
        signal = value["analysis_signal"]
        if (
            not isinstance(signal, dict)
            or set(signal) != {"sha256", "sample_rate", "frames"}
            or not isinstance(signal.get("sha256"), str)
            or not _SHA256.fullmatch(signal["sha256"])
            or signal.get("sample_rate") != case.sample_rate
            or signal.get("frames") != case.frames
        ):
            raise InvalidDetectionError(f"case {case_id}のanalysis signal identityが不正です")
        recipe_evidence = value["activity_recipe"]
        try:
            if not isinstance(recipe_evidence, dict) or set(recipe_evidence) != {"recipe_id", "revision"}:
                raise ValueError
            recipe = ActivityRecipe(**recipe_evidence["revision"])
            recipe.validate()
        except (TypeError, ValueError, InvalidActivityRecipeError) as exc:
            raise InvalidDetectionError(f"case {case_id}のactivity Recipeが不正です") from exc
        if recipe_evidence["recipe_id"] != recipe.recipe_id:
            raise InvalidDetectionError(f"case {case_id}のactivity Recipe IDが一致しません")
        implementation = value["activity_implementation"]
        if (
            not isinstance(implementation, dict)
            or set(implementation) != {"algorithm", "artifact_sha256", "numpy_version"}
            or implementation.get("algorithm") != recipe.method
            or not isinstance(implementation.get("artifact_sha256"), str)
            or not _SHA256.fullmatch(implementation["artifact_sha256"])
            or not isinstance(implementation.get("numpy_version"), str)
            or not implementation["numpy_version"]
        ):
            raise InvalidDetectionError(f"case {case_id}のactivity実装identityが不正です")
        identities.add((recipe.recipe_id, implementation["artifact_sha256"]))
        if not isinstance(value["candidates"], list):
            raise InvalidDetectionError(f"case {case_id}のcandidatesは配列である必要があります")
        loaded[case_id] = [
            detection_from_dict(item, source_frames=cases[case_id].frames)
            for item in value["candidates"]
        ]
    if len(identities) > 1:
        raise InvalidDetectionError("同じ評価へ異なるactivity Recipeまたは実装を混在できません")
    return loaded


# L1は音楽らしい区間を見つけられたか、L4は欲しい劇伴だけを選べたかを測る。
LAYERS = {"activity-discovery": ("activity_discovery", evaluate_activity_discovery,
                                 "activity_coverage"),
          "semantic-role": ("semantic_role", evaluate_occurrences, "wanted_coverage")}


def evaluate(corpus: BgmCorpus, detections_by_case: dict[str, list],
             layer: str = "activity-discovery") -> list[dict]:
    _, evaluator, _ = LAYERS[layer]
    return [
        evaluator(case, detections_by_case[case.case_id] if case.case_id in detections_by_case else None)
        for case in corpus.cases
    ]


def build_summary(corpus: BgmCorpus, results: list[dict],
                  layer: str = "activity-discovery") -> dict:
    summary = summarize_occurrence_results(corpus, results)
    key = LAYERS[layer][2]
    scored = [item for item in results if item.get(key) is not None]
    worst = min(scored, key=lambda item: (item[key], item["case_id"])) if scored else None
    return {
        **summary,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        f"worst_{key}_case": worst["case_id"] if worst else None,
        f"worst_{key}": worst[key] if worst else None,
    }


def write_report(path: Path, summary: dict, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}-", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for record in (summary, *results):
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="版付きBGM corpus JSONL")
    parser.add_argument("detections", type=Path, help="case別のactivity候補 JSONL")
    parser.add_argument("output", type=Path, help="評価結果 JSONL")
    parser.add_argument("--layer", choices=sorted(LAYERS), default="activity-discovery",
                        help="activity-discovery=音楽らしい区間の発見、semantic-role=欲しい劇伴の選別")
    args = parser.parse_args(argv)

    corpus = load_bgm_corpus(args.manifest, verify_sources=True)
    results = evaluate(corpus, load_detections(args.detections, corpus), args.layer)
    summary = build_summary(corpus, results, args.layer)
    write_report(args.output, summary, results)
    print(
        f"BGM評価完了: {summary['case_count']} cases, "
        f"hard failures={summary['hard_failure_count']}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
