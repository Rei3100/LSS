"""分離済みmusic-like信号からBGM候補区間JSONLを生成する。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.bgm.activity import discover_activity, load_activity_recipe  # noqa: E402
from lss.validation.activity_evidence import (  # noqa: E402
    build_activity_detection_record,
    file_sha256,
)
from lss.validation.bgm_corpus import load_bgm_corpus  # noqa: E402


def write_detection(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}-", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="case source identityを固定するBGM corpus JSONL")
    parser.add_argument("input", type=Path, help="source timelineを保つmusic-like音声")
    parser.add_argument("recipe", type=Path, help="activity Recipe JSON")
    parser.add_argument("case_id", help="BGM corpusのcase_id")
    parser.add_argument("output", type=Path, help="検出区間 JSONL")
    args = parser.parse_args(argv)

    corpus = load_bgm_corpus(args.manifest, verify_sources=True)
    matches = [case for case in corpus.cases if case.case_id == args.case_id]
    if len(matches) != 1:
        raise ValueError(f"corpusにcase_idが一意に存在しません: {args.case_id}")
    case = matches[0]
    input_sha256 = file_sha256(args.input)
    samples, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if file_sha256(args.input) != input_sha256:
        raise ValueError(f"activity解析中にinputが変更されました: {args.input}")
    recipe = load_activity_recipe(args.recipe)
    result = discover_activity(samples.T, sample_rate, recipe)
    if file_sha256(case.source_path) != case.source_sha256:
        raise ValueError(f"activity解析中にcorpus sourceが変更されました: {case.source_path}")
    record = build_activity_detection_record(
        case, args.input, recipe, result, analysis_signal_sha256=input_sha256,
    )
    write_detection(args.output, record)
    print(
        f"BGM activity候補: {len(result.candidates)}件, "
        f"recipe={result.recipe_id}, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
