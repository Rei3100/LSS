"""版付きBGM corpusを現行分離backendとactivity Recipe群で実測する。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.validation.bgm_baseline import run_bgm_baseline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("recipes", type=Path, nargs="+", help="activity Recipe JSON群")
    parser.add_argument("output", type=Path, help="新規baseline run directory")
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--model-id")
    stage.add_argument("--source-passthrough", action="store_true")
    parser.add_argument("--stem-target", default="Instrumental")
    args = parser.parse_args(argv)
    scoreboard_path = run_bgm_baseline(
        args.manifest, args.recipes, args.output,
        model_id=None if args.source_passthrough else args.model_id,
        stem_target=args.stem_target,
    )
    scoreboard = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    best = scoreboard["recipes"][0]
    print(
        f"BGM baseline完了（未昇格）: best={best['recipe_id']}, "
        f"hard failures={best['summary']['hard_failure_count']}, output={scoreboard_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
