"""現在の既定と違うモデル・設定で作られた結果を、再処理の候補として一覧にする（読み取りのみ）。

    python scripts/list_reprocess_candidates.py [--verify-sources] [--out candidates.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lss.assets.reprocess import reprocess_candidates  # noqa: E402
from lss.core import session  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sources", action="store_true", help="原本の内容SHA-256まで確かめる")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = reprocess_candidates(verify_sources=args.verify_sources)
    except (OSError, ValueError) as exc:
        print(json.dumps({
            "status": "FAILED", "error_code": "reprocess_listing_failed",
            "message": f"再処理候補を確認できません: {exc}",
        }, ensure_ascii=False), file=sys.stderr)
        return 1
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is None:
        print(text)
    else:
        try:
            session._write_json_atomic(args.out, result)
        except (OSError, ValueError) as exc:
            print(json.dumps({
                "status": "FAILED", "error_code": "reprocess_export_failed",
                "message": f"再処理候補の保存に失敗しました: {exc}",
            }, ensure_ascii=False), file=sys.stderr)
            return 1
        print(json.dumps({"counts": result["counts"], "candidates": len(result["candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
