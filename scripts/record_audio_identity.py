"""既存の完了記録へ、復号sampleの同一性を後から残す。音声は読むだけ。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lss.core import session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=None,
                        help="完了台帳を持つ状態領域。省略時は通常の保存先。")
    parser.add_argument("--dry-run", action="store_true",
                        help="台帳を書き換えず、記録できる件数だけを数える。")
    args = parser.parse_args()
    summary = session.record_missing_audio_identity(state_root=args.state_root, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
