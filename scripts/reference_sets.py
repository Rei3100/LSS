"""作品ごとの参照録音の集まりへ、手元のファイルやURLから取得した録音を入れ、状態を確かめる。

    python scripts/reference_sets.py add some-series <folder of reference recordings>
    python scripts/reference_sets.py add-url some-series https://music.youtube.com/playlist?list=...
    python scripts/reference_sets.py show some-series
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lss.bgm import references  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add")
    add.add_argument("set_id")
    add.add_argument("paths", type=Path, nargs="+")
    add_url = commands.add_parser("add-url")
    add_url.add_argument("set_id")
    add_url.add_argument("url")
    show = commands.add_parser("show")
    show.add_argument("set_id")
    args = parser.parse_args(argv)
    if args.command == "add":
        added = references.add_local(args.set_id, args.paths)
        print(json.dumps({"added": len(added)}, ensure_ascii=False))
    elif args.command == "add-url":
        added, failures = references.add_acquired(args.set_id, args.url)
        print(json.dumps({"added": len(added), "failed": [
            {"url": f.requested_url, "kind": f.failure.kind.value} for f in failures]}, ensure_ascii=False))
    else:
        resolution = references.resolve(args.set_id)
        print(json.dumps({"available": len(resolution.paths), "missing": resolution.missing}, ensure_ascii=False,
                         indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
