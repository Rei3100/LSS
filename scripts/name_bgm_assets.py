"""既存公開cueへ自動で仮題を付け、独立JSONと表示名付きplaylistを出す。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lss.assets.naming import NamingStore
from lss.bgm.provisional_naming import name_published
from lss.core.session import _write_json_atomic


def run(paths: list[Path], output: Path, *, evidence_by_asset: dict | None = None) -> dict:
    store = NamingStore(output / "names")
    from mutagen.flac import FLAC

    rows = []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        asset_id = "bgm-cue:" + FLAC(path)["LSS_BGM_CUE_KEY"][0]
        extra = (evidence_by_asset or {}).get(asset_id)
        rows.append(name_published(path, store, extra_evidence=extra))
    report = {"type": "bgm_naming_demonstration", "schema_version": 1,
              "source_files_modified": False, "rows": rows}
    _write_json_atomic(output / "report.json", report)
    playlist = ["#EXTM3U"]
    table = []
    for row in rows:
        playlist.extend([f"#EXTINF:{row['duration_seconds']:.3f},{row['title']}", row["path"]])
        basis = json.dumps(row["evidence"], ensure_ascii=False)
        table.append(f"<tr><td>{html.escape(row['title'])}</td><td>{html.escape(row['kind'])}</td>"
                     f"<td><a href='{html.escape(Path(row['path']).as_uri(), quote=True)}'>音声</a></td>"
                     f"<td><details><summary>根拠</summary><pre>{html.escape(basis)}</pre></details></td></tr>")
    (output / "BGM.m3u8").write_text("\n".join(playlist) + "\n", encoding="utf-8")
    (output / "index.html").write_text(
        "<!doctype html><html lang='ja'><meta charset='utf-8'><title>LSS BGMの自動仮題</title>"
        "<style>body{font-family:system-ui;margin:2rem;background:#111827;color:#e5e7eb}"
        "td{padding:1rem;border-bottom:1px solid #374151}a{color:#93c5fd}pre{white-space:pre-wrap}</style>"
        "<h1>BGMの自動仮題</h1><p>「仮題」はLSSの暫定的な記述です。公式曲名ではありません。</p>"
        "<p><a href='BGM.m3u8'>表示名付きプレイリスト</a></p><table>" + "".join(table) + "</table></html>",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="+", type=Path, help="公開BGM FLAC、またはそのフォルダ")
    parser.add_argument("--output", type=Path, required=True, help="命名記録だけの保存先")
    parser.add_argument("--evidence", type=Path, help="asset_idごとの既存の局所context/確認済み同定metadata JSON")
    args = parser.parse_args()
    paths = []
    for path in args.assets:
        paths.extend(path.rglob("*.flac") if path.is_dir() else [path])
    if not paths:
        parser.error("公開BGMが見つかりません")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else None
    if evidence is not None and not isinstance(evidence, dict):
        parser.error("命名証拠はasset_idをkeyに持つobjectが必要です")
    report = run(paths, args.output.resolve(), evidence_by_asset=evidence)
    for row in report["rows"]:
        print(row["title"])


if __name__ == "__main__":
    main()
