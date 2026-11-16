"""既知録音の一致をDevelopment番組で確かめる。Holdoutは使わない。

    python scripts/probe_bgm_identity.py --corpus C:\\LSS-TestCorpus --out .state/validation/bgm-identity-development-r1.json

各作品の References（Released_Music・OP/ED）で索引を作り、同じ作品のDevelopment番組を照合する。
別作品の参照との照合は、同じ録音が入っているはずのない誤検出の対照として数える。
一致しないことは劇伴の不在を意味しない。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lss.bgm import identity  # noqa: E402

AUDIO = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus"}
MEDIA = {".mkv", ".mp4", ".webm", ".ts", ".flac", ".wav", ".m4a"}


def references_of(series: Path) -> list[Path]:
    return sorted(p for p in series.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, default=None,
                        help="番組の代わりに照合する音声（例: 分離した伴奏）を <root>/<作品>/LSS/<番組名> - Instrumental.flac から読む")
    parser.add_argument("--min-block-explained", type=float, default=None,
                        help="Developmentでのしきい値比較用。既定はidentity.MIN_BLOCK_EXPLAINED")
    args = parser.parse_args(argv)
    if args.min_block_explained is not None:
        identity.MIN_BLOCK_EXPLAINED = args.min_block_explained
    references_root = args.corpus / "References"
    sources_root = args.corpus / "Sources" / "Development"

    indexes = {}
    for series in sorted(p for p in references_root.iterdir() if p.is_dir()):
        started = time.monotonic()
        paths = references_of(series)
        indexes[series.name] = identity.ReferenceIndex.build(paths)
        print(json.dumps({"series": series.name, "references": len(paths),
                          "seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)

    episodes = []
    for series in sorted(p for p in sources_root.iterdir() if p.is_dir()):
        for media in sorted(p for p in series.iterdir() if p.is_file() and p.suffix.lower() in MEDIA):
            started = time.monotonic()
            query_path = media
            if args.query_root is not None:
                query_path = args.query_root / series.name / "LSS" / f"{media.stem} - Instrumental.flac"
                if not query_path.is_file():
                    print(json.dumps({"missing_query": str(query_path)}, ensure_ascii=False), flush=True)
                    continue
            samples = identity.decode_mono(query_path)
            spec_frames = max(0, 1 + (samples.size - identity.N_FFT) // identity.HOP)
            entry = {"series": series.name, "source": str(media), "sha256": identity._sha256(media),
                     "query": str(query_path),
                     "seconds": round(spec_frames * identity.FRAME_SECONDS, 1), "own": [], "cross": {}}
            for name, index in indexes.items():
                found = [m.to_dict() for m in identity.match(samples, index)]
                if name == series.name:
                    entry["own"] = found
                else:
                    entry["cross"][name] = found
            entry["elapsed_seconds"] = round(time.monotonic() - started, 1)
            episodes.append(entry)
            own_exact = [m for m in entry["own"] if m["identity_state"] == "known_exact"]
            cross_exact = sum(1 for items in entry["cross"].values() for m in items if m["identity_state"] == "known_exact")
            print(json.dumps({"series": series.name, "own_exact": len(own_exact), "own_candidates": len(entry["own"]),
                              "cross_exact": cross_exact, "elapsed": entry["elapsed_seconds"]},
                             ensure_ascii=False), flush=True)
            for m in own_exact:
                print("   ", m["reference_id"], m["query_start_seconds"], "-", m["query_end_seconds"],
                      "ref@", m["reference_start_seconds"], "z", m["z"], "votes", m["votes"],
                      "explained", m["waveform_explained"], "null", m["waveform_null_max"],
                      "alt", m["alternative_alignments"], flush=True)

    payload = {
        "kind": "bgm_identity_development_probe", "method_version": identity.METHOD_VERSION,
        "thresholds": {"min_votes": identity.MIN_VOTES, "min_seconds": identity.MIN_SECONDS,
                       "min_block_explained": identity.MIN_BLOCK_EXPLAINED, "min_run_blocks": identity.MIN_RUN_BLOCKS,
                       "min_explained_margin": identity.MIN_EXPLAINED_MARGIN},
        "created_at": datetime.now(timezone.utc).isoformat(), "split": "Development",
        "note": "positive exact-recording evidence only; no match is not absence",
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
