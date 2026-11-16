"""解析器の要求する標本化周波数へ、**解析用の複製**を作る。

却下済みSFCをanalyzerとして使うとき、modelは44.1kHz固定で、違う周波数の入力は
adapterが拒否する（無言で変換しない）。一方、実際の素材は48kHzで届く。

このscriptは、測るためだけの複製を別directoryへ作る。元の候補・推定は変えない。
作った複製は**解析用の信号**であり、公開する成果物ではない。何をどう変換したかを
記録に残す。

    .venv/Scripts/python.exe scripts/prepare_analysis_resample.py \
        --rate 44100 --out <解析用dir> <file> [<file> ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHOD = "scipy_resample_poly_analysis_only_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.sources:
        path = source.resolve()
        data, rate = sf.read(str(path), always_2d=True, dtype="float64")
        before = sha256(path)
        if rate == args.rate:
            converted, method = data, "copied_without_conversion"
        else:
            divisor = gcd(args.rate, rate)
            converted = resample_poly(data, args.rate // divisor, rate // divisor, axis=0)
            method = METHOD
        if not np.isfinite(converted).all():
            raise SystemExit(f"変換後の音声が不正です: {path}")
        target = args.out / f"{path.stem}.wav"
        sf.write(str(target), converted, args.rate, subtype="FLOAT")
        if sha256(path) != before:
            raise SystemExit("変換中に元の音声が変わりました")
        rows.append({
            "source": {"path": str(path), "sha256": before, "sample_rate": rate,
                       "frames": int(data.shape[0])},
            "analysis_copy": {"path": str(target.resolve()), "sha256": sha256(target),
                              "sample_rate": args.rate, "frames": int(converted.shape[0]),
                              "representation": "analysis_only_resampled_copy",
                              "publication_state": "not_published"},
            "method": method,
        })

    payload = {
        "type": "analysis_resample",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_sample_rate": args.rate,
        "count": len(rows),
        "converted": sum(1 for row in rows if row["method"] == METHOD),
        "files": rows,
        "claim_boundary": [
            "測るためだけの複製であり、公開する成果物ではない",
            "元の候補・推定は変えていない（前後でhashを確認している）",
            "解析器が要求する周波数へ合わせただけで、品質の主張はしない",
        ],
    }
    record = args.out / "resample.json"
    with record.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"count": len(rows), "converted": payload["converted"],
                      "rate": args.rate}, ensure_ascii=False))
    print(f"record: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
