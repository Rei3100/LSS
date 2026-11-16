"""区間が「音楽だけで鳴っているか」を、原本と既発売録音だけから測る。

BGMの処理は場所によって意味が違う。音楽だけが鳴っている区間は、分離をかけるより
**原本をそのまま取る方が高品質で費用も0**になる。台詞や効果音が重なる区間だけが
分離を必要とし、音楽が無い区間は処理しても意味がない。その振り分けの材料を作る。

教師にする量：`lss/bgm/identity.py` の `waveform_explained`。既発売録音を短い線形
フィルタ（音量・EQ・位相差を許す）へ通して、番組音声をどれだけ説明できるかを表す。
台詞や効果音が重なるほど値が下がるため、**「音楽だけで鳴っている度合い」の直接の測定**
になる。意味的な判断ではなく物理量で、人の試聴を要さない。

重要：照合は**原本mix**に対して行う。分離後の信号へ当てると分離器の挙動が混ざり、
教師として使えなくなる。この probe は分離器を一切使わない。

    .venv/Scripts/python.exe scripts/probe_bgm_region_alone.py \
        --original <original.flac> --references <Released_Musicのdir> [--references <dir>] \
        --out <record dir> [--cache <fingerprint cache dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.bgm import identity  # noqa: E402


def _summary(values: np.ndarray) -> dict | None:
    if not values.size:
        return None
    return {"count": int(values.size),
            "minimum": round(float(values.min()), 4),
            "p25": round(float(np.percentile(values, 25)), 4),
            "median": round(float(np.median(values)), 4),
            "p75": round(float(np.percentile(values, 75)), 4),
            "maximum": round(float(values.max()), 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--references", required=True, action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args(argv)

    original = args.original.resolve()
    reference_paths: list[Path] = []
    for directory in args.references:
        reference_paths.extend(sorted(p for p in Path(directory).resolve().iterdir()
                                      if p.is_file() and p.suffix.lower() in {".flac", ".wav", ".m4a", ".mp3"}))
    if not reference_paths:
        raise SystemExit("参照録音が見つかりません")

    started = time.perf_counter()
    index = identity.ReferenceIndex.build(reference_paths, cache_dir=args.cache)
    index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    signal = identity.decode_mono(original)
    decode_seconds = time.perf_counter() - started

    started = time.perf_counter()
    matches = identity.match(signal, index)
    match_seconds = time.perf_counter() - started

    known = [m for m in matches if m.identity_state == "known_exact" and m.waveform_explained is not None]
    records = []
    for m in known:
        item = m.to_dict()
        item["seconds"] = round(m.query_end_seconds - m.query_start_seconds, 3)
        records.append(item)
    records.sort(key=lambda item: item["query_start_seconds"])

    explained = np.array([item["waveform_explained"] for item in records]) if records else np.zeros(0)
    seconds = np.array([item["seconds"] for item in records]) if records else np.zeros(0)
    bands = {}
    for name, low, high in (("music_alone_0.80+", 0.80, 1.01),
                            ("mixed_0.50-0.80", 0.50, 0.80),
                            ("overlaid_below_0.50", 0.0, 0.50)):
        mask = (explained >= low) & (explained < high)
        bands[name] = {"spans": int(mask.sum()), "minutes": round(float(seconds[mask].sum() / 60), 3)}

    payload = {
        "probe": "bgm-region-alone",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label or original.stem,
        "claim": "既発売録音が原本をどれだけ説明するか。意味的なBGM判定でも公開可否でもない",
        "original": {"path": str(original), "signal_seconds": round(signal.size / identity.SAMPLE_RATE, 3)},
        "references": {"count": len(reference_paths),
                       "directories": [str(Path(d).resolve()) for d in args.references]},
        "matching_settings_id": identity.matching_settings_id(),
        "fingerprint_settings_id": identity.fingerprint_settings_id(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform()},
        "seconds": {"index": round(index_seconds, 2), "decode": round(decode_seconds, 2),
                    "match": round(match_seconds, 2)},
        "known_exact_spans": len(records),
        "matched_minutes": round(float(seconds.sum() / 60), 3),
        "waveform_explained": _summary(explained),
        "bands": bands,
        "spans": records,
        "limits": [
            "原本mixに対する照合であり、分離器は使っていない",
            "既発売録音を持つcueだけが対象。未発売cueはここに現れない",
            "explainedは音楽だけで鳴っている度合いの測定であって、欲しい劇伴かどうかではない",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "report.json"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"label": payload["label"], "spans": len(records),
                      "matched_minutes": payload["matched_minutes"],
                      "explained": payload["waveform_explained"], "bands": bands},
                     ensure_ascii=False), flush=True)
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
