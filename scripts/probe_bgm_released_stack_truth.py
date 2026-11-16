"""重ね合わせが実番組でも音楽を取り出せているかを、既発売録音で答え合わせする。

制御corpusでは正解の音楽stemがあるので測れた。実番組には無い——**ただし既発売cueなら
録音そのものが手元にある**。同じ既発売録音が複数の話に出ていれば、その回どうしを重ねて
音楽を推定し、推定が既発売録音でどれだけ説明できるかを見られる。推定は他の話の原本mixだけから
作るので、**既発売録音は答え合わせにしか使っていない**。

測るのは音量やEQに依らない量：短い線形フィルタへ通した既発売録音で、その信号の何割を
説明できるか。原本の区間そのままと推定とで比べる。推定のほうが高ければ、
**乗っていたものが減って録音そのものへ近づいた**ということ。

`probe_bgm_occurrence_stack.py` の制御corpus版に対する、実番組版。

    .venv/Scripts/python.exe scripts/probe_bgm_released_stack_truth.py \
        --episode series-ep1=<original.flac> --episode series-ep8=<...> \
        --references <Released_Musicのdir> --out <record dir>

分離modelを使わない。意味役割は判定しない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.bgm import identity, occurrence_stack as stack  # noqa: E402

AUDIO_SUFFIXES = {".flac", ".wav", ".m4a", ".mp3", ".opus", ".ogg", ".aac"}
MIN_SECONDS = 6.0
MAX_SECONDS = 30.0
PAD_SECONDS = 0.5


def read_span(path: Path, start_seconds: float, frames: int, channels: int, *,
              rate: int | None = None) -> np.ndarray:
    """指定の位置・長さで読む。`rate` を渡すとその周波数へ合わせる（測るためだけの変換）。

    既発売録音と番組の標本化周波数は揃っているとは限らない。揃えずに比べると、
    同じ録音でも線形フィルタでは繋がらず、何も説明できないように見える。
    """
    with sf.SoundFile(str(path)) as stream:
        source_rate = stream.samplerate
        if rate is not None and source_rate != rate:
            wanted = int(np.ceil(frames * source_rate / rate)) + source_rate
            raw = read_span(path, start_seconds, wanted, channels)
            divisor = gcd(rate, source_rate)
            converted = resample_poly(raw, rate // divisor, source_rate // divisor, axis=0)
            out = np.zeros((frames, channels), dtype=np.float64)
            length = min(frames, converted.shape[0])
            out[:length] = converted[:length]
            return out
        rate = source_rate
        begin = int(round(start_seconds * rate))
        out = np.zeros((frames, channels), dtype=np.float64)
        if begin >= len(stream):
            return out
        stream.seek(max(0, begin))
        data = stream.read(frames + min(0, begin), dtype="float64", always_2d=True)
        if data.shape[1] > channels:
            data = data[:, :channels]
        offset = max(0, -begin)
        length = min(frames - offset, data.shape[0])
        if length > 0:
            out[offset:offset + length, :data.shape[1]] = data[:length]
    return out


def explained_by(signal: np.ndarray, reference: np.ndarray,
                 recipe: stack.OccurrenceStackRecipe, *, sample_rate: int) -> float:
    """短い線形フィルタへ通した `reference` で、`signal` の何割を説明できるか。"""
    energy = float(np.sum(signal * signal))
    if energy <= 0.0:
        return 0.0
    aligned, _ = stack.align_to(signal, reference, sample_rate=sample_rate)
    if not np.any(aligned):
        return 0.0
    fitted = stack.fit_explained_jointly(signal, [aligned], recipe, sample_rate=sample_rate)
    return float(max(0.0, 1.0 - float(np.sum((signal - fitted) ** 2)) / energy))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, action="append", help="label=original.flac")
    parser.add_argument("--references", required=True, action="append", type=Path)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fir-taps", type=int, default=9)
    parser.add_argument("--block-seconds", type=float, default=0.1)
    args = parser.parse_args(argv)

    episodes = {}
    for item in args.episode:
        label, _, path = item.partition("=")
        episodes[label] = Path(path).resolve()
    reference_paths: list[Path] = []
    for directory in args.references:
        reference_paths.extend(sorted(p for p in Path(directory).resolve().iterdir()
                                      if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES))
    if not reference_paths:
        raise SystemExit("既発売録音が要ります")

    started = time.perf_counter()
    index = identity.ReferenceIndex.build(reference_paths, cache_dir=args.cache)
    groups: dict[str, list[dict]] = defaultdict(list)
    for label, path in sorted(episodes.items()):
        for match in identity.match(identity.decode_mono(path), index):
            if match.identity_state != "known_exact":
                continue
            seconds = match.query_end_seconds - match.query_start_seconds
            if seconds < MIN_SECONDS:
                continue
            groups[match.reference_sha256].append({
                "episode": label,
                "query_start_seconds": match.query_start_seconds,
                "seconds": min(seconds, MAX_SECONDS),
                "reference_path": match.reference_path,
                "reference_start_seconds": match.reference_start_seconds,
            })
    match_seconds = time.perf_counter() - started

    recipe = stack.OccurrenceStackRecipe(
        fir_taps=args.fir_taps, block_seconds=args.block_seconds,
        combine=stack.COMBINE_JOINT)
    rows = []
    for reference_sha, occurrences in sorted(groups.items()):
        if len(occurrences) < 2:
            continue
        for position, target in enumerate(occurrences):
            path = episodes[target["episode"]]
            info = sf.info(str(path))
            rate, channels = int(info.samplerate), int(info.channels)
            seconds = min(target["seconds"], min(item["seconds"] for item in occurrences))
            frames = int(round(seconds * rate))
            if frames < int(MIN_SECONDS * rate):
                continue
            mix = read_span(path, target["query_start_seconds"] - PAD_SECONDS, frames, channels)
            others = []
            for index_other, other in enumerate(occurrences):
                if index_other == position:
                    continue
                window = read_span(episodes[other["episode"]],
                                   other["query_start_seconds"] - PAD_SECONDS, frames, channels)
                if np.any(window):
                    others.append(stack.align_to(mix, window, sample_rate=rate)[0])
            if not others:
                continue
            estimate = stack.fit_explained_jointly(mix, others, recipe, sample_rate=rate)
            released = read_span(Path(target["reference_path"]),
                                 target["reference_start_seconds"] - PAD_SECONDS, frames, channels,
                                 rate=rate)
            if not np.any(released):
                continue
            raw_explained = explained_by(mix, released, recipe, sample_rate=rate)
            estimate_explained = explained_by(estimate, released, recipe, sample_rate=rate)
            rows.append({
                "reference_sha256": reference_sha,
                "reference_name": Path(target["reference_path"]).name,
                "episode": target["episode"],
                "query_start_seconds": round(target["query_start_seconds"], 3),
                "seconds": round(seconds, 3),
                "occurrence_count": len(occurrences),
                "others_used": len(others),
                "raw_explained_by_released": round(raw_explained, 6),
                "estimate_explained_by_released": round(estimate_explained, 6),
                "gain": round(estimate_explained - raw_explained, 6),
                "estimate_energy_share": round(
                    float(np.sum(estimate ** 2) / np.sum(mix ** 2)) if np.any(mix) else 0.0, 6),
            })
        print(json.dumps({"reference": Path(occurrences[0]["reference_path"]).name,
                          "occurrences": len(occurrences), "rows": len(rows)},
                         ensure_ascii=False), flush=True)

    def summarise(values):
        if not values:
            return None
        array = np.array(values, dtype=np.float64)
        return {"count": int(array.size), "median": round(float(np.median(array)), 4),
                "min": round(float(array.min()), 4), "max": round(float(array.max()), 4)}

    payload = {
        "probe": "bgm-released-stack-truth",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": "重ね合わせた推定が、原本の区間より既発売録音そのものへ近づいているか。"
                 "推定は他の話の原本mixだけから作り、既発売録音は答え合わせにしか使っていない",
        "episodes": {label: str(path) for label, path in sorted(episodes.items())},
        "references": {"count": len(reference_paths)},
        "matching_settings_id": identity.matching_settings_id(),
        "recipe_id": recipe.recipe_id,
        "recipe": recipe.to_dict(),
        "recording_groups": sum(1 for items in groups.values() if len(items) >= 2),
        "row_count": len(rows),
        "raw_explained_by_released": summarise(
            [row["raw_explained_by_released"] for row in rows]),
        "estimate_explained_by_released": summarise(
            [row["estimate_explained_by_released"] for row in rows]),
        "gain": summarise([row["gain"] for row in rows]),
        "improved": sum(1 for row in rows if row["gain"] > 0),
        "estimate_energy_share": summarise([row["estimate_energy_share"] for row in rows]),
        "seconds": {"match": round(match_seconds, 1),
                    "total": round(time.perf_counter() - started, 1)},
        "rows": rows,
        "limits": [
            "既発売録音を持つcueだけで測っている。未発売cueで同じとは限らない",
            "既発売録音は番組と標本化周波数が違うので、測るためだけに合わせている",
            "説明できる割合であって、聴いた印象ではない。試聴していない",
            "推定は元mixから線形に取り出したもので、劇伴の完成品ではない",
            "取り出せなかった音楽は残差へ落ちる。残った割合も併記している",
            "公開・昇格・既定変更はしていない",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "report.json"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: payload[k] for k in
                      ("row_count", "recording_groups", "raw_explained_by_released",
                       "estimate_explained_by_released", "gain", "improved",
                       "estimate_energy_share", "seconds")}, ensure_ascii=False))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
