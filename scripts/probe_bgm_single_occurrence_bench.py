"""一度しか鳴らない劇伴の取り出し方を、実番組の正解付き区間で比べる。

繰り返しを使う経路は、同じ録音が2回以上鳴るcueにしか効かない。利用者の手元には多くの場合
1話しかなく、多くの劇伴は一度しか鳴らない。その取り出し方（分離器・却下済み部品の組み合わせ）
を、制御corpusではなく実番組で測る。

正解：既発売録音と一致する区間では、原本の区間に実際に鳴っている録音の成分を、短い線形フィルタ
へ通した既発売録音の当てはめとして得られる（`probe_bgm_stack_separator_order.py` と同じ量）。
繰り返しは要らない。各出力を、その成分に対する信号対誤差比（SDR）で比べる。

比べる出力（`--view 話:名前=path` で渡す。全話で揃わない視点はその話では飛ばす）:

- ``raw``：原本の区間そのまま
- 渡した視点そのもの（例：``inst``＝既定Recipeの伴奏、``sfc_music``）
- SFCの3stem（``sfc_music``/``sfc_effects``/``sfc_speech``）が揃う話では組み合わせも作る:
  - ``sfc_mask_raw``：3stemから時間周波数ごとの音楽の配分を作り、原本へかける
  - ``sfc_mask_inst``：同じ配分を ``inst`` へかける
  - ``consensus``：``inst`` と ``sfc_music`` の時間周波数ごとの小さい振幅を取る

分離modelの推論はしない。渡したファイルを読むだけ。既発売録音は正解にしか使わない。

    .venv/Scripts/python.exe scripts/probe_bgm_single_occurrence_bench.py \
        --episode series-ep1=<original.flac> --references <dir> \
        --view series-ep1:inst=<Instrumental.flac> --view series-ep1:sfc_music=<music.wav> ... \
        --out <record dir>
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.bgm import identity, occurrence_stack as stack  # noqa: E402
from scripts.probe_bgm_released_stack_truth import AUDIO_SUFFIXES, PAD_SECONDS, read_span  # noqa: E402
from scripts.probe_bgm_stack_separator_order import fitted_by, headroom_scale  # noqa: E402

MIN_SECONDS = 6.0
MAX_SECONDS = 30.0
N_FFT = 2048
HOP = 512
SFC = ("sfc_music", "sfc_effects", "sfc_speech")


def sdr_db(estimate: np.ndarray, truth: np.ndarray) -> float | None:
    error = float(np.sum((estimate - truth) ** 2))
    energy = float(np.sum(truth * truth))
    if energy <= 0.0 or error <= 0.0:
        return None
    return round(10.0 * np.log10(energy / error), 3)


def _stft(signal: np.ndarray, rate: int):
    from scipy.signal import stft

    return [stft(signal[:, channel], fs=rate, nperseg=N_FFT, noverlap=N_FFT - HOP)[2]
            for channel in range(signal.shape[1])]


def _istft(spectra, rate: int, frames: int) -> np.ndarray:
    from scipy.signal import istft

    out = np.zeros((frames, len(spectra)))
    for channel, spectrum in enumerate(spectra):
        _, restored = istft(spectrum, fs=rate, nperseg=N_FFT, noverlap=N_FFT - HOP)
        length = min(restored.size, frames)
        out[:length, channel] = restored[:length]
    return out


def music_share_mask(views: dict[str, np.ndarray], rate: int):
    """SFCの3stemの時間周波数ごとの音楽の配分（0〜1）。"""
    powers = {name: [np.abs(spectrum) ** 2 for spectrum in _stft(views[name], rate)] for name in SFC}
    return [powers["sfc_music"][channel]
            / (powers["sfc_music"][channel] + powers["sfc_effects"][channel]
               + powers["sfc_speech"][channel] + 1e-20)
            for channel in range(len(powers["sfc_music"]))]


def combinations(raw: np.ndarray, views: dict[str, np.ndarray], rate: int) -> dict[str, np.ndarray]:
    found = {}
    frames = raw.shape[0]
    if all(name in views for name in SFC):
        mask = music_share_mask(views, rate)
        found["sfc_mask_raw"] = _istft([m * s for m, s in zip(mask, _stft(raw, rate))], rate, frames)
        if "inst" in views:
            found["sfc_mask_inst"] = _istft(
                [m * s for m, s in zip(mask, _stft(views["inst"], rate))], rate, frames)
    if "inst" in views and "sfc_music" in views:
        inst, music = _stft(views["inst"], rate), _stft(views["sfc_music"], rate)
        found["consensus"] = _istft(
            [np.where(np.abs(a) <= np.abs(b), a, b) for a, b in zip(inst, music)], rate, frames)
    return found


def summarise(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    array = np.array(values, dtype=np.float64)
    return {"count": int(array.size), "median": round(float(np.median(array)), 3),
            "p10": round(float(np.percentile(array, 10)), 3), "min": round(float(array.min()), 3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, action="append", help="話=original.flac")
    parser.add_argument("--view", action="append", default=[], help="話:名前=path")
    parser.add_argument("--references", required=True, action="append", type=Path)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    episodes = {}
    for item in args.episode:
        label, _, path = item.partition("=")
        episodes[label] = Path(path).resolve()
    views: dict[str, dict[str, Path]] = defaultdict(dict)
    for item in args.view:
        head, _, path = item.partition("=")
        label, _, name = head.partition(":")
        views[label][name] = Path(path).resolve()
    for label, named in views.items():
        original = sf.info(str(episodes[label]))
        for name, path in named.items():
            info = sf.info(str(path))
            # 同じ時間軸の出力だけを比べる。ずれていれば測り方が壊れている。
            if (info.samplerate != original.samplerate
                    or abs(info.frames - original.frames) > original.samplerate // 100):
                raise SystemExit(f"時間軸が原本と一致しません: {label}:{name}")
    reference_paths: list[Path] = []
    for directory in args.references:
        reference_paths.extend(sorted(p for p in Path(directory).resolve().rglob("*")
                                      if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES))

    started = time.perf_counter()
    index = identity.ReferenceIndex.build(reference_paths, cache_dir=args.cache)
    recipe = stack.OccurrenceStackRecipe(fir_taps=9, block_seconds=0.1, combine=stack.COMBINE_JOINT)
    rows = []
    self_checks = []
    for label, path in sorted(episodes.items()):
        info = sf.info(str(path))
        rate, channels = int(info.samplerate), int(info.channels)
        scales = {name: (headroom_scale(view) if view.suffix.lower() == ".flac" else 1.0)
                  for name, view in views.get(label, {}).items()}
        for match in identity.match(identity.decode_mono(path), index):
            if match.identity_state != "known_exact":
                continue
            seconds = min(match.query_end_seconds - match.query_start_seconds, MAX_SECONDS)
            if seconds < MIN_SECONDS:
                continue
            frames = int(round(seconds * rate))
            start = match.query_start_seconds - PAD_SECONDS
            raw = read_span(path, start, frames, channels)
            released = read_span(Path(match.reference_path),
                                 match.reference_start_seconds - PAD_SECONDS, frames, channels, rate=rate)
            if not np.any(released) or not np.any(raw):
                continue
            truth = fitted_by(raw, released, recipe, sample_rate=rate)
            if len(self_checks) < 3:
                self_checks.append(sdr_db(fitted_by(released, released, recipe, sample_rate=rate),
                                          released))
            signals = {"raw": raw}
            for name, view in views.get(label, {}).items():
                signals[name] = read_span(view, start, frames, channels) * scales[name]
            signals.update(combinations(raw, signals, rate))
            rows.append({
                "episode": label, "reference": Path(match.reference_path).name,
                "start_seconds": round(match.query_start_seconds, 3), "seconds": round(seconds, 3),
                "raw_music_share": round(float(np.sum(truth ** 2) / np.sum(raw ** 2)), 4),
                "sdr_db": {name: sdr_db(signal, truth) for name, signal in signals.items()},
            })
        print(json.dumps({"episode": label, "rows": len(rows)}, ensure_ascii=False), flush=True)

    names = sorted({name for row in rows for name in row["sdr_db"]})
    paired = {}
    for name in names:
        if name == "raw":
            continue
        gains = [row["sdr_db"][name] - row["sdr_db"]["raw"] for row in rows
                 if row["sdr_db"].get(name) is not None and row["sdr_db"].get("raw") is not None]
        paired[f"{name}_vs_raw"] = {"better": sum(1 for g in gains if g > 0), **(summarise(gains) or {})}
    payload = {
        "probe": "bgm-single-occurrence-bench",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": "一度しか鳴らない劇伴を1話だけから取り出すとき、どの出力・組み合わせが実番組の録音成分へ最も近いか",
        "episodes": {label: str(path) for label, path in sorted(episodes.items())},
        "views": {label: {name: str(path) for name, path in named.items()} for label, named in views.items()},
        "references": {"count": len(reference_paths)},
        "self_check_sdr_db": self_checks,
        "row_count": len(rows),
        "by_view": {name: summarise([row["sdr_db"].get(name) for row in rows]) for name in names},
        "paired": paired,
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
        "limits": [
            "既発売録音に一致する区間だけ。未発売の劇伴で同じとは限らない",
            "正解は原本を既発売録音で線形に当てはめた成分で、真のstemではない",
            "試聴していない。公開・昇格・既定変更はしていない",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "report.json"
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({k: payload[k] for k in ("row_count", "self_check_sdr_db", "by_view", "paired")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
