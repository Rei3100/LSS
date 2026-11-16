"""重ね合わせと既定の分離器を、どの順で組むと既発売録音へ最も近づくかを答え合わせする。

決める選択：候補の音楽推定を「重ね合わせのみ（現行）」のままにするか、既定Recipeの
伴奏（台詞・歌を除いたもの）を材料に使う直列へ替えるか。

比べる形（どれも同じ区間・同じ回の組）:

- ``raw``：原本の区間そのまま
- ``stack``：原本mixどうしの重ね合わせ（現行）
- ``sep``：既定Recipeの伴奏の区間そのまま
- ``sep_stack``：伴奏どうしの重ね合わせ

指標は2つ。**片方だけでは薄い切れ端が高得点になる**ので両方を見る。

- ``precision``：短い線形フィルタへ通した既発売録音で、その信号の何割を説明できるか
  （`probe_bgm_released_stack_truth.explained_by` と同じ量）
- ``retention``：その当てはめで説明された成分のエネルギーを、原本の区間での同じ量で割ったもの。
  録音の成分をどれだけ残したか。原本は定義上1
- ``sdr_db``：原本の区間に実際に鳴っている録音の成分（原本を既発売録音で当てはめたもの）を
  正解とみなしたときの信号対誤差比。取りこぼしも混ざりものも誤差に入るので、上の2つを一つにまとめる

伴奏は通常の分離経路が作った既存のファイルを使う（`scripts/prepare_identity_instrumental.py`）。
この測定では推論しない。既発売録音は答え合わせにしか使わない。

    .venv/Scripts/python.exe scripts/probe_bgm_stack_separator_order.py \
        --episode series-ep137=<original.flac> --instrumental series-ep137=<Instrumental.flac> ... \
        --references <Released_Musicのdir> --out <record dir>
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
from scripts.probe_bgm_released_stack_truth import (  # noqa: E402
    AUDIO_SUFFIXES, MAX_SECONDS, MIN_SECONDS, PAD_SECONDS, read_span)

VARIANTS = ("raw", "stack", "sep", "sep_stack")
# 試作：当てはめ先の回そのものへ、回どうしの一致度（coherence）から作った時間周波数の利得をかける。
COHERENCE_SETTINGS = {f"coh_s{smooth}_p{power:g}": (smooth, power)
                      for smooth in (0.25, 0.5, 1.0) for power in (1.0, 2.0)}


def coherence_estimate(target, others, rate, *, smooth_seconds, power,
                       n_fft=2048, hop=512, freq_bins=3):
    """回どうしで一致している時間周波数の成分だけを、当てはめ先の回から残す。

    他の回を当てはめて足すのではないので、他の回の台詞が混ざらない。利得は一致度（0〜1）。
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import istft, stft

    width = max(1, int(round(smooth_seconds * rate / hop))) | 1

    def smooth(values):
        return uniform_filter1d(uniform_filter1d(values, width, axis=1), freq_bins, axis=0)

    out = np.zeros_like(target)
    for channel in range(target.shape[1]):
        _, _, spectrum = stft(target[:, channel], fs=rate, nperseg=n_fft, noverlap=n_fft - hop)
        auto = smooth(np.abs(spectrum) ** 2)
        gains = []
        for other in others:
            _, _, reference = stft(other[:, channel], fs=rate, nperseg=n_fft, noverlap=n_fft - hop)
            cross = spectrum * np.conj(reference)
            joint = np.abs(smooth(cross.real) + 1j * smooth(cross.imag))
            power_product = np.maximum(auto, 0.0) * np.maximum(smooth(np.abs(reference) ** 2), 0.0)
            gains.append(np.nan_to_num(np.clip(joint / np.sqrt(power_product + 1e-20), 0.0, 1.0)))
        gain = np.max(gains, axis=0) ** power
        _, restored = istft(spectrum * gain, fs=rate, nperseg=n_fft, noverlap=n_fft - hop)
        length = min(restored.size, target.shape[0])
        out[:length, channel] = restored[:length]
    return out


def fitted_by(signal: np.ndarray, reference: np.ndarray,
              recipe: stack.OccurrenceStackRecipe, *, sample_rate: int) -> np.ndarray:
    """短い線形フィルタへ通した `reference` で `signal` を当てはめた成分。"""
    if not np.any(signal):
        return np.zeros_like(signal)
    aligned, _ = stack.align_to(signal, reference, sample_rate=sample_rate)
    if not np.any(aligned):
        return np.zeros_like(signal)
    return stack.fit_explained_jointly(signal, [aligned], recipe, sample_rate=sample_rate)


def measure(signal, reference, recipe, *, sample_rate, raw_fitted_energy, truth=None):
    energy = float(np.sum(signal * signal))
    fitted = fitted_by(signal, reference, recipe, sample_rate=sample_rate)
    if energy <= 0.0:
        return {"precision": 0.0, "retention": 0.0, "energy_share": 0.0}, fitted
    residual = float(np.sum((signal - fitted) ** 2))
    fitted_energy = float(np.sum(fitted * fitted))
    error = None if truth is None else float(np.sum((signal - truth) ** 2))
    return {
        "sdr_db": (None if truth is None or not error
                   else round(10.0 * np.log10(float(np.sum(truth * truth)) / error), 3)),
        "precision": round(max(0.0, 1.0 - residual / energy), 6),
        "retention": (round(fitted_energy / raw_fitted_energy, 6)
                      if raw_fitted_energy and raw_fitted_energy > 0 else None),
    }, fitted


def headroom_scale(path: Path) -> float:
    """保存時に下げた振幅を戻す倍率。戻さないと保持の割合が4倍小さく見える。"""
    from mutagen.flac import FLAC

    if Path(path).suffix.lower() != ".flac":
        return 1.0  # WAV等はLSSのheadroom規約で書かれていない
    shift = (FLAC(str(path)).get("LSS_OUTPUT_HEADROOM_SHIFT") or ["0"])[0]
    return float(2.0 ** int(shift))


def estimate_from(mix_path, target, occurrences, position, frames, channels, rate, recipe,
                  scale=lambda label: 1.0):
    mix = read_span(mix_path(target["episode"]), target["query_start_seconds"] - PAD_SECONDS,
                    frames, channels) * scale(target["episode"])
    others = []
    for index_other, other in enumerate(occurrences):
        if index_other == position:
            continue
        window = read_span(mix_path(other["episode"]), other["query_start_seconds"] - PAD_SECONDS,
                           frames, channels) * scale(other["episode"])
        if np.any(window):
            others.append(stack.align_to(mix, window, sample_rate=rate)[0])
    if not others:
        return mix, None
    estimate_from.last_others = others
    return mix, stack.fit_explained_jointly(mix, others, recipe, sample_rate=rate)


def summarise(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    array = np.array(values, dtype=np.float64)
    return {"count": int(array.size), "median": round(float(np.median(array)), 4),
            "p10": round(float(np.percentile(array, 10)), 4),
            "min": round(float(array.min()), 4), "max": round(float(array.max()), 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, action="append", help="label=original.flac")
    parser.add_argument("--instrumental", required=True, action="append",
                        help="label=既定Recipeの伴奏.flac")
    parser.add_argument("--references", required=True, action="append", type=Path)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--self-check", action="store_true",
                        help="既発売録音そのものを信号として通し、precision≈1 を確かめる")
    args = parser.parse_args(argv)

    def pairs(items):
        found = {}
        for item in items:
            label, _, path = item.partition("=")
            found[label] = Path(path).resolve()
        return found

    episodes = pairs(args.episode)
    instrumentals = pairs(args.instrumental)
    if set(episodes) != set(instrumentals):
        raise SystemExit("話と伴奏のラベルが一致しません")
    for label in episodes:
        original, separated = sf.info(str(episodes[label])), sf.info(str(instrumentals[label]))
        # 同じ場所を比べていることを先に確かめる。時間軸がずれていれば測り方が壊れている。
        if (original.samplerate != separated.samplerate or original.channels != separated.channels
                or abs(original.frames - separated.frames) > original.samplerate // 100):
            raise SystemExit(f"原本と伴奏の時間軸が一致しません: {label}")
    scales = {label: headroom_scale(path) for label, path in instrumentals.items()}
    reference_paths: list[Path] = []
    for directory in args.references:
        reference_paths.extend(sorted(p for p in Path(directory).resolve().iterdir()
                                      if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES))

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
                "episode": label, "query_start_seconds": match.query_start_seconds,
                "seconds": min(seconds, MAX_SECONDS),
                "reference_path": match.reference_path,
                "reference_start_seconds": match.reference_start_seconds,
            })

    recipe = stack.OccurrenceStackRecipe(fir_taps=9, block_seconds=0.1,
                                         combine=stack.COMBINE_JOINT)
    rows = []
    self_checks = []
    for reference_sha, occurrences in sorted(groups.items()):
        if len(occurrences) < 2:
            continue
        for position, target in enumerate(occurrences):
            info = sf.info(str(episodes[target["episode"]]))
            rate, channels = int(info.samplerate), int(info.channels)
            seconds = min(target["seconds"], min(item["seconds"] for item in occurrences))
            frames = int(round(seconds * rate))
            if frames < int(MIN_SECONDS * rate):
                continue
            mix, stacked = estimate_from(episodes.__getitem__, target, occurrences, position,
                                         frames, channels, rate, recipe)
            if stacked is None:
                continue
            aligned_others = estimate_from.last_others
            separated, sep_stacked = estimate_from(instrumentals.__getitem__, target, occurrences,
                                                   position, frames, channels, rate, recipe,
                                                   scale=scales.__getitem__)
            if sep_stacked is None:
                continue
            released = read_span(Path(target["reference_path"]),
                                 target["reference_start_seconds"] - PAD_SECONDS, frames, channels,
                                 rate=rate)
            if not np.any(released):
                continue
            raw_fitted = fitted_by(mix, released, recipe, sample_rate=rate)
            raw_fitted_energy = float(np.sum(raw_fitted * raw_fitted))
            measured = {}
            signals = dict(zip(VARIANTS, (mix, stacked, separated, sep_stacked)))
            for name, (smooth_seconds, power) in COHERENCE_SETTINGS.items():
                signals[name] = coherence_estimate(mix, aligned_others, rate,
                                                   smooth_seconds=smooth_seconds, power=power)
            for name, signal in signals.items():
                measured[name], _ = measure(signal, released, recipe, sample_rate=rate,
                                            raw_fitted_energy=raw_fitted_energy, truth=raw_fitted)
                measured[name]["energy_share"] = round(
                    float(np.sum(signal ** 2) / np.sum(mix ** 2)) if np.any(mix) else 0.0, 6)
            if args.self_check and len(self_checks) < 3:
                check, _ = measure(released, released, recipe, sample_rate=rate,
                                   raw_fitted_energy=raw_fitted_energy)
                self_checks.append(check["precision"])
            rows.append({
                "reference_name": Path(target["reference_path"]).name,
                "episode": target["episode"],
                "query_start_seconds": round(target["query_start_seconds"], 3),
                "seconds": round(seconds, 3), "occurrence_count": len(occurrences),
                **measured,
            })
        print(json.dumps({"reference": Path(occurrences[0]["reference_path"]).name,
                          "occurrences": len(occurrences), "rows": len(rows)},
                         ensure_ascii=False), flush=True)

    def compare(first, second):
        both = [(row[first]["precision"], row[second]["precision"],
                 row[first]["retention"], row[second]["retention"]) for row in rows]
        sdr = [(row[first]["sdr_db"], row[second]["sdr_db"]) for row in rows
               if row[first]["sdr_db"] is not None and row[second]["sdr_db"] is not None]
        return {
            "sdr_better": sum(1 for a, b in sdr if b > a),
            "sdr_gain_db": summarise([b - a for a, b in sdr]),
            "precision_better": sum(1 for a, b, *_ in both if b > a),
            "precision_gain": summarise([b - a for a, b, *_ in both]),
            "retention_ratio": summarise([d / c for *_, c, d in both
                                          if c is not None and d is not None and c > 0]),
        }

    payload = {
        "probe": "bgm-stack-separator-order",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": "候補の音楽推定を、重ね合わせのみのままにするか、既定Recipeの伴奏を"
                    "材料にした直列へ替えるか",
        "episodes": {label: str(path) for label, path in sorted(episodes.items())},
        "instrumentals": {label: str(path) for label, path in sorted(instrumentals.items())},
        "instrumental_headroom_scale": scales,
        "references": {"count": len(reference_paths)},
        "matching_settings_id": identity.matching_settings_id(),
        "recipe_id": recipe.recipe_id,
        "self_check_precision": self_checks,
        "row_count": len(rows),
        "variants": {name: {"sdr_db": summarise([row[name]["sdr_db"] for row in rows]),
                            "precision": summarise([row[name]["precision"] for row in rows]),
                            "retention": summarise([row[name]["retention"] for row in rows]),
                            "energy_share": summarise([row[name]["energy_share"] for row in rows])}
                     for name in (*VARIANTS, *COHERENCE_SETTINGS)},
        "comparisons": {"stack_vs_raw": compare("raw", "stack"),
                        "sep_vs_raw": compare("raw", "sep"),
                        "sep_vs_stack": compare("stack", "sep"),
                        "sep_stack_vs_stack": compare("stack", "sep_stack"),
                        "sep_stack_vs_sep": compare("sep", "sep_stack"),
                        **{f"{name}_vs_stack": compare("stack", name) for name in COHERENCE_SETTINGS}},
        "seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
        "limits": [
            "既発売録音を持つcueだけで測っている。未発売cueで同じとは限らない",
            "伴奏は歌と台詞を除く分離器の出力で、効果音は残る。効果音の混同はこの指標に効果音として現れる",
            "説明できる割合と保持の割合であって、聴いた印象ではない。試聴していない",
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
    print(json.dumps({key: payload[key] for key in
                      ("row_count", "self_check_precision", "variants", "comparisons", "seconds")},
                     ensure_ascii=False, indent=1))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
