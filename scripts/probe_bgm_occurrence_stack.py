"""同じ録音の複数の回を重ねたとき、音楽がどれだけ取り出せるかを制御corpusで測る。

**この測定がどの選択肢を決めるか**：回ごとの推定の束ね方（中央値／平均／最小振幅／
最も当てはまる1つ）と、フィルタの長さ・ブロック長。効果が無ければ線を保留する。

制御corpus（`lv-bgm-002-recurrence-r1〜r3`）には、同じ録音が回ごとに違う音量・違う台詞・
違うduckingで1〜6回鳴る番組と、**回ごとの正解の音楽stem**（`references/<話>/music.flac`）が
ある。したがって試聴せずに答え合わせができる。

**回どうしは録音の同じ場所を鳴らしているとは限らない。** このcorpusは回ごとに録音の別の区間を
使う。したがって比べるのは「回の全体」ではなく、**録音として重なっている部分だけ**。
重なりは `construction.json` の source/output の対応から正確に求める。製品の経路でも
候補の区間は再演探しが返した共有区間なので、同じ条件になる。

比較の基準は「原本の区間をそのまま使った場合」。推定がそれより良くならなければ意味がない。

    .venv/Scripts/python.exe scripts/probe_bgm_occurrence_stack.py \
        --corpus .state/validation/lv-bgm-002-recurrence-r2 --out <record dir>

分離modelを使わない。参照録音も使わない。意味役割は判定しない。
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

from lss.bgm import occurrence_stack as stack  # noqa: E402

# 掃引する宣言。当てはめは設定ごとに1回で済ませ、束ね方だけを後から変える。
TAPS = (9, 17)
BLOCK_SECONDS = (0.1, 0.25)
COMBINES = (stack.COMBINE_MEDIAN, stack.COMBINE_MEAN, stack.COMBINE_MIN_MAGNITUDE,
            stack.COMBINE_BEST, stack.COMBINE_DEBIASED, stack.COMBINE_JOINT,
            stack.COMBINE_TWO_STAGE)
# 1回の区間で見る長さの上限。費用を抑えるためで、範囲の主張には使わない。
# 20秒あれば制御corpusの全20区間が台詞と重なる（40dB未満）。
MAX_SECONDS = 20.0
# これより綺麗な区間は、取り除くものが無いので改善量を問わない。
ALREADY_CLEAN_DB = 40.0
# これより短い共有部分は、録音の同定でも扱わない長さなので比べない。
MIN_SHARED_SECONDS = 6.0


def ratio_db(truth: np.ndarray, estimate: np.ndarray) -> float:
    """正解の音楽に対する、推定の誤差の小ささ（dB）。大きいほど良い。"""
    error = float(np.sum((truth - estimate) ** 2))
    energy = float(np.sum(truth ** 2))
    if energy <= 0.0:
        return float("nan")
    if error <= 0.0:
        return 120.0
    return float(min(120.0, 10.0 * np.log10(energy / error)))


def read_span(path: Path, start: int, frames: int) -> np.ndarray:
    with sf.SoundFile(str(path)) as stream:
        stream.seek(start)
        return stream.read(frames, dtype="float64", always_2d=True)


def load_occurrences(corpus: Path) -> tuple[dict, dict]:
    """話ごとの音声の場所と、cueごとの使用箇所（録音のどこを鳴らしているかを含む）。"""
    episodes = {}
    for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") != "case":
            continue
        episodes[record["case_id"]] = {
            "mix": corpus / record["source"]["path"],
            "music": corpus / "references" / record["case_id"] / "music.flac",
            "sample_rate": record["source"]["sample_rate"],
        }

    construction = json.loads((corpus / "construction.json").read_text(encoding="utf-8"))
    cues = defaultdict(list)
    for case in construction["cases"]:
        for component in case.get("components") or []:
            occurrence = component.get("occurrence") or {}
            cue_id = occurrence.get("cue_identity_id")
            if cue_id is None:
                continue
            cues[cue_id].append({
                "episode": case["case_id"],
                "output_start": component["output_start_frame"],
                "output_end": component["output_end_frame"],
                "source_start": component["source_start_frame"],
                "source_end": component["source_end_frame"],
                "fade": component.get("fade_frames") or 0,
                "gain_db": component.get("gain_db"),
            })
    return episodes, dict(cues)


def shared_window(target: dict, other: dict) -> tuple[int, int] | None:
    """二つの使用箇所が**録音として**重なっている部分を、番組の時刻で返す（targetの側）。"""
    low = max(target["source_start"] + target["fade"], other["source_start"] + other["fade"])
    high = min(target["source_end"] - target["fade"], other["source_end"] - other["fade"])
    if high <= low:
        return None
    return (target["output_start"] + (low - target["source_start"]),
            target["output_start"] + (high - target["source_start"]))


def mapped_window(occurrence: dict, source_low: int, source_high: int) -> tuple[int, int]:
    """録音の区間を、その使用箇所の番組の時刻へ直す。"""
    return (occurrence["output_start"] + (source_low - occurrence["source_start"]),
            occurrence["output_start"] + (source_high - occurrence["source_start"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, action="append", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)

    rows = []
    skipped = {"already_clean": 0, "no_music": 0, "no_shared_recording": 0, "too_short": 0}
    started = time.perf_counter()
    for corpus_path in args.corpus:
        corpus = corpus_path.resolve()
        episodes, cues = load_occurrences(corpus)
        if not (corpus / "references").is_dir():
            raise SystemExit(f"正解の音楽stemがありません: {corpus}")
        for cue_id, occurrences in sorted(cues.items()):
            if len(occurrences) < 2:
                continue
            rate = episodes[occurrences[0]["episode"]]["sample_rate"]
            limit = int(round(args.max_seconds * rate))
            minimum = int(round(MIN_SHARED_SECONDS * rate))
            for position, target in enumerate(occurrences):
                # targetと録音として重なる回だけを使い、重なりの共通部分で比べる。
                usable = []
                low = target["source_start"] + target["fade"]
                high = target["source_end"] - target["fade"]
                for index, other in enumerate(occurrences):
                    if index == position:
                        continue
                    window = shared_window(target, other)
                    if window is None:
                        continue
                    usable.append(other)
                    low = max(low, other["source_start"] + other["fade"])
                    high = min(high, other["source_end"] - other["fade"])
                if not usable:
                    skipped["no_shared_recording"] += 1
                    continue
                if high - low < minimum:
                    skipped["too_short"] += 1
                    continue
                high = min(high, low + limit)
                frames = high - low
                episode = episodes[target["episode"]]
                target_start, _ = mapped_window(target, low, high)
                truth = read_span(episode["music"], target_start, frames)
                mix = read_span(episode["mix"], target_start, frames)
                others = []
                for other in usable:
                    other_start, _ = mapped_window(other, low, high)
                    others.append(read_span(episodes[other["episode"]]["mix"],
                                            other_start, frames))
                baseline = ratio_db(truth, mix)
                if not np.isfinite(baseline):
                    skipped["no_music"] += 1
                    continue  # 正解の音楽が無い区間。改善量を定義できない
                if baseline >= ALREADY_CLEAN_DB:
                    skipped["already_clean"] += 1
                    continue  # 既に音楽だけの区間。取り除くものが無い
                for taps in TAPS:
                    for block in BLOCK_SECONDS:
                        recipe = stack.OccurrenceStackRecipe(
                            fir_taps=taps, block_seconds=block, combine=stack.COMBINE_MEDIAN)
                        fitted, aligned, lags = [], [], []
                        for other in others:
                            shifted, lag = stack.align_to(mix, other, sample_rate=rate)
                            if not np.any(shifted):
                                continue
                            aligned.append(shifted)
                            lags.append(lag)
                            fitted.append(stack.fit_explained(
                                mix, shifted, recipe, sample_rate=rate))
                        if not fitted:
                            continue
                        for combine in COMBINES:
                            if combine == stack.COMBINE_JOINT:
                                music = stack.fit_explained_jointly(
                                    mix, aligned, recipe, sample_rate=rate)
                            elif combine == stack.COMBINE_TWO_STAGE:
                                music = stack.two_stage_estimate(
                                    mix, aligned, recipe, sample_rate=rate)
                            else:
                                music = stack.combine_estimates(fitted, combine, target=mix)
                            estimate = ratio_db(truth, music)
                            rows.append({
                                "corpus": corpus.name,
                                "cue": cue_id,
                                "episode": target["episode"],
                                "shared_source_start": low,
                                "seconds": round(frames / rate, 3),
                                "occurrence_count": len(usable) + 1,
                                "others_used": len(fitted),
                                "max_abs_lag_samples": int(max(abs(v) for v in lags)),
                                "fir_taps": taps,
                                "block_seconds": block,
                                "combine": combine,
                                "baseline_db": round(baseline, 3),
                                "estimate_db": round(estimate, 3),
                                "gain_db": round(estimate - baseline, 3),
                            })
            print(json.dumps({"cue": cue_id, "occurrences": len(occurrences),
                              "rows": len(rows)}, ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - started

    def summarise(subset):
        if not subset:
            return None
        gains = np.array([row["gain_db"] for row in subset], dtype=np.float64)
        return {"count": int(gains.size), "median_gain_db": round(float(np.median(gains)), 3),
                "min_gain_db": round(float(gains.min()), 3),
                "max_gain_db": round(float(gains.max()), 3),
                "improved": int((gains > 0).sum())}

    by_combine = {combine: summarise([row for row in rows if row["combine"] == combine])
                  for combine in COMBINES}
    by_setting = {f"taps={taps},block={block}": summarise(
        [row for row in rows if row["fir_taps"] == taps and row["block_seconds"] == block])
        for taps in TAPS for block in BLOCK_SECONDS}
    by_count = {str(count): summarise([row for row in rows if row["occurrence_count"] == count])
                for count in sorted({row["occurrence_count"] for row in rows})}

    best = max((row for row in rows), key=lambda row: row["gain_db"], default=None)
    best_setting = max(
        ((key, value) for key, value in by_setting.items() if value),
        key=lambda item: item[1]["median_gain_db"], default=(None, None))
    best_combine = max(
        ((key, value) for key, value in by_combine.items() if value),
        key=lambda item: item[1]["median_gain_db"], default=(None, None))

    payload = {
        "probe": "bgm-occurrence-stack",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": "同じ録音の別の回で説明できるぶんを音楽としたときの、正解の音楽stemに対する改善量。"
                 "分離modelも参照録音も使っていない",
        "decides_between": ["束ね方（中央値／平均／最小振幅／最も当てはまる1つ／縮み戻し／同時当てはめ）",
                            "フィルタの長さとブロック長", "効果が無ければ保留"],
        "max_seconds": args.max_seconds,
        "row_count": len(rows),
        "skipped_occurrences": skipped,
        "already_clean_db": ALREADY_CLEAN_DB,
        "by_combine": by_combine,
        "by_setting": by_setting,
        "by_occurrence_count": by_count,
        "best_setting": best_setting[0],
        "best_combine": best_combine[0],
        "best_row": best,
        "seconds": round(elapsed, 1),
        "rows": rows,
        "limits": [
            "制御corpusでの測定であり、実番組の再演で同じ改善が出るとは限らない",
            "正解の音楽stemはmixの構成要素そのもので、実番組には存在しない",
            "1回の区間で見る長さに上限を置いている。範囲の主張には使わない",
            "改善量は音楽の取り出しやすさであり、欲しい劇伴かどうかの判定ではない",
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
                      ("row_count", "by_combine", "by_occurrence_count",
                       "best_setting", "best_combine", "seconds")}, ensure_ascii=False))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
