"""保存した未発売候補の中身を、却下済み分離器をanalyzerとして使って測る。

`materialize_bgm_series_candidates.py` が保存した候補は「同じ録音が繰り返されている区間」
でしかない。そこに何が鳴っているか（音楽か、台詞か、効果音か）は未判定のまま。
このprobeは候補の音声をspeech/music/effectsへ分け、元mixに対する取り分をdBで測る。

最終rendererとして却下したmodelを、別のrole（analyzer）で使う。`AGENTS.md`のとおり、
落ちた候補は部品として保持する。ここで得るのは**意味役割の判定ではなく、証拠の一本**。

    # 1. 候補の音声をstemへ分ける（GPU）
    .providers/sfc-locoformer/.venv/Scripts/python.exe scripts/infer_bgm_sfc.py \
        --model <model dir> --out <stem dir> <candidates/*.flac>
    # 2. 取り分を測る（CPU）
    .venv/Scripts/python.exe scripts/probe_bgm_candidate_content.py \
        --record <*-candidates/record.json> --stems <stem dir> --out <record dir>

試聴していない。公開・昇格・既定変更はしていない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lss.bgm import content_guard, series_cues  # noqa: E402

STEMS = ("speech", "music", "effects")
# 音楽が台詞・効果音を上回っていることだけを要求する。余裕は取らない
MUSIC_OVER_SPEECH_DB = 0.0
MUSIC_OVER_EFFECTS_DB = 0.0


def level_db(signal: np.ndarray, reference: float) -> float:
    """元mixのRMSに対する、そのstemのRMS（dB）。"""
    value = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    if reference <= 0.0 or value <= 0.0:
        return -120.0
    return float(20.0 * np.log10(value / reference))


def read_signal(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), always_2d=True, dtype="float32")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, action="append", type=Path,
                        help="series_cue_run の series-cues.json（または同じ形のrecord.json）")
    parser.add_argument("--stems", required=True, action="append", type=Path,
                        help="infer_bgm_sfc.py の出力dir")
    parser.add_argument("--analysis-dir", type=Path, default=None,
                        help="解析用に周波数を合わせた複製のdir（`prepare_analysis_resample.py`）")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    stem_roots = [path.resolve() for path in args.stems]

    analysis_dir = args.analysis_dir.resolve() if args.analysis_dir is not None else None

    def shares(source: Path):
        """その音声そのものを基準にした、stemごとの取り分。見つからなければ None。"""
        found = None
        for root in stem_roots:
            if (root / source.stem).is_dir():
                found = root / source.stem
                break
        if found is None:
            return None
        # stemは解析用の複製から作られているので、基準も同じ複製から取る。
        measured = source
        if analysis_dir is not None and (analysis_dir / f"{source.stem}.wav").is_file():
            measured = analysis_dir / f"{source.stem}.wav"
        signal = read_signal(measured)
        reference = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
        levels = {}
        for stem in STEMS:
            stem_path = found / f"{stem}.wav"
            if not stem_path.is_file():
                raise SystemExit(f"stemが足りません: {stem_path}")
            levels[stem] = round(level_db(read_signal(stem_path), reference), 3)
        return levels

    rows, missing = [], []
    for record_path in args.record:
        record = json.loads(record_path.resolve().read_text(encoding="utf-8"))
        series = record_path.resolve().parent.name.replace("-candidates", "").replace("stacked-", "")
        for candidate in record["candidates"]:
            source = Path(candidate["artifact"]["path"])
            levels = shares(source)
            if levels is None:
                missing.append(source.stem)
                continue
            row = {
                "name": source.stem,
                "series": series,
                "target": series_cues.GUARD_TARGET_OCCURRENCE,
                "recording_key": candidate["recording_key"],
                "source_id": candidate["source_id"],
                "start_seconds": candidate["start_seconds"],
                "seconds": candidate["seconds"],
                "occurrence_count": candidate.get("occurrence_count"),
                "episode_count": candidate.get("episode_count"),
                "explained_median": candidate.get("explained_median"),
                **{f"{stem}_db": levels[stem] for stem in STEMS},
                "music_minus_effects_db": round(levels["music"] - levels["effects"], 3),
                "music_minus_speech_db": round(levels["music"] - levels["speech"], 3),
            }
            estimate = candidate.get("music_estimate")
            if estimate:
                estimate_levels = shares(Path(estimate["path"]))
                if estimate_levels is None:
                    missing.append(Path(estimate["path"]).stem)
                    continue
                row["estimate"] = {
                    "name": Path(estimate["path"]).stem,
                    "recording_key": candidate["recording_key"],
                    # recording_keyはシリーズの中でだけ一意なので、行に作品名を持たせる。
                    "series": series,
                    "target": series_cues.GUARD_TARGET_ESTIMATE,
                    "recipe_id": estimate["recipe_id"],
                    "combined_explained": estimate["combined_explained"],
                    "occurrence_count": estimate["occurrence_count"],
                    **{f"{stem}_db": estimate_levels[stem] for stem in STEMS},
                    "music_minus_effects_db": round(
                        estimate_levels["music"] - estimate_levels["effects"], 3),
                    "music_minus_speech_db": round(
                        estimate_levels["music"] - estimate_levels["speech"], 3),
                }
                row["estimate"]["effects_change_db"] = round(
                    row["estimate"]["music_minus_effects_db"] - row["music_minus_effects_db"], 3)
                row["estimate"]["speech_change_db"] = round(
                    row["estimate"]["music_minus_speech_db"] - row["music_minus_speech_db"], 3)
            rows.append(row)

    if missing:
        raise SystemExit(f"stemが見つからない候補があります: {missing}")

    def summary(values: list[float]) -> dict | None:
        if not values:
            return None
        array = np.array(values, dtype=np.float64)
        return {"count": int(array.size), "median": round(float(np.median(array)), 3),
                "min": round(float(array.min()), 3), "max": round(float(array.max()), 3)}

    recipe = content_guard.ContentGuardRecipe(
        music_over_speech_db=MUSIC_OVER_SPEECH_DB,
        music_over_effects_db=MUSIC_OVER_EFFECTS_DB)

    def screen(subset, key):
        """関門をかけ、行ごとの結果を書き戻す。かけどころが違えば結果も違う。"""
        result = content_guard.screen_candidates(subset, recipe)
        state = {item["name"]: content_guard.KEEP for item in result["kept"]}
        state.update({item["name"]: item["reason"] for item in result["rejected"]})
        for item in subset:
            item[key] = state[item["name"]]
        return result

    screened = screen(rows, "guard_state")
    estimates = [row["estimate"] for row in rows if row.get("estimate")]
    screened_estimate = screen(estimates, "guard_state") if estimates else None
    for row in rows:
        row["estimate_guard_state"] = (row["estimate"]["guard_state"]
                                       if row.get("estimate") else None)

    by_series: dict[str, dict] = {}
    for series in sorted({row["series"] for row in rows}):
        subset = [row for row in rows if row["series"] == series]
        by_series[series] = {
            "count": len(subset),
            "music_minus_effects_db": summary([row["music_minus_effects_db"] for row in subset]),
            "music_minus_speech_db": summary([row["music_minus_speech_db"] for row in subset]),
            "guard": {state: sum(1 for row in subset if row["guard_state"] == state)
                      for state in sorted({row["guard_state"] for row in subset})},
        }

    payload = {
        "probe": "bgm-candidate-content",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": "保存した未発売候補それぞれの、speech/music/effectsの取り分。"
                 "却下済み分離器をanalyzerとして使っている。意味役割は判定しない",
        "analyzer_role": "rejected_renderer_kept_as_analyzer",
        "content_guard": {series_cues.GUARD_TARGET_OCCURRENCE: screened,
                          series_cues.GUARD_TARGET_ESTIMATE: screened_estimate},
        "candidate_count": len(rows),
        "by_series": by_series,
        "overall": {
            "music_minus_effects_db": summary([row["music_minus_effects_db"] for row in rows]),
            "music_minus_speech_db": summary([row["music_minus_speech_db"] for row in rows]),
            "guard": {state: sum(1 for row in rows if row["guard_state"] == state)
                      for state in sorted({row["guard_state"] for row in rows})},
        },
        "estimate_count": len(estimates),
        # 製品入口へそのまま渡せる形。録音ごとに対象を2つとも並べる。
        "guard_rows": [
            {k: v for k, v in item.items()
             if k in ("recording_key", "target", "speech_db", "music_db", "effects_db",
                      "series", "name")}
            for row in rows for item in ([row] + ([row["estimate"]] if row.get("estimate") else []))],
        "guard_by_target": {
            series_cues.GUARD_TARGET_OCCURRENCE: {
                state: sum(1 for row in rows if row["guard_state"] == state)
                for state in sorted({row["guard_state"] for row in rows})},
            series_cues.GUARD_TARGET_ESTIMATE: (
                {state: sum(1 for row in rows if row["estimate_guard_state"] == state)
                 for state in sorted({row["estimate_guard_state"] for row in rows})}
                if estimates else None)},
        "estimate_change": {
            key: summary([row["estimate"][key] for row in rows if row.get("estimate")])
            for key in ("effects_change_db", "speech_change_db")},
        "rows": sorted(rows, key=lambda row: (row["series"], row["name"])),
        "limits": [
            "取り分は分離器の出力であり、正解音源ではない。分離器自体は最終rendererとして却下済み",
            "関門を通っても、欲しい劇伴かどうかは判定していない",
            "既製の効果音・繰り返しの台詞は、この測定だけでは劇伴と区別しきれない",
            "試聴していない。公開・昇格・既定変更・Holdout消費はしていない",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "analysis.json"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"candidates": len(rows), "kept": screened["kept_count"],
                      "rejected": screened["rejected_count"],
                      "overall": payload["overall"],
                      "estimate_change": payload["estimate_change"],
                      "guard_by_target": payload["guard_by_target"]}, ensure_ascii=False))
    print(f"analysis: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
