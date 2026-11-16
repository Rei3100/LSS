"""1話だけから、未発売の劇伴の候補を曲単位で取り出す。

`docs/BGM.md` の「最小のfirst-class BGM slice」を1話単体で通す経路。繰り返しは要らない
（一度しか鳴らない曲もfirst-class）。段は次の順に進む。

1. **外す**：参照録音（OST・OP/EDのTVサイズ等）が手元にあれば照合し、既発売の録音と
   主題歌の区間を外す。取り出し直す必要が無いため。参照が無い場合は外せないと記録する。
2. **見つける**：BGMの3分類（台詞・音楽・効果音）で学習した分離器の出力を解析に使い、
   1秒窓ごとの音楽の取り分を数秒の中央値でならしてから、ヒステリシスで区間にする。
   分離器が台詞＋効果音へ漏らす音楽は散発的で、劇伴は連続しているので、ならすと分かれる。
   取り分は「音楽が優勢か」なので、台詞・効果音の下で小さく鳴る劇伴は取り分が低く落ちる。
   そこで、音楽の出力が下限以上の大きさで途切れずに続く区間も、別の根拠として区間にする
   （`sustained_seconds`）。分離器が台詞のある場面で効果音を音楽へ誤配分する漏れは、効果音の
   鳴る窓にだけ出て間の窓で下限を大きく割るので、途切れない存在にはならない。
   曲の長さに満たない断片は候補にしない。ただし、それ自体で完結した短い音楽（アイキャッチ・
   ジングル等）は長さだけで捨てない（`short_seconds` / `short_dominant_share`）。短い区間と、
   長い区間から離れた短い途切れない存在をつなぎ、区間の丸ごとが音楽で埋まっていれば候補にする。
   長い区間を切った・既知の録音を外した残りの断片は、これまでどおり候補にしない。
3. **分ける**：音楽が途切れずにつながる曲どうしを、曲が替わる位置の候補で分ける
   （`cue_boundaries`：大きさの段差・音色・ステレオの広がりの変化を合わせる）。
4. **関門**：区間の中で音楽が続いている窓の割合（取り分が続く窓、または途切れない存在の窓）。
   台詞だけ・効果音だけの区間はここで落ちる。
   台詞より大きいことは要求しない（台詞の下で鳴る劇伴は正常。`docs/BGM.md`）。
   欲しい劇伴かどうか（意味role）は判定しない。
5. **仕上げる**：音楽の出力に効果音が強く混ざっていそうな候補だけ、その候補を含むブロックを
   解析器へもう一度通し、明確に良くなった時だけ2回目を採る（`lss.bgm.second_pass`）。
6. **出す**：通った区間を、分離器の音楽の出力と原本の区間（raw/no-op）の二つで切り出し、
   出典・時刻・測定値・Recipeを付けて **非公開の保留候補** として保存する
   （`candidate_store`：単独のファイルにも出典が残り、同じ候補の再実行は検証して再利用する）。

分離器の推論はここでは行わない。解析用の3 stemを受け取る（製品は `lss.bgm.sfc_analysis`
を常駐workerで動かす）。既定の伴奏modelは伴奏抽出Capability用であり、この経路では使わない。

区間の範囲は音楽の活動から決めたもので、曲の切れ目の確定ではない（Cue formationは未解決）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from lss.bgm import cue_boundaries


SCHEMA_VERSION = 1
METHOD = "episode-single-occurrence-cues-v1"
RECORD_NAME = "episode-cues.json"
PUBLICATION_STATE = "not_published"
STEMS = ("music", "speech", "effects")
RENDERERS = ("separator_music", "raw")

DISPOSITION_HELD = "held"
DISPOSITION_REJECTED = "rejected_music_not_sustained"
DISPOSITION_EXCLUDED = "excluded_known_recording"
SERIALIZATION_POLICY = "declared_power_of_two_headroom_v1"


@dataclass(frozen=True)
class CueOrigin:
    """候補ファイルへ残す出典。原本ファイルと、区間を見つけた解析の同一性。"""

    source_path: Path
    source_sha256: str
    source_id: str
    analysis_id: str
    runtime_id: str
    time_offset_numerator: int = 0
    time_offset_denominator: int = 1


class EpisodeCueError(ValueError):
    """1話の候補づくりの宣言、または渡された材料が契約に合わない。"""


@dataclass(frozen=True)
class EpisodeCueRecipe:
    """1話の候補づくりの宣言。どれかが変われば識別子が変わる。"""

    window_seconds: float
    smoothing_windows: int
    onset_share: float
    offset_share: float
    floor_music_db: float
    bridge_seconds: float
    minimum_cue_seconds: float
    minimum_active_fraction: float
    exclusion_pad_seconds: float
    split: cue_boundaries.CueBoundaryRecipe | None = None
    # 音楽の出力が `floor_music_db` 以上で途切れずに続く長さ。これ以上続けば取り分が低くても
    # 区間にする。Noneなら使わない（取り分だけで決める旧来の宣言）。
    sustained_seconds: float | None = None
    # 完結した短い音楽の宣言。`minimum_cue_seconds` 未満でも、他の区間から離れて完結し、
    # この長さ以上で窓の `short_present_fraction` 以上が下限以上なら候補にする。Noneなら使わない。
    short_seconds: float | None = None
    short_present_fraction: float = 0.8
    # さらに短い音楽は、全ての窓で取り分がこれ以上（音楽がほぼ単独）なら `short_dominant_seconds`
    # 以上で候補にする。台詞のある場面の効果音の漏れは取り分が高くても1窓で途切れる。
    short_dominant_share: float | None = None
    short_dominant_seconds: float = 2.0
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD
    partial_window_policy: str = "include_final_partial_v1"

    def validate(self) -> None:
        if self.partial_window_policy != "include_final_partial_v1":
            raise EpisodeCueError("Unsupported partial-window policy")
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise EpisodeCueError("未対応のschemaです")
        if (isinstance(self.smoothing_windows, bool) or not isinstance(self.smoothing_windows, int)
                or self.smoothing_windows < 1 or self.smoothing_windows % 2 == 0):
            raise EpisodeCueError("smoothing_windowsは1以上の奇数である必要があります")
        for name, low, high in (("window_seconds", 0.05, 10.0), ("onset_share", 0.0, 1.0),
                                ("offset_share", 0.0, 1.0), ("floor_music_db", -150.0, 0.0),
                                ("bridge_seconds", 0.0, 60.0), ("minimum_cue_seconds", 0.0, 600.0),
                                ("minimum_active_fraction", 0.0, 1.0),
                                ("exclusion_pad_seconds", 0.0, 60.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not low <= float(value) <= high:
                raise EpisodeCueError(f"{name}は{low}以上{high}以下である必要があります")
        if self.offset_share > self.onset_share:
            raise EpisodeCueError("offset_shareはonset_share以下である必要があります")
        if self.split is not None:
            self.split.validate()
        if self.sustained_seconds is not None and (
                isinstance(self.sustained_seconds, bool) or not isinstance(self.sustained_seconds, Real)
                or not self.window_seconds <= float(self.sustained_seconds) <= 600.0):
            raise EpisodeCueError("sustained_secondsは窓の長さ以上600以下である必要があります")
        for name in ("short_seconds", "short_dominant_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                      or not self.window_seconds <= float(value) <= self.minimum_cue_seconds):
                raise EpisodeCueError(f"{name}は窓の長さ以上minimum_cue_seconds以下である必要があります")
        for name in ("short_present_fraction", "short_dominant_share"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                      or not 0.0 < float(value) <= 1.0):
                raise EpisodeCueError(f"{name}は0より大きく1以下である必要があります")
        if self.short_dominant_share is not None and self.short_seconds is None:
            raise EpisodeCueError("short_dominant_shareはshort_secondsと一緒に宣言します")

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        # 使わない宣言は識別子を変えない（旧来の宣言と同じ識別子のまま比べられる）。
        if body["sustained_seconds"] is None:
            body.pop("sustained_seconds")
        if body["short_seconds"] is None:
            for name in ("short_seconds", "short_present_fraction", "short_dominant_share",
                         "short_dominant_seconds"):
                body.pop(name)
        elif body["short_dominant_share"] is None:
            body.pop("short_dominant_share")
            body.pop("short_dominant_seconds")
        return body

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-episode-cues-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _subtract(interval: tuple[float, float],
              excluded: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """区間から除外区間を引いた残り。"""
    pieces = [interval]
    for low, high in sorted(excluded):
        next_pieces = []
        for start, end in pieces:
            if high <= start or low >= end:
                next_pieces.append((start, end))
                continue
            if low > start:
                next_pieces.append((start, low))
            if high < end:
                next_pieces.append((high, end))
        pieces = next_pieces
    return pieces


def _level_db(energy: float) -> float:
    return round(10.0 * np.log10(max(energy, 1e-20)), 3)


def measure_span(stem_paths: Mapping[str, Path], start_seconds: float, end_seconds: float) -> dict:
    """区間の中の3 stemのエネルギー（dB）。関門と記録に使う。"""
    import soundfile as sf

    levels = {}
    for stem in STEMS:
        with sf.SoundFile(str(stem_paths[stem])) as stream:
            rate = stream.samplerate
            stream.seek(int(round(start_seconds * rate)))
            data = stream.read(int(round((end_seconds - start_seconds) * rate)),
                               dtype="float64", always_2d=True)
        levels[stem] = _level_db(float(np.mean(np.square(data))) if data.size else 0.0)
    return {"music_db": levels["music"], "speech_db": levels["speech"], "effects_db": levels["effects"],
            "music_over_speech_db": round(levels["music"] - levels["speech"], 3),
            "music_over_effects_db": round(levels["music"] - levels["effects"], 3)}


def window_shares(stem_paths: Mapping[str, Path], window_seconds: float) -> tuple[np.ndarray, np.ndarray, int]:
    """窓ごとの音楽の取り分と、音楽の大きさ（dB）。窓は重ならない。"""
    import soundfile as sf

    streams = [sf.SoundFile(str(stem_paths[stem])) for stem in STEMS]
    try:
        rate = streams[0].samplerate
        frames = min(stream.frames for stream in streams)
        size = max(1, int(round(window_seconds * rate)))
        shares, levels = [], []
        for first in range(0, frames, size):
            count = min(size, frames - first)
            energies = [float(np.mean(np.square(stream.read(count, dtype="float32", always_2d=True))))
                        for stream in streams]
            total = sum(energies)
            shares.append(energies[0] / total if total > 0 else 0.0)
            levels.append(10.0 * np.log10(max(energies[0], 1e-20)))
    finally:
        for stream in streams:
            stream.close()
    return np.asarray(shares), np.asarray(levels), rate


def _smoothed(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values.copy()
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[index:index + width]) for index in range(values.size)])


def sustained_windows(levels: np.ndarray, recipe: EpisodeCueRecipe) -> np.ndarray:
    """音楽の出力が下限以上で、宣言した長さ以上途切れずに続く窓の印。"""
    marked = np.zeros(levels.size, dtype=bool)
    if recipe.sustained_seconds is None:
        return marked
    need = int(np.ceil(float(recipe.sustained_seconds) / float(recipe.window_seconds) - 1e-9))
    present = levels >= recipe.floor_music_db
    start = None
    for index, value in enumerate([*present, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= need:
                marked[start:index] = True
            start = None
    return marked


def _bridged(intervals: Iterable[tuple[int, int]], recipe: EpisodeCueRecipe) -> list[tuple[int, int]]:
    bridge = int(round(recipe.bridge_seconds / recipe.window_seconds))
    merged: list[list[int]] = []
    for low, high in sorted(intervals):
        if merged and low - merged[-1][1] <= bridge:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return [(low, high) for low, high in merged]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def share_intervals(shares: np.ndarray, levels: np.ndarray,
                    recipe: EpisodeCueRecipe) -> list[tuple[int, int]]:
    """ならした取り分のヒステリシスで、音楽が優勢な窓番号の区間 [開始, 終了) を返す。"""
    smooth = _smoothed(np.where(levels >= recipe.floor_music_db, shares, 0.0), recipe.smoothing_windows)
    intervals, start = [], None
    for index, value in enumerate(smooth):
        if start is None and value >= recipe.onset_share:
            start = index
        elif start is not None and value < recipe.offset_share:
            intervals.append((start, index))
            start = None
    if start is not None:
        intervals.append((start, int(smooth.size)))
    return _bridged(intervals, recipe)


def active_intervals(shares: np.ndarray, levels: np.ndarray,
                     recipe: EpisodeCueRecipe) -> list[tuple[int, int]]:
    """取り分の区間と途切れない存在の区間を合わせ、窓番号の区間 [開始, 終了) を返す。"""
    return _bridged([*share_intervals(shares, levels, recipe),
                     *_runs(sustained_windows(levels, recipe))], recipe)


def joined_gap_splits(shares: np.ndarray, levels: np.ndarray, recipe: EpisodeCueRecipe) -> list[int]:
    """途切れない存在がつないだ、取り分の区間どうしの隙間の切れ目（窓番号）。

    取り分の区間の隙間は、存在をつなぎにする前は曲の切れ目として働いていた。つないだ区間を
    1つの候補にすると別の曲まで一続きになる（実番組11話で既知劇伴の端の捕捉が22→11件）ので、
    隙間の中で音楽が最も小さい窓で切る。隙間の小さい音楽は前後どちらかの候補に残る。
    """
    shares_only = share_intervals(shares, levels, recipe)
    splits = []
    for low, high in active_intervals(shares, levels, recipe):
        inside = [item for item in shares_only if item[0] >= low and item[1] <= high]
        for (_, first_end), (second_start, _) in zip(inside, inside[1:]):
            splits.append(first_end + int(np.argmin(levels[first_end:second_start])))
    return splits


def _windows(seconds: float, recipe: EpisodeCueRecipe) -> int:
    return int(np.ceil(float(seconds) / float(recipe.window_seconds) - 1e-9))


def short_intervals(shares: np.ndarray, levels: np.ndarray, intervals: list[tuple[int, int]],
                    recipe: EpisodeCueRecipe) -> tuple[list[tuple[int, int]], np.ndarray]:
    """完結した短い音楽の候補区間と、その根拠の窓の印を返す。

    `minimum_cue_seconds` 未満の区間と、長い区間から `bridge_seconds` より離れた短い途切れない
    存在（`short_seconds` 以上、または全窓で取り分が `short_dominant_share` 以上の
    `short_dominant_seconds` 以上）をつなぐ。長い区間へは加えないので、曲の区切りは変わらない。
    """
    marked = np.zeros(levels.size, dtype=bool)
    if recipe.short_seconds is None:
        return [], marked
    minimum = _windows(recipe.minimum_cue_seconds, recipe)
    bridge = int(round(recipe.bridge_seconds / recipe.window_seconds))
    long = [item for item in intervals if item[1] - item[0] >= minimum]
    short = [item for item in intervals if item[1] - item[0] < minimum]

    def isolated(low: int, high: int) -> bool:
        return all(start - high > bridge or low - end > bridge for start, end in long)

    present = levels >= recipe.floor_music_db
    runs = [item for item in _runs(present)
            if item[1] - item[0] >= _windows(recipe.short_seconds, recipe) and isolated(*item)]
    if recipe.short_dominant_share is not None:
        runs += [item for item in _runs(present & (shares >= recipe.short_dominant_share))
                 if item[1] - item[0] >= _windows(recipe.short_dominant_seconds, recipe) and isolated(*item)]
    for low, high in runs:
        marked[low:high] = True
    return _bridged([*short, *runs], recipe), marked


def _observed_fraction(mask, first, start, end, window) -> float:
    times = (first + np.arange(len(mask))) * window
    weights = np.maximum(0., np.minimum(times + window, end) - np.maximum(times, start))
    return float(np.sum(np.asarray(mask) * weights) / weights.sum()) if weights.sum() else 0.


def _complete_short(first: int, last: int, shares: np.ndarray, levels: np.ndarray,
                    recipe: EpisodeCueRecipe, *, start_seconds=None, end_seconds=None) -> bool:
    """区間の丸ごとが、短くても音楽として十分な根拠を持つか。"""
    present = levels[first:last] >= recipe.floor_music_db
    if recipe.short_seconds is None or not present.size:
        return False
    start = first * recipe.window_seconds if start_seconds is None else start_seconds
    end = last * recipe.window_seconds if end_seconds is None else end_seconds
    if (end - start >= recipe.short_seconds
            and _observed_fraction(present, first, start, end, recipe.window_seconds) >= recipe.short_present_fraction):
        return True
    return (recipe.short_dominant_share is not None
            and end - start >= recipe.short_dominant_seconds
            and bool(np.all(present & (shares[first:last] >= recipe.short_dominant_share))))


def plan_episode_cues(stem_paths: Mapping[str, Path], recipe: EpisodeCueRecipe, *,
                      known_spans: Iterable[tuple[float, float]] = ()) -> dict:
    """音楽が続く区間から、既知の録音を外し、関門をかけた候補の一覧を作る。音声は作らない。"""
    recipe.validate()
    import soundfile as sf

    shares, levels, rate = window_shares(stem_paths, recipe.window_seconds)
    infos = [sf.info(stem_paths[stem]) for stem in STEMS]
    source_seconds = min(info.frames / info.samplerate for info in infos)
    window = float(recipe.window_seconds)
    pad = float(recipe.exclusion_pad_seconds)
    excluded = [(max(0.0, low - pad), high + pad) for low, high in known_spans]
    intervals = active_intervals(shares, levels, recipe)
    sustained = sustained_windows(levels, recipe)
    gap_splits = [index * window for index in joined_gap_splits(shares, levels, recipe)]
    short, short_evidence = short_intervals(shares, levels, intervals, recipe)
    if recipe.short_seconds is not None:
        minimum = _windows(recipe.minimum_cue_seconds, recipe)
        long = [item for item in intervals if item[1] - item[0] >= minimum]
        intervals = sorted(long + short)
    else:
        long = intervals
    cues = []
    excluded_seconds = 0.0
    for low, high in intervals:
        interval = (low * window, min(high * window, source_seconds))
        pieces = _subtract(interval, excluded)
        excluded_seconds += (interval[1] - interval[0]) - sum(end - start for start, end in pieces)
        spans = []
        own_splits = gap_splits if (low, high) in long else []
        for outer_start, outer_end in pieces:
            inner = [outer_start, *[at for at in own_splits if outer_start < at < outer_end], outer_end]
            for piece_start, piece_end in zip(inner, inner[1:]):
                boundaries = ([] if recipe.split is None else
                              [item["seconds"] for item in cue_boundaries.split_positions(
                                  stem_paths["music"], piece_start, piece_end, recipe.split)])
                points = [piece_start, *boundaries, piece_end]
                spans.extend(zip(points, points[1:]))
        for start, end in spans:
            first = int(start / window)
            last = max(first + 1, int(math.ceil(end / window)))
            inside = shares[first:last]
            is_short = end - start < float(recipe.minimum_cue_seconds)
            # 短い候補は、区間を丸ごと（切った残りや既知の録音を外した残りではなく）認める。
            if is_short and not ((start, end) == interval
                                 and _complete_short(first, last, shares, levels, recipe,
                                                     start_seconds=start, end_seconds=end)):
                continue
            held_on = (inside >= recipe.offset_share) | sustained[first:last] | short_evidence[first:last]
            if is_short:
                held_on = held_on | (levels[first:last] >= recipe.floor_music_db)
            active = _observed_fraction(held_on, first, start, end, window)
            measured = measure_span(stem_paths, start, end)
            cues.append({
                "cue_id": f"cue-{len(cues) + 1:03d}",
                "start_seconds": round(start, 3), "end_seconds": end if end == source_seconds else round(end, 3),
                "seconds": round(end - start, 3), "active_fraction": round(active, 4),
                "median_share": round(float(np.median(inside)) if inside.size else 0.0, 4),
                # 関門を通した根拠の内訳。取り分が続く窓と、途切れない存在の窓の割合。
                "share_fraction": round(_observed_fraction(inside >= recipe.offset_share, first, start, end, window), 4)
                if inside.size else 0.0,
                "sustained_fraction": round(_observed_fraction(sustained[first:last], first, start, end, window), 4)
                if inside.size else 0.0,
                # 最小候補長に満たず、完結した短い音楽として残した候補か。
                "short_cue": bool(is_short),
                **measured,
                "disposition": (DISPOSITION_HELD if active >= recipe.minimum_active_fraction
                                else DISPOSITION_REJECTED),
            })
    return {"recipe_id": recipe.recipe_id, "sample_rate": rate,
            "source_seconds": source_seconds,
            "activity_interval_count": len(intervals),
            "known_span_count": len(excluded), "excluded_known_seconds": round(excluded_seconds, 3),
            "cues": cues}


def _sha256(path: Path) -> str:
    from lss.bgm.candidate_store import signal_sha256

    return signal_sha256(path)


def _materialize_cut(signal: Path, signal_sha256: str, start: float, end: float, target: Path, *,
                     origin: CueOrigin, recipe_id: str, evidence: dict) -> dict:
    """区間を含む最小のsample範囲を、出典付きの非公開候補として保存する。

    振幅は変えない（大きすぎる場合は2の冪で下げて記録する）。同じ候補が既にあれば検証して再利用する。
    """
    import soundfile as sf

    from lss.bgm.candidate_store import CandidateOrigin, materialize_candidate

    info = sf.info(str(signal))
    first = max(0, math.floor(start * info.samplerate))
    last = min(info.frames, math.ceil(end * info.samplerate))
    if last <= first:
        raise EpisodeCueError(f"切り出す区間がありません: {signal}")
    record = materialize_candidate(
        signal, target,
        origin=CandidateOrigin(
            origin.source_sha256, signal_sha256, origin.analysis_id, origin.runtime_id, recipe_id,
            first, last, info.samplerate, origin.time_offset_numerator, origin.time_offset_denominator,
            serialization_policy=SERIALIZATION_POLICY,
        ),
        source_id=origin.source_id, source_path=origin.source_path, evidence=evidence,
    )
    return {**record, "publication_state": PUBLICATION_STATE}


def known_recording_spans(source: Path, reference_paths: Iterable[Path], *,
                          cache_dir: Path | None = None) -> list[dict]:
    """参照録音と一致する区間（既発売・主題歌）。参照が無ければ空。"""
    from lss.bgm import identity

    references = list(reference_paths)
    if not references:
        return []
    index = identity.ReferenceIndex.build(references, cache_dir=cache_dir)
    spans = []
    for match in identity.match(identity.decode_mono(Path(source)), index):
        if match.identity_state == "known_exact":
            spans.append({"start_seconds": round(match.query_start_seconds, 3),
                          "end_seconds": round(match.query_end_seconds, 3),
                          "reference": Path(match.reference_path).name})
    return spans


def _refined(plan: dict, cue_ids) -> list[dict]:
    """磨き直す保留候補。``cue_ids`` がNoneなら全て（従来どおり）。指定外の候補は1回目の音楽の出力のまま残す。"""
    held = [cue for cue in plan["cues"] if cue["disposition"] == DISPOSITION_HELD]
    if cue_ids is None:
        return held
    wanted = set(cue_ids)
    for cue in held:
        if cue["cue_id"] not in wanted:
            cue["refinement"] = "not_routed"
    return [cue for cue in held if cue["cue_id"] in wanted]


def _second_pass(plan: dict, paths: Mapping[str, Path], recipe, refine, cue_ids=None) -> dict:
    """試す価値のある保留候補だけを2回目へ送り、候補ごとに採否を決める。記録の本文を返す。"""
    import soundfile as sf

    from lss.bgm import second_pass, sfc_analysis

    held = _refined(plan, cue_ids)
    for cue in held:
        cue["second_pass"] = {"eligibility": second_pass.eligibility(
            paths, cue["start_seconds"], cue["end_seconds"], recipe)}
    eligible = [cue for cue in held if cue["second_pass"]["eligibility"]["eligible"]]
    summary = {"recipe": recipe.to_dict(), "recipe_id": recipe.recipe_id,
               "eligible_count": len(eligible), "adopted_count": 0}
    if not eligible or refine is None:
        summary["status"] = "not_needed" if not eligible else "not_available"
        for cue in held:
            cue["second_pass"]["adopted"] = False
        return summary
    info = sf.info(str(paths["music"]))
    blocks = sfc_analysis.blocks_for([(cue["start_seconds"], cue["end_seconds"]) for cue in eligible],
                                     info.frames, info.samplerate)
    result = refine(blocks)
    second_music = Path(result["music"]).resolve()
    second_info = sf.info(str(second_music))
    if (second_info.samplerate, second_info.frames, second_info.channels) != (
            info.samplerate, info.frames, info.channels):
        raise EpisodeCueError("2回目の音楽の出力が1回目と同じ時間軸ではありません")
    summary.update({"status": "ran", "blocks": blocks, "analysis_id": result["analysis_id"],
                    "runtime_id": result["runtime_id"],
                    "signal": {"path": str(second_music), "sha256": _sha256(second_music)}})
    for cue in held:
        decision = cue["second_pass"]
        if not decision["eligibility"]["eligible"]:
            decision["adopted"] = False
            continue
        decision["adoption"] = second_pass.adoption(paths["music"], second_music, cue["start_seconds"],
                                                    cue["end_seconds"], recipe)
        decision["adopted"] = bool(decision["adoption"]["adopt"])
        summary["adopted_count"] += decision["adopted"]
    return summary


def _cue_frames(cue: dict, rate: int, frames: int) -> tuple[int, int]:
    """候補を切り出す sample 範囲（`_materialize_cut` と同じ丸め）。"""
    return (max(0, math.floor(cue["start_seconds"] * rate)),
            min(frames, math.ceil(cue["end_seconds"] * rate)))


def _read_frames(path: Path, first: int, last: int) -> np.ndarray:
    import soundfile as sf

    with sf.SoundFile(str(path)) as stream:
        stream.seek(first)
        return stream.read(last - first, dtype="float64", always_2d=True)


def _vocal_recovery(plan: dict, paths: Mapping[str, Path], second: dict | None, recipe, recover,
                    target: Path, mixture: Path, cue_ids=None) -> dict:
    """効果音の出力へ誤配分された歌唱を、試す価値のある保留候補だけ4分類で回収し、候補ごとに採否を決める。

    ``recover(input_role, signal_path, spans)`` は信号の区間 [開始frame, 終了frame) を4分類へ通し、
    ``{"stems": [{"span", "path", "sha256"}], "analysis_id", "runtime_id"}``（``music_vocals``）を返す。
    4分類へ送るのは、1回目の効果音の出力（試す価値のある候補）と、その中に足す価値のある歌唱が
    あった候補の採用中の音楽・原本（``mixture``、解析stemと同じ時間軸）だけ。
    採った候補の challenger は、全長の ``target``（採らない所は0）へ書く。記録の本文を返す。
    """
    import soundfile as sf

    from lss.bgm import vocal_recovery

    info = sf.info(str(paths["music"]))
    rate, frames = info.samplerate, info.frames
    second_music = Path(second["signal"]["path"]) if second and second.get("status") == "ran" else None
    held = _refined(plan, cue_ids)

    def current_of(cue: dict) -> Path:
        return second_music if cue.get("second_pass", {}).get("adopted") else paths["music"]

    for cue in held:
        first, last = _cue_frames(cue, rate, frames)
        cue["vocal_recovery"] = {"frames": [first, last], "eligibility": vocal_recovery.eligibility(
            _read_frames(current_of(cue), first, last), _read_frames(paths["effects"], first, last), recipe)}
    eligible = [cue for cue in held if cue["vocal_recovery"]["eligibility"]["eligible"]]
    summary = {"recipe": recipe.to_dict(), "recipe_id": recipe.recipe_id,
               "eligible_count": len(eligible), "adopted_count": 0}
    if not eligible or recover is None:
        summary["status"] = "not_needed" if not eligible else "not_available"
        for cue in held:
            cue["vocal_recovery"]["adopted"] = False
        return summary

    def run(role: str, signal: Path, cues: list[dict]) -> tuple[dict, dict]:
        result = recover(role, signal, [cue["vocal_recovery"]["frames"] for cue in cues])
        by_span = {tuple(item["span"]): Path(item["path"]) for item in result["stems"]}
        if set(by_span) != {tuple(cue["vocal_recovery"]["frames"]) for cue in cues}:
            raise EpisodeCueError("4分類の出力が求めた区間と一致しません")
        return by_span, result

    effects_vocals, effects_run = run("first_pass_effects", paths["effects"], eligible)
    runs = {"first_pass_effects": {key: effects_run[key] for key in ("analysis_id", "runtime_id")}}
    loaded = {}
    worth = []
    for cue in eligible:
        first, last = cue["vocal_recovery"]["frames"]
        ve = sf.read(str(effects_vocals[(first, last)]), dtype="float64", always_2d=True)[0]
        current = _read_frames(current_of(cue), first, last)
        if ve.shape != current.shape:
            raise EpisodeCueError("4分類の出力が候補区間の長さと一致しません")
        loaded[cue["cue_id"]] = (current, ve)
        # 効果音の出力の中の歌唱が、gate前でも意味のある量に届かなければ、候補の歌唱を測るまでもない。
        if (10 * np.log10(max(float(np.sum(ve ** 2)), 1e-30))
                - 10 * np.log10(max(float(np.sum(current ** 2)), 1e-30))) >= recipe.minimum_added_over_music_db:
            worth.append(cue)
        else:
            cue["vocal_recovery"].update(adopted=False, adoption={
                "adopt": False, "features": {}, "reasons": ["nothing_meaningful_to_add"]})
    music_vocals = {}
    mixture_vocals = {}
    if worth:
        found, result = run("mixture", mixture, worth)
        runs["mixture"] = {key: result[key] for key in ("analysis_id", "runtime_id")}
        for cue in worth:
            mixture_vocals[cue["cue_id"]] = found[tuple(cue["vocal_recovery"]["frames"])]
    for source in {current_of(cue) for cue in worth}:
        group = [cue for cue in worth if current_of(cue) == source]
        role = "second_pass_music" if source == second_music else "first_pass_music"
        found, result = run(role, source, group)
        runs[role] = {key: result[key] for key in ("analysis_id", "runtime_id")}
        for cue in group:
            music_vocals[cue["cue_id"]] = found[tuple(cue["vocal_recovery"]["frames"])]
    adopted = {}
    for cue in worth:
        current, ve = loaded[cue["cue_id"]]
        vm = sf.read(str(music_vocals[cue["cue_id"]]), dtype="float64", always_2d=True)[0]
        vx = sf.read(str(mixture_vocals[cue["cue_id"]]), dtype="float64", always_2d=True)[0]
        challenger, decision = vocal_recovery.recover(current, ve, vm, vx, rate, recipe)
        cue["vocal_recovery"].update(adopted=bool(decision["adopt"]), adoption=decision)
        if decision["adopt"]:
            adopted[cue["cue_id"]] = challenger
    summary.update({"status": "ran", "worth_count": len(worth), "adopted_count": len(adopted), "runs": runs})
    if adopted:
        # 採った候補の区間だけを書いた全長の信号。候補はここから同じ時刻で切り出す。
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        with sf.SoundFile(str(temporary), "w", samplerate=rate, channels=info.channels,
                          subtype="FLOAT", format="WAV") as stream:
            block = np.zeros((1 << 18, info.channels), dtype=np.float32)
            written = 0
            while written < frames:
                count = min(block.shape[0], frames - written)
                stream.write(block[:count])
                written += count
        with sf.SoundFile(str(temporary), "r+") as stream:
            for cue in held:
                if cue["cue_id"] in adopted:
                    stream.seek(cue["vocal_recovery"]["frames"][0])
                    stream.write(adopted[cue["cue_id"]].astype(np.float32))
        temporary.replace(target)
        summary["signal"] = {"path": str(target), "sha256": _sha256(target)}
    return summary


def run_episode_cues(source: Path, stem_paths: Mapping[str, Path], out_dir: Path,
                     recipe: EpisodeCueRecipe, *, origin: CueOrigin,
                     reference_paths: Iterable[Path] = (),
                     cache_dir: Path | None = None, raw_source: Path | None = None,
                     provenance: Mapping[str, Any] | None = None,
                     second_pass_recipe=None, refine=None,
                     vocal_recovery_recipe=None, recover=None, work_dir: Path | None = None,
                     plan: dict | None = None, refine_cue_ids=None) -> Path:
    """1話から保留候補を作り、音声と記録を保存する。記録のパスを返す。

    ``source`` は解析stemと同じ時間軸の音声。``raw_source`` を渡すと、原本の区間
    （raw/no-op候補）はそちらから同じ時刻で切る（解析用に周波数を変えた複製ではなく、
    元の周波数の復号から切るため）。``origin`` は候補ファイルへ残す原本と解析の同一性。

    ``second_pass_recipe`` を渡すと、試す価値のある候補を ``refine(blocks)`` で解析器へもう一度
    通す。``refine`` は1回目の音楽の出力を同じブロック格子で解析し、``{"music", "analysis_id",
    "runtime_id"}`` を返す。採った候補は2回目の出力から、1回目の出力も別に残す。

    ``vocal_recovery_recipe`` を渡すと、効果音の出力へ誤配分された歌唱を ``recover`` で4分類へ通し、
    採用中の音楽へ安全に足せる候補だけ足す（`lss.bgm.vocal_recovery`）。採った候補は足した信号から、
    足す前の候補も別に残す。足した全長の信号は ``work_dir``（省略時は ``out_dir``）へ置く。

    ``plan`` を渡すと区間計画を作り直さない（同じ宣言・同じstemで作った ``plan_episode_cues`` の結果）。
    ``refine_cue_ids`` を渡すと、2回目と歌唱回収はその保留候補だけに試す（公開に使う区間だけを磨き直す）。
    ブロック・区間ごとの推論は他の区間の有無で変わらないので、試した候補の採否と音は全てに試した時と同じ。
    """
    import soundfile as sf

    source = Path(source).resolve()
    paths = {stem: Path(stem_paths[stem]).resolve() for stem in STEMS}
    original = sf.info(str(source))
    for stem, path in paths.items():
        info = sf.info(str(path))
        # 解析stemが原本と同じ時間軸でなければ、区間の時刻が原本を指さない。
        if info.samplerate != original.samplerate or abs(info.frames - original.frames) > original.samplerate // 100:
            raise EpisodeCueError(f"解析stemの時間軸が原本と一致しません: {stem}")
    source_sha = _sha256(source)
    references = [Path(path).resolve() for path in reference_paths]
    known = known_recording_spans(source, references, cache_dir=cache_dir)
    if plan is None:
        plan = plan_episode_cues(paths, recipe,
                                 known_spans=[(item["start_seconds"], item["end_seconds"]) for item in known])
    elif plan.get("recipe_id") != recipe.recipe_id:
        raise EpisodeCueError("渡された区間計画の宣言が一致しません")
    second = (None if second_pass_recipe is None
              else _second_pass(plan, paths, second_pass_recipe, refine, refine_cue_ids))
    second_music = Path(second["signal"]["path"]) if second and second.get("status") == "ran" else None

    out = Path(out_dir).resolve()
    vocal = (None if vocal_recovery_recipe is None
             else _vocal_recovery(plan, paths, second, vocal_recovery_recipe, recover,
                                  Path(work_dir or out).resolve() / "music-vocal-recovered.wav", source,
                                  refine_cue_ids))
    vocal_music = Path(vocal["signal"]["path"]) if vocal and vocal.get("signal") else None
    audio_dir = out / "candidates"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw = Path(raw_source).resolve() if raw_source is not None else source
    # 候補ごとに全長の信号を読み直さない。保存時の照合はcandidate_storeが行う。
    music_sha = _sha256(paths["music"])
    raw_sha = _sha256(raw)
    second_sha = second["signal"]["sha256"] if second_music is not None else None
    for cue in plan["cues"]:
        if cue["disposition"] != DISPOSITION_HELD:
            continue
        stem = f"{cue['cue_id']}-{cue['start_seconds']:.0f}s"
        measured = {key: cue[key] for key in ("cue_id", "active_fraction", "median_share",
                                              "music_over_speech_db", "music_over_effects_db")}
        adopted = bool(cue.get("second_pass", {}).get("adopted"))
        recovered = bool(cue.get("vocal_recovery", {}).get("adopted"))
        first_cut = lambda target: _materialize_cut(  # noqa: E731
            paths["music"], music_sha, cue["start_seconds"], cue["end_seconds"], target,
            origin=origin, recipe_id=plan["recipe_id"],
            evidence={"method": METHOD, "representation": "separator_music_analysis_signal", **measured})
        current_origin = origin
        if adopted:
            current_origin = CueOrigin(
                source_path=origin.source_path, source_sha256=origin.source_sha256,
                source_id=origin.source_id,
                analysis_id=f"{second['analysis_id']}+{second['recipe_id']}",
                runtime_id=second["runtime_id"], time_offset_numerator=origin.time_offset_numerator,
                time_offset_denominator=origin.time_offset_denominator)
            music_artifact = _materialize_cut(
                second_music, second_sha, cue["start_seconds"], cue["end_seconds"],
                audio_dir / f"{stem}-music-second-pass.flac",
                origin=current_origin, recipe_id=plan["recipe_id"],
                evidence={"method": METHOD, "representation": "separator_music_second_pass",
                          "first_pass_signal_sha256": music_sha,
                          "adoption": cue["second_pass"]["adoption"]["features"], **measured})
        else:
            music_artifact = first_cut(audio_dir / f"{stem}-music.flac")
        before_recovery = music_artifact
        if recovered:
            runs = vocal["runs"]
            four_stem = "+".join(sorted({runs[role]["analysis_id"] for role in runs}))
            vocal_origin = CueOrigin(
                source_path=origin.source_path, source_sha256=origin.source_sha256,
                source_id=origin.source_id,
                analysis_id=f"{current_origin.analysis_id}+{vocal['recipe_id']}+{four_stem}",
                runtime_id="+".join([current_origin.runtime_id]
                                    + sorted({runs[role]["runtime_id"] for role in runs})),
                time_offset_numerator=origin.time_offset_numerator,
                time_offset_denominator=origin.time_offset_denominator)
            music_artifact = _materialize_cut(
                vocal_music, vocal["signal"]["sha256"], cue["start_seconds"], cue["end_seconds"],
                audio_dir / f"{stem}-music-vocal-recovered.flac",
                origin=vocal_origin, recipe_id=plan["recipe_id"],
                evidence={"method": METHOD, "representation": "separator_music_vocal_recovered",
                          "before_recovery_signal_sha256": (second_sha if adopted else music_sha),
                          "adoption": cue["vocal_recovery"]["adoption"]["features"], **measured})
        cue["artifacts"] = {
            "separator_music": music_artifact,
            "raw": _materialize_cut(
                raw, raw_sha, cue["start_seconds"], cue["end_seconds"], audio_dir / f"{stem}-raw.flac",
                origin=origin, recipe_id=plan["recipe_id"],
                evidence={"method": METHOD, "representation": "original_mixture_no_op",
                          "detected_from_signal_sha256": music_sha, **measured}),
        }
        if recovered:
            # 足す前の候補も残す（どちらを採ったかを後から比べ直せるように）。
            cue["artifacts"]["separator_music_before_vocal_recovery"] = before_recovery
        if adopted:
            # 採らなかった1回目も残す（どちらを採ったかを後から比べ直せるように）。
            cue["artifacts"]["separator_music_first_pass"] = first_cut(audio_dir / f"{stem}-music.flac")
    if _sha256(source) != source_sha:
        raise EpisodeCueError("処理中に原本が変わりました")
    held = [cue for cue in plan["cues"] if cue["disposition"] == DISPOSITION_HELD]
    payload = {
        "type": "bgm_episode_cue_run", "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED_HELD_CANDIDATES" if held else "COMPLETED_NO_CANDIDATE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": source_sha,
                   "seconds": plan["source_seconds"]},
        "original": {"path": str(Path(origin.source_path).resolve()), "sha256": origin.source_sha256,
                     "source_id": origin.source_id,
                     "time_offset": [origin.time_offset_numerator, origin.time_offset_denominator]},
        "analysis": {"analysis_id": origin.analysis_id, "runtime_id": origin.runtime_id},
        "second_pass": second or {"status": "not_declared"},
        "vocal_recovery": vocal or {"status": "not_declared"},
        "analysis_stems": {stem: {"path": str(path), "sha256": _sha256(path)}
                           for stem, path in paths.items()},
        "recipe": recipe.to_dict(), "recipe_id": plan["recipe_id"],
        "input_preparation": dict(provenance or {}),
        "references": {"count": len(references),
                       "known_recording_exclusion": "applied" if references else "not_available"},
        "known_spans": known,
        "plan": plan,
        "held_count": len(held),
        "held_seconds": round(sum(cue["seconds"] for cue in held), 3),
        "rejected_count": sum(1 for cue in plan["cues"] if cue["disposition"] == DISPOSITION_REJECTED),
        "publication_state": PUBLICATION_STATE, "promotion_allowed": False,
        "claim_boundary": [
            "音楽が台詞と効果音を上回る区間を見つけたところまで。欲しい劇伴かどうかは判定していない",
            "区間は音楽の活動から決めた範囲で、曲の切れ目は確定していない",
            "参照録音が無い既発売曲・主題歌は外れていない",
            "非公開の保留候補であり、正式なBGM成果物ではない",
        ],
    }
    record = out / RECORD_NAME
    temporary = record.with_name(f".{record.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, record)
    return record


def default_recipe() -> EpisodeCueRecipe:
    """最初の宣言（2026-09-19）。窓1秒・5秒の中央値。値は実番組の既知の劇伴の窓と、
    制御素材の対象不在の窓の分布から置いた仮の値で、recallと対象不在で確かめて改める。"""
    return EpisodeCueRecipe(
        window_seconds=1.0, smoothing_windows=5, onset_share=0.15, offset_share=0.08,
        floor_music_db=-60.0, bridge_seconds=3.0, minimum_cue_seconds=8.0,
        minimum_active_fraction=0.5, exclusion_pad_seconds=1.0,
        # 実番組の既知の劇伴の境目58件で、端で捉える数が7→18件、既知の曲の内側を切る数は1件のまま
        # （`.state/validation/lv-bgm-cue-boundaries-sfc-r1/`）。
        split=cue_boundaries.CueBoundaryRecipe(minimum_interval_seconds=40.0, minimum_gap_seconds=20.0,
                                               score_threshold=1.5),
        # 台詞・効果音の下の小音量の劇伴。取り分だけでは30秒中8秒しか残らなかった制御素材が
        # 30秒、同じ条件の別の2曲と歌唱も0〜12秒→30秒になり、対象不在（台詞のみ・効果音のみ・
        # 台詞＋効果音）の保留秒は変わらない。実番組11話で既知劇伴のrecallは平均0.987→1.0
        # （`.state/validation/bgm-quiet-under-dialogue-r1/`）。長さは最小候補長と同じ。
        sustained_seconds=8.0,
        # 完結した短い音楽（アイキャッチ・ジングル等）。制御素材で3〜7秒の短い曲が、取り分が高くても
        # 台詞・効果音の下でも最小候補長で全て落ちていた（SFCは音楽として出していた）。3秒以上の
        # 途切れない存在、または取り分0.9以上が2秒続く完結した区間を残す。台詞のある場面の効果音の
        # 漏れは存在が最長2窓・取り分0.9以上は1窓で途切れる。実番組11話で曲単位の指標は変わらず、
        # 短い候補は6件26秒増えた（`.state/validation/bgm-short-cues-r1/`）。
        short_seconds=3.0, short_dominant_share=0.9,
    )
