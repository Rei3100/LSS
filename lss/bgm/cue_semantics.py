"""保留候補を曲単位（cue）へ分け直し、欲しい劇伴として公開してよいかを自動で決める。

`episode_cues` の保留候補は「音楽が台詞・効果音を上回る・途切れず続く」区間で、欲しい劇伴かどうかも、
一曲かどうかも決めていない。実番組では候補が話の9割以上を覆い、複数の曲が一つの候補にまとまる
（false merge）・一つの曲が複数の候補へ分かれる（split）・音程のある効果音や鳴き声が候補になる。

このStageは、候補を作った観測（SFCの音楽の出力の窓ごとの大きさ）に、別familyのタグ付け（`event_tagging`、
AudioSetのPANNs CNN14をSFCの音楽の出力へかけたもの）の窓ごとのクラス確率と埋め込みを足して、次を行う。

1. **まとめる**：隙間なく続く保留候補を一続き（stretch）にする。候補の境目は音楽の活動から決めたもので、
   曲の切れ目とは限らない（実番組で一つの曲が複数の候補に分かれていた）。
2. **切る**：タグ付けが「音楽に聞こえない」窓が ``gap_seconds`` 以上続く所（音楽が止まる所）で切り、
   その窓は外す（台詞・鳴き声・効果音）。実番組の既知の劇伴は、多くがこの止まり目を両端に持つ。
3. **分ける**：切った区間を二つに分けたとき、前後それぞれの平均埋め込みの距離（1−cos）が最大の位置を探し、
   ``split_distance`` 以上なら分けて、それぞれでまた探す。埋め込みは編成・雰囲気の観測で、別の曲への
   替わり目で遠い。ただし一曲の内側でも展開で大きく変わる曲があるので、分ける閾値は高く置き、
   分けなかった区間の内側の最大が ``doubt_distance`` 以上なら一曲とは言えないとしてHELDにする。
4. **決める**：分けた区間ごとに、次のいずれかにする（`SEMANTIC_CLASSES`）。
   - ``NON_BGM``：音楽に聞こえる窓が足りない。
   - ``KNOWN_RELEASED_MUSIC``：登録済みの参照録音と一致する（参照が無ければ使わない。不一致は根拠にしない）。
   - ``OTHER_NON_TARGET_MUSIC``：legacy_exclusion_v1で、話の冒頭・末尾にある長い歌もの（主題歌の形）。
   - ``UNKNOWN_OR_UNRELEASED_WANTED_BGM``：音楽が続き、長さが一曲の範囲で、両端が音楽の止まり目
     か強い替わり目で、内側に替わり目の疑いが無い。
   - ``UNCERTAIN``：上のどれとも安全に言えない（legacy_exclusion_v1の歌唱、端が候補の外へ続く、替わり目の疑い等）。HELDに残る。

欲しい劇伴を切り落として境界を綺麗にするより、曖昧ならHELDに残す。参照録音が無いこと・一致しないことを
「BGMではない」の根拠にしない。判定に正解・試聴・外部の記録を使わない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable

import numpy as np
from lss.sound import audioset

SCHEMA_VERSION = 1
METHOD = "cue-semantics-tagged-gap-split-v1"
VOCAL_POLICIES = ("legacy_exclusion_v1", "wanted_music_v2")

WANTED = "UNKNOWN_OR_UNRELEASED_WANTED_BGM"
KNOWN_RELEASED = "KNOWN_RELEASED_MUSIC"
OTHER_NON_TARGET = "OTHER_NON_TARGET_MUSIC"
NON_BGM = "NON_BGM"
UNCERTAIN = "UNCERTAIN"
# 別の原本で既に公開したcueと同じ組み立て済み音声（`lss.bgm.published_duplicates`）。判定後に付け替える。
DUPLICATE_PUBLISHED = "DUPLICATE_OF_PUBLISHED_CUE"
SEMANTIC_CLASSES = (WANTED, KNOWN_RELEASED, OTHER_NON_TARGET, NON_BGM, UNCERTAIN, DUPLICATE_PUBLISHED)

# 区間の端の根拠
EDGE_EPISODE = "episode_edge"      # 話の始まり・終わり
EDGE_SILENCE = "music_gap"         # 音楽に聞こえない窓が続いた（音楽の始まり・終わり）
EDGE_CHANGE = "strong_change"      # 埋め込みの強い替わり目で分けた
EDGE_CANDIDATE = "candidate_edge"  # 保留候補の外側でも音楽が続く（曲の頭・尻尾が候補の外へ切れている）

MUSIC_CLASS = audioset.INDEX["Music"]
SINGING_CLASSES = tuple(audioset.INDEX[name] for name in (
    "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Male singing", "Female singing",
    "Child singing", "Synthetic singing", "Rapping", "Humming", "Vocal music", "A capella",
))


class CueSemanticsError(ValueError):
    """判定の宣言か材料が契約に合わない。"""


@dataclass(frozen=True)
class CueSemanticsRecipe:
    """区切り・判定の宣言。どれかが変われば識別子が変わる。"""

    music_probability: float          # 窓を「音楽に聞こえる」とするタグの確率（SFCの音楽の出力に対して）
    floor_music_db: float             # SFCの音楽の出力がこれ未満の窓は音楽とみなさない
    gap_seconds: int                  # 音楽に聞こえない窓がこれ以上続けば切る
    split_distance: float             # 前後の平均埋め込みの距離がこれ以上なら分ける
    doubt_distance: float             # 分けなかった区間の内側の最大距離がこれ以上なら一曲とは言えない
    minimum_part_seconds: int         # 分けたとき・内側を測るときの片側の最小の長さ
    minimum_wanted_seconds: float     # 公開する劇伴の最小の長さ
    maximum_wanted_seconds: float     # 公開する劇伴の最大の長さ（これより長い区間は複数曲の疑い）
    minimum_music_fraction: float     # 公開する区間で音楽に聞こえる窓の割合の下限
    non_bgm_music_fraction: float     # これ未満ならNON_BGM
    singing_probability: float        # 窓を歌ありとする歌唱系クラスの確率（最大）
    maximum_singing_fraction: float   # 公開する区間で歌ありの窓の割合の上限
    singing_burst_seconds: int        # この長さの中に
    singing_burst_windows: int        # 歌ありの窓がこれ以上あれば、区間全体の割合が低くても歌がある（主題歌の断片等）
    theme_minimum_seconds: float      # 主題歌の形：長さ
    theme_head_fraction: float        # 主題歌の形：話の冒頭からの割合
    theme_tail_fraction: float        # 主題歌の形：話の末尾からの割合
    theme_singing_fraction: float     # 主題歌の形：歌ありの窓の割合
    known_fraction: float             # 参照一致がこの割合以上を覆えばKNOWN_RELEASED
    # 音楽の止まり目の端を、SFCの音楽の出力がこの大きさ以上で続く間だけ外へ延ばす最大の長さ（フェードの頭・尻尾）。
    # 0なら延ばさない。延ばした分は候補の外・別の区間へは入れない。
    edge_extension_seconds: int = 0
    edge_extension_floor_db: float = -60.0
    # 音楽が続くかの割合（``minimum_music_fraction`` と比べる）で、``segment`` と同じく両側を音楽の窓に挟まれた
    # ``gap_seconds`` 未満の内側の隙間を音楽の続きとして数える。ただし生の窓の割合がこの下限以上のときだけ
    # （短い隙間だらけの本体を通さない）。1.0なら隙間を数えない（生の割合のまま。導入前の宣言と同じ識別子）。
    continuity_raw_floor: float = 1.0
    vocal_policy: str = "legacy_exclusion_v1"
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD
    fractional_end_policy: str = "materialized_candidate_end_v1"

    def validate(self) -> None:
        if self.fractional_end_policy != "materialized_candidate_end_v1":
            raise CueSemanticsError("Unsupported fractional-end policy")
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise CueSemanticsError("未対応のschemaです")
        if self.vocal_policy not in VOCAL_POLICIES:
            raise CueSemanticsError(f"vocal_policyは{VOCAL_POLICIES}のいずれかである必要があります")
        for name, low, high in (
                ("music_probability", 0.0, 1.0), ("floor_music_db", -150.0, 0.0),
                ("split_distance", 0.0, 2.0), ("doubt_distance", 0.0, 2.0),
                ("minimum_wanted_seconds", 0.0, 600.0), ("maximum_wanted_seconds", 0.0, 3600.0),
                ("minimum_music_fraction", 0.0, 1.0), ("non_bgm_music_fraction", 0.0, 1.0),
                ("singing_probability", 0.0, 1.0), ("maximum_singing_fraction", 0.0, 1.0),
                ("theme_minimum_seconds", 0.0, 600.0), ("theme_head_fraction", 0.0, 1.0),
                ("theme_tail_fraction", 0.0, 1.0), ("theme_singing_fraction", 0.0, 1.0),
                ("known_fraction", 0.0, 1.0), ("continuity_raw_floor", 0.0, 1.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not low <= float(value) <= high:
                raise CueSemanticsError(f"{name}は{low}以上{high}以下である必要があります")
        for name, low, high in (("gap_seconds", 1, 60), ("minimum_part_seconds", 2, 120),
                                ("singing_burst_seconds", 1, 60), ("singing_burst_windows", 1, 60),
                                ("edge_extension_seconds", 0, 30)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise CueSemanticsError(f"{name}は{low}以上{high}以下の整数である必要があります")
        if self.singing_burst_windows > self.singing_burst_seconds:
            raise CueSemanticsError("singing_burst_windowsはsinging_burst_seconds以下である必要があります")
        if self.doubt_distance > self.split_distance:
            raise CueSemanticsError("doubt_distanceはsplit_distance以下である必要があります")
        if self.minimum_wanted_seconds > self.maximum_wanted_seconds:
            raise CueSemanticsError("minimum_wanted_secondsはmaximum_wanted_seconds以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        # 延長を使わない宣言は識別子を変えない（延長を足す前の宣言と同じ識別子のまま比べられる）。
        if body["edge_extension_seconds"] == 0:
            body.pop("edge_extension_seconds")
            body.pop("edge_extension_floor_db")
        # 隙間を数えない宣言（生の割合）も識別子を変えない。
        if body["continuity_raw_floor"] == 1.0:
            body.pop("continuity_raw_floor")
        # Preserve historical recipe identity and replay when using the legacy behavior.
        if body["vocal_policy"] == "legacy_exclusion_v1":
            body.pop("vocal_policy")
        return body

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-cue-semantics-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class WindowEvidence:
    """1秒格子（窓 i は原本の [i, i+1) 秒。タグ付けの窓は中心 i+0.5 秒）の観測。長さは揃える。"""

    music_db: np.ndarray             # SFCの音楽の出力の大きさ（dB）
    music_probability: np.ndarray    # タグ付け（SFCの音楽の出力）の Music の確率
    singing_probability: np.ndarray  # 同じく歌唱系クラスの確率の最大
    embeddings: np.ndarray           # 同じく埋め込み (窓, 次元)

    @classmethod
    def from_arrays(cls, music_db, probabilities, embeddings) -> "WindowEvidence":
        probabilities = np.asarray(probabilities, dtype=np.float32)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        music_db = np.asarray(music_db, dtype=np.float64)
        count = min(len(music_db), len(probabilities), len(embeddings))
        if (count == 0 or probabilities.ndim != 2 or embeddings.ndim != 2
                or probabilities.shape[1] <= max(SINGING_CLASSES + (MUSIC_CLASS,))):
            raise CueSemanticsError("窓ごとの観測が空か、クラス数が足りません")
        if not (np.isfinite(music_db[:count]).all() and np.isfinite(probabilities[:count]).all()
                and np.isfinite(embeddings[:count]).all()):
            raise CueSemanticsError("窓ごとの観測に有限でない値があります")
        return cls(music_db=music_db[:count], music_probability=probabilities[:count, MUSIC_CLASS],
                   singing_probability=probabilities[:count, list(SINGING_CLASSES)].max(axis=1),
                   embeddings=embeddings[:count])

    @property
    def count(self) -> int:
        return int(len(self.music_db))


def music_windows(evidence: WindowEvidence, recipe: CueSemanticsRecipe) -> np.ndarray:
    return (evidence.music_probability >= recipe.music_probability) & (evidence.music_db >= recipe.floor_music_db)


def fill_short_gaps(mask: np.ndarray, gap_seconds: int) -> np.ndarray:
    """両側を音楽の窓に挟まれた ``gap_seconds`` 未満の隙間を音楽の続きとして埋めた複製。端の隙間は埋めない。"""
    filled = np.array(mask, dtype=bool)
    for start, end in _runs(~filled):
        if end - start < gap_seconds and start > 0 and end < len(filled):
            filled[start:end] = True
    return filled


def continuity_fraction(mask: np.ndarray, weights: np.ndarray, recipe: CueSemanticsRecipe) -> tuple[float, float]:
    """本体の窓（``weights`` は窓ごとの本体に入る秒）で、音楽が続く割合と生の割合を返す。

    続く割合は ``segment`` が切らない短い内側の隙間を音楽として数える。生の割合が
    ``continuity_raw_floor`` 未満なら隙間を数えず、生の割合をそのまま返す。
    """
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if not total:
        return 0.0, 0.0
    raw = float(np.sum(mask * weights) / total)
    if raw < recipe.continuity_raw_floor:
        return raw, raw
    return float(np.sum(fill_short_gaps(mask, recipe.gap_seconds) * weights) / total), raw


def split_profile(embeddings: np.ndarray, start: int, end: int, minimum: int) -> tuple[float, int | None]:
    """[start, end) を t で二つに分けたときの、前後の平均埋め込みの距離（1−cos）の最大と、その位置。

    片側が ``minimum`` 窓に満たない位置は測らない。区間の外は使わない。
    """
    if end - start < 2 * minimum:
        return 0.0, None
    block = embeddings[start:end].astype(np.float64)
    prefix = np.cumsum(block, axis=0)
    total = prefix[-1]
    best, position = 0.0, None
    for cut in range(minimum, end - start - minimum + 1):
        before = prefix[cut - 1] / cut
        after = (total - prefix[cut - 1]) / (end - start - cut)
        norm = float(np.linalg.norm(before) * np.linalg.norm(after))
        distance = 1.0 - float(before @ after) / norm if norm > 0 else 0.0
        if distance > best:
            best, position = distance, start + cut
    return best, position


def _stretches(candidates: Iterable[tuple[float, float]]) -> list[tuple[int, int, list[tuple[float, float]]]]:
    """隙間なく（1秒以内で）続く候補を一続きにする。窓番号 [開始, 終了) と元の候補を返す。"""
    ordered = sorted((float(a), float(b)) for a, b in candidates)
    groups: list[list[tuple[float, float]]] = []
    for span in ordered:
        if groups and span[0] - groups[-1][-1][1] <= 1.0:
            groups[-1].append(span)
        else:
            groups.append([span])
    return [(int(np.floor(group[0][0] + 1e-6)), int(np.ceil(group[-1][1] - 1e-6)), group) for group in groups]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _split(embeddings: np.ndarray, start: int, end: int, recipe: CueSemanticsRecipe) -> list[tuple]:
    """強い替わり目で再帰的に二分する。(開始, 終了, 前の替わり目の距離 or None, 次の替わり目の距離 or None)。"""
    distance, position = split_profile(embeddings, start, end, recipe.minimum_part_seconds)
    if position is None or distance < recipe.split_distance:
        return [(start, end, None, None)]
    left = _split(embeddings, start, position, recipe)
    right = _split(embeddings, position, end, recipe)
    left[-1] = (*left[-1][:3], distance)
    right[0] = (right[0][0], right[0][1], distance, right[0][3])
    return left + right


def segment(candidates: Iterable[tuple[float, float]], evidence: WindowEvidence,
            recipe: CueSemanticsRecipe) -> list[dict]:
    """保留候補の範囲だけを、音楽の止まり目で切り、強い替わり目で分ける。区間と端の根拠を返す。"""
    recipe.validate()
    music = music_windows(evidence, recipe)
    segments = []
    for low, high, members in _stretches(candidates):
        high = min(high, evidence.count)
        if high <= low:
            continue
        # 短い隙間（gap_seconds未満）は音楽の続きとして埋める。長い隙間で切る。
        inside = fill_short_gaps(music[low:high], recipe.gap_seconds)
        for run_start, run_end in _runs(inside):
            first, last = low + run_start, low + run_end

            def edge(position: int, is_start: bool, distance) -> dict:
                if distance is not None:
                    return {"kind": EDGE_CHANGE, "distance": round(float(distance), 4)}
                if position <= 0 or position >= evidence.count:
                    return {"kind": EDGE_EPISODE}
                if (is_start and run_start > 0) or (not is_start and run_end < len(inside)):
                    return {"kind": EDGE_SILENCE}
                # 候補の外側でも音楽に聞こえる窓が続くなら、曲の頭・尻尾が候補の外へ切れている。
                outside = (music[max(0, position - recipe.gap_seconds):position] if is_start
                           else music[position:position + recipe.gap_seconds])
                if outside.size and not outside.any():
                    return {"kind": EDGE_SILENCE}
                return {"kind": EDGE_CANDIDATE}

            for start, end, before, after in _split(evidence.embeddings, first, last, recipe):
                start_edge, end_edge = edge(start, True, before), edge(end, False, after)
                core = [float(start), float(end)]
                if recipe.edge_extension_seconds and start_edge["kind"] == EDGE_SILENCE:
                    reach = start
                    while (start - reach < recipe.edge_extension_seconds and reach - 1 >= low
                           and not music[reach - 1] and evidence.music_db[reach - 1] >= recipe.edge_extension_floor_db):
                        reach -= 1
                    start_edge["extended_seconds"] = start - reach
                    start = reach
                if recipe.edge_extension_seconds and end_edge["kind"] == EDGE_SILENCE:
                    reach = end
                    while (reach - end < recipe.edge_extension_seconds and reach + 1 <= high
                           and not music[reach] and evidence.music_db[reach] >= recipe.edge_extension_floor_db):
                        reach += 1
                    end_edge["extended_seconds"] = reach - end
                    end = reach
                segments.append({
                    "start_seconds": float(start), "end_seconds": min(float(end), max(b for _, b in members)),
                    "core_seconds": [core[0], min(core[1], max(b for _, b in members))],
                    "start_edge": start_edge, "end_edge": end_edge,
                    "stretch": [float(low), float(high)],
                    "candidates": [[round(a, 3), round(b, 3)] for a, b in members
                                   if min(b, end) - max(a, start) > 0],
                })
    return segments


def _fraction(mask: np.ndarray) -> float:
    return round(float(np.mean(mask)), 4) if mask.size else 0.0


def decide(segments: list[dict], evidence: WindowEvidence, recipe: CueSemanticsRecipe, *,
           source_seconds: float, known_spans: Iterable[tuple[float, float]] = (),
           additional_music_windows=None) -> list[dict]:
    """区間ごとに意味の分類と理由を決める。公開してよいのは ``WANTED`` だけ。

    ``known_spans`` は登録済みの参照録音と完全に一致した原本の区間（秒）。無ければ空で、
    その場合は既発売かどうかを判断しない（一致しないことを未発売の根拠にもしない）。
    """
    recipe.validate()
    if source_seconds <= 0:
        raise CueSemanticsError("原本の長さが必要です")
    music = music_windows(evidence, recipe)
    if additional_music_windows is not None:
        positive=np.asarray(additional_music_windows)
        if positive.shape!=music.shape or positive.dtype!=np.dtype(bool):
            raise CueSemanticsError("Additional source music evidence needs one boolean per window")
        music=music|positive
    singing = evidence.singing_probability >= recipe.singing_probability
    known = [(float(a), float(b)) for a, b in known_spans]
    decided = []
    for item in segments:
        # 判定は延長前の本体で測る（延長はタグが音楽と言わなくなった後のフェードで、音楽の割合を下げるため）。
        core_start, core_end = map(float, item.get("core_seconds", (item["start_seconds"], item["end_seconds"])))
        start, end = max(0, int(np.floor(core_start))), min(evidence.count, int(np.ceil(core_end)))
        seconds = core_end-core_start
        # Fractional source intervals still contain evidence. An empty slice
        # created by int truncation cannot prove unknown music absent.
        indexes = np.arange(start,end)
        weights = np.maximum(0., np.minimum(indexes+1,core_end)-np.maximum(indexes,core_start))
        def fraction(mask):
            return round(float(np.sum(mask*weights)/np.sum(weights)),4) if weights.sum() else 0.
        # 音楽が続くかは、segmentが切らない短い内側の隙間を音楽として数える（生の割合の下限付き）。
        music_fraction, raw_music_fraction = (round(value, 4) for value in
                                              continuity_fraction(music[start:end], weights, recipe))
        singing_fraction = fraction(singing[start:end] & music[start:end])
        sung = (singing[start:end] & music[start:end]).astype(int)
        width = recipe.singing_burst_seconds
        burst = int(np.convolve(sung, np.ones(width, dtype=int), mode="valid").max()) if sung.size >= width else int(sung.sum())
        inner, _ = split_profile(evidence.embeddings, start, end, recipe.minimum_part_seconds)
        known_seconds = sum(max(0.0, min(core_end, b) - max(core_start, a)) for a, b in known)
        head = core_start / source_seconds <= recipe.theme_head_fraction
        tail = core_end / source_seconds >= 1.0 - recipe.theme_tail_fraction
        theme_shape = (seconds >= recipe.theme_minimum_seconds and (head or tail)
                       and singing_fraction >= recipe.theme_singing_fraction)
        singing_present = (singing_fraction > recipe.maximum_singing_fraction
                           or burst >= recipe.singing_burst_windows)
        features = {"seconds": seconds, "music_fraction": music_fraction,
                    "raw_music_fraction": raw_music_fraction, "singing_fraction": singing_fraction,
                    "singing_burst_windows": burst,
                    "singing_present": bool(singing_present), "theme_shape": bool(theme_shape),
                    "inner_split_distance": round(float(inner), 4),
                    "known_fraction": round(known_seconds / seconds, 4) if seconds else 0.0,
                    "position_start": round(core_start / source_seconds, 4),
                    "position_end": round(core_end / source_seconds, 4)}
        reasons = []
        if raw_music_fraction < recipe.non_bgm_music_fraction:
            if item.get("unresolved_recording_regions"):
                label = UNCERTAIN
                reasons.append("observed_recording_role_unresolved")
            else:
                label = NON_BGM
                reasons.append("music_not_heard")
        elif known and known_seconds >= recipe.known_fraction * seconds:
            label = KNOWN_RELEASED
            reasons.append("registered_reference_exact_match")
        elif recipe.vocal_policy == "legacy_exclusion_v1" and theme_shape:
            label = OTHER_NON_TARGET
            reasons.append("theme_song_shape_with_singing")
        else:
            if seconds < recipe.minimum_wanted_seconds:
                reasons.append("shorter_than_wanted_minimum")
            if seconds > recipe.maximum_wanted_seconds:
                reasons.append("longer_than_one_cue")
            if music_fraction < recipe.minimum_music_fraction:
                reasons.append("music_not_continuous")
            if recipe.vocal_policy == "legacy_exclusion_v1" and singing_present:
                reasons.append("singing_present")
            if inner >= recipe.doubt_distance:
                reasons.append("possible_change_inside")
            for side in ("start_edge", "end_edge"):
                if item[side]["kind"] == EDGE_CANDIDATE:
                    reasons.append(f"{side}_music_continues_outside_candidate")
            if known_seconds > 0:
                reasons.append("partly_registered_reference")
            label = UNCERTAIN if reasons else WANTED
        decided.append({**item, "semantic_class": label, "reasons": reasons, "features": features})
    return decided


def voice_preserving_refinement(candidates: list[dict], nominated, evidence: WindowEvidence,
                                recipe: CueSemanticsRecipe) -> tuple[list[str], dict]:
    """Do not send an observed musical voice through a destructive second pass.

    Candidate materialization selects one first/second-pass signal per candidate,
    so protect the entire candidate even if only a small overlapping phrase is
    observed. This does not authorize mixture restoration or assert a voice absent.
    """
    recipe.validate()
    kept = set(nominated)
    protected = []
    if recipe.vocal_policy == "wanted_music_v2":
        for cue in candidates:
            low = max(0, int(np.floor(cue["start_seconds"])))
            high = min(evidence.count, int(np.ceil(cue["end_seconds"])))
            observed = np.flatnonzero(evidence.singing_probability[low:high] >= recipe.singing_probability)
            if cue["cue_id"] in kept and len(observed):
                kept.remove(cue["cue_id"])
                protected.append({"cue_id": cue["cue_id"], "observed_windows": (observed + low).tolist(),
                                  "reason": "retain_first_pass_musical_voice"})
    return sorted(kept), {"policy": "observed-musical-voice-first-pass-v1", "protected": protected,
                          "original_mixture_authority": "none", "absence_authority": "none"}


def default_recipe() -> CueSemanticsRecipe:
    """最初の宣言。値は開発用の実番組（4作品の開発話数。holdoutは見ていない）で置いた
    （`.state/validation/bgm-semantic-disposition-r1/`）。

    - 分ける0.19・疑い0.13：既知の劇伴の替わり目の前後全体の距離は0.175〜0.26、一曲の内側（編集・展開を含む）は
      中央値0.05で、展開の大きい曲は0.14〜0.28まで届く。疑いを0.15にすると公開は増えるが、既知の曲の途中切れと
      未知の曲とのmergeが出る。高確信だけを公開するので0.13。
    - ``legacy_exclusion_v1`` の歌唱閾値は、過去の分類を再生するため残す。製品の既定 ``wanted_music_v2`` は歌唱や
      主題歌の形を記録するが、それらだけでwanted musicを保留・対象外にしない。
    - 音楽の止まり目の延長は使わない（既知の曲の頭の誤差が悪化し、音楽に聞こえない秒が増えるだけだった）。
    - 続く割合は、segmentが切らない3秒未満の内側の隙間を音楽として数える（ep1 u11：小さい鉄琴の曲で、タグが
      1〜2秒だけ0.5を割る5窓のために35/40=0.875で保留されていた）。生の割合0.8未満の本体は数えない（半分が
      短い隙間の本体を通さない）。
    """
    return CueSemanticsRecipe(
        music_probability=0.5, floor_music_db=-60.0, gap_seconds=3,
        split_distance=0.19, doubt_distance=0.13, minimum_part_seconds=8,
        minimum_wanted_seconds=10.0, maximum_wanted_seconds=150.0,
        minimum_music_fraction=0.9, non_bgm_music_fraction=0.3,
        singing_probability=0.15, maximum_singing_fraction=0.1, singing_burst_seconds=6,
        singing_burst_windows=3, theme_minimum_seconds=45.0,
        theme_head_fraction=0.2, theme_tail_fraction=0.15, theme_singing_fraction=0.25, known_fraction=0.5,
        vocal_policy="wanted_music_v2", continuity_raw_floor=0.8)
