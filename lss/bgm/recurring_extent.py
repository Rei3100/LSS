"""別の回に同じ録音が現れることだけから、候補が一曲かどうかを読めるようにする。

`docs/BGM.md` の「Cue formation」「Occurrence（使用箇所）」「Single-occurrenceはfirst-class」
に対応する。既発売の参照録音は番組の一部にしか付かないので、参照が無い曲の範囲は
既知録音だけでは読めない。別の回に同じ録音が出るなら、その区間は一つの録音の一続きの
再生であり、参照が無くても範囲の根拠になる。

ここでは記録済みの共通録音を、範囲の読み取りが受け取る形へ直すだけである。音を作らず、
照合もせず、意味的role・公開可否・同一性の名前は決めない。共通録音が無いことは、
その候補が一曲でない証拠にも、単独出現でない証拠にもしない。
"""
from __future__ import annotations

from numbers import Real
from typing import Any

from lss.bgm.cue_extent import (
    CueExtentRecipe,
    combine_readings,
    matches_from_identity_evidence,
    read_cue_extent,
)

SCHEMA_VERSION = 1
METHOD = "recurring-recording-occurrence-v1"
EVIDENCE_KIND = "recurring_recording"
# 同じずれで隣り合う区間は、一つの再生が切れて記録されたものとして読む。
# `lss/bgm/identity.py` の共通録音のつなぎ方と同じ許容を使う。
SAME_OFFSET_SECONDS = 0.05
CONTIGUOUS_SECONDS = 2.5


class RecurringExtentError(ValueError):
    """渡された共通録音の記録が契約に合わない。"""


def _seconds(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise RecurringExtentError(f"{context}が不正です: {value!r}")
    return float(value)


def _span(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise RecurringExtentError("共通録音の区間はオブジェクトである必要があります")
    other = raw.get("other_episode")
    if not isinstance(other, str) or not other.strip():
        raise RecurringExtentError("共通録音にother_episodeが必要です")
    start = _seconds(raw.get("start_seconds"), "start_seconds")
    end = _seconds(raw.get("end_seconds"), "end_seconds")
    if end <= start:
        raise RecurringExtentError(f"共通録音の区間が空です: {other}")
    other_start = _seconds(raw.get("other_start_seconds"), "other_start_seconds")
    other_end = _seconds(raw.get("other_end_seconds"), "other_end_seconds")
    if other_end <= other_start:
        raise RecurringExtentError(f"相手側の区間が空です: {other}")
    explained = raw.get("waveform_explained")
    if explained is None:
        raise RecurringExtentError(f"共通録音にwaveform_explainedが必要です: {other}")
    explained = _seconds(explained, "waveform_explained")
    if not 0.0 <= explained <= 1.0:
        raise RecurringExtentError(f"waveform_explainedは0以上1以下である必要があります: {other}")
    same_timeline = raw.get("same_timeline", False)
    if not isinstance(same_timeline, bool):
        raise RecurringExtentError(f"same_timelineは真偽値である必要があります: {other}")
    return {"other_episode": other.strip(), "start_seconds": start, "end_seconds": end,
            "other_start_seconds": other_start, "other_end_seconds": other_end,
            "waveform_explained": explained, "same_timeline": same_timeline}


def _same_recording(first: dict, second: dict) -> bool:
    """同じ録音かどうかは、相手側の区間か自分側の区間が重なるかで読む。

    同じ録音が相手の回で二度鳴っていれば相手側は離れるが、自分側は同じ区間を指す。
    逆に自分の回で二度鳴っていれば自分側は離れるが、相手側は同じ区間を指す。
    どちらも重ならなくても、同じずれで隣り合うなら一つの再生が切れて記録されたと読む。
    同じ番組の中の再演（`same_timeline`）では、相手側の区間も同じ時間軸で読める。
    """
    if first["other_episode"] != second["other_episode"]:
        return False
    if (first["other_start_seconds"] < second["other_end_seconds"]
            and second["other_start_seconds"] < first["other_end_seconds"]):
        return True
    if (first["start_seconds"] < second["end_seconds"]
            and second["start_seconds"] < first["end_seconds"]):
        return True
    if first["same_timeline"] and second["same_timeline"]:
        # 同じ番組の中の再演では、相手側の区間も自分側と同じ時間軸にある。
        # 片方の登場がもう片方の相手側と重なるなら、同じ録音である。
        for left, right in ((first, second), (second, first)):
            if (left["other_start_seconds"] < right["end_seconds"]
                    and right["start_seconds"] < left["other_end_seconds"]):
                return True
    offsets = [row["other_start_seconds"] - row["start_seconds"] for row in (first, second)]
    if abs(offsets[0] - offsets[1]) > SAME_OFFSET_SECONDS:
        return False
    gap = max(first["start_seconds"] - second["end_seconds"],
              second["start_seconds"] - first["end_seconds"])
    return gap <= CONTIGUOUS_SECONDS


def recurring_matches(spans: Any) -> list[dict]:
    """共通録音の区間を、範囲の読み取りが受け取る一致の形へ直す。

    同じ録音を指す区間には同じ名前を付ける。名前は録音の題名ではなく、この読み取りの
    中だけで通じる呼び名である。
    """
    if not isinstance(spans, list):
        raise RecurringExtentError("共通録音の一覧が必要です")
    rows = [_span(raw) for raw in spans]
    rows.sort(key=lambda row: (row["other_episode"], row["start_seconds"],
                               row["other_start_seconds"]))

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _same_recording(rows[left], rows[right]):
                parent[find(right)] = find(left)

    names: dict[int, str] = {}
    for index, row in enumerate(rows):
        root = find(index)
        if root not in names:
            names[root] = (f"recurring:{row['other_episode']}:"
                           f"{len(names):04d}")
    return [{
        "reference_id": names[find(index)],
        "query_start_seconds": row["start_seconds"],
        "query_end_seconds": row["end_seconds"],
        "offset_seconds": row["other_start_seconds"] - row["start_seconds"],
        "similarity": row["waveform_explained"],
        "other_episode": row["other_episode"],
    } for index, row in enumerate(rows)]


def spans_for_episode(occurrences: Any, *, first: str, second: str) -> dict[str, list[dict]]:
    """番組の組ごとの共通録音を、番組ごとの区間へ並べ直す。"""
    if not isinstance(occurrences, list):
        raise RecurringExtentError("共通録音の一覧が必要です")
    if not isinstance(first, str) or not isinstance(second, str) or first == second:
        raise RecurringExtentError("別々の番組の識別子が必要です")
    spans: dict[str, list[dict]] = {first: [], second: []}
    for raw in occurrences:
        if not isinstance(raw, dict):
            raise RecurringExtentError("共通録音はオブジェクトである必要があります")
        explained = raw.get("waveform_explained")
        spans[first].append(_span({
            "other_episode": second,
            "start_seconds": raw.get("first_start_seconds"),
            "end_seconds": raw.get("first_end_seconds"),
            "other_start_seconds": raw.get("second_start_seconds"),
            "other_end_seconds": raw.get("second_end_seconds"),
            "waveform_explained": explained}))
        spans[second].append(_span({
            "other_episode": first,
            "start_seconds": raw.get("second_start_seconds"),
            "end_seconds": raw.get("second_end_seconds"),
            "other_start_seconds": raw.get("first_start_seconds"),
            "other_end_seconds": raw.get("first_end_seconds"),
            "waveform_explained": explained}))
    return spans


def spans_from_evidence(spans) -> list[dict]:
    """（開始, 終了, 根拠）の並びを、範囲の読み取りが受け取る形へ直す。

    相手側の時刻を持たない根拠は落とす。無い値を作らない。
    """
    rows = []
    for start, end, detail in spans:
        if detail.get("other_start_seconds") is None or detail.get("other_end_seconds") is None:
            continue
        rows.append({"other_episode": detail["other_episode"],
                     "start_seconds": float(start), "end_seconds": float(end),
                     "other_start_seconds": float(detail["other_start_seconds"]),
                     "other_end_seconds": float(detail["other_end_seconds"]),
                     "waveform_explained": detail.get("waveform_explained") or 0.0,
                     "same_timeline": bool(detail.get("same_timeline", False))})
    return rows


def read_extents_for_candidates(candidates: list[dict], spans, recipe: CueExtentRecipe,
                                *, gap_readings: dict | None = None) -> None:
    """候補ごとに「一曲だと言えるか」を付ける。読み取りだけで、候補は捨てない。

    既知録音の一致、同じ録音の再登場、候補の内側で音楽が止まることを別々に読んでから
    まとめる。食い違えば保留になる。根拠が無い候補には読み取りを付けない。
    """
    matches = recurring_matches(spans_from_evidence(spans))
    for candidate in candidates:
        start, end = float(candidate["start_seconds"]), float(candidate["end_seconds"])
        inside = [row for row in matches
                  if row["query_start_seconds"] < end and start < row["query_end_seconds"]]
        readings = [read_cue_extent(start, end, matches_from_identity_evidence(candidate),
                                    recipe),
                    read_recurring_extent(start, end, inside, recipe)]
        gap = (gap_readings or {}).get(candidate["candidate_id"])
        if gap is not None:
            readings.append(gap)
        if all(row["state"] == "unresolved" for row in readings):
            continue
        candidate["cue_extent"] = combine_readings(readings)


def read_recurring_extent(
    start_seconds: float, end_seconds: float, matches: list, recipe: CueExtentRecipe,
) -> dict:
    """再登場の証拠だけで候補の範囲を読む。複数曲だとはここでは言わない。

    呼び名が分かれていても、同じ録音の別の部分が相手の回の別の場所で鳴っているだけの
    ことがある。再登場の証拠は範囲の下限を示せても、二つの区間が別の録音だとは示せない。
    不確かな同一性で候補を捨てないため、分かれた場合は保留にする。
    """
    read = read_cue_extent(start_seconds, end_seconds, matches, recipe,
                           evidence_kind=EVIDENCE_KIND)
    if read["state"] != "multiple_cues":
        return read
    return {"state": "unresolved", "recipe_id": read["recipe_id"],
            "evidence": {**read["evidence"],
                         "reason": "recurring_names_may_be_one_recording"}}
