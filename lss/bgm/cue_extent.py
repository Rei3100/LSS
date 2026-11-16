"""候補が一曲かどうかを、既知録音との一致だけから読み取る。

`docs/BGM.md` の「Cue formation」と「Known OSTとIdentity」に対応する。音を作らず、
ファイルを書かず、照合もしない。既に記録された一致から結論と理由だけを返す。

一致は疎なので、多くの候補は`unresolved`のままになる。`unresolved`は「一曲ではない」
ではなく「一曲だと示せていない」である。判断に使うしきい値は呼び出し側が先に宣言する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

SCHEMA_VERSION = 1
METHOD = "released-reference-cue-extent-v1"
STATES = ("single_cue", "multiple_cues", "unresolved")
# 一致の出どころ。既発売の参照録音と、別の回に同じ録音が出ること（未発売を含む）を混ぜない。
EVIDENCE_KINDS = ("released_recording", "recurring_recording")
UNRESOLVED_REASONS = (
    "no_usable_match",
    "single_recording_coverage_below_minimum",
    "matched_offsets_not_one_continuous_playback",
)


class CueExtentError(ValueError):
    """範囲の読み取りの宣言、または渡された一致が契約に合わない。"""


@dataclass(frozen=True)
class CueExtentRecipe:
    """一曲だと言うために必要な一致の強さ・覆い・連続性を先に固定する。"""

    minimum_similarity: float
    minimum_single_recording_coverage: float
    maximum_offset_spread_seconds: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise CueExtentError("未対応の範囲読み取りschemaです")
        for field in ("minimum_similarity", "minimum_single_recording_coverage"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not 0.0 < float(value) <= 1.0):
                raise CueExtentError(f"{field}は0より大きく1以下である必要があります")
        value = self.maximum_offset_spread_seconds
        if (isinstance(value, bool) or not isinstance(value, Real) or float(value) < 0.0
                or float(value) == float("inf")):
            raise CueExtentError("maximum_offset_spread_secondsは0以上の有限な秒数である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-cue-extent-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _seconds(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise CueExtentError(f"{context}が不正です: {value!r}")
    return float(value)


def _match(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise CueExtentError("一致はオブジェクトである必要があります")
    reference_id = raw.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id.strip():
        raise CueExtentError("一致にreference_idが必要です")
    start = _seconds(raw.get("query_start_seconds"), "query_start_seconds")
    end = _seconds(raw.get("query_end_seconds"), "query_end_seconds")
    if end <= start:
        raise CueExtentError(f"一致の区間が空です: {reference_id}")
    return {
        "reference_id": reference_id.strip(), "query_start_seconds": start,
        "query_end_seconds": end,
        "offset_seconds": _seconds(raw.get("offset_seconds"), "offset_seconds"),
        "similarity": _seconds(raw.get("similarity"), "similarity"),
    }


def _covered_seconds(spans: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def read_cue_extent(
    start_seconds: float, end_seconds: float, matches: list, recipe: CueExtentRecipe,
    *, evidence_kind: str = "released_recording",
) -> dict:
    """候補一つの範囲を読む。公開可否も意味的roleも決めない。"""
    recipe.validate()
    if evidence_kind not in EVIDENCE_KINDS:
        raise CueExtentError(f"未対応の一致の出どころです: {evidence_kind}")
    start, end = _seconds(start_seconds, "start_seconds"), _seconds(end_seconds, "end_seconds")
    if end <= start:
        raise CueExtentError("候補の区間が空です")
    rows = [_match(raw) for raw in matches]
    usable = [row for row in rows
              if row["similarity"] >= recipe.minimum_similarity
              and row["query_start_seconds"] < end and start < row["query_end_seconds"]]
    references = sorted({row["reference_id"] for row in usable})

    def result(state: str, **extra) -> dict:
        if state not in STATES:
            raise CueExtentError(f"未対応の範囲の状態です: {state}")
        return {"state": state, "recipe_id": recipe.recipe_id,
                "evidence": {"matched_reference_ids": references,
                             "usable_match_count": len(usable),
                             "evidence_kind": evidence_kind, **extra}}

    if len(references) > 1:
        # 別々の録音が一つの候補に並んでいる。曲単位ではないことが示された。
        return result("multiple_cues", reason=f"more_than_one_{evidence_kind}")
    if not references:
        return result("unresolved", reason="no_usable_match")
    covered = _covered_seconds([(max(start, row["query_start_seconds"]),
                                 min(end, row["query_end_seconds"])) for row in usable])
    coverage = covered / (end - start)
    offsets = [row["offset_seconds"] for row in usable]
    spread = max(offsets) - min(offsets)
    common = {"single_recording_coverage": coverage, "offset_spread_seconds": spread}
    if coverage < recipe.minimum_single_recording_coverage:
        return result("unresolved", reason="single_recording_coverage_below_minimum", **common)
    if spread > recipe.maximum_offset_spread_seconds:
        # 同じ録音でも、参照側の位置が揃わなければ一続きの再生とは言えない。
        return result("unresolved", reason="matched_offsets_not_one_continuous_playback", **common)
    return result("single_cue", reason=f"one_continuous_{evidence_kind}", **common)


def combine_readings(readings: list) -> dict:
    """複数の読み取りをまとめる。一つでも複数曲と言えば複数曲、根拠は全部残す。

    「一曲である」は、複数曲と言う読み取りが一つも無いときにだけ通す。読み取り同士が
    食い違ったまま公開へ進めない。
    """
    rows = list(readings)
    if not rows:
        raise CueExtentError("まとめる読み取りがありません")
    for row in rows:
        if not isinstance(row, dict) or row.get("state") not in STATES:
            raise CueExtentError(f"読み取りの状態が不正です: {row!r}")
    evidence = {"readings": [{"state": row["state"], "evidence": row.get("evidence")}
                             for row in rows]}
    if any(row["state"] == "multiple_cues" for row in rows):
        return {"state": "multiple_cues", "evidence": {**evidence, "reason": "at_least_one_reading_says_more_than_one_cue"}}
    if any(row["state"] == "single_cue" for row in rows):
        return {"state": "single_cue", "evidence": {**evidence, "reason": "a_reading_says_one_cue_and_none_disagree"}}
    return {"state": "unresolved", "evidence": {**evidence, "reason": "no_reading_decides"}}


def matches_from_identity_evidence(candidate: dict) -> list[dict]:
    """候補に記録済みの既知録音の一致を、範囲の読み取りが受け取る形へ直す。

    記録にある値だけを使う。参照側の位置が無い根拠は、一続きの再生かどうかを言えない
    ので使わない。
    """
    if not isinstance(candidate, dict):
        raise CueExtentError("候補はオブジェクトである必要があります")
    start = _seconds(candidate.get("start_seconds"), "start_seconds")
    end = _seconds(candidate.get("end_seconds"), "end_seconds")
    if end <= start:
        raise CueExtentError("候補の区間が空です")
    rows = []
    for item in candidate.get("identity_evidence") or []:
        if not isinstance(item, dict):
            raise CueExtentError("一致の根拠はオブジェクトである必要があります")
        reference_start = item.get("reference_start_seconds")
        covered = (item.get("covered_fraction") or 0.0) * (end - start)
        if reference_start is None or covered <= 0.0:
            continue
        rows.append({
            "reference_id": item.get("reference_id") or item["reference_sha256"],
            "query_start_seconds": start, "query_end_seconds": start + covered,
            "offset_seconds": _seconds(reference_start, "reference_start_seconds") - start,
            "similarity": item.get("waveform_explained") or 0.0,
        })
    return rows


SPAN_TOLERANCE_SECONDS = 0.05


def extent_for_span(start_seconds: float, end_seconds: float, children: list) -> dict | None:
    """親の区間の範囲を、その中にある子候補の読み取りからまとめる。

    子候補は親を切り分けたものなので、`combine_readings` のように同じ区間の複数の読み取り
    としては扱えない。決まった読み取りが二つ以上あれば、その親は一曲ではない。

    一曲だと言えるのは、親の端まで覆う子が一つだけ一曲だと言ったときに限る。それ以外は
    何も返さない。根拠の無い候補へ読み取りを付けないためである。
    """
    start = _seconds(start_seconds, "start_seconds")
    end = _seconds(end_seconds, "end_seconds")
    if end <= start:
        raise CueExtentError("親の区間が空です")
    inside = []
    for child in children:
        if not isinstance(child, dict):
            raise CueExtentError("子候補はオブジェクトである必要があります")
        reading = child.get("cue_extent")
        if not isinstance(reading, dict) or reading.get("state") not in STATES:
            continue
        child_start = _seconds(child.get("start_seconds"), "start_seconds")
        child_end = _seconds(child.get("end_seconds"), "end_seconds")
        if (child_start < start - SPAN_TOLERANCE_SECONDS
                or child_end > end + SPAN_TOLERANCE_SECONDS):
            continue
        inside.append((child_start, child_end, reading))

    decided = [row for row in inside if row[2]["state"] != "unresolved"]
    evidence = {"child_reading_count": len(inside),
                "decided_child_count": len(decided),
                "readings": [{"start_seconds": row[0], "end_seconds": row[1],
                              "state": row[2]["state"],
                              "evidence": row[2].get("evidence")} for row in decided]}
    if any(row[2]["state"] == "multiple_cues" for row in decided):
        return {"state": "multiple_cues",
                "evidence": {**evidence, "reason": "a_child_holds_more_than_one_cue"}}
    if len(decided) > 1:
        return {"state": "multiple_cues",
                "evidence": {**evidence, "reason": "more_than_one_child_is_its_own_cue"}}
    if len(decided) == 1:
        child_start, child_end, reading = decided[0]
        if (abs(child_start - start) <= SPAN_TOLERANCE_SECONDS
                and abs(child_end - end) <= SPAN_TOLERANCE_SECONDS):
            return {"state": "single_cue",
                    "evidence": {**evidence, "reason": "one_child_is_one_cue_and_covers_the_span"}}
    return None
