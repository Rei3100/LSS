"""記録済みの一致から、曲が入れ替わる裏付けのある境目を読み取る。

`docs/BGM.md` の「Cue formation」と「Known OSTとIdentity」に対応する。音を作らず、
照合もしない。既に記録された一致だけを読み直し、別々の録音が続けて鳴っている位置を返す。

信号の内側だけを見る観測（落ち込み・自己相似・止まり）は、単独では曲の境目を決められ
なかった。ここで返す位置は信号の外側からの根拠なので、その観測を数えて確かめるための
材料になる。一致は疎なので、これで番組の全ての境目が並ぶわけではない。

同じ録音が続けて一致する場合は境目として数えない。参照位置の飛びは繰り返しの多い曲で
説明できないことが分かっているためである。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

SCHEMA_VERSION = 1
METHOD = "recorded-identity-join-v1"
DROP_REASONS = (
    "similarity_below_minimum",
    "shorter_than_minimum",
    "overlaps_a_stronger_match",
)


class IdentityJoinError(ValueError):
    """境目の読み取りの宣言、または渡された一致が契約に合わない。"""


@dataclass(frozen=True)
class IdentityJoinRecipe:
    """境目と言うために必要な一致の強さ・長さと、間に入れる隙間を先に固定する。"""

    minimum_similarity: float
    minimum_match_seconds: float
    maximum_join_seconds: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise IdentityJoinError("未対応の境目読み取りschemaです")
        value = self.minimum_similarity
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not 0.0 < float(value) <= 1.0):
            raise IdentityJoinError("minimum_similarityは0より大きく1以下である必要があります")
        value = self.minimum_match_seconds
        if (isinstance(value, bool) or not isinstance(value, Real) or float(value) <= 0.0
                or float(value) == float("inf")):
            raise IdentityJoinError("minimum_match_secondsは有限な正数である必要があります")
        value = self.maximum_join_seconds
        if (isinstance(value, bool) or not isinstance(value, Real) or float(value) < 0.0
                or float(value) == float("inf")):
            raise IdentityJoinError("maximum_join_secondsは0以上の有限な秒数である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-identity-join-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _seconds(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise IdentityJoinError(f"{context}が不正です: {value!r}")
    return float(value)


def _match(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise IdentityJoinError("一致はオブジェクトである必要があります")
    reference_id = raw.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id.strip():
        raise IdentityJoinError("一致にreference_idが必要です")
    start = _seconds(raw.get("query_start_seconds"), "query_start_seconds")
    end = _seconds(raw.get("query_end_seconds"), "query_end_seconds")
    if end <= start:
        raise IdentityJoinError(f"一致の区間が空です: {reference_id}")
    similarity = _seconds(raw.get("similarity"), "similarity")
    if not 0.0 <= similarity <= 1.0:
        raise IdentityJoinError(f"similarityは0以上1以下である必要があります: {reference_id}")
    return {"reference_id": reference_id, "start_seconds": start, "end_seconds": end,
            "similarity": similarity}


def _accept_without_overlap(usable: list[dict]) -> tuple[list[dict], list[dict]]:
    """重なった一致は同時には読めない。似ている方を残し、落とした方を理由付きで返す。"""
    accepted: list[dict] = []
    dropped: list[dict] = []
    order = sorted(usable, key=lambda row: (-row["similarity"], row["start_seconds"],
                                            row["reference_id"]))
    for row in order:
        if any(row["start_seconds"] < kept["end_seconds"]
               and kept["start_seconds"] < row["end_seconds"] for kept in accepted):
            dropped.append({**row, "reason": "overlaps_a_stronger_match"})
            continue
        accepted.append(row)
    accepted.sort(key=lambda row: row["start_seconds"])
    dropped.sort(key=lambda row: row["start_seconds"])
    return accepted, dropped


def _spans(accepted: list[dict], recipe: IdentityJoinRecipe) -> list[dict]:
    """同じ録音が続く一致は一つの再生としてまとめる。飛びは境目として数えない。"""
    spans: list[dict] = []
    for row in accepted:
        if (spans and spans[-1]["reference_id"] == row["reference_id"]
                and row["start_seconds"] - spans[-1]["end_seconds"] <= recipe.maximum_join_seconds):
            spans[-1]["end_seconds"] = max(spans[-1]["end_seconds"], row["end_seconds"])
            spans[-1]["match_count"] += 1
            spans[-1]["maximum_similarity"] = max(spans[-1]["maximum_similarity"],
                                                  row["similarity"])
            continue
        spans.append({"reference_id": row["reference_id"],
                      "start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"],
                      "match_count": 1, "maximum_similarity": row["similarity"]})
    return spans


def read_joins(matches: Any, recipe: IdentityJoinRecipe) -> dict:
    """一致の並びから、別々の録音が続けて鳴る位置だけを境目として返す。"""
    recipe.validate()
    if not isinstance(matches, list):
        raise IdentityJoinError("一致の一覧が必要です")
    rows = [_match(raw) for raw in matches]

    usable, dropped = [], []
    for row in rows:
        if row["similarity"] < recipe.minimum_similarity:
            dropped.append({**row, "reason": "similarity_below_minimum"})
        elif row["end_seconds"] - row["start_seconds"] < recipe.minimum_match_seconds:
            dropped.append({**row, "reason": "shorter_than_minimum"})
        else:
            usable.append(row)

    accepted, overlapped = _accept_without_overlap(usable)
    dropped.extend(overlapped)
    dropped.sort(key=lambda row: (row["start_seconds"], row["reference_id"]))
    spans = _spans(accepted, recipe)

    joins, separated = [], []
    for before, after in zip(spans, spans[1:]):
        if before["reference_id"] == after["reference_id"]:
            # 離れていても同じ録音である。参照位置の飛びは境目の根拠にしない。
            continue
        gap = after["start_seconds"] - before["end_seconds"]
        record = {
            "before_reference_id": before["reference_id"],
            "after_reference_id": after["reference_id"],
            "earliest_seconds": before["end_seconds"],
            "latest_seconds": after["start_seconds"],
            "seconds": (before["end_seconds"] + after["start_seconds"]) / 2.0,
            "gap_seconds": gap,
        }
        (joins if gap <= recipe.maximum_join_seconds else separated).append(record)

    return {
        "recipe_id": recipe.recipe_id,
        "match_count": len(rows),
        "usable_count": len(usable),
        "accepted_count": len(accepted),
        "dropped": dropped,
        "spans": spans,
        "joins": joins,
        "separated_changes": separated,
    }
