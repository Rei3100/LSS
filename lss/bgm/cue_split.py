"""根拠のある止まりの位置で、候補の区間を切り分ける。

音は作らず、ファイルも書かない。区間と、その区間がどこから来たかだけを返す。切った
断片は新しい候補ではなく、同じ親候補の一部として扱う。

止まりの内側は音楽が無いので、どちらの断片にも入れない。短すぎる断片は捨てずに、
短いという理由を付けて返す。何を公開するかはここでは決めない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

SCHEMA_VERSION = 1
METHOD = "split-at-evidenced-music-stops-v1"


class CueSplitError(ValueError):
    """切り分けの宣言、または渡された区間が契約に合わない。"""


@dataclass(frozen=True)
class CueSplitRecipe:
    minimum_piece_seconds: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise CueSplitError("未対応の切り分けschemaです")
        value = self.minimum_piece_seconds
        if (isinstance(value, bool) or not isinstance(value, Real)
                or value != value or value <= 0 or value == float("inf")):
            raise CueSplitError("minimum_piece_secondsは有限な正数である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-cue-split-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _seconds(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise CueSplitError(f"{context}が不正です: {value!r}")
    return float(value)


def split_at_gaps(
    start_seconds: float, end_seconds: float, gaps: list, recipe: CueSplitRecipe,
) -> dict:
    """止まりの外側だけを断片として返す。親の区間と使った止まりを残す。"""
    recipe.validate()
    start, end = _seconds(start_seconds, "start_seconds"), _seconds(end_seconds, "end_seconds")
    if end <= start:
        raise CueSplitError("候補の区間が空です")

    spans = []
    for gap in gaps:
        if not isinstance(gap, dict):
            raise CueSplitError("止まりはオブジェクトである必要があります")
        gap_start = _seconds(gap.get("start_seconds"), "止まりのstart_seconds")
        gap_end = _seconds(gap.get("end_seconds"), "止まりのend_seconds")
        if gap_end <= gap_start:
            raise CueSplitError("止まりの区間が空です")
        if gap_start < start or gap_end > end:
            raise CueSplitError("止まりが候補の外にあります")
        spans.append((gap_start, gap_end))
    spans.sort()
    for (first_start, first_end), (second_start, _) in zip(spans, spans[1:]):
        if second_start < first_end:
            raise CueSplitError("止まりが重なっています")

    pieces = []
    cursor = start
    for gap_start, gap_end in spans + [(end, end)]:
        seconds = gap_start - cursor
        if seconds > 0:
            pieces.append({
                "start_seconds": cursor, "end_seconds": gap_start, "seconds": seconds,
                "usable": seconds >= recipe.minimum_piece_seconds,
                "reason": ("within_declared_minimum_piece_length" if seconds >= recipe.minimum_piece_seconds
                           else "piece_below_minimum_length"),
            })
        cursor = gap_end
    return {
        "recipe_id": recipe.recipe_id,
        "parent": {"start_seconds": start, "end_seconds": end, "seconds": end - start},
        "gaps_used": [{"start_seconds": item[0], "end_seconds": item[1]} for item in spans],
        "pieces": pieces,
        "usable_piece_count": sum(1 for piece in pieces if piece["usable"]),
    }
