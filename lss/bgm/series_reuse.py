"""シリーズ内で繰り返される録音を、組ごとの一致から録音単位のまとまりへ直す。

原本mixどうしの照合は「話Aの区間」と「話Bの区間」の**組**を返す。同じ録音が3話に出れば
組は3つできるので、そのまま数えると1曲が3件に割れる。このStageは組をたどって
録音単位のまとまりへ直し、話ごとの使用箇所を並べる。

あわせて二つを分ける。

- **既発売**：使用箇所が既発売録音との一致で説明できるなら、そのまとまりは既知として印を付ける。
  `docs/BGM.md` のとおり、既知録音は同定はするが重複公開しない。
- **区間ごと同一の使い回し**：説明率がほぼ1.0のまとまりは、台詞違いで同じ曲が鳴っているのではなく
  音声が丸ごと同じ区間の流用（局ID、アイキャッチ、バンク）である。実番組の測定では、
  corpus-c の3話で11件・corpus-a の3話で3件がこの形で、corpus-a の3件は全て0.2秒位置の14.0秒だった
  （`.state/validation/lv-bgm-series-unreleased-r1/disposition.json`）。劇伴として扱うと
  毎話同じ区間を曲として拾ってしまう。

まとまりを作るだけで、使用箇所を統合・削除しない。意味役割（欲しい劇伴か）は判定しない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
METHOD = "series-recurrence-grouping-v1"

REUSE_IDENTICAL = "identical_segment_reuse"
REUSE_OTHER_CONTEXT = "same_recording_other_context"
KNOWN_RELEASED = "released_reference_match"


class SeriesReuseError(ValueError):
    """まとまりの宣言、または渡された一致が契約に合わない。"""


@dataclass(frozen=True)
class SeriesReuseRecipe:
    """シリーズ内の再演をまとめる宣言。"""

    identical_explained: float
    released_overlap_share: float
    occurrence_overlap_share: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise SeriesReuseError("未対応の再演まとめschemaです")
        for name in ("identical_explained", "released_overlap_share", "occurrence_overlap_share"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or value != value
                    or not 0.0 < float(value) <= 1.0):
                raise SeriesReuseError(f"{name}は0より大きく1以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-series-reuse-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SeriesReuseError(f"{context}が未指定です")
    return value.strip()


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise SeriesReuseError(f"{context}が不正です: {value!r}")
    return float(value)


def _span(value: Any, prefix: str, record: Mapping[str, Any]) -> tuple[str, float, float]:
    label = _text(record.get(prefix), f"{prefix}")
    start = _number(record.get(f"{prefix}_start_seconds"), f"{prefix}_start_seconds")
    end = _number(record.get(f"{prefix}_end_seconds"), f"{prefix}_end_seconds")
    if end <= start:
        raise SeriesReuseError(f"{prefix}の区間が空です")
    del value
    return label, start, end


def overlap_share(first: tuple[float, float], second: tuple[float, float]) -> float:
    """短い方に対する重なりの割合。"""
    shared = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    shortest = min(first[1] - first[0], second[1] - second[0])
    return shared / shortest if shortest > 0 else 0.0


def covered_by(span: tuple[float, float], spans: Iterable[Any], share: float) -> bool:
    """区間が、与えた区間群にどれだけ覆われているか。"""
    start, end = span
    if end <= start:
        return True
    total = 0.0
    for other in spans:
        other_start, other_end = float(other[0]), float(other[1])
        total += max(0.0, min(end, other_end) - max(start, other_start))
    return total / (end - start) >= share


class _Union:
    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, first, second) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def group_recurrence(
    pairs: Iterable[Any], recipe: SeriesReuseRecipe, *,
    released_spans: Mapping[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """組ごとの一致を、録音単位のまとまりへ直す。

    同じ話の中で十分に重なる区間は同じ使用箇所とみなし、組でつながった使用箇所は
    同じ録音のまとまりにする。まとまりごとに、既発売かどうかと、区間ごと同一の
    使い回しかどうかを付ける。
    """
    recipe.validate()
    released = {key: [(float(item[0]), float(item[1])) for item in value]
                for key, value in (released_spans or {}).items()}

    nodes: list[tuple[str, float, float]] = []
    records = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise SeriesReuseError("一致はオブジェクトである必要があります")
        first = _span(None, "first", pair)
        second = _span(None, "second", pair)
        explained = _number(pair.get("waveform_explained", 0.0), "waveform_explained")
        if not 0.0 <= explained <= 1.0:
            raise SeriesReuseError(f"waveform_explainedは0以上1以下である必要があります: {explained}")
        records.append((first, second, explained))
        nodes.extend((first, second))

    union = _Union()
    for node in nodes:
        union.find(node)
    # 同じ話の中で十分重なる区間は同じ使用箇所
    for index, node in enumerate(nodes):
        for other in nodes[index + 1:]:
            if node[0] != other[0]:
                continue
            if overlap_share((node[1], node[2]), (other[1], other[2])) >= recipe.occurrence_overlap_share:
                union.union(node, other)
    for first, second, _ in records:
        union.union(first, second)

    components: dict[Any, list[tuple[str, float, float]]] = {}
    for node in nodes:
        components.setdefault(union.find(node), []).append(node)
    explained_of: dict[Any, list[float]] = {}
    for first, second, explained in records:
        explained_of.setdefault(union.find(first), []).append(explained)

    groups = []
    for position, (root, members) in enumerate(sorted(
            components.items(), key=lambda item: (-len(set(item[1])), sorted(set(item[1]))[0]))):
        # 同じ話の中で十分重なる区間は一つの使用箇所にまとめる。同じ出現を二重に数えない。
        merged: list[list] = []
        for label, start, end in sorted(set(members)):
            for span in merged:
                if span[0] != label:
                    continue
                if overlap_share((span[1], span[2]), (start, end)) >= recipe.occurrence_overlap_share:
                    span[1] = min(span[1], start)
                    span[2] = max(span[2], end)
                    break
            else:
                merged.append([label, start, end])
        occurrences = [{"source_id": label, "start_seconds": start, "end_seconds": end,
                        "seconds": round(end - start, 6)}
                       for label, start, end in sorted(merged)]
        values = sorted(explained_of.get(root, []))
        middle = values[len(values) // 2] if values else 0.0
        is_released = any(covered_by((item["start_seconds"], item["end_seconds"]),
                                     released.get(item["source_id"], ()), recipe.released_overlap_share)
                          for item in occurrences)
        groups.append({
            "recording_key": f"series-recurrence-{position:03d}",
            "occurrence_count": len(occurrences),
            "episode_count": len({item["source_id"] for item in occurrences}),
            "occurrences": occurrences,
            "explained_values": values,
            "explained_median": round(middle, 6),
            "released": is_released,
            "reuse_class": (REUSE_IDENTICAL if middle >= recipe.identical_explained
                            else REUSE_OTHER_CONTEXT),
            "known_state": KNOWN_RELEASED if is_released else "unresolved",
        })

    unreleased = [group for group in groups if not group["released"]]
    return {
        "recipe_id": recipe.recipe_id,
        "recipe": recipe.to_dict(),
        "group_count": len(groups),
        "released_count": sum(1 for group in groups if group["released"]),
        "unreleased_count": len(unreleased),
        "unreleased_other_context": sum(
            1 for group in unreleased if group["reuse_class"] == REUSE_OTHER_CONTEXT),
        "unreleased_identical_reuse": sum(
            1 for group in unreleased if group["reuse_class"] == REUSE_IDENTICAL),
        "groups": groups,
        "claim_boundary": [
            "同じ録音が鳴っていることまでで、欲しい劇伴かどうかは判定しない",
            "区間ごと同一の使い回しは別クラスであり、劇伴として数えない",
            "まとまりを作るだけで使用箇所を統合・削除しない",
        ],
    }
