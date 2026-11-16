"""同じ録音の使用箇所を、重なりの少なさで順位付けし、主題歌を対象から外す。

実番組の測定（`.state/validation/lv-bgm-region-routing-r1/`）で次が分かっている。

- 欲しい劇伴は、この対象領域ではほぼ常に台詞・効果音と重なる。原本をそのまま使える
  ほど重なりの無い区間は、実質OP/EDだけだった。
- 同じ録音でも使用箇所ごとに重なり具合が大きく違う。corpus-c 第1話の
  `孤独・恐怖・トラウマ` は6回登場し、説明率は0.697から0.261まで開いた。

そこでこのStageは二つを決める。

1. **主題歌の除外**：OP/EDの録音と確実に一致した使用箇所は、欲しい劇伴ではないので
   対象から外す。どの録音が主題歌かは呼び出し側が渡す（corpusの置き場所に依存しない）。
2. **最良の使用箇所**：同じ録音の使用箇所のうち、重なりが最も少ないものを抽出元に選ぶ。

`waveform_explained` は既発売録音または別の使用箇所が、その区間の音をどれだけ説明
できるかで、台詞・効果音が重なるほど下がる。意味的な判断ではなく物理量なので、
人の試聴を要さない。長さだけで最良を決めない（`docs/BGM.md`「最良の使用箇所」）。

順位を付けるだけで、使用箇所を統合・削除しない。重なりの多い使用箇所も、そう記録して
保持する。単独出現は比較対象が無いだけで、それ自体は欠格ではない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable

SCHEMA_VERSION = 1
METHOD = "least-overlaid-occurrence-v1"

EVIDENCE_RELEASED = "released_reference"
EVIDENCE_MUTUAL = "mutual_occurrence"
EVIDENCE_SOURCES = frozenset({EVIDENCE_RELEASED, EVIDENCE_MUTUAL})

EXCLUDED_THEME = "theme_opening_or_ending"
HEAVILY_OVERLAID = "heavily_overlaid"
SINGLE_OCCURRENCE = "single_occurrence"
SELECTED = "least_overlaid"


class OccurrenceQualityError(ValueError):
    """順位付けの宣言、または渡された使用箇所が契約に合わない。"""


@dataclass(frozen=True)
class OccurrenceQualityRecipe:
    """重なりの少なさで使用箇所を選ぶ宣言。"""

    minimum_explained: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise OccurrenceQualityError("未対応の使用箇所順位schemaです")
        value = self.minimum_explained
        if (isinstance(value, bool) or not isinstance(value, Real) or value != value
                or not 0.0 <= float(value) <= 1.0):
            raise OccurrenceQualityError("minimum_explainedは0以上1以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-occurrence-quality-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OccurrenceQualityError(f"{context}が未指定です")
    return value.strip()


def _seconds(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise OccurrenceQualityError(f"{context}が不正です: {value!r}")
    return float(value)


def _explained(value: Any, context: str) -> float:
    number = _seconds(value, context)
    if not 0.0 <= number <= 1.0:
        raise OccurrenceQualityError(f"{context}は0以上1以下である必要があります: {number}")
    return number


def _checked(occurrence: Any) -> dict[str, Any]:
    if not isinstance(occurrence, dict):
        raise OccurrenceQualityError("使用箇所はオブジェクトである必要があります")
    start = _seconds(occurrence.get("start_seconds"), "start_seconds")
    end = _seconds(occurrence.get("end_seconds"), "end_seconds")
    if end <= start:
        raise OccurrenceQualityError("使用箇所の区間が空です")
    evidence = occurrence.get("evidence_source", EVIDENCE_RELEASED)
    if evidence not in EVIDENCE_SOURCES:
        raise OccurrenceQualityError(f"evidence_sourceが不正です: {evidence!r}")
    return {
        "occurrence_id": _text(occurrence.get("occurrence_id"), "occurrence_id"),
        "recording_key": _text(occurrence.get("recording_key"), "recording_key"),
        "source_id": _text(occurrence.get("source_id"), "source_id"),
        "start_seconds": start,
        "end_seconds": end,
        "seconds": round(end - start, 6),
        "waveform_explained": _explained(occurrence.get("waveform_explained"), "waveform_explained"),
        "evidence_source": evidence,
    }


def mutual_explained_scores(
    pairwise: Any, occurrence_ids: Iterable[str],
) -> dict[str, float]:
    """使用箇所どうしの説明率から、各回の「他の回にどれだけ説明されるか」を出す。

    参照録音が無い曲でも重なりの少なさを測るための入口。`pairwise` は
    `{"対象の回|説明する回": 説明率}`。対象の回ごとに、他の全ての回からの説明率の
    平均を返す。実番組の既発売cue 7録音のうち6録音で、既発売録音を使った順位と
    同じ回を最良に選ぶことを確かめている（`lv-bgm-region-routing-r1/occurrence-mutual`）。
    """
    ids = [_text(value, "occurrence_id") for value in occurrence_ids]
    if len(set(ids)) != len(ids):
        raise OccurrenceQualityError("occurrence_idが重複しています")
    if len(ids) < 2:
        raise OccurrenceQualityError("相互説明には2つ以上の使用箇所が必要です")
    if not isinstance(pairwise, dict):
        raise OccurrenceQualityError("pairwiseはオブジェクトである必要があります")
    scores = {}
    for target in ids:
        values = []
        for other in ids:
            if other == target:
                continue
            key = f"{target}|{other}"
            if key not in pairwise:
                raise OccurrenceQualityError(f"相互説明が足りません: {key}")
            values.append(_explained(pairwise[key], key))
        scores[target] = round(sum(values) / len(values), 6)
    return scores


def select_best_occurrences(
    occurrences: Iterable[Any], recipe: OccurrenceQualityRecipe, *,
    theme_recording_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """録音ごとに使用箇所を重なりの少ない順へ並べ、抽出元を一つ選ぶ。

    `theme_recording_keys` に挙げた録音（OP/ED）は対象から外す。確実な一致だけを
    渡すこと。順位付けは使用箇所を統合・削除しない。
    """
    recipe.validate()
    themes = {key for key in theme_recording_keys if isinstance(key, str) and key.strip()}
    grouped: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for occurrence in occurrences:
        item = _checked(occurrence)
        if item["occurrence_id"] in seen:
            raise OccurrenceQualityError(f"occurrence_idが重複しています: {item['occurrence_id']}")
        seen.add(item["occurrence_id"])
        grouped.setdefault(item["recording_key"], []).append(item)

    groups = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda item: (-item["waveform_explained"], -item["seconds"],
                              item["start_seconds"], item["occurrence_id"]),
        )
        is_theme = key in themes
        explained = [item["waveform_explained"] for item in items]
        ranked = []
        for position, item in enumerate(items):
            reasons = []
            if is_theme:
                reasons.append(EXCLUDED_THEME)
            elif position == 0:
                reasons.append(SELECTED)
                if len(items) == 1:
                    reasons.append(SINGLE_OCCURRENCE)
            if item["waveform_explained"] < recipe.minimum_explained:
                reasons.append(HEAVILY_OVERLAID)
            ranked.append({**item, "rank": position, "reasons": reasons})
        evidence_sources = sorted({item["evidence_source"] for item in items})
        groups.append({
            "recording_key": key,
            "theme_excluded": is_theme,
            "evidence_sources": evidence_sources,
            "occurrence_count": len(items),
            "selected_occurrence_id": None if is_theme else ranked[0]["occurrence_id"],
            "selected_explained": None if is_theme else ranked[0]["waveform_explained"],
            "explained_spread": round(max(explained) - min(explained), 6),
            "occurrences": ranked,
        })

    usable = [group for group in groups if not group["theme_excluded"]]
    return {
        "recipe_id": recipe.recipe_id,
        "recipe": recipe.to_dict(),
        "recording_count": len(groups),
        "theme_excluded_count": sum(1 for group in groups if group["theme_excluded"]),
        "selected_count": len(usable),
        "selected_above_minimum": sum(
            1 for group in usable if group["selected_explained"] >= recipe.minimum_explained),
        "groups": groups,
        "claim_boundary": [
            "重なりの少なさの順位であり、欲しい劇伴かどうかの判断ではない",
            "主題歌の除外は確実な録音一致にだけ適用する",
            "順位付けは使用箇所を統合・削除しない",
        ],
    }
