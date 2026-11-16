"""再演から見つけた録音のまとまりから、切り出す区間を一つ選ぶ。

`series_reuse` が録音単位のまとまりを作り、`occurrence_quality` が重なりの少ない
使用箇所を選ぶ。このStageはその二つを受けて、**実際に保存する区間**を決める。

区間の端に足す余裕は、実番組で測った範囲の誤差から決める
（`.state/validation/lv-bgm-series-unreleased-r1/disposition.json`）。

- 開始の誤差は中央値0.157秒・p90 0.96秒。再演の境界をほぼそのまま使える。
- 終了の誤差は中央値1.296秒・p90 6.30秒。曲の終わりは台詞の開始でduckingやフェードが
  かかり共有内容が崩れるため、開始より緩い。

したがって開始と終了で別の余裕を持つ。左右対称の余裕にしない。

選ぶだけで、音声は作らない。既発売のまとまりと、区間ごと同一の使い回しは対象にしない。
意味役割（欲しい劇伴か）は判定しない。保存した区間は非公開の候補であって、
正式なBGM成果物ではない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Mapping

from lss.bgm import series_reuse

SCHEMA_VERSION = 1
METHOD = "recurrence-cue-extraction-span-v1"

SKIP_RELEASED = "released_reference_match"
SKIP_EXCLUDED = "excluded_recording"
SKIP_IDENTICAL = "identical_segment_reuse"
SKIP_NO_OCCURRENCE = "no_selected_occurrence"
SKIP_TOO_SHORT = "below_minimum_seconds"


class CueSelectionError(ValueError):
    """切り出し区間の宣言、または渡されたまとまりが契約に合わない。"""


@dataclass(frozen=True)
class CueSelectionRecipe:
    """どの区間をどれだけの余裕で切り出すかの宣言。"""

    start_pad_seconds: float
    end_pad_seconds: float
    minimum_seconds: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise CueSelectionError("未対応の切り出しschemaです")
        for name in ("start_pad_seconds", "end_pad_seconds"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or value != value
                    or float(value) < 0.0):
                raise CueSelectionError(f"{name}は0以上の有限な数値である必要があります")
        value = self.minimum_seconds
        if (isinstance(value, bool) or not isinstance(value, Real) or value != value
                or float(value) <= 0.0):
            raise CueSelectionError("minimum_secondsは正の数値である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-cue-selection-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise CueSelectionError(f"{context}が不正です: {value!r}")
    return float(value)


def select_extraction_spans(
    grouped: Mapping[str, Any], recipe: CueSelectionRecipe, *,
    selected_occurrence_ids: Iterable[str] = (),
    excluded_recording_keys: Iterable[str] = (),
    source_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """まとまりごとに、切り出す区間を一つ決める。

    `selected_occurrence_ids` は `occurrence_quality` が選んだ回の識別子
    （`"{source_id}@{start:.3f}"` の形）。渡さない場合は最長の使用箇所を使う。
    `excluded_recording_keys` に挙げた録音（主題歌など、前段で対象から外したもの）は
    区間を作らない。前段が外した録音をここで拾い直さないための明示の入口。
    `source_seconds` があれば、余裕を足した区間を素材の長さで丸める。
    """
    recipe.validate()
    if not isinstance(grouped, Mapping) or not isinstance(grouped.get("groups"), list):
        raise CueSelectionError("series_reuseのまとまりが必要です")
    chosen = {value for value in selected_occurrence_ids if isinstance(value, str)}
    excluded = {value for value in excluded_recording_keys if isinstance(value, str)}
    lengths = {key: _number(value, "source_seconds") for key, value in (source_seconds or {}).items()}

    selected, skipped = [], []
    for group in grouped["groups"]:
        key = group.get("recording_key")
        if key in excluded:
            skipped.append({"recording_key": key, "reason": SKIP_EXCLUDED})
            continue
        if group.get("released"):
            skipped.append({"recording_key": key, "reason": SKIP_RELEASED})
            continue
        if group.get("reuse_class") != series_reuse.REUSE_OTHER_CONTEXT:
            skipped.append({"recording_key": key, "reason": SKIP_IDENTICAL})
            continue
        occurrences = group.get("occurrences") or []
        if not occurrences:
            skipped.append({"recording_key": key, "reason": SKIP_NO_OCCURRENCE})
            continue
        picked = None
        for occurrence in occurrences:
            identifier = f"{occurrence['source_id']}@{_number(occurrence['start_seconds'], 'start'):.3f}"
            if identifier in chosen:
                picked = occurrence
                break
        if picked is None:
            picked = max(occurrences, key=lambda item: _number(item["end_seconds"], "end")
                         - _number(item["start_seconds"], "start"))
        start = _number(picked["start_seconds"], "start_seconds")
        end = _number(picked["end_seconds"], "end_seconds")
        if end - start < recipe.minimum_seconds:
            skipped.append({"recording_key": key, "reason": SKIP_TOO_SHORT,
                            "seconds": round(end - start, 3)})
            continue
        padded_start = max(0.0, start - recipe.start_pad_seconds)
        padded_end = end + recipe.end_pad_seconds
        limit = lengths.get(picked["source_id"])
        if limit is not None:
            padded_end = min(padded_end, limit)
        # 同じ録音が他のどの回で鳴っているか。来歴として残し、
        # 複数の回を使う処理（`occurrence_stack`）の入口にもなる。
        others = [{"source_id": item["source_id"],
                   "start_seconds": round(_number(item["start_seconds"], "start"), 3),
                   "end_seconds": round(_number(item["end_seconds"], "end"), 3)}
                  for item in occurrences if item is not picked]
        selected.append({
            "recording_key": key,
            "source_id": picked["source_id"],
            "other_occurrences": others,
            "occurrence_start_seconds": round(start, 3),
            "occurrence_end_seconds": round(end, 3),
            "start_seconds": round(padded_start, 3),
            "end_seconds": round(padded_end, 3),
            "seconds": round(padded_end - padded_start, 3),
            "occurrence_count": group.get("occurrence_count"),
            "episode_count": group.get("episode_count"),
            "explained_median": group.get("explained_median"),
            "selected_by": "occurrence_quality" if picked is not None and chosen else "longest_occurrence",
        })

    return {
        "recipe_id": recipe.recipe_id,
        "recipe": recipe.to_dict(),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected": selected,
        "skipped": skipped,
        "claim_boundary": [
            "非公開の候補区間であり、正式なBGM成果物ではない",
            "欲しい劇伴かどうかは判定していない",
            "余裕の幅は実番組で測った範囲の誤差から決めている",
        ],
    }
