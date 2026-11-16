"""候補区間の中身を見て、繰り返す台詞・繰り返す効果音を落とす関門。

再演から取り出した候補は「同じ録音が複数回鳴っている区間」でしかない。番組の中で
繰り返されるのは劇伴だけではなく、決め台詞と既製の効果音も同じ強さで繰り返される。
このStageは候補の中身を分けて測った取り分を受け、**音楽が台詞と効果音の両方を
上回っていること**を要求する。

単独の閾値が答えを決めるのではない。再演・既発売の差し引き・録音単位のまとまり・
使用箇所の選択がすでに候補を作っており、この関門はそのうち性質の分かった
2種類の偽陽性だけを落とす。

測った実番組（4作品11話・候補18件）での値
（`.state/validation/lv-bgm-series-unreleased-r1/candidates-content/analysis.json`）:

- 既発売録音に一致する対照8件（確実に音楽）は、music−speechが最小+7.2dB、
  music−effectsが最小+1.6dB。どちらも全件が正。
- 候補18件のうち3件はmusic−speechが負（−3.1 / −3.8 / −8.9dB）、
  別の3件はmusic−effectsが負（−7.0 / −1.4 / −0.5dB）。合わせて6件。
- したがって0dBの関門は、確実な音楽を1件も落とさずに候補6件を落とす。

取り分は却下済み分離器をanalyzerとして使って得る。最終rendererとして落ちたことは
部品としての抹消ではない（`AGENTS.md`）。この値は正解音源ではないので、関門は
「音楽が優勢であること」までしか主張しない。欲しい劇伴かどうかは判定しない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
METHOD = "candidate-content-dominance-guard-v1"

ANALYZER_ROLE = "rejected_renderer_kept_as_analyzer"

REJECT_SPEECH = "speech_over_music"
REJECT_EFFECTS = "effects_over_music"
KEEP = "music_over_speech_and_effects"


class ContentGuardError(ValueError):
    """関門の宣言、または渡された測定が契約に合わない。"""


@dataclass(frozen=True)
class ContentGuardRecipe:
    """音楽が台詞・効果音をどれだけ上回っていれば通すかの宣言。"""

    music_over_speech_db: float
    music_over_effects_db: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise ContentGuardError("未対応の関門schemaです")
        for name in ("music_over_speech_db", "music_over_effects_db"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or value != value:
                raise ContentGuardError(f"{name}は有限な数値である必要があります")
            if abs(float(value)) > 120.0:
                raise ContentGuardError(f"{name}が測定範囲の外です: {value}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-content-guard-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _level(record: Mapping[str, Any], stem: str) -> float:
    value = record.get(f"{stem}_db")
    if isinstance(value, bool) or not isinstance(value, Real) or value != value:
        raise ContentGuardError(f"{stem}_dbが不正です: {value!r}")
    return float(value)


def _name(record: Mapping[str, Any]) -> str:
    for key in ("name", "recording_key", "occurrence_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ContentGuardError("候補の名前が未指定です")


def screen_candidates(
    measurements: Iterable[Any], recipe: ContentGuardRecipe,
) -> dict[str, Any]:
    """測った取り分から、音楽が優勢な候補だけを残す。

    `measurements` の各要素は `speech_db` / `music_db` / `effects_db` を持つ。
    どちらの余裕も満たさない候補は、どちらで落ちたかを付けて `rejected` へ回す。
    候補を書き換えず、順序も変えない。
    """
    recipe.validate()
    kept, rejected = [], []
    for record in measurements:
        if not isinstance(record, Mapping):
            raise ContentGuardError("測定はオブジェクトである必要があります")
        name = _name(record)
        music = _level(record, "music")
        speech = _level(record, "speech")
        effects = _level(record, "effects")
        over_speech = music - speech
        over_effects = music - effects
        row = {
            "name": name,
            "music_minus_speech_db": round(over_speech, 3),
            "music_minus_effects_db": round(over_effects, 3),
        }
        if over_speech < recipe.music_over_speech_db:
            rejected.append({**row, "reason": REJECT_SPEECH})
        elif over_effects < recipe.music_over_effects_db:
            rejected.append({**row, "reason": REJECT_EFFECTS})
        else:
            kept.append({**row, "state": KEEP})

    return {
        "recipe_id": recipe.recipe_id,
        "recipe": recipe.to_dict(),
        "analyzer_role": ANALYZER_ROLE,
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "kept": kept,
        "rejected": rejected,
        "claim_boundary": [
            "音楽が台詞・効果音を上回っていることまでで、欲しい劇伴かどうかは判定しない",
            "取り分は却下済み分離器の出力であり、正解音源ではない",
            "落とすのは性質の分かった2種類の偽陽性だけで、候補の作成自体はしない",
        ],
    }
