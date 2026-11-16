"""シリーズの原本mixから未発売cueの候補を作る経路を、一つのRecipeとして宣言する。

これまで別々のStageとscriptに散っていた手順を、製品が実行できる一つの宣言にまとめる。

0. 話どうしの照合に加え、`include_within_episode` を宣言すると**同じ話の中の繰り返し**も拾う。
   1話しか無い入力でも、同じ録音が2回鳴っていれば候補になる。
1. `series_reuse`：組ごとの一致を録音単位のまとまりへ直し、既発売と区間丸ごと流用を分ける。
2. `occurrence_quality`：録音ごとに最も重なりの少ない使用箇所を選ぶ。主題歌は外す。
3. `cue_selection`：実番組で測った誤差ぶんの余裕を付けて、切り出す区間を決める。
4. `content_guard`：切り出した中身を測れる場合に、繰り返す台詞・既製の効果音を落とす。

このモジュールは音声を読まない。組ごとの一致（`identity.shared_recordings` の結果）と、
既発売録音との一致区間を受け取って、**どこを切り出すか**までを返す。切り出しと測定は
呼び出し側が行う。4番目の関門は分離器の出力を要するので、測定を渡せたときだけ適用し、
適用したかどうかを結果へ残す。

参照録音が無くても1〜3は動く。既発売録音が手元にあれば、その区間を差し引いて
「未発売」の精度が上がる。どちらの場合も、結果は**非公開の候補**であって
正式なBGM成果物ではない。意味役割（欲しい劇伴か）は判定しない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Mapping

from lss.bgm import (content_guard, cue_selection, occurrence_quality, occurrence_stack,
                     series_reuse)

SCHEMA_VERSION = 1
METHOD = "series-unreleased-cue-route-v1"

GUARD_APPLIED = "applied"
GUARD_NOT_MEASURED = "not_measured"
# 測定が無いとき、経路が自分で計算した印だけで選ぶ。関門より弱い根拠なので別の名前にする。
GUARD_MARKS_ONLY = "marks_only"
REJECT_UNTRUSTED_ESTIMATE = "untrusted_music_estimate"

# 関門を何に対してかけるか。原本の区間そのままか、重ね合わせで推定した音楽か。
GUARD_TARGET_OCCURRENCE = "occurrence_span"
GUARD_TARGET_ESTIMATE = "music_estimate"
GUARD_TARGETS = frozenset({GUARD_TARGET_OCCURRENCE, GUARD_TARGET_ESTIMATE})

# **同じ波形がそのまま繰り返されている**ことの印。再演の説明率も、重ね合わせが残す割合も
# 高く出る。毎回ちがう台詞の下にある劇伴はどちらも低い。
#
# 当てはめていない材料で試したところ、**これは「効果音」の印ではない**
# （`.state/validation/lv-bgm-stock-effect-holdout-r1/`）。主題歌のように毎回同じ音源が
# 流れる**音楽**にも同じように鳴る（既知の音楽11群のうち5群で鳴った）。
# 候補31件で効果音と一致して見えたのは、主題歌が既発売の差し引きで先に消えていたため。
# したがって**この印だけで候補を落とさない**。
MARK_IDENTICAL_REPEAT = "likely_identical_repeat"
# 旧名。記録の互換のために残す。
MARK_STOCK_EFFECT = MARK_IDENTICAL_REPEAT


class SeriesCueError(ValueError):
    """経路の宣言、または渡された材料が契約に合わない。"""


@dataclass(frozen=True)
class SeriesCueRecipe:
    """未発売cueを取り出す経路の宣言。段ごとのRecipeを持つ。"""

    reuse: series_reuse.SeriesReuseRecipe
    quality: occurrence_quality.OccurrenceQualityRecipe
    selection: cue_selection.CueSelectionRecipe
    guard: content_guard.ContentGuardRecipe
    guard_target: str = GUARD_TARGET_OCCURRENCE
    include_within_episode: bool = False
    stock_effect_explained: float = 1.0
    stock_effect_energy_share: float = 1.0
    select_without_measurement: bool = False
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise SeriesCueError("未対応の経路schemaです")
        if self.guard_target not in GUARD_TARGETS:
            raise SeriesCueError(f"未対応の関門の対象です: {self.guard_target!r}")
        for name in ("include_within_episode", "select_without_measurement"):
            if not isinstance(getattr(self, name), bool):
                raise SeriesCueError(f"{name}は真偽値である必要があります")
        for name in ("stock_effect_explained", "stock_effect_energy_share"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or value != value
                    or not 0.0 < float(value) <= 1.0):
                raise SeriesCueError(f"{name}は0より大きく1以下である必要があります")
        for name in ("reuse", "quality", "selection", "guard"):
            stage = getattr(self, name)
            if not hasattr(stage, "validate") or not hasattr(stage, "recipe_id"):
                raise SeriesCueError(f"{name}に段のRecipeが必要です")
            stage.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stage_ids(self) -> dict[str, str]:
        self.validate()
        return {name: getattr(self, name).recipe_id
                for name in ("reuse", "quality", "selection", "guard")}

    @property
    def recipe_id(self) -> str:
        body = json.dumps({"method": self.method, "schema_version": self.schema_version,
                           "guard_target": self.guard_target,
                           "include_within_episode": self.include_within_episode,
                           "stock_effect_explained": self.stock_effect_explained,
                           "stock_effect_energy_share": self.stock_effect_energy_share,
                           "select_without_measurement": self.select_without_measurement,
                           "stages": self.stage_ids},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-series-cues-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def marks_for(explained: Any, energy_share: Any,
              recipe: SeriesCueRecipe) -> list[str]:
    """分離器を使わずに付けられる印。今は「同じ波形がそのまま繰り返されている」の一つだけ。

    同じ波形の繰り返しは、再演の説明率も、重ね合わせが残す割合も高く出る。
    毎回ちがう台詞の下にある劇伴はどちらも低い。

    実番組13話の候補31件では、分離器で効果音優勢と判定された5件が
    「説明率0.585以上かつ残した割合0.384以上」に全て入り、残る26件は1件も入らなかった。
    **しかしこれは効果音の印ではない。** 当てはめていない材料（既発売録音に一致する＝
    独立に音楽と分かっている11群）で試すと、5群で鳴った。主題歌のように毎回同じ音源が
    流れる音楽にも同じように鳴る。候補31件で一致して見えたのは、主題歌が既発売の
    差し引きで先に消えていたため。

    **印であって、落とす判断ではない。** この印だけで候補を落とさない。
    """
    recipe.validate()
    if explained is None or energy_share is None:
        return []
    if (float(explained) >= recipe.stock_effect_explained
            and float(energy_share) >= recipe.stock_effect_energy_share):
        return [MARK_STOCK_EFFECT]
    return []


def occurrence_id(source_id: str, start_seconds: float) -> str:
    """使用箇所の識別子。段をまたいで同じ綴りを使う。"""
    return f"{source_id}@{float(start_seconds):.3f}"


def plan_cues(
    pairs: Iterable[Any], recipe: SeriesCueRecipe, *,
    released_spans: Mapping[str, Iterable[Any]] | None = None,
    source_seconds: Mapping[str, float] | None = None,
    theme_recording_keys: Iterable[str] = (),
    mutual_explained: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """組ごとの一致から、切り出す区間までを決める。音声は読まない。

    `mutual_explained` は使用箇所ごとの「他の回にどれだけ説明されるか」。渡せば
    重なりの少ない回を回ごとに選べる。渡さない場合はまとまりの説明率で代用し、
    どちらを使ったかを結果へ残す。
    """
    recipe.validate()
    grouped = series_reuse.group_recurrence(pairs, recipe.reuse, released_spans=released_spans)
    scores = {key: float(value) for key, value in (mutual_explained or {}).items()}

    occurrences = []
    for group in grouped["groups"]:
        if group["released"] or group["reuse_class"] != series_reuse.REUSE_OTHER_CONTEXT:
            continue
        for item in group["occurrences"]:
            identifier = occurrence_id(item["source_id"], item["start_seconds"])
            occurrences.append({
                "occurrence_id": identifier,
                "recording_key": group["recording_key"],
                "source_id": item["source_id"],
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
                "waveform_explained": scores.get(identifier, group["explained_median"]),
                "evidence_source": occurrence_quality.EVIDENCE_MUTUAL,
            })

    ranked = occurrence_quality.select_best_occurrences(
        occurrences, recipe.quality, theme_recording_keys=theme_recording_keys)
    selected = [group["selected_occurrence_id"] for group in ranked["groups"]
                if group["selected_occurrence_id"] is not None]
    # 前段が外した主題歌を、最長の使用箇所として拾い直さない。
    excluded = [group["recording_key"] for group in ranked["groups"]
                if group["theme_excluded"]]
    spans = cue_selection.select_extraction_spans(
        grouped, recipe.selection, selected_occurrence_ids=selected,
        excluded_recording_keys=excluded, source_seconds=source_seconds)

    return {
        "recipe_id": recipe.recipe_id,
        "stage_ids": recipe.stage_ids,
        "occurrence_ranking_basis": "mutual_occurrence" if scores else "group_explained_median",
        "grouped": grouped,
        "occurrence_quality": ranked,
        "selection": spans,
        "spans": spans["selected"],
        "span_count": len(spans["selected"]),
        "claim_boundary": [
            "非公開の候補であり、正式なBGM成果物ではない",
            "同じ録音が繰り返されていることまでで、欲しい劇伴かどうかは判定しない",
            "一度しか鳴らない曲はこの経路に現れない",
            "「未発売」は手元の参照に無いという意味であり、作品をまたいで件数を比べない",
        ],
    }


def _measurement_key(record: Mapping[str, Any]) -> str:
    for key in ("recording_key", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise SeriesCueError("測定にrecording_keyが必要です")


def _measurement_target(record: Mapping[str, Any], fallback: str) -> str:
    value = record.get("target")
    if value is None:
        return fallback
    if value not in GUARD_TARGETS:
        raise SeriesCueError(f"未対応の測定の対象です: {value!r}")
    return value


def apply_content_guard(
    plan: Mapping[str, Any], recipe: SeriesCueRecipe,
    measurements: Iterable[Any] | None = None, *,
    untrusted_recording_keys: Iterable[str] = (),
    marked_recording_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """切り出した中身の測定があれば、性質の分かった偽陽性を落とす。

    測定を渡さない場合は関門を適用せず、全区間を「未測定」として返す。
    落とさなかったことを「音楽だと確かめた」と読み替えないための区別。

    測定が何を測ったものか（原本の区間か、推定した音楽か）は各測定の `target` が名乗り、
    既定のかけどころは宣言の `guard_target` が持つ。

    `untrusted_recording_keys` に挙げた録音は、推定が信用できない（元mixをほとんど
    残していない）ものとして、**原本の区間の測定で判定する**。実番組の既発売cueでは、
    残した割合が下限に届かない推定は原本の区間より録音から遠かった
    （`.state/validation/lv-bgm-released-stack-r1/disposition.json`）。
    """
    recipe.validate()
    if not isinstance(plan, Mapping) or not isinstance(plan.get("spans"), list):
        raise SeriesCueError("plan_cuesの結果が必要です")
    spans = plan["spans"]
    untrusted = {value for value in untrusted_recording_keys if isinstance(value, str)}
    marked = {value for value in marked_recording_keys if isinstance(value, str)}
    if measurements is None and recipe.select_without_measurement:
        # 分離器の測定が無くても、経路が自分で計算した印だけで選ぶ。
        reasons = {}
        for span in spans:
            key = span["recording_key"]
            if key in marked:
                reasons[key] = MARK_STOCK_EFFECT
            elif key in untrusted:
                reasons[key] = REJECT_UNTRUSTED_ESTIMATE
        held = [span for span in spans if span["recording_key"] not in reasons]
        rejected = [{**span, "reason": reasons[span["recording_key"]]}
                    for span in spans if span["recording_key"] in reasons]
        return {"guard_state": GUARD_MARKS_ONLY, "guard": None,
                "guard_target": recipe.guard_target,
                "marked_recording_keys": sorted(marked & {s["recording_key"] for s in spans}),
                "untrusted_recording_keys": sorted(untrusted & {s["recording_key"] for s in spans}),
                "held": held, "rejected": rejected, "held_count": len(held),
                "rejected_count": len(rejected),
                "claim_boundary": [
                    "分離器の測定を使っていない。経路が自分で計算した印だけで選んでいる",
                    "印は「既製の効果音らしい」であって、欲しい劇伴かどうかの判定ではない",
                    "推定が信用できない候補は、判定できないものとして落としている",
                ]}
    if measurements is None:
        return {"guard_state": GUARD_NOT_MEASURED, "guard": None,
                "guard_target": recipe.guard_target,
                "held": list(spans), "rejected": [], "held_count": len(spans),
                "rejected_count": 0,
                "claim_boundary": ["関門を適用していない。音楽が優勢かどうかは確かめていない"]}

    by_target: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in measurements:
        if not isinstance(record, Mapping):
            raise SeriesCueError("測定はオブジェクトである必要があります")
        # recording_keyはシリーズの中でだけ一意。複数シリーズの測定を混ぜて渡せない。
        slot = (_measurement_key(record), _measurement_target(record, recipe.guard_target))
        if slot in by_target:
            raise SeriesCueError(f"測定が重複しています: {slot[0]} / {slot[1]}")
        by_target[slot] = record

    chosen, targets_used, missing = [], {}, []
    for span in spans:
        key = span["recording_key"]
        target = GUARD_TARGET_OCCURRENCE if key in untrusted else recipe.guard_target
        record = by_target.get((key, target))
        if record is None:
            missing.append(f"{key} / {target}")
            continue
        targets_used[key] = target
        chosen.append({**record, "name": key})
    if missing:
        raise SeriesCueError(f"測定の無い区間があります: {sorted(missing)}")

    screened = content_guard.screen_candidates(chosen, recipe.guard)
    rejected_of = {item["name"]: item["reason"] for item in screened["rejected"]}
    held = [span for span in spans if span["recording_key"] not in rejected_of]
    rejected = [{**span, "reason": rejected_of[span["recording_key"]],
                 "guard_target": targets_used[span["recording_key"]]}
                for span in spans if span["recording_key"] in rejected_of]
    return {
        "guard_state": GUARD_APPLIED,
        "guard_target": recipe.guard_target,
        "guard_targets_used": targets_used,
        "untrusted_recording_keys": sorted(untrusted & {span["recording_key"] for span in spans}),
        "guard": screened,
        "held": held,
        "rejected": rejected,
        "held_count": len(held),
        "rejected_count": len(rejected),
        "claim_boundary": [
            "音楽が台詞・効果音を上回っていることまでで、欲しい劇伴かどうかは判定しない",
            "取り分は却下済み分離器の出力であり、正解音源ではない",
            "推定が信用できない録音は、原本の区間の測定で判定している",
        ],
    }


# 実番組の測定から決めた既定。数値の出どころは
# `.state/validation/lv-bgm-series-unreleased-r1/disposition.json`。
DEFAULT_RECIPE = SeriesCueRecipe(
    # 説明率0.95以上は区間ごと同一の使い回し。実測で二群の間は0.458と0.976で空いている。
    reuse=series_reuse.SeriesReuseRecipe(
        identical_explained=0.95, released_overlap_share=0.5, occurrence_overlap_share=0.5),
    # 重なりが大きすぎる回は抽出元として印を付ける。順位付け自体は全ての回に付ける。
    quality=occurrence_quality.OccurrenceQualityRecipe(minimum_explained=0.5),
    # 範囲の誤差のp90（開始0.96秒・終了6.30秒）から取る。左右非対称。
    selection=cue_selection.CueSelectionRecipe(
        start_pad_seconds=1.0, end_pad_seconds=6.5, minimum_seconds=4.0),
    # 既発売録音に一致する対照8件は両方の余裕が正（最小+7.2dB／+1.6dB）。0dBで足りる。
    guard=content_guard.ContentGuardRecipe(
        music_over_speech_db=0.0, music_over_effects_db=0.0),
    # 関門は原本の区間ではなく**推定した音楽**にかける。実番組18件で、原本の区間に
    # かけると台詞優勢3件（＝台詞の下の劇伴）まで落ちるが、推定にかけると
    # 本当に繰り返される既製の効果音1件だけが落ちる。
    guard_target=GUARD_TARGET_ESTIMATE,
    # 同じ話の中の繰り返しも拾う。実番組で corpus-c の3話が候補2件→5件になり、
    # **1話だけの入力でも候補が出る**ようになる。corpus-a・corpus-d では増減なし。
    # 費用は照合が5割増し（corpus-c の3話で140秒→209秒）。
    include_within_episode=True,
    # 実番組13話・候補31件で、効果音優勢と判定された5件を全て捕らえ、残る26件を
    # 1件も捕らえない帯の真ん中（効果音側の下限は0.585/0.384、他候補の上は0.470/0.414と
    # 0.608/0.271）。ただし当てはめていない材料では音楽にも鳴るので、印に留める。
    stock_effect_explained=0.53,
    stock_effect_energy_share=0.33,
    # 印だけで選ぶのは**まだ安全と示せていない**。当てはめていない材料では、
    # 主題歌のような『毎回同じ音源の音楽』にも印が鳴る（11群中5群）。
    # 候補31件で分離器と一致したのは、主題歌が既発売の差し引きで先に消えていた場合の話。
    # 差し引ける参照が無い作品では、本物の音楽を落としかねない。
    select_without_measurement=False,
)


# 制御corpus2つ（flat / ducked）での測定から決めた、音楽推定の既定
# （`.state/validation/lv-bgm-occurrence-stack-r1/report.json`）。
# 同時当てはめが他の束ね方を大きく上回る（中央値 +5.79dB に対し次点 +2.58dB）。
# ブロックは短いほど良く（0.1秒で +8.74dB、1.0秒で +2.84dB）、tap数はほぼ効かないので
# 最も軽いものを取る（9/17/33/65 で +8.74/+8.63/+8.36/+8.13dB）。
# 残した割合の下限は、実番組の既発売cue 30件での答え合わせから決めた
# （`.state/validation/lv-bgm-released-stack-r1/`）。0.10以上の19件は17件で録音そのものへ
# 近づき、0.10未満の11件は3件しか近づかなかった。
DEFAULT_STACK_RECIPE = occurrence_stack.OccurrenceStackRecipe(
    fir_taps=9, block_seconds=0.1, combine=occurrence_stack.COMBINE_JOINT,
    minimum_energy_share=0.10)
