"""公開するcueごとに、どの表現（Recipe）で保存するかを選ぶ。重い処理はその表現が要る区間にだけ掛ける。

`cue_semantics` が公開してよいと決めた区間（``WANTED``）について、次のいずれかを選ぶ。

- ``RAW``（``raw_original_mixture``）：原本mixの区間をそのまま使う。原本のsample rate・channel構成・音色・
  残響・微細な成分をそのまま保つ。分離器の出力と原本の差（分離器が取り除いた成分）が区間のどこでも小さく、
  原本に台詞が聞こえない時だけ選ぶ。取り除く物が無いのに分離器を通すと、変質（44.1kHz化、分離の揺らぎ）だけが残る。
- ``PROCESSED``（``sfc_music_refined``）：現在の既知良好の表現。SFCの音楽の出力に、条件付きの2回目と歌唱回収を
  掛けたもの。台詞・効果音が重なる区間。2回目と歌唱回収は、この表現を選んだ区間の保留候補にだけ掛ける。
- ``UNCERTAIN``：安い観測だけでは原音で足りると言い切れない区間。速さのために欲しい音を賭けず、
  既知良好の ``PROCESSED`` へ戻す（fallbackであることを記録する）。

公開しない区間（``cue_semantics`` のNON_BGM・主題歌・既発売・UNCERTAIN、候補の外）は ``SKIP`` で、
追加の重い処理を掛けない（保留候補は1回目の音楽の出力と原本の区間のまま残る）。

表現は ``REPRESENTATIONS`` に名前で登録する。今の判定が選べるのは二つだが、区間ごとの観測から表現を選ぶ
形なので、4分類の器楽・stem別のspecialist・使用箇所の再構成等を、同じ判定の選択肢として足せる。
判定は正解・参照・試聴を使わない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD = "cue-representation-residual-routing-v1"

RAW = "raw_original_mixture"
PROCESSED = "sfc_music_refined"
# 経路の状態（表現とは別。UNCERTAINは既知良好の表現へ戻したことを示す）
STATE_SKIP, STATE_RAW, STATE_PROCESSED, STATE_UNCERTAIN = "SKIP", "RAW", "PROCESSED", "UNCERTAIN"
REPRESENTATIONS = {
    RAW: {"description": "原本mixの区間（no-op）", "sample_rate": "source_native", "heavy_processing": []},
    PROCESSED: {"description": "SFCの音楽の出力＋条件付き2回目＋歌唱回収（既知良好）", "sample_rate": 44100,
                "heavy_processing": ["sfc_second_pass", "four_stem_vocal_recovery"]},
}
KNOWN_GOOD = PROCESSED
SPEECH_CLASS = 0


class CueRepresentationError(ValueError):
    """表現の判定の宣言か材料が契約に合わない。"""


@dataclass(frozen=True)
class RepresentationRecipe:
    """表現を選ぶ宣言。どれかが変われば識別子が変わる。"""

    raw_maximum_residual_db: float     # 原音にする：どの1秒窓でも（原本−分離器の音楽）が音楽よりこれ以下
    raw_maximum_speech_probability: float  # 原音にする：原本のタグのSpeech確率がどの窓でもこれ以下
    processed_minimum_residual_db: float   # 処理済みと言い切る：これを超える窓がある（取り除く物がはっきりある）
    processed_minimum_speech_probability: float  # 処理済みと言い切る：Speech確率がこれを超える窓がある
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD
    partial_window_policy: str = "include_final_partial_v1"

    def validate(self) -> None:
        if self.partial_window_policy != "include_final_partial_v1":
            raise CueRepresentationError("Unsupported partial-window policy")
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise CueRepresentationError("未対応のschemaです")
        for name, low, high in (("raw_maximum_residual_db", -120.0, 0.0),
                                ("processed_minimum_residual_db", -120.0, 20.0),
                                ("raw_maximum_speech_probability", 0.0, 1.0),
                                ("processed_minimum_speech_probability", 0.0, 1.0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not low <= float(value) <= high:
                raise CueRepresentationError(f"{name}は{low}以上{high}以下である必要があります")
        if self.processed_minimum_residual_db < self.raw_maximum_residual_db:
            raise CueRepresentationError("processed_minimum_residual_dbはraw_maximum_residual_db以上である必要があります")
        if self.processed_minimum_speech_probability < self.raw_maximum_speech_probability:
            raise CueRepresentationError("processed_minimum_speech_probabilityはraw_maximum_speech_probability以上です")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-cue-representation-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class RoutingEvidence:
    """1秒格子（窓 i は原本の [i, i+1) 秒）の観測。長さは揃える。"""

    residual_db: np.ndarray        # (原本mix − SFCの音楽の出力) のエネルギーの、SFCの音楽の出力に対するdB
    mix_speech_probability: np.ndarray  # 原本mixのタグのSpeech確率

    @classmethod
    def from_arrays(cls, residual_db, mix_probabilities) -> "RoutingEvidence":
        residual = np.asarray(residual_db, dtype=np.float64)
        probabilities = np.asarray(mix_probabilities, dtype=np.float32)
        count = min(len(residual), len(probabilities))
        if count == 0 or probabilities.ndim != 2:
            raise CueRepresentationError("表現の判定の観測が空です")
        if not (np.isfinite(residual[:count]).all() and np.isfinite(probabilities[:count]).all()):
            raise CueRepresentationError("表現の判定の観測に有限でない値があります")
        return cls(residual_db=residual[:count], mix_speech_probability=probabilities[:count, SPEECH_CLASS])


def residual_levels(mixture: str, music: str, *, window_seconds: float = 1.0) -> np.ndarray:
    """原本mix（解析用の複製）と分離器の音楽の出力を1回だけ読み、1秒窓ごとの（mix−music）の音楽に対するdB。"""
    import soundfile as sf

    with sf.SoundFile(mixture) as mix_stream, sf.SoundFile(music) as music_stream:
        if mix_stream.samplerate != music_stream.samplerate:
            raise CueRepresentationError("原本の解析用複製と音楽の出力の時間軸が一致しません")
        size = int(round(window_seconds * mix_stream.samplerate))
        levels = []
        while True:
            mix = mix_stream.read(size, dtype="float64", always_2d=True)
            music = music_stream.read(size, dtype="float64", always_2d=True)
            if not len(mix) and not len(music):
                break
            if mix.shape != music.shape:
                raise CueRepresentationError("Residual observations have different source timelines")
            residual = float(np.mean((mix - music) ** 2))
            levels.append(10.0 * np.log10(max(residual, 1e-20) / max(float(np.mean(music ** 2)), 1e-20)))
    return np.asarray(levels)


def choose(start_seconds: float, end_seconds: float, evidence: RoutingEvidence,
           recipe: RepresentationRecipe) -> dict:
    """公開する区間の表現を選ぶ。経路の状態・表現・理由・特徴量を返す。"""
    recipe.validate()
    first, last = int(start_seconds), int(np.ceil(end_seconds))
    residual = evidence.residual_db[first:last]
    speech = evidence.mix_speech_probability[first:last]
    if residual.size == 0:
        raise CueRepresentationError("表現を選ぶ区間の観測がありません")
    features = {"residual_db_max": round(float(residual.max()), 2),
                "residual_db_p90": round(float(np.percentile(residual, 90)), 2),
                "residual_db_median": round(float(np.median(residual)), 2),
                "mix_speech_max": round(float(speech.max()), 3),
                "mix_speech_p90": round(float(np.percentile(speech, 90)), 3)}
    reasons = []
    if residual.max() > recipe.processed_minimum_residual_db:
        reasons.append("separator_removes_material")
    if speech.max() > recipe.processed_minimum_speech_probability:
        reasons.append("speech_in_mixture")
    if reasons:
        return {"state": STATE_PROCESSED, "representation": PROCESSED, "fallback": False,
                "reasons": reasons, "features": features}
    if residual.max() <= recipe.raw_maximum_residual_db and speech.max() <= recipe.raw_maximum_speech_probability:
        return {"state": STATE_RAW, "representation": RAW, "fallback": False,
                "reasons": ["mixture_equals_music"], "features": features}
    return {"state": STATE_UNCERTAIN, "representation": KNOWN_GOOD, "fallback": True,
            "reasons": ["raw_not_established"], "features": features}


def default_recipe() -> RepresentationRecipe:
    """最初の宣言。corpus-a の5話の既発売曲の区間（参照を原本へ当てはめた近似正解、28区間）で置いた
    （`.state/validation/bgm-work-avoidance-r1/raw-study.json`）。

    - 取り除く成分が全窓で-20dB以下・Speech 0.2以下の区間は、原音と処理済みの近似正解への近さの差が0.03〜0.09dB
      （分離しても良くならない）。原音なら原本のsample rate・stereo・音色をそのまま保てる。
    - 取り除く成分が-10dBを超える窓がある区間（台詞・効果音が重なる大半の劇伴）は、処理済みが原音より
      数dB〜15dB良い。
    - その間（主題歌の中の-13〜-15dBの1窓等）は処理済みの利得が+0.02〜0.4dBと小さいが、何かを取り除いている
      ので言い切らず、既知良好の処理済みへ戻す。
    """
    return RepresentationRecipe(raw_maximum_residual_db=-20.0, raw_maximum_speech_probability=0.2,
                                processed_minimum_residual_db=-10.0, processed_minimum_speech_probability=0.5)


def certify_raw(route, source, first_music, rate, *, allowed, native_grid):
    """A coarse no-op nomination cannot bypass sample-level source checks.

    Losing this permission retains separated music. It supplies no authority
    to expand later speech repair, discard a quiet cue, or restore contaminants.
    """
    from lss.separation import selective
    if route["representation"] != RAW:
        return route
    plan = selective.plan(source, first_music, rate, allowed=allowed, witness=first_music)
    covered = sum(b-a for a,b in plan["original_regions"]+plan["silent_regions"])
    admitted = bool(native_grid and covered == len(source))
    check = {"state":"original_admitted" if admitted else "retain_separated_music",
             "native_grid":bool(native_grid),"covered_frames":covered,"frames":len(source),
             "policy":plan["policy"],"additional_processing_authority":"none"}
    if admitted:
        return {**route,"original_check":check}
    return {**route,"state":STATE_UNCERTAIN,"representation":PROCESSED,"fallback":True,
            "reasons":[*route["reasons"],"original_samples_not_established"],"original_check":check}
