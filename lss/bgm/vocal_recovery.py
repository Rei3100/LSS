"""効果音の出力へ誤配分された歌唱を、候補区間だけ4分類で見つけ、安全な時だけ劇伴の候補へ戻す。

SFC（台詞・音楽・効果音の3分類）は、短い歌唱などで同じ声を音楽と効果音の出力へ分けて入れることが
ある（制御素材で音楽0.46・効果音0.49）。SFCは左右を別々に推論するので、同じ時間の同じ声が片方の
channelでは音楽、もう片方では効果音へ行く（足す成分の左右差は+80dB以上）。効果音の出力を
丸ごと音楽へ足すと効果音も戻り、原本の全体を4分類へ置き換えると正常な候補を壊す。そこで次の3段で、
採用中の音楽（1回目か、2回目を採った候補は2回目）へ足すかを決める。正解・参照・試聴は使わない。

1. **試す価値（eligibility）**：候補区間の1回目の効果音の出力が、採用中の音楽に対して
   ``minimum_effects_db`` 以上ある時だけ4分類へ送る。これより小さい効果音からは、足す価値のある
   量（3.の ``minimum_added_over_music_db``）が出てこない。
2. **回収（recovery）**：効果音の出力の区間を4分類へ通し、``music_vocals`` を V_E とする。
   V_E に足す価値のある量がある候補だけ、採用中の音楽と原本（完成mix）の同じ区間も4分類へ通し、
   それぞれの ``music_vocals`` を V_M（候補に既にある歌唱）・V_X（原本の歌唱）とする。
   原本の4分類は置き換えには使わず、足す成分を確かめる証拠にだけ使う。100ms枠ごとに次の両方を満たす
   枠だけを足す:
   - V_E が V_M より ``gate_margin_db`` を超えて大きくない（**候補に既に同じ時間の歌唱がある**。
     前後 ``gate_dilation_seconds`` だけ広げる。1回目の振り分けが音の出だし・終わりで1枠ずれる分）;
   - 足すと候補の歌唱が V_X へ近づく（|V_X - V_M - V_E| < |V_X - V_M|）。器楽の一部を4分類が歌唱と
     見なした候補へ、同時に鳴る別の音（効果音・台詞）を足すことを防ぐ。実番組の既発売曲の区間で、
     1つ目だけでは近似正解に対して-1.1dB悪化させた採用があった。
   枠の0/1を ``gate_smoothing_seconds`` のHann窓で滑らかにし、足し始め・止めでクリックを作らない。
   challenger = 採用中の音楽 + gate·V_E。
3. **採否（adoption）**：次の全てを満たす時だけ採る。拮抗・不明は採用中の音楽のまま。
   - 足す量が意味を持つ（候補の音楽に対して ``minimum_added_over_music_db`` 以上、かつ候補に既にある
     歌唱に対して ``minimum_added_over_vocals_db`` 以上）;
   - 足す量が候補に既にある歌唱を大きく超えない（``maximum_added_over_vocals_db`` 以下。超える時は
     別の声を足している疑いがある）;
   - 4分類が効果音の中で歌唱と見なしたもののうち、候補の歌唱と同時に鳴る割合が
     ``minimum_kept_fraction`` 以上（大半が候補の歌唱と別の時間にあるなら、それは別の声の疑い）;
   - 候補の歌唱と同時に鳴る分のうち、原本の歌唱と整合する割合が ``minimum_agreeing_fraction`` 以上;
   - 足す成分と候補の歌唱が**同じ声**（足す枠の振幅スペクトルの余弦類似度の、足す量で重み付けした
     平均が ``minimum_same_voice_similarity`` 以上）。SFCが分けて入れた同じ歌唱は同じ音高・倍音を持つ
     （制御素材の改善ケースで0.64〜0.99）。実番組の既発売器楽曲の区間で近似正解を-1.1dB悪化させた
     採用は0.38で、原本の4分類もその音を歌唱と見なしていた（器楽の一部を歌唱と見なした候補に、同時に
     鳴る叫び等の声を足していた）;
   - 足した後の歌唱（V_M + 足す成分）の左右の相関が、候補に既にある歌唱より ``maximum_stereo_change``
     を超えて下がらず、左右の大きさの差が max(既にある歌唱の差, ``maximum_level_difference_db``) を
     超えて偏らない（別の位置の音を足していない）。1回目は左右を別々に推論するので、同じ歌唱が
     片方のchannelだけ効果音へ行くことがあり、足す成分だけの定位は片寄っていてよい;
   - 時間位置を ``maximum_lag_samples`` 以上ずらさない。

2回目（`lss.bgm.second_pass`）の「取り除くのが安全か」とは別の判断である。足す判断は、足す成分が
候補に既にある歌唱と同じ時間にあり、同程度の大きさで、原本の歌唱の欠けた分を埋めることを根拠にする。

実測：`.state/validation/bgm-effects-vocal-recovery-r1/`。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

METHOD = "bgm-effects-vocal-recovery-v1"


@dataclass(frozen=True)
class VocalRecoveryRecipe:
    """試す価値・回収・採否の宣言。どれかが変われば識別子が変わる。"""

    minimum_effects_db: float = -40.0
    hop_seconds: float = 3.0
    frame_seconds: float = 0.1
    gate_margin_db: float = 10.0
    gate_dilation_seconds: float = 0.1
    gate_smoothing_seconds: float = 0.05
    minimum_added_over_music_db: float = -35.0
    minimum_added_over_vocals_db: float = -20.0
    maximum_added_over_vocals_db: float = 6.0
    minimum_kept_fraction: float = 0.2
    minimum_agreeing_fraction: float = 0.0
    minimum_same_voice_similarity: float = 0.55
    similarity_fft: int = 2048
    similarity_max_hz: float = 8000.0
    maximum_stereo_change: float = 0.1
    maximum_level_difference_db: float = 6.0
    maximum_lag_samples: int = 1
    method: str = METHOD

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-vocal-recovery-v1-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def default_recipe() -> VocalRecoveryRecipe:
    return VocalRecoveryRecipe()


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), 1e-30)))


def _energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _stereo(value, name: str, frames: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name}は有限のstereo（frames, 2）である必要があります")
    if frames is not None and array.shape[0] != frames:
        raise ValueError(f"{name}の長さが候補区間と一致しません")
    return array


def eligibility(current_music: np.ndarray, effects: np.ndarray, recipe: VocalRecoveryRecipe) -> dict:
    """候補区間の採用中の音楽と1回目の効果音の出力だけで、4分類へ送る価値があるかを返す。"""
    music = _stereo(current_music, "採用中の音楽")
    effects = _stereo(effects, "効果音の出力", music.shape[0])
    em, ee = _energy(music), _energy(effects)
    features = {"effects_over_music_db": round(_db(ee) - _db(em), 2)}
    reasons = []
    if em <= 0:
        reasons.append("no_music")
    if features["effects_over_music_db"] < recipe.minimum_effects_db:
        reasons.append("effects_negligible")
    return {"eligible": not reasons, "features": features, "reasons": reasons}


def _frames(x: np.ndarray, size: int) -> np.ndarray:
    count = -(-x.shape[0] // size)
    return np.pad(x, ((0, count * size - x.shape[0]), (0, 0))).reshape(count, size, -1)


def _smoothed(on: np.ndarray, frames: int, size: int, rate: int, recipe: VocalRecoveryRecipe) -> np.ndarray:
    weight = np.repeat(on.astype(np.float64), size)[:frames]
    width = max(1, int(round(recipe.gate_smoothing_seconds * rate)))
    if width > 1:
        # 区間の端は端の値を延長して滑らかにする（区間の外を0とみなして端で弱めない）。
        window = np.hanning(width + 2)[1:-1]
        padded = np.pad(weight, (width // 2, width - 1 - width // 2), mode="edge")
        weight = np.convolve(padded, window / window.sum(), mode="valid")
    return np.clip(weight, 0.0, 1.0)


def coactive_frames(effects_vocals: np.ndarray, music_vocals: np.ndarray, rate: int,
                    recipe: VocalRecoveryRecipe) -> np.ndarray:
    """候補に既に同じ時間の歌唱がある枠の印（前後 ``gate_dilation_seconds`` だけ広げる）。"""
    size = max(1, int(round(recipe.frame_seconds * rate)))
    ee = np.mean(_frames(effects_vocals, size) ** 2, axis=(1, 2))
    em = np.mean(_frames(music_vocals, size) ** 2, axis=(1, 2))
    on = (em > 0) & (ee <= em * 10 ** (recipe.gate_margin_db / 10))
    widened = on.copy()
    for step in range(1, int(round(recipe.gate_dilation_seconds / recipe.frame_seconds)) + 1):
        widened[step:] |= on[:-step]
        widened[:-step] |= on[step:]
    return widened


def agreeing_frames(effects_vocals: np.ndarray, music_vocals: np.ndarray, mixture_vocals: np.ndarray,
                    rate: int, recipe: VocalRecoveryRecipe) -> np.ndarray:
    """足すと、候補の歌唱が原本（完成mix）の歌唱へ近づく枠の印。"""
    size = max(1, int(round(recipe.frame_seconds * rate)))
    missing = _frames(mixture_vocals - music_vocals, size)
    added = _frames(effects_vocals, size)
    reduction = np.sum(missing ** 2, axis=(1, 2)) - np.sum((missing - added) ** 2, axis=(1, 2))
    return reduction > 0


def gate(effects_vocals: np.ndarray, music_vocals: np.ndarray, rate: int, recipe: VocalRecoveryRecipe,
         mixture_vocals: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """足す枠を1にした、滑らかなsampleごとの重みと枠の印。

    候補に既に同じ時間の歌唱がある枠のうち、``mixture_vocals`` を渡した時は、足すと原本の歌唱へ
    近づく枠だけを残す。
    """
    size = max(1, int(round(recipe.frame_seconds * rate)))
    on = coactive_frames(effects_vocals, music_vocals, rate, recipe)
    if mixture_vocals is not None:
        on = on & agreeing_frames(effects_vocals, music_vocals, mixture_vocals, rate, recipe)
    return _smoothed(on, effects_vocals.shape[0], size, rate, recipe), on


def same_voice_similarity(effects_vocals: np.ndarray, music_vocals: np.ndarray, weight: np.ndarray,
                          rate: int, recipe: VocalRecoveryRecipe) -> float | None:
    """足す枠で、V_E と V_M の振幅スペクトル（左右の和）の余弦類似度を、足す量で重み付けした平均。

    SFCが同じ歌唱を分けて入れたなら同じ音高・倍音を持つ。叫びや台詞、器楽の一部とは倍音の位置が違う。
    """
    size = max(1, int(round(recipe.frame_seconds * rate)))
    n_fft = int(recipe.similarity_fft)
    top = max(2, int(recipe.similarity_max_hz / (rate / n_fft)))
    window = np.hanning(n_fft)
    total, weights = 0.0, 0.0
    for start in range(0, effects_vocals.shape[0], size):
        center = start + size // 2
        low = max(0, min(center - n_fft // 2, effects_vocals.shape[0] - n_fft))
        if effects_vocals.shape[0] < n_fft:
            break
        w = float(weight[min(center, weight.shape[0] - 1)])
        if w <= 0:
            continue
        a = np.abs(np.fft.rfft(effects_vocals[low:low + n_fft].T * window, axis=-1)).sum(axis=0)[1:top]
        b = np.abs(np.fft.rfft(music_vocals[low:low + n_fft].T * window, axis=-1)).sum(axis=0)[1:top]
        denominator = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
        if denominator <= 0:
            continue
        energy = w * float(np.sum(a ** 2))
        total += energy * float(np.sum(a * b)) / denominator
        weights += energy
    return total / weights if weights > 0 else None


def _stereo_correlation(x: np.ndarray) -> float:
    left, right = x[:, 0], x[:, 1]
    denominator = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)))
    return float(np.sum(left * right) / denominator) if denominator > 0 else 0.0


def _level_difference_db(x: np.ndarray) -> float:
    return _db(float(np.sum(x[:, 0] ** 2))) - _db(float(np.sum(x[:, 1] ** 2)))


def _lag(first: np.ndarray, second: np.ndarray, limit: int) -> int:
    a, b = first.mean(axis=1), second.mean(axis=1)
    size = 1 << int(np.ceil(np.log2(a.size + b.size)))
    correlation = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.concatenate([np.arange(0, limit + 1), np.arange(-limit, 0)])
    values = np.concatenate([correlation[:limit + 1], correlation[-limit:]])
    return int(lags[int(np.argmax(values))])


def recover(current_music: np.ndarray, effects_vocals: np.ndarray, music_vocals: np.ndarray,
            mixture_vocals: np.ndarray, rate: int, recipe: VocalRecoveryRecipe) -> tuple[np.ndarray, dict]:
    """challenger（採用中の音楽 + gate·V_E）と、採否の判断を返す。正解は使わない。

    ``mixture_vocals`` は原本（完成mix）の同じ区間を4分類した歌唱 V_X。足す成分がSFCの誤配分で
    分かれた歌唱なら、足すと候補の歌唱は V_X へ近づく。別の音なら遠ざかる。
    """
    music = _stereo(current_music, "採用中の音楽")
    effects_vocals = _stereo(effects_vocals, "効果音の出力の歌唱", music.shape[0])
    music_vocals = _stereo(music_vocals, "採用中の音楽の歌唱", music.shape[0])
    mixture_vocals = _stereo(mixture_vocals, "原本の歌唱", music.shape[0])
    size = max(1, int(round(recipe.frame_seconds * rate)))
    coactive = coactive_frames(effects_vocals, music_vocals, rate, recipe)
    coactive_added = effects_vocals * _smoothed(coactive, music.shape[0], size, rate, recipe)[:, None]
    weight, on = gate(effects_vocals, music_vocals, rate, recipe, mixture_vocals)
    added = effects_vocals * weight[:, None]
    challenger = music + added
    e_music, e_added = _energy(music), _energy(added)
    e_vocals, e_effects_vocals = _energy(music_vocals), _energy(effects_vocals)
    e_coactive = _energy(coactive_added)
    missing = mixture_vocals - music_vocals
    features = {
        "added_over_music_db": round(_db(e_added) - _db(e_music), 2),
        "added_over_vocals_db": round(_db(e_added) - _db(e_vocals), 2),
        "music_vocals_over_music_db": round(_db(e_vocals) - _db(e_music), 2),
        "effects_vocals_over_music_db": round(_db(e_effects_vocals) - _db(e_music), 2),
        "kept_fraction": round(e_coactive / e_effects_vocals, 4) if e_effects_vocals > 0 else 0.0,
        "agreeing_fraction": round(e_added / e_coactive, 4) if e_coactive > 0 else 0.0,
        "mixture_agreement": (round((_energy(missing) - _energy(missing - added)) / e_added, 4)
                              if e_added > 0 else None),
        "same_voice_similarity": None,
        "gate_open_fraction": round(float(np.mean(on)), 4),
        "singing_stereo_change": round(_stereo_correlation(music_vocals + added)
                                       - _stereo_correlation(music_vocals), 4),
        "singing_level_difference_db": round(_level_difference_db(music_vocals + added), 2),
        "existing_level_difference_db": round(_level_difference_db(music_vocals), 2),
        "lag_samples": _lag(added, music_vocals, max(1, int(0.01 * rate))) if e_added > 0 and e_vocals > 0 else 0,
    }
    reasons = []
    if e_added <= 0 or features["added_over_music_db"] < recipe.minimum_added_over_music_db:
        reasons.append("nothing_meaningful_to_add")
    elif features["added_over_vocals_db"] < recipe.minimum_added_over_vocals_db:
        reasons.append("negligible_against_existing_singing")
    if features["added_over_vocals_db"] > recipe.maximum_added_over_vocals_db:
        reasons.append("exceeds_existing_singing")
    if e_coactive > 0 and features["kept_fraction"] < recipe.minimum_kept_fraction:
        reasons.append("mostly_outside_existing_singing")
    if e_coactive > 0 and features["agreeing_fraction"] < recipe.minimum_agreeing_fraction:
        reasons.append("disagrees_with_mixture_singing")
    if e_added > 0:
        similarity = same_voice_similarity(effects_vocals, music_vocals, weight, rate, recipe)
        features["same_voice_similarity"] = None if similarity is None else round(similarity, 4)
        if similarity is None or similarity < recipe.minimum_same_voice_similarity:
            reasons.append("different_voice")
    if e_added > 0 and (
            features["singing_stereo_change"] < -recipe.maximum_stereo_change
            or abs(features["singing_level_difference_db"]) > max(
                abs(features["existing_level_difference_db"]), recipe.maximum_level_difference_db)):
        reasons.append("stereo_changed")
    if abs(features["lag_samples"]) > recipe.maximum_lag_samples:
        reasons.append("timing_changed")
    if not np.isfinite(challenger).all():
        reasons.append("non_finite")
    return challenger, {"adopt": not reasons, "features": features, "reasons": reasons}
