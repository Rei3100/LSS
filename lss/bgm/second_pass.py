"""SFX混入が強い劇伴候補だけを、解析器（SFC）へもう一度通し、良くなった時だけ採る。

SFCは台詞のある場面などで効果音の一部を音楽の出力へ入れる。その音楽の出力をもう一度SFCへ
通すと、混ざった効果音が取れることがある。一方、音楽だけの区間ではわずかな誤差が足されるだけで、
歌唱では欲しい声まで削ることがある。そこで二つを別に判断する。

1. **試す価値（eligibility）**：1回目の出力だけで判断する。区間の中に、音楽と競合する
   非音楽の音（効果音か台詞）がそれなりにあり、音楽が優勢でない区間。優勢かは、1秒窓の取り分の
   中央値と区間全体のエネルギーの取り分の両方で見る（突発的な強い効果音は中央値に現れない）。
   音楽が優勢で競合の無い区間は2回目で良くなる余地が無いので試さない。
2. **採用（adoption）**：1回目と2回目の音楽の出力だけを比べる。正解もSFCの2回目の台詞・効果音の
   出力も使わない。2回目が取り除いた成分R（1回目-2回目）が
   - 意味のある量で（取り除いた量が区間の-25dB以上）、
   - 突発的な瞬間に集まり（Rの上位20%の100ms区間にRのエネルギーの90%以上）、
   - Rが小さい時間（音楽だけの時間）では2回目が1回目と同じ大きさで（中央値-0.25dB以内）、
   - その時間の帯域ごとの大きさも欠けず（どの帯域も-1dB以内）、
   - 左右の相関と時間位置を変えていない
   時だけ採る。欲しい音楽を広く削る・弱める・歪める2回目はこれで落ちる。拮抗・不明なら1回目。
3. **文脈**：2回目は、1回目と同じ120秒のブロック格子（前後12秒の文脈）で、候補を含むブロックの
   1回目の音楽の出力全体を入れる。候補だけを切って入れると、切り方で結果が大きく変わるため。

実測：`.state/validation/bgm-second-pass-r1/`。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

METHOD = "bgm-second-pass-v1"
FRAME_SECONDS = 0.1
BAND_COUNT = 24


@dataclass(frozen=True)
class SecondPassRecipe:
    """試す価値と採用の宣言。どれかが変われば識別子が変わる。"""

    competing_effects_db: float = -3.0
    competing_speech_db: float = -6.0
    dominant_share: float = 0.9
    minimum_removed_db: float = -25.0
    minimum_removed_concentration: float = 0.90
    minimum_steady_db: float = -0.25
    minimum_band_db: float = -1.0
    maximum_stereo_change: float = 0.3
    maximum_lag_samples: int = 1
    method: str = METHOD

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "bgm-second-pass-v1-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _read(path: Path, start: float, end: float) -> np.ndarray:
    import soundfile as sf

    with sf.SoundFile(str(path)) as stream:
        rate = stream.samplerate
        stream.seek(int(round(start * rate)))
        return stream.read(int(round((end - start) * rate)), dtype="float64", always_2d=True)


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(value, 1e-30)))


def _frames(x: np.ndarray, rate: int) -> np.ndarray:
    size = max(1, int(round(FRAME_SECONDS * rate)))
    count = x.shape[0] // size
    return x[:count * size].reshape(count, size, x.shape[1])


def eligibility(stem_paths: Mapping[str, Path], start: float, end: float,
                recipe: SecondPassRecipe) -> dict:
    """1回目の3出力だけで、この区間を2回目に送る価値があるかを返す。"""
    import soundfile as sf

    rate = sf.info(str(stem_paths["music"])).samplerate
    music, speech, effects = (_read(stem_paths[name], start, end) for name in ("music", "speech", "effects"))
    em, es, ef = (float(np.sum(x ** 2)) for x in (music, speech, effects))
    shares = []
    for index in range(0, music.shape[0] - rate + 1, rate):
        window = [float(np.sum(x[index:index + rate] ** 2)) for x in (music, speech, effects)]
        shares.append(window[0] / max(sum(window), 1e-30))
    share = float(np.median(shares)) if shares else 1.0
    energy_share = em / max(em + es + ef, 1e-30)
    features = {"effects_over_music_db": round(_db(ef) - _db(em), 2),
                "speech_over_music_db": round(_db(es) - _db(em), 2),
                "music_share_median": round(share, 4), "music_share_energy": round(energy_share, 4)}
    competing = (features["effects_over_music_db"] >= recipe.competing_effects_db
                 or features["speech_over_music_db"] >= recipe.competing_speech_db)
    not_dominant = share < recipe.dominant_share or energy_share < recipe.dominant_share
    reasons = []
    if not competing:
        reasons.append("no_competing_non_music")
    if not not_dominant:
        reasons.append("music_dominant")
    return {"eligible": competing and not_dominant and em > 0, "features": features, "reasons": reasons}


def _band_energy(frames: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(frames.mean(axis=2), axis=1)) ** 2
    edges = np.unique(np.geomspace(2, spectrum.shape[1] - 1, BAND_COUNT + 1).astype(int))
    return np.stack([spectrum[:, a:b].sum(axis=1) for a, b in zip(edges, edges[1:])], axis=1)


def _stereo_correlation(x: np.ndarray) -> float:
    if x.shape[1] < 2:
        return 1.0
    left, right = x[:, 0], x[:, 1]
    denominator = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)))
    return float(np.sum(left * right) / denominator) if denominator > 0 else 0.0


def _lag(first: np.ndarray, second: np.ndarray, limit: int) -> int:
    a, b = first.mean(axis=1), second.mean(axis=1)
    size = 1 << int(np.ceil(np.log2(a.size + b.size)))
    correlation = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.concatenate([np.arange(0, limit + 1), np.arange(-limit, 0)])
    values = np.concatenate([correlation[:limit + 1], correlation[-limit:]])
    return int(lags[int(np.argmax(values))])


def adoption(first_music: Path, second_music: Path, start: float, end: float,
             recipe: SecondPassRecipe) -> dict:
    """1回目と2回目の音楽の出力だけで、2回目が明確に良くなったかを返す。"""
    import soundfile as sf

    rate = sf.info(str(first_music)).samplerate
    first, second = _read(first_music, start, end), _read(second_music, start, end)
    frames_count = min(first.shape[0], second.shape[0])
    first, second = first[:frames_count], second[:frames_count]
    removed = first - second
    e1 = np.mean(_frames(first, rate) ** 2, axis=(1, 2))
    e2 = np.mean(_frames(second, rate) ** 2, axis=(1, 2))
    er = np.mean(_frames(removed, rate) ** 2, axis=(1, 2))
    active = e1 > 1e-12
    total_first, total_removed = float(np.sum(first ** 2)), float(np.sum(removed ** 2))
    if not active.any() or total_first <= 0 or not np.isfinite(second).all():
        return {"adopt": False, "features": {}, "reasons": ["no_music_to_compare"]}
    steady = active & (er <= np.median(er[active]))
    ordered = np.sort(er[active])[::-1]
    top = max(1, int(0.2 * ordered.size))
    band_first = _band_energy(_frames(first, rate)[steady]).sum(axis=0)
    band_second = _band_energy(_frames(second, rate)[steady]).sum(axis=0)
    band_change = 10.0 * np.log10(np.maximum(band_second, 1e-30) / np.maximum(band_first, 1e-30))
    features = {
        "removed_db": round(_db(total_removed) - _db(total_first), 2),
        "removed_concentration": round(float(ordered[:top].sum() / max(ordered.sum(), 1e-30)), 4),
        "steady_db": round(float(np.median(10.0 * np.log10(np.maximum(e2[steady], 1e-30)
                                                            / np.maximum(e1[steady], 1e-30)))), 3),
        "band_min_db": round(float(band_change.min()), 3),
        "stereo_change": round(_stereo_correlation(second) - _stereo_correlation(first), 4),
        "lag_samples": _lag(first, second, int(0.01 * rate)),
    }
    reasons = []
    if features["removed_db"] < recipe.minimum_removed_db:
        reasons.append("nothing_meaningful_removed")
    if features["removed_concentration"] < recipe.minimum_removed_concentration:
        reasons.append("removal_spread_over_time")
    if features["steady_db"] < recipe.minimum_steady_db:
        reasons.append("music_weakened")
    if features["band_min_db"] < recipe.minimum_band_db:
        reasons.append("spectral_loss")
    if abs(features["stereo_change"]) > recipe.maximum_stereo_change:
        reasons.append("stereo_changed")
    if abs(features["lag_samples"]) > recipe.maximum_lag_samples:
        reasons.append("timing_changed")
    return {"adopt": not reasons, "features": features, "reasons": reasons}


def default_recipe() -> SecondPassRecipe:
    return SecondPassRecipe()
