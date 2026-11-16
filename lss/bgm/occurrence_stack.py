"""同じ録音の複数の使用箇所を重ねて、分離modelを使わずに音楽だけを推定する。

再演から作った候補は、定義上2回以上鳴っている。回ごとに乗っている台詞と効果音は違うが、
**音楽は同じ録音**である。したがって、ある回Aを別の回Bで説明できるぶんが音楽の推定になる。

説明の仕方は録音の同定と同じ形を使う。短い線形フィルタ（音量・EQ・位相の違いを許す）に
Bを通してAへ最小二乗で当てはめ、当てはまったものを音楽、残差を台詞・効果音とする。
AとBの台詞・効果音は互いに無相関なので、当てはめは共通成分＝音楽へ寄る。

回が3つ以上あるときの使い方は2通りあり、宣言（`combine`）が選ぶ。

- 回ごとに当てはめてから束ねる（中央値・平均・最小振幅・最も当てはまる1つ・縮み戻し）。
- **全部の回を同時に当てはめる**（`joint`）。回ごとの当てはめは音楽より小さく出るので、
  あとから束ねてもその縮みは消えない。一つの設計行列に並べて一度に解けば、
  回どうしの重なりを使って最も説明できる組み合わせが直接求まる。

`docs/BGM.md`「複数の使用箇所による証拠と再構成」のとおり、単純な波形平均を当然の処理に
しない。どれが良いかは制御corpusで測ってから決める
（`.state/validation/lv-bgm-occurrence-stack-r1/report.json`）。

分離modelを一つも使わない。参照録音も要らない。必要なのは同じ録音が2回以上鳴っていること
だけで、それはこの経路の候補が満たしている。

**推定が信用できるかは、推定が元mixのどれだけを残したかで分かる**（`minimum_energy_share`）。
実番組の既発売cue 30件で答え合わせしたところ、残した割合が0.10以上の19件は17件で
録音そのものへ近づいたのに対し、0.10未満の11件は3件しか近づかなかった
（`.state/validation/lv-bgm-released-stack-r1/`）。正解も分離modelも要らない判定なので、
未発売cueにもそのまま使える。

音量の正規化・時間軸の変更・左右の入れ替えはしない。推定は元の回Aの時間軸・channel数のまま返す。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 1
METHOD = "short-fir-common-component-v1"

COMBINE_MEDIAN = "median"
COMBINE_MEAN = "mean"
COMBINE_MIN_MAGNITUDE = "min_magnitude"
COMBINE_BEST = "best_single"
COMBINE_DEBIASED = "debiased"
# 回ごとに当てはめてから束ねるのではなく、全部の回を同時に当てはめる。
COMBINE_JOINT = "joint"
# 一度推定した音楽を寄せ集めて参照を作り、それを当てはめ直す。
COMBINE_TWO_STAGE = "two_stage_joint"
COMBINES = frozenset({COMBINE_MEDIAN, COMBINE_MEAN, COMBINE_MIN_MAGNITUDE,
                      COMBINE_BEST, COMBINE_DEBIASED, COMBINE_JOINT,
                      COMBINE_TWO_STAGE})

# 当てはめの縮みを戻すときの下限。これ以下の回は音楽の取り分が小さすぎて戻せない。
MIN_ATTENUATION = 0.05

# 位置合わせで探すずれの上限。候補の区間には前後に余裕が付いている。
MAX_LAG_SECONDS = 3.0


class OccurrenceStackError(ValueError):
    """重ね合わせの宣言、または渡された使用箇所が契約に合わない。"""


@dataclass(frozen=True)
class OccurrenceStackRecipe:
    """回どうしをどう当てはめ、どう束ねるかの宣言。"""

    fir_taps: int
    block_seconds: float
    combine: str
    minimum_energy_share: float = 0.0
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise OccurrenceStackError("未対応の重ね合わせschemaです")
        if (isinstance(self.fir_taps, bool) or not isinstance(self.fir_taps, int)
                or not 1 <= self.fir_taps <= 1025 or self.fir_taps % 2 == 0):
            raise OccurrenceStackError("fir_tapsは1以上1025以下の奇数である必要があります")
        value = self.block_seconds
        if (isinstance(value, bool) or not isinstance(value, Real) or value != value
                or not 0.0 < float(value) <= 10.0):
            raise OccurrenceStackError("block_secondsは0より大きく10以下である必要があります")
        if self.combine not in COMBINES:
            raise OccurrenceStackError(f"未対応の束ね方です: {self.combine!r}")
        value = self.minimum_energy_share
        if (isinstance(value, bool) or not isinstance(value, Real) or value != value
                or not 0.0 <= float(value) <= 1.0):
            raise OccurrenceStackError("minimum_energy_shareは0以上1以下である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return f"bgm-occurrence-stack-v{SCHEMA_VERSION}-" + hashlib.sha256(body.encode()).hexdigest()[:16]


def _checked(signal: Any, context: str) -> np.ndarray:
    array = np.asarray(signal)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise OccurrenceStackError(f"{context}は(フレーム, channel)の音声である必要があります")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise OccurrenceStackError(f"{context}に有限でない値が含まれています")
    return array


def best_lag(target: np.ndarray, other: np.ndarray, *, max_lag: int) -> int:
    """`other` を何標本ずらすと `target` に最もよく重なるか。

    返す値は「`other` の先頭が `target` の何標本目に当たるか」。channelは足して見る。
    """
    a = target.sum(axis=1)
    b = other.sum(axis=1)
    size = 1 << int(np.ceil(np.log2(max(a.size + b.size, 4))))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    values = np.concatenate([corr[:max_lag + 1], corr[-max_lag:]]) if max_lag else corr[:1]
    return int(lags[int(np.argmax(np.abs(values)))])


def _shifted(other: np.ndarray, lag: int, frames: int, channels: int) -> np.ndarray:
    """`other` を `lag` だけずらして、`target` と同じ長さ・同じchannel数へ揃える。"""
    out = np.zeros((frames, channels), dtype=np.float64)
    if other.shape[1] == 1 and channels > 1:
        other = np.repeat(other, channels, axis=1)
    elif other.shape[1] > channels:
        other = other[:, :channels]
    elif other.shape[1] < channels:
        other = np.pad(other, ((0, 0), (0, channels - other.shape[1])), mode="edge")
    start = max(0, lag)
    source_start = max(0, -lag)
    length = min(frames - start, other.shape[0] - source_start)
    if length > 0:
        out[start:start + length] = other[source_start:source_start + length]
    return out


def align_to(target: np.ndarray, other: np.ndarray, *,
             sample_rate: int) -> tuple[np.ndarray, int]:
    """`other` を `target` の時間軸へ揃える。揃えた信号と使ったずれを返す。"""
    frames, channels = target.shape
    max_lag = min(frames - 1, int(round(MAX_LAG_SECONDS * sample_rate)))
    lag = best_lag(target, other, max_lag=max_lag)
    return _shifted(other, lag, frames, channels), lag


def _block_plan(frames: int, recipe: OccurrenceStackRecipe, sample_rate: int):
    block = max(recipe.fir_taps * 2, int(round(recipe.block_seconds * sample_rate)))
    block = min(block, frames)
    hop = max(1, block // 2)
    window = np.hanning(block + 2)[1:-1] if block > 2 else np.ones(block)
    return block, hop, window


def _solve(design: np.ndarray, y: np.ndarray, taps: int) -> np.ndarray:
    """正規方程式で解く。tap数は小さく、blockは長いので条件は良い。

    数値的な保険として、対角へごく小さな量を足す（解を実質変えない）。
    """
    gram = design.T @ design
    ridge = 1e-9 * float(np.trace(gram)) / max(1, gram.shape[0])
    try:
        return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), design.T @ y)
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coefficients


def fit_explained_jointly(
    target: np.ndarray, others: Sequence[np.ndarray], recipe: OccurrenceStackRecipe, *,
    sample_rate: int,
) -> np.ndarray:
    """複数の回を**同時に**当てはめる。回ごとに当てはめてから束ねるのとは違う。

    回ごとに当てはめると、それぞれの推定は音楽より小さく出る（縮み）。あとから束ねても
    縮みは消えない。全部の回を一つの設計行列に並べて一度に解けば、回どうしの重なりを
    使って最も説明できる組み合わせが直接求まる。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    recipe.validate()
    if not others:
        raise OccurrenceStackError("同時に当てはめる使用箇所がありません")
    frames, channels = target.shape
    block, hop, window = _block_plan(frames, recipe, sample_rate)
    half = recipe.fir_taps // 2

    estimate = np.zeros_like(target)
    weight = np.zeros(frames, dtype=np.float64)
    for channel in range(channels):
        y_all = target[:, channel]
        padded = [np.pad(other[:, channel], (half, recipe.fir_taps - half - 1))
                  for other in others]
        for begin in range(0, max(1, frames - block + 1), hop):
            end = min(begin + block, frames)
            size = end - begin
            if size < recipe.fir_taps:
                continue
            columns = [sliding_window_view(item[begin:end + recipe.fir_taps - 1], recipe.fir_taps)
                       for item in padded]
            design = np.concatenate(columns, axis=1)
            if not np.any(design):
                continue
            coefficients = _solve(design, y_all[begin:end], recipe.fir_taps)
            taper = window[:size] if size == block else np.hanning(size + 2)[1:-1]
            estimate[begin:end, channel] += (design @ coefficients) * taper
            if channel == 0:
                weight[begin:end] += taper
    positive = weight > 1e-12
    estimate[positive] /= weight[positive, None]
    estimate[~positive] = 0.0
    return estimate


def fit_explained(
    target: np.ndarray, other: np.ndarray, recipe: OccurrenceStackRecipe, *,
    sample_rate: int,
) -> np.ndarray:
    """`other` を短い線形フィルタへ通して `target` へ当てはめ、当てはまったぶんを返す。

    重なり半分のブロックごとに係数を求め、Hann窓で繋ぐ。区間の途中で音量やduckingが
    変わっても追えるようにするため、フィルタは一本にしない。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    recipe.validate()
    frames, channels = target.shape
    block, hop, window = _block_plan(frames, recipe, sample_rate)
    half = recipe.fir_taps // 2

    estimate = np.zeros_like(target)
    weight = np.zeros(frames, dtype=np.float64)
    for channel in range(channels):
        y_all = target[:, channel]
        x_all = other[:, channel]
        padded = np.pad(x_all, (half, recipe.fir_taps - half - 1))
        for begin in range(0, max(1, frames - block + 1), hop):
            end = min(begin + block, frames)
            size = end - begin
            if size < recipe.fir_taps:
                continue
            y = y_all[begin:end]
            design = sliding_window_view(padded[begin:end + recipe.fir_taps - 1], recipe.fir_taps)
            if not np.any(design):
                continue
            coefficients = _solve(design, y, recipe.fir_taps)
            taper = window[:size] if size == block else np.hanning(size + 2)[1:-1]
            estimate[begin:end, channel] += (design @ coefficients) * taper
            if channel == 0:
                weight[begin:end] += taper
    positive = weight > 1e-12
    estimate[positive] /= weight[positive, None]
    estimate[~positive] = 0.0
    return estimate


def attenuations(estimates: Sequence[np.ndarray], target: np.ndarray) -> list[float]:
    """最小二乗の当てはめが音楽をどれだけ縮めたかを、回どうしの突き合わせから求める。

    回Bを回Aへ最小二乗で当てはめると、Bに乗っている台詞・効果音のぶんだけ
    推定は音楽より**小さく**出る（縮み率 α = Bの音楽の取り分 / (1+取り分)）。
    このズレは一定方向なので、そのまま足しても消えない。

    回が2つ以上あれば縮み率そのものを測れる。回B・Cの推定 g_B・g_C は同じ音楽 M の
    α倍なので、⟨g_B, g_C⟩ = α_B α_C ⟨M, M⟩、⟨g_B, A⟩ = α_B ⟨M, M⟩ が成り立つ。
    ここから α_B = ⟨g_B, g_C⟩ / ⟨g_C, A⟩ が、Mを知らずに出る。
    """
    values = []
    for index, estimate in enumerate(estimates):
        ratios = []
        for other_index, other in enumerate(estimates):
            if other_index == index:
                continue
            denominator = float(np.sum(other * target))
            if abs(denominator) <= 1e-30:
                continue
            ratios.append(float(np.sum(estimate * other)) / denominator)
        usable = [value for value in ratios if value > MIN_ATTENUATION]
        values.append(min(1.0, float(np.median(usable))) if usable else 1.0)
    return values


def _scaled_to(reference: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """`signal` を一つの係数で `reference` へ合わせる。回ごとの音量差を揃えるため。"""
    denominator = float(np.sum(signal * signal))
    if denominator <= 1e-30:
        return signal
    return signal * (float(np.sum(reference * signal)) / denominator)


def two_stage_estimate(
    target: np.ndarray, aligned: Sequence[np.ndarray], recipe: OccurrenceStackRecipe, *,
    sample_rate: int,
) -> np.ndarray:
    """一度推定した音楽を寄せ集めて参照を作り、それを当てはめ直す。

    一段目の推定は、当てはめ先の回が持っていて他の回が持っていない音楽を落とす。
    回ごとの推定を**共通の参照**へまとめてから当てはめ直すと、regressorが元mixより
    綺麗になるぶん、当てはめが縮みにくくなる。

    回が1つしか無いときは一段目と同じ（寄せ集める相手がいない）。
    """
    if not aligned:
        raise OccurrenceStackError("同時に当てはめる使用箇所がありません")
    first = fit_explained_jointly(target, aligned, recipe, sample_rate=sample_rate)
    if len(aligned) < 2:
        return first
    parts = [first]
    for index, other in enumerate(aligned):
        rest = [target] + [item for position, item in enumerate(aligned) if position != index]
        estimate = fit_explained_jointly(other, rest, recipe, sample_rate=sample_rate)
        parts.append(_scaled_to(first, estimate))
    reference = np.mean(np.stack(parts, axis=0), axis=0)
    if not np.any(reference):
        return first
    return fit_explained_jointly(target, [reference], recipe, sample_rate=sample_rate)


def combine_estimates(estimates: Sequence[np.ndarray], combine: str,
                      target: np.ndarray | None = None) -> np.ndarray:
    """回ごとの推定を一つへ束ねる。束ね方は宣言が決める。"""
    if not estimates:
        raise OccurrenceStackError("束ねる推定がありません")
    stack = np.stack(estimates, axis=0)
    if combine == COMBINE_MEAN:
        return stack.mean(axis=0)
    if combine == COMBINE_MEDIAN:
        return np.median(stack, axis=0)
    if combine == COMBINE_MIN_MAGNITUDE:
        index = np.argmin(np.abs(stack), axis=0)
        return np.take_along_axis(stack, index[None], axis=0)[0]
    if combine == COMBINE_BEST:
        if target is None:
            raise OccurrenceStackError("best_singleには当てはめ先が必要です")
        residuals = [float(np.sum((target - item) ** 2)) for item in estimates]
        return estimates[int(np.argmin(residuals))]
    if combine == COMBINE_DEBIASED:
        if target is None:
            raise OccurrenceStackError("debiasedには当てはめ先が必要です")
        if len(estimates) < 2:
            # 回が1つでは縮み率を測れない。縮んだままの推定を返す。
            return np.asarray(estimates[0], dtype=np.float64)
        scaled = [item / value for item, value in
                  zip(estimates, attenuations(estimates, target))]
        return np.median(np.stack(scaled, axis=0), axis=0)
    raise OccurrenceStackError(f"未対応の束ね方です: {combine!r}")


def estimate_music(
    target: Any, others: Iterable[Any], recipe: OccurrenceStackRecipe, *,
    sample_rate: int,
) -> dict[str, Any]:
    """`target` の回に鳴っている音楽を、他の回から推定する。

    返すのは推定信号と、使った回ごとのずれ・説明率。推定は `target` と同じ長さ・
    同じchannel数で、音量の正規化や時間軸の変更はしていない。
    """
    recipe.validate()
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise OccurrenceStackError("sample_rateは正の整数である必要があります")
    signal = _checked(target, "target")
    frames = signal.shape[0]

    estimates, used, aligned_others = [], [], []
    energy = float(np.sum(signal * signal))
    for position, item in enumerate(others):
        other = _checked(item, f"others[{position}]")
        aligned, lag = align_to(signal, other, sample_rate=sample_rate)
        if not np.any(aligned):
            continue
        aligned_others.append(aligned)
        fitted = fit_explained(signal, aligned, recipe, sample_rate=sample_rate)
        residual = float(np.sum((signal - fitted) ** 2))
        estimates.append(fitted)
        used.append({"index": position, "lag_samples": lag,
                     "lag_seconds": round(lag / sample_rate, 6),
                     "explained": round(max(0.0, 1.0 - residual / energy), 6) if energy > 0 else 0.0})
    if not estimates:
        raise OccurrenceStackError("重ねられる使用箇所がありません")

    if recipe.combine == COMBINE_JOINT:
        music = fit_explained_jointly(signal, aligned_others, recipe, sample_rate=sample_rate)
    elif recipe.combine == COMBINE_TWO_STAGE:
        music = two_stage_estimate(signal, aligned_others, recipe, sample_rate=sample_rate)
    else:
        music = combine_estimates(estimates, recipe.combine, target=signal)
    residual = float(np.sum((signal - music) ** 2))
    # 推定が元mixのどれだけを残したか。小さすぎる推定は当てはめの残りかすで、信用できない。
    share = float(np.sum(music * music) / energy) if energy > 0 else 0.0
    return {
        "recipe_id": recipe.recipe_id,
        "recipe": recipe.to_dict(),
        "sample_rate": sample_rate,
        "occurrence_count": len(estimates) + 1,
        "used": used,
        "combined_explained": round(max(0.0, 1.0 - residual / energy), 6) if energy > 0 else 0.0,
        "energy_share": round(share, 6),
        "meets_minimum_energy_share": bool(share >= recipe.minimum_energy_share),
        "music": music,
        "residual": signal - music,
        "claim_boundary": [
            "分離modelを使わず、同じ録音の別の回で説明できるぶんを音楽としている",
            "説明できない音楽（その回だけの演奏差・ducking後の欠落）は残差へ落ちる",
            "欲しい劇伴かどうかは判定しない",
            "残した割合が下限に届かない推定は、当てはめの残りかすであり信用しない",
        ],
    }
