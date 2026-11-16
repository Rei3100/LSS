"""同じ録音の二つの使用箇所を突き合わせ、どちらに余計な音が多いかを読む。

`docs/BGM.md` の「複数の使用箇所による証拠」と「最良の使用箇所」に対応する。参照録音も
正解音源も要らない。同じ録音が二度鳴るなら、両方に共通して残る成分はその録音に由来する
可能性が高く、片方にだけ乗っている成分は、その場面で重なった台詞・効果音や処理の違いで
ある可能性が高い。

比べる前に二つの大きさを揃える。片方が小さく鳴っているだけで「余計が少ない」と読むと、
音量を下げられた場面ほど良いことになってしまうためである。

ここで返すのは観測であって、音を作り直す処理ではない。共通部分を取り出して繋ぎ直したり、
差分を引いて「きれいにする」ことはしない。順位を決めるしきい値は呼び出し側が先に宣言し、
差が小さいときは順位を付けない。

制作上の理由（ducking、EQ、編集）で余計に見える差が出ることがある。ここでの「余計」は
「共通していない」という意味であり、不要音の証明ではない。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
METHOD = "paired-occurrence-excess-energy-v1"
STATES = ("first_has_less_excess", "second_has_less_excess", "unresolved")
_SILENCE_FLOOR_DBFS = -200.0


class OccurrenceContrastError(ValueError):
    """突き合わせの宣言、または渡された音声が契約に合わない。"""


@dataclass(frozen=True)
class OccurrenceContrastRecipe:
    """比べ方と、順位を付けてよい差の大きさを先に固定する。"""

    window_frames: int
    hop_frames: int
    minimum_seconds: float
    minimum_excess_share_margin: float
    schema_version: int = SCHEMA_VERSION
    method: str = METHOD

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != METHOD:
            raise OccurrenceContrastError("未対応の突き合わせschemaです")
        for field in ("window_frames", "hop_frames"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise OccurrenceContrastError(f"{field}は正の整数である必要があります")
        if self.hop_frames > self.window_frames:
            raise OccurrenceContrastError("hop_framesはwindow_frames以下である必要があります")
        for field in ("minimum_seconds", "minimum_excess_share_margin"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or float(value) <= 0.0):
                raise OccurrenceContrastError(f"{field}は有限な正数である必要があります")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def recipe_id(self) -> str:
        self.validate()
        body = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (f"bgm-occurrence-contrast-v{SCHEMA_VERSION}-"
                + hashlib.sha256(body.encode()).hexdigest()[:16])


def _mono(samples: Any, context: str) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim == 2:
        if values.shape[0] <= 0 or values.shape[1] <= 0:
            raise OccurrenceContrastError(f"{context}に音がありません")
        values = values.mean(axis=0)
    if values.ndim != 1 or values.size <= 0:
        raise OccurrenceContrastError(f"{context}は(channels, frames)か1次元である必要があります")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise OccurrenceContrastError(f"{context}に有限でない値があります")
    return values.astype(np.float64, copy=False)


def _dbfs(value: float) -> float:
    return _SILENCE_FLOOR_DBFS if value <= 0 else max(_SILENCE_FLOOR_DBFS,
                                                      10.0 * math.log10(value))


def _magnitudes(signal: np.ndarray, recipe: OccurrenceContrastRecipe) -> np.ndarray:
    window = np.hanning(recipe.window_frames)
    count = 1 + (signal.size - recipe.window_frames) // recipe.hop_frames
    return np.stack([
        np.abs(np.fft.rfft(signal[index * recipe.hop_frames:
                                  index * recipe.hop_frames + recipe.window_frames] * window))
        for index in range(count)])


def compare(first: Any, second: Any, sample_rate: int,
            recipe: OccurrenceContrastRecipe) -> dict:
    """二つの使用箇所の、共通していない成分の大きさを返す。

    二つは既に同じ録音の同じ位置として整列している前提である。長さが違うときは短い方に
    合わせ、合わせた長さだけを比べる。
    """
    recipe.validate()
    if type(sample_rate) is not int or sample_rate <= 0:
        raise OccurrenceContrastError("正のsample rateが必要です")
    left, right = _mono(first, "first"), _mono(second, "second")
    frames = min(left.size, right.size)
    if frames < max(recipe.window_frames, int(recipe.minimum_seconds * sample_rate)):
        raise OccurrenceContrastError("突き合わせるには宣言した長さの音声が必要です")
    left, right = left[:frames], right[:frames]

    magnitudes = [_magnitudes(left, recipe), _magnitudes(right, recipe)]
    energies = [float(np.sum(item ** 2)) for item in magnitudes]
    if min(energies) <= 0:
        raise OccurrenceContrastError("どちらかの使用箇所に音がありません")
    # 片方が小さく鳴っているだけで「余計が少ない」と読まないよう、先に大きさを揃える。
    gain = float(np.sum(magnitudes[0] * magnitudes[1]) / energies[1])
    if gain <= 0:
        raise OccurrenceContrastError("二つの使用箇所の大きさを揃えられません")
    scaled = [magnitudes[0], magnitudes[1] * gain]

    common = np.minimum(scaled[0], scaled[1])
    excess = [float(np.sum((item - common) ** 2)) for item in scaled]
    total = [float(np.sum(item ** 2)) for item in scaled]
    shared = float(np.sum(common ** 2))
    # 自分の大きさに対する割合で見る。大きさを揃えた後でも残る偏りを避ける。
    shares = [item / value for item, value in zip(excess, total)]
    difference = shares[0] - shares[1]

    if abs(difference) < recipe.minimum_excess_share_margin:
        state = "unresolved"
        reason = "excess_difference_below_margin"
    elif difference < 0:
        state, reason = "first_has_less_excess", "first_adds_less_than_the_common_part"
    else:
        state, reason = "second_has_less_excess", "second_adds_less_than_the_common_part"
    return {
        "state": state, "reason": reason, "recipe_id": recipe.recipe_id,
        "compared_seconds": frames / sample_rate,
        "level_gain": gain,
        "excess_energy_dbfs": [_dbfs(value) for value in excess],
        "excess_share": shares,
        "common_energy_dbfs": _dbfs(shared),
        "excess_share_difference": difference,
    }
