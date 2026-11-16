"""観測のためだけに、原本と推定を区間ごとに読み合わせる。成果物は変更しない。

`docs/QUALITY_AND_VALIDATION.md` の「音声パイプラインの忠実度」に従い、標本化率の変更は
透明な便宜ではなく明示した変換として扱う。使った条件は観測へ添えて記録する。

長い素材を丸ごと持たないために、一定の長さの区間へ区切って読む。区間ごとに
「推定の大きさ」「原本の大きさ」「両者の内積」だけを積み、信号そのものは持ち越さない。
この三つがあれば、原本から推定を引いた相手の大きさを区間ごとに復元できる。

モノへ落とさない。伴奏は左右に広く、歌唱は中央に寄るため、モノ化は比を偏らせる。
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SCHEMA_VERSION = 1
METHOD = "observation-only-segment-energies-v1"
RESAMPLER = "scipy.signal.resample_poly-default-v1"


class ObservationReadError(RuntimeError):
    """観測のために読み合わせられない。"""


def segment_energies(mixture_path: Path, estimate_path: Path, *,
                     segment_seconds: float, estimate_gain: float = 1.0) -> dict:
    """区間ごとの（推定の大きさ, 原本の大きさ, 内積）と、使った条件を返す。

    推定の標本化率へ合わせて読む。原本の率が違う場合だけ区間ごとに合わせ、
    合わせたことを記録に残す。`estimate_gain` は保存時に掛けた倍率を戻すためのもので、
    原本と同じ大きさで比べるために要る。
    """
    if segment_seconds <= 0:
        raise ObservationReadError("区間の長さは0より大きい必要があります")
    if not estimate_gain > 0:
        raise ObservationReadError("estimate_gainは正の値である必要があります")
    with sf.SoundFile(str(estimate_path)) as estimate, sf.SoundFile(str(mixture_path)) as mixture:
        rate, channels = int(estimate.samplerate), int(estimate.channels)
        mix_rate = int(mixture.samplerate)
        if channels != int(mixture.channels):
            raise ObservationReadError("原本と推定のchannel数が違います")
        divisor = gcd(rate, mix_rate)
        up, down = rate // divisor, mix_rate // divisor
        # 区間ごとに率を合わせると、合わせ込みの立ち上がりが区間の端へ入り、
        # 原本と推定の対応がずれる。前後へ余白を付けて合わせ、余白を捨てる。
        margin_in = 0 if mix_rate == rate else down * max(1, mix_rate // down)
        mix_step = max(down, int(round(segment_seconds * mix_rate)) // down * down)
        step = mix_step * up // down
        rows = []
        position = 0
        while True:
            taken = estimate.read(step, dtype="float64", always_2d=True)
            if taken.shape[0] <= 0:
                break
            start = max(0, position - margin_in)
            mixture.seek(start)
            source = mixture.read(margin_in + mix_step + margin_in, dtype="float64",
                                  always_2d=True)
            if source.shape[0] <= 0:
                break
            head = position - start
            if mix_rate != rate:
                source = resample_poly(source, up, down, axis=0)
                head = head * up // down
            source = source[head:head + step]
            position += mix_step
            frames = min(taken.shape[0], source.shape[0])
            if frames <= 0:
                break
            taken, source = taken[:frames] * estimate_gain, source[:frames]
            rows.append({"frames": frames,
                         "estimate_energy": float(np.sum(taken ** 2)),
                         "mixture_energy": float(np.sum(source ** 2)),
                         "cross_energy": float(np.sum(taken * source))})
    if not rows:
        raise ObservationReadError("観測に使える音が読めません")
    return {"revision": {"schema_version": SCHEMA_VERSION, "method": METHOD,
                         "sample_rate": rate, "channels": channels,
                         "segment_seconds": segment_seconds,
                         "estimate_gain": estimate_gain,
                         "mixture_resampled": (None if mix_rate == rate else
                                               {"from_rate": mix_rate, "to_rate": rate,
                                                "resampler": RESAMPLER})},
            "segments": rows}


def other_energy(row: dict) -> float:
    """区間の「原本から推定を引いた相手」の大きさ。信号を持たずに復元する。"""
    value = row["mixture_energy"] + row["estimate_energy"] - 2.0 * row["cross_energy"]
    return max(0.0, value)
