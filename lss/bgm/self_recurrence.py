"""同じ番組の中で同じ録音が二度以上鳴る区間を、番組一本だけから見つける。

`docs/BGM.md` の「Occurrence（使用箇所）」に対応する。別の回の再登場は曲の範囲の根拠に
なるが、手元に一話しかない番組では使えない。同じ番組の中の再演も同じ種類の根拠であり、
参照録音も別の回も要らない。

見つけ方は別の回との共通録音と同じで、指紋の一致と波形の説明率による。自分自身との
自明な一致（同じ時刻）は除く。自己相似の変化（novelty）とは別物で、ここでは同じ録音で
あることだけを根拠にする。意味的role・公開可否・曲の名前は決めない。

見つからないことは、その番組に繰り返しが無い証拠にも、単独出現である証拠にもしない。
"""
from __future__ import annotations

from numbers import Real

import numpy as np

from lss.bgm import identity

SCHEMA_VERSION = 1
METHOD = "within-source-repeated-recording-v1"
# 同じ場所を指す自明な一致を除くための間隔。曲の最小長（6秒）より十分長く取る。
MINIMUM_SEPARATION_SECONDS = 20.0
# 同じ再演が、わずかにずれた整列として二度記録されることがある。両側がこれ以上重なる
# 組は一つの再演として読む。
SAME_PAIR_OVERLAP = 0.5
# 索引の区間の切り方を変えて二度探す。一つの区間に二度の登場が収まると、同じ参照の
# 別整列として畳まれてしまう。区間の境目が間に落ちる切り方があれば拾える。
CHUNK_PHASE_RATIOS = (0.0, 0.5)


class SelfRecurrenceError(ValueError):
    """番組一本からの再演の探し方が契約に合わない。"""


def _overlaps(first: tuple[float, float], second: tuple[float, float]) -> bool:
    overlap = min(first[1], second[1]) - max(first[0], second[0])
    shorter = min(first[1] - first[0], second[1] - second[0])
    return shorter > 0 and overlap / shorter >= SAME_PAIR_OVERLAP


def _same_pair(first: identity.SharedOccurrence, second: identity.SharedOccurrence) -> bool:
    """両側が重なる組は、同じ再演をわずかに違う整列で読んだものとして扱う。"""
    return (_overlaps((first.first_start_seconds, first.first_end_seconds),
                      (second.first_start_seconds, second.first_end_seconds))
            and _overlaps((first.second_start_seconds, first.second_end_seconds),
                          (second.second_start_seconds, second.second_end_seconds)))


def _valid(occurrence: identity.SharedOccurrence) -> bool:
    return (occurrence.first_end_seconds > occurrence.first_start_seconds
            and occurrence.second_end_seconds > occurrence.second_start_seconds
            and occurrence.blocks > 0)


def repeated_recordings(
    signal: np.ndarray, *, chunk_seconds: float = identity.SHARED_CHUNK_SECONDS,
    minimum_separation_seconds: float = MINIMUM_SEPARATION_SECONDS,
) -> list[identity.SharedOccurrence]:
    """一本の信号の中で、同じ録音が別の時刻にも鳴る区間の組を返す。

    返る組は「先に鳴る区間」「後で鳴る区間」の順で、同じ組を向きだけ変えて重複させない。
    """
    if isinstance(minimum_separation_seconds, bool) or not isinstance(minimum_separation_seconds, Real):
        raise SelfRecurrenceError("minimum_separation_secondsは秒数である必要があります")
    if float(minimum_separation_seconds) < 0.0:
        raise SelfRecurrenceError("minimum_separation_secondsは0以上である必要があります")
    data = np.asarray(signal)
    if data.ndim != 1 or data.size <= 0:
        raise SelfRecurrenceError("単一channelの信号が必要です")

    size = int(chunk_seconds * identity.SAMPLE_RATE)
    # 一区間に収まる信号では、切り方を変えても同じ索引になる。無駄に二度探さない。
    ratios = CHUNK_PHASE_RATIOS if data.size > size else CHUNK_PHASE_RATIOS[:1]
    found: list[identity.SharedOccurrence] = []
    for ratio in ratios:
        shift = int(size * ratio)
        if shift >= data.size:
            continue
        seconds = shift / identity.SAMPLE_RATE
        for item in identity.shared_recordings(data[shift:], data, chunk_seconds=chunk_seconds):
            found.append(identity.SharedOccurrence(
                item.first_start_seconds + seconds, item.first_end_seconds + seconds,
                item.second_start_seconds, item.second_end_seconds,
                item.waveform_explained, item.blocks))
    separation = float(minimum_separation_seconds)
    ordered: list[identity.SharedOccurrence] = []
    for occurrence in found:
        if not _valid(occurrence):
            continue
        if abs(occurrence.first_start_seconds - occurrence.second_start_seconds) < separation:
            # 同じ時刻を指す一致は、繰り返しの証拠にならない。
            continue
        if occurrence.first_start_seconds <= occurrence.second_start_seconds:
            ordered.append(occurrence)
            continue
        ordered.append(identity.SharedOccurrence(
            occurrence.second_start_seconds, occurrence.second_end_seconds,
            occurrence.first_start_seconds, occurrence.first_end_seconds,
            occurrence.waveform_explained, occurrence.blocks))

    kept: list[identity.SharedOccurrence] = []
    for occurrence in sorted(ordered, key=lambda item: (-item.blocks,
                                                        -(item.waveform_explained or 0.0),
                                                        item.first_start_seconds)):
        if any(_same_pair(occurrence, existing) for existing in kept):
            continue
        kept.append(occurrence)
    kept.sort(key=lambda item: (item.first_start_seconds, item.second_start_seconds))
    return kept


def spans_for_source(occurrences: list, *, source_key: str) -> list[dict]:
    """番組の中の再演を、曲の範囲の読み取りが受け取る区間へ並べ直す。

    両方の登場を同じ扱いで残す。相手は同じ番組なので、`other_episode` はこの番組自身を指す。
    """
    if not isinstance(source_key, str) or not source_key.strip():
        raise SelfRecurrenceError("番組の識別子が必要です")
    spans = []
    for occurrence in occurrences:
        first = (occurrence.first_start_seconds, occurrence.first_end_seconds)
        second = (occurrence.second_start_seconds, occurrence.second_end_seconds)
        for own, other in ((first, second), (second, first)):
            spans.append({
                "other_episode": source_key.strip(),
                "start_seconds": own[0], "end_seconds": own[1],
                "other_start_seconds": other[0], "other_end_seconds": other[1],
                "waveform_explained": occurrence.waveform_explained or 0.0,
                "same_timeline": True,
            })
    spans.sort(key=lambda row: (row["start_seconds"], row["other_start_seconds"]))
    return spans
