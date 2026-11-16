"""D-8「継承しない・再計算する」タグ群のトラック単位scan(I23)。

Integrated Loudness(ITU-R BS.1770)はpyloudnormへvendor委譲し再実装しない(A10・metrics.pyと同方針)。
True Peakは同規格Annex2の4倍オーバーサンプル法。Loudness RangeはEBU R128 Tech3342の簡易実装
(3秒非重複窓のshort-term loudnessへ絶対/相対ゲートをかけP95-P10)。Dynamic Rangeは公開されている
DR14アルゴリズム(3秒ブロックのRMS上位20%平均とセカンドピークの比・チャンネル平均)。
foobar2000純正コンポーネントとの数値完全一致は保証しない(非公開実装のため)が、
「元曲の値をコピーすると嘘になる」というR1違反を避けるための実測再計算という目的は満たす。

`samples`は(num_channels, num_samples)を規約とする(lss.harness.metrics.compute_decision_metricsと同じ)。
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import lfilter, resample_poly

REFERENCE_LOUDNESS_LUFS = -18.0  # ReplayGain 2.0の慣例基準値(BS.1770系実装での一般的な採用値)
BLOCK_SECONDS = 3.0
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = 20.0
TRUE_PEAK_OVERSAMPLE = 4
TRUE_PEAK_BLOCK_FRAMES = 1 << 18
TRUE_PEAK_OVERLAP_FRAMES = 64
LOUDNESS_FILTER_BLOCK_FRAMES = 1 << 18
FILE_SCAN_BLOCK_FRAMES = 1 << 18

FrameReader = Callable[[int, int], np.ndarray]


def _frames_by_channels(samples: np.ndarray) -> np.ndarray:
    """(channels, frames)を pyloudnorm/scipy が期待する(frames, channels)へ転置する。"""
    return samples.T if samples.ndim == 2 else samples[:, np.newaxis]


def sample_peak_linear(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples))) if samples.size else 0.0


def _sample_peak_from_reader(reader: FrameReader, total_frames: int) -> float:
    peak = 0.0
    for start in range(0, total_frames, FILE_SCAN_BLOCK_FRAMES):
        data = reader(start, min(FILE_SCAN_BLOCK_FRAMES, total_frames - start))
        if data.size:
            peak = max(peak, float(np.max(np.abs(data))))
    return peak


def _integrated_loudness_from_reader(
    reader: FrameReader, total_frames: int, channels: int, samplerate: int, dtype,
) -> float:
    meter = pyln.Meter(samplerate)
    if total_frames < meter.block_size * samplerate:
        return float("-inf")  # 計測窓に満たない音声の値を推測しない。
    filter_states = {
        name: [np.zeros(max(len(stage.a), len(stage.b)) - 1) for _ in range(channels)]
        for name, stage in meter._filters.items()
    }
    block_energies: list[np.ndarray] = []
    block_seconds = meter.block_size
    step_ratio = 1.0 - meter.overlap
    buffered = np.empty((0, channels), dtype=dtype)
    buffer_start = 0
    processed = 0
    next_block = 0

    def bounds(index: int) -> tuple[int, int]:
        lower = int(block_seconds * (index * step_ratio) * samplerate)
        upper = int(block_seconds * (index * step_ratio + 1) * samplerate)
        return lower, upper

    def collect_available(*, allow_partial: bool = False, target_blocks: int | None = None) -> None:
        nonlocal buffered, buffer_start, next_block
        while target_blocks is None or next_block < target_blocks:
            lower, upper = bounds(next_block)
            if upper > processed and not allow_partial:
                break
            if lower >= processed:
                break
            local_start = lower - buffer_start
            local_end = min(upper, processed) - buffer_start
            block = buffered[local_start:local_end]
            denominator = block_seconds * samplerate
            block_energies.append(np.sum(np.square(block), axis=0) / denominator)
            next_block += 1
        next_lower, _ = bounds(next_block)
        drop = max(0, min(next_lower - buffer_start, buffered.shape[0]))
        if drop:
            buffered = buffered[drop:]
            buffer_start += drop

    for start in range(0, total_frames, LOUDNESS_FILTER_BLOCK_FRAMES):
        filtered = reader(
            start, min(LOUDNESS_FILTER_BLOCK_FRAMES, total_frames - start),
        ).copy()
        for name, stage in meter._filters.items():
            for channel in range(channels):
                values, state = lfilter(
                    stage.b, stage.a, filtered[:, channel], zi=filter_states[name][channel],
                )
                filtered[:, channel] = stage.passband_gain * values
                filter_states[name][channel] = state
        buffered = np.concatenate((buffered, filtered), axis=0)
        processed += filtered.shape[0]
        collect_available()

    duration = total_frames / samplerate
    target_blocks = int(np.round(((duration - block_seconds) / (block_seconds * step_ratio))) + 1)
    collect_available(allow_partial=True, target_blocks=target_blocks)
    energies = np.asarray(block_energies, dtype=np.float64).T
    channel_gains = np.asarray([1.0, 1.0, 1.0, 1.41, 1.41][:channels])
    with np.errstate(divide="ignore", invalid="ignore"):
        block_loudness = -0.691 + 10.0 * np.log10(np.sum(channel_gains[:, None] * energies, axis=0))
    absolute = block_loudness >= -70.0
    if not np.any(absolute):
        return float("-inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        absolute_mean = np.mean(energies[:, absolute], axis=1)
        relative_gate = -0.691 + 10.0 * np.log10(np.sum(channel_gains * absolute_mean)) - 10.0
    gated = (block_loudness > relative_gate) & (block_loudness > -70.0)
    if not np.any(gated):
        return float("-inf")
    gated_mean = np.nan_to_num(np.mean(energies[:, gated], axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-0.691 + 10.0 * np.log10(np.sum(channel_gains * gated_mean)))


def integrated_loudness_lufs(samples: np.ndarray, samplerate: int) -> float:
    data = _frames_by_channels(samples)
    return _integrated_loudness_from_reader(
        lambda start, frames: data[start:start + frames],
        data.shape[0], data.shape[1], samplerate, data.dtype,
    )


def _true_peak_from_reader(
    reader: FrameReader, total_frames: int, *, oversample: int,
) -> tuple[float, int, int]:
    if total_frames == 0:
        return float("-inf"), 0, 0
    peak_linear = 0.0
    peak_index_over = 0
    clipped_oversampled = 0
    for core_start in range(0, total_frames, TRUE_PEAK_BLOCK_FRAMES):
        core_end = min(core_start + TRUE_PEAK_BLOCK_FRAMES, total_frames)
        read_start = max(0, core_start - TRUE_PEAK_OVERLAP_FRAMES)
        read_end = min(total_frames, core_end + TRUE_PEAK_OVERLAP_FRAMES)
        oversampled = resample_poly(
            reader(read_start, read_end - read_start), oversample, 1, axis=0,
        )
        local_start = (core_start - read_start) * oversample
        local_end = local_start + (core_end - core_start) * oversample
        per_frame_peak = np.max(np.abs(oversampled[local_start:local_end]), axis=1)
        local_peak_index = int(np.argmax(per_frame_peak))
        local_peak = float(per_frame_peak[local_peak_index])
        if local_peak > peak_linear:
            peak_linear = local_peak
            peak_index_over = core_start * oversample + local_peak_index
        clipped_oversampled += int(
            np.sum(np.abs(oversampled[local_start:local_end]) >= 1.0)
        )
    peak_dbtp = 20.0 * float(np.log10(peak_linear)) if peak_linear > 0 else float("-inf")
    peak_position_sample = peak_index_over // oversample
    # True Peak Scannerの現行タグ規約に合わせ、全チャンネルの超過値を合計したうえで
    # オーバーサンプル倍率を戻し、元サンプルレート相当の件数にする。
    clipped = int(round(clipped_oversampled / oversample))
    return peak_dbtp, peak_position_sample, clipped


def true_peak(samples: np.ndarray, samplerate: int, oversample: int = TRUE_PEAK_OVERSAMPLE) -> tuple[float, int, int]:
    """(true_peak_dbtp, peak_position_sample, clipped_sample_count)を返す。

    peak_position_sampleは原サンプルレート基準のindex(オーバーサンプル後indexを間引き戻したもの)。
    clipped_sample_countは全チャンネルで絶対値が1.0(0dBTP)以上になった値を、
    元サンプルレート相当へ換算したサンプル数。
    """
    data = _frames_by_channels(samples)
    return _true_peak_from_reader(
        lambda start, frames: data[start:start + frames], data.shape[0], oversample=oversample,
    )


def _loudness_range_from_reader(
    reader: FrameReader, total_frames: int, samplerate: int,
) -> float:
    block_frames = int(BLOCK_SECONDS * samplerate)
    if block_frames <= 0 or total_frames < block_frames:
        return 0.0

    meter = pyln.Meter(samplerate)
    loudness_values = []
    for start in range(0, total_frames - block_frames + 1, block_frames):
        block = reader(start, block_frames)
        try:
            value = meter.integrated_loudness(block)
        except Exception:
            continue
        if np.isfinite(value) and value > ABSOLUTE_GATE_LUFS:
            loudness_values.append(value)

    if len(loudness_values) < 2:
        return 0.0
    mean = float(np.mean(loudness_values))
    gated = [v for v in loudness_values if v > mean - RELATIVE_GATE_LU]
    if len(gated) < 2:
        return 0.0
    return float(np.percentile(gated, 95) - np.percentile(gated, 10))


def loudness_range_lu(samples: np.ndarray, samplerate: int) -> float:
    """EBU R128 Loudness Range簡易実装。3秒に満たない/ゲート後1点以下ならR1違反を避け0.0を返す。"""
    data = _frames_by_channels(samples)
    return _loudness_range_from_reader(
        lambda start, frames: data[start:start + frames], data.shape[0], samplerate,
    )


def _dynamic_range_from_reader(
    reader: FrameReader, total_frames: int, channels: int, samplerate: int,
) -> float:
    block_frames = int(BLOCK_SECONDS * samplerate)
    n_blocks = total_frames // block_frames if block_frames > 0 else 0
    if n_blocks < 1:
        return 0.0

    rms_by_channel = [[] for _ in range(channels)]
    peaks_by_channel = [[] for _ in range(channels)]
    for block_index in range(n_blocks):
        block = reader(block_index * block_frames, block_frames).astype(np.float64)
        for channel in range(channels):
            values = block[:, channel]
            rms_by_channel[channel].append(float(np.sqrt(2.0 * np.mean(values ** 2))))
            peaks_by_channel[channel].append(float(np.max(np.abs(values))))

    channel_drs = []
    for rms_list, peak_list in zip(rms_by_channel, peaks_by_channel):
        top_count = max(1, int(np.ceil(0.2 * n_blocks)))
        rms_top20 = float(np.mean(sorted(rms_list, reverse=True)[:top_count]))
        peaks_sorted = sorted(peak_list, reverse=True)
        peak_second = peaks_sorted[1] if len(peaks_sorted) > 1 else peaks_sorted[0]
        if rms_top20 > 0 and peak_second > 0:
            channel_drs.append(20.0 * float(np.log10(peak_second / rms_top20)))

    return float(np.mean(channel_drs)) if channel_drs else 0.0


def dynamic_range_db(samples: np.ndarray, samplerate: int) -> float:
    """DR14アルゴリズム: 3秒ブロックのRMS上位20%平均とセカンドピークの比(dB)・チャンネル平均。"""
    data = _frames_by_channels(samples)
    return _dynamic_range_from_reader(
        lambda start, frames: data[start:start + frames],
        data.shape[0], data.shape[1], samplerate,
    )


def _scan_values(
    *, samplerate: int, loudness: float, sample_peak: float,
    peak_result: tuple[float, int, int], loudness_range: float, dynamic_range: float,
) -> dict:
    peak_dbtp, peak_position_sample, clipped = peak_result
    measured = bool(np.isfinite(loudness))
    plr = peak_dbtp - loudness if measured and np.isfinite(peak_dbtp) else 0.0
    return {
        "samplerate": samplerate,
        "loudness_lufs": loudness,
        "sample_peak_linear": sample_peak,
        "loudness_measured": measured,
        "track_gain_db": REFERENCE_LOUDNESS_LUFS - loudness if measured else 0.0,
        "loudness_range_lu": loudness_range,
        "true_peak_dbtp": peak_dbtp,
        "true_peak_position_sample": peak_position_sample,
        "clipped_samples": clipped,
        "plr_db": plr,
        "dynamic_range_db": dynamic_range,
    }


def scan_track(samples: np.ndarray, samplerate: int) -> dict:
    """D-8「再計算する」タグ群を1回でまとめて返す(トラック単位・アルバムスキャンはしない=I23)。"""
    loudness = integrated_loudness_lufs(samples, samplerate)
    return _scan_values(
        samplerate=samplerate, loudness=loudness,
        sample_peak=sample_peak_linear(samples),
        peak_result=true_peak(samples, samplerate),
        loudness_range=loudness_range_lu(samples, samplerate),
        dynamic_range=dynamic_range_db(samples, samplerate),
    )


def scan_track_file(
    path: Path, *, start: int = 0, frames: int | None = None, gain: float = 1.0,
) -> dict:
    """音声ファイルの指定範囲を全量ロードせず、トラック単位でscanする。

    `gain`は直列化時に宣言された固定倍率(2の冪)。再計算タグは「保存されたファイル」を
    説明しなければならないため、走査時点で同じ倍率をかけた値を測る。
    """
    if not (gain > 0.0) or not math.isfinite(gain):
        raise ValueError(f"scanの倍率が不正です: {gain}")
    with sf.SoundFile(path) as stream:
        available = len(stream)
        if start < 0 or start > available:
            raise ValueError(f"scan開始位置が音声範囲外です: start={start}, frames={available}")
        total_frames = available - start if frames is None else frames
        if total_frames < 0 or start + total_frames > available:
            raise ValueError(
                f"scan範囲が音声範囲外です: start={start}, frames={total_frames}, available={available}"
            )
        channels = stream.channels
        samplerate = stream.samplerate

        def read(relative_start: int, frame_count: int) -> np.ndarray:
            stream.seek(start + relative_start)
            data = stream.read(frame_count, dtype="float32", always_2d=True)
            if data.shape != (frame_count, channels) or not np.isfinite(data).all():
                raise ValueError(f"scan中に音声ファイルが変更されたか、不正なサンプルがあります: {path}")
            return data if gain == 1.0 else data * np.float32(gain)

        loudness = _integrated_loudness_from_reader(
            read, total_frames, channels, samplerate, np.float32,
        )
        return _scan_values(
            samplerate=samplerate, loudness=loudness,
            sample_peak=_sample_peak_from_reader(read, total_frames),
            peak_result=_true_peak_from_reader(
                read, total_frames, oversample=TRUE_PEAK_OVERSAMPLE,
            ),
            loudness_range=_loudness_range_from_reader(read, total_frames, samplerate),
            dynamic_range=_dynamic_range_from_reader(
                read, total_frames, channels, samplerate,
            ),
        )
