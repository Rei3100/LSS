"""分離の作業物と最終成果の、段階ごとの最大容量を推論前に確認する。"""
from __future__ import annotations

import math
import shutil
from fractions import Fraction
from pathlib import Path

from lss.output import writer


def _existing_directory(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        parent = path.parent
        if parent == path:
            raise FileNotFoundError(f"保存領域を確認できません: {path}")
        path = parent
    if not path.is_dir():
        raise NotADirectoryError(f"保存領域がディレクトリではありません: {path}")
    return path


def _volume(path: Path) -> int:
    return path.stat().st_dev


def required_capacity_by_volume(
    work_root: Path, destination: Path, *, input_copy_bytes: int,
    raw_stems_bytes: int, final_outputs_bytes: int,
) -> dict[int, tuple[Path, int]]:
    """入力コピーは公開前に消えるため、同時に存在する各段階の最大値を使う。"""
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in (input_copy_bytes, raw_stems_bytes, final_outputs_bytes)):
        raise ValueError("容量は0以上の整数である必要があります")
    work, output = _existing_directory(work_root), _existing_directory(destination)
    work_volume, output_volume = _volume(work), _volume(output)
    if work_volume == output_volume:
        return {work_volume: (work, raw_stems_bytes + max(input_copy_bytes, final_outputs_bytes))}
    return {
        work_volume: (work, raw_stems_bytes + input_copy_bytes),
        output_volume: (output, final_outputs_bytes),
    }


def ensure_processing_capacity(
    prepared_input: Path, work_root: Path, destination: Path, *,
    source_frames: int, source_samplerate: int, output_samplerate: int,
    channels: int, generated_stem_count: int, final_outputs_bytes: int,
) -> dict[int, tuple[Path, int]]:
    """MSSTの入力コピー・FLOAT WAV全内部stem・最終FLACの併存を計上する。

    復号済み入力はすでにfree容量から引かれている。公開対象外の相補stemも、分離時に
    実際に作られるので作業容量に含める。容量不足で品質設定を下げる経路は持たない。
    """
    values = (source_frames, source_samplerate, output_samplerate, channels, generated_stem_count)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("分離容量の見積もり条件が不正です")
    # 整数演算でceilを求め、長尺やレート変換で必要sample数を過小評価しない。
    frames = (source_frames * output_samplerate + source_samplerate - 1) // source_samplerate
    raw_pcm = frames * channels * generated_stem_count * 4
    raw_bytes = math.ceil(raw_pcm * Fraction(str(writer.OUTPUT_HEADROOM_RATIO))) + generated_stem_count * writer.OUTPUT_OVERHEAD_BYTES_PER_STEM
    required = required_capacity_by_volume(
        work_root, destination, input_copy_bytes=prepared_input.stat().st_size,
        raw_stems_bytes=raw_bytes, final_outputs_bytes=final_outputs_bytes,
    )
    for directory, needed in required.values():
        free = shutil.disk_usage(directory).free
        if free < needed:
            raise writer.InsufficientOutputSpaceError(
                f"分離・保存に必要な空き容量が不足しています: path={directory}, "
                f"free_bytes={free}, needed_bytes={needed}"
            )
    return required
