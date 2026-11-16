"""ピークVRAM計測(D-5 L5・T320要件4)。torch.cuda.max_memory_allocatedの薄いラッパー。"""
from __future__ import annotations

import gc

import torch


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def release_unused_vram() -> None:
    """推論モデルの参照を回収し、PyTorchの未使用CUDA予約領域をOS側へ返す。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
