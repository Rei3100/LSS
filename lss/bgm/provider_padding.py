"""短いstereo入力でも、providerの反射境界と元のsample位置を保つ。"""
from __future__ import annotations


def reflect_pad_last_axis(audio, padding: tuple[int, int]):
    """任意長の反射padding。端点を重複しない周期2*(N-1)で定義する。"""
    import torch

    if (
        not isinstance(padding, tuple) or len(padding) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in padding)
        or audio.ndim < 1 or audio.shape[-1] <= 0
    ):
        raise ValueError("空でない音声と0以上の整数paddingが必要です")
    left, right = padding
    length = audio.shape[-1]
    if length == 1:
        return audio.expand(*audio.shape[:-1], left + 1 + right).clone()
    period = 2 * (length - 1)
    positions = torch.arange(-left, length + right, device=audio.device) % period
    indices = torch.where(positions < length, positions, period - positions)
    return audio.index_select(-1, indices)


def padding_revision() -> dict:
    return {"schema_version": 1, "method": "periodic_reflection_without_duplicate_endpoints",
            "single_sample_policy": "repeat_single_sample", "original_samples": "unchanged",
            "provider_chunk_hop_batch": "unchanged", "output_timeline": "original_frames_only"}
