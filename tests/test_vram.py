import torch

from lss.harness import vram
from lss.harness.vram import peak_vram_mb, reset_peak_vram


def test_peak_vram_mb_is_zero_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert peak_vram_mb() == 0.0


def test_reset_peak_vram_does_not_raise_without_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reset_peak_vram()


def test_release_unused_vram_collects_objects_and_releases_cuda_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(vram.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    vram.release_unused_vram()

    assert calls == ["gc", "cuda"]
