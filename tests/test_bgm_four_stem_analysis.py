"""4分類は原本の絶対時刻に固定したchunk格子で区間だけを推論し、切り方で出力を変えない。"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import four_stem_analysis as fs

RATE = fs.SAMPLE_RATE
_REAL = fs.FourStemSeparator


class _Model:
    """chunkごとに内容が変わる非線形の偽model（chunkの位置と文脈に依存する出力）。"""

    def __init__(self):
        self.batches = 0

    def __call__(self, pieces):
        import torch

        self.batches += 1
        scale = 1.0 + pieces.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return {role: pieces * scale * (index + 1) for index, role in enumerate(fs.ROLES)}


def _separator(hop_seconds=3.0):
    import torch

    separator = _REAL.__new__(_REAL)
    separator._torch = torch
    separator.device = "cpu"
    separator.config = fs.inference_config(hop_seconds)
    separator.model = _Model()
    separator.window = torch.hann_window(separator.config["chunk_frames"], periodic=True, dtype=torch.float32)
    return separator


def _audio(seconds=40):
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.1, (seconds * RATE, 2)).astype(np.float32)


def test_a_region_is_the_same_whatever_the_crop_or_neighbouring_regions():
    audio = _audio()
    separator = _separator()
    inner = (int(12.3 * RATE), int(17.9 * RATE))

    alone = separator.separate(audio, *inner, roles=("music_vocals",))["music_vocals"]
    wider = separator.separate(audio, int(8.0 * RATE), int(25.0 * RATE), roles=("music_vocals",))["music_vocals"]

    offset = inner[0] - int(8.0 * RATE)
    np.testing.assert_allclose(alone, wider[offset:offset + alone.shape[0]], atol=1e-6)


def test_an_identity_model_reconstructs_the_input_including_the_file_start():
    audio = _audio(20)
    separator = _separator(1.5)
    separator.model = lambda pieces: {role: pieces for role in fs.ROLES}

    out = separator.separate(audio, 0, 9 * RATE, roles=("music_vocals",))["music_vocals"]

    np.testing.assert_allclose(out, audio[:9 * RATE], atol=1e-6)


def test_the_grid_is_anchored_to_source_time():
    config = fs.inference_config(3.0)
    starts = fs.grid_chunks(10 * RATE, 13 * RATE, config)

    assert all(start % config["hop_frames"] == 0 for start in starts)
    assert starts[0] < 10 * RATE - config["chunk_frames"] + config["hop_frames"] + 1
    assert starts[-1] < 13 * RATE <= starts[-1] + config["hop_frames"]
    with pytest.raises(fs.FourStemAnalysisError):
        fs.inference_config(2.5)


def test_spans_are_written_as_their_own_files(tmp_path, monkeypatch):
    signal = tmp_path / "effects.wav"
    sf.write(str(signal), _audio(20), RATE, subtype="FLOAT")
    monkeypatch.setattr(fs, "FourStemSeparator", lambda weights, hop, device="cuda:0": _separator(hop))

    result = fs.separate_spans(signal, tmp_path / "out", tmp_path / "w.ckpt", [[RATE, 3 * RATE], [5 * RATE, 6 * RATE]],
                               hop_seconds=3.0)

    items = result["stems"]["music_vocals"]
    assert [item["span"] for item in items] == [[RATE, 3 * RATE], [5 * RATE, 6 * RATE]]
    assert [sf.info(item["path"]).frames for item in items] == [2 * RATE, RATE]
    with pytest.raises(fs.FourStemAnalysisError):
        fs.separate_spans(signal, tmp_path / "bad", tmp_path / "w.ckpt", [[2 * RATE, RATE]], hop_seconds=3.0)
    with pytest.raises(fs.FourStemAnalysisError):
        fs.separate_spans(signal, tmp_path / "bad", tmp_path / "w.ckpt", [[0, RATE]], hop_seconds=3.0,
                          roles=("drums",))


def test_the_revision_follows_the_hop_and_the_pinned_weights():
    assert fs.analyzer_revision(3.0)["analyzer_id"] != fs.analyzer_revision(1.5)["analyzer_id"]
    revision = fs.analyzer_revision(3.0)
    assert revision["weights"]["sha256"] == fs.WEIGHTS["sha256"]
    assert revision["weights"]["license"] == "CC-BY-NC-4.0"
    assert any(item["path"].endswith("bandsplit.py") for item in revision["implementation"])
