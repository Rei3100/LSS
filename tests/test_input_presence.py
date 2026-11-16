"""小さな音を消さず、全サンプルの有限値と厳密な無音だけを判定する。"""
import numpy as np
import pytest
import soundfile as sf

from lss.core import presence


def test_exact_silence_has_auditable_zero_evidence(tmp_path, monkeypatch):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros((1001, 2)), 48000, subtype="FLOAT")
    monkeypatch.setattr(presence, "SCAN_BLOCK_FRAMES", 63)
    result = presence.inspect_signal(path)
    assert result.exactly_silent
    assert result.frames == 1001 and result.sample_rate == 48000 and result.channels == 2
    zero = result.zero_result()
    assert zero["validation_result"]["hard_gates_passed"] is True
    assert zero["reason"] == "no_wanted_target"
    assert zero["validation_result"]["evidence"]["checked_frames"] == 1001


@pytest.mark.parametrize("value", [1e-6, -1e-6, np.nextafter(np.float32(0), np.float32(1))])
def test_even_one_tiny_sample_is_not_absence(tmp_path, value):
    path = tmp_path / "quiet.wav"
    samples = np.zeros((1000, 2), dtype=np.float32)
    samples[-1, 1] = value
    sf.write(path, samples, 44100, subtype="FLOAT")
    result = presence.inspect_signal(path)
    assert not result.exactly_silent
    with pytest.raises(ValueError):
        result.zero_result()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_samples_after_valid_audio_are_rejected(tmp_path, monkeypatch, value):
    path = tmp_path / "broken.wav"
    samples = np.ones((1000, 2), dtype=np.float32) * 0.1
    samples[-1, 1] = value
    sf.write(path, samples, 44100, subtype="FLOAT")
    monkeypatch.setattr(presence, "SCAN_BLOCK_FRAMES", 71)
    with pytest.raises(presence.InvalidInputSignalError, match="NaN|無限大"):
        presence.inspect_signal(path)


def test_unreadable_tail_is_not_accepted_as_silence(tmp_path, monkeypatch):
    path = tmp_path / "truncated.wav"
    sf.write(path, np.zeros((1000, 2)), 44100)
    read = sf.SoundFile.read

    def truncated(audio, count, **kwargs):
        return read(audio, min(count, 100), **kwargs) if audio.tell() < 100 else np.empty((0, 2))

    monkeypatch.setattr(sf.SoundFile, "read", truncated)
    with pytest.raises(presence.InvalidInputSignalError, match="最後まで"):
        presence.inspect_signal(path)


