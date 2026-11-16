import numpy as np
import pytest

from lss.output import scan


def _sine(seconds: float, samplerate: int, freq: float = 440.0, amplitude: float = 0.5, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * samplerate)) / samplerate
    wave = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    return np.tile(wave, (channels, 1))


def test_sample_peak_linear_matches_max_abs():
    samples = _sine(1.0, 44100, amplitude=0.7)
    assert scan.sample_peak_linear(samples) == pytest.approx(0.7, abs=1e-6)


def test_sample_peak_linear_empty_is_zero():
    assert scan.sample_peak_linear(np.zeros((2, 0))) == 0.0


def test_integrated_loudness_louder_signal_is_higher():
    quiet = _sine(5.0, 44100, amplitude=0.1)
    loud = _sine(5.0, 44100, amplitude=0.5)
    assert scan.integrated_loudness_lufs(loud, 44100) > scan.integrated_loudness_lufs(quiet, 44100)


def test_true_peak_is_at_least_sample_peak():
    samples = _sine(1.0, 44100, amplitude=0.9)
    peak_dbtp, position, clipped = scan.true_peak(samples, 44100)
    sample_peak_db = 20 * np.log10(scan.sample_peak_linear(samples))
    assert peak_dbtp >= sample_peak_db - 1e-6
    assert 0 <= position < samples.shape[1]
    assert clipped == 0


def test_true_peak_detects_clipping_at_full_scale():
    samples = np.ones((2, 1000), dtype=np.float64)
    _, _, clipped = scan.true_peak(samples, 44100)
    oversampled = scan.resample_poly(samples.T, scan.TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    expected = round(np.sum(np.abs(oversampled) >= 1.0) / scan.TRUE_PEAK_OVERSAMPLE)
    assert clipped == expected


def test_true_peak_clipped_count_includes_each_channel():
    left_only = np.vstack([np.ones(1000), np.zeros(1000)])
    stereo = np.ones((2, 1000))

    left_count = scan.true_peak(left_only, 44100)[2]
    stereo_count = scan.true_peak(stereo, 44100)[2]

    assert left_count > 0
    assert stereo_count in {left_count * 2 - 1, left_count * 2}


def test_true_peak_empty_returns_negative_infinity():
    peak_dbtp, position, clipped = scan.true_peak(np.zeros((2, 0)), 44100)
    assert peak_dbtp == float("-inf")
    assert position == 0
    assert clipped == 0


def test_true_peak_chunks_long_audio_with_filter_overlap(monkeypatch):
    samplerate = 44100
    timeline = np.arange(5000) / samplerate
    samples = np.vstack([
        0.91 * np.sin(2 * np.pi * 997 * timeline),
        0.73 * np.sin(2 * np.pi * 1301 * timeline),
    ]).astype(np.float32)
    from scipy.signal import resample_poly as reference_resample

    full = reference_resample(samples.T, scan.TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    expected_peak = float(np.max(np.abs(full)))
    seen_lengths = []

    def bounded_resample(data, *args, **kwargs):
        seen_lengths.append(data.shape[0])
        return reference_resample(data, *args, **kwargs)

    monkeypatch.setattr(scan, "TRUE_PEAK_BLOCK_FRAMES", 1000)
    monkeypatch.setattr(scan, "resample_poly", bounded_resample)

    peak_dbtp, position, clipped = scan.true_peak(samples, samplerate)

    assert len(seen_lengths) == 5
    assert max(seen_lengths) <= 1000 + 2 * scan.TRUE_PEAK_OVERLAP_FRAMES
    assert 10 ** (peak_dbtp / 20) == pytest.approx(expected_peak, rel=1e-5)
    assert 0 <= position < samples.shape[1]
    assert clipped == 0


def test_true_peak_chunked_clipped_count_matches_full_resample(monkeypatch):
    samples = np.ones((2, 5000), dtype=np.float32) * 1.2
    full = scan.resample_poly(samples.T, scan.TRUE_PEAK_OVERSAMPLE, 1, axis=0)
    expected = round(np.sum(np.abs(full) >= 1.0) / scan.TRUE_PEAK_OVERSAMPLE)
    monkeypatch.setattr(scan, "TRUE_PEAK_BLOCK_FRAMES", 1000)

    assert scan.true_peak(samples, 44100)[2] == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_streaming_integrated_loudness_matches_pyloudnorm(dtype, monkeypatch):
    samplerate = 44100
    samples = _sine(2.37, samplerate, freq=997, amplitude=0.37).astype(dtype)
    expected = scan.pyln.Meter(samplerate).integrated_loudness(samples.T)
    seen_lengths = []
    original_lfilter = scan.lfilter

    def bounded_lfilter(data, *args, **kwargs):
        seen_lengths.append(data.shape[0])
        return original_lfilter(data, *args, **kwargs)

    monkeypatch.setattr(scan, "LOUDNESS_FILTER_BLOCK_FRAMES", 10000)
    monkeypatch.setattr(scan, "lfilter", bounded_lfilter)

    actual = scan.integrated_loudness_lufs(samples, samplerate)

    assert len(seen_lengths) > 4
    assert max(seen_lengths) <= 10000
    assert actual == pytest.approx(expected, abs=2e-5)


def test_loudness_range_short_signal_is_zero():
    samples = _sine(1.0, 44100)  # 3秒ブロックに満たない
    assert scan.loudness_range_lu(samples, 44100) == 0.0


def test_loudness_range_varying_amplitude_is_positive():
    samplerate = 44100
    quiet = _sine(4.0, samplerate, amplitude=0.05)
    loud = _sine(4.0, samplerate, amplitude=0.8)
    samples = np.concatenate([quiet, loud, quiet], axis=1)
    assert scan.loudness_range_lu(samples, samplerate) >= 0.0


def test_dynamic_range_short_signal_is_zero():
    samples = _sine(1.0, 44100)
    assert scan.dynamic_range_db(samples, 44100) == 0.0


def test_dynamic_range_is_positive_for_varying_amplitude_signal():
    samplerate = 44100
    quiet = _sine(4.0, samplerate, amplitude=0.02)
    loud = _sine(4.0, samplerate, amplitude=0.9)
    samples = np.concatenate([quiet, loud, quiet], axis=1)
    assert scan.dynamic_range_db(samples, samplerate) > 0.0


def test_scan_track_returns_all_required_keys():
    samples = _sine(9.0, 44100, amplitude=0.5)
    result = scan.scan_track(samples, 44100)
    expected_keys = {
        "samplerate", "loudness_lufs", "sample_peak_linear", "track_gain_db",
        "loudness_range_lu", "true_peak_dbtp", "true_peak_position_sample",
        "clipped_samples", "plr_db", "dynamic_range_db",
    }
    assert expected_keys <= result.keys()


def test_scan_track_file_matches_trimmed_array_without_full_read(tmp_path, monkeypatch):
    samplerate = 44100
    samples = _sine(9.0, samplerate, freq=997, amplitude=0.37).astype(np.float32)
    path = tmp_path / "stem.wav"
    sf_data = samples.T
    scan.sf.write(path, sf_data, samplerate, subtype="FLOAT")
    start = 1234
    frames = samples.shape[1] - start - 2345
    expected = scan.scan_track(samples[:, start:start + frames], samplerate)
    real_sound_file = scan.sf.SoundFile
    reads = []

    class BoundedSoundFile:
        def __init__(self, *args, **kwargs):
            self.stream = real_sound_file(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __len__(self):
            return len(self.stream)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, frame_count, *args, **kwargs):
            reads.append(frame_count)
            return self.stream.read(frame_count, *args, **kwargs)

    monkeypatch.setattr(scan.sf, "SoundFile", BoundedSoundFile)
    actual = scan.scan_track_file(path, start=start, frames=frames)

    assert reads
    assert max(reads) <= max(
        scan.FILE_SCAN_BLOCK_FRAMES,
        scan.TRUE_PEAK_BLOCK_FRAMES + 2 * scan.TRUE_PEAK_OVERLAP_FRAMES,
        int(scan.BLOCK_SECONDS * samplerate),
    )
    for key in expected:
        if isinstance(expected[key], float):
            assert actual[key] == pytest.approx(expected[key], abs=2e-5)
        else:
            assert actual[key] == expected[key]


def test_scan_track_file_rejects_range_past_end(tmp_path):
    path = tmp_path / "stem.wav"
    scan.sf.write(path, np.zeros((100, 2), dtype=np.float32), 44100)

    with pytest.raises(ValueError, match="scan範囲"):
        scan.scan_track_file(path, start=50, frames=51)


@pytest.mark.parametrize("samples", [_sine(0.1, 44100), np.zeros((2, 44100))])
def test_unmeasurable_loudness_does_not_invent_amplification(samples):
    from lss.output.tags import recalculated_tags

    result = scan.scan_track(samples, 44100)
    assert result["loudness_measured"] is False
    assert result["track_gain_db"] == 0.0
    tags = recalculated_tags(result)
    assert "REPLAYGAIN_TRACK_GAIN" not in tags
    assert "TRUEPEAK_SCANNER_TRACK_PLR" not in tags
    assert "REPLAYGAIN_TRACK_PEAK" in tags
