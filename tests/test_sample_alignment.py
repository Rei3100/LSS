import numpy as np
import pytest
import soundfile as sf

from lss.validation.sample_alignment import observe_files, observe_pair


def test_channel_specific_lag_is_not_hidden_by_dominant_channel():
    rng = np.random.default_rng(76)
    raw = rng.normal(size=(2, 4096))
    raw[1] *= 0.01
    candidate = raw.copy()
    candidate[1] = np.roll(candidate[1], 7)
    observed = observe_pair(raw, candidate)
    assert observed[0]["status"] == "observed_zero_lag"
    assert observed[1]["status"] == "observed_nonzero_lag"
    assert abs(observed[1]["best_lag_frames"]) == 7


def test_gain_difference_is_diagnostic_only_and_audio_is_unchanged():
    raw = np.random.default_rng(12).normal(size=(2, 4096))
    estimate = raw * 0.3
    copy = estimate.copy()
    assert all(row["status"] == "observed_zero_lag" for row in observe_pair(raw, estimate))
    np.testing.assert_array_equal(estimate, copy)


def test_constant_zero_and_repetitive_audio_are_unresolved_not_zero_lag_proof():
    assert all(row["status"] == "unresolved_no_time_information" for row in observe_pair(np.ones((2, 100)), np.ones((2, 100))))
    repeated = np.tile([1., -1.], (2, 100))
    assert all(row["status"] == "unresolved_ambiguous_peak" for row in observe_pair(repeated, repeated))


def test_unrelated_audio_does_not_claim_an_offset():
    rng = np.random.default_rng(13)
    assert all(row["status"] == "unresolved_low_correlation"
               for row in observe_pair(rng.normal(size=(2, 4096)), rng.normal(size=(2, 4096))))


def test_file_sum_and_declared_headroom_preserve_original_samples(tmp_path):
    raw = np.random.default_rng(34).normal(scale=0.1, size=(88200, 2))
    paths = [tmp_path / name for name in ("raw.wav", "music.wav", "speech.wav")]
    sf.write(paths[0], raw, 44100, subtype="DOUBLE")
    sf.write(paths[1], raw * 0.2, 44100, subtype="DOUBLE")
    sf.write(paths[2], raw * 0.6, 44100, subtype="DOUBLE")
    rows = observe_files(paths[0], paths[1:], gains=[2, 1])
    assert len(rows) == 5
    assert all(channel["status"] == "observed_zero_lag" for row in rows for channel in row["channels"])
    sf.write(paths[1], raw[:-1], 44100, subtype="DOUBLE")
    with pytest.raises(ValueError, match="timeline"):
        observe_files(paths[0], paths[1:])


@pytest.mark.parametrize("estimate", [np.ones((1, 100)), np.full((2, 100), np.nan), np.ones((2, 100), dtype=complex)])
def test_invalid_stereo_is_refused(estimate):
    with pytest.raises(ValueError):
        observe_pair(np.ones((2, 100)), estimate)
