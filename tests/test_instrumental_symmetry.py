import numpy as np
import pytest

from scripts.probe_instrumental_symmetry import TRANSFORMS, VARIANTS, combine, transform


@pytest.mark.parametrize("frames", [1, 2, 11, 401])
@pytest.mark.parametrize("name", TRANSFORMS)
def test_inverse_preserves_each_channel_sample_and_input(frames, name):
    source = np.arange(2 * frames).reshape(2, frames).astype(float)
    original = source.copy()
    result = transform(transform(source, name), name)
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(source, original)
    result[0, 0] = -123
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("members", VARIANTS.values())
def test_average_undoes_transform_before_combining_and_keeps_time(members):
    audio = np.zeros((2, 13))
    audio[0, 4] = 1.0
    audio[1, 8] = -0.5
    estimates = {name: transform(audio, name) for name in members}
    np.testing.assert_array_equal(combine(estimates, members), audio)


def test_asymmetric_errors_are_averaged_in_original_channel_order():
    audio = np.array([[1., 2., 3.], [-2., -4., -6.]])
    estimates = {"identity": audio.copy(), "swap": transform(audio, "swap")}
    estimates["swap"][0] += 0.2
    combined = combine(estimates, ("identity", "swap"))
    np.testing.assert_allclose(combined[0], audio[0])
    np.testing.assert_allclose(combined[1], audio[1] + 0.1)


def test_missing_or_duplicate_transforms_or_timeline_are_refused():
    estimates = {"identity": np.ones((2, 4))}
    for members in ((), ("identity", "identity"), ("identity", "negative")):
        with pytest.raises(ValueError):
            combine(estimates, members)
    estimates["negative"] = np.ones((2, 5))
    with pytest.raises(ValueError, match="時間軸"):
        combine(estimates, ("identity", "negative"))


@pytest.mark.parametrize("values", [np.ones((1, 4)), np.empty((2, 0)), np.full((2, 4), np.nan),
                                   np.full((2, 4), np.inf), np.ones((2, 4), dtype=complex)])
def test_invalid_audio_is_refused(values):
    with pytest.raises(ValueError):
        transform(values, "identity")
