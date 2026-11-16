import numpy as np
import pytest

from lss.bgm.cue_segmentation import (
    CueSegmentationError,
    CueSegmentationRecipe,
    segment_cues,
)

RATE = 22050


def _recipe(**overrides):
    values = {
        "n_fft": 2048,
        "stft_hop_frames": 512,
        "feature_seconds": 1.0,
        "feature_hop_seconds": 0.5,
        "band_count": 16,
        "low_hz": 55.0,
        "high_hz": 8000.0,
        "kernel_seconds": 3.0,
        "minimum_boundary_gap_seconds": 5.0,
        "novelty_threshold": 0.2,
        "chroma_weight": 0.5,
    }
    values.update(overrides)
    return CueSegmentationRecipe(**values)


def _tone(seconds, partials):
    time = np.arange(int(seconds * RATE)) / RATE
    value = sum(gain * np.sin(2 * np.pi * frequency * time) for frequency, gain in partials)
    return 0.2 * value


def _stereo(mono):
    return np.stack([mono, mono])


def test_finds_the_join_between_two_different_pieces():
    first = _tone(15.0, [(440.0, 1.0), (880.0, 0.5)])
    second = _tone(15.0, [(330.0, 1.0), (1650.0, 0.5)])

    result = segment_cues(_stereo(np.concatenate([first, second])), RATE, _recipe())

    assert len(result.boundaries) == 1
    assert abs(result.boundaries[0].seconds - 15.0) <= 1.0
    assert result.boundaries[0].novelty > 0.2


def test_reports_no_boundary_while_nothing_changes():
    result = segment_cues(_stereo(_tone(30.0, [(440.0, 1.0), (880.0, 0.5)])), RATE, _recipe())

    assert result.boundaries == ()
    assert max(result.novelty) < 1e-9


def test_keeps_the_declared_minimum_gap_between_boundaries():
    parts = [
        _tone(8.0, [(440.0, 1.0)]),
        _tone(2.0, [(330.0, 1.0)]),
        _tone(8.0, [(550.0, 1.0)]),
    ]
    signal = _stereo(np.concatenate(parts))

    close = segment_cues(signal, RATE, _recipe(minimum_boundary_gap_seconds=1.0,
                                               kernel_seconds=1.5, novelty_threshold=0.05))
    apart = segment_cues(signal, RATE, _recipe(minimum_boundary_gap_seconds=6.0,
                                               kernel_seconds=1.5, novelty_threshold=0.05))

    assert len(close.boundaries) >= 2
    assert len(apart.boundaries) == 1


def test_novelty_is_reported_as_an_observation_not_a_decision():
    result = segment_cues(
        _stereo(np.concatenate([_tone(15.0, [(440.0, 1.0)]), _tone(15.0, [(330.0, 1.0)])])),
        RATE, _recipe(novelty_threshold=1.0),
    )

    assert result.boundaries == ()
    assert max(result.novelty) > 0.2


def test_frozen_recipe_identity():
    assert _recipe().recipe_id.startswith("bgm-cue-segmentation-v1-")
    assert _recipe().recipe_id == _recipe().recipe_id
    assert _recipe().recipe_id != _recipe(chroma_weight=0.25).recipe_id


def test_a_stationary_signal_scores_zero_while_a_change_scores_high():
    """しきい値は信号ごとの最大値ではなく、共通の目盛りの上で意味を持つ。"""
    same = segment_cues(_stereo(_tone(30.0, [(440.0, 1.0)])), RATE, _recipe())
    changed = segment_cues(
        _stereo(np.concatenate([_tone(15.0, [(440.0, 1.0), (880.0, 0.5)]),
                                _tone(15.0, [(330.0, 1.0), (1650.0, 0.5)])])),
        RATE, _recipe(),
    )

    assert max(same.novelty) < 1e-9
    assert max(changed.novelty) > 0.2


@pytest.mark.parametrize("overrides", [
    {"n_fft": 3000},
    {"stft_hop_frames": 4096},
    {"feature_hop_seconds": 2.0},
    {"low_hz": 9000.0},
    {"novelty_threshold": 1.5},
    {"chroma_weight": -0.1},
    {"band_count": 0},
    {"method": "checkerboard"},
])
def test_rejects_unusable_declarations(overrides):
    with pytest.raises(CueSegmentationError):
        _recipe(**overrides).validate()


def test_rejects_an_interval_shorter_than_the_kernel():
    with pytest.raises(CueSegmentationError, match="短すぎます"):
        segment_cues(_stereo(_tone(4.0, [(440.0, 1.0)])), RATE, _recipe())


def test_rejects_a_band_above_the_nyquist_frequency():
    with pytest.raises(CueSegmentationError, match="Nyquist"):
        segment_cues(_stereo(_tone(30.0, [(440.0, 1.0)])), RATE, _recipe(high_hz=20000.0))


def test_rejects_non_finite_audio():
    values = _tone(30.0, [(440.0, 1.0)])
    values[100] = np.nan

    with pytest.raises(CueSegmentationError, match="有限"):
        segment_cues(_stereo(values), RATE, _recipe())
