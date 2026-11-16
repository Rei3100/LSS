from dataclasses import replace
import json

import numpy as np
import pytest

from lss.bgm.routing import DisagreementRoutingRecipe, analyze_stereo, route_stereo


def recipe():
    return DisagreementRoutingRecipe(window_frames=64, hop_frames=32)


def stems(music, speech=None):
    return {"Music": music, "Speech": np.zeros_like(music) if speech is None else speech,
            "Effects": np.zeros_like(music)}


@pytest.mark.parametrize("frames", [1, 31, 129, 513])
def test_agreement_keeps_instrumental_exactly(frames):
    parent = np.random.default_rng(8).normal(0, 0.1, (2, frames))
    result = route_stereo(parent, stems(parent * 0.95), recipe())
    np.testing.assert_array_equal(result.audio, parent)
    assert np.count_nonzero(result.music_weights) == 0
    assert json.loads(json.dumps(result.windows))


def test_removed_component_switches_to_music_without_reference():
    music = np.random.default_rng(2).normal(0, 0.1, (2, 513))
    parent = music * 0.1
    copies = parent.copy(), music.copy()
    result = route_stereo(parent, stems(music), recipe())
    np.testing.assert_array_equal(result.audio, music)
    np.testing.assert_array_equal(result.music_weights, np.ones(513))
    np.testing.assert_array_equal(parent, copies[0])
    np.testing.assert_array_equal(music, copies[1])


def test_low_music_share_does_not_switch_from_dialogue_disagreement():
    music = np.ones((2, 129)) * 0.01
    parent = music * 0.1
    result = route_stereo(parent, stems(music, np.ones_like(music)), recipe())
    np.testing.assert_array_equal(result.audio, parent)
    assert all(not row["use_music"] for row in result.windows)


def test_silence_is_not_hallucinated_and_single_sample_is_supported():
    silent = np.zeros((2, 1))
    result = route_stereo(silent, stems(silent), recipe())
    np.testing.assert_array_equal(result.audio, silent)
    assert result.windows[0]["disagreement_share"] == 0


def test_switch_interpolation_is_common_to_both_channels_and_convex():
    t = np.arange(513)
    music = np.vstack((np.sin(t * 0.2), np.cos(t * 0.3))) * 0.1
    parent = music.copy()
    parent[:, 250:] *= 0.1
    result = route_stereo(parent, stems(music), recipe())
    assert np.all((result.music_weights >= 0) & (result.music_weights <= 1))
    assert np.any((result.music_weights > 0) & (result.music_weights < 1))
    expected = parent * (1 - result.music_weights) + music * result.music_weights
    np.testing.assert_array_equal(result.audio, expected)


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"hop_frames": True}, {"hop_frames": 65},
    {"minimum_music_share": float("nan")}, {"minimum_music_share": 1.01},
    {"minimum_music_rms": 0}, {"minimum_disagreement_share": True},
    {"transition": "independent-channel"},
])
def test_invalid_rules_fail_before_audio_processing(change):
    with pytest.raises(ValueError):
        replace(recipe(), **change).validate()


@pytest.mark.parametrize("bad", [np.zeros((1, 17)), np.zeros((2, 0)),
                                  np.full((2, 17), np.nan), np.ones((2, 17), dtype=complex)])
def test_invalid_audio_is_refused(bad):
    with pytest.raises(ValueError):
        route_stereo(bad, stems(bad), recipe())


def test_missing_stem_or_different_timeline_is_refused():
    parent = np.zeros((2, 17))
    with pytest.raises(ValueError, match="全推定"):
        route_stereo(parent, {"Music": parent}, recipe())
    with pytest.raises(ValueError, match="timeline"):
        route_stereo(parent, stems(np.zeros((2, 18))), recipe())


def test_every_output_affecting_rule_has_distinct_identity():
    parent = recipe()
    for change in ({"window_frames": 128}, {"hop_frames": 16}, {"minimum_music_share": 0.3},
                   {"minimum_disagreement_share": 0.2}, {"minimum_music_rms": 0.002}):
        assert replace(parent, **change).recipe_id != parent.recipe_id


@pytest.mark.parametrize("block_frames", [1, 7, 64, 513])
def test_energy_block_boundaries_do_not_change_observations_or_audio(monkeypatch, block_frames):
    import lss.bgm.routing as routing

    # 左右の桁が異なり、累積和の丸めがブロック順序に敏感な入力。
    rng = np.random.default_rng(42)
    music = rng.normal(size=(2, 513)) * np.exp(rng.uniform(-12, 5, size=(2, 513)))
    parent = music.copy()
    parent[:, 200:320] *= 0.1
    inputs = stems(music, rng.normal(0, 0.1, size=music.shape))
    monkeypatch.setattr(routing, "ENERGY_BLOCK_FRAMES", music.shape[1])
    whole = route_stereo(parent, inputs, recipe())
    monkeypatch.setattr(routing, "ENERGY_BLOCK_FRAMES", block_frames)
    blocked = route_stereo(parent, inputs, recipe())
    assert blocked.windows == whole.windows
    assert analyze_stereo(parent, inputs, recipe()) == whole.windows
    np.testing.assert_array_equal(blocked.music_weights, whole.music_weights)
    np.testing.assert_array_equal(blocked.audio, whole.audio)


def test_energy_overflow_remains_a_hard_failure():
    parent = np.full((2, 17), 1e300)
    with pytest.raises(ValueError, match="有限範囲"):
        analyze_stereo(parent, stems(parent), recipe())
