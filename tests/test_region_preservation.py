import numpy as np
import pytest

from lss.bgm import dialogue_residual as dr
from lss.separation import regions, selective


def test_tiled_weights_and_replacement_keep_outside_pcm_and_stereo_exact():
    rate, frames = 8000, 8000 * 12
    rng = np.random.default_rng(65)
    base = rng.normal(size=(frames, 2)).astype('float32') * 1e-6
    repair = base * [.1, .4]
    spans = [[4000, 13000], [42000, 75000]]
    full = regions.inside_weight(frames, spans, rate, .08, guard_seconds=.05)
    pieces = [regions.inside_weight(frames, spans, rate, .08, guard_seconds=.05,
                                    start=a, length=min(997, frames-a))
              for a in range(0, frames, 997)]
    assert np.array_equal(np.concatenate(pieces), full)
    before = base.copy()
    out = regions.blend(base, repair, full)
    assert np.array_equal(out[full == 0], base[full == 0])
    assert np.array_equal(out[full == 1], repair[full == 1])
    assert np.array_equal(base, before)
    assert np.max(np.abs(np.diff(full))) < .003


def test_bgm_speech_repair_cannot_change_adjacent_wanted_music():
    rate = 8000
    t = np.arange(rate * 6) / rate
    music = np.column_stack((.2*np.sin(2*np.pi*317*t), .1*np.sin(2*np.pi*619*t)))
    inst = music * .5
    spans = [[rate * 2, rate * 3]]
    out = dr.region_consensus(music, inst, spans, rate, dr.default_recipe())
    assert np.array_equal(out[:rate*2], music[:rate*2])
    assert np.array_equal(out[rate*3:], music[rate*3:])
    assert np.sum(out[rate*2+2000:rate*3-2000]**2) < np.sum(music[rate*2+2000:rate*3-2000]**2)


def test_removed_diffuse_noise_is_not_restored_even_under_strong_tones():
    rate = 8000
    t = np.arange(rate * 4) / rate
    wanted = (.2*np.column_stack((np.sin(2*np.pi*317*t), np.sin(2*np.pi*619*t)))).astype('float32')
    noise = np.random.default_rng(281).normal(size=wanted.shape)
    spectrum = np.fft.rfft(noise, axis=0)
    frequencies = np.fft.rfftfreq(len(noise), 1/rate)
    spectrum[(frequencies < 250) | (frequencies > 900)] = 0
    noise = np.fft.irfft(spectrum, n=len(noise), axis=0)
    noise *= .0001 / np.sqrt(np.mean(noise**2, axis=0))
    source, estimate = wanted + noise.astype('float32'), wanted + .5*noise.astype('float32')
    plan = selective.plan(source, estimate, rate, witness=estimate)
    assert not plan['original_regions']
    assert np.array_equal(selective.render(source, estimate, plan), estimate)


def test_saved_v5_plan_still_replays_without_replanning():
    rate = 8000
    t = np.arange(rate * 2) / rate
    x = np.column_stack((np.sin(2*np.pi*317*t), np.sin(2*np.pi*619*t))).astype('float32') * 1e-7
    y = x * .999
    plan = {'policy': selective.LEGACY_V5_POLICY, 'frames': len(x), 'sample_rate': rate,
            'original_regions': [[0, len(x)]], 'silent_regions': []}
    assert np.array_equal(selective.render(x, y, plan), x)


def test_collinear_removed_sound_is_not_restored_from_a_level_difference():
    rate = 8000
    t = np.arange(rate * 4) / rate
    wanted = (.2*np.column_stack((np.sin(2*np.pi*317*t), np.sin(2*np.pi*619*t)))).astype('float32')
    unwanted = .001*wanted
    source, estimate = wanted+unwanted, wanted+.5*unwanted
    plan = selective.plan(source, estimate, rate, witness=estimate)
    assert not plan['original_regions']
    assert np.array_equal(selective.render(source, estimate, plan), estimate)
    tiny = wanted*1e-7
    kept = tiny*.999
    plan = selective.plan(tiny, kept, rate)
    assert not plan['original_regions'] and not plan['silent_regions']
    assert np.array_equal(selective.render(tiny, kept, plan), kept)


def test_bgm_only_computes_needed_context_without_resetting_global_phase(monkeypatch):
    rng = np.random.default_rng(96)
    rate, frames = 44100, 44100 * 8 + 171
    music = rng.normal(size=(frames, 2)) * .01
    inst = music * [.7, .9] + rng.normal(size=music.shape) * .0005
    spans = [[rate+137, rate*2+371], [rate*5+25, rate*5+22050]]
    recipe = dr.default_recipe()
    expected = regions.blend(music, dr.consensus(music, inst, recipe),
                             regions.inside_weight(frames, spans, rate, recipe.local_ramp_seconds))
    original, seen = dr.consensus, []
    def observed(a, b, r):
        seen.append(len(a))
        return original(a, b, r)
    monkeypatch.setattr(dr, 'consensus', observed)
    actual = dr.region_consensus(music, inst, spans, rate, recipe)
    assert sum(seen) < frames * .3
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-17)
    w = regions.inside_weight(frames, spans, rate, recipe.local_ramp_seconds)
    assert np.array_equal(actual[w == 0], music[w == 0])


@pytest.mark.parametrize('spans', [[[0, 100], [90, 200]], [[True, 100]], [[0, 201]], [[0., 100]]])
def test_invalid_support_cannot_authorize_a_blend(spans):
    with pytest.raises(ValueError):
        regions.inside_weight(200, spans, 8000, .08)
