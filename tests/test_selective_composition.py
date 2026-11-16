from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from lss.separation import selective as s


def music(rate=8000, seconds=4, gain=.2):
    t = np.arange(round(rate * seconds)) / rate
    return (gain * np.column_stack((np.sin(2*np.pi*317*t), np.sin(2*np.pi*619*t)))).astype('float32')


def test_clean_original_preserves_tiny_notes_and_stereo_without_normalization():
    x = music(gain=1e-7)
    x[:, 1] *= .02
    estimate = x + .0005*np.gradient(x, axis=0)
    p = s.plan(x, estimate, 8000)
    assert p['original_regions'] == [[0, len(x)]] and not p['silent_regions']
    assert np.array_equal(s.render(x, estimate, p), x)


def test_voice_over_quiet_music_is_uncertain_and_never_muted():
    target = music(gain=1e-7)
    source = music(gain=.2) + target
    p = s.plan(source, target, 8000)
    assert not p['silent_regions'] and not p['original_regions']
    assert np.array_equal(s.render(source, target, p), target)


def test_stereo_transient_and_quiet_high_frequency_contaminants_veto_raw():
    x = music()
    estimate = x.copy()
    # A 2ms contaminant must not disappear in a quarter-second average.
    x[4000:4016, 1] += .1
    p = s.plan(x, estimate, 8000)
    assert all(not (low <= 4000 < high) for low, high in p['original_regions'])
    # A quiet high instrument missing from the estimate is uncertain, rather
    # than proof that all disagreement is unwanted material.
    t = np.arange(len(x))/8000
    x = music(gain=.2)
    x[:, 1] += (.001*np.sin(2*np.pi*2900*t)).astype('float32')
    p = s.plan(x, music(gain=.2), 8000)
    assert not p['original_regions']


def test_clean_islands_and_zero_core_have_inside_fades_and_exact_outside_pcm():
    x = music(seconds=6)
    y = x + .0005*np.gradient(x, axis=0)
    x[16000:24000] += music(seconds=1, gain=.2)
    x[32000:40000] = 0
    y[32000:40000] = .0001
    p = s.plan(x, y, 8000)
    assert p['original_regions'] and p['silent_regions'] == [[32000, 40000]]
    out = s.render(x, y, p)
    w = np.maximum(s.weight(len(x), p['original_regions'], 8000), s.weight(len(x), p['silent_regions'], 8000))
    assert np.array_equal(out[w == 0], y[w == 0])
    assert np.array_equal(out[w == 1], x[w == 1])
    assert np.all(out[34000:38000] == 0)
    # The smooth switch has no extra single-sample jump relative to the music.
    assert np.max(np.abs(np.diff(w))) < .003


@pytest.mark.parametrize('native,rate', [(48000,44100), (44100,48000), (8000,8000)])
def test_bounded_alignment_has_global_phase_and_keeps_fractional_tail(tmp_path, native, rate):
    x = music(native, seconds=1.137)
    path = tmp_path/'source.wav'
    sf.write(path, x, native, subtype='FLOAT')
    from math import gcd
    d = gcd(native, rate)
    expected = resample_poly(x, rate//d, native//d, axis=0) if native != rate else x
    pieces = [s.aligned_read(path, rate, a, min(997, len(expected)-a))
              for a in range(0, len(expected), 997)]
    assert np.array_equal(np.concatenate(pieces), expected)


def test_staging_and_replay_match_across_tile_edges_and_preserve_inputs(tmp_path):
    x = music(seconds=19.13)
    y = x + .0005*np.gradient(x, axis=0)
    y[8000*7:8000*9] *= .3
    x[8000*15:8000*16] = 0
    source, estimate = tmp_path/'source.wav', tmp_path/'estimate.wav'
    sf.write(source, x, 8000, subtype='FLOAT')
    sf.write(estimate, y, 8000, subtype='FLOAT')
    before = (source.read_bytes(), estimate.read_bytes())
    receipt = s.process(source, estimate, tmp_path/'composed.wav')
    data, rate = sf.read(receipt['staged_path'], dtype='float32', always_2d=True)
    assert np.array_equal(data, s.render(x, y, receipt))
    assert before == (source.read_bytes(), estimate.read_bytes())
    assert data.shape == x.shape and rate == 8000


def test_invalid_or_incomplete_signal_cannot_publish_and_forged_absence_fails(tmp_path):
    x = music()
    with pytest.raises(ValueError, match='aligned'):
        s.plan(x, x[:-1], 8000)
    bad = x.copy(); bad[1, 1] = np.nan
    with pytest.raises(ValueError, match='aligned'):
        s.plan(x, bad, 8000)
    p = s.plan(x, x, 8000)
    p.update(original_regions=[], silent_regions=[[0, len(x)]])
    with pytest.raises(ValueError, match='source-zero'):
        s.render(x, x, p)


def test_semantic_speech_veto_cannot_be_overridden_by_separator_agreement():
    x = music()
    p = s.plan(x, x, 8000, allowed=np.zeros(16, dtype=bool))
    assert not p['original_regions']
    import json
    mixed = np.ones(16, dtype=bool); mixed[4:8] = False
    p = s.plan(x, x, 8000, allowed=mixed)
    assert p['original_regions']
    assert json.loads(json.dumps(p)) == p


def test_quiet_contaminant_in_a_strong_instrument_band_is_not_restored():
    wanted = music()
    t = np.arange(len(wanted))/8000
    contaminant = (.001*np.sin(2*np.pi*710*t))[:,None].astype('float32')
    source = wanted+contaminant
    estimate = wanted+.5*contaminant
    p = s.plan(source, estimate, 8000)
    assert not p['original_regions']
    assert np.array_equal(s.render(source, estimate, p), estimate)


def test_original_cannot_undo_successful_later_effect_cleanup():
    wanted = music()
    t = np.arange(len(wanted))/8000
    unwanted = (.05*np.sin(2*np.pi*710*t))[:,None].astype('float32')
    source = wanted+unwanted
    # The first separator retains an effect that a later stage removes.
    # Comparing only that first estimate would restore the entire unwanted tone.
    p = s.plan(source, source, 8000, witness=wanted)
    assert not p['original_regions']
    assert p['original_witness'] == 'first_and_selected_processed'
    assert np.array_equal(s.render(source, wanted, p), wanted)
    with pytest.raises(ValueError, match='aligned'):
        s.plan(source, source, 8000, witness=wanted[:-1])
