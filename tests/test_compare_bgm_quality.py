import numpy as np
import pytest

from scripts.compare_bgm_quality import load_asset, measure_case, paired_decision, spectral_vector, spectrum, summarize
from scripts.research_tests import digest


def test_summary_preserves_local_failure_even_when_whole_row_passes():
    rows = [{'id': 'a', 'decision': {'reasons': []}, 'windows': [
        {'start_seconds': 0, 'decision': {'reasons': [], 'delta_retention_db': 0}},
        {'start_seconds': 3, 'decision': {'reasons': ['catastrophic_wanted_loss'], 'delta_retention_db': -9}}]},
        {'id': 'b', 'decision': {'reasons': ['wanted_loss']}, 'windows': [
            {'start_seconds': 0, 'decision': {'reasons': ['wanted_loss'], 'delta_retention_db': -2}},
            {'start_seconds': 3, 'decision': {'reasons': [], 'delta_retention_db': None}}]}]
    s = summarize(rows)
    assert s == {'rows': 2, 'whole_row_harms': 1, 'rows_with_local_harm': 2,
                 'windows': 4, 'harmful_windows': 2, 'catastrophic_windows': 1,
                 'worst_retention': {'id': 'a', 'start_seconds': 3, 'delta_retention_db': -9},
                 'disposition': 'REJECT_BLANKET_REPLACEMENT'}
    assert summarize([])['worst_retention'] is None
    assert summarize([{'id': 'x', 'decision': {'reasons': []}, 'windows': []}])['disposition'] == 'HOLD'


def test_silencing_improves_bleed_but_cannot_pass_retention():
    rng = np.random.default_rng(42)
    t = rng.normal(0, .1, (44100*3, 2))
    x = t + rng.normal(0, .005, t.shape)
    st, sx = spectrum(t), spectrum(x)
    major = np.zeros(st.shape, dtype=bool)
    base = spectral_vector(st, st, sx, major, 44100)
    silent = spectral_vector(st*0, st, sx, major, 44100)
    assert base['wanted_retention_db'] == pytest.approx(0)
    assert 'catastrophic_wanted_loss' in paired_decision(base, silent)['reasons']


def test_band_loss_cannot_be_hidden_by_louder_other_band():
    rng = np.random.default_rng(2)
    st = spectrum(rng.normal(size=(44100*3, 2)))
    sy = st.copy()
    f = np.fft.rfftfreq(2048, 1/44100)
    sy[:, (f >= 250) & (f < 2000)] *= .1
    sy[:, f >= 8000] *= 2
    major = np.zeros(st.shape, bool)
    b = spectral_vector(st, st, st, major, 44100)
    c = spectral_vector(sy, st, st, major, 44100)
    assert c['wanted_retention_db'] > b['wanted_retention_db']
    assert 'band_loss_250_2000' in paired_decision(b, c)['reasons']


def test_no_target_does_not_fabricate_retention():
    st = np.zeros((2, 1025, 12), dtype=complex)
    v = spectral_vector(st+1, st, st+2, np.ones(st.shape, bool), 44100)
    assert v['wanted_retention_db'] is None
    assert paired_decision(v, v)['delta_retention_db'] is None


def test_actual_window_measurement_retains_tail_and_detects_stereo_collapse(tmp_path):
    rng = np.random.default_rng(19)
    rate = 44100
    x = rng.normal(0, .1, (rate*7, 2))
    y = x.copy()
    y[rate*3:rate*6] = np.repeat(y[rate*3:rate*6].mean(axis=1, keepdims=True), 2, axis=1)
    yp = tmp_path/'y.npy'
    xp = tmp_path/'x.npy'
    np.save(yp, y)
    np.save(xp, x)
    attr = tmp_path/'attribution.npz'
    np.savez(attr, major=np.zeros(spectrum(x).shape, bool))
    def asset(p):
        return {'path': str(p), 'sha256': digest(p), 'frames': len(x)}
    c = {'id': 'control', 'episode': 'control', 'segment_seconds': [100, 107],
         'strata': [], 'truth': 'controlled', 'mix': asset(xp), 'target': asset(xp),
         'baseline': asset(xp), 'candidate': asset(yp), 'attribution': asset(attr)}
    result = measure_case(c, rate)
    assert [(w['start_seconds'], w['end_seconds']) for w in result['windows']] == [(0, 3), (3, 6), (6, 7)]
    assert 'wanted_stereo_width_regression' in result['windows'][1]['decision']['reasons']
    # Hash-bound inputs cannot silently be replaced by another run.
    np.save(yp, x)
    with pytest.raises(ValueError, match='Changed asset'):
        measure_case(c, rate)


def test_loader_rejects_short_changed_or_nonstereo_output(tmp_path):
    p = tmp_path/'x.npy'
    np.save(p, np.zeros((99, 2)))
    a = {'path': str(p), 'sha256': digest(p), 'frames': 100}
    with pytest.raises(ValueError, match='Incomplete asset'):
        load_asset(a, 44100)
    np.save(p, np.zeros((100, 1)))
    a['sha256'] = digest(p)
    with pytest.raises(ValueError, match='native stereo'):
        load_asset(a, 44100)
