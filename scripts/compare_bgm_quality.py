"""Replay frozen real-mix comparisons, retaining local damage and reference limits.

Protocol assets are hash-bound NPY/NPZ/WAV files, already aligned to one timeline.
No fitting, inference, normalization, oracle selection, or Product mutation occurs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lss.validation.stereo_quality import db_ratio, energy, stereo_quality
from scripts.research_tests import digest, read

NFFT, HOP = 2048, 512
BANDS = ((0, 250), (250, 2000), (2000, 8000), (8000, 14000), (14000, 20000))
POLICY = {"window_seconds": 3, "retention_loss_db": 1.0,
          "catastrophic_loss_db": 3.0, "band_min_target_share": 0.01,
          "minimum_target_bins": 100, "error_regression_db": 1.0,
          "stereo_width_regression_db": 3.0}


def load_asset(asset: dict, rate: int) -> np.ndarray:
    path = Path(asset['path'])
    if digest(path) != asset['sha256']:
        raise ValueError(f'Changed asset: {path}')
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            x = data[asset['key']].astype(np.float64)
    elif path.suffix == '.npy':
        x = np.load(path, allow_pickle=False).astype(np.float64)
    else:
        x, sr = sf.read(path, dtype='float64', always_2d=True)
        if sr != rate:
            raise ValueError('Sample rate mismatch; resampling is not implicit')
    start = asset.get('start_frame', 0)
    stop = start + asset['frames']
    if start < 0 or stop > len(x):
        raise ValueError('Incomplete asset')
    x = x[start:stop]
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite native stereo')
    return x * asset.get('decode_gain', 1.0)


def spectrum(x: np.ndarray) -> np.ndarray:
    # Same centered zero-padded scipy STFT as the historical native audit.
    return stft(x.T, nperseg=NFFT, noverlap=NFFT-HOP)[2]


def spectral_vector(y: np.ndarray, t: np.ndarray, x: np.ndarray,
                    major: np.ndarray, rate: int, active_floor: float | None = None) -> dict:
    """Fixed source/reference masks; event attribution remains a model proxy."""
    pt, py, pn = np.abs(t)**2, np.abs(y)**2, np.abs(x-t)**2
    if active_floor is None:
        active_floor = 1e-7 * max(float(pt.max()), float(pn.max()))
    active = (pt + pn) > active_floor
    wanted = active & (pt >= 10 * pn)
    unwanted = active & (pn >= 10 * pt)
    total, dominant = float(pt.sum()), float(pt[wanted].sum())
    ratio = lambda a, b: db_ratio(float(a), float(b)) if b > 0 else None
    ret = ratio(py[wanted].sum(), dominant)
    freqs = np.fft.rfftfreq(NFFT, 1/rate)
    bands = []
    for lo, hi in BANDS:
        mask = wanted & ((freqs >= lo) & (freqs < hi))[None, :, None]
        power = float(pt[mask].sum())
        bands.append({'hz': [lo, hi], 'bins': int(mask.sum()),
                      'target_share': power/dominant if dominant else 0.0,
                      'retention_db': ratio(py[mask].sum(), power)})
    both = wanted[0] & wanted[1]
    def width(z):
        mid, side = (z[0]+z[1])/2, (z[0]-z[1])/2
        return db_ratio(float((np.abs(side[both])**2).sum()),
                        float((np.abs(mid[both])**2).sum()))
    side_power = float((np.abs((t[0]-t[1])/2)[both]**2).sum())
    width_error = width(y)-width(t) if side_power > 1e-7*total and both.sum() >= 100 else None
    return {'reference_error_snr_db': db_ratio(total, float((np.abs(y-t)**2).sum())) if total > 0 else None,
            'wanted_retention_db': ret, 'wanted_bins': int(wanted.sum()),
            'wanted_share': dominant/total if total else 0.0,
            'missing_energy_db': ratio((np.maximum(np.abs(t)-np.abs(y), 0)**2).sum(), total),
            'wanted_stereo_width_error_db': width_error,
            'speech_proxy_bleed_db': ratio(py[unwanted & major].sum(), total),
            'effects_proxy_bleed_db': ratio(py[unwanted & ~major].sum(), total),
            'bands': bands}


def paired_decision(base: dict, candidate: dict) -> dict:
    """One fixed local diagnostic rule, never a Product promotion rule."""
    reasons = []
    delta = None
    if min(base['wanted_bins'], candidate['wanted_bins']) >= POLICY['minimum_target_bins']:
        delta = candidate['wanted_retention_db'] - base['wanted_retention_db']
        if delta < -POLICY['retention_loss_db']:
            reasons.append('wanted_loss')
        if delta < -POLICY['catastrophic_loss_db']:
            reasons.append('catastrophic_wanted_loss')
    for b, c in zip(base['bands'], candidate['bands']):
        if (b['bins'] >= POLICY['minimum_target_bins']
                and b['target_share'] >= POLICY['band_min_target_share']
                and c['retention_db'] is not None and b['retention_db'] is not None
                and c['retention_db'] - b['retention_db'] < -POLICY['catastrophic_loss_db']):
            reasons.append(f'band_loss_{b["hz"][0]}_{b["hz"][1]}')
    if (base['reference_error_snr_db'] is not None and candidate['reference_error_snr_db'] is not None
            and candidate['reference_error_snr_db'] < base['reference_error_snr_db']-POLICY['error_regression_db']):
        reasons.append('reference_error_regression')
    bw, cw = base['wanted_stereo_width_error_db'], candidate['wanted_stereo_width_error_db']
    if bw is not None and cw is not None and abs(cw) > abs(bw)+POLICY['stereo_width_regression_db']:
        reasons.append('wanted_stereo_width_regression')
    return {'reasons': reasons, 'delta_retention_db': delta}


def measure_case(case: dict, rate: int) -> dict:
    waves = {name: load_asset(case[name], rate) for name in ('mix', 'target', 'baseline', 'candidate')}
    if len({v.shape for v in waves.values()}) != 1:
        raise ValueError('Timeline mismatch')
    with np.load(case['attribution']['path'], allow_pickle=False) as data:
        major = data['major'].astype(bool)
    s = {k: spectrum(v) for k, v in waves.items()}
    if major.shape != s['mix'].shape:
        raise ValueError('Attribution timeline mismatch')
    active_floor = 1e-7 * max(float((np.abs(s['target'])**2).max()),
                            float((np.abs(s['mix']-s['target'])**2).max()))
    vectors = {k: spectral_vector(s[k], s['target'], s['mix'], major, rate, active_floor)
               for k in ('baseline', 'candidate')}
    stereo = {k: stereo_quality(waves[k].T, waves['target'].T) for k in vectors}
    decision = paired_decision(vectors['baseline'], vectors['candidate'])
    if any(stereo['candidate']['spatial_component_lost'].values()) and not any(stereo['baseline']['spatial_component_lost'].values()):
        decision['reasons'].append('spatial_component_lost')
    windows = []
    step = int(rate * POLICY['window_seconds'])
    for a in range(0, len(waves['mix']), step):
        b = min(a+step, len(waves['mix']))
        # Assign each STFT center once; include the partial final interval explicitly.
        sel = (np.arange(s['mix'].shape[-1])*HOP >= a) & (np.arange(s['mix'].shape[-1])*HOP < b)
        v = {k: spectral_vector(s[k][..., sel], s['target'][..., sel], s['mix'][..., sel], major[..., sel], rate, active_floor)
             for k in vectors}
        windows.append({'start_seconds': a/rate, 'end_seconds': b/rate, **v,
                        'decision': paired_decision(v['baseline'], v['candidate'])})
    return {'id': case['id'], 'episode': case['episode'], 'segment_seconds': case['segment_seconds'],
            'strata': case['strata'], 'truth': case['truth'], **vectors, 'stereo': stereo,
            'decision': decision, 'windows': windows}


def summarize(rows: list[dict]) -> dict:
    windows = [(r['id'], w) for r in rows for w in r['windows']]
    known = [(rid, w) for rid, w in windows if w['decision']['delta_retention_db'] is not None]
    worst = min(known, key=lambda p: p[1]['decision']['delta_retention_db']) if known else None
    whole = sum(bool(r['decision']['reasons']) for r in rows)
    local = sum(any(w['decision']['reasons'] for w in r['windows']) for r in rows)
    return {'rows': len(rows), 'whole_row_harms': whole, 'rows_with_local_harm': local,
            'windows': len(windows),
            'harmful_windows': sum(bool(w['decision']['reasons']) for _, w in windows),
            'catastrophic_windows': sum('catastrophic_wanted_loss' in w['decision']['reasons'] for _, w in windows),
            'worst_retention': {'id': worst[0], 'start_seconds': worst[1]['start_seconds'],
                                'delta_retention_db': worst[1]['decision']['delta_retention_db']} if worst else None,
            'disposition': 'REJECT_BLANKET_REPLACEMENT' if whole or local else 'HOLD'}


def compare(protocol: dict) -> dict:
    if protocol['schema_version'] != 1 or protocol['policy'] != POLICY:
        raise ValueError('Unsupported protocol/policy')
    cases = protocol['cases']
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Empty or duplicate population')
    # Verify every declared dependency before measurement and again before publishing.
    assets = [a for c in cases for a in (c[k] for k in ('mix', 'target', 'baseline', 'candidate', 'attribution'))]
    assets += protocol['dependencies']
    def verify():
        for a in assets:
            if digest(Path(a['path'])) != a['sha256']:
                raise ValueError(f'Changed dependency: {a["path"]}')
    verify()
    rows = []
    for case in cases:
        rows.append(measure_case(case, protocol['sample_rate']))
        print(f'Measured {case["id"]}', flush=True)
    verify()
    return {'schema_version': 1, 'scope': 'DEVELOPMENT_APPROXIMATE_REFERENCE',
            'protocol': protocol, 'summary': summarize(rows), 'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('protocol', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Result already exists; retain each attempt')
    result = compare(read(args.protocol))
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
