"""Freeze the existing 42-row real-mix research bank; never call it holdout.

Uses cached Musicless on original mix, not Musicless-after-Product or dialogue-first.
The fitted T and proxy attribution are reused, never fitted to the candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
import platform
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root


import numpy as np
import soundfile as sf
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_bgm_quality import POLICY
from scripts.research_tests import digest, read


def prepare(repo: Path, cache: Path) -> dict:
    valid = dev_state_root(repo) / 'validation'
    bank = valid / 'bgm-model-dependence-r1/bank.json'
    manifest = valid / 'bgm-candidate-headroom-r1/case-manifest.json'
    native = {r['id']: r for r in read(manifest)['rows']}
    def asset(path, **fields):
        return {'path': str(path.resolve()), 'sha256': digest(path), **fields}
    cases = []
    for r in read(bank)['rows']:
        rid = r['id']
        folder = cache / 'bgm-model-dependence-r1/crops' / rid
        path = folder / 'row.npz'
        with np.load(path, allow_pickle=False) as z:
            frames, channels = z['X'].shape
            offset = int(z['offset'])
            if channels != 2 or z['T'].shape != z['X'].shape:
                raise ValueError(f'Invalid native row: {rid}')
        cand = cache / 'bgm-breadth-sweep-r1/crops' / rid / 'musicless/music_residual.wav'
        info = sf.info(cand)
        if info.samplerate != 44100 or info.channels != 2 or offset+frames > info.frames:
            raise ValueError(f'Invalid cached candidate: {rid}')
        source = asset(path, key='X', frames=frames)
        cases.append({'id': rid, 'episode': r['episode'], 'segment_seconds': r['segment_seconds'],
                      'strata': native[rid]['strata'], 'truth': 'REUSED_FITTED_REFERENCE_NOT_HOLDOUT',
                      'native_split_label': r['split'], 'mix': source,
                      'target': {**source, 'key': 'T'},
                      'baseline': asset(cache / 'bgm-sfx-mechanism-r1/rows' / rid / 'Y10.npy', frames=frames),
                      'candidate': asset(cand, start_frame=offset, frames=frames),
                      'attribution': asset(cache / 'bgm-breadth-sweep-r1/ctx' / f'{rid}.npz')})
    if len(cases) != 42 or set(native) != {r['id'] for r in cases}:
        raise ValueError('Historical 42-row population changed')
    dependencies = [bank, manifest, ROOT/'scripts/compare_bgm_quality.py',
                    ROOT/'scripts/prepare_bgm_quality.py', ROOT/'scripts/research_tests.py',
                    ROOT/'lss/validation/stereo_quality.py',
                    valid/'bgm-breadth-sweep-r1/sweep_lib.py',
                    valid/'bgm-sfx-mechanism-r1/build_y10.py',
                    valid/'bgm-model-dependence-r1/bank.py',
                    valid/'bgm-published-quality-audit-r1/audit_reference.py']
    return {'schema_version': 1, 'sample_rate': 44100, 'policy': POLICY,
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__},
            'population': '42 reused development intervals, corpus-a and corpus-c; correlated/overlapping views',
            'baseline_identity': 'Historical Y10 cached published composition; not present installed EXE',
            'candidate_identity': 'Historical cached Musicless standalone on original mix; not post-Y10 or dialogue-first',
            'reference_limit': 'T: existing source-fitted released reference, 0.1 s / 9-tap fit. Arrangement/unmodelled music and reference-fit errors are not separable from residue.',
            'preprocessing': '44100 Hz stereo snapshot; no new alignment, fit, normalization or resampling; scipy STFT 2048/512 centered zero padding; 3 s windows plus final partial window; 10:1 source/target masks, floor 1e-7; majority speech/effects attribution is a proxy.',
            'unresolved': ['Individual instrument identity/loss', 'subjective naturalness', 'real recurrence splice seams', 'untouched holdout', 'current installed release quality'],
            'dependencies': [asset(p) for p in dependencies], 'cases': cases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Protocol already exists')
    result = prepare(args.repo, args.cache)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f'Frozen {len(result["cases"])} cases')


if __name__ == '__main__':
    main()
