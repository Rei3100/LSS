"""Run the prepared full-BF16 natural-audio comparison after the GPU blackout.

No model download or Product mutation. gpu_run owns the one GPU lease and caps
the child tree at an explicit duty (default 25%); this parent records global
sampled resource release. Duty 1 runs without the duty cap and retains all guards.
The source WAV identities match the native trial; model preprocessing, precision
and system templates differ, so this compares complete configurations.
"""
try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import hashlib
import re
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import local_llm
from probe_audio_understanding import _enforce_gpu_blackout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='r4', help='unique run label such as r5-constrained')
    parser.add_argument('--constrained-captions', action='store_true')
    parser.add_argument('--duty', type=float, default=0.25)
    args = parser.parse_args(argv)
    if not 0 < args.duty <= 1:
        parser.error('duty must be greater than zero and at most one')
    if not re.fullmatch(r'r[0-9]+(?:-[a-z0-9]+)*', args.label):
        parser.error('label must be r followed by digits and optional lowercase alphanumeric suffixes')
    _enforce_gpu_blackout()  # Before sampling, CUDA operations or child launch.
    evidence = dev_evidence_root(ROOT) / 'release-evidence/local-ai-audio-r1'
    out = evidence / f'natural-comparison-{args.label}'
    try:
        require_evidence_writable(out)
        out.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError('immutable run identity already exists; inspect it before another trial') from exc
    result = out / 'run.json'
    command = [sys.executable, str(ROOT / 'scripts/gpu_run.py'), '--duty', str(args.duty), '--purpose',
        f'local-ai-natural-{args.label}', '--', str(ROOT / '.cache/local-ai/omni-env/Scripts/python.exe'),
        str(ROOT / 'scripts/probe_audio_understanding.py'), '--run', '--evidence', str(evidence),
        '--output', str(result), '--prompt-profile', 'english', '--audio-manifest',
        str(dev_evidence_root(ROOT) / 'release-evidence/local-ai-challengers-r1/musiccaps/acquisition.json')]
    if args.constrained_captions:
        command.append('--constrained-captions')
    before = local_llm._vram_mib()
    sampler = local_llm.Sampler(interval=2.0)
    sampler.start()
    started = time.monotonic()
    code = None
    try:
        code = subprocess.run(command, cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW).returncode
    finally:
        sampler.stop()
        release_seconds, after = local_llm._vram_release(before)
        record = {'returncode':code, 'command':command, 'duty':args.duty,
            'wall_seconds':round(time.monotonic()-started,3), 'global_vram_before_mib':before,
            'global_vram_peak_mib':sampler.vram_peak, 'global_vram_after_mib':after,
            'global_ram_available_min_gb':sampler.ram_min,
            'release_seconds_within_150_mib_of_before':release_seconds,
            'probe_sha256':hashlib.sha256((ROOT/'scripts/probe_audio_understanding.py').read_bytes()).hexdigest(),
            'scope':'Whole-machine sampled counters, not exclusive process attribution. Full BF16 and fixed resource floors retained.'}
        (out/'resources.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
