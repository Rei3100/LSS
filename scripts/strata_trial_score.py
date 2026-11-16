"""Wait for a Strata matrix, then reuse the unchanged historical repair oracles.

    python scripts/strata_trial_score.py TRIAL_DIR --out OUTPUT_DIR --watch

The watcher does no inference and never changes drafts or Product files. The
worker runs on one CPU with a 50% hard cap; failure of the historical positive
control aborts grading rather than blaming the model for a harness timeout.
Strict JSON and the existing fence decoder are separate scoring views of the
same replies, not additional model attempts or extra independent repair cases.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import strata_trials as trials


def export(trial_dir, thinking):
    summary = json.loads((trial_dir / 'summary.json').read_text(encoding='utf-8'))
    identity = json.loads((trial_dir / 'identity.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed' or summary['completed'] != summary['total']:
        raise ValueError('incomplete matrix is not a completed repair population')
    cases = json.loads((trial_dir / 'cases.json').read_text(encoding='utf-8'))
    by_id = {case['id']: case for case in cases}
    entries = []
    for path in sorted(trial_dir.glob('*.json')):
        if len(path.stem) != 64:
            continue
        row = json.loads(path.read_text(encoding='utf-8'))
        if not row.get('id', '').startswith(thinking + '/'):
            continue
        case = by_id[row['id']]
        key = trials.digest({'identity': identity, 'case': case, 'request': row['request']})
        if key != row.get('key') or key != path.stem:
            raise ValueError('request/identity mismatch: ' + str(path))
        response = row.get('response')
        if response is not None and trials.digest(response) != row.get('response_digest'):
            raise ValueError('response integrity mismatch: ' + str(path))
        choice = (response or {}).get('choices', [{}])[0]
        entries.append({'id': row['id'].split('/', 1)[1],
                        'response': (choice.get('message') or {}).get('content') or '',
                        'finish_reason': choice.get('finish_reason'),
                        'status': row.get('status'), 'error': row.get('error')})
    expected = {case['id'].split('/', 1)[1] for case in cases if case['id'].startswith(thinking + '/')}
    if {row['id'] for row in entries} != expected:
        raise ValueError('matrix receipts are missing or duplicated')
    return {'model_sha256': trials.digest(identity['installation']['model_files']),
            'model_identity_kind': 'composite digest of pinned shard identities',
            'strata_identity': identity, 'replays': entries}


def evaluate(trial_dir, out):
    from score_local_ai_cpu_trial import score
    from summarize_local_ai_repairs import summarize
    reports = []
    for thinking in ('off', 'low'):
        source = out / f'{thinking}-replies.json'
        trials.atomic(source, export(trial_dir, thinking))
        for fences in (False, True):
            label = f'{thinking}-' + ('fences' if fences else 'strict')
            path = out / f'{label}.json'
            report = score(source, adapt_fences=fences)
            trials.atomic(path, report)
            reports.append({'view': label, **summarize(path)})
    trials.atomic(out / 'repair-summary.json', {'status': 'completed', 'runs': reports,
                  'scope': 'Five common functions plus optional sixth, two thinking choices; adapter views are not additional attempts'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    trial_dir = trials.output_directory(args.trial)
    out = trials.output_directory(args.out)
    if args.worker:
        evaluate(trial_dir, out)
        return 0
    while True:
        resource_path = trial_dir / 'resources-summary.json'
        resource = json.loads(resource_path.read_text(encoding='utf-8')) if resource_path.exists() else {}
        status = resource.get('status')
        if status == 'completed':
            break
        if status in ('failed', 'aborted-memory-pressure', 'interrupted'):
            trials.atomic(out / 'repair-summary.json', {'status': 'not-scored-resource-failure', 'resource': resource,
                                                       'scope': 'No repair-quality denominator is claimed'})
            return 3
        if not args.watch:
            raise ValueError('matrix is not stopped and completed; use --watch to wait')
        time.sleep(30)
    from quiet_cpu import run_capped
    command = [sys.executable, str(Path(__file__).resolve()), str(trial_dir), '--out', str(out), '--worker']
    with (out / 'score.log').open('ab') as log:
        accounting = run_capped(command, cwd=trials.ROOT, env=os.environ.copy(), stdout=log, duty=.50)
    rc = accounting['returncode']
    trials.atomic(out / 'cpu-receipt.json', {'returncode': rc, 'accounting': accounting, 'one_cpu_duty': .50})
    if rc:
        trials.atomic(out / 'grading-failed.json', {'status': 'harness-failed', 'returncode': rc,
                                                  'scope': 'Inspect score.log; do not count as model failure'})
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
