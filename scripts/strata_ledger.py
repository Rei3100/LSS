"""Net-value ledger for work handed to local Strata, and the gate that closes a job kind that does not pay.

    python scripts/strata_ledger.py report
    python scripts/strata_ledger.py gate code      # exit 0 = hand this kind to Strata, 1 = cloud does it itself

Strata earns its GPU time only when the cloud model spends less than it would have written the result itself.
A task that passes within ``PAYING_ATTEMPTS`` attempts saves the cloud the drafting. A task that exhausts its
attempts costs the cloud the draft AND a failed round trip, so it is a loss. The gate looks at the last
``WINDOW`` finished tasks of one kind: at least ``MIN_PAYING`` of them paying keeps the kind open, fewer closes it
(cloud only) until a recorded change reopens it with ``reopen``. A kind with fewer than ``WINDOW`` records is on
probation: open, so that it can collect evidence. The ledger is ``.state/strata-ledger.jsonl`` of the main checkout, shared by every worktree (one row per task,
appended by ``strata_task.py`` after a run).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _shared_root(root=ROOT):
    """The main checkout that owns the shared git directory, so every worktree reads and appends one ledger.

    A per-worktree ledger split the evidence (8 rows in one lane, 2 in another, 0 on main) and kept every kind
    on probation forever. Falls back to this checkout when git is unavailable.
    """
    import subprocess
    try:
        common = subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=root,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return Path(root)
    return Path(common).parent if common else Path(root)


LEDGER = (dev_state_root() if 'LSS_DEV_STATE' in os.environ
          else dev_state_root(_shared_root())) / 'strata-ledger.jsonl'
WINDOW = 6
MIN_PAYING = 4
PAYING_ATTEMPTS = 2


def _rows(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def ingest(summary_path, *, ledger=LEDGER, note=None):
    """Append one row per task of a finished ``strata_task`` summary; returns the rows added."""
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    rows = [{'at': now, 'task': t['id'], 'kind': t.get('kind', 'code'), 'status': t['status'],
             'attempts': t['attempts'], 'tokens': t['tokens'], 'seconds': t['seconds'],
             'paying': t['status'] == 'PASS' and t['attempts'] <= PAYING_ATTEMPTS,
             'engine': (summary.get('identity') or {}).get('engine') if isinstance(summary.get('identity'), dict) else None,
             'note': note} for t in summary['tasks']]
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows


def reopen(kind, reason, *, ledger=LEDGER):
    """Start a new evidence window for a kind after a recorded change (model, engine, prompt, task shape)."""
    if not reason.strip():
        raise ValueError('reopen needs a reason')
    with Path(ledger).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'kind': kind,
                                 'reopen': reason}, ensure_ascii=False) + '\n')


def verdict(kind, *, ledger=LEDGER):
    window = []
    for row in _rows(ledger):
        if row.get('kind') != kind:
            continue
        if 'reopen' in row:
            window = []
        else:
            window.append(row)
    recent = window[-WINDOW:]
    paying = sum(1 for r in recent if r['paying'])
    if len(recent) < WINDOW:
        state = 'probation'
    else:
        state = 'open' if paying >= MIN_PAYING else 'closed'
    return {'kind': kind, 'state': state, 'paying': paying, 'finished': len(recent),
            'failed': sum(1 for r in recent if r['status'] != 'PASS'),
            'seconds': round(sum(r['seconds'] for r in recent), 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('report')
    gate = sub.add_parser('gate')
    gate.add_argument('kind')
    again = sub.add_parser('reopen')
    again.add_argument('kind')
    again.add_argument('reason')
    args = parser.parse_args(argv)
    if args.command == 'gate':
        found = verdict(args.kind)
        print(json.dumps(found, ensure_ascii=False))
        return 0 if found['state'] in ('open', 'probation') else 1
    if args.command == 'reopen':
        reopen(args.kind, args.reason)
        return 0
    kinds = sorted({r['kind'] for r in _rows(LEDGER)})
    for kind in kinds or ['code', 'test']:
        print(json.dumps(verdict(kind), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
