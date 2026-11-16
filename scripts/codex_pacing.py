"""Conservative five-hour advice from bounded, explicit local Codex log tails.

This does not control the host or guarantee account availability. Weekly usage is
not a pacing input. Missing/stale/expired observations fail closed for new agent
work. Percentages of the same shared limit are never added together. The highest
fresh observation wins. A passed reset deadline alone never proves renewal.

Read at substantial batch boundaries. The static thresholds below are conservative
advice, not host enforcement or a throughput target; current explicit user batch
instructions govern delegation. Native account reads avoid stale rollout logs.
OS-owned work can continue while requests wait, subject to its resource guards.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

TAIL_BYTES = 512 * 1024
FRESH_SECONDS = 300


def observations(path: Path):
    with path.open('rb') as stream:
        stream.seek(0, 2)
        start = max(0, stream.tell() - TAIL_BYTES)
        stream.seek(start)
        if start:
            stream.readline()  # Discard a partial line.
        lines = stream.read(TAIL_BYTES).splitlines()
    for line in lines:
        try:
            event = json.loads(line)
            limits = event['payload']['rate_limits']
            primary = limits['primary']
            stamp = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                continue
            used, reset = primary['used_percent'], primary['resets_at']
            if (type(used) not in (int, float) or not math.isfinite(used)
                    or not 0 <= used <= 100 or type(reset) is not int
                    or primary.get('window_minutes') != 300):
                continue
            yield {'timestamp': stamp.timestamp(), 'used_percent': used,
                   'reset_epoch': reset, 'source': str(path.resolve())}
        except (ValueError, TypeError, KeyError, AttributeError):
            continue


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def native_observations(response, *, now, source):
    """Convert a fresh native read; never infer permission from a percentage."""
    if not isinstance(response, dict):
        raise ValueError('Malformed native account response')
    allowed = response.get('ordinaryUsageAllowed')
    if allowed is False:
        raise ValueError('Backend denies ordinary included usage')
    buckets = response.get('rateLimitsByLimitId') or {}
    if not isinstance(buckets, dict):
        raise ValueError('Malformed native rate-limit buckets')
    snapshot = buckets.get('codex', response.get('rateLimits', {}))
    if not isinstance(snapshot, dict):
        raise ValueError('Malformed native primary snapshot')
    primary = snapshot.get('primary') or {}
    if not isinstance(primary, dict):
        raise ValueError('Malformed native primary window')
    used, reset = primary.get('usedPercent'), primary.get('resetsAt')
    if (type(used) not in (int, float) or not math.isfinite(used) or not 0 <= used <= 100
            or type(reset) is not int or not now < reset <= now + 18000
            or primary.get('windowDurationMins') != 300):
        raise ValueError('Native read has no valid current five-hour primary observation')
    return [{'timestamp': now, 'used_percent': used, 'reset_epoch': reset, 'source': source}]


def advise(samples, *, now, active_workers=0):
    if type(active_workers) is not int or active_workers < 0:
        raise ValueError('active_workers must be a nonnegative integer')
    fresh = [s for s in samples if 0 <= now - s['timestamp'] <= FRESH_SECONDS
             and now < s['reset_epoch'] <= now + 18000]
    result = {'checked_utc': utc(now), 'weekly_pacing': False,
              'active_workers': active_workers, 'allow_delegation': False,
              'allow_parent_step': False, 'max_steps_before_refresh': 1,
              'interrupt_workers': bool(active_workers),
              'scope': 'Local advice only; no host limit enforcement or availability guarantee.'}
    if not fresh:
        return {**result, 'action': 'mechanical_only',
                'reason': 'missing_stale_or_expired_primary_observation'}
    # Highest fresh usage is conservative even if a concurrent observation lags.
    chosen = max(fresh, key=lambda s: (s['used_percent'], s['timestamp']))
    used, reset = chosen['used_percent'], chosen['reset_epoch']
    elapsed = max(0, 18000 - (reset - now))
    target = min(90, elapsed / 18000 * 90 + 8)
    ahead = used > target
    next_check = reset + 2 if used >= 90 else max(now + 60, reset - 18000 + max(0, used - 8) / 90 * 18000)
    interrupt = bool(active_workers and used >= 75)
    parent = used < 90 and not ahead and not interrupt
    delegate = parent and used < 55 and active_workers == 0 and reset - now >= 1800
    return {**result, 'observation_utc': utc(chosen['timestamp']),
            'used_percent': used, 'reset_epoch': reset, 'reset_utc': utc(reset),
            'observed_sources': sorted({s['source'] for s in fresh}),
            'target_used_percent': round(target, 2), 'ahead_of_elapsed_time': ahead,
            'allow_parent_step': parent, 'allow_delegation': delegate,
            'interrupt_workers': interrupt, 'next_check_utc': utc(next_check),
            'action': ('checkpoint_and_interrupt_workers' if interrupt else
                       'one_bounded_parent_step' if parent else 'quiet_mechanical_only')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-log', type=Path, action='append', default=[])
    parser.add_argument('--native-runtime', type=Path,
                        help='Read current account limits through this installed Codex app-server binary; no model call')
    parser.add_argument('--active-workers', type=int, default=0)
    parser.add_argument('--require', choices=['parent', 'delegation'])
    args = parser.parse_args(argv)
    if not args.session_log and not args.native_runtime:
        parser.error('--session-log or --native-runtime is required')
    from dev_hot_paths import HotPathError, require_hot
    samples, errors = [], []
    if args.native_runtime:
        try:
            from codex_hook_status import native_call
            response = native_call(str(args.native_runtime), Path.cwd(), 'account/rateLimits/read', None)['response']
            samples.extend(native_observations(response, now=time.time(), source='native:account/rateLimits/read'))
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(type(exc).__name__)
    for path in args.session_log:
        try:
            require_hot(path, 'Codex pacing log')
            samples.extend(observations(path))
        except (OSError, ValueError, HotPathError) as exc:
            errors.append(type(exc).__name__)
    result = advise(samples, now=time.time(), active_workers=args.active_workers)
    # An unreadable supplied concurrent source must not hide its consumption.
    if errors:
        result.update(action='mechanical_only', allow_parent_step=False,
                      allow_delegation=False, interrupt_workers=bool(args.active_workers),
                      source_errors=errors)
    print(json.dumps(result, ensure_ascii=False))
    if args.require:
        return 0 if result['allow_parent_step' if args.require == 'parent' else 'allow_delegation'] else 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
