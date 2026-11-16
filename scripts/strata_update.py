"""Signed, side-by-side, gated and reversible Strata updates for the development runtime.

    python scripts/strata_update.py check            # signed feed vs pin; exit 0 current, 10 update available
    python scripts/strata_update.py apply            # one signed step, smoke + fixed bench, revert on failure
    python scripts/strata_update.py rollback         # manual: components rollback + previous pin

Rules (every one is a refusal, never a silent skip, and writes a receipt to `.state/strata-updates/`):
- Only the signed feed (built-in keys, `ok`, status `available`) of the release tag GitHub names as latest is used;
  never an unsigned `latest` file, and the feed's replay-protection sequence may not go backwards.
- One step: a single `strata update`; the engine must go up within the same 0.N series unless --allow-minor; a version
  that was reverted before (`rejected` in the pin) needs --force.
- The official updater installs the new engine/Control beside the old one and test-starts it; the engine archive hash
  must equal the signed feed's SHA256 when the archive is kept on disk.
- Gate: `strata_dev.py smoke --bench` (exact JSON/identifier controls plus the generic coding bench). Fewer previously
  passing tasks than the baseline minus the tolerance, a failed smoke or a decode-speed drop below 70% reverts with
  `strata components rollback` and the previous pin.
- Nothing runs while the LSS GPU lease is held (exit 3, receipt `skipped`), so the updater never fights a job.
  The schedule is the user-level agent kit's task "Agent Kit DevTools Update" (every 15 min, at logon and at session
  start; `strata` entry of ~/.agent-kit/updaters.json runs `apply` and treats exit 3 as blocked).
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import strata_pin  # noqa: E402

REPO = 'https://github.com/Niko1221/Strata'
RECEIPTS = dev_state_root(ROOT) / 'strata-updates'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
GATE_TASKS = ('parse_duration,split_csv_line,merge_intervals,shortest_path,find_duplicate_files,safe_join,LRUCache,'
              'extract_semver,toposort,fix_rolling_average,write_test_slugify,longctx_add_trimmed_mean,longctx_fix_quantiles')
SPEED_FLOOR = 0.70


class Refused(RuntimeError):
    """The update is not applied; the message is the receipt's reason."""


# ---------------------------------------------------------------- pure decisions (unit-tested with fakes)

def decide(report: dict, pin: dict, *, allow_minor=False, force=False) -> dict:
    """Turn `strata update --check --output json` into an action: current / update / refuse (+reason)."""
    if not report.get('ok'):
        return {'action': 'refuse', 'reason': f"feed check failed: {report.get('error', {}).get('message', report)}"}
    feed = report.get('feed', {})
    if feed.get('trusted_by') != 'built-in keys':
        return {'action': 'refuse', 'reason': f"feed is not signed by the built-in keys ({feed.get('trusted_by')!r})"}
    engine = report.get('engine', {})
    offered, installed = engine.get('offered'), engine.get('installed')
    base = {'feed_sequence': feed.get('sequence'), 'installed_engine': (installed or {}).get('version'),
            'offered_engine': (offered or {}).get('version'), 'control_offered': report.get('control', {}).get('offered'),
            'control_installed': report.get('control', {}).get('installed')}
    if not offered or engine.get('status') != 'available':
        return {'action': 'current', **base, 'reason': f"engine status {engine.get('status')!r}"}
    if not installed or not installed.get('signed', False):
        return {'action': 'refuse', **base, 'reason': 'installed engine is not a signed package'}
    sha = offered.get('sha256', '')
    if len(sha) != 64:
        return {'action': 'refuse', **base, 'reason': 'offered engine has no SHA256 in the signed feed'}
    new, old = strata_pin.version_tuple(offered['version']), strata_pin.version_tuple(pin['engine'])
    if new <= old:
        return {'action': 'current', **base, 'reason': 'offered engine is not newer than the pin'}
    if new[:2] != old[:2] and not allow_minor:
        return {'action': 'refuse', **base, 'reason': f"{pin['engine']} -> {offered['version']} is not one patch step (use --allow-minor)"}
    if offered['version'] in pin.get('rejected', []) and not force:
        return {'action': 'refuse', **base, 'reason': f"{offered['version']} was reverted before (use --force)"}
    if base['installed_engine'] != pin['engine']:
        return {'action': 'refuse', **base, 'reason': f"installed engine {base['installed_engine']} differs from the pin {pin['engine']}; resolve first"}
    sequence = pin.get('feed_sequence') or 0
    if (feed.get('sequence') or 0) < sequence:
        return {'action': 'refuse', **base, 'reason': 'feed sequence went backwards (replayed feed)'}
    return {'action': 'update', **base, 'engine_archive': offered['url'], 'engine_sha256': sha,
            'download_bytes': offered.get('size')}


def regression(baseline: dict | None, result: dict, tolerance=1) -> str | None:
    """Reason to revert, or None. `baseline`/`result` hold passed_ids, ids_run and decode_tok_s."""
    if not result.get('smoke_passed', False):
        return 'smoke failed'
    if not result.get('ids_run'):
        return f"coding bench produced no result (gate exit {result.get('exit')})"
    if baseline and baseline.get('ids_run'):
        common = set(baseline.get('ids_run') or []) & set(result.get('ids_run') or [])
        lost = sorted((set(baseline.get('passed_ids', [])) & common) - set(result.get('passed_ids', [])))
        if len(lost) > tolerance:
            return f'quality regression: {len(lost)} previously passing tasks now fail ({", ".join(lost)})'
        speed, base = result.get('decode_tok_s'), baseline.get('decode_tok_s')
        if speed and base and speed < SPEED_FLOOR * base:
            return f'decode speed {speed} tok/s is below {SPEED_FLOOR:.0%} of baseline {base}'
    return None


def step_pin(pin: dict, decision: dict, tag: str, control: str, baseline: dict | None) -> dict:
    new = {'release_tag': tag, 'terminal': tag.lstrip('v'), 'control': control, 'engine': decision['offered_engine'],
           'engine_archive': decision['engine_archive'], 'engine_sha256': decision['engine_sha256'],
           'install_sha256': pin['install_sha256'], 'feed_sequence': decision['feed_sequence'],
           'bench_baseline': baseline, 'pinned_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    return strata_pin.advance(pin, new)


# ---------------------------------------------------------------- real operations (replaced by fakes in tests)

class Ops:
    def __init__(self):
        import strata_dev
        self.dev = strata_dev
        self.home, self.model = strata_dev.paths('coder')

    def latest_tag(self) -> str:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        # Python's own Windows store load misses issuers Schannel fetches on demand (CERTIFICATE_VERIFY_FAILED on
        # github.com); the venv's certifi bundle verifies, as in lss/acquisition/provider.py.
        import ssl
        import certifi
        https = urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where()))
        opener = urllib.request.build_opener(NoRedirect, https)
        try:
            opener.open(urllib.request.Request(f'{REPO}/releases/latest', method='HEAD'), timeout=60)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get('Location', '')
            if exc.code in (301, 302, 303, 307, 308) and '/releases/tag/' in location:
                return location.rsplit('/', 1)[1]
        raise Refused('could not read the latest release tag from GitHub')

    def cli_json(self, *args, timeout=3600):
        terminal = self.home / 'terminal' / 'current'
        proc = subprocess.run([str(terminal / 'python' / 'python.exe'), str(terminal / 'strata.py'), *args],
                              env=self.dev.environment(self.home), capture_output=True, text=True, timeout=timeout,
                              creationflags=FLAGS, stdin=subprocess.DEVNULL)
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {'ok': False, 'error': {'message': proc.stderr[-500:]}}
        except ValueError:
            return {'ok': False, 'error': {'message': (proc.stdout + proc.stderr)[-500:]}}

    def check(self, tag: str) -> dict:
        return self.cli_json('update', '--check', '--output', 'json', '--feed', f'{REPO}/releases/download/{tag}')

    def lease_held(self) -> bool:
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.CreateMutexW.restype = ctypes.c_void_p
        api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        api.ReleaseMutex.argtypes = api.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = api.CreateMutexW(None, False, 'Local\\LSS-CUDA-Device-0-v1')
        if not handle:
            return True
        try:
            state = api.WaitForSingleObject(handle, 0)
            if state in (0, 0x80):
                api.ReleaseMutex(handle)
                return False
            return True
        finally:
            api.CloseHandle(handle)

    def server_running(self) -> bool:
        with socket.socket() as probe:
            probe.settimeout(1)
            return probe.connect_ex(('127.0.0.1', self.dev.PORT)) == 0

    def run_update(self, tag: str, out: Path) -> dict:
        """The official updater under the GPU lease (it test-starts the new engine on the GPU)."""
        self.dev.recover_stale(self.home)  # a dead launch's record otherwise blocks the engine step as "serving"
        worker = [sys.executable, str(Path(__file__).resolve()), '_update-worker', '--tag', tag, '--out', str(out)]
        command = [sys.executable, str(ROOT / 'scripts' / 'gpu_run.py'), '--purpose', 'Strata update', '--owned-job', '--duty', '1',
                   '--protect', str(self.home), '--protect', str(self.model.parent), '--', *worker]
        code = subprocess.run(command, creationflags=FLAGS, stdin=subprocess.DEVNULL).returncode
        report = json.loads(out.read_text(encoding='utf-8')) if out.exists() else {'ok': False, 'error': {'message': f'worker exit {code}'}}
        return report

    def archive_sha(self, name: str) -> str | None:
        for root in (self.home / 'components', self.home / 'downloads'):
            for candidate in root.rglob(name) if root.exists() else []:
                with candidate.open('rb') as stream:
                    return hashlib.file_digest(stream, 'sha256').hexdigest()
        return None

    def gate(self, label: str, out: Path) -> dict:
        """Smoke + generic bench in ONE model load under the lease; returns the summary the comparison needs."""
        smoke = out / 'smoke.json'
        code = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'strata_dev.py'), 'smoke', '--out', str(smoke), '--bench', label,
                               '--bench-only', GATE_TASKS], creationflags=FLAGS, stdin=subprocess.DEVNULL).returncode
        result = {'smoke_passed': False, 'exit': code}
        with contextlib.suppress(OSError, ValueError):
            result['smoke_passed'] = bool(json.loads(smoke.read_text(encoding='utf-8')).get('passed'))
        with contextlib.suppress(OSError, ValueError):
            result.update(bench_digest(json.loads((out / 'bench' / f'{label}.json').read_text(encoding='utf-8'))))
        return result

    def rollback(self) -> dict:
        return self.cli_json('components', 'rollback', '--output', 'json')

    def installed_engine(self) -> str | None:
        report = self.cli_json('components', 'engine', '--output', 'json')
        return (report.get('engine') or report).get('version') if isinstance(report, dict) else None


def bench_digest(bench: dict) -> dict:
    rows = [r for r in bench['rows'] if r.get('status') != 'skipped-context']
    summary = bench['summary']
    return {'ids_run': [r['id'] for r in rows], 'passed_ids': [r['id'] for r in rows if r.get('passed')],
            'decode_tok_s': summary.get('decode_tok_s_median'), 'label': summary.get('label')}


# ---------------------------------------------------------------- orchestration

def write_receipt(receipt: dict, directory: Path = RECEIPTS) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    receipt['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    path = directory / f"{receipt['started_utc'].replace(':', '')}-{receipt['command']}.json"
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)
    return path


@contextlib.contextmanager
def single_instance(directory: Path = RECEIPTS):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'apply.lock'
    if lock.exists():
        try:
            pid = int(lock.read_text())
            alive = pid != os.getpid() and subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True,
                                                          text=True, creationflags=FLAGS).stdout.count(str(pid)) > 0
        except ValueError:
            alive = False
        if alive:
            raise Refused(f'another Strata update (pid {pid}) is running')
    lock.write_text(str(os.getpid()))
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()


def do_check(ops, pin, **options) -> dict:
    tag = ops.latest_tag()
    decision = decide(ops.check(tag), pin, **options)
    decision['tag'] = tag
    return decision


def do_apply(ops, pin, *, directory: Path = RECEIPTS, allow_minor=False, force=False, tolerance=1, dry_run=False,
             save=strata_pin.save) -> tuple[int, dict]:
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    receipt = {'command': 'apply', 'started_utc': started, 'pin_before': pin, 'status': 'running'}
    out = directory / started.replace(':', '')
    write_receipt(receipt, directory)

    def finish(status, code, **extra):
        receipt.update(status=status, **extra)
        write_receipt(receipt, directory)
        return code, receipt

    try:
        decision = do_check(ops, pin, allow_minor=allow_minor, force=force)
        receipt['decision'] = decision
        if decision['action'] == 'current':
            return finish('current', 0)
        if decision['action'] == 'refuse':
            return finish('refused', 2, reason=decision['reason'])
        if dry_run:
            return finish('dry-run', 0)
        if ops.lease_held() or ops.server_running():
            return finish('skipped', 3, reason='GPU lease held or a Strata server is running')
        out.mkdir(parents=True, exist_ok=True)
        update = ops.run_update(decision['tag'], out / 'update.json')
        receipt['update'] = update
        if not update.get('ok'):
            return finish('failed-update', 1, reason=str(update.get('error', update))[:500], reverted=False)
        kept = ops.archive_sha(decision['engine_archive'])
        if kept is not None and kept != decision['engine_sha256']:
            ops.rollback()
            return finish('reverted', 1, reason='installed engine archive hash differs from the signed feed', reverted=True)
        control = (update.get('control') or {}).get('installed') or decision.get('control_offered') or pin['control']
        candidate = step_pin(pin, decision, decision['tag'], str(control), pin.get('bench_baseline'))
        save(candidate)  # the gate runs with the candidate pin (strata_dev reads it in the child process)
        label = f"update-{decision['offered_engine']}"
        gate = ops.gate(label, out)
        receipt['gate'] = gate
        why = regression(pin.get('bench_baseline'), gate, tolerance)
        if why:
            rolled = ops.rollback()
            save(strata_pin.revert(candidate))
            receipt['rollback'] = rolled
            return finish('reverted', 1, reason=why, reverted=True)
        baseline = {k: gate.get(k) for k in ('label', 'ids_run', 'passed_ids', 'decode_tok_s')}
        save(step_pin(pin, decision, decision['tag'], str(control), baseline))
        return finish('applied', 0, pin_after=step_pin(pin, decision, decision['tag'], str(control), baseline))
    except Refused as exc:
        return finish('refused', 2, reason=str(exc))
    except Exception as exc:  # unattended: leave a receipt, and put the previous version back if a step was begun
        reverted = False
        if receipt.get('update', {}).get('ok'):
            with contextlib.suppress(Exception):
                ops.rollback()
                save(pin)
                reverted = True
        return finish('error', 1, reason=f'{type(exc).__name__}: {exc}', reverted=reverted)


def update_worker(tag: str, out: Path) -> int:
    import strata_dev
    home, _model = strata_dev.paths('coder')
    ops = Ops()
    report = ops.cli_json('update', '--output', 'json', '--wait', '0', '--feed', f'{REPO}/releases/download/{tag}')
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report.get('ok') else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['check', 'apply', 'rollback', '_update-worker'])
    parser.add_argument('--allow-minor', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tolerance', type=int, default=1, help='previously passing bench tasks that may fail before reverting')
    parser.add_argument('--tag')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.command == '_update-worker':
        return update_worker(args.tag, args.out)
    ops, pin = Ops(), strata_pin.load()
    if args.command == 'check':
        decision = do_check(ops, pin, allow_minor=args.allow_minor, force=args.force)
        print(json.dumps(decision, indent=2))
        return {'current': 0, 'update': 10}.get(decision['action'], 2)
    with single_instance():
        if args.command == 'rollback':
            if ops.lease_held() or ops.server_running():
                print('GPU lease held or a Strata server is running', file=sys.stderr)
                return 3
            rolled = ops.rollback()
            strata_pin.save(strata_pin.revert(pin))
            write_receipt({'command': 'rollback', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                           'status': 'rolled-back', 'result': rolled})
            return 0 if rolled.get('ok', True) else 1
        code, receipt = do_apply(ops, pin, allow_minor=args.allow_minor, force=args.force, tolerance=args.tolerance,
                                 dry_run=args.dry_run)
        print(json.dumps({k: receipt.get(k) for k in ('status', 'reason', 'decision')}, indent=2))
        return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Refused as exc:
        print(f'strata_update: {exc}', file=sys.stderr)
        raise SystemExit(2)
