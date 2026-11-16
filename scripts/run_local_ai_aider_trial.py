"""Bounded Aider-prompt/parser comparison on one historical function; no auto-apply."""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import local_ai_bench as bench
import local_ai_replay_r2 as replay
import local_ai_aider_trial as guarded
import local_llm

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = dev_evidence_root(ROOT) / 'release-evidence/local-ai-challengers-r1/aider'
CASE_PARENT = ROOT / '.cache/local-ai'
COMMIT_SHORT = '10e946b6'
TARGET_PATH = 'scripts/state_status.py'
RUNTIME = CASE_PARENT / 'aider-env/Scripts/python.exe'
CLIENT = ROOT / 'scripts/local_ai_aider_trial.py'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def load_task():
    commit = bench.git('rev-parse', COMMIT_SHORT).strip()
    before = bench.git('show', f'{commit}^:{TARGET_PATH}')
    after = bench.git('show', f'{commit}:{TARGET_PATH}')
    constants = bench.regex_constants(after)
    constants = {key: constants[key] for key in ('STAMP_PREFIX', 'LANE_PREFIX', 'LANE_HEADER_PREFIX')}
    return {'commit': commit, 'issue': bench.git('log', '-1', '--format=%s', commit).strip(),
            'before': bench.function_source(before, '_normalise'),
            'after': bench.function_source(after, '_normalise'),
            'constants': constants, 'cases': bench.behavioral_cases('_normalise')}


def contract(task):
    return ('Repair the historical _normalise function in candidate.py. Keep its signature. '
            'Use only the given constants. Preserve durable lines; ignore the generated timestamp, '
            'lane rows and lane header. Return exactly one SEARCH/REPLACE block for candidate.py. '
            'No imports, extra functions, file access, shell commands or additional files.\n'
            + json.dumps({'issue': task['issue'], 'available_constants': task['constants']}, ensure_ascii=False) + '\n')


def check_controls(task, evaluate=replay.evaluate_code):
    old = evaluate(task['before'], '_normalise', task['constants'], task['cases'])
    fixed = evaluate(task['after'], '_normalise', task['constants'], task['cases'])
    if old.get('pass') or not fixed.get('pass'):
        raise RuntimeError('historical baseline/positive controls invalid')
    return {'baseline': old, 'positive_control': fixed}


def child_environment():
    # Explicit allowlist: never inherit API keys or provider settings.
    env = {key: os.environ[key] for key in (
        'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT',
        'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'LOCALAPPDATA', 'APPDATA') if key in os.environ}
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='-1',
               LITELLM_LOCAL_MODEL_COST_MAP='True', OPENAI_API_KEY='local-dummy',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def run(label, *, serve=local_llm.serving, evaluate=replay.evaluate_code, child=subprocess.run):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', label):
        raise ValueError('label must contain only lowercase alphanumerics separated by hyphens')
    from dev_hot_paths import require_hot
    for path in (ROOT, EVIDENCE, CASE_PARENT, RUNTIME, CLIENT): require_hot(path, 'Aider wrapper')
    out = EVIDENCE / label
    case = CASE_PARENT / ('aider-case-' + label)
    if out.exists() or case.exists(): raise ValueError('immutable trial label already exists')
    require_evidence_writable(out)
    out.mkdir(parents=False, exist_ok=False)
    record = {'status': 'unmeasured', 'stage': 'controls', 'label': label,
              'scope': 'guarded Aider prompt/parser variant; not a full autonomous agent', 'unverified': True}
    started = time.monotonic()
    paths = snapshot = None
    try:
        task = load_task()
        record.update(commit=task['commit'], target='_normalise', controls=check_controls(task, evaluate))
        case.mkdir(exist_ok=False)
        (case / 'candidate.py').write_text(task['before'], encoding='utf-8')
        (case / 'contract.txt').write_text(contract(task), encoding='utf-8')
        paths = guarded.CasePaths(case)
        snapshot = guarded.snapshot_case(paths)
        record['case_before_sha256'] = snapshot
        model = local_llm.get_model('qwen3.5-4b')
        require_hot(model.path, 'Aider model weights')
        require_hot(local_llm.llama_dir(), 'Aider model runtime')
        record['identity'] = {'model': model.name, 'model_sha256': sha(model.path),
            'model_expected_sha256': model.sha256, 'runtime_sha256': sha(local_llm.llama_dir() / 'llama-server.exe'),
            'runtime_build': local_llm.BUILD,
            'runtime_dlls_sha256': {p.name: sha(p) for p in sorted(local_llm.llama_dir().glob('*.dll'))},
            'aider_python_sha256': sha(RUNTIME), 'client_sha256': sha(CLIENT),
            'wrapper_sha256': sha(Path(__file__)), 'mode': 'guarded-aider-prompt-parser', 'ctx': 16384}
        if model.sha256 and record['identity']['model_sha256'].lower() != model.sha256.lower():
            raise ValueError('model SHA mismatch')
        record['stage'] = 'server'
        with serve(model.name, ctx=16384, purpose='guarded Aider historical proposal') as url:
            transcript = out / 'transcript.json'
            command = [str(RUNTIME), str(CLIENT), '--case', str(case), '--base-url', url,
                       '--transcript', str(transcript)]
            record['stage'] = 'client'
            with (out / 'stdout.txt').open('wb') as stdout, (out / 'stderr.txt').open('wb') as stderr:
                result = child(command, cwd=str(case), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                               env=child_environment(), timeout=240, creationflags=subprocess.CREATE_NO_WINDOW)
            record['client_returncode'] = result.returncode
        record['case_after_sha256'] = guarded.snapshot_case(paths)
        if record['case_after_sha256'] != snapshot: raise RuntimeError('case changed during proposal')
        record['transcript'] = str(transcript)
        if result.returncode:
            error = json.loads(transcript.read_text(encoding='utf-8')).get('error', '') if transcript.exists() else ''
            format_errors = ('expected exactly one candidate edit block', 'invalid response size',
                             'only candidate edits permitted', 'edit must uniquely match',
                             'expected one _normalise', 'exactly one edit', 'shell output refused')
            record['status'] = 'format_or_guard_rejected' if any(item in error for item in format_errors) or error.startswith('SyntaxError:') else 'unmeasured'
            record['client_error'] = error or None
            record['stage'] = 'client_rejected'
        else:
            proposal = json.loads((out / 'stdout.txt').read_text(encoding='utf-8'))
            if proposal.get('status') != 'proposal_only' or proposal.get('case_sha256') != snapshot:
                raise ValueError('unexpected proposal receipt')
            source = proposal['proposal']
            record['proposal'] = source
            record['aider_identity'] = proposal['identity']
            record['stage'] = 'oracle'
            record['oracle'] = evaluate(source, '_normalise', task['constants'], task['cases'])
            record['status'] = 'evaluated'
            record['oracle_scope'] = 'bounded historical behavioral cases only'
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if paths is not None and snapshot is not None:
            try:
                record['case_after_sha256'] = guarded.snapshot_case(paths)
                if record['case_after_sha256'] != snapshot:
                    record['status'] = 'unmeasured'
                    record['error'] = 'case changed during proposal'
            except Exception as exc:
                record['status'] = 'unmeasured'
                record['error'] = f'case verification failed: {type(exc).__name__}: {exc}'
        record['wall_s'] = round(time.monotonic() - started, 3)
        (out / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    result = run(args.label)
    print(json.dumps({'status': result['status'], 'stage': result['stage'], 'label': args.label}))
    return 0 if result['status'] == 'evaluated' else 1


if __name__ == '__main__': raise SystemExit(main())
