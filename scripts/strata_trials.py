"""Frozen Strata x LSS variation matrix; receipts are drafts, never applied code.

Called inside strata_dev's owned server lifetime. Every request and response is
saved, including failed controls and truncations. Repeated cases have distinct
identities. JSON-schema is Strata's prompt-and-validate transport, not a claim
of native constrained decoding. Historical repair evaluation is a later stage.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import hashlib
import json
from pathlib import Path
import time

import strata_dev as runtime

ROOT = Path(__file__).resolve().parents[1]
HISTORY = dev_evidence_root(ROOT) / 'release-evidence/local-ai-challengers-r1/historical-requests.jsonl'
ANSWERS = {'experiment-held': 'held', 'experiment-unknown': 'unknown', 'music-choir': 'keep',
           'music-partial': 'no', 'music-scene': 'no', 'music-currentbest': 'accepted'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic(path, value):
    require_evidence_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def matrix():
    history = [json.loads(line) for line in HISTORY.read_text(encoding='utf-8').splitlines() if line.strip()]
    cases = []
    for thinking in ('off', 'low'):
        for original in history:
            ident = original['id']
            decision = ident in ANSWERS
            prompt = original['prompt'] + ('\nReturn exactly one choice word, without explanation.' if decision else '')
            cases.append({'id': f'{thinking}/{ident}', 'family': 'decision' if decision else 'historical-repair',
                          'thinking': thinking, 'prompt': prompt,
                          'max_tokens': 512 if decision else original['max_tokens'],
                          'oracle': {'type': 'exact', 'value': ANSWERS[ident]} if decision else {'type': 'draft-code-json'}})
        for form in ('plain', 'json_schema'):
            for repeat in (1, 2):
                cases.append({'id': f'{thinking}/japanese-{form}-{repeat}', 'family': 'japanese-json',
                              'thinking': thinking, 'max_tokens': 512,
                              'prompt': 'Return only JSON with title exactly 夜明けの窓辺 and cue_id exactly LSS_CurrentBest_cue_001. Do not translate or change either string.',
                              'oracle': {'type': 'json', 'value': {'title': '夜明けの窓辺', 'cue_id': 'LSS_CurrentBest_cue_001'}},
                              'format': form})
        cases.append({'id': f'{thinking}/tool-currentbest', 'family': 'tool-call', 'thinking': thinking,
                      'max_tokens': 512, 'prompt': 'Call select_current_best with revision_id exactly accepted_cue_001. The newer candidate is HELD and must not be selected.',
                      'oracle': {'type': 'tool', 'name': 'select_current_best', 'arguments': {'revision_id': 'accepted_cue_001'}}})
    for lines in (96, 240, 420):
        filler = '\n'.join(f'def unrelated_{i:04d}(value): return value + {i}' for i in range(lines))
        for placement in ('beginning', 'middle', 'end'):
            fact = 'ACTIVE_CURRENTBEST_ID = "accepted_cue_001"\nNEWER_HELD_ID = "held_cue_999"\n'
            if placement == 'beginning':
                document = fact + filler
            elif placement == 'end':
                document = filler + '\n' + fact
            else:
                middle = len(filler) // 2
                document = filler[:middle] + '\n' + fact + '\n' + filler[middle:]
            cases.append({'id': f'context/{lines}-lines-{placement}', 'family': 'long-context', 'thinking': 'off',
                          'max_tokens': 128, 'prompt': 'Read this development fixture. Return the exact ACTIVE_CURRENTBEST_ID value and nothing else.\n<fixture>\n' + document + '\n</fixture>\nWhich revision is ACTIVE_CURRENTBEST_ID?',
                          'oracle': {'type': 'exact', 'value': 'accepted_cue_001'}})
    return cases


def payload(case, model):
    body = {'model': model, 'messages': [{'role': 'user', 'content': case['prompt']}],
            'temperature': 0, 'max_tokens': case['max_tokens']}
    body['chat_template_kwargs'] = ({'enable_thinking': False} if case['thinking'] == 'off'
                                    else {'reasoning_effort': 'low'})
    oracle = case['oracle']
    if case.get('format') == 'json_schema':
        body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'lss_title', 'strict': True,
                                  'schema': {'type': 'object', 'properties': {'title': {'type': 'string'},
                                             'cue_id': {'type': 'string'}}, 'required': ['title', 'cue_id'],
                                             'additionalProperties': False}}}
    if oracle['type'] == 'tool':
        body['tools'] = [{'type': 'function', 'function': {'name': oracle['name'],
                         'description': 'Record the accepted revision identifier in this test only.',
                         'parameters': {'type': 'object', 'properties': {'revision_id': {'type': 'string'}},
                                        'required': ['revision_id'], 'additionalProperties': False}}}]
        body['tool_choice'] = {'type': 'function', 'function': {'name': oracle['name']}}
    return body


def judge(case, reply):
    choice = reply['choices'][0]
    message = choice['message']
    content = message.get('content') or ''
    oracle = case['oracle']
    if choice.get('finish_reason') not in ('stop', 'tool_calls'):
        return {'status': 'incomplete', 'passed': False}
    try:
        if oracle['type'] == 'exact':
            passed = content.strip() == oracle['value']
        elif oracle['type'] == 'json':
            passed = json.loads(content) == oracle['value']
        elif oracle['type'] == 'tool':
            calls = message.get('tool_calls') or []
            passed = (len(calls) == 1 and calls[0]['function']['name'] == oracle['name']
                      and json.loads(calls[0]['function']['arguments']) == oracle['arguments'])
        else:
            parsed = json.loads(content)
            passed = isinstance(parsed, dict) and isinstance(parsed.get('code'), str) and bool(parsed['code'].strip())
            # A parseable draft is NOT a successful repair.
            return {'status': 'draft-ready' if passed else 'invalid-draft', 'passed': None,
                    'format_valid': passed, 'repair_evaluation': 'pending'}
    except (ValueError, KeyError, TypeError):
        passed = False
    return {'status': 'complete', 'passed': passed}


def reusable(row, key):
    try:
        return (row['key'] == key and row['response_digest'] == digest(row['response'])
                and row['response']['choices'][0]['finish_reason'] in ('stop', 'tool_calls')
                and row.get('status') not in ('error', 'incomplete'))
    except (KeyError, TypeError, IndexError):
        return False


def output_directory(output):
    require_evidence_writable(output)
    out = output.resolve()
    from dev_hot_paths import require_hot
    require_hot(out, 'Strata trial receipts')
    allowed = (ROOT / '.cache/local-ai').resolve(), (dev_evidence_root(ROOT) / 'release-evidence').resolve()
    if not any(out.is_relative_to(root) and out != root for root in allowed):
        raise ValueError('trial output must be inside .cache/local-ai or release-evidence')
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(model, output, *, variant='standard', duty=0.10):
    out = output_directory(output)
    home, _model_path = runtime.paths('coder')
    identity = {'installation': json.loads((home / 'lss-install-coder.json').read_text(encoding='utf-8')),
                'runner_sha256': runtime.sha256(Path(__file__)),
                'runtime_wrapper_sha256': runtime.sha256(Path(runtime.__file__)),
                'history_sha256': runtime.sha256(HISTORY), 'model': model,
                'variant': variant, 'process_duty': duty, 'route_tail_skip': 0,
                'gpu_controller_sha256': runtime.sha256(ROOT / 'scripts/gpu_run.py'),
                'process_job_sha256': runtime.sha256(ROOT / 'scripts/strata_process_job.py'),
                'effective_config_sha256': runtime.sha256(
                    (home if variant == 'standard' else home / 'variants' / variant)
                    / 'user-state/Strata/launch/iq1_m-8k.json')}
    cases = matrix()
    atomic(out / 'cases.json', cases)
    atomic(out / 'identity.json', identity)
    rows = []
    for case in cases:
        body = payload(case, model)
        key = digest({'identity': identity, 'case': case, 'request': body})
        path = out / f'{key}.json'
        old = None
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
        if old and reusable(old, key):
            row = {**old, 'cached': True}
        else:
            row = {'id': case['id'], 'family': case['family'], 'key': key, 'request': body, 'cached': False}
            before = time.monotonic()
            try:
                reply = runtime.request('/v1/chat/completions', body, timeout=3600)
                row.update(response=reply, response_digest=digest(reply), **judge(case, reply))
            except Exception as exc:
                row.update(status='error', passed=False, error=f'{type(exc).__name__}: {exc}')
            row['seconds'] = time.monotonic() - before
            atomic(path, row)
        rows.append({name: row.get(name) for name in ('id', 'family', 'status', 'passed', 'format_valid', 'seconds', 'cached', 'error')})
        atomic(out / 'summary.json', {'identity': identity, 'total': len(cases), 'completed': len(rows),
                                    'controls_passed': sum(r['passed'] is True for r in rows),
                                    'rows': rows, 'status': 'running' if len(rows) < len(cases) else 'completed',
                                    'scope': 'Frozen controls and unverified code drafts; no Product promotion'})
        print(f"Strata trial {len(rows)}/{len(cases)} {case['id']}: {row['status']} passed={row.get('passed')}", flush=True)
