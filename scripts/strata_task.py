"""Hand bounded coding jobs to the local Strata model in any repository.

    python scripts/strata_task.py tasks.jsonl --repo <path> --out <receipt dir> [--attempts 3]

Each JSONL row: id, file (repo-relative .py), function, spec (the contract), test (shell command run
in --repo that judges the result), and optional context_files (repo-relative, truncated to fit),
max_tokens, test_timeout. A test-writing task names test_file and test_function instead of
file and function; the draft is then spliced into test_file. One model load covers the whole file.
Strata drafts one top-level ``def``; it is spliced in, the test runs, and a failure restores the
original bytes and feeds the failure tail into the next attempt. A passing change stays in the
working tree, unstaged and uncommitted, for the caller to review. Only the target file is ever
written, and every path must stay inside --repo.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import os
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import local_ai_queue as queue

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CTX = 8192
CHARS_PER_TOKEN = 3          # conservative: code tokenises to roughly 3 to 4 characters
PROMPT_OVERHEAD_TOKENS = 200
TAIL_LINES = 60
SCHEMA = {'type': 'object', 'properties': {'code': {'type': 'string'}},
          'required': ['code'], 'additionalProperties': False}
FIELDS = {'id', 'file', 'function', 'spec', 'context_files', 'test', 'test_file',
          'test_function', 'max_tokens', 'test_timeout'}
SYSTEM = ('You write one Python function for a repository you cannot see beyond the excerpts given. '
          'Follow the contract exactly.')
CONTRACT = ('Output contract: reply with only a JSON object {"code": "<source>"} where <source> is '
            'exactly one top-level Python def named {name}, with its decorators if any, and nothing '
            'else: no imports, no module docstring, no other definitions, no prose.')


def sha(value):
    data = value if isinstance(value, bytes) else str(value).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


# ---- source splicing -------------------------------------------------------------------------

def _lines(text):
    return re.findall(r'[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+', text)


def _top_defs(tree, name):
    return [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]


def replace_function(source, name, new_code):
    """Return source with the top-level def ``name`` replaced (decorators included), else appended.

    ValueError when new_code is not exactly one def called ``name`` or when source already holds
    several top-level defs of that name. Methods and nested defs of the same name are ignored.
    Line endings of the original are kept.
    """
    try:
        new_tree = ast.parse(new_code.replace('\r\n', '\n'))
    except SyntaxError as exc:
        raise ValueError(f'new code does not parse: {exc}') from exc
    if len(new_tree.body) != 1 or len(_top_defs(new_tree, name)) != 1:
        raise ValueError(f'new code must be exactly one top-level def {name}, got '
                         f'{[type(n).__name__ for n in new_tree.body]}')
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f'target source does not parse: {exc}') from exc
    found = _top_defs(tree, name)
    if len(found) > 1:
        raise ValueError(f'{len(found)} top-level defs named {name} in target; refusing to guess')
    eol = '\r\n' if '\r\n' in source else ('\r' if '\r' in source and '\n' not in source else '\n')
    code = new_code.replace('\r\n', '\n').replace('\r', '\n').lstrip('\n').rstrip() + '\n'
    code = code.replace('\n', eol)
    if not found:
        if not source.strip():
            return code
        base = source if source.endswith(('\n', '\r')) else source + eol
        return base + eol + eol + code
    node = found[0]
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = _lines(source)
    return ''.join(lines[:first - 1]) + code + ''.join(lines[node.end_lineno:])


def function_source(source, name):
    """Current text of the top-level def (decorators included), or '' when it does not exist."""
    try:
        found = _top_defs(ast.parse(source), name)
    except SyntaxError:
        return ''
    if len(found) != 1:
        return ''
    node = found[0]
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return ''.join(_lines(source)[first - 1:node.end_lineno])


# ---- tasks -----------------------------------------------------------------------------------

def inside(repo, rel):
    """Resolve rel under repo; refuse absolute paths and any escape (including via links)."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or Path(rel).drive:
        raise ValueError(f'path must be repo-relative: {rel!r}')
    root = Path(repo).resolve()
    path = (root / rel).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'path escapes the repository: {rel!r}')
    return path


def read_tasks(path, repo):
    tasks, ids = [], set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - FIELDS:
            raise ValueError(f'line {number}: invalid task fields {sorted(set(row) - FIELDS) if isinstance(row, dict) else ""}')
        ident = row.get('id')
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError(f'line {number}: nonempty unique id required')
        writes_test = row.get('test_file') is not None
        file = row.get('test_file') if writes_test else row.get('file')
        function = row.get('test_function') if writes_test else row.get('function')
        if writes_test and not isinstance(function, str):
            raise ValueError(f'line {number}: test_file requires test_function')
        for key, value in (('spec', row.get('spec')), ('test', row.get('test'))):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'line {number}: {key} required')
        if not isinstance(function, str) or not function.isidentifier():
            raise ValueError(f'line {number}: function must be an identifier')
        if not isinstance(file, str) or not file.endswith('.py'):
            raise ValueError(f'line {number}: target must be a repo-relative .py path')
        inside(repo, file)
        context = row.get('context_files') or []
        if not isinstance(context, list) or not all(isinstance(c, str) for c in context):
            raise ValueError(f'line {number}: context_files must be a list of paths')
        for item in context:
            inside(repo, item)
        tokens = row.get('max_tokens', 1536)
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= 8192:
            raise ValueError(f'line {number}: max_tokens must be 1..8192')
        timeout = row.get('test_timeout', 600)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError(f'line {number}: test_timeout must be positive')
        tasks.append({'id': ident, 'file': file, 'function': function, 'spec': row['spec'],
                      'test': row['test'], 'context_files': context, 'max_tokens': tokens,
                      'test_timeout': timeout, 'writes_test': writes_test})
        ids.add(ident)
    if not tasks:
        raise ValueError('task file is empty')
    return tasks


def tail(text, limit):
    kept = '\n'.join(text.splitlines()[-TAIL_LINES:])
    return kept if len(kept) <= limit else '...' + kept[-limit:]


def build_prompt(task, current, contexts, feedback, ctx, max_tokens):
    """Messages that fit ctx. The spec is never cut; context files shrink first, then feedback, then source."""
    name = task['function']
    budget = max(0, (ctx - max_tokens - PROMPT_OVERHEAD_TOKENS) * CHARS_PER_TOKEN)
    contract = CONTRACT.replace('{name}', name)
    head = f"Task contract:\n{task['spec']}\n\nTarget: {task['file']} :: {name}\n"
    fixed = len(SYSTEM) + len(head) + len(contract)
    room = budget - fixed
    feedback_text = ''
    if feedback:
        allowance = max(400, room // 3)
        feedback_text = ('\nYour previous attempt failed. Last output of the judging command:\n'
                         + tail(feedback, allowance) + '\n')
        room -= len(feedback_text)
    source_text = ''
    if current:
        source_text = f'\nCurrent source of {name} (replace it):\n{current}\n'
    else:
        source_text = f'\n{name} does not exist yet in {task["file"]}.\n'
    if len(source_text) > max(0, room):
        source_text = source_text[:max(200, room)] + '\n...[truncated]\n'
    room -= len(source_text)
    blocks = []
    usable = [(rel, text) for rel, text in contexts if text]
    if usable and room > 200:
        share = room // len(usable)
        for rel, text in usable:
            header = f'\nContext excerpt from {rel}:\n'
            allowed = share - len(header) - 20
            if allowed < 80:
                continue
            body = text if len(text) <= allowed else text[:allowed] + '\n...[truncated]'
            blocks.append(header + body + '\n')
    user = head + source_text + ''.join(blocks) + feedback_text + '\n' + contract
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]


def run_tests(command, repo, timeout):
    try:
        done = subprocess.run(command, shell=True, cwd=repo, capture_output=True, text=True,
                              errors='replace', timeout=timeout)
        return done.returncode, (done.stdout or '') + (done.stderr or '')
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b'').decode('utf-8', 'replace')
        return None, (out or '') + f'\n[test command timed out after {timeout}s]'


JUDGE_RAN = (0, 5)   # pytest: all passed / nothing collected


def preflight_judges(tasks, repo):
    """Run each test-writing task's judge once on the unchanged file before any GPU or ledger work.

    Before the new test exists the file's other tests must pass (or none are collected); anything else means
    the judge itself cannot run (a shell that cannot start ``.venv/Scripts/python.exe``, a broken import) and
    every draft would be recorded as a Strata failure without ever being judged. Code-writing judges may fail
    legitimately before the draft exists, so they are not preflighted.
    """
    for task in tasks:
        if not task.get('writes_test'):
            continue
        code, output = run_tests(task['test'], repo, task['test_timeout'])
        if code not in JUDGE_RAN:
            raise ValueError(f"judge for {task['id']} does not run on the unchanged file (exit {code}): "
                             f"{tail(output, 600).strip()}")


def restore(path, original):
    if original is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(original)


def run_task(task, repo, url, *, complete, extra, ctx, attempts):
    target = inside(repo, task['file'])
    start = time.monotonic()
    receipt = {'id': task['id'], 'file': task['file'], 'function': task['function'],
               'test': task['test'], 'attempts': [], 'status': 'FAIL', 'tokens': 0,
               'kind': 'test' if task.get('writes_test') else 'code'}
    original = target.read_bytes() if target.exists() else None
    contexts = []
    for rel in task['context_files']:
        try:
            contexts.append((rel, inside(repo, rel).read_text(encoding='utf-8', errors='replace')))
        except OSError:
            contexts.append((rel, ''))
    max_tokens = min(task['max_tokens'], max(64, ctx // 2))
    feedback = ''
    try:
        for number in range(1, attempts + 1):
            record = {'attempt': number}
            receipt['attempts'].append(record)
            before = time.monotonic()
            text = original.decode('utf-8') if original is not None else ''
            messages = build_prompt(task, function_source(text, task['function']), contexts,
                                    feedback, ctx, max_tokens)
            record['prompt_sha256'] = sha(json.dumps(messages, sort_keys=True))
            record['prompt_chars'] = sum(len(m['content']) for m in messages)
            try:
                reply = complete(url, messages, SCHEMA, max_tokens=max_tokens, extra=extra)
                record['response_sha256'] = sha(reply.text)
                usage = getattr(reply, 'usage', None) or {}
                record['tokens'] = usage.get('total_tokens') or (
                    (usage.get('prompt_tokens') or 0) + (usage.get('completion_tokens') or 0))
                receipt['tokens'] += record['tokens']
                finish = ((reply.raw or {}).get('choices') or [{}])[0].get('finish_reason')
                record['finish_reason'] = finish
                if finish not in (None, 'stop'):
                    raise ValueError(f'reply ended with finish_reason={finish}; shorten the function')
                queue.check_python_def({'python_def': task['function']}, {'text': reply.text})
                code = json.loads(reply.text)['code']
                new_text = replace_function(text, task['function'], code)
                record['code_sha256'] = sha(code)
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                feedback = f'Your reply was rejected before testing: {record["error"]}'
                record['seconds'] = round(time.monotonic() - before, 3)
                continue
            target.write_bytes(new_text.encode('utf-8'))
            code_exit, output = run_tests(task['test'], repo, task['test_timeout'])
            record['test_exit'] = code_exit
            record['test_tail'] = tail(output, 4000)
            record['seconds'] = round(time.monotonic() - before, 3)
            if code_exit == 0:
                receipt['status'] = 'PASS'
                receipt['code_sha256'] = record['code_sha256']
                return receipt
            restore(target, original)
            feedback = output if output.strip() else f'test exited with {code_exit} and no output'
    finally:
        if receipt['status'] != 'PASS':
            restore(target, original)
        receipt['seconds'] = round(time.monotonic() - start, 3)
    return receipt


def safe_name(ident):
    return re.sub(r'[^A-Za-z0-9._-]', '_', ident)[:80] + '-' + sha(ident)[:8]


def run_tasks(tasks, repo, out, *, serving, complete, extra, identify, attempts=3, context_size=None):
    repo, out = Path(repo).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    receipts = []
    fingerprint = identify()
    with serving('strata') as url:
        ctx = context_size(url) if callable(context_size) else (context_size or DEFAULT_CTX)
        for task in tasks:
            receipt = run_task(task, repo, url, complete=complete, extra=extra, ctx=ctx, attempts=attempts)
            queue.atomic_json(out / (safe_name(task['id']) + '.json'), receipt)
            receipts.append(receipt)
    summary = {'identity': fingerprint, 'repo': str(repo), 'ctx': ctx, 'attempts_allowed': attempts,
               'tasks': [{'id': r['id'], 'kind': r.get('kind', 'code'), 'status': r['status'], 'attempts': len(r['attempts']),
                          'tokens': r['tokens'], 'seconds': r['seconds']} for r in receipts],
               'status': 'PASS' if all(r['status'] == 'PASS' for r in receipts) else 'FAIL',
               'seconds': round(time.monotonic() - start, 3)}
    queue.atomic_json(out / 'summary.json', summary)
    return summary


def human(summary):
    lines = [f"{item['status']:4} {item['id']}  attempts={item['attempts']} tokens={item['tokens']} "
             f"{item['seconds']}s" for item in summary['tasks']]
    passed = sum(item['status'] == 'PASS' for item in summary['tasks'])
    lines.append(f"{summary['status']}: {passed}/{len(summary['tasks'])} tasks, {summary['seconds']}s")
    return '\n'.join(lines)


def server_context(_url):
    """The running server's context size (/props), else the profile's 8K."""
    import strata_dev as runtime
    try:
        props = runtime.request('/props', timeout=30)
        value = props.get('default_generation_settings', {}).get('n_ctx') or props.get('n_ctx')
        if isinstance(value, int) and value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_CTX


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks')
    parser.add_argument('--repo', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--profile', help='Strata model profile (coder, qwen36); default: the policy profile, else coder')
    parser.add_argument('--no-ledger', action='store_true',
                        help='do not append to the shared ledger (comparisons and replays, not cloud work saved)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.profile:
        os.environ['LSS_STRATA_PROFILE'] = args.profile   # inherited by the gpu_run worker
    if not 1 <= args.attempts <= 10:
        parser.error('--attempts must be 1..10')
    try:
        repo = Path(args.repo).resolve()
        if not repo.is_dir():
            raise ValueError(f'--repo is not a directory: {repo}')
        tasks = read_tasks(args.tasks, repo)
        out = Path(args.out).resolve()
        if out == repo:
            raise ValueError('--out must not be the repository root')
        if not args.worker:
            preflight_judges(tasks, repo)
        import strata_queue
        import strata_dev as runtime
        if args.worker:
            summary = run_tasks(tasks, repo, out, serving=strata_queue.serving,
                                complete=strata_queue.complete, extra=strata_queue.DEFAULTS,
                                identify=lambda: strata_queue.identity(strata_queue.model()),
                                attempts=args.attempts, context_size=server_context)
            print(human(summary))
            return 0 if summary['status'] == 'PASS' else 1
        from dev_quiet_mode import GPUDeferred, require_gpu_allowed
        try:
            require_gpu_allowed()
        except GPUDeferred as exc:
            print(str(exc), file=sys.stderr)
            return 3
        out.mkdir(parents=True, exist_ok=True)
        profile = runtime.active_profile()
        home, model_path = runtime.paths(profile)
        command = [sys.executable, str(ROOT / 'scripts/gpu_run.py'), '--purpose', 'Strata coding tasks',
                   '--owned-job', '--cap-receipt', str(out / 'cap.json'), '--duty', str(runtime.study_duty()),
                   '--protect', str(home), '--protect', str(model_path.parent), '--protect', str(out), '--',
                   sys.executable, str(Path(__file__).resolve()), str(Path(args.tasks).resolve()),
                   '--repo', str(repo), '--out', str(out), '--attempts', str(args.attempts), '--worker']
        command[4:4] = runtime.performance_args()
        status = runtime.supervise_trial(command, out, admission=runtime.admission_requirements(profile))
        if (out / 'summary.json').is_file() and not args.no_ledger:
            import strata_ledger
            strata_ledger.ingest(out / 'summary.json', note=None if profile == 'coder' else f'profile {profile}')   # net-value ledger: strata_ledger.py gate <kind>
        return status
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
