import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import strata_task as subject

PY = sys.executable.replace('\\', '/')
SOURCE = 'def add(a, b):\n    return 0\n\n\ndef other():\n    return 1\n'


def reply(code, finish='stop'):
    text = json.dumps({'code': code})
    return SimpleNamespace(text=text, usage={'total_tokens': 10},
                           raw={'choices': [{'finish_reason': finish, 'message': {'content': text}}]})


def harness(tmp_path, replies, source=SOURCE, **task_extra):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'm.py').write_bytes(source.encode())
    prompts, loads = [], []

    @contextlib.contextmanager
    def serving(name, **kw):
        loads.append(name)
        yield 'url'

    queue_ = list(replies)

    def complete(url, messages, schema, *, max_tokens, extra):
        prompts.append(messages)
        return queue_.pop(0)

    test = (f'"{PY}" -c "import sys; sys.path.insert(0, \'.\'); import m; '
            f'sys.exit(0 if m.add(2, 3) == 5 else 1)"')
    task = {'id': 't1', 'file': 'm.py', 'function': 'add', 'spec': 'add two numbers SPEC-MARK',
            'test': test, 'context_files': [], 'max_tokens': 256, 'test_timeout': 60,
            'writes_test': False, **task_extra}

    def go(tasks=None, attempts=3, ctx=8192):
        return subject.run_tasks(tasks or [task], repo, tmp_path / 'out', serving=serving, complete=complete,
                                 extra={}, identify=lambda: {'model': 'fake'}, attempts=attempts,
                                 context_size=ctx)
    return repo, prompts, loads, go


GOOD = 'def add(a, b):\n    return a + b'
BAD = 'def add(a, b):\n    return a - b'


def test_success_first_try(tmp_path):
    repo, prompts, loads, go = harness(tmp_path, [reply(GOOD)])
    summary = go()
    assert summary['status'] == 'PASS' and loads == ['strata'] and len(prompts) == 1
    assert (repo / 'm.py').read_text() == 'def add(a, b):\n    return a + b\n\n\ndef other():\n    return 1\n'
    receipt = json.loads(next((tmp_path / 'out').glob('t1-*.json')).read_text())
    assert receipt['status'] == 'PASS' and receipt['attempts'][0]['test_exit'] == 0
    assert len(receipt['attempts'][0]['prompt_sha256']) == 64
    assert (tmp_path / 'out/summary.json').exists()


def test_failure_feedback_then_success(tmp_path):
    repo, prompts, loads, go = harness(tmp_path, [reply(BAD), reply(GOOD)])
    assert go()['status'] == 'PASS'
    assert 'Your previous attempt failed' not in prompts[0][1]['content']
    assert 'Your previous attempt failed' in prompts[1][1]['content']
    assert loads == ['strata']
    assert 'return a + b' in (repo / 'm.py').read_text()


def test_all_attempts_fail_leaves_bytes_identical(tmp_path):
    repo, prompts, loads, go = harness(tmp_path, [reply(BAD), reply('def nope(): pass'), reply('not json')],
                                       source=SOURCE.replace('\n', '\r\n'))
    before = (repo / 'm.py').read_bytes()
    summary = go()
    assert summary['status'] == 'FAIL' and len(prompts) == 3
    assert (repo / 'm.py').read_bytes() == before
    assert not list(repo.glob('*.partial'))


def test_path_escape_refused(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    for bad in ('../x.py', str(tmp_path / 'x.py'), '.'):
        with pytest.raises(ValueError):
            subject.inside(repo, bad)
    tasks = tmp_path / 'tasks.jsonl'
    tasks.write_text(json.dumps({'id': 'a', 'file': '../evil.py', 'function': 'f', 'spec': 's', 'test': 't'}))
    with pytest.raises(ValueError):
        subject.read_tasks(tasks, repo)
    tasks.write_text(json.dumps({'id': 'a', 'file': 'ok.py', 'function': 'f', 'spec': 's', 'test': 't',
                                 'context_files': ['../../secret.txt']}))
    with pytest.raises(ValueError):
        subject.read_tasks(tasks, repo)


def test_test_file_task_targets_test_file(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    tasks = tmp_path / 'tasks.jsonl'
    tasks.write_text(json.dumps({'id': 'a', 'file': 'src.py', 'function': 'f', 'spec': 's', 'test': 't',
                                 'test_file': 'tests/test_x.py', 'test_function': 'test_f'}))
    task = subject.read_tasks(tasks, repo)[0]
    assert (task['file'], task['function'], task['writes_test']) == ('tests/test_x.py', 'test_f', True)


def test_replace_function_cases():
    r = subject.replace_function
    dec = '@staticmethod\ndef f():\n    return 1\n\n\nx = 2\n'
    assert r(dec, 'f', 'def f():\n    return 9') == 'def f():\n    return 9\n\n\nx = 2\n'
    crlf = 'def f():\r\n    return 1\r\n\r\n\r\ndef g():\r\n    pass\r\n'
    out = r(crlf, 'f', 'def f():\n    return 2\n    ')
    assert out == 'def f():\r\n    return 2\r\n\r\n\r\ndef g():\r\n    pass\r\n'
    assert r('x = 1\n', 'f', 'def f():\n    pass') == 'x = 1\n\n\ndef f():\n    pass\n'
    assert r('x = 1', 'f', 'def f():\n    pass') == 'x = 1\n\n\ndef f():\n    pass\n'
    with pytest.raises(ValueError):
        r('x = 1\n', 'f', 'def g():\n    pass')
    with pytest.raises(ValueError):
        r('x = 1\n', 'f', 'import os\ndef f():\n    pass')
    with pytest.raises(ValueError):
        r('def f():\n    pass\n\n\ndef f():\n    pass\n', 'f', 'def f():\n    pass')
    methods = 'class C:\n    def f(self):\n        return 1\n\n\ndef h():\n    def f():\n        return 2\n'
    assert r(methods, 'f', 'def f():\n    pass') == methods + '\n\ndef f():\n    pass\n'
    decorated_new = r('def f():\n    pass\n', 'f', '@dec\ndef f():\n    pass')
    assert decorated_new == '@dec\ndef f():\n    pass\n'


def test_decorators_of_old_function_are_replaced_whole():
    src = '@a\n@b\ndef f():\n    pass\nz = 1\n'
    assert subject.replace_function(src, 'f', 'def f():\n    return 1') == 'def f():\n    return 1\nz = 1\n'


def test_context_truncation_keeps_spec():
    task = {'function': 'f', 'file': 'm.py', 'spec': 'SPEC ' * 40}
    big = [('a.py', 'A' * 100000), ('b.py', 'B' * 100000)]
    messages = subject.build_prompt(task, 'def f():\n    pass\n', big, 'E' * 50000, 2048, 512)
    text = messages[1]['content']
    assert task['spec'] in text
    assert 'def f():' in text and 'exactly one top-level Python def named f' in text
    assert len(text) < (2048 - 512) * subject.CHARS_PER_TOKEN
    assert '...[truncated]' in text
    assert text.count('E') < 50000 / 4


def test_preflight_rejects_a_test_judge_that_cannot_run(tmp_path):
    broken = {'id': 'broken', 'writes_test': True, 'test': f'"{PY}" -c "import sys; sys.exit(9)"', 'test_timeout': 30}
    with pytest.raises(ValueError, match='does not run on the unchanged file'):
        subject.preflight_judges([broken], tmp_path)


def test_preflight_accepts_running_judges_and_skips_code_tasks(tmp_path):
    passing = {'id': 'ok', 'writes_test': True, 'test': f'"{PY}" -c "import sys; sys.exit(5)"', 'test_timeout': 30}
    code_task = {'id': 'code', 'writes_test': False, 'test': f'"{PY}" -c "import sys; sys.exit(1)"', 'test_timeout': 30}
    subject.preflight_judges([passing, code_task], tmp_path)
