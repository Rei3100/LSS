import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import strata_ledger as subject


def summary(tmp_path, rows, name='summary.json'):
    path = tmp_path / name
    path.write_text(json.dumps({'identity': {'engine': '0.1.55'}, 'tasks': [
        {'id': f't{i}', 'kind': kind, 'status': status, 'attempts': attempts, 'tokens': 100, 'seconds': 10.0}
        for i, (kind, status, attempts) in enumerate(rows)]}), encoding='utf-8')
    return path


def test_new_kind_is_on_probation_and_stays_open(tmp_path):
    ledger = tmp_path / 'l.jsonl'
    subject.ingest(summary(tmp_path, [('code', 'FAIL', 3)] * 3), ledger=ledger)
    found = subject.verdict('code', ledger=ledger)
    assert found['state'] == 'probation' and found['finished'] == 3


def test_kind_closes_when_too_few_tasks_pay(tmp_path):
    ledger = tmp_path / 'l.jsonl'
    rows = [('code', 'PASS', 1)] * 3 + [('code', 'PASS', 3), ('code', 'FAIL', 3), ('code', 'FAIL', 3)]
    subject.ingest(summary(tmp_path, rows), ledger=ledger)
    found = subject.verdict('code', ledger=ledger)
    assert found == {'kind': 'code', 'state': 'closed', 'paying': 3, 'finished': 6, 'failed': 2, 'seconds': 60.0}


def test_passing_on_a_late_attempt_does_not_pay(tmp_path):
    ledger = tmp_path / 'l.jsonl'
    subject.ingest(summary(tmp_path, [('test', 'PASS', 3)] * 6), ledger=ledger)
    assert subject.verdict('test', ledger=ledger)['state'] == 'closed'


def test_kinds_are_judged_apart_and_reopen_starts_a_new_window(tmp_path):
    ledger = tmp_path / 'l.jsonl'
    subject.ingest(summary(tmp_path, [('code', 'PASS', 1)] * 6 + [('test', 'FAIL', 3)] * 6), ledger=ledger)
    assert subject.verdict('code', ledger=ledger)['state'] == 'open'
    assert subject.verdict('test', ledger=ledger)['state'] == 'closed'
    subject.reopen('test', 'new engine 0.1.56', ledger=ledger)
    assert subject.verdict('test', ledger=ledger)['state'] == 'probation'
    assert subject.verdict('code', ledger=ledger)['state'] == 'open'


def test_gate_exit_code_follows_the_state(tmp_path, monkeypatch, capsys):
    ledger = tmp_path / 'l.jsonl'
    subject.ingest(summary(tmp_path, [('code', 'FAIL', 3)] * 6), ledger=ledger)
    monkeypatch.setattr(subject, 'LEDGER', ledger)
    monkeypatch.setattr(subject.verdict, '__kwdefaults__', {'ledger': ledger})
    assert subject.main(['gate', 'code']) == 1
    assert subject.main(['gate', 'test']) == 0


def test_rows_skip_a_torn_last_line(tmp_path):
    ledger = tmp_path / 'l.jsonl'
    subject.ingest(summary(tmp_path, [('code', 'PASS', 1)] * 2), ledger=ledger)
    with ledger.open('a', encoding='utf-8') as handle:
        handle.write('{"kind": "code", "paying"')   # an append cut off by a crash
    assert len(subject._rows(ledger)) == 2


def test_report_prints_one_verdict_per_recorded_kind(tmp_path, capsys, monkeypatch):
    ledger = tmp_path / 'l.jsonl'
    subject.ingest(summary(tmp_path, [('code', 'PASS', 1)] * 6 + [('test', 'FAIL', 3)] * 6), ledger=ledger)
    monkeypatch.setattr(subject, 'LEDGER', ledger)
    monkeypatch.setattr(subject.verdict, '__kwdefaults__', {'ledger': ledger})
    assert subject.main(['report']) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert {(row['kind'], row['state']) for row in lines} == {('code', 'open'), ('test', 'closed')}
    assert len(lines) == 2


def test_ledger_lives_in_the_main_checkout_of_a_worktree(tmp_path, monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        class Result:
            stdout = str(tmp_path / 'main' / '.git') + '\n'
        return Result()

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert subject._shared_root(tmp_path / 'worktree') == tmp_path / 'main'

    def raise_os_error(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(subprocess, 'run', raise_os_error)
    assert subject._shared_root(tmp_path / 'worktree') == tmp_path / 'worktree'
