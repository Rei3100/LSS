import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import batch
import codex_focus


def queue(tmp_path, name, kind='primary', primary=None):
    brief = tmp_path / 'brief.json'
    if not brief.exists():
        brief.write_text(json.dumps({'user_request': 'verbatim request', 'requested_outcomes': ['result']}), encoding='utf-8')
    q = {'steps': [{'name': 'step', 'cmd': ['payload']}], 'done': name + '.done',
         'focus': {'objective': 'objective', 'user_result': 'result',
                   'kind': kind, 'done_when': 'finished', 'return_to': 'primary', 'brief_path': brief.name},
         'review_path': name + '.review.json'}
    if primary:
        q['primary_queue'] = primary.name
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(q), encoding='utf-8')
    (tmp_path / q['review_path']).write_text(json.dumps({'queue_sha256': codex_focus.canonical_sha(q),
        'brief_sha256': hashlib.sha256(brief.read_bytes()).hexdigest(), 'decision': 'ALLOW',
        'reviewer': 'independent', 'requested_outcome': 'result', 'why_required_now': 'needed',
        'what_changes_next': 'primary advances'}), encoding='utf-8')
    return path, q


def test_denied_run_and_start_never_launch(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_SESSION_ID', 'session')
    path, q = queue(tmp_path, 'bad')
    (tmp_path / q['review_path']).unlink()
    import hidden_run
    with patch.object(batch, 'run_command', side_effect=AssertionError('payload started')), \
         patch.object(hidden_run, 'main', side_effect=AssertionError('background started')):
        assert batch.run_queue(path) == 1
        assert batch.start_queue(path) == 1
    assert json.loads(path.with_suffix('.receipt.json').read_text())['status'].startswith('failed Codex focus')
    assert (tmp_path / 'bad.done').read_text().startswith('status failed')


def test_denial_preserves_prior_receipt_and_marker(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_SESSION_ID', 'session')
    path, q = queue(tmp_path, 'past')
    receipt = path.with_suffix('.receipt.json')
    marker = tmp_path / q['done']
    receipt.write_bytes(b'prior completed execution')
    marker.write_bytes(b'status ok\n')
    (tmp_path / q['review_path']).unlink()
    assert batch.run_queue(path) == 1
    assert receipt.read_bytes() == b'prior completed execution'
    assert marker.read_bytes() == b'status ok\n'


def test_self_review_and_unrequested_outcome_are_denied(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_SESSION_ID', 'session')
    path, q = queue(tmp_path, 'self')
    review_path = tmp_path / q['review_path']
    original = json.loads(review_path.read_text())
    for changes in ({'reviewer': '/root'}, {'requested_outcome': 'more benchmark tooling'}):
        review_path.write_text(json.dumps({**original, **changes}))
        try:
            codex_focus.admit(path, q)
            assert False, 'invalid review was accepted'
        except ValueError:
            pass


def test_non_codex_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv('CODEX_SESSION_ID', raising=False)
    monkeypatch.delenv('CODEX_THREAD_ID', raising=False)
    path = tmp_path / 'plain.json'
    path.write_text(json.dumps({'steps': [{'name': 'step', 'cmd': ['payload']}]}))
    with patch.object(batch, 'run_command', return_value=0) as run:
        assert batch.run_queue(path) == 0
        run.assert_called_once()


def test_multiple_support_and_stale_brief(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_SESSION_ID', 'session')
    primary, _ = queue(tmp_path, 'primary')
    support, sq = queue(tmp_path, 'support', 'support', primary)
    codex_focus.admit(support, sq)
    codex_focus.admit(support, sq)
    (tmp_path / 'brief.json').write_text(json.dumps({'user_request': 'verbatim request',
        'requested_outcomes': ['result', 'new']}), encoding='utf-8')
    try:
        codex_focus.admit(support, sq)
        assert False, 'changed brief was accepted'
    except ValueError:
        pass


def test_review_binding_and_no_nested_support(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_THREAD_ID', 'thread')
    monkeypatch.delenv('CODEX_SESSION_ID', raising=False)
    primary, pq = queue(tmp_path, 'primary')
    support, sq = queue(tmp_path, 'support', 'support', primary)
    nested, nq = queue(tmp_path, 'nested', 'support', support)
    for path, value in [(primary, {**pq, 'done': 'changed'}), (nested, nq)]:
        try:
            codex_focus.admit(path, value)
            assert False, 'admission unexpectedly succeeded'
        except ValueError:
            pass
