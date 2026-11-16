from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_ai_aider_trial as trial


@pytest.fixture
def case(tmp_path):
    root = tmp_path / 'case'
    root.mkdir()
    (root / 'candidate.py').write_text('def _normalise(text):\n    return text\n', encoding='utf-8')
    (root / 'contract.txt').write_text('contract', encoding='utf-8')
    return trial.CasePaths(root)


@pytest.mark.parametrize('url', ['https://127.0.0.1:1/v1', 'http://localhost:3/v1', 'http://127.0.0.1/v1',
    'http://127.0.0.1:3/v1?x=1', 'http://user@127.0.0.1:3/v1', 'http://127.0.0.1:3/v1/', 'http://127.0.0.1:65536/v1'])
def test_only_exact_loopback_url(url):
    with pytest.raises(trial.GuardError): trial.local_base_url(url)
    assert trial.local_base_url('http://127.0.0.1:8080/v1')


def test_path_allowlist_rejects_outside_parent_unknown_and_alias(case, tmp_path):
    outside = tmp_path / 'outside.py'
    outside.write_text('secret')
    (case.root / 'unknown.py').write_text('secret')
    for path in [outside, '../outside.py', 'unknown.py', 'candidate.py/../candidate.py']:
        with pytest.raises((trial.GuardError, OSError)): case.resolve(path)
    assert case.resolve('candidate.py') == case.root / 'candidate.py'


def test_symlink_to_external_is_rejected(case, tmp_path, monkeypatch):
    # Patch path status instead of requiring Windows symlink privileges.
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda p: p == case.root / 'candidate.py' or original(p))
    with pytest.raises(trial.GuardError): case.resolve('candidate.py')


def test_guarded_coder_rejects_unknown_before_any_read_and_proposes_in_memory(case):
    reads = []
    class IO:
        def read_text(self, filename, *args, **kwargs):
            reads.append(filename)
            return Path(filename).read_text(encoding='utf-8')
    class Coder:
        def get_edits(self): return self.edits
    def parse(content, fence, valid):
        assert valid == []
        return content
    coder_type, io_type = trial.guarded_types(Coder, IO, parse)
    io = io_type(); io.case_paths = case
    coder = coder_type(); coder.io = io; coder.case_paths = case; coder.shell_commands = []; coder.fence = ('```', '```')
    coder.partial_response_content = [('unknown.py', 'return text', 'return text.strip()')]
    with pytest.raises((trial.GuardError, OSError)): coder.apply_updates()
    assert reads == []
    coder.partial_response_content = [('candidate.py', 'return text', 'return text.strip()')]
    assert coder.apply_updates() == set()
    assert 'return text.strip()' in coder.proposal
    assert 'return text.strip()' not in (case.root / 'candidate.py').read_text()
    with pytest.raises(trial.GuardError): io.write_text('candidate.py', 'x')
    assert not io.confirm_ask('anything')


def test_candidate_structure_and_shell_commands_are_refused(case):
    class IO:
        def read_text(self, filename, *a, **kw): return Path(filename).read_text()
    class Coder: pass
    def parser(content, fence, valid): return content
    coder_type, io_type = trial.guarded_types(Coder, IO, parser)
    io = io_type(); io.case_paths = case
    coder = coder_type(); coder.io = io; coder.case_paths = case; coder.shell_commands = []; coder.fence = ('```', '```')
    coder.partial_response_content = [(None, 'rm file')]
    with pytest.raises(trial.GuardError, match='shell'): coder.apply_updates()
    coder.partial_response_content = [('candidate.py', 'return text', 'return text\ndef added(): pass')]
    with pytest.raises(trial.GuardError, match='one _normalise'): coder.apply_updates()
    assert (case.root / 'candidate.py').read_text().endswith('return text\n')


def test_response_rejects_prose_unknown_paths_and_shell():
    good = 'candidate.py\n```python\n<<<<<<< SEARCH\nreturn x\n=======\nreturn y\n>>>>>>> REPLACE\n```'
    assert trial.single_edit_response(good) == good
    for bad in [good + '\nsecret.py', good.replace('candidate.py', '../secret.py'),
                good + '\n```sh\nwhoami\n```']:
        with pytest.raises(trial.GuardError): trial.single_edit_response(bad)


def test_request_is_loopback_dummy_key_and_bounded(monkeypatch):
    import http.client
    calls = []
    good = 'candidate.py\n```\n<<<<<<< SEARCH\nreturn x\n=======\nreturn y\n>>>>>>> REPLACE\n```'
    class Response:
        status = 200
        def read(self, limit):
            assert limit == 250001
            import json
            return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': good}}]}).encode()
    class Connection:
        def __init__(self, host, port, timeout): calls.append((host, port, timeout))
        def request(self, method, path, body, headers): calls.append((method, path, headers))
        def getresponse(self): return Response()
        def close(self): calls.append('closed')
    monkeypatch.setattr(http.client, 'HTTPConnection', Connection)
    assert trial.request_local('http://127.0.0.1:1234/v1', [], 4) == good
    assert calls[0] == ('127.0.0.1', 1234, 4)
    assert calls[1][2]['Authorization'] == 'Bearer local-dummy'
    assert calls[-1] == 'closed'


def test_request_capture_survives_format_and_finish_rejection(monkeypatch):
    import hashlib
    import http.client
    import json
    raw = json.dumps({'choices':[{'finish_reason':'length', 'message':{'content':'partial'}}]}).encode()
    class Response:
        status = 200
        def read(self, limit): return raw
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, *args, **kwargs): pass
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(http.client, 'HTTPConnection', Connection)
    capture = {}
    with pytest.raises(trial.GuardError, match='incomplete'):
        trial.request_local('http://127.0.0.1:44/v1', [], capture=capture)
    assert capture['raw_response_utf8'] == raw.decode()
    assert capture['raw_response_sha256'] == hashlib.sha256(raw).hexdigest()
    assert len(capture['request_sha256']) == 64
