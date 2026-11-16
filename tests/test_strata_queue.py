import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import strata_queue as subject
from quiet_cpu import quiet_wall_timeout


@pytest.fixture(autouse=True)
def quiet_policy(tmp_path, monkeypatch):
    import dev_quiet_mode
    monkeypatch.setattr(dev_quiet_mode, 'POLICY', tmp_path / 'absent.json')


def test_queue_load_once_resume_validation_and_attempts(tmp_path, monkeypatch):
    pytest.importorskip('jsonschema')
    calls = []
    @contextlib.contextmanager
    def serve(*args, **kwargs):
        calls.append('load')
        yield 'local'
    def reply(text, finish='stop'):
        return SimpleNamespace(text=text, usage={}, raw={'choices': [{'finish_reason': finish, 'message': {'content': text}}]})
    response = [reply('{"value": 3}')]
    def complete(*args, **kwargs):
        calls.append('request')
        return response[0]
    monkeypatch.setattr(subject, 'serving', serve)
    monkeypatch.setattr(subject, 'complete', complete)
    requests = [{'id': 'a', 'messages': [{'role':'user','content':'x'}], 'max_tokens':20,
                 'schema': {'type':'object','properties':{'value':{'type':'integer'}},'required':['value']}}]
    result = subject.run_worker(requests, tmp_path, {'model_sha256':'one'})
    assert result['status'] == 'drafts_ready'
    assert subject.run_worker(requests, tmp_path, {'model_sha256':'one'})['cached'] == 1
    assert calls == ['load', 'request']
    response[0] = reply('{"value":"bad"}')
    bad = subject.run_worker(requests, tmp_path, {'model_sha256':'two'})
    assert bad['status'] == 'failed'
    receipt = Path(bad['entries'][0]['receipt'])
    assert json.loads(receipt.read_text(encoding='utf-8'))['text'] == '{"value":"bad"}'
    response[0] = reply('{"value":', 'length')
    assert subject.run_worker(requests, tmp_path, {'model_sha256':'two'})['status'] == 'failed'
    assert len(list((tmp_path/'attempts').glob('*.json'))) == 1
    response[0] = reply('{"value": 4}')
    assert subject.run_worker(requests, tmp_path, {'model_sha256':'two'})['cached'] == 0
    assert len(list((tmp_path/'attempts').glob('*.json'))) == 2


def test_external_refs_are_rejected():
    pytest.importorskip('jsonschema')
    for schema in ({'$ref':'https://example.com/a'}, {'$dynamicRef':'//example.com/a'}, {'$id':'https://example.com/'}):
        with pytest.raises(ValueError):
            subject.check_schemas([{'schema':schema}])
    subject.check_schemas([{'schema':{'$ref':'#/$defs/a', '$defs':{'a':{'type':'string'}}}}])


def test_file_hash_cache_rehashes_on_change(tmp_path):
    path = tmp_path / 'weights'
    path.write_bytes(b'original')
    cache = {}
    first = subject.file_hash(path, cache)
    assert subject.file_hash(path, cache) == first
    path.write_bytes(b'changed')
    assert subject.file_hash(path, cache) != first


def test_rejects_unsupported_modes(tmp_path):
    with pytest.raises(SystemExit):
        subject.main(['unused', '--out', str(tmp_path), '--cpu'])
    with pytest.raises(SystemExit):
        subject.main(['unused', '--out', str(tmp_path), '--profile', 'test-vectors'])


def test_quiet_policy_defers_before_identity_or_hashing(tmp_path, monkeypatch, capsys):
    requests = tmp_path / 'requests.json'
    requests.write_text('{"id":"a","messages":[{"role":"user","content":"x"}]}\n')
    monkeypatch.setattr(subject.strata_trials, 'output_directory', lambda path: path)
    import dev_quiet_mode
    monkeypatch.setattr(dev_quiet_mode, 'require_gpu_allowed', lambda: (_ for _ in ()).throw(dev_quiet_mode.GPUDeferred('quiet')))
    monkeypatch.setattr(subject, 'identity', lambda *_: pytest.fail('identity must not run'))
    monkeypatch.setattr(subject, 'file_hash', lambda *_: pytest.fail('hashing must not run'))
    monkeypatch.setattr(subject.runtime, 'supervise_trial', lambda *a, **k: pytest.fail('server path must not run'))
    assert subject.main([str(requests), '--out', str(tmp_path / 'out')]) == 3
    assert 'quiet' in capsys.readouterr().err


def test_declared_draft7_dependencies_and_unsupported_dialect():
    schema = {'$schema': 'http://json-schema.org/draft-07/schema#', 'type': 'object', 'dependencies': {'a': ['b']}}
    subject.check_schemas([{'schema': schema}])
    with pytest.raises(Exception):
        subject.validate_receipt({'schema': schema}, {'text': '{"a":1}'})
    subject.validate_receipt({'schema': schema}, {'text': '{"a":1,"b":2}'})
    with pytest.raises(ValueError, match='dialect'):
        subject.check_schemas([{'schema': {'$schema': 'https://unknown.invalid/schema'}}])


def test_failed_start_does_not_stop_unowned_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, 'verify_owned_worker', lambda: None)
    calls = []
    monkeypatch.setattr(subject.runtime, 'paths', lambda _: (tmp_path, tmp_path / 'model'))
    monkeypatch.setattr(subject.runtime, 'start_ready', lambda *args: (_ for _ in ()).throw(RuntimeError('startup failed')))
    monkeypatch.setattr(subject.runtime, 'cli', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match='startup failed'):
        with subject.serving('strata'):
            pass
    assert calls == []


def test_unowned_listener_cannot_serve(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, 'verify_owned_worker', lambda: None)
    monkeypatch.setattr(subject.runtime, 'paths', lambda _: (tmp_path, tmp_path / 'model'))
    monkeypatch.setattr(subject.runtime, 'start_ready', lambda *args: None)
    monkeypatch.setattr(subject, 'verify_owned_listener', lambda port: (_ for _ in ()).throw(RuntimeError('unowned')))
    with pytest.raises(RuntimeError, match='unowned'):
        with subject.serving('strata'):
            pytest.fail('unowned listener cannot serve')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Job ownership')
def test_listener_job_membership_cpu_only():
    import subprocess
    from strata_process_job import OwnedProcessJob
    code = ("import socket,sys; sys.path.insert(0,sys.argv[1]); "
            "import strata_queue as q; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
            "assert q.verify_owned_listener(s.getsockname()[1]) == __import__('os').getpid(); print('owned',flush=True)")
    job = OwnedProcessJob()
    try:
        process = job.start([sys.executable, '-c', code, str(Path(subject.__file__).parent)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = process.communicate(timeout=quiet_wall_timeout(15))
        assert process.returncode == 0, errors.decode(errors='replace')
        assert output.strip() == b'owned'
    finally:
        job.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Job ownership')
def test_listener_outside_job_is_rejected_cpu_only():
    import socket
    import subprocess
    from strata_process_job import OwnedProcessJob
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        code = ("import sys; sys.path.insert(0,sys.argv[1]); import strata_queue as q; "
                "q.verify_owned_listener(int(sys.argv[2]))")
        job = OwnedProcessJob()
        try:
            process = job.start([sys.executable, '-c', code, str(Path(subject.__file__).parent),
                                 str(listener.getsockname()[1])], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, errors = process.communicate(timeout=quiet_wall_timeout(15))
            assert process.returncode != 0
            assert b'not a live member' in errors
        finally:
            job.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Job ownership')
def test_nested_jobs_verify_inner_not_outer():
    import subprocess
    from strata_process_job import OwnedProcessJob
    inner = ("import socket,os,sys;sys.path.insert(0,sys.argv[1]);import strata_queue as q;"
             "s=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
             "assert q.verify_owned_listener(s.getsockname()[1])==os.getpid();print('nested-owned',flush=True)")
    outer = ("import sys,subprocess;sys.path.insert(0,sys.argv[1]);from strata_process_job import OwnedProcessJob;"
             "j=OwnedProcessJob();p=j.start([sys.executable,'-c',sys.argv[2],sys.argv[1]],"
             "stdout=subprocess.PIPE,stderr=subprocess.PIPE);from quiet_cpu import quiet_wall_timeout;out,err=p.communicate(timeout=quiet_wall_timeout(15));"
             "j.close();sys.stdout.buffer.write(out);sys.stderr.buffer.write(err);sys.exit(p.returncode)")
    job = OwnedProcessJob()
    try:
        process = job.start([sys.executable, '-c', outer, str(Path(subject.__file__).parent), inner],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = process.communicate(timeout=quiet_wall_timeout(20))
        assert process.returncode == 0, errors.decode(errors='replace')
        assert output.strip() == b'nested-owned'
    finally:
        job.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Job ownership')
def test_nested_job_verifies_sibling_listener():
    import subprocess
    from strata_process_job import OwnedProcessJob
    listener = ("import socket,sys;s=socket.socket();s.bind(('127.0.0.1',0));s.listen();"
                "print(s.getsockname()[1],flush=True);sys.stdin.buffer.read(1)")
    verifier = ("import sys;sys.path.insert(0,sys.argv[1]);import strata_queue as q;"
                "q.verify_owned_listener(int(sys.argv[2]));print('verified',flush=True)")
    inner = ("import sys,subprocess;"
             "p=subprocess.Popen([sys.executable,'-c',sys.argv[2]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);"
             "port=p.stdout.readline().decode().strip();"
             "r=subprocess.run([sys.executable,'-c',sys.argv[3],sys.argv[1],port],capture_output=True);"
             "p.kill();p.communicate();sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)")
    outer = ("import sys,subprocess;sys.path.insert(0,sys.argv[1]);from strata_process_job import OwnedProcessJob;"
             "j=OwnedProcessJob();p=j.start([sys.executable,'-c',sys.argv[2],sys.argv[1],sys.argv[3],sys.argv[4]],"
             "stdout=subprocess.PIPE,stderr=subprocess.PIPE);sys.path.insert(0,sys.argv[1]);from quiet_cpu import quiet_wall_timeout;out,err=p.communicate(timeout=quiet_wall_timeout(15));"
             "j.close();sys.stdout.buffer.write(out);sys.stderr.buffer.write(err);sys.exit(p.returncode)")
    job = OwnedProcessJob()
    try:
        process = job.start([sys.executable, '-c', outer, str(Path(subject.__file__).parent), inner, listener, verifier],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = process.communicate(timeout=quiet_wall_timeout(20))
        assert process.returncode == 0, errors.decode(errors='replace')
        assert output.strip() == b'verified'
    finally:
        job.close()


def test_worker_without_owned_job_cannot_start_model(monkeypatch):
    started = []
    def reject():
        raise RuntimeError('worker is not a member of the owned Job')
    monkeypatch.setattr(subject, 'verify_owned_worker', reject)
    monkeypatch.setattr(subject.runtime, 'start_ready', lambda *args: started.append(True))
    with pytest.raises(RuntimeError, match='not a member'):
        with subject.serving('strata'):
            pytest.fail('unowned worker cannot serve')
    assert not started


def test_structured_output_502_preserves_failure_without_another_generation(monkeypatch):
    import io
    import urllib.error
    import strata_queue
    bodies = []

    def request(route, body=None, **kwargs):
        if route == '/v1/models':
            return {'data': [{'id': 'strata-coder'}]}
        bodies.append(body)
        if len(bodies) == 1:
            raise urllib.error.HTTPError('u', 502, 'Bad Gateway', {}, io.BytesIO(b'{"error": {"code": "structured_output_failed"}}'))
        return {'choices': [{'message': {'content': '{"code": "x"}'}}], 'usage': {}}

    monkeypatch.setattr(strata_queue.runtime, 'request', request)
    monkeypatch.setattr(strata_queue, 'verify_owned_listener', lambda port: 1)
    with pytest.raises(RuntimeError, match='HTTP 502.*structured_output_failed') as failure:
        strata_queue.complete('u', [{'role': 'user', 'content': 'q'}], {'type': 'object'}, max_tokens=2400, extra={'temperature': 0})
    assert isinstance(failure.value.__cause__, urllib.error.HTTPError)
    assert len(bodies) == 1
    assert bodies[0]['reasoning_budget_tokens'] == 1200
