"""Local requests cannot inherit a proxy or follow an external redirect."""
import json
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import threading
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_llm as llm


@pytest.fixture
def server(monkeypatch):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append(self.path)
            if self.path == '/redirect':
                self.send_response(302)
                self.send_header('Location', 'https://external.invalid/never-follow')
                self.end_headers()
                return
            body = b'llamacpp:prompt_tokens_total 12\n' if self.path == '/metrics' else b'{"status":"ok"}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            data = json.dumps({'echo': body}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
    thread.start()
    monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1')
    monkeypatch.setenv('https_proxy', 'http://127.0.0.1:1')
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=120)
        assert not thread.is_alive()


def test_real_loopback_ignores_environment_proxy(server):
    base, calls = server
    assert llm._http_json(base+'/chat', {'private_fixture': 'local-only'}, timeout=120) == {
        'echo': {'private_fixture': 'local-only'}}
    with llm._local_open(base+'/health', timeout=120) as response:
        assert json.loads(response.read()) == {'status': 'ok'}
    assert calls == [{'private_fixture': 'local-only'}, '/health']


def test_health_and_metrics_keep_their_existing_deadlines(tmp_path, monkeypatch):
    calls = []
    class Response(io.BytesIO):
        status = 200
    def local_open(url, *, timeout):
        calls.append((url, timeout))
        return Response(b'llamacpp:prompt_tokens_total 12\n')
    monkeypatch.setattr(llm, '_local_open', local_open)
    class Process:
        def poll(self):
            return None
    base = 'http://127.0.0.1:12345'
    llm._wait_health(Process(), base, 10, tmp_path/'unused.log')
    assert llm._metric(base, 'llamacpp:prompt_tokens_total') == 12
    assert calls == [(base+'/health', 3), (base+'/metrics', 5)]


def test_redirect_never_contacts_another_host(server):
    base, calls = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        llm._local_open(base+'/redirect', timeout=120)
    assert exc.value.code == 302
    exc.value.close()
    assert calls == ['/redirect']


@pytest.mark.parametrize('url', ['https://127.0.0.1:1/chat', 'http://localhost:1/chat',
    'http://external.invalid/chat', 'http://127.0.0.1.external.invalid/chat',
    'http://user:password@127.0.0.1:1/chat', 'http://127.0.0.1:1/chat#fragment',
    'file:///C:/not-a-model-request'])
def test_nonlocal_or_ambiguous_url_rejected_before_any_opener(monkeypatch, url):
    monkeypatch.setattr(llm.urllib.request, 'build_opener', lambda *args: pytest.fail('Network path opened'))
    with pytest.raises(ValueError):
        llm._http_json(url, {'private_fixture': 'must stay local'})
