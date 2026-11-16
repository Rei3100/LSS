"""Native model requests stay on loopback even with ambient proxy settings."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lss.bgm import scene_native


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
                self.send_header('Location', 'http://127.0.0.1:1/must-not-follow')
                self.end_headers()
                return
            data = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
    for name in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
        monkeypatch.setenv(name, 'http://127.0.0.1:1')
    for name in ('no_proxy', 'NO_PROXY'):
        monkeypatch.setenv(name, '')
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', calls
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=120)
        assert not thread.is_alive()


def test_real_loopback_ignores_proxy(server):
    base, calls = server
    payload = {'private_fixture': 'local-only'}
    request = urllib.request.Request(base + '/chat', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with scene_native._local_open(request, timeout=120) as response:
        assert json.loads(response.read()) == {'echo': payload}
    with scene_native._local_open(base + '/health', timeout=120) as response:
        assert json.loads(response.read()) == {'status': 'ok'}
    assert calls == [payload, '/health']


def test_redirect_is_not_followed(server):
    base, calls = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        scene_native._local_open(base + '/redirect', timeout=120)
    assert exc.value.code == 302
    exc.value.close()
    assert calls == ['/redirect']


@pytest.mark.parametrize('url', [
    'https://127.0.0.1:1/chat', 'http://localhost:1/chat',
    'http://external.invalid/chat', 'http://127.0.0.1.external.invalid/chat',
    'http://user:password@127.0.0.1:1/chat', 'http://127.0.0.1:1/chat#fragment',
    'file:///C:/not-a-model-request',
])
def test_invalid_url_rejected_before_opener(monkeypatch, url):
    def fail(*args):
        pytest.fail('Network path opened')
    monkeypatch.setattr(scene_native.urllib.request, 'build_opener', fail)
    with pytest.raises(ValueError):
        scene_native._local_open(url, timeout=120)


def test_completion_keeps_request_body_and_deadline(monkeypatch):
    import io

    calls = []
    model = scene_native.NativeSceneModels.__new__(scene_native.NativeSceneModels)
    model._port = 12345
    monkeypatch.setattr(model, '_load_llm', lambda: None)

    def local_open(request, *, timeout):
        calls.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(b'{"choices":[{"message":{"content":"answer"}}]}')

    monkeypatch.setattr(scene_native, '_local_open', local_open)
    assert model._ask('private prompt', 42) == 'answer'
    url, body, timeout = calls[0]
    assert url == 'http://127.0.0.1:12345/v1/chat/completions'
    assert body['messages'] == [{'role': 'user', 'content': 'private prompt'}]
    assert body['max_tokens'] == 42
    assert body['t_max_predict_ms'] == scene_native.LLM['max_seconds'] * 1000
    assert timeout == scene_native.LLM['max_seconds'] + 60
