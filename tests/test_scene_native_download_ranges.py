"""Bounded download integrity tests use tiny bytes and no public network."""
import hashlib
import io
import pytest
from lss.bgm import scene_native as native


class Response(io.BytesIO):
    def __init__(self, body, status, content_range=None):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Range': content_range} if content_range else {}


def setup(monkeypatch, tmp_path):
    body = b'abcdefghij'
    target = tmp_path / 'model.gguf'
    part = tmp_path / 'model.gguf.part'
    monkeypatch.setattr(native, 'DOWNLOAD_RANGE_BYTES', 4)
    monkeypatch.setattr(native, '_download_ssl_context', lambda: object())
    return body, target, part, hashlib.sha256(body).hexdigest()


def test_sequential_bounded_ranges_preserve_full_digest(monkeypatch, tmp_path):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    seen = []
    def fetch(request, **kwargs):
        value = request.get_header('Range')
        seen.append(value)
        lo, hi = map(int, value[6:].split('-'))
        return Response(body[lo:hi+1], 206, f'bytes {lo}-{hi}/{len(body)}')
    monkeypatch.setattr(native.urllib.request, 'urlopen', fetch)
    native._download('https://example.invalid/model', target, sha256=digest, size=len(body))
    assert seen == ['bytes=0-3', 'bytes=4-7', 'bytes=8-9']
    assert target.read_bytes() == body and not part.exists()


def test_existing_prefix_resumes_from_exact_eof(monkeypatch, tmp_path):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    part.write_bytes(body[:6])
    def fetch(request, **kwargs):
        assert request.get_header('Range') == 'bytes=6-9'
        return Response(body[6:], 206, 'bytes 6-9/10')
    monkeypatch.setattr(native.urllib.request, 'urlopen', fetch)
    native._download('https://example.invalid/model', target, sha256=digest, size=10)
    assert target.read_bytes() == body


@pytest.mark.parametrize('header', [None, 'bytes 0-3/10', 'bytes 6-8/10', 'bytes 6-9/11'])
def test_wrong_range_cannot_change_existing_prefix(monkeypatch, tmp_path, header):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    part.write_bytes(body[:6])
    monkeypatch.setattr(native.urllib.request, 'urlopen', lambda *a, **k: Response(body[6:], 206, header))
    with pytest.raises(native.SceneModelsError, match='invalid_content_range'):
        native._download('https://example.invalid/model', target, sha256=digest, size=10)
    assert part.read_bytes() == body[:6] and not target.exists()


def test_ignored_range_restarts_full_response_safely(monkeypatch, tmp_path):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    part.write_bytes(b'old prefix')
    monkeypatch.setattr(native.urllib.request, 'urlopen', lambda *a, **k: Response(body, 200))
    # A full-sized incorrect partial is digest-checked without a new request.
    part.write_bytes(b'old')
    native._download('https://example.invalid/model', target, sha256=digest, size=10)
    assert target.read_bytes() == body


def test_complete_partial_promotes_without_network(monkeypatch, tmp_path):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    part.write_bytes(body)
    monkeypatch.setattr(native.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network'))
    native._download('https://example.invalid/model', target, sha256=digest, size=10)
    assert target.read_bytes() == body and not part.exists()


def test_truncated_chunk_keeps_only_valid_received_prefix(monkeypatch, tmp_path):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(native.urllib.request, 'urlopen', lambda *a, **k: Response(body[:2], 206, 'bytes 0-3/10'))
    with pytest.raises(native.SceneModelsError, match='truncated_response'):
        native._download('https://example.invalid/model', target, sha256=digest, size=10)
    assert part.read_bytes() == body[:2] and not target.exists()


def test_oversized_chunk_never_publishes(monkeypatch, tmp_path):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(native.urllib.request, 'urlopen', lambda *a, **k: Response(body[:5], 206, 'bytes 0-3/10'))
    with pytest.raises(native.SceneModelsError, match='oversized_response'):
        native._download('https://example.invalid/model', target, sha256=digest, size=10)
    assert part.stat().st_size == 4 and not target.exists()


def test_wrong_complete_digest_cannot_publish(monkeypatch, tmp_path):
    body, target, part, digest = setup(monkeypatch, tmp_path)
    part.write_bytes(b'xxxxxxxxxx')
    with pytest.raises(native.SceneModelsError):
        native._download('https://example.invalid/model', target, sha256=digest, size=10)
    assert not target.exists() and not part.exists()


@pytest.mark.parametrize("state", ["missing", "partial", "tampered"])
def test_cached_only_refuses_before_network_or_file_mutation(monkeypatch, tmp_path, state):
    target = tmp_path / "new-directory" / "model.gguf"
    part = target.with_name(target.name + ".part")
    if state != "missing":
        target.parent.mkdir()
        (part if state == "partial" else target).write_bytes(b"untrusted")
    before = {p.name: p.read_bytes() for p in target.parent.glob("*")} if target.parent.exists() else None
    monkeypatch.setenv("LSS_SCENE_NO_ACQUIRE", "1")
    monkeypatch.setattr(native.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network attempted"))
    monkeypatch.setattr(native, "_download_ssl_context", lambda: pytest.fail("TLS context created"))
    with pytest.raises(native.SceneModelsError, match="scene_native_acquisition_disabled"):
        native._download("https://example.invalid/model", target, sha256=hashlib.sha256(b"trusted").hexdigest(), size=7)
    after = {p.name: p.read_bytes() for p in target.parent.glob("*")} if target.parent.exists() else None
    assert before == after


def test_cached_only_reuses_only_fully_pinned_existing_bytes(monkeypatch, tmp_path):
    target = tmp_path / "model.gguf"
    target.write_bytes(b"trusted")
    stamp = target.stat().st_mtime_ns
    monkeypatch.setenv("LSS_SCENE_NO_ACQUIRE", "1")
    monkeypatch.setattr(native.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network attempted"))
    native._download("https://example.invalid/model", target, sha256=hashlib.sha256(b"trusted").hexdigest(), size=7)
    assert target.read_bytes() == b"trusted" and target.stat().st_mtime_ns == stamp


@pytest.mark.parametrize("mode", ["", "true", "invalid"])
def test_malformed_cached_only_policy_never_enables_acquisition(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("LSS_SCENE_NO_ACQUIRE", mode)
    monkeypatch.setattr(native.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network attempted"))
    target = tmp_path / "no-directory" / "model.gguf"
    with pytest.raises(native.SceneModelsError, match="scene_native_acquisition_disabled"):
        native._download("https://example.invalid/model", target, sha256="0" * 64, size=10)
    assert not target.parent.exists()
