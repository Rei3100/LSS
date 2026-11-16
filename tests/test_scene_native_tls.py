from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest

from lss.bgm import scene_native


def test_download_context_adds_certifi_to_system_context(monkeypatch):
    calls = []

    class Context:
        check_hostname = True
        verify_mode = scene_native.ssl.CERT_REQUIRED

        def load_verify_locations(self, *, cafile):
            calls.append(cafile)

    context = Context()
    monkeypatch.setattr(scene_native.ssl, "create_default_context", lambda: context)
    monkeypatch.setitem(sys.modules, "certifi", SimpleNamespace(where=lambda: "bundle/cacert.pem"))

    assert scene_native._download_ssl_context() is context
    assert calls == ["bundle/cacert.pem"]
    assert context.check_hostname and context.verify_mode == scene_native.ssl.CERT_REQUIRED


def test_download_context_without_certifi_keeps_verified_system_context(monkeypatch):
    context = scene_native.ssl.create_default_context()
    monkeypatch.setattr(scene_native.ssl, "create_default_context", lambda: context)
    monkeypatch.setitem(sys.modules, "certifi", None)

    assert scene_native._download_ssl_context() is context
    assert context.check_hostname and context.verify_mode == scene_native.ssl.CERT_REQUIRED


@pytest.mark.parametrize("body, expected_size, expected_sha, succeeds", [
    (b"verified bytes", 14, hashlib.sha256(b"verified bytes").hexdigest(), True),
    (b"wrong", 5, "0" * 64, False),
])
def test_download_uses_verified_context_timeout_and_preserves_integrity_checks(
    monkeypatch, tmp_path, body, expected_size, expected_sha, succeeds
):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):
            nonlocal body
            result, body = body, b""
            return result

    calls = []
    context = object()
    monkeypatch.setattr(scene_native, "_download_ssl_context", lambda: context)
    monkeypatch.setattr(scene_native.urllib.request, "urlopen", lambda request, **kwargs: (calls.append((request, kwargs)) or Response()))
    target = tmp_path / "artifact.bin"

    if succeeds:
        scene_native._download("https://example.invalid/artifact", target,
                               sha256=expected_sha, size=expected_size)
        assert target.read_bytes() == b"verified bytes"
    else:
        with pytest.raises(scene_native.SceneModelsError):
            scene_native._download("https://example.invalid/artifact", target,
                                   sha256=expected_sha, size=expected_size)
        assert not target.exists()

    assert calls and calls[0][1] == {"timeout": 60, "context": context}
