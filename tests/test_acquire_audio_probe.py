"""Public model acquisition: bounded writes, verified identity and strict resumption."""
import hashlib
import io
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acquire_audio_probe as acquire


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}
        if content_range:
            self.headers["Content-Range"] = content_range


def test_resume_verifies_complete_bytes(tmp_path, monkeypatch):
    data = b"the exact pinned weight bytes"
    (tmp_path / "weight.safetensors.part").write_bytes(data[:5])
    def response(request, **kwargs):
        assert request.headers["Range"] == "bytes=5-"
        return Response(data[5:], 206, f"bytes 5-{len(data)-1}/{len(data)}")
    monkeypatch.setattr(acquire.urllib.request, "urlopen", response)
    result = acquire.acquire("weight.safetensors", len(data), hashlib.sha256(data).hexdigest(),
                             rate=1024 * 1024, destination=tmp_path)
    assert (tmp_path / "weight.safetensors").read_bytes() == data
    assert result["resumed_bytes"] == 5
    assert not (tmp_path / "weight.safetensors.part").exists()


def test_server_ignoring_resume_cannot_append_full_body(tmp_path, monkeypatch):
    partial = tmp_path / "weight.safetensors.part"
    partial.write_bytes(b"keep")
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **kw: Response(b"full body", 200))
    with pytest.raises(ValueError, match="exact resume range"):
        acquire.acquire("weight.safetensors", 9, "wrong", rate=1024 * 1024, destination=tmp_path)
    assert partial.read_bytes() == b"keep"


def test_wrong_hash_cannot_publish_complete_file(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **kw: Response(b"incorrect"))
    with pytest.raises(ValueError, match="verification failed"):
        acquire.acquire("weight.safetensors", 9, hashlib.sha256(b"different").hexdigest(),
                        rate=1024 * 1024, destination=tmp_path)
    assert not (tmp_path / "weight.safetensors").exists()
    assert (tmp_path / "weight.safetensors.part").read_bytes() == b"incorrect"


def test_rate_wait_is_applied_before_next_read(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *a, **kw: Response(b"12345"))
    monkeypatch.setattr(acquire.time, "monotonic", lambda: 0.0)
    waits = []
    monkeypatch.setattr(acquire.time, "sleep", waits.append)
    acquire.acquire("weight.safetensors", 5, hashlib.sha256(b"12345").hexdigest(),
                    rate=5, destination=tmp_path)
    assert waits == [1.0]


def test_explicit_challenger_identity_controls_download_url(tmp_path, monkeypatch):
    def response(request, **kwargs):
        assert request.full_url == "https://huggingface.co/Owner/New/resolve/pinned/weight.gguf"
        return Response(b"new")
    monkeypatch.setattr(acquire.urllib.request, "urlopen", response)
    acquire.acquire("weight.gguf", 3, hashlib.sha256(b"new").hexdigest(),
                    rate=1024 * 1024, destination=tmp_path,
                    repository="Owner/New", revision="pinned")
