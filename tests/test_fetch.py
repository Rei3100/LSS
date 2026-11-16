"""モデル取得のサイズ・SHA-256検証と、取得失敗時の既存ファイル保全。"""
import hashlib
import threading

import pytest

from lss.harness import fetch


class Response:
    def __init__(self, data, *, url="https://example.com/weight.ckpt", headers=None, history=None):
        self.data = data
        self.url = url
        self.headers = headers or {}
        self.history = history or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


@pytest.mark.parametrize("downloaded", [b"", b"short"])
def test_incomplete_download_does_not_replace_existing_cache(tmp_path, monkeypatch, downloaded):
    dest = tmp_path / "weight.ckpt"
    dest.write_bytes(b"old")
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response(downloaded))
    with pytest.raises(fetch.DownloadIntegrityError, match="サイズが不正"):
        fetch.download("https://example.com/weight.ckpt", dest, expected_size=10)
    assert dest.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [dest]


def test_wrong_size_cache_is_replaced_only_after_complete_download(tmp_path, monkeypatch):
    dest = tmp_path / "weight.ckpt"
    dest.write_bytes(b"old")
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response(b"new weight"))
    fetch.download("https://example.com/weight.ckpt", dest, expected_size=10)
    assert dest.read_bytes() == b"new weight"


def test_frozen_model_cache_uses_visible_models_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("LSS_MODEL_DIR", raising=False)
    monkeypatch.setattr(fetch, "is_frozen", lambda: True)
    monkeypatch.setattr(fetch, "app_root", lambda: tmp_path)

    assert fetch.model_cache_dir() == tmp_path / "models"


def test_same_size_corrupt_cache_is_replaced_after_sha256_validation(tmp_path, monkeypatch):
    dest = tmp_path / "weight.ckpt"
    dest.write_bytes(b"bad weight")
    valid = b"new weight"
    expected_sha256 = hashlib.sha256(valid).hexdigest()
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response(valid))

    fetch.download(
        "https://example.com/weight.ckpt", dest,
        expected_size=len(valid), expected_sha256=expected_sha256,
    )

    assert dest.read_bytes() == valid


def test_bad_download_sha256_preserves_existing_cache(tmp_path, monkeypatch):
    dest = tmp_path / "weight.ckpt"
    dest.write_bytes(b"old weight")
    downloaded = b"bad weight"
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response(downloaded))

    with pytest.raises(fetch.DownloadIntegrityError, match="SHA-256"):
        fetch.download(
            "https://example.com/weight.ckpt", dest,
            expected_size=len(downloaded), expected_sha256=hashlib.sha256(b"valid").hexdigest(),
        )

    assert dest.read_bytes() == b"old weight"


def test_download_reports_byte_progress(tmp_path, monkeypatch):
    data = b"downloaded model"
    dest = tmp_path / "weight.ckpt"
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response(data))
    events = []

    fetch.download(
        "https://example.com/weight.ckpt", dest,
        expected_size=len(data), progress=lambda completed, total: events.append((completed, total)),
    )

    assert events == [(0, len(data)), (len(data), len(data))]


@pytest.mark.parametrize("url", [
    "http://example.com/weight.ckpt",
    "https://user:secret@example.com/weight.ckpt",
    "weight.ckpt",
])
def test_download_rejects_non_https_or_embedded_credentials(tmp_path, monkeypatch, url):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("通信してはいけない"))

    with pytest.raises(fetch.InvalidModelSourceError, match="HTTPS"):
        fetch.download(url, tmp_path / "weight.ckpt")


def test_download_rejects_redirect_from_https_to_http(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch.requests, "get",
        lambda *a, **k: Response(b"weight", url="http://mirror.example.com/weight.ckpt"),
    )

    with pytest.raises(fetch.InvalidModelSourceError, match="HTTPS"):
        fetch.download("https://example.com/weight.ckpt", tmp_path / "weight.ckpt")
    assert not (tmp_path / "weight.ckpt").exists()


def test_download_rejects_insecure_intermediate_redirect(tmp_path, monkeypatch):
    insecure_hop = Response(b"", url="http://mirror.example.com/weight.ckpt")
    monkeypatch.setattr(
        fetch.requests, "get",
        lambda *a, **k: Response(
            b"weight", url="https://cdn.example.com/weight.ckpt", history=[insecure_hop],
        ),
    )

    with pytest.raises(fetch.InvalidModelSourceError, match="HTTPS"):
        fetch.download("https://example.com/weight.ckpt", tmp_path / "weight.ckpt")

    assert not (tmp_path / "weight.ckpt").exists()


def test_download_rejects_malformed_sha256_before_network_access(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("通信してはいけない"))

    with pytest.raises(fetch.InvalidModelSourceError, match="SHA-256"):
        fetch.download(
            "https://example.com/weight.ckpt", tmp_path / "weight.ckpt",
            expected_sha256="not-a-sha256",
        )


def test_download_stops_when_stream_exceeds_limit_and_preserves_cache(tmp_path, monkeypatch):
    dest = tmp_path / "config.yaml"
    dest.write_bytes(b"old config")
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Response(b"0123456789"))

    with pytest.raises(fetch.DownloadTooLargeError, match="上限"):
        fetch.download("https://example.com/config.yaml", dest, max_bytes=5)

    assert dest.read_bytes() == b"old config"
    assert not dest.with_suffix(".yaml.part").exists()


def test_network_failure_has_stable_error_and_preserves_existing_cache(tmp_path, monkeypatch):
    dest = tmp_path / "weight.ckpt"
    dest.write_bytes(b"old weight")
    monkeypatch.setattr(
        fetch.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(fetch.requests.Timeout("timed out")),
    )

    with pytest.raises(fetch.ModelDownloadError, match="モデル資産を取得できません"):
        fetch.download("https://example.com/weight.ckpt", dest, expected_size=20)

    assert dest.read_bytes() == b"old weight"
    assert not dest.with_suffix(".ckpt.part").exists()


def test_concurrent_downloads_use_separate_partial_files(tmp_path, monkeypatch):
    data = b"concurrent model data"
    barrier = threading.Barrier(2)

    class ConcurrentResponse(Response):
        def iter_content(self, chunk_size):
            barrier.wait(timeout=5)
            yield data

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: ConcurrentResponse(data))
    destination = tmp_path / "weight.ckpt"
    errors = []

    def run():
        try:
            fetch.download(
                "https://example.com/weight.ckpt", destination,
                expected_size=len(data), expected_sha256=hashlib.sha256(data).hexdigest(),
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert destination.read_bytes() == data
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("model_id", ["../outside", "a/b", "CON", "name."])
def test_model_paths_rejects_unsafe_model_id(tmp_path, monkeypatch, model_id):
    monkeypatch.setenv("LSS_MODEL_DIR", str(tmp_path / "models"))

    with pytest.raises(fetch.InvalidModelSourceError):
        fetch.model_paths(
            model_id,
            "https://example.com/weight.ckpt",
            "https://example.com/config.yaml",
        )


@pytest.mark.parametrize("url", [
    "https://example.com/",
    "https://example.com/%2e%2e",
    "https://example.com/a%2fb.ckpt",
])
def test_model_paths_rejects_unsafe_asset_name(tmp_path, monkeypatch, url):
    monkeypatch.setenv("LSS_MODEL_DIR", str(tmp_path / "models"))

    with pytest.raises(fetch.InvalidModelSourceError):
        fetch.model_paths("model", url, "https://example.com/config.yaml")
