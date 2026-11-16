"""URL入力を、モデル選択なしでローカル入力と同じ製品要求へつなぐ。"""
from functools import partial

import pytest

from lss import application
from lss.acquisition import service, store
from tests.test_acquisition import (
    FakeBackend, NO_WAIT, YtDlpBackend, _flac, roots, serve,
)
from tests.test_application import _default_catalog


INTENT = application.UserIntent.EXTRACT_ACCOMPANIMENT


class Queue:
    def __init__(self):
        self.calls = []

    def add(self, paths, **options):
        self.calls.append((paths, options))
        return [f"item-{i}" for i in range(len(paths))]


def test_real_http_acquisition_enters_the_resolved_product_route(roots, tmp_path, monkeypatch, serve):
    _default_catalog(tmp_path, monkeypatch)
    media = _flac(tmp_path / "source.flac")
    url, handler = serve(media.read_bytes(), "source.flac", "audio/flac")
    queue = Queue()

    result = application.submit(
        application.ProcessingRequest.from_urls([url], INTENT), queue,
        acquirer=partial(service.acquire, backend=YtDlpBackend(), pacing=NO_WAIT),
    )

    assert result.outcome is None and result.queue_item_ids == ("item-0",)
    [acquisition] = result.acquisitions
    assert acquisition.source_path.read_bytes() == handler.payload
    assert queue.calls == [([acquisition.source_path], {
        "model_id": "technical-model", "stem_target": "Instrumental", "include_existing": True,
    })]
    [record] = store.records_for_sha256(acquisition.source_sha256)
    assert record["location"]["requested_url"] == url
    assert record["tool"]["version"]


def test_playlist_failure_keeps_successful_and_local_inputs(roots, tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    media = _flac(tmp_path / "source.flac")
    backend = FakeBackend({"x1": media, "x2": media},
                          failures={"x2": [store.FailureKind.AUTH_EXPIRED]}, playlist=["x1", "x2"])
    queue = Queue()
    request = application.ProcessingRequest((media,), INTENT, ("https://example.test/playlist",))

    result = application.submit(request, queue,
                               acquirer=partial(service.acquire, backend=backend, pacing=NO_WAIT))

    assert result.outcome is None and len(result.queue_item_ids) == 2
    first, failed = result.acquisitions
    assert first.source_sha256 == store.file_sha256(media)
    assert queue.calls[0][0] == [media, first.source_path]
    assert failed.error_code == "auth_expired" and not failed.retryable
    assert failed.source_path is None and "SECRET" not in failed.message and "abc" not in failed.message


def test_default_acquirer_uses_the_existing_service(roots, tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    backend = FakeBackend({"x1": _flac(tmp_path / "source.flac")})
    acquire = service.acquire
    calls = []

    def configured(url):
        calls.append(url)
        return acquire(url, backend=backend, pacing=NO_WAIT)

    monkeypatch.setattr(service, "acquire", configured)
    result = application.submit(
        application.ProcessingRequest.from_urls(["https://example.test/x1"], INTENT), Queue(),
    )
    assert calls == ["https://example.test/x1"] and result.queue_item_ids


@pytest.mark.parametrize("intent", [application.UserIntent.EXTRACT_BGM_CUES, INTENT])
def test_resolution_failure_does_not_acquire_sources(tmp_path, monkeypatch, intent):
    if intent == INTENT:
        monkeypatch.setattr(application.catalog, "default_model", lambda: (_ for _ in ()).throw(
            PermissionError("台帳を読めません")))
    queue = Queue()
    result = application.submit(
        application.ProcessingRequest.from_urls(["https://example.test/x1"], intent), queue,
        acquirer=lambda url: pytest.fail("未対応・解決失敗では素材を取得しない"),
    )
    assert result.outcome.disposition in {"UNSUPPORTED", "FAILED"}
    assert not queue.calls and result.acquisitions == ()


def test_bgm_url_is_not_acquired_before_the_single_local_source_boundary(tmp_path):
    queue = Queue()
    result = application.submit(
        application.ProcessingRequest.from_urls(
            ["https://example.test/episode"], application.UserIntent.EXTRACT_BGM_CUES,
        ),
        queue,
        acquirer=lambda url: pytest.fail("BGMの未対応URLを取得しない"),
        bgm_runner=lambda request: pytest.fail("ローカル原本なしで候補処理を始めない"),
    )

    assert result.outcome.disposition == "UNSUPPORTED"
    assert result.outcome.error_code == "bgm_requires_single_local_source"
    assert result.acquisitions == () and not queue.calls


@pytest.mark.parametrize("failure", [
    store.FailureKind.AUTH_EXPIRED, store.FailureKind.NETWORK, store.FailureKind.CANCELLED,
])
def test_failed_acquisition_is_distinct_from_zero_and_unsupported(tmp_path, monkeypatch, failure):
    _default_catalog(tmp_path, monkeypatch)
    queue = Queue()
    url = "https://example.test/x1?v=123&token=SECRET#SECRET"
    def acquire(url):
        return [service.EntryOutcome(url, failure=store.AcquisitionFailure(failure, "token=SECRET"))]

    result = application.submit(application.ProcessingRequest.from_urls([url], INTENT), queue,
                                acquirer=acquire)

    assert result.outcome.disposition == "FAILED" and result.outcome.zero_result is None
    assert not result.queue_item_ids and not queue.calls
    [entry] = result.acquisitions
    assert entry.error_code == failure.value and entry.retryable == (failure in store.RETRYABLE)
    assert entry.url == "https://example.test/x1?v=123" and "SECRET" not in entry.message


def test_unexpected_failure_does_not_abort_other_urls(roots, tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    backend = FakeBackend({"x1": _flac(tmp_path / "source.flac")})
    def acquire(url):
        if url.endswith("broken"):
            raise PermissionError("認証状態を読めません token=SECRET")
        return service.acquire(url, backend=backend, pacing=NO_WAIT)

    result = application.submit(application.ProcessingRequest.from_urls(
        ["https://example.test/broken", "https://example.test/x1"], INTENT), Queue(), acquirer=acquire)

    assert result.outcome is None and len(result.queue_item_ids) == 1
    assert result.acquisitions[0].error_code == "acquisition_failed"
    assert "SECRET" not in result.acquisitions[0].message
    assert result.acquisitions[1].source_path.is_file()


def test_empty_playlist_is_failed_acquisition(tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    result = application.submit(application.ProcessingRequest.from_urls(
        ["https://example.test/empty"], INTENT), Queue(), acquirer=lambda url: [])
    assert result.outcome.disposition == "FAILED"
    assert result.acquisitions[0].error_code == "unavailable"


def test_offline_reuse_retains_content_identity(tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    source = tmp_path / "cached.flac"
    source.write_bytes(b"cached")
    acquired = store.AcquiredSource(source, store.file_sha256(source), "example:x1", True, {})
    result = application.submit(application.ProcessingRequest.from_urls(
        ["https://example.test/x1"], INTENT), Queue(),
        acquirer=lambda url: [service.EntryOutcome(url, acquired, offline_reuse=True)])
    [entry] = result.acquisitions
    assert entry.reused and entry.offline_reuse and entry.source_sha256 == acquired.sha256


def test_local_request_does_not_call_acquirer(tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    request = application.ProcessingRequest.from_paths([tmp_path / "source.wav"], INTENT)
    result = application.submit(request, Queue(), acquirer=lambda url: pytest.fail("ローカル入力"))
    assert result.queue_item_ids == ("item-0",) and result.acquisitions == ()


@pytest.mark.parametrize("urls", [("",), (" ",), (123,)])
def test_empty_or_non_string_url_is_rejected(urls):
    with pytest.raises(ValueError, match="URL"):
        application.ProcessingRequest.from_urls(urls, INTENT)


def test_development_requires_a_fixed_local_source(tmp_path):
    request = application.ProcessingRequest((tmp_path / "source.wav",),
                                           application.UserIntent.EXTRACT_BGM_CUES,
                                           ("https://example.test/x1",))
    outcome = application.run_bgm_development(request, tmp_path / "unused-plan.json")
    assert outcome.error_code == "development_bgm_requires_single_source"


def test_url_request_repr_does_not_include_credentials():
    request = application.ProcessingRequest.from_urls(["https://example.test/x1?token=SECRET"], INTENT)
    assert "SECRET" not in repr(request)


@pytest.mark.parametrize("url, safe", [
    ("https://account:SECRET@example.test/x1?v=123&sig=SECRET", "https://example.test/x1?v=123"),
    ("http://account:SECRET@[::1]:8080/x1?token=SECRET", "http://[::1]:8080/x1"),
    ("https://account:SECRET@example.test:invalid/x1", "[不正なURL]"),
])
def test_acquisition_sanitizers_remove_userinfo_without_breaking_ipv6(url, safe):
    assert store.sanitize_url(url) == safe
    assert "SECRET" not in store.sanitize_text(f"取得失敗: {url} token=SECRET")


def test_repeated_url_before_processing_returns_the_same_queued_item(roots, tmp_path, monkeypatch):
    from lss.core.jobs import SeparationQueue
    _default_catalog(tmp_path, monkeypatch)
    backend = FakeBackend({"x1": _flac(tmp_path / "source.flac")})
    queue = SeparationQueue(lambda job: pytest.fail("登録だけ"), state_path=tmp_path / "queue.json")
    request = application.ProcessingRequest.from_urls(["https://example.test/x1"], INTENT)
    acquire = partial(service.acquire, backend=backend, pacing=NO_WAIT)
    first = application.submit(request, queue, acquirer=acquire)
    stamp = (tmp_path / "queue.json").stat().st_mtime_ns
    again = application.submit(request, queue, acquirer=acquire)

    assert again.queue_item_ids == first.queue_item_ids and again.outcome is None
    assert len(queue.snapshot()) == 1 and again.acquisitions[0].reused
    assert (tmp_path / "queue.json").stat().st_mtime_ns == stamp and backend.downloads == ["x1"]


def test_queue_registration_failure_retains_the_acquired_source(roots, tmp_path, monkeypatch):
    _default_catalog(tmp_path, monkeypatch)
    backend = FakeBackend({"x1": _flac(tmp_path / "source.flac")})
    queue = Queue()
    def fail(*args, **kwargs):
        raise OSError("台帳を保存できません token=SECRET")
    queue.add = fail
    result = application.submit(application.ProcessingRequest.from_urls(
        ["https://example.test/x1"], INTENT), queue,
        acquirer=partial(service.acquire, backend=backend, pacing=NO_WAIT))

    assert result.outcome.error_code == "queue_registration_failed"
    assert "SECRET" not in result.outcome.message
    assert result.acquisitions[0].source_path.is_file()
