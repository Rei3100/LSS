"""URL取得の保存・来歴・再利用・中断再開・失敗分類。外部サービスへは接続しない。

実際のyt-dlpはローカルHTTPサーバーの直リンクで動かし、取得手段の実物を通す。
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.acquisition import service, store
from lss.acquisition.store import AcquisitionFailure, AuthMethod, FailureKind
import sys

from lss.acquisition.ytdlp import Description, YtDlpBackend as _Backend, classify


@pytest.fixture
def roots(tmp_path, monkeypatch):
    state = tmp_path / "state"
    library = tmp_path / "library"
    monkeypatch.setenv("LSS_STATE_DIR", str(state))
    monkeypatch.setenv(store.LIBRARY_ENV, str(library))
    return state, library


def _flac(path: Path, *, seconds: float = 1.0, channels: int = 2, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal((int(44100 * seconds), channels)) * 0.1).astype(np.float32)
    sf.write(path, samples, 44100, subtype="PCM_16")
    return path


# 仮想環境のyt-dlpモジュールを、配備版と同じく別プロセスの実行物として使う。
MODULE = (sys.executable, "-m", "yt_dlp")


def YtDlpBackend(**options):  # noqa: N802
    return _Backend(MODULE, **options)


NO_WAIT = service.Pacing(between_items=(0.0, 0.0), backoff_start=0.0, attempts=3)


class FakeBackend:
    """取得手段の差し替え。descriptionとダウンロード結果だけを決める。"""

    def __init__(self, media: dict[str, Path], failures: dict[str, list[FailureKind]] | None = None,
                 playlist: list[str] | None = None) -> None:
        self.media = media
        self.failures = failures or {}
        self.playlist = playlist
        self.downloads: list[str] = []

    def resolve(self, url, auth):
        ids = self.playlist if self.playlist is not None else [url.split("?", 1)[0].rsplit("/", 1)[-1]]
        return [{"id": source_id, "webpage_url": f"https://example.test/{source_id}"} for source_id in ids]

    def describe(self, entry, auth):
        source_id = entry["id"]
        size = self.media[source_id].stat().st_size if source_id in self.media else None
        record = {
            "logical_source": f"example:{source_id}",
            "authority": {"extractor_key": "Example", "id": source_id},
            "location": {"webpage_url": entry["webpage_url"]},
            "auth": auth.provenance(),
            "tool": {"name": "fake", "version": "1"},
            "format": {"selected": [{"format_id": "a", "filesize": size}]},
            "metadata": {"title": f"曲 {source_id}"},
            "reacquire": {"url": entry["webpage_url"], "auth_method": auth.method},
        }
        return Description(record["logical_source"], "Example", f"曲 {source_id} [{source_id}]", record, {})

    def download(self, description, work, auth, on_progress=None, *, should_stop=None):
        source_id = description.logical_source.split(":", 1)[1]
        pending = self.failures.get(source_id) or []
        if pending:
            raise AcquisitionFailure(pending.pop(0), "https://cdn.example.test/a?sig=SECRET token=abc")
        self.downloads.append(source_id)
        work.mkdir(parents=True, exist_ok=True)
        staged = work / f"source{self.media[source_id].suffix}"
        staged.write_bytes(self.media[source_id].read_bytes())
        return staged


def test_acquisition_publishes_immutable_source_with_provenance(roots, tmp_path):
    state, library = roots
    media = _flac(tmp_path / "a.flac")
    backend = FakeBackend({"x1": media})

    [outcome] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)

    assert outcome.ok and not outcome.source.reused
    path = outcome.source.path
    assert path == (library / "Example" / "曲 x1 [x1].flac").resolve()
    assert path.read_bytes() == media.read_bytes()
    record = store.records_for_sha256(outcome.source.sha256)[0]
    assert record["logical_source"] == "example:x1"
    assert record["content"]["sha256"] == store.file_sha256(path)
    assert {k: record["media"][k] for k in ("audio_channels", "sample_rate")} == {"audio_channels": 2, "sample_rate": 44100}
    timeline = record["media"]["timeline"]
    assert abs(timeline["difference_seconds"]) < 0.05 and timeline["decoded_seconds"] > 0
    assert record["auth"] == {"method": "none"}
    assert record["location"]["requested_url"] == "https://example.test/x1"
    assert record["acquired_at"]
    # 作業領域は公開後に残さない。
    assert not store.work_dir_for("example:x1").exists()


def test_same_logical_source_is_reused_and_force_reacquires_without_overwriting(roots, tmp_path):
    media = _flac(tmp_path / "a.flac")
    backend = FakeBackend({"x1": media})
    [first] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    [again] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    assert again.source.reused and again.source.path == first.source.path
    assert backend.downloads == ["x1"]

    # 同じ内容の意図的な取り直しは同じファイルを指し、版だけが増える。
    [forced] = service.acquire("https://example.test/x1", backend=backend, force=True, pacing=NO_WAIT)
    assert not forced.source.reused and forced.source.path == first.source.path
    assert [r["intent"] for r in store.load_revisions("example:x1")] == ["acquisition", "forced_reacquisition"]

    # 取得元の内容が変わったら、既存ファイルを上書きせず別名で置く。
    backend.media["x1"] = _flac(tmp_path / "b.flac", seed=1)
    [changed] = service.acquire("https://example.test/x1", backend=backend, force=True, pacing=NO_WAIT)
    assert changed.source.path.name == "曲 x1 [x1] (2).flac"
    assert first.source.path.read_bytes() == media.read_bytes()
    assert store.find_valid("example:x1")[0] == changed.source.path


def test_deleted_or_corrupted_source_is_reacquired(roots, tmp_path):
    backend = FakeBackend({"x1": _flac(tmp_path / "a.flac")})
    [first] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    data = bytearray(first.source.path.read_bytes())
    data[-1] ^= 0xFF
    first.source.path.write_bytes(bytes(data))

    [second] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    assert not second.source.reused
    assert store.file_sha256(second.source.path) == second.source.sha256
    assert backend.downloads == ["x1", "x1"]


def test_retryable_failures_back_off_and_permanent_failures_do_not_stop_the_playlist(roots, tmp_path):
    sleeps: list[float] = []
    backend = FakeBackend(
        {"a": _flac(tmp_path / "a.flac"), "b": _flac(tmp_path / "b.flac", seed=2),
         "c": _flac(tmp_path / "c.flac", seed=3)},
        failures={"a": [FailureKind.RATE_LIMITED], "b": [FailureKind.AUTH_REQUIRED]},
        playlist=["a", "b", "c"],
    )
    pacing = service.Pacing(between_items=(1.0, 1.0), backoff_start=10.0, backoff_limit=40.0, attempts=3)
    outcomes = service.acquire("https://example.test/list", backend=backend, pacing=pacing,
                               sleep=sleeps.append)

    assert [o.ok for o in outcomes] == [True, False, True]
    assert outcomes[1].failure.kind is FailureKind.AUTH_REQUIRED and not outcomes[1].failure.retryable
    # 秘匿値は診断文に残らない。
    assert "SECRET" not in str(outcomes[1].failure) and "abc" not in str(outcomes[1].failure)
    # 混雑では待って同じ項目をやり直し、項目の間にも間隔を空ける。
    # 混雑を示された後は、以後の項目の間隔も広げたままにする。
    assert sleeps == [10.0, 2.0, 2.0]
    assert backend.downloads == ["a", "c"]


def test_offline_resolve_reuses_a_verified_acquisition_for_the_same_url(roots, tmp_path):
    backend = FakeBackend({"x1": _flac(tmp_path / "a.flac")})
    [first] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)

    class Offline(FakeBackend):
        def resolve(self, url, auth):
            raise AcquisitionFailure(FailureKind.NETWORK, "timed out")

    offline = Offline({})
    [reused] = service.acquire("https://example.test/x1", backend=offline, pacing=NO_WAIT, sleep=lambda _s: None)
    assert reused.ok and reused.offline_reuse and reused.source.path == first.source.path
    [unknown] = service.acquire("https://example.test/other", backend=offline, pacing=NO_WAIT, sleep=lambda _s: None)
    assert unknown.failure.kind is FailureKind.NETWORK


def test_media_without_audio_is_not_published(roots, tmp_path):
    broken = tmp_path / "broken.m4a"
    broken.write_bytes(b"not media at all" * 100)
    backend = FakeBackend({"x1": broken})
    [outcome] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    assert outcome.failure.kind is FailureKind.INVALID_MEDIA
    assert not list(roots[1].rglob("*.m4a"))
    assert store.load_revisions("example:x1") == []


def test_concurrent_acquisition_of_the_same_source_is_refused(roots):
    work = store.work_dir_for("example:x1")
    with store.WorkLock(work):
        with pytest.raises(AcquisitionFailure) as caught:
            with store.WorkLock(work):
                pass
    assert caught.value.kind is FailureKind.BUSY


def test_browser_cookie_auth_records_only_the_method_class():
    auth = AuthMethod.browser_cookies("Brave", "Profile 3")
    assert auth.provenance() == {"method": "browser_cookies", "browser": "brave", "profile_specified": True}


def test_sanitizers_keep_location_but_drop_secrets():
    assert store.sanitize_url("https://www.youtube.com/watch?v=abc&token=zzz#t=1") == \
        "https://www.youtube.com/watch?v=abc"
    assert store.sanitize_url("https://user:pw@example.test/a") == "https://example.test/a"
    text = store.sanitize_text(('GET https://rr1.googlevideo.com/videoplayback?expire=1&sig=AB failed; ' + 'Cookie' + chr(58) + ' SID=xyz'))
    assert "sig=" not in text and "xyz" not in text and "https://rr1.googlevideo.com/videoplayback?…" in text


def test_a_malformed_cookie_file_line_never_reaches_receipts_or_logs():
    """yt-dlpは読めないCookieファイルの行を、値ごと `WARNING: skipping cookie file entry ...` へ書く。
    警告は取得の来歴（diagnostics.warnings）とログへ残るので、その行の中身を落とす。"""
    from lss.acquisition.ytdlp import _Run

    # An exported line without its expiry column (6 fields): yt-dlp reports "invalid length 6".
    line = chr(9).join(['.youtube.com', 'TRUE', '/', 'TRUE', '__Secure-3PSID', 'g.a000SECRETVALUE\n'])
    stderr = f"WARNING: skipping cookie file entry due to invalid length 6: {line!r}\n"
    [warning] = _Run(0, [], stderr).warnings()
    assert "SECRETVALUE" not in warning and "__Secure-3PSID" not in warning
    assert warning.startswith("WARNING: skipping cookie file entry")
    # A Cookie header keeps no pair, not only its first one.
    header = store.sanitize_text(('' + 'Cookie' + chr(58) + ' SID=one; HSID=two; SSID=three'))
    assert "one" not in header and "two" not in header and "three" not in header


@pytest.mark.parametrize(("message", "kind"), [
    ("ERROR: [youtube] x: Sign in to confirm you're not a bot. Use --cookies-from-browser", FailureKind.AUTH_REQUIRED),
    ("ERROR: [youtube] x: Video unavailable. This video has been removed by the uploader", FailureKind.UNAVAILABLE),
    ("ERROR: [youtube] x: The uploader has not made this video available in your country", FailureKind.REGION_RESTRICTED),
    ("ERROR: Unable to download webpage: <urlopen error timed out>", FailureKind.NETWORK),
    ("ERROR: HTTP Error 429: Too Many Requests", FailureKind.RATE_LIMITED),
    ("ERROR: Unsupported URL: https://example.test/", FailureKind.UNSUPPORTED_URL),
    ("ERROR: could not find brave cookies database in " + "Q" + chr(58) + "/synthetic-profile", FailureKind.AUTH_UNAVAILABLE),
    ("ERROR: Failed to decrypt with DPAPI. See  https://github.com/yt-dlp/yt-dlp/issues/10927", FailureKind.AUTH_UNAVAILABLE),
])
def test_ytdlp_errors_map_to_contract_failures(message, kind):
    from yt_dlp.utils import DownloadError

    assert classify(DownloadError(message)).kind is kind


# --- 実際のyt-dlpをローカルHTTPで通す ----------------------------------------

class _Media(BaseHTTPRequestHandler):
    payload = b""
    content_type = "application/octet-stream"
    cut_first_response_at: int | None = None  # 先頭からの応答を途中で切る
    requests: list[str | None] = []

    def log_message(self, *_args):
        pass

    def do_HEAD(self):
        self._respond(head=True)

    def do_GET(self):
        self._respond(head=False)

    def _respond(self, head: bool):
        data = type(self).payload
        start = 0
        header = self.headers.get("Range")
        if not head:
            type(self).requests.append(header)
        if header and header.startswith("bytes="):
            start = int(header[6:].split("-", 1)[0] or 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", type(self).content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        if head:
            return
        cut = type(self).cut_first_response_at
        if cut is not None and start == 0:
            self.wfile.write(data[:cut])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(data[start:])


@pytest.fixture
def serve():
    servers = []

    def start(payload: bytes, name: str, content_type: str):
        handler = type("Handler", (_Media,), {"payload": payload, "requests": [],
                                              "cut_first_response_at": None, "content_type": content_type})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}/{name}", handler

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def media_server(tmp_path, serve):
    return serve(_flac(tmp_path / "served.flac", seconds=3.0).read_bytes(), "song.flac", "audio/flac")


def test_real_ytdlp_downloads_the_original_bytes_without_conversion(roots, media_server):
    url, handler = media_server
    events: list[dict] = []
    [outcome] = service.acquire(url, backend=YtDlpBackend(), pacing=NO_WAIT, on_progress=events.append)

    assert outcome.ok, outcome.failure
    assert outcome.source.path.read_bytes() == handler.payload
    record = outcome.source.record
    assert record["logical_source"] == "generic:song"
    assert record["tool"]["name"] == "yt-dlp" and record["tool"]["version"]
    assert record["format"]["transcoded"] is False and record["format"]["selected"][0]["ext"] == "flac"
    assert record["reacquire"]["url"] == url
    assert any(event.get("stage") == "download" for event in events)
    assert "http_headers" not in json.dumps(record)


def test_real_ytdlp_resumes_an_interrupted_download_from_the_partial_file(roots, media_server):
    url, handler = media_server
    handler.cut_first_response_at = len(handler.payload) // 3
    backend = YtDlpBackend(extra_args=["--retries", "0"])

    [interrupted] = service.acquire(url, backend=backend,
                                    pacing=service.Pacing(backoff_start=0.0, attempts=1))
    assert interrupted.failure is not None and interrupted.failure.retryable
    partial = list(store.work_dir_for("generic:song").glob("*.part"))
    assert partial and 0 < partial[0].stat().st_size < len(handler.payload)
    assert not list(roots[1].rglob("*.flac"))

    handler.cut_first_response_at = None
    [resumed] = service.acquire(url, backend=backend, pacing=NO_WAIT)
    assert resumed.ok, resumed.failure
    assert resumed.source.path.read_bytes() == handler.payload
    assert any(header and header != "bytes=0-" for header in handler.requests)


def test_acquired_source_enters_the_same_separation_queue(roots, tmp_path):
    from lss.core.jobs import SeparationQueue

    backend = FakeBackend({"x1": _flac(tmp_path / "a.flac")})
    queue = SeparationQueue(lambda job: {"status": "ok"}, state_path=tmp_path / "queue.json")
    intake = service.enqueue(queue, "https://example.test/x1", backend=backend, pacing=NO_WAIT)
    assert len(intake.queue_ids) == 1
    [item] = queue.snapshot()
    assert Path(item.path) == intake.outcomes[0].source.path


def test_reacquisition_uses_the_recorded_location_after_the_source_is_deleted(roots, tmp_path):
    backend = FakeBackend({"x1": _flac(tmp_path / "a.flac")})
    [first] = service.acquire("https://example.test/x1?token=zzz", backend=backend, pacing=NO_WAIT)
    first.source.path.unlink()

    again = service.reacquire(first.source.sha256, backend=backend, pacing=NO_WAIT)
    assert again.outcome.ok and again.identical
    assert again.outcome.source.path == first.source.path and first.source.path.is_file()
    # 所在は記録されたページURLで、秘匿クエリは使わない。
    assert store.load_revisions("example:x1")[0]["location"]["requested_url"] == "https://example.test/x1"

    backend.media["x1"] = _flac(tmp_path / "b.flac", seed=5)
    first.source.path.unlink()
    changed = service.reacquire(first.source.sha256, backend=backend, pacing=NO_WAIT)
    assert changed.outcome.ok and not changed.identical

    with pytest.raises(KeyError):
        service.reacquire("0" * 64, backend=backend)


def test_opus_in_webm_is_kept_as_served_and_decodes_for_separation(roots, tmp_path, serve):
    """YouTubeの最良音声と同じ形（WebM/Opus）を、変換せず取得して分離の入力準備へ渡せる。"""
    import subprocess

    from lss.core.audio_input import prepare_input
    from lss.paths import resource_root

    source = _flac(tmp_path / "tone.flac", seconds=3.0)
    webm = tmp_path / "served.webm"
    subprocess.run([str(resource_root() / "bin" / "ffmpeg.exe"), "-nostdin", "-v", "error", "-i", str(source),
                    "-ar", "48000", "-c:a", "libopus", "-b:a", "128k", str(webm)], check=True)
    url, handler = serve(webm.read_bytes(), "clip.webm", "audio/webm")

    [outcome] = service.acquire(url, backend=YtDlpBackend(), pacing=NO_WAIT)
    assert outcome.ok, outcome.failure
    assert outcome.source.path.suffix == ".webm"
    assert outcome.source.path.read_bytes() == handler.payload
    media = outcome.source.record["media"]
    assert (media["audio_channels"], media["sample_rate"]) == (2, 48000)
    assert abs(media["timeline"]["decoded_seconds"] - 3.0) <= 0.02

    with prepare_input(outcome.source.path, work_root=tmp_path / "work") as prepared:
        info = sf.info(prepared)
    assert (info.channels, info.samplerate) == (2, 48000)
    assert abs(info.frames - 3 * 48000) <= 48000 // 50


def test_rejection_with_browser_cookies_means_the_login_must_be_renewed():
    from yt_dlp.utils import DownloadError

    from lss.acquisition.ytdlp import _failure

    rejected = DownloadError("ERROR: [niconico] sm1: This video requires login. Use --cookies-from-browser")
    assert _failure(rejected, AuthMethod()).kind is FailureKind.AUTH_REQUIRED
    assert _failure(rejected, AuthMethod.browser_cookies("firefox")).kind is FailureKind.AUTH_EXPIRED


def test_hls_audio_like_niconico_and_tver_is_concatenated_without_conversion(roots, tmp_path):
    """HLSの音声（ニコニコ・TVerの選択形式と同じ配信方式）を、変換せずにつないで分離入力へ復号できる。"""
    import subprocess

    from lss.core.audio_input import prepare_input
    from lss.paths import resource_root

    source = _flac(tmp_path / "tone.flac", seconds=7.0)
    hls = tmp_path / "hls"
    hls.mkdir()
    subprocess.run([str(resource_root() / "bin" / "ffmpeg.exe"), "-nostdin", "-v", "error", "-i", str(source),
                    "-ar", "48000", "-c:a", "aac", "-b:a", "128k", "-f", "hls", "-hls_time", "2",
                    "-hls_playlist_type", "vod", "-hls_segment_filename", str(hls / "seg%03d.ts"),
                    str(hls / "index.m3u8")], check=True)
    files = {path.name: path.read_bytes() for path in hls.iterdir()}

    class Directory(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            name = self.path.lstrip("/").split("?", 1)[0]
            if name not in files:
                self.send_error(404)
                return
            kind = "application/vnd.apple.mpegurl" if name.endswith(".m3u8") else "video/mp2t"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(files[name])))
            self.end_headers()
            self.wfile.write(files[name])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Directory)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/index.m3u8"
        [outcome] = service.acquire(url, backend=YtDlpBackend(), pacing=NO_WAIT)
    finally:
        server.shutdown()
        server.server_close()

    assert outcome.ok, outcome.failure
    segments = b"".join(files[name] for name in sorted(files) if name.endswith(".ts"))
    assert outcome.source.path.read_bytes() == segments
    # 中身はMPEG-TSなので、.mp4ではなく中身と一致する拡張子で置く（合成したHLS音声の形式）。
    assert outcome.source.path.suffix == ".ts"
    assert outcome.source.record["format"]["container_detected"] == "mpegts"
    assert outcome.source.record["format"]["selected"][0]["protocol"] == "m3u8_native"
    with prepare_input(outcome.source.path, work_root=tmp_path / "work") as prepared:
        info = sf.info(prepared)
    assert (info.channels, info.samplerate) == (2, 48000)
    assert abs(info.frames - 7 * 48000) <= 48000 // 20


def test_stop_request_keeps_the_partial_file_and_skips_remaining_items(roots, media_server):
    url, handler = media_server
    checks = {"n": 0}

    def stop_after_first_progress():
        checks["n"] += 1
        return checks["n"] > 1

    [stopped] = service.acquire(url, backend=YtDlpBackend(), pacing=NO_WAIT,
                                should_stop=stop_after_first_progress)
    assert stopped.failure.kind is FailureKind.CANCELLED and not stopped.failure.retryable
    assert not list(roots[1].rglob("*.flac"))
    assert store.load_revisions("generic:song") == []

    [resumed] = service.acquire(url, backend=YtDlpBackend(), pacing=NO_WAIT)
    assert resumed.ok and resumed.source.path.read_bytes() == handler.payload

    backend = FakeBackend({}, playlist=["a", "b"])
    outcomes = service.acquire("https://example.test/list", backend=backend, pacing=NO_WAIT,
                               should_stop=lambda: True)
    assert [o.failure.kind for o in outcomes] == [FailureKind.CANCELLED, FailureKind.CANCELLED]
    assert backend.downloads == []


def test_insufficient_space_is_reported_before_downloading(roots, tmp_path, monkeypatch):
    from collections import namedtuple

    backend = FakeBackend({"x1": _flac(tmp_path / "a.flac")})
    real = service.shutil.disk_usage
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(service.shutil, "disk_usage", lambda _path: usage(10, 10, 1024))
    [outcome] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    assert outcome.failure.kind is FailureKind.INSUFFICIENT_SPACE
    assert backend.downloads == []

    monkeypatch.setattr(service.shutil, "disk_usage", real)
    [again] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    assert again.ok
    # サイズを申告しない形式（HLS等）では、推測で止めない。
    assert service.expected_bytes({"format": {"selected": [{"format_id": "hls"}]}}) is None


def test_a_stale_tool_failure_triggers_a_verified_update_and_one_retry(roots, tmp_path, monkeypatch):
    from lss.acquisition import provider

    media = _flac(tmp_path / "a.flac")

    class Stale(FakeBackend):
        def describe(self, entry, auth):
            raise AcquisitionFailure(FailureKind.EXTRACTOR_FAILED, "Unable to extract player response")

    installed = {"done": False}
    backends = []

    def default_backend():
        backend = FakeBackend({"x1": media}) if installed["done"] else Stale({"x1": media})
        backends.append(backend)
        return backend

    checks = []

    def ensure(*, after_failure=False, **_kw):
        checks.append(after_failure)
        if after_failure:
            installed["done"] = True
            return {"status": "installed"}
        return None

    monkeypatch.setattr(service, "default_backend", default_backend)
    monkeypatch.setattr(provider, "ensure", ensure)
    [outcome] = service.acquire("https://example.test/x1", pacing=NO_WAIT)
    assert outcome.ok and checks == [False, True] and len(backends) == 2

    # 新しい版が無ければやり直さず、失敗をそのまま返す。
    installed["done"] = False
    monkeypatch.setattr(provider, "ensure", lambda **_kw: None)
    [failed] = service.acquire("https://example.test/x2", pacing=NO_WAIT)
    assert failed.failure.kind is FailureKind.EXTRACTOR_FAILED


def test_default_library_is_inside_the_application_folder(monkeypatch):
    from lss.paths import app_root

    monkeypatch.delenv(store.LIBRARY_ENV, raising=False)
    assert store.library_root() == app_root() / "Downloads"
    # 状態領域の中に置くと、画面の履歴から内部ファイルとして除外されてしまう。
    assert not store.library_root().is_relative_to(app_root() / ".state")


def test_adding_the_download_library_folder_includes_acquired_containers_only_there(roots, tmp_path):
    from lss.core.session import iter_input_files

    _, library = roots
    acquired = library / "Youtube" / "曲 [x].webm"
    acquired.parent.mkdir(parents=True)
    acquired.write_bytes(b"webm")
    tver = library / "TVer" / "回 [ep].ts"
    tver.parent.mkdir(parents=True)
    tver.write_bytes(b"ts")
    elsewhere = tmp_path / "videos" / "clip.mp4"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"mp4")

    assert set(iter_input_files([library])) == {acquired.resolve(), tver.resolve()}
    assert iter_input_files([elsewhere.parent]) == []
    # 直接指定したファイルは従来どおり受け付ける。
    assert iter_input_files([elsewhere]) == [elsewhere.resolve()]


def test_outputs_of_acquired_sources_inherit_recorded_title_without_guessing_the_artist(roots, tmp_path):
    from lss.output import writer

    media = tmp_path / "a.webm.flac"
    import shutil as _shutil

    _shutil.copyfile(_flac(tmp_path / "plain.flac"), media)
    backend = FakeBackend({"x1": media})
    [outcome] = service.acquire("https://example.test/x1", backend=backend, pacing=NO_WAIT)
    tags = writer.read_source_metadata(outcome.source.path).tags
    assert tags["TITLE"] == "曲 x1"
    assert "ARTIST" not in tags  # 投稿者名を曲のアーティストにしない

    # 内容が記録と違えば、記録の事実を付けない。
    outcome.source.path.write_bytes(outcome.source.path.read_bytes() + b"\0")
    assert "TITLE" not in writer.read_source_metadata(outcome.source.path).tags


def test_container_duration_is_read_from_the_ffmpeg_input_header():
    header = "Input #0, mov,mp4,m4a, from 'a.m4a':\n  Duration: 00:34:15.64, start: 0.000000, bitrate: 134 kb/s\n"
    assert service.parse_container_duration(header) == pytest.approx(2055.64)
    assert service.parse_container_duration("  Duration: N/A, bitrate: N/A") is None


def test_timeline_records_audio_missing_against_the_container_duration(roots, tmp_path):
    import subprocess

    from lss.paths import resource_root

    state, library = roots
    source = _flac(tmp_path / "long.flac", seconds=20.0)
    m4a = tmp_path / "served.m4a"
    subprocess.run([str(resource_root() / "bin" / "ffmpeg.exe"), "-nostdin", "-v", "error", "-i", str(source),
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(m4a)], check=True)
    data = m4a.read_bytes()
    broken = tmp_path / "broken.m4a"
    broken.write_bytes(data[: len(data) // 2])  # 見出しと先頭は読めるが、後半の音声が無い
    backend = FakeBackend({"b1": broken})

    [outcome] = service.acquire("https://example.test/b1", backend=backend, pacing=NO_WAIT)

    # 途中で切れた取得物は、コンテナの時刻より復号できた標本が短いことが来歴に残る。
    assert outcome.ok
    timeline = outcome.source.record["media"]["timeline"]
    assert timeline["container_seconds"] == pytest.approx(20.0, abs=0.1)
    assert timeline["difference_seconds"] < -5
