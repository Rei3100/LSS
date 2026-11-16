"""Interrupted first-use downloads stay resumable: free-space admission counts only the bytes still to fetch."""
import hashlib
import io
from types import SimpleNamespace

from lss.bgm import scene_native as native

GIB = 1024 ** 3


class Response(io.BytesIO):
    def __init__(self, body, status, content_range=None):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Range': content_range} if content_range else {}


def test_resume_needs_space_only_for_the_missing_suffix(monkeypatch, tmp_path):
    body = b'abcdefghij'
    target = tmp_path / 'model.gguf'
    part = tmp_path / 'model.gguf.part'
    part.write_bytes(body[:6])
    size = 10 * GIB  # declared size; the fake server only serves the logical suffix below
    # The interrupted prefix already occupies its space: free = remaining + reserve, not size + reserve.
    monkeypatch.setattr(native.shutil, 'disk_usage',
                        lambda _: SimpleNamespace(free=(size - 6) + 4 * GIB))
    monkeypatch.setattr(native, '_download_ssl_context', lambda: object())
    monkeypatch.setattr(native, 'DOWNLOAD_RANGE_BYTES', size)
    requested = []

    def fetch(request, **kwargs):
        requested.append(request.get_header('Range'))
        raise native.SceneModelsError('stop_after_admission')

    monkeypatch.setattr(native.urllib.request, 'urlopen', fetch)
    try:
        native._download('https://example.invalid/model', target, sha256=hashlib.sha256(body).hexdigest(),
                         size=size)
    except native.SceneModelsError as exc:
        assert str(exc) == 'stop_after_admission'
    assert requested == [f'bytes=6-{size - 1}']
    assert part.read_bytes() == body[:6]


def test_bootstrap_capacity_credits_bytes_already_downloaded(monkeypatch, tmp_path):
    root = tmp_path / 'native'
    weights = native._weights_path(root, native.LLM_SPEC)
    weights.parent.mkdir(parents=True)
    partial = weights.with_name(weights.name + '.part')
    partial.write_bytes(b'x' * 1000)
    downloads = root / 'downloads'
    downloads.mkdir()
    archive_bytes = 0
    for engine in (native.WHISPER_ENGINE, native.LLAMA_ENGINE):
        for spec in engine['archives']:
            archive_bytes += spec['bytes']
    reserve = 4 * GIB
    full = native.ASR_SPEC['bytes'] + native.LLM_SPEC['bytes'] + archive_bytes * 4 + reserve
    # Exactly the full first-run requirement minus the 1000 bytes the user already holds.
    monkeypatch.setattr(native.shutil, 'disk_usage', lambda _: SimpleNamespace(free=full - 1000))
    native._bootstrap_capacity(root)
