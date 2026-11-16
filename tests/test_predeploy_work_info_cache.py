"""A torn page cache or an unknown page charset is "page unavailable", never a permanent failure."""
from __future__ import annotations

import hashlib
import io

import pytest

from lss.bgm import work_info
from tests.test_bgm_work_info import LISTING, Courtesy, opener_for

URL = "https://example.jp/anime/episodes/"


def test_torn_page_cache_is_refetched_instead_of_failing_forever(tmp_path):
    store = work_info._root(tmp_path) / "pages" / (hashlib.sha256(URL.encode()).hexdigest()[:24] + ".json")
    store.parent.mkdir(parents=True)
    store.write_text('{"url": "' + URL + '", "te', encoding="utf-8")  # a write cut off by power loss / crash
    opener, calls = opener_for({"https://example.jp/robots.txt": "User-agent: *\nAllow: /", URL: LISTING})
    entry = work_info.fetch(URL, tmp_path, opener=opener, courtesy=Courtesy())
    assert "ホノトカゲ" in entry["text"] and URL in calls
    assert work_info.fetch(URL, tmp_path, opener=opener_for({})[0], courtesy=Courtesy())["text"] == entry["text"]


def test_unknown_declared_charset_is_page_unavailable(monkeypatch):
    class Response(io.BytesIO):
        class headers:
            @staticmethod
            def get_content_charset():
                return "x-unknown-charset"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(work_info.urllib.request, "urlopen", lambda *a, **k: Response(b"<p>x</p>"))
    with pytest.raises(work_info.WorkInfoUnavailable):
        work_info._http(URL)
