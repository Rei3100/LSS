from __future__ import annotations

import pytest

from lss.bgm import work_info

LISTING = """<html><body><h2>第150話 灯台へ、風に乗って！</h2><p>ソラが浜辺でホノトカゲと紙の帆を組み立て、風向きを調べる。</p>
<h2>第151話 ミナと星の時計</h2><p>ミナとフワモンは広場で止まった星時計を直し、新しい歯車を探す。</p></body></html>"""


class Courtesy:
    def __init__(self):
        self.slots, self.reports = [], []

    def slot(self, url, purpose="request"):
        import contextlib

        self.slots.append(url)
        return contextlib.nullcontext()

    def report(self, url, outcome=None):
        self.reports.append(outcome)


def opener_for(pages):
    calls = []

    def opener(url):
        calls.append(url)
        if url not in pages:
            raise work_info.WorkInfoUnavailable(url)
        return pages[url]
    return opener, calls


def test_episode_blocks_are_read_from_the_page_lss_fetched_itself(tmp_path):
    opener, calls = opener_for({"https://example.jp/robots.txt": "User-agent: *\nAllow: /",
                                "https://example.jp/anime/episodes/": LISTING})
    found = work_info.official_episode_pages("https://example.jp/anime/episodes/", tmp_path, pages=2,
                                             opener=opener, courtesy=Courtesy())
    assert set(found) == {150, 151} and "ホノトカゲ" in found[150]["text"] and "フワモン" in found[151]["text"]
    # Stored once: a second run reads nothing from the network.
    opener2, calls2 = opener_for({})
    again = work_info.official_episode_pages("https://example.jp/anime/episodes/", tmp_path, pages=1,
                                             opener=opener2, courtesy=Courtesy())
    assert again[150]["text"] == found[150]["text"] and calls2 == []


def test_robots_txt_that_refuses_lss_is_honoured(tmp_path):
    opener, calls = opener_for({"https://example.jp/robots.txt": "User-agent: LSS-work-info\nDisallow: /",
                                "https://example.jp/anime/episodes/": LISTING})
    with pytest.raises(work_info.WorkInfoUnavailable):
        work_info.fetch("https://example.jp/anime/episodes/", tmp_path, opener=opener, courtesy=Courtesy())
    assert calls == ["https://example.jp/robots.txt"]


def test_offline_or_missing_pages_give_nothing_instead_of_failing(tmp_path):
    opener, _ = opener_for({})
    assert work_info.official_episode_pages("https://example.jp/x/", tmp_path, opener=opener, courtesy=Courtesy()) == {}
    with pytest.raises(work_info.WorkInfoUnavailable):
        work_info.fetch("http://example.jp/plain", tmp_path, opener=opener, courtesy=Courtesy())


def test_naming_context_uses_official_episode_before_wikipedia(tmp_path):
    opener, calls = opener_for({"https://official.jp/robots.txt": "User-agent: *\nAllow: /",
                                "https://official.jp/anime/": '<a href="/anime/episodes/">各話あらすじ</a>',
                                "https://official.jp/anime/episodes/": LISTING})
    found = work_info.naming_context("公式サイト\nhttps://official.jp/anime/", 150, "作品", tmp_path,
                                     opener=opener, courtesy=Courtesy())
    assert found["source_kind"] == "official_episode"
    assert found["url"] == "https://official.jp/anime/episodes/"
    assert "ホノトカゲ" in found["text"]
    assert found["text_sha256"] == __import__("hashlib").sha256(found["text"].encode("utf-8")).hexdigest()
    assert found["page_text_sha256"]
    assert not any("wikipedia.org" in url for url in calls)


def test_wikipedia_work_url_is_quoted_and_h1_must_match_title(tmp_path):
    url = "https://ja.wikipedia.org/wiki/%E4%BD%9C%E5%93%81%2F%20%E7%89%B9%E5%88%A5%E7%B7%A8"
    html = "<html><h1>作品／ 特別編</h1><p>公開されている作品の説明。</p></html>"
    opener, calls = opener_for({"https://ja.wikipedia.org/robots.txt": "User-agent: *\nAllow: /", url: html})
    found = work_info.naming_context("", None, "作品/ 特別編", tmp_path, opener=opener, courtesy=Courtesy())
    assert found["source_kind"] == "wikipedia_work"
    assert found["url"] == url
    assert found["text"] and len(found["text"]) <= 20000
    assert found["page_text_sha256"]
    assert calls == ["https://ja.wikipedia.org/robots.txt", url]

    other_root = tmp_path / "wrong"
    wrong, _ = opener_for({"https://ja.wikipedia.org/robots.txt": "User-agent: *\nAllow: /", url:
                           "<h1>作品 (曖昧さ回避)</h1><p>説明</p>"})
    assert work_info.naming_context("", None, "作品/ 特別編", other_root, opener=wrong,
                                    courtesy=Courtesy()) is None

    marked, _ = opener_for({"https://ja.wikipedia.org/robots.txt": "User-agent: *\nAllow: /", url:
                            "<h1>作品／ 特別編</h1><p>曖昧さ回避のためのページです。</p><p>別作品</p>"})
    assert work_info.naming_context("", None, "作品/ 特別編", tmp_path / "marked", opener=marked,
                                    courtesy=Courtesy()) is None

    later_match, _ = opener_for({"https://ja.wikipedia.org/robots.txt": "User-agent: *\nAllow: /", url:
                                "<h1>別作品</h1><h1>作品／ 特別編</h1><p>説明</p>"})
    assert work_info.naming_context("", None, "作品/ 特別編", tmp_path / "later-match", opener=later_match,
                                    courtesy=Courtesy()) is None

    later_mismatch, _ = opener_for({"https://ja.wikipedia.org/robots.txt": "User-agent: *\nAllow: /", url:
                                   "<h1>作品／ 特別編</h1><h1>別作品</h1><p>説明</p>"})
    assert work_info.naming_context("", None, "作品/ 特別編", tmp_path / "later-mismatch", opener=later_mismatch,
                                    courtesy=Courtesy())["source_kind"] == "wikipedia_work"


def test_unavailable_official_site_falls_back_and_wikipedia_cache_avoids_http(tmp_path):
    wiki_url = "https://ja.wikipedia.org/wiki/%E4%BD%9C%E5%93%81"
    opener, calls = opener_for({
        "https://official.jp/robots.txt": "User-agent: *\nAllow: /",
        "https://ja.wikipedia.org/robots.txt": "User-agent: *\nAllow: /",
        wiki_url: "<h1>作品</h1><p>公式ページが利用できない場合に使う作品情報。</p>",
    })
    first = work_info.naming_context("公式サイト\nhttps://official.jp/anime/", 151, "作品", tmp_path,
                                     opener=opener, courtesy=Courtesy())
    assert first["source_kind"] == "wikipedia_work"
    assert "https://official.jp/anime/" in calls
    assert calls[-2:] == ["https://ja.wikipedia.org/robots.txt", wiki_url]

    no_network, repeated_calls = opener_for({})
    second = work_info.naming_context("", None, "作品", tmp_path, opener=no_network, courtesy=Courtesy())
    assert second == first
    assert repeated_calls == []


def test_wikipedia_unavailable_refused_offline_and_legacy_cache_without_h1(tmp_path):
    url = "https://ja.wikipedia.org/wiki/%E4%BD%9C%E5%93%81"
    refused, calls = opener_for({"https://ja.wikipedia.org/robots.txt": "User-agent: LSS-work-info\nDisallow: /"})
    assert work_info.naming_context("", None, "作品", tmp_path / "refused", opener=refused,
                                    courtesy=Courtesy()) is None
    assert calls == ["https://ja.wikipedia.org/robots.txt"]

    offline, _ = opener_for({})
    assert work_info.naming_context("", None, "作品", tmp_path / "offline", opener=offline,
                                    courtesy=Courtesy()) is None

    legacy_root = tmp_path / "legacy"
    store = work_info._root(legacy_root) / "pages" / (__import__("hashlib").sha256(url.encode()).hexdigest()[:24] + ".json")
    store.parent.mkdir(parents=True)
    store.write_text(__import__("json").dumps({"url": url, "text": "作品\n説明"}), encoding="utf-8")
    no_network, legacy_calls = opener_for({})
    assert work_info.naming_context("", None, "作品", legacy_root, opener=no_network,
                                    courtesy=Courtesy()) is None
    assert legacy_calls == []
