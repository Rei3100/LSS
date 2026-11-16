"""iTunes adapter: raw store titles, discs/tracks/lengths, album artist credit, artwork, offline via fixtures."""
import contextlib
import json
import urllib.parse
from pathlib import Path

import pytest

from lss.metadata import fetch
from lss.metadata.sources import itunes

ID = 1900000001


def _album_data():
    head = {"wrapperType": "collection", "collectionId": ID,
            "collectionName": "テレビアニメ 星空冒険団～ハルとカイの冒険～ オリジナル・サウンドトラック",
            "collectionViewUrl": f"https://music.apple.com/jp/album/{ID}", "artistName": "ソラノアオ",
            "releaseDate": "2024-06-19T00:00:00Z", "artworkUrl100": "https://is1-ssl.mzstatic.com/synthetic.jpg"}
    tracks = [{"wrapperType": "track", "kind": "song", "trackName": f"合成曲{i}", "trackNumber": i,
               "discNumber": 1, "discCount": 1, "artistName": "ソラノアオ", "trackTimeMillis": 60000}
              for i in range(1, 43)]
    tracks[0].update(trackName="序曲～Tv Anime Ver.1 (Instrumental)", trackTimeMillis=15240)
    tracks[30]["trackName"] = "ソララインズの合図 (Instrumental)"
    tracks[10]["trackName"] = "灯りのうた"
    return {"resultCount": 43, "results": [head, *tracks]}


class Courtesy:
    def __init__(self):
        self.slots, self.reports = [], []

    @contextlib.contextmanager
    def slot(self, url, purpose="request"):
        self.slots.append((url, purpose))
        yield

    def report(self, url, outcome=None):
        self.reports.append((url, outcome))


def opener_for(requests):
    def opener(url):
        if url.endswith("/robots.txt"):
            return "User-agent: *\nAllow: /\n"
        requests.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        if "/lookup" in url and query["id"] == [str(ID)]:
            return json.dumps(_album_data(), ensure_ascii=False)
        if "/search" in url:
            return json.dumps({"results": [_album_data()["results"][0]]}, ensure_ascii=False)
        error = fetch.Unavailable(url)
        error.status = 404
        raise error
    return opener


def test_album_keeps_store_titles_raw_and_reads_discs_tracks_and_lengths(tmp_path):
    requests, courtesy = [], Courtesy()
    release = itunes.album(ID, tmp_path, opener=opener_for(requests), courtesy=courtesy)
    assert requests == [f"https://itunes.apple.com/lookup?id={ID}&entity=song&country=jp&lang=ja_jp"]
    assert release.source == "itunes"
    assert release.title == "テレビアニメ 星空冒険団～ハルとカイの冒険～ オリジナル・サウンドトラック"
    assert release.release_date == "2024-06-19"
    assert release.discs == 1 and len(release.tracks) == 42
    assert [t.number for t in release.tracks] == list(range(1, 43))
    assert {t.disc for t in release.tracks} == {1}
    first = release.tracks[0]
    assert first.title == "序曲～Tv Anime Ver.1 (Instrumental)" and first.titles["ja"] == first.title
    assert first.length_seconds == pytest.approx(15.24)
    # the label's mistake and suffix survive untouched
    assert release.tracks[30].title == "ソララインズの合図 (Instrumental)"
    assert release.tracks[10].title == "灯りのうた"
    assert [(c.role, c.role_key, c.name) for c in release.credits] == [("artistName", "artist", "ソラノアオ")]
    assert all(not t.credits for t in release.tracks)       # no composerName / differing artist in this album
    assert release.related_urls and release.related_urls[0].startswith("https://is1-ssl.mzstatic.com/")
    assert release.raw_sha256 and release.fetched_at
    assert courtesy.slots and courtesy.slots[0][1] == "itunes"


def test_album_is_cached_after_the_first_read(tmp_path):
    requests = []
    itunes.album(ID, tmp_path, opener=opener_for(requests), courtesy=Courtesy())
    itunes.album(ID, tmp_path, opener=opener_for(requests), courtesy=Courtesy())
    assert len(requests) == 1


def test_composer_name_becomes_a_track_credit(tmp_path):
    data = _album_data()
    data["results"][1]["composerName"] = "ソラノアオ"
    data["results"][2]["artistName"] = "作家申"

    def opener(url):
        return "User-agent: *\nAllow: /\n" if url.endswith("/robots.txt") else json.dumps(data, ensure_ascii=False)
    release = itunes.album(ID, tmp_path, opener=opener, courtesy=Courtesy())
    assert [(c.role, c.role_key, c.name) for c in release.tracks[0].credits] == [("composerName", "composer", "ソラノアオ")]
    assert [(c.role_key, c.name) for c in release.tracks[1].credits] == [("artist", "作家申")]


def test_search_albums_lists_collections_without_tracks(tmp_path):
    requests = []
    found = itunes.search_albums("星空冒険団 ハルとカイ", tmp_path, opener=opener_for(requests), courtesy=Courtesy())
    assert requests and requests[0].startswith("https://itunes.apple.com/search?term=")
    assert "country=jp&lang=ja_jp&media=music&entity=album" in requests[0]
    assert len(found) == 1 and found[0].source == "itunes" and found[0].tracks == ()
    assert found[0].release_date == "2024-06-19"
    assert found[0].url.startswith("https://music.apple.com/jp/album/")


def test_unavailable_gives_empty_search_and_raises_for_album(tmp_path):
    def refuse(url):
        error = fetch.Unavailable(url)
        error.status = 404
        raise error
    assert itunes.search_albums("x", tmp_path, opener=refuse, courtesy=Courtesy()) == []
    with pytest.raises(fetch.Unavailable):
        itunes.album(1, tmp_path / "other", opener=refuse, courtesy=Courtesy())
    assert itunes.search_albums("  ", tmp_path) == []
