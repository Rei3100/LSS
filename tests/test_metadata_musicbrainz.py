from __future__ import annotations

import contextlib
import json
import urllib.parse
from pathlib import Path

import pytest

from lss.metadata import fetch
from lss.metadata.sources import musicbrainz as mb

ALPHA = "00000000-0000-4000-8000-000000000001"
ALPHA_TITLE = "星空冒険団 〜ハルとカイの冒険〜"
OTHER_COMPOSER_IDS = ("00000000-0000-4000-8000-000000000002", "00000000-0000-4000-8000-000000000003",
               "00000000-0000-4000-8000-000000000004")


def _read(name: str) -> str:
    artist = {"name": "Artist Y", "aliases": [{"name": "ソラノアオ", "locale": "ja", "primary": True}]}
    credit = [{"name": "ソラノアオ", "artist": artist}]
    title = ALPHA_TITLE + " オリジナル・サウンドトラック"
    release = {"id": ALPHA, "title": title, "date": "2024-06-19", "barcode": "4900000000001",
               "artist-credit": credit, "release-group": {"id": "synthetic-group-a"},
               "label-info": [{"catalog-number": "SYNX-10001", "label": {"name": "Synthetic Label"}}],
               "relations": [{"target-type": "url", "url": {"resource": "https://vgmdb.net/album/900001"}}]}
    if name == "lookup_alpha.json":
        tracks = [{"position": i, "title": f"合成曲{i}", "length": 60000,
                   "recording": {"artist-credit": credit}} for i in range(1, 43)]
        tracks[0].update(title="序曲 (TV Anime ver. 1)", length=15240)
        tracks[9]["recording"]["relations"] = [
            {"target-type": "artist", "type": "arranger", "artist": artist},
            {"target-type": "work", "type": "performance", "work": {"relations": [
                {"target-type": "artist", "type": "composer", "artist": {"name": "作家甴"}}]}}]
        tracks[18]["recording"]["artist-credit"] = [{"artist": {"name": "作家电"}}]
        return json.dumps({**release, "media": [{"position": 1, "tracks": tracks}]}, ensure_ascii=False)
    if name in ("search_series_c_artist.json", "search_series_c_title.json"):
        return '{"count":0,"releases":[]}'
    releases = [release]
    if name == "search_franchise_title.json":
        releases += [{**release, "id": mbid, "release-group": {"id": "other-" + mbid},
                      "artist-credit": [{"name": "別作家", "artist": {"name": "別作家"}}]}
                     for mbid in OTHER_COMPOSER_IDS]
    return json.dumps({"count": len(releases), "releases": releases}, ensure_ascii=False)


class Courtesy:
    @contextlib.contextmanager
    def slot(self, url, purpose="request", **_):
        yield

    def report(self, url, outcome=None, **_):
        pass


class Server:
    """Serves authored synthetic MusicBrainz bodies; records every URL asked."""

    def __init__(self, searches: dict[str, str]):
        self.searches, self.asked = searches, []

    def __call__(self, url: str) -> str:
        if url.endswith("/robots.txt"):
            return "User-agent: *\nDisallow:\n"
        self.asked.append(url)
        if f"/release/{ALPHA}?" in url:
            return _read("lookup_alpha.json")
        if "/release/?" in url:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["query"][0]
            for marker, name in self.searches.items():
                if marker in query:
                    return _read(name)
            return '{"count":0,"releases":[]}'
        raise fetch.Unavailable(f"{url}: HTTP 404")


def _alpha_server():
    return Server({'artist:"ソラノアオ"': "search_alpha_artist.json", '"ハルとカイの冒険"': "search_alpha_title.json"})


def test_finds_alpha_release_with_full_fields(tmp_path):
    server = _alpha_server()
    found = mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], "2023-04-14", tmp_path, opener=server, courtesy=Courtesy())
    assert [r.url for r in found] == [f"https://musicbrainz.org/release/{ALPHA}"]
    r = found[0]
    assert r.source == "musicbrainz" and r.release_date == "2024-06-19"
    assert r.title == "星空冒険団 〜ハルとカイの冒険〜 オリジナル・サウンドトラック"
    assert r.catalog == "SYNX-10001" and r.barcode == "4900000000001" and r.label == "Synthetic Label"
    assert r.discs == 1 and len(r.tracks) == 42
    assert {t.disc for t in r.tracks} == {1} and r.tracks[0].number == 1 and r.tracks[-1].number == 42
    assert r.tracks[0].title == "序曲 (TV Anime ver. 1)" and r.tracks[0].length_seconds == pytest.approx(15.24)
    assert "https://vgmdb.net/album/900001" in r.related_urls
    assert r.raw_sha256 and r.fetched_at
    assert [(c.role_key, c.name) for c in r.credits] == [("artist", "ソラノアオ")]
    assert "Artist Y" in r.credits[0].names


def test_track_credits_from_recording_and_work_relations(tmp_path):
    r = mb.release(ALPHA, tmp_path, opener=_alpha_server(), courtesy=Courtesy())
    track10 = r.tracks[9]
    keys = {(c.role_key, c.name) for c in track10.credits}
    assert ("arranger", "ソラノアオ") in keys
    assert ("composer", "作家甴") in keys
    # MusicBrainz states role credits only for some tracks; every track carries its recording artist credit.
    assert any(all(c.role_key == "artist" for c in t.credits) for t in r.tracks)
    assert ("artist", "作家电") in {(c.role_key, c.name) for c in r.tracks[18].credits}


def test_instrument_and_vocal_attributes_become_role_keys():
    artist = {"name": "山田太郎", "sort-name": "Yamada, Taro"}
    inst = mb._credits_for("instrument", ["flute", "piccolo"], artist, None)
    assert [c.role_key for c in inst] == ["instrument:flute", "instrument:piccolo"]
    assert mb._credits_for("vocal", ["lead vocals"], artist, None)[0].role_key == "vocals"
    assert mb._credits_for("mix", [], artist, None)[0].role_key == "mix"


def test_japanese_alias_preferred_for_romanized_artist():
    artist = {"name": "Artist Y", "sort-name": "Artist Y", "aliases": [{"name": "ソラノアオ", "locale": "ja", "primary": True}]}
    name, names = mb._artist_name(artist)
    assert name == "ソラノアオ" and "Artist Y" in names


def test_composer_filter_rejects_other_series_ost(tmp_path):
    server = Server({'artist:"ソラノアオ"': "search_alpha_artist.json", '"星空冒険団"': "search_franchise_title.json"})
    found = mb.search_soundtracks("星空冒険団", ["ソラノアオ"], "2019-01-01", tmp_path, opener=server, courtesy=Courtesy())
    assert [r.url.rsplit("/", 1)[1] for r in found] == [ALPHA]
    assert not any(mbid in url for url in server.asked for mbid in OTHER_COMPOSER_IDS)   # Other composers are never even looked up


def test_composer_name_is_whitespace_and_width_insensitive(tmp_path):
    found = mb.search_soundtracks(ALPHA_TITLE, ["ソラノ アオ"], None, tmp_path, opener=_alpha_server(), courtesy=Courtesy())
    assert len(found) == 1


def test_wrong_composer_returns_nothing(tmp_path):
    server = Server({'artist:"別作家"': "search_alpha_artist.json", '"ハルとカイの冒険"': "search_alpha_title.json"})
    assert mb.search_soundtracks(ALPHA_TITLE, ["別作家"], None, tmp_path, opener=server, courtesy=Courtesy()) == []


def test_not_before_excludes_earlier_release(tmp_path):
    assert mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], "2024-07-01", tmp_path, opener=_alpha_server(), courtesy=Courtesy()) == []
    assert mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], "2024-06-19", tmp_path, opener=_alpha_server(), courtesy=Courtesy())


def test_series_without_soundtrack_returns_empty(tmp_path):
    server = Server({'artist:"作家畐"': "search_series_c_artist.json", '"仮面の勇者マイナ"': "search_series_c_title.json"})
    assert mb.search_soundtracks("仮面の勇者マイナ", ["作家畐"], "2026-01-01", tmp_path, opener=server, courtesy=Courtesy()) == []


def test_offline_search_returns_empty_not_exception(tmp_path):
    def offline(url):
        if url.endswith("/robots.txt"):
            raise fetch.Unavailable("offline")
        raise fetch.Unavailable(f"{url}: URLError")

    assert mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], None, tmp_path, opener=offline, courtesy=Courtesy()) == []


def test_release_lookup_offline_raises_unavailable(tmp_path):
    def offline(url):
        raise fetch.Unavailable("offline")

    with pytest.raises(fetch.Unavailable):
        mb.release(ALPHA, tmp_path, opener=offline, courtesy=Courtesy())


def test_second_search_is_served_from_cache(tmp_path):
    server = _alpha_server()
    mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], None, tmp_path, opener=server, courtesy=Courtesy())
    asked = len(server.asked)
    mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], None, tmp_path, opener=server, courtesy=Courtesy())
    assert len(server.asked) == asked


def _failing(status):
    def opener(url):
        if url.endswith("/robots.txt"):
            return "User-agent: *\nDisallow:\n"
        error = fetch.Unavailable(f"{url}: HTTP {status}")
        if status:
            error.status = status
        raise error
    return opener


def test_strict_search_raises_on_transient_failure(tmp_path):
    for status in (503, None):
        with pytest.raises(fetch.Unavailable):
            mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], None, tmp_path / str(status), opener=_failing(status),
                                  courtesy=Courtesy(), strict=True)


def test_strict_search_treats_not_found_as_no_soundtrack(tmp_path):
    assert mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], None, tmp_path, opener=_failing(404),
                                 courtesy=Courtesy(), strict=True) == []


def test_non_strict_search_still_returns_empty_on_transient_failure(tmp_path):
    assert mb.search_soundtracks(ALPHA_TITLE, ["ソラノアオ"], None, tmp_path, opener=_failing(503),
                                 courtesy=Courtesy()) == []
