from __future__ import annotations

import contextlib
from pathlib import Path

from lss.metadata import fetch
from lss.metadata.sources import vgmdb

def _album_html():
    def spans(css, ja, en):
        return f'<span class="{css}" lang="ja">{ja}</span><span class="{css}" lang="en">{en}</span>'

    def info(label, value):
        return f'<tr><td><span class="label"><b>{label}</b></span></td><td>{value}</td></tr>'

    def credit(en, ja, people):
        label = spans("artistname", ja, en)
        links = "".join('<a href="/artist/1">' + spans("artistname", local, latin) + '</a>'
                        for local, latin in people)
        return f'<tr class="maincred"><td>{label}</td><td>{links}</td></tr>'

    title = spans("albumtitle", "テレビアニメ「星空冒険団～ハルとカイの冒険～」オリジナル・サウンドトラック",
                  "Series Alpha: Haru to Kai no Bouken Original Soundtrack")
    head = ''.join(info(k, v) for k, v in [("Catalog Number", "SYNX-10001"), ("Barcode", "4900000000001"),
                                         ("Release Date", "Jun 19, 2024"), ("Label", "Synthetic Label")])
    credits = credit("Composer", "作曲", [("ソラノアオ", "Artist Y")])
    credits += credit("Flute", "フルート", [("作家畋", "Player A"), ("奏者乙", "Player B"), ("奏者丙", "Player C")])
    credits += credit("Performer", "演奏", [("作家界", "Ensemble A")])
    credits += credit("Alto flute", "アルトフルート", [("奏者乙", "Player B")])
    credits += credit("Conductor", "指揮", [("指揮者甲", "Conductor A")])
    lists = '<div id="tlnav"><a class="smallfont" rel="tl1" href="#">Japanese</a><a class="smallfont" rel="tl2" href="#">English</a></div>'
    for ident, language in [(1, "ja"), (2, "en")]:
        lists += f'<span class="tl" id="tl{ident}"><b>Disc 1</b><table>'
        for i in range(1, 43):
            name = ("序曲～Tv Anime Ver.1" if i == 1 else "名前のない灯り" if i == 3 else f"合成曲{i}") if language == "ja" else ("Prelude ~ TV Anime Ver. 1" if i == 1 else f"Synthetic Track {i}")
            length = "0:15" if i == 1 else "1:00"
            lists += f'<tr class="rolebit"><td>{i:02d}</td><td>{name}</td><td>{length}</td></tr>'
        lists += '</table></span>'
    notes = 'M-1, 2, 10, 20, 25 - Original Composer: Artist B<br>M-19 - Original Composer: Artist C<br>Lyricist: Artist Y (M-11)'
    return f'<html><h1>{title}</h1><table>{head}</table><div id="collapse_credits"><table>{credits}</table></div>{lists}<div id="notes">{notes}</div><a href="/web/20260617011718/">Snapshot</a></html>'


class Courtesy:
    def __init__(self):
        self.reports = []

    @contextlib.contextmanager
    def slot(self, url, purpose=None):
        yield

    def report(self, url, outcome=None):
        self.reports.append((url, outcome))


def opener_for(pages: dict):
    seen = []

    def opener(url):
        seen.append(url)
        if url.endswith("/robots.txt"):
            return "User-agent: *\nAllow: /\n"
        if url in pages:
            return pages[url]
        error = fetch.Unavailable(f"{url}: HTTP 404")
        error.status = 404
        raise error

    opener.seen = seen
    return opener


def load(tmp_path):
    body = _album_html()
    opener = opener_for({vgmdb.ARCHIVE_URL.format(id=900001): body})
    return vgmdb.album(900001, tmp_path, opener=opener, courtesy=Courtesy()), opener


def test_album_id_from_url():
    assert vgmdb.album_id_from_url("https://vgmdb.net/album/900001") == 900001
    assert vgmdb.album_id_from_url("https://web.archive.org/web/2025/https://vgmdb.net/album/42?x=1") == 42
    assert vgmdb.album_id_from_url("https://vgmdb.net/artist/26997") is None


def test_release_fields(tmp_path):
    release, opener = load(tmp_path)
    assert release.source == "vgmdb"
    assert release.url == "https://vgmdb.net/album/900001"
    assert release.title == "テレビアニメ「星空冒険団～ハルとカイの冒険～」オリジナル・サウンドトラック"
    assert release.titles["en"].startswith("Series Alpha: Haru to Kai no Bouken")
    assert release.catalog == "SYNX-10001"
    assert release.barcode == "4900000000001"
    assert release.release_date == "2024-06-19"
    assert release.label == "Synthetic Label"
    assert release.discs == 1
    assert release.raw_sha256 and release.fetched_at
    assert release.related_urls == ("https://web.archive.org/web/20260617011718/https://vgmdb.net/album/900001",)
    assert "Original Composer: Artist C" in release.notes
    assert any(url.startswith("https://web.archive.org/web/") for url in opener.seen)


def test_tracklist_in_japanese(tmp_path):
    release, _ = load(tmp_path)
    assert len(release.tracks) == 42
    first, third = release.tracks[0], release.tracks[2]
    assert (first.disc, first.number) == (1, 1)
    assert first.title == "序曲～Tv Anime Ver.1"
    assert first.titles["en"] == "Prelude ~ TV Anime Ver. 1"
    assert first.length_seconds == 15.0
    assert third.title == "名前のない灯り"
    assert all(track.m_number is None for track in release.tracks)


def test_album_credits(tmp_path):
    release, _ = load(tmp_path)
    by_role = {}
    for credit in release.credits:
        by_role.setdefault(credit.role_key, []).append(credit)
    composer = by_role["composer"][0]
    assert composer.role == "Composer / 作曲"
    assert composer.name == "ソラノアオ" and "Artist Y" in composer.names
    flute = [credit.name for credit in by_role["instrument:flute"]]
    assert "作家畋" in flute and len(flute) == 3
    assert by_role["performer"][0].name == "作家界"
    assert "instrument:alto flute" in by_role and "conductor" in by_role


def test_track_level_credits_from_notes(tmp_path):
    release, _ = load(tmp_path)
    tracks = {track.number: track for track in release.tracks}

    def credits(number, key):
        return [c.name for c in tracks[number].credits if c.role_key == key]

    for number in (1, 2, 10, 20, 25):
        assert credits(number, "original composer") == ["Artist B"]
    assert credits(19, "original composer") == ["Artist C"]
    assert credits(11, "lyricist") == ["ソラノアオ"]
    assert "Artist Y" in [c for c in tracks[11].credits if c.role_key == "lyricist"][0].names
    assert tracks[3].credits == ()
    assert credits(3, "original composer") == []


def test_unavailable_archive_is_none(tmp_path):
    opener = opener_for({})
    assert vgmdb.album(999999, tmp_path, opener=opener, courtesy=Courtesy()) is None
    # remembered as a miss: no second request
    count = len(opener.seen)
    assert vgmdb.album(999999, tmp_path, opener=opener, courtesy=Courtesy()) is None
    assert len(opener.seen) == count


def test_non_album_page_is_none(tmp_path):
    opener = opener_for({vgmdb.ARCHIVE_URL.format(id=5): "<html><title>Just a moment...</title></html>"})
    assert vgmdb.album(5, tmp_path, opener=opener, courtesy=Courtesy()) is None
    assert vgmdb.album("not-a-number", tmp_path, opener=opener, courtesy=Courtesy()) is None


def test_note_credit_forms():
    notes = "M-3, 5-6 - Arranger: Taro Yamada\nLyricist: A (M-1), B (M-2)\nComposer: Someone"
    found = vgmdb._note_credits(notes, {})
    assert [c.name for c in found[3]] == ["Taro Yamada"] and [c.name for c in found[6]] == ["Taro Yamada"]
    assert [c.name for c in found[1]] == ["A"] and [c.name for c in found[2]] == ["B"]
    assert 4 not in found
