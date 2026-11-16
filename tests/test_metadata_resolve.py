"""Series album reconciliation: volume marks, middle dot, store suffixes, ranking, every credited role."""
import pytest

from lss.metadata import resolve as R
from lss.metadata.model import Credit, Release, Track


@pytest.mark.parametrize("title, series, expected", [
    ("星空冒険団 オリジナル・サウンドトラック", "星空冒険団",
     "星空冒険団 オリジナル・サウンドトラック"),
    ("仮面の勇者マイナ オリジナルサウンドトラック Vol.2", "仮面の勇者マイナ", "仮面の勇者マイナ オリジナル・サウンドトラック"),
    ("光の戦士テノ オリジナル サウンドトラック vol.1", "光の戦士テノ", "光の戦士テノ オリジナル・サウンドトラック"),
    ("TVアニメ「作品」オリジナル・サウンドトラック 第2弾", "作品", "TVアニメ「作品」オリジナル・サウンドトラック"),
    ("作品 オリジナル・サウンドトラック (Vol. 3)", "作品", "作品 オリジナル・サウンドトラック"),
    ("Vol.1", "作品", "作品 オリジナル・サウンドトラック"),
    ("オリジナルサウンドトラック vol.1", "作品", "作品 オリジナル・サウンドトラック"),
])
def test_album_title_removes_volume_marks_and_spells_the_middle_dot(title, series, expected):
    assert R.album_title(title, series) == expected


def test_store_suffix_only_when_the_store_marks_the_whole_album():
    titles = [f"曲{n} (Instrumental)" for n in range(41)] + ["曲名60"]
    assert R.store_suffix_pattern(titles)
    assert not R.store_suffix_pattern(["曲 (Instrumental)", "歌", "別の歌", "もう一つ"])
    assert R.strip_store_suffix("ソララインジングの合図 (Instrumental)") == "ソララインジングの合図"


def test_ranking_prefers_booklet_sources_over_store_typos_and_keeps_raw():
    choice = R.choose({"itunes": "ソララインジングの合図", "musicbrainz": "ソラライジングの合図"})
    assert choice.value == "ソラライジングの合図" and choice.source == "musicbrainz"
    assert choice.raw["itunes"] == "ソララインジングの合図"
    assert R.choose({"itunes": None, "vgmdb": ""}).value is None


def release(source, titles, lengths, credits=(), catalog=None, title="OST"):
    return Release(source=source, url=source, title=title, catalog=catalog,
                   tracks=tuple(Track(1, n + 1, t, length_seconds=l) for n, (t, l) in enumerate(zip(titles, lengths))),
                   credits=tuple(credits))


def test_merge_volume_strips_store_suffix_and_unions_credits():
    mb = release("musicbrainz", ["ソラライジングの合図", "主題"], [10.0, 20.0],
                 title="星空冒険団 オリジナル・サウンドトラック")
    it = release("itunes", ["ソララインジングの合図 (Instrumental)", "主題 (Instrumental)"], [10.2, 19.9],
                 credits=[Credit("artistName", "artist", "ソラノアオ")])
    vg = release("vgmdb", ["ソラライジングの合図", "主題"], [10.0, 20.0],
                 credits=[Credit("Composer / 作曲", "composer", "ソラノアオ"), Credit("Flute / フルート", "instrument:flute", "作家畋")])
    assert R.same_release(mb, it) and R.same_release(mb, vg)
    volume = R.merge_volume([it, vg, mb])
    first = volume["tracks"][0]
    assert first["title"].value == "ソラライジングの合図" and first["title"].raw["itunes"] == "ソララインジングの合図"
    assert volume["title"].source == "musicbrainz"
    tags, untagged = R.credit_tags(first["credits"])
    assert tags == {"COMPOSER": ["ソラノアオ"], "FLUTE": ["作家畋"]} and untagged == []


def test_series_album_continues_discs_across_volumes_and_reserves_the_unreleased_disc():
    v1 = R.merge_volume([release("musicbrainz", ["a", "b"], [1, 2], title="作品 オリジナルサウンドトラック Vol.1")])
    v2 = R.merge_volume([release("musicbrainz", ["c"], [3], title="作品 オリジナルサウンドトラック Vol.2")])
    album = R.series_album("作品", [v1, v2])
    assert album["title"] == "作品 オリジナル・サウンドトラック"
    assert [(t.disc, t.number, t.title.value) for t in album["tracks"]] == [(1, 1, "a"), (1, 2, "b"), (2, 1, "c")]
    assert album["official_discs"] == 2 and album["unreleased_disc"] == 3
    assert R.series_album("作品", [])["title"] == "作品 オリジナル・サウンドトラック"


def test_each_role_keeps_its_own_tag_and_release_identifiers_are_not_tags():
    credits = [Credit("Sound Effects / 効果", "sound effects", "某"), Credit("Conductor / 指揮", "conductor", "指揮者"),
               Credit("歌", "vocals", "歌手"), Credit("Original Composer", "original composer", "作家甴"),
               Credit("Alto Flute / アルトフルート", "instrument:alto flute", "作家畋"),
               Credit("B Clarinet / B管クラリネット", "instrument:b clarinet", "星ヒネリト"),
               Credit("Performer / 演奏", "performer", "ENSEMBLE Nebulafonica"), Credit("コーラス", "", "合唱団"),
               Credit("Label", "label", "SME"), Credit("Catalog", "catalog", "SYNX-10001"), Credit("振付", "", "先生")]
    tags, untagged = R.credit_tags(credits)
    assert tags == {"SOUND EFFECTS": ["某"], "CONDUCTOR": ["指揮者"], "VOCALS": ["歌手"], "ORIGINALCOMPOSER": ["作家甴"],
                    "ALTO FLUTE": ["作家畋"], "B CLARINET": ["星ヒネリト"], "PERFORMER": ["ENSEMBLE Nebulafonica"],
                    "CHORUS": ["合唱団"]}
    assert [c.name for c in untagged] == ["先生"]


def test_a_person_is_written_the_same_way_across_sources():
    mb = release("musicbrainz", ["a", "b"], [1, 2])
    mb = Release(**{**mb.__dict__, "tracks": (Track(1, 1, "a", length_seconds=1, credits=(
        Credit("composer", "composer", "作家甴", ("作家甴", "Cloud, Aster")),)), Track(1, 2, "b", length_seconds=2))})
    vg = release("vgmdb", ["a", "b"], [1, 2])
    vg = Release(**{**vg.__dict__, "tracks": (Track(1, 1, "a", length_seconds=1), Track(1, 2, "b", length_seconds=2, credits=(
        Credit("Original Composer", "original composer", "Aster Cloud"),
        Credit("Original Composer", "original composer", "Artist C"))))})
    volume = R.merge_volume([vg, mb])
    second = R.credit_tags(volume["tracks"][1]["credits"])[0]
    assert second["ORIGINALCOMPOSER"] == ["作家甴", "Artist C"]


def test_real_sources_unify_vgmdb_romanized_original_composers_with_musicbrainz():
    from pathlib import Path
    import contextlib
    from lss.metadata.sources import musicbrainz, vgmdb

    root = Path(__file__).parent / "fixtures" / "metadata"
    mb_body = (root / "musicbrainz" / "lookup_ae026de5.json").read_text(encoding="utf-8")
    vg_body = (root / "vgmdb" / "album_900001.html").read_text(encoding="utf-8")

    def opener(url):
        if url.endswith("robots.txt"):
            return "User-agent: *\nAllow: /\n"
        return mb_body if "musicbrainz.org" in url else vg_body

    class Courtesy:
        @contextlib.contextmanager
        def slot(self, url, purpose=None):
            yield

        def report(self, url, outcome=None):
            pass
    import tempfile
    with tempfile.TemporaryDirectory() as state:
        mb = musicbrainz.release("00000000-0000-4000-8000-000000000001", Path(state), opener=opener, courtesy=Courtesy())
        vg = vgmdb.album(900001, Path(state), opener=opener, courtesy=Courtesy())
    assert R.same_release(mb, vg)
    volume = R.merge_volume([vg, mb])
    tags = {row["number"]: R.credit_tags(row["credits"])[0] for row in volume["tracks"]}
    assert "作家甴" in tags[1]["ORIGINALCOMPOSER"] and "Artist B" not in tags[1]["ORIGINALCOMPOSER"]
    assert "COMPOSER" not in tags[1] and tags[1]["ARRANGER"] == ["ソラノアオ"]
    assert tags[10]["COMPOSER"] == ["作家甴"] and tags[5]["COMPOSER"] == ["ソラノアオ"]
    assert tags[19]["ORIGINALCOMPOSER"] == ["作家电"]
    assert tags[11]["LYRICIST"] == ["ソラノアオ"] and tags[5]["FLUTE"]
    assert all("ARTIST" not in t for t in tags.values())


def test_store_flattened_discs_align_by_running_order_and_lengths_and_never_set_structure():
    cd = Release("vgmdb", "v", "二枚組", tracks=(Track(1, 1, "一", length_seconds=60), Track(1, 2, "二", length_seconds=70),
                                                Track(2, 1, "三", length_seconds=80), Track(2, 2, "四", length_seconds=90)))
    store = Release("itunes", "i", "二枚組", tracks=(Track(1, 1, "一", length_seconds=60.4), Track(1, 2, "二", length_seconds=70),
                                                 Track(1, 3, "三 (誤記)", length_seconds=80.2), Track(1, 4, "四", length_seconds=99)))
    volume = R.merge_volume([store, cd])
    rows = {(t["disc"], t["number"]): t for t in volume["tracks"]}
    assert set(rows) == {(1, 1), (1, 2), (2, 1), (2, 2)} and volume["discs"] == 2
    assert rows[(2, 1)]["title"].raw == {"vgmdb": "三", "itunes": "三 (誤記)"}
    assert "itunes" not in rows[(2, 2)]["title"].raw          # lengths disagree: not the same track


def test_an_encoding_placeholder_loses_to_the_same_title_with_the_real_character():
    title = "ホシノウタ・ユメノサンポミ"
    # Preserve the original single unrepresentable separator -> '?' damage.
    damaged = title.replace("・", "・".encode("ascii", errors="replace").decode("ascii"))
    assert damaged == "ホシノウタ?ユメノサンポミ"
    choice = R.choose({"musicbrainz": damaged, "vgmdb": title})
    assert choice.value == "ホシノウタ・ユメノサンポミ" and choice.source == "vgmdb"
    # A real question mark that every source writes stays; a different title is not a repair.
    assert R.choose({"musicbrainz": "誰?", "vgmdb": "誰?"}).value == "誰?"
    assert R.choose({"musicbrainz": "誰?", "vgmdb": "別題"}).source == "musicbrainz"
