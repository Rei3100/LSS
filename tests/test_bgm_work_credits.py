"""Work composer credit from published staff lists: verbatim names only, never another series' composer."""
import contextlib
import urllib.parse
from pathlib import Path

from lss.bgm import collection_metadata, work_credits as K

W = K.WIKI


class Courtesy:
    @contextlib.contextmanager
    def slot(self, url, purpose=None):
        yield

    def report(self, url, state=None):
        pass


def site(pages):
    """opener over {title: html}; robots.txt allows everything; unknown pages are offline."""
    def opener(url):
        if url.endswith("/robots.txt"):
            return "User-agent: *\nAllow: /\n"
        title = urllib.parse.unquote(url[len(W):]) if url.startswith(W) else None
        if title not in pages:
            from lss.bgm.work_info import WorkInfoUnavailable
            raise WorkInfoUnavailable(url)
        return pages[title]
    return opener


def html(h1, lines, links=()):
    anchors = "".join(f'<a href="/wiki/{urllib.parse.quote(t)}">{t}</a>' for t in links)
    return f"<html><body><h1>{h1}</h1>{anchors}" + "".join(f"<p>{line}</p>" for line in lines) + "</body></html>"


def credits(tmp_path, pages, work, subtitle):
    return K.credits(work, subtitle, tmp_path, opener=site(pages), courtesy=Courtesy())


def test_staff_line_names_notes_and_changes():
    assert K.staff_music("スタッフ\n音楽 - 作家甶[30]\n撮影 - 某") == ["作家甶"]
    assert K.staff_music("音楽 - 作家畉[注 7]（1） → 作家甸、作家甹、作家町\n") == ["作家甸", "作家甹", "作家町"]
    assert K.staff_music("音楽 - Artist F[9][注 1]\n…\n音楽 - 別の人\n") == ["Artist F"]
    assert K.staff_music("音楽プロデューサー - 作家畊畍\n主題歌\n作曲 - Artist Q・Artist R") == []
    assert K.staff_music("") == []


def test_work_title_page_gives_composer(tmp_path):
    pages = {"仮面の勇者マイナ": html("仮面の勇者マイナ", ["スタッフ", "音楽 - 作家甶[30]"])}
    found = credits(tmp_path, pages, "仮面の勇者マイナ", "モダンウサギと走るカメ")
    assert found["composers"] == ["作家甶"] and found["matched"] == "work_title_page"


def test_franchise_page_uses_only_the_linked_series_listing_this_episode(tmp_path):
    pages = {
        "星空冒険団": html("星空冒険団", ["作曲家 一覧"], links=[
            "星空冒険団 (1997年のアニメ)", "星空冒険団 (2023年のアニメ)"]),
        "星空冒険団 (1997年のアニメ)": html("星空冒険団 (1997年のアニメ)", ["音楽 - 作家畎畏", "第1話 ホシナカマ！きみにきめた！"]),
        "星空冒険団 (2023年のアニメ)": html("星空冒険団 (2023年のアニメ)", ["音楽 - ソラノアオ[3]", "出発！星空の港へ！！"]),
    }
    found = credits(tmp_path, pages, "星空冒険団", "出発！星空の港へ！！")
    assert found["composers"] == ["ソラノアオ"] and "2023" in urllib.parse.unquote(found["url"])
    # An episode no linked page lists, on a franchise page with no staff line: unknown, not a guess.
    assert credits(tmp_path, pages, "星空冒険団", "どこにも無い題") is None


def test_wrong_title_disambiguation_offline_and_disabled_are_unknown(tmp_path, monkeypatch):
    assert credits(tmp_path, {"作品": html("別の作品", ["音楽 - 誰か"])}, "作品", None) is None
    assert credits(tmp_path / "b", {"作品": html("作品", ["作品 (曖昧さ回避)", "音楽 - 誰か"])}, "作品", None) is None
    assert credits(tmp_path / "c", {}, "作品", None) is None
    monkeypatch.setenv("LSS_WORK_INFO", "0")
    assert credits(tmp_path / "d", {"作品": html("作品", ["音楽 - 誰か"])}, "作品", None) is None


def test_composers_become_artist_list_with_source_and_channel_never_does():
    def credit(work, subtitle):
        assert (work, subtitle) == ("光の戦士テノ", "謎の少年")
        return {"composers": ["作家甸", "作家甹", "作家町"], "url": W + "x"}
    shown = collection_metadata.presentation(Path("x.mkv"), {
        "title": "『光の戦士テノ』第11話「謎の少年」-公式配信-", "ARTIST": "光の戦士公式"}, credit=credit)
    assert shown["tags"]["ARTIST"] == ["作家甸", "作家甹", "作家町"]
    assert '"artist":"' + W + 'x"' in shown["tags"][collection_metadata.PROVENANCE_TAG]
    unknown = collection_metadata.presentation(Path("x.mkv"), {"title": "『作品』第1話「一」", "ARTIST": "公式"},
                                               credit=lambda work, subtitle: None)
    assert "ARTIST" not in unknown["tags"]
