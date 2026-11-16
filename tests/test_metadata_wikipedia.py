"""Wikipedia series adapter from authored synthetic article fixtures: first date, 音楽 staff, episode listing, theme songs."""
import contextlib
import urllib.parse
from pathlib import Path

from lss.metadata import fetch
from lss.metadata.sources import wikipedia

PAGES = {"星空冒険団": "series_a", "星空冒険団 (2023年のアニメ)": "series_b",
         "仮面の勇者マイナ": "series_c", "角醒ハンター オメガベル": "series_d", "光の戦士テノ": "series_e"}


def _paragraphs(lines):
    return "".join(f"<p>{line}</p>" for line in lines)


def _schedule(rows, merged=False):
    body = '<table class="wikitable"><tr><th>話数</th><th>初放送日</th></tr>'
    for number, date in rows:
        if merged and number == 1:
            body += f'<tr><td>第1話</td><td rowspan="2">{date}</td></tr>'
        elif merged and number == 2:
            body += '<tr><td>第2話</td></tr>'
        else:
            body += f'<tr><td>{number if isinstance(number, str) else f"第{number}話"}</td><td>{date}</td></tr>'
    return body + '</table>'


def _article(title, start, music, songs, schedule, subtitle=""):
    return (f'<html><h1>{title}</h1><table class="infobox"><tr><th>放送期間</th><td>{start} -</td></tr></table>'
            + _paragraphs([subtitle, "スタッフ", "音楽 - " + music, "主題歌", *songs, "放送日程"])
            + schedule + '</html>')


def _pages():
    child = "星空冒険団 (2023年のアニメ)"
    links = f'<a href="{wikipedia.WIKI}{urllib.parse.quote(child)}">{child}</a>'
    franchise = f'<html><h1>星空冒険団</h1>{links}</html>'
    songs = ["オープニングテーマ「テーマ甲」", "Artist Mによる主題歌。作詞・作曲・編曲はArtist N。",
             "オープニングテーマ「THEME G」", "作曲 - Artist J / 編曲 - 作家畀",
             "エンディングテーマ「テーマ申」", "編曲 - Artist J",
             "オープニングテーマ「テーマ甴」", "作曲 - Artist K", "挿入歌「テーマ电」", "歌 - 歌手甲"]
    for i in range(5):
        songs += [f'エンディングテーマ「合成主題歌{i}」', '歌 - 歌手乙']
    milestones = {1: "2023年4月14日", 2: "2023年4月14日", 3: "4月21日", 6: "2024年4月12日",
                  12: "2025年4月11日", 20: "2026年5月22日", 21: "2026年8月28日",
                  22: "2026年9月4日", 23: "2026年9月4日", 24: "2026年9月18日"}
    schedule = _schedule([(i, milestones.get(i, f'2023年5月{min(i, 28)}日')) for i in range(1, 25)], merged=True)
    alpha = _article(child, "2023年4月14日", "ソラノアオ[1]", songs, schedule, "はじまりの灯り 前編")
    alpha += '<!-- {"Number":{"wt":"第23話"}} {"Aux6":{"wt":"9月4日"}} {"Aux6":{"wt":"9月4日"}} -->'
    beta = _article("仮面の勇者マイナ", "2026年9月6日", "作家甶", [
        '主題歌「Theme 83」', '作詞 - 作家畁 / 作曲 - Artist Q・Artist R / 編曲 - Artist Q / 歌 - Artist S'],
        '<table><tr><th>話数</th><th>配信日</th></tr><tr><td>第一話</td><td>2026年9月7日</td></tr></table>'
        + _schedule([("第一話", "2026年9月6日"), ("第二話", "9月13日")]), "旅人と灯台守")
    gamma = _article("角醒ハンター オメガベル", "2026年7月26日", "Artist F", [
        '主題歌「テーマ男」', '作詩 - ソラノアカ', '挿入歌「THEME E」',
        '作詞 - ソラノアカ / 作曲・編曲 - 作家畅 / 歌 - 作家畆'],
        _schedule([(i, "2026年7月26日" if i == 1 else "8月2日" if i == 2 else f'8月{min(2+i,28)}日') for i in range(1, 14)]))
    delta = _article("光の戦士テノ", "2026年7月4日", "旧作家[注 1]（1） → 作家甸、作家甹、作家町", [
        '主題歌「THEME STARS」', 'Artist Vによる主題歌。作詞は作家畈、作曲は作家甸と作家甹と作家町、編曲は作家畉。'],
        _schedule([(i, "2026年7月4日" if i == 1 else f'7月{min(4+i,28)}日') for i in range(1, 17)]
                  + [("総集編", "7月30日")]), "空から来た旅人")
    return {"series_a": franchise, "series_b": alpha, "series_c": beta, "series_d": gamma, "series_e": delta,
            "series_disambiguation": franchise.replace('</h1>', '</h1><a property="mw:PageProp/Category" href="曖昧さ回避">分類</a>'),
            "ended_series_b": alpha.replace('2023年4月14日 -</td>', '2023年4月14日 - 2026年10月9日</td>', 1)}


class Courtesy:
    def __init__(self):
        self.slots = []

    @contextlib.contextmanager
    def slot(self, url, purpose="request"):
        self.slots.append((url, purpose))
        yield

    def report(self, url, outcome=None):
        pass


def make_opener(requests=None, pages=PAGES):
    def opener(url):
        if url.endswith("/robots.txt"):
            return "User-agent: *\nAllow: /\n"
        if requests is not None:
            requests.append(url)
        name = pages.get(urllib.parse.unquote(url[len(wikipedia.WIKI):]))
        if name:
            return _pages()[name]
        error = fetch.Unavailable(url)
        error.status = 404
        raise error
    return opener


def run(tmp_path, title, subtitle, kind="tv"):
    return wikipedia.series(title, subtitle, kind, tmp_path, opener=make_opener(), courtesy=Courtesy())


def roles(song):
    return [(c.role_key, c.name) for c in song["credits"]]


def test_franchise_page_defers_to_the_series_page_that_lists_the_episode(tmp_path):
    series = run(tmp_path, "星空冒険団", "はじまりの灯り 前編")
    assert series.title == "星空冒険団 (2023年のアニメ)"
    assert series.url.endswith("%282023%E5%B9%B4%E3%81%AE%E3%82%A2%E3%83%8B%E3%83%A1%29")
    assert (series.kind, series.first_date, series.episode_listed) == ("tv", "2023-04-14", True)
    assert series.music == ("ソラノアオ",)
    assert series.raw_sha256 and series.fetched_at


def test_franchise_page_alone_is_not_a_series(tmp_path):
    assert run(tmp_path, "星空冒険団", None) is None
    assert run(tmp_path, "星空冒険団", "存在しないサブタイトル") is None


def test_series_alpha_theme_songs_carry_credits_from_prose(tmp_path):
    songs = {s["title"]: s for s in run(tmp_path, "星空冒険団", "はじまりの灯り 前編").songs}
    assert len(songs) >= 10
    first = songs["テーマ甲"]
    assert first["kind"] == "オープニングテーマ"
    assert roles(first) == [("vocals", "Artist M"), ("lyricist", "Artist N"), ("composer", "Artist N"),
                            ("arranger", "Artist N")]
    assert ("composer", "Artist J") in roles(songs["THEME G"]) and ("arranger", "作家畀") in roles(songs["THEME G"])
    assert ("arranger", "Artist J") in roles(songs["テーマ申"])
    assert ("composer", "Artist K") in roles(songs["テーマ甴"])
    assert songs["テーマ电"]["kind"] == "挿入歌"


def test_series_c_dates_music_and_opening_song(tmp_path):
    series = run(tmp_path, "仮面の勇者マイナ", "旅人と灯台守")
    assert (series.kind, series.first_date, series.episode_listed) == ("tv", "2026-09-06", True)
    assert series.music == ("作家甶",)
    assert len(series.songs) == 1
    song = series.songs[0]
    assert song["title"] == "Theme 83" and song["kind"] == "主題歌"
    assert roles(song) == [("lyricist", "作家畁"), ("composer", "Artist Q"), ("composer", "Artist R"),
                           ("arranger", "Artist Q"), ("vocals", "Artist S")]


def test_series_d_staff_credit_and_theme_and_insert_songs(tmp_path):
    series = run(tmp_path, "角醒ハンター オメガベル", None)
    assert series.first_date == "2026-07-26"
    assert series.music == ("Artist F",)
    assert series.episode_listed is False
    songs = {s["title"]: s for s in series.songs}
    assert set(songs) == {"テーマ男", "THEME E"}
    assert songs["THEME E"]["kind"] == "挿入歌"
    assert roles(songs["THEME E"]) == [("lyricist", "ソラノアカ"), ("composer", "作家畅"),
                                           ("arranger", "作家畅"), ("vocals", "作家畆")]
    assert ("lyricist", "ソラノアカ") in roles(songs["テーマ男"])   # written 作詩 on the page


def test_corpus_c_keeps_names_after_the_change_arrow_and_reads_prose_song_credits(tmp_path):
    series = run(tmp_path, "光の戦士テノ", "空から来た旅人")
    assert series.first_date == "2026-07-04" and series.episode_listed is True
    assert series.music == ("作家甸", "作家甹", "作家町")
    song = series.songs[0]
    assert song["title"] == "THEME STARS"
    assert roles(song) == [("vocals", "Artist V"), ("lyricist", "作家畈"), ("composer", "作家甸"),
                           ("composer", "作家甹"), ("composer", "作家町"), ("arranger", "作家畉")]


def test_kind_must_match_the_article_infobox(tmp_path):
    assert run(tmp_path, "仮面の勇者マイナ", None, "movie") is None
    assert run(tmp_path, "仮面の勇者マイナ", None, "tv").first_date == "2026-09-06"


def test_first_date_falls_back_to_the_first_schedule_row():
    assert wikipedia._schedule_first_date(["放送日程", "初放送日", "話数", "2026年", "07月04日", "第1話"]) == "2026-07-04"
    assert wikipedia._schedule_first_date(["放送日程", "2025年12月3日"]) == "2025-12-03"


def test_movie_release_date_is_read_from_the_infobox_release_row():
    box = '<table class="infobox"><tr><th>公開</th><td>2023年7月14日<sup>[1]</sup></td></tr></table>'
    assert wikipedia._infobox(box) == {"公開": "2023年7月14日"}
    assert wikipedia._iso("2023年7月14日 [注 1]") == "2023-07-14"
    assert wikipedia._iso("2023年7月") == "2023-07"


def test_unavailable_or_misnamed_article_is_none(tmp_path):
    assert wikipedia.series("存在しない作品", None, "tv", tmp_path, opener=make_opener(), courtesy=Courtesy()) is None
    assert wikipedia.series("  ", None, "tv", tmp_path, opener=make_opener(), courtesy=Courtesy()) is None
    # an article whose own title differs from the requested work is not accepted
    pages = {"別の作品": "series_c"}
    assert wikipedia.series("別の作品", None, "tv", tmp_path, opener=make_opener(pages=pages), courtesy=Courtesy()) is None


def test_pages_are_read_once_and_paced(tmp_path):
    requests, courtesy = [], Courtesy()
    for _ in range(2):
        wikipedia.series("仮面の勇者マイナ", None, "tv", tmp_path, opener=make_opener(requests), courtesy=courtesy)
    assert len(requests) == 1
    assert courtesy.slots and courtesy.slots[0][1] == "wikipedia"


def test_episode_dates_series_b_follow_the_page_schedule(tmp_path):
    series = run(tmp_path, "星空冒険団", "はじまりの灯り 前編")
    dates = dict(series.episode_dates)
    assert len(dates) == 24
    assert dates[1] == dates[2] == "2023-04-14"      # merged (rowspan) date cell covers two episodes
    assert dates[3] == "2023-04-21"                   # year-less "4月21日" continues 2023
    assert dates[6] == "2024-04-12" and dates[12] == "2025-04-11"
    assert dates[20] == "2026-05-22"
    assert 999 not in dates


def test_episode_dates_other_works(tmp_path):
    series_c = dict(run(tmp_path, "仮面の勇者マイナ", "旅人と灯台守").episode_dates)
    assert series_c[1] == "2026-09-06" and series_c[2] == "2026-09-13"        # kanji numbers; the 配信日 table is ignored
    corpus_c = dict(run(tmp_path, "光の戦士テノ", "空から来た旅人").episode_dates)
    assert corpus_c[1] == "2026-07-04" and len(corpus_c) == 16                  # 総集編 rows carry no number: skipped
    omega = dict(run(tmp_path, "角醒ハンター オメガベル", None).episode_dates)
    assert omega[1] == "2026-07-26" and omega[2] == "2026-08-02" and len(omega) == 13


def test_year_less_dates_roll_over_to_the_next_year():
    html = ('<table class="wikitable"><tr><th>話数</th><th>初放送日</th></tr>'
            '<tr><td>第1話</td><td>2025年12月20日</td></tr><tr><td>第2話</td><td>12月27日</td></tr>'
            '<tr><td>第3話</td><td>1月3日</td></tr><tr><td>特別編</td><td>1月4日</td></tr></table>')
    assert wikipedia._episode_dates(html) == ((1, "2025-12-20"), (2, "2025-12-27"), (3, "2026-01-03"))
    assert wikipedia._episode_dates("<table><tr><td>x</td></tr></table>") == ()


# --- ongoing continuation (episode newer than the article's last listed one) ---------------------------------

DISAMBIGUATION = {**PAGES, "星空冒険団": "series_disambiguation"}


def run_disambiguation(tmp_path, subtitle, number, pages=DISAMBIGUATION):
    return wikipedia.series("星空冒険団", subtitle, "tv", tmp_path, opener=make_opener(pages=pages),
                            courtesy=Courtesy(), episode_number=number)


def test_listed_subtitle_records_the_subtitle_rule(tmp_path):
    assert run(tmp_path, "星空冒険団", "はじまりの灯り 前編").match == "subtitle_listed"


def test_ongoing_article_that_has_not_caught_up_is_accepted_by_episode_number(tmp_path):
    series = run_disambiguation(tmp_path, "まだ載っていない話", 25)
    assert series is not None and series.match == "ongoing_continuation"
    assert series.title == "星空冒険団 (2023年のアニメ)"
    assert series.episode_listed is False
    assert dict(series.episode_dates).get(25) is None   # nothing inferred for the unlisted episode
    assert run_disambiguation(tmp_path, "まだ載っていない話", 32).match == "ongoing_continuation"   # N + 8


def test_ongoing_continuation_window_is_bounded(tmp_path):
    assert run_disambiguation(tmp_path, "まだ載っていない話", 33) is None   # N + 9
    assert run_disambiguation(tmp_path, "まだ載っていない話", 24) is None   # not newer than the article
    assert run_disambiguation(tmp_path, "まだ載っていない話", None) is None


def test_ended_series_never_continues(tmp_path):
    pages = {**DISAMBIGUATION, "星空冒険団 (2023年のアニメ)": "ended_series_b"}
    assert run_disambiguation(tmp_path, "まだ載っていない話", 25, pages) is None


def test_disambiguation_page_is_a_link_source_not_a_series(tmp_path):
    assert run_disambiguation(tmp_path, None, None) is None
    series = run_disambiguation(tmp_path, "はじまりの灯り 前編", 1)
    assert series.match == "subtitle_listed"


def test_page_states_the_same_date_for_two_adjacent_episodes():
    html = _pages()["series_b"]
    dates = dict(wikipedia._episode_dates(html))
    assert (dates[21], dates[22], dates[23], dates[24]) == ("2026-08-28", "2026-09-04", "2026-09-04", "2026-09-18")
    assert '"Number":{"wt":"第23話"}' in html and html.count('"Aux6":{"wt":"9月4日"}') == 2

