import json
from pathlib import Path

import pytest

from lss.bgm import collection_metadata as C


@pytest.mark.parametrize("title, work, episode, subtitle", [
    ("【公式】アニメ「星空冒険団」第1話「はじまりの灯り 前編」", "星空冒険団", "第1話", "はじまりの灯り 前編"),
    ("【公式】『仮面の勇者マイナ』　第1話「子(うさぎ)旅人と灯台守」", "仮面の勇者マイナ", "第1話", "子(うさぎ)旅人と灯台守"),
    ("見逃し配信【角醒ハンター オメガベル】 『第拾弍話　取り戻せ！アオVSシロ！』[公式]",
     "角醒ハンター オメガベル", "第拾弍話", "取り戻せ！アオVSシロ！"),
    ("『光の戦士テノ』第10話「合宿へ出発！」-公式配信-", "光の戦士テノ", "第10話", "合宿へ出発！"),
    ("【公式】アニメ「星空冒険団」第145話「驚愕！メガソラウサと神秘の地！？」（期間限定一挙配信）",
     "星空冒険団", "第145話", "驚愕！メガソラウサと神秘の地！？"),
    ("『作品』第3話", "作品", "第3話", None),
])
def test_episode_titles_from_real_distribution_forms(title, work, episode, subtitle):
    assert C.parse_episode_title(title) == {"work": work, "episode": episode, "subtitle": subtitle}


@pytest.mark.parametrize("title", ["", "第1話「タイトル」", "【公式】第1話「タイトル」", "ホシナカマ 名場面集", None])
def test_no_work_or_no_episode_is_not_invented(title):
    assert C.parse_episode_title(title) is None


def test_one_album_per_work_episode_as_disc_and_never_channel_or_album_artist():
    tags = {"title": "『光の戦士テノ』第11話「謎の少年」-公式配信-", "ARTIST": "光の戦士公式 series_e OFFICIAL",
            "DATE": "20260919", "COMMENT": "AbCdEfGhIjK"}
    result = C.presentation(Path("x.mkv"), tags)
    other = C.presentation(Path("y.mkv"), {"title": "『光の戦士テノ』第1話「新」-公式配信-"})
    assert result["album"] == other["album"] == "光の戦士テノ オリジナル・サウンドトラック"
    assert "DISCNUMBER" not in result["tags"] and "TRACKNUMBER" not in result["tags"]
    assert result["tags"]["DATE"] == "2026-09-19"
    assert not {"ARTIST", "ALBUMARTIST"} & set(result["tags"])
    assert json.loads(result["tags"][C.PROVENANCE_TAG]) == {
        "album": "container_title", "date": "container_date"}


@pytest.mark.parametrize("marker, number", [("第137話", 137), ("第１２話", 12), ("第壱話", 1), ("第拾弍話", 12),
                                            ("第九話", 9), ("第二十三話", 23), ("第百五話", 105), ("第一〇話", 10), ("第〇話", None)])
def test_episode_numbers_from_arabic_kanji_and_daiji(marker, number):
    assert C.episode_number(marker) == number


def test_file_name_is_used_when_container_has_no_title_and_unparsed_keeps_file_name_album():
    named = C.presentation(Path(('D' + chr(58) + '/v/『作品A』第2話「題」.mkv')), {})
    assert named["album"] == "作品A オリジナル・サウンドトラック" and "DISCNUMBER" not in named["tags"]
    assert "DATE" not in named["tags"]
    plain = C.presentation(Path(('D' + chr(58) + '/v/recording 2026.mkv')), {"date": "unknown"})
    assert plain["album"] == "recording 2026 オリジナル・サウンドトラック"
    assert set(plain["tags"]) == {C.PROVENANCE_TAG}


def test_invalid_dates_are_dropped():
    assert "DATE" not in C.presentation(Path("a.mkv"), {"DATE": "2026-13-40"})["tags"]
    assert C.presentation(Path("a.mkv"), {"creation_time": "2025-01-02T03:04:05.000000Z"})["tags"]["DATE"] == "2025-01-02"


def test_unreadable_source_keeps_file_name_presentation(tmp_path, monkeypatch):
    from lss.core import media_metadata

    source = tmp_path / "『作品B』第5話「五」.mkv"
    source.write_bytes(b"not media")

    def fail(_path, **_kw):
        raise RuntimeError("ffmpeg failed")
    monkeypatch.setattr(media_metadata, "format_tags", fail)
    C._read.cache_clear()
    assert C.for_source(source)["album"] == "作品B オリジナル・サウンドトラック"
    assert C.for_source(None) is None
