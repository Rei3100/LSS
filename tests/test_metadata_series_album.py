"""series_album.resolve_source fills the episode's first broadcast date from the series article."""
from pathlib import Path

from lss.metadata import resolve, series_album
from lss.metadata.model import Series
from lss.metadata.sources import wikipedia


def _series(dates):
    return Series(source="wikipedia", url="https://ja.wikipedia.org/wiki/X", title="X", kind="tv",
                  first_date="2026-07-04", music=(), episode_listed=True, episode_dates=dates)


def _resolve(tmp_path, monkeypatch, series, name="[SyntheticGroup] 光の戦士テノ 第2話「星あかりの観測小屋へ」.mkv"):
    monkeypatch.setattr(wikipedia, "series", lambda *a, **k: series)
    monkeypatch.setattr(series_album, "volumes", lambda *a, **k: [])
    monkeypatch.setattr(resolve, "series_album", lambda work, vols: {
        "title": work, "title_choice": resolve.Choice(None, None, {}), "official_discs": 0, "unreleased_disc": False,
        "tracks": []})
    return series_album.resolve_source(Path(name), {}, tmp_path)


def test_episode_date_comes_from_the_series_schedule(tmp_path, monkeypatch):
    summary = _resolve(tmp_path, monkeypatch, _series(((1, "2026-07-04"), (2, "2026-07-11"))))
    assert summary["episode"]["order"] == 2
    assert summary["episode"]["date"] == "2026-07-11"


def test_episode_date_is_none_when_unlisted_or_no_series(tmp_path, monkeypatch):
    assert _resolve(tmp_path, monkeypatch, _series(((1, "2026-07-04"),)))["episode"]["date"] is None
    assert _resolve(tmp_path, monkeypatch, None)["episode"]["date"] is None


# --- transient failure is not "no soundtrack" ------------------------------------------------------------------

import pytest  # noqa: E402

from lss.metadata import fetch  # noqa: E402
from lss.metadata.model import Credit  # noqa: E402


def _volume():
    return {"title": resolve.Choice("ハルとカイの冒険 オリジナル・サウンドトラック", "musicbrainz",
                                    {"musicbrainz": "ハルとカイの冒険 オリジナル・サウンドトラック", "itunes": "Series Alpha OST"}),
            "discs": 2,
            "tracks": [{"disc": 2, "number": 3, "title": resolve.Choice("ブルー", "vgmdb", {"vgmdb": "ブルー", "itunes": "ブル-"}),
                        "length_seconds": 61.5, "m_number": resolve.Choice("M12", "vgmdb", {"vgmdb": "M12"}),
                        "credits": (Credit(role="作曲", role_key="composer", name="ソラノアオ", names=("ソラノアオ", "Neburin")),)}]}


def _name():
    return "[X] 星空冒険団 第137話「サブ」.mkv"


def test_transient_failure_reuses_last_complete_decision_and_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(wikipedia, "series", lambda *a, **k: _series(()))
    monkeypatch.setattr(series_album, "volumes", lambda *a, **k: [_volume()])
    first = series_album.resolve_source(Path(_name()), {}, tmp_path)
    assert first["complete"] is True and len(first["album"]["tracks"]) == 1

    def down(*a, **k):
        raise fetch.Unavailable("HTTP 503")
    monkeypatch.setattr(series_album, "volumes", down)
    again = series_album.resolve_source(Path(_name()), {}, tmp_path)
    assert again["complete"] is False
    assert again["key"] == first["key"]
    assert again["album"] == first["album"]   # rebuilt exactly (Choices, credits, titles, discs)


def test_transient_failure_without_prior_record_is_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(wikipedia, "series", lambda *a, **k: _series(()))

    def down(*a, **k):
        raise fetch.Unavailable("offline")
    monkeypatch.setattr(series_album, "volumes", down)
    summary = series_album.resolve_source(Path(_name()), {}, tmp_path)
    assert summary["complete"] is False and summary["album"]["tracks"] == []
    assert not list((tmp_path / "metadata-library").glob("*/album.json"))   # a partial is never recorded


def test_episode_number_is_passed_to_the_series_lookup(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(wikipedia, "series", lambda *a, **k: seen.update(k) or None)
    monkeypatch.setattr(series_album, "volumes", lambda *a, **k: [])
    series_album.resolve_source(Path(_name()), {}, tmp_path)
    assert seen["episode_number"] == 137


def test_incomplete_results_are_not_cached(tmp_path, monkeypatch):
    video = tmp_path / "a.mkv"
    video.write_bytes(b"x")
    series_album._CACHE.clear()
    calls = []
    results = [{"complete": False}, {"complete": True}, {"complete": True}]

    def fake(*a, **k):
        calls.append(1)
        return results[len(calls) - 1]
    monkeypatch.setattr(series_album, "resolve_source", fake)
    assert series_album.for_source(video, tmp_path)["complete"] is False
    assert series_album.for_source(video, tmp_path)["complete"] is True   # asked again
    assert series_album.for_source(video, tmp_path)["complete"] is True   # now served from the cache
    assert len(calls) == 2
    series_album._CACHE.clear()
