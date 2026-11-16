"""Falsifiers for the series album on the Product line (video-metadata merge).

User-owned tags survive the projection; the input video is never touched; a re-run numbers identically; a
source that cannot be asked yields unknown fields, never guesses; distinct pieces never share a number or a
title; other episodes are written only at their CurrentBest public path. No test reaches the network: every
source read goes through ``fetch.http``, which these tests replace.
"""
import hashlib
import os
from pathlib import Path

import pytest
from mutagen.flac import FLAC

from lss.bgm import listening, scene_names
from lss.core import audio_identity
from lss.metadata import fetch, library as ledger, resolve, series_album
from lss.metadata.model import Release, Series, Track
from lss.metadata.sources import wikipedia
from lss.output import finalizer
from tests.test_bgm_collection_numbering import cue, library_for, published_as_current_best


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Any source read that a test does not answer itself is a failed read (never the network)."""
    def down(url, accept="*/*"):
        raise fetch.Unavailable(f"{url}: offline")
    monkeypatch.setattr(fetch, "http", down)
    series_album._CACHE.clear()
    yield
    series_album._CACHE.clear()


def _album():
    return resolve.series_album("作品", [resolve.merge_volume([Release(
        "musicbrainz", "u", "作品 オリジナルサウンドトラック Vol.1", tracks=(Track(1, 1, "主題", length_seconds=60),))])])


def _numbers(path):
    tags = FLAC(path)
    return tags["DISCNUMBER"][0], tags["TRACKNUMBER"][0]


def _episode(tmp_path, label, order, starts, source):
    return [cue(tmp_path, f"{label} {start}", start, source) for start in starts]


def test_lss_writes_the_whole_library_tag_set_over_listener_edits(tmp_path, monkeypatch):
    root = tmp_path / "state"
    ep3 = _episode(tmp_path, "ep3", 3, (20, 100), "c" * 64)
    pcm = {p: audio_identity.audio_identity(p)["pcm_sha256"] for p in ep3}
    first = FLAC(ep3[0])
    first.update(DATE="1999-01-01", COMMENT="利用者のメモ")
    first.save()
    listening.finish_outputs(ep3, root, library=library_for("第3話", 3, _album()))
    tags = FLAC(ep3[0])
    assert tags["DATE"] == FLAC(ep3[1])["DATE"] == ["2023-04-14"]   # LSS's value, not the listener's
    assert tags["COMMENT"] == ["利用者のメモ"]                        # a tag LSS never writes is untouched
    assert "user_owned" not in finalizer.receipt(ep3[0])
    assert all(audio_identity.audio_identity(p)["pcm_sha256"] == pcm[p] for p in ep3)


def test_the_input_video_is_never_touched(tmp_path):
    from lss import application

    video = tmp_path / "[X] 作品 第3話「サブ」.mkv"
    video.write_bytes(b"not really a video" * 64)
    before = (hashlib.sha256(video.read_bytes()).hexdigest(), video.stat().st_mtime_ns, video.stat().st_size)
    root = tmp_path / "state"
    library = application._bgm_library(video, root)
    assert library is not None and library["complete"] is False
    listening.finish_outputs(_episode(tmp_path, "ep3", 3, (20,), "c" * 64), root, library=library)
    assert (hashlib.sha256(video.read_bytes()).hexdigest(), video.stat().st_mtime_ns, video.stat().st_size) == before


def test_rerunning_the_same_series_gives_identical_numbering_and_records_one_revision(tmp_path, monkeypatch):
    root = tmp_path / "state"
    ep3 = _episode(tmp_path, "ep3", 3, (20, 100), "c" * 64)
    ep1 = _episode(tmp_path, "ep1", 1, (50,), "d" * 64)
    published_as_current_best(monkeypatch, ep3 + ep1)
    listening.finish_outputs(ep3, root, library=library_for("第3話", 3, _album()))
    listening.finish_outputs(ep1, root, library=library_for("第1話", 1, _album()))
    seen = {p: (_numbers(p), FLAC(p)["TITLE"]) for p in ep3 + ep1}
    revisions = ledger.read(root, "series-test")["numbering_revisions"]
    # Processing episode 1 after 3 moved episode 3's pieces: recorded once, as a revision, not silently.
    assert len(revisions) == 1 and {c["asset_id"] for c in revisions[0]["changes"]} == {
        f"bgm-cue:{FLAC(p)['LSS_BGM_CUE_KEY'][0]}" for p in ep3}
    for order in ((ep1, "第1話", 1), (ep3, "第3話", 3), (ep1, "第1話", 1)):
        listening.finish_outputs(order[0], root, library=library_for(order[1], order[2], _album()))
    assert {p: (_numbers(p), FLAC(p)["TITLE"]) for p in ep3 + ep1} == seen
    assert ledger.read(root, "series-test")["numbering_revisions"] == revisions


def test_distinct_pieces_never_share_a_number_or_a_title(tmp_path, monkeypatch):
    root = tmp_path / "state"
    # Same times in two episodes: the provisional time titles alone would collide in one series album.
    ep1 = _episode(tmp_path, "ep1", 1, (20, 100), "d" * 64)
    ep3 = _episode(tmp_path, "ep3", 3, (20, 100), "c" * 64)
    published_as_current_best(monkeypatch, ep1 + ep3)
    listening.finish_outputs(ep1, root, library=library_for("第1話", 1, _album()))
    listening.finish_outputs(ep3, root, library=library_for("第3話", 3, _album()))
    files = ep1 + ep3
    assert len({_numbers(p) for p in files}) == len(files)
    assert len({FLAC(p)["TITLE"][0] for p in files}) == len(files)
    assert FLAC(ep3[0])["TITLE"][0].startswith("第3話 00:20 ")


def test_one_recording_in_two_episodes_is_one_piece_even_when_grouped_later(tmp_path, monkeypatch):
    root = tmp_path / "state"
    ep1 = _episode(tmp_path, "ep1", 1, (20, 100), "d" * 64)
    ep3 = _episode(tmp_path, "ep3", 3, (40,), "c" * 64)
    published_as_current_best(monkeypatch, ep1 + ep3)
    listening.finish_outputs(ep1, root, library=library_for("第1話", 1, _album()))
    # Series Memory groups ep1's second cue with ep3's only after ep1 was numbered alone.
    same = [f"bgm-cue:{FLAC(p)['LSS_BGM_CUE_KEY'][0]}" for p in (ep1[1], ep3[0])]
    monkeypatch.setattr(listening, "_series_labels", lambda state: {asset: "劇伴#1" for asset in same})
    listening.finish_outputs(ep3, root, library=library_for("第3話", 3, _album()))
    assert _numbers(ep1[1]) == _numbers(ep3[0])                     # the refreshed ep1 file follows its group
    assert _numbers(ep1[0]) != _numbers(ep3[0])
    assert FLAC(ep1[1])["USE_EPISODE"] == ["第1話", "第3話"]


def test_other_episodes_are_written_only_at_their_current_best_path(tmp_path, monkeypatch):
    root = tmp_path / "state"
    ep3 = _episode(tmp_path, "ep3", 3, (20,), "c" * 64)
    listening.finish_outputs(ep3, root, library=library_for("第3話", 3, _album()))
    before = ep3[0].read_bytes()
    monkeypatch.setattr("lss.bgm.current_best.published_files", lambda root, sha: {})   # no CurrentBest record
    listening.finish_outputs(_episode(tmp_path, "ep1", 1, (50,), "d" * 64), root,
                             library=library_for("第1話", 1, _album()))
    assert ep3[0].read_bytes() == before    # a remembered path alone is never a write target


def _series():
    return Series(source="wikipedia", url="https://ja.wikipedia.org/wiki/作品", title="作品", kind="tv",
                  first_date="2023-04-14", music=("作曲家",), episode_listed=True, episode_dates=((1, "2023-04-14"),))


def test_a_source_that_cannot_be_asked_leaves_fields_unknown_and_records_nothing(tmp_path):
    summary = series_album.resolve_source(Path("[X] 作品 第3話「サブ」.mkv"), {}, tmp_path)
    assert summary["complete"] is False and summary["series"] is None
    assert summary["album"]["tracks"] == [] and summary["episode"]["date"] is None
    assert not (tmp_path / series_album.DIRECTORY).exists()     # neither a decision nor an alias is recorded
    files = _episode(tmp_path, "ep3", 3, (20,), "c" * 64)
    listening.finish_outputs(files, tmp_path / "state", library=summary)
    tags = FLAC(files[0])
    for unknown in ("DATE", "FIRST_USE_DATE", "COMPOSER", "SERIES_MUSIC", "M_NUMBER", "ARTIST"):
        assert unknown not in tags, unknown


def test_wikipedia_unreachable_is_not_no_series(tmp_path, monkeypatch):
    calls = []

    def flaky(url, accept="*/*"):
        calls.append(url)
        error = fetch.Unavailable(f"{url}: HTTP 503")
        error.status = 503
        raise error
    class Courtesy:   # no pacing or backoff waits in a test
        from contextlib import nullcontext

        def slot(self, url, purpose=None):
            return self.nullcontext()

        def report(self, url, reason=None):
            pass
    monkeypatch.setattr(fetch, "http", flaky)
    with pytest.raises(fetch.Unavailable):
        wikipedia.series("作品", "サブ", "tv", tmp_path, strict=True, courtesy=Courtesy())
    assert wikipedia.series("作品", "サブ", "tv", tmp_path, courtesy=Courtesy()) is None   # lenient reading unchanged
    assert calls


def test_offline_episodes_join_their_series_once_it_is_identified(tmp_path, monkeypatch):
    root = tmp_path / "state"
    offline = series_album.resolve_source(Path("[X] 作品 第3話「サブ」.mkv"), {}, root)
    ep3 = _episode(tmp_path, "ep3", 3, (20,), "c" * 64)
    listening.finish_outputs(ep3, root, library=offline)
    assert offline["provisional_key"] is None
    # Later the series is identified (online): its ledger adopts the offline episode, numbered with the rest.
    monkeypatch.setattr(wikipedia, "series", lambda *a, **k: _series())
    monkeypatch.setattr(series_album, "volumes", lambda *a, **k: [])
    online = series_album.resolve_source(Path("[X] 作品 第1話「サブ」.mkv"), {}, root)
    assert online["complete"] is True and online["provisional_key"] == offline["key"]
    published_as_current_best(monkeypatch, ep3)
    ep1 = _episode(tmp_path, "ep1", 1, (50,), "d" * 64)
    listening.finish_outputs(ep1, root, library=online)
    assert _numbers(ep1[0]) == ("1", "1") and _numbers(ep3[0]) == ("1", "2")
    assert FLAC(ep3[0])["DATE"] == ["2023-04-14"]
    revision = ledger.read(root, online["key"])["numbering_revisions"][-1]
    assert {"asset_id": f"bgm-cue:{FLAC(ep3[0])['LSS_BGM_CUE_KEY'][0]}", "from": "provisional",
            "to": {"disc": "1", "track": "2"}} in revision["changes"]
    # Offline again: the identified series' key and recorded facts are kept, the date is not invented anew.
    monkeypatch.setattr(wikipedia, "series", lambda *a, **k: (_ for _ in ()).throw(fetch.Unavailable("offline")))
    again = series_album.resolve_source(Path("[X] 作品 第3話「サブ」.mkv"), {}, root)
    assert again["complete"] is False and again["key"] == online["key"]
    assert again["series"]["first_date"] == "2023-04-14" and again["episode"]["date"] is None


def test_a_later_recording_never_takes_a_name_another_recording_of_the_work_has(tmp_path):
    root = tmp_path / "state"
    facts = {"work_key": "work", "work_title": "作品", "episode": 1, "cast": []}
    order = {"name": "探しもの", "use": "探す", "direction": "ゆったりと", "category": "日常・のどか",
             "evidence": "一緒に探そう"}
    for n, asset in enumerate(("bgm-cue:a", "bgm-cue:b")):
        scene_names.record_cue(root, asset, pcm_sha256=str(n) * 64, source_sha256="a" * 64, facts=facts,
                               start_seconds=30. + 60 * n, end_seconds=35. + 60 * n,
                               heard=[{"start": 31. + 60 * n, "end": 33. + 60 * n, "text": "一緒に探そう"}],
                               order=order, model={"id": "fixture"})
        scene_names.settle(root)
    names = {k: r["name"] for k, r in scene_names.read(root)["recordings"].items()}
    assert names == {"asset:bgm-cue:a": "探しもの", "asset:bgm-cue:b": "探しもの その2"}
    scene_names.settle(root)
    assert {k: r["name"] for k, r in scene_names.read(root)["recordings"].items()} == names   # fixed once given


def test_legacy_scene_backend_keeps_the_lanes_execution_profile(monkeypatch):
    from lss.bgm import scene_runtime

    monkeypatch.setattr(scene_runtime, "backend", lambda: "transformers")
    monkeypatch.setenv("LSS_SCENE_EXECUTION_PROFILE", "scene_paced60")
    assert scene_runtime.create_models().execution_profile == "scene_paced60"
    monkeypatch.delenv("LSS_SCENE_EXECUTION_PROFILE")
    assert scene_runtime.create_models().execution_profile == "legacy"
    assert os.environ.get("LSS_SCENE_EXECUTION_PROFILE") is None
