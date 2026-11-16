"""Episode playback order is separate from recording M numbers; LSS owns every tag it writes."""
import json

import numpy as np
from mutagen.flac import FLAC

from lss.bgm import cue_publication, current_best, listening, scene_job, scene_names
from lss.core import audio_identity
from lss.output import finalizer

RATE = 44100


def cue(root, name, start, source="a" * 64, *, known=True):
    frames = 5 * RATE
    t = np.arange(frames) / RATE
    audio = np.column_stack((.2 * np.sin(2 * np.pi * 220 * t), .1 * np.sin(2 * np.pi * 330 * t)))
    first = round(start * RATE)
    key = cue_publication.cue_key(source, "test", first, first + frames, RATE)
    path = root / f"{name}.flac"
    provenance = {"type": "bgm_published_cue", "source": {"sha256": source},
                  "start_frame": first, "end_frame": first + frames, "sample_rate": RATE}
    if known:
        provenance["start_seconds"] = float(start)
    cue_publication.publish(audio, RATE, path, key=key, title=name, album="Episode collection",
                            provenance=provenance, input_sha256=source, recipe_id="test", version="0.1.0")
    return path


def numbers(path):
    tags = FLAC(path)
    return tags.get("TRACKNUMBER"), tags.get("DISCNUMBER")


def test_reversed_inputs_number_each_source_independently_of_album_and_m_number(tmp_path):
    early = cue(tmp_path, "last filename", 10)
    late = cue(tmp_path, "first filename", 30)
    other = cue(tmp_path, "same album other episode", 5, "b" * 64)
    paths = [late, other, early]
    pcm = {path: audio_identity.audio_identity(path) for path in paths}
    listening.finish_outputs(paths, tmp_path / "state")
    assert numbers(early) == (["1"], ["1"])
    assert numbers(late) == (["2"], ["1"])
    assert numbers(other) == (["1"], ["1"])
    assert "M_NUMBER" not in FLAC(late)
    assert all(audio_identity.audio_identity(path) == pcm[path] for path in paths)
    assert listening.current(paths, tmp_path / "state", collection=True)


def test_missing_numbering_and_late_insertion_refresh_collection_without_scanning_again(tmp_path, monkeypatch):
    root = tmp_path / "state"
    early, late = cue(tmp_path, "early", 10), cue(tmp_path, "late", 30)
    for path in (early, late):
        listening.finish(path, root)
    assert listening.current([early, late], root)
    assert not listening.current([early, late], root, collection=True)
    listening.finish_outputs([late, early], root)
    middle = cue(tmp_path, "new middle", 20)
    listening.finish(middle, root)
    paths = [late, middle, early]
    pcm = {path: audio_identity.audio_identity(path) for path in paths}
    assert not listening.current(paths, root, collection=True)
    monkeypatch.setattr(finalizer, "scan_track_file", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PCM scan must be reused")))
    listening.finish_outputs(paths, root)
    assert [numbers(p) for p in (early, middle, late)] == [([str(n)], ["1"]) for n in (1, 2, 3)]
    before = {path: path.read_bytes() for path in paths}
    assert all(row["reused"] for row in listening.finish_outputs(tuple(reversed(paths)), root))
    assert all(path.read_bytes() == before[path] and audio_identity.audio_identity(path) == pcm[path] for path in paths)
    assert listening.current(paths, root, collection=True)


def test_foreign_edits_are_replaced_by_lss_numbers_title_and_album_on_reconciliation(tmp_path):
    root = tmp_path / "state"
    early, late = cue(tmp_path, "early", 10), cue(tmp_path, "late", 30)
    listening.finish_outputs([early, late], root)
    own_title = FLAC(late)["TITLE"]
    tags = FLAC(late)
    tags.update(TRACKNUMBER="99", DISCNUMBER="7", TITLE="他者の曲名", ALBUM="他者のアルバム")
    tags.save()
    assert not listening.current([late, early], root, collection=True)
    middle = cue(tmp_path, "middle", 20)
    listening.finish_outputs([late, middle, early], root)
    tags = FLAC(late)
    assert numbers(late) == (["3"], ["1"])
    assert tags["TITLE"] == own_title and tags["ALBUM"] == ["Episode collection"]
    assert "user_owned" not in finalizer.receipt(late)
    assert listening.current([late, early, middle], root, collection=True)


def test_unreceipted_existing_numbers_are_replaced_by_lss_playback_order(tmp_path):
    root = tmp_path / "state"
    late = cue(tmp_path, "late", 30)
    tags = FLAC(late)
    tags.update(TRACKNUMBER="9", DISCNUMBER="9")
    tags.save()
    listening.finish_outputs([late], root)
    assert numbers(late) == (["1"], ["1"])
    early = cue(tmp_path, "early", 10)
    listening.finish_outputs([late, early], root)
    assert numbers(late) == (["2"], ["1"])
    assert listening.current([early, late], root, collection=True)


def test_single_file_scene_refresh_keeps_collection_track_two_and_identical_pcm(tmp_path, monkeypatch):
    root = tmp_path / "state"
    early, late = cue(tmp_path, "early", 10), cue(tmp_path, "late", 30)
    listening.finish_outputs([late, early], root)
    before = audio_identity.audio_identity(late)
    tags = FLAC(late)
    asset = tags["LSS_NAMING_ASSET_ID"][0]
    scene_names.record_cue(root, asset, pcm_sha256=before["pcm_sha256"], source_sha256="a" * 64,
        facts={"work_key": "work", "work_title": "作品", "episode": 1, "cast": []},
        start_seconds=30., end_seconds=35., heard=[{"start": 31., "end": 33., "text": "一緒に探そう"}],
        order={"name": "探しもの", "use": "探す", "direction": "ゆったりと", "category": "日常・のどか", "evidence": "一緒に探そう"},
        model={"id": "fixture"})
    scene_names.settle(root)
    monkeypatch.setattr(current_best, "published_files", lambda *args: {
        asset.removeprefix("bgm-cue:"): {"public_path": str(late), "pcm_sha256": before["pcm_sha256"]}})
    assert scene_job.refresh_work(root, "work", None)["refreshed"] == 1
    assert FLAC(late)["TITLE"] == ["探しもの（仮題）"]
    assert numbers(late) == (["2"], ["1"])
    assert audio_identity.audio_identity(late) == before
    assert listening.current([late, early], root, collection=True)
    assert listening.finish(late, root)["reused"]


def test_unknown_source_position_does_not_invent_an_order_or_block_audio(tmp_path):
    a, b = cue(tmp_path, "known", 10), cue(tmp_path, "unknown", 30, known=False)
    rows = listening.finish_outputs([b, a], tmp_path / "state")
    assert len(rows) == 2
    assert numbers(a) == (None, None) and numbers(b) == (None, None)
    assert listening.current([a, b], tmp_path / "state", collection=True)


def library_for(episode, order, album):
    return {"key": "series-test", "work": "作品", "episode": {"label": episode, "order": order, "date": None},
            "series": {"first_date": "2023-04-14", "music": ["作曲家"]}, "album": album}


def published_as_current_best(monkeypatch, paths):
    """Register ``paths`` as their sources' CurrentBest files: only those are written by another source's run."""
    rows = {}
    for path in paths:
        tags = FLAC(path)
        rows.setdefault(tags["LSS_INPUT_SHA256"][0], {})[tags["LSS_BGM_CUE_KEY"][0]] = {
            "public_path": str(path), "pcm_sha256": audio_identity.audio_identity(path)["pcm_sha256"]}
    monkeypatch.setattr(current_best, "published_files", lambda root, sha: rows.get(sha, {}))


def test_series_album_joins_episodes_renumbers_by_broadcast_and_refreshes_other_episodes(tmp_path, monkeypatch):
    from lss.metadata import resolve as R
    from lss.metadata.model import Release, Track

    album = R.series_album("作品", [R.merge_volume([Release("musicbrainz", "u", "作品 オリジナルサウンドトラック Vol.1",
                                                           tracks=(Track(1, 1, "主題", length_seconds=60),))])])
    root = tmp_path / "state"
    ep3 = [cue(tmp_path, "ep3 a", 100, "c" * 64), cue(tmp_path, "ep3 b", 20, "c" * 64)]
    pcm = {p: audio_identity.audio_identity(p) for p in ep3}
    listening.finish_outputs(ep3, root, library=library_for("第3話", 3, album))
    tags = FLAC(ep3[1])
    assert tags["ALBUM"] == ["作品 オリジナル・サウンドトラック"] and tags["DATE"] == ["2023-04-14"]
    assert (tags["DISCNUMBER"], tags["TRACKNUMBER"]) == (["2"], ["1"]) and FLAC(ep3[0])["TRACKNUMBER"] == ["2"]
    assert tags["USE_EPISODE"] == ["第3話"] and tags["USE_START"] == ["00:20"] and tags["SERIES_MUSIC"] == ["作曲家"]
    assert "TOTALDISCS" not in tags and "ARTIST" not in tags and "RELATION" not in tags
    assert listening.current(ep3, root, collection=True, library=library_for("第3話", 3, album))
    edited = FLAC(ep3[0]); edited.update(SERIES_MUSIC="利用者の記入"); edited.save()
    # An earlier episode processed later takes the first numbers; episode 3's files are refreshed (tag-only).
    published_as_current_best(monkeypatch, ep3)
    ep1 = [cue(tmp_path, "ep1 a", 50, "d" * 64)]
    listening.finish_outputs(ep1, root, library=library_for("第1話", 1, album))
    assert FLAC(ep1[0])["TRACKNUMBER"] == ["1"]
    assert FLAC(ep3[1])["TRACKNUMBER"] == ["2"] and FLAC(ep3[0])["TRACKNUMBER"] == ["3"]
    assert FLAC(ep3[0])["SERIES_MUSIC"] == ["作曲家"]   # LSS rewrites its own credit tag
    assert all(audio_identity.audio_identity(p) == pcm[p] for p in ep3)
    # A single-file refresh without the library keeps the projection.
    listening.finish(ep3[1], root)
    assert FLAC(ep3[1])["USE_EPISODE"] == ["第3話"] and FLAC(ep3[1])["TRACKNUMBER"] == ["2"]
    assert listening.current(ep3, root, collection=True, library=library_for("第3話", 3, album))
