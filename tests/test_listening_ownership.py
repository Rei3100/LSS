"""LSS owns every tag, ReplayGain value and cover of the files it creates; files it did not create stay untouched.

A player- or tag-editor-edited value is not a second owner. A reuse run rewrites LSS's complete tag set from
Product truth (naming record, measurement) and restores whatever differs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC, Picture

from lss.bgm import cue_publication, listening
from lss.core import audio_identity
from lss.output import finalizer, result_report, results


def _cue(tmp_path, name="source - BGM 00m00s.flac", title="source BGM 00m00s", **extra):
    rate, frames = 44100, 44100 * 7
    t = np.arange(frames) / rate
    signal = np.column_stack((0.2 * np.sin(2 * np.pi * 220 * t), 0.1 * np.sin(2 * np.pi * 420 * t)))
    key = cue_publication.cue_key("a" * 64, "accepted-recipe", 0, frames, rate)
    path = tmp_path / name
    cue_publication.publish(signal, rate, path, key=key, title=title, album="source BGM",
                            input_sha256="a" * 64, recipe_id="accepted-recipe", version="0.1.0",
                            provenance={"type": "bgm_published_cue", "source": {"sha256": "a" * 64},
                                        "start_frame": 0, "end_frame": frames, "sample_rate": rate, **extra})
    return path


def _accepted(path, *, title="source BGM 00m00s"):
    tags = FLAC(path)
    return {"key": tags["LSS_BGM_CUE_KEY"][0], "provenance": json.loads(tags["LSS_BGM_PROVENANCE"][0]),
            "audio": audio_identity.audio_identity(path), "input_sha256": tags["LSS_INPUT_SHA256"][0],
            "recipe_id": tags["LSS_RECIPE_ID"][0], "headroom_shift": 0, "version": "0.1.0", "album": "source BGM"}


def _picture(desc, kind=3):
    picture = Picture()
    picture.type, picture.mime, picture.desc = kind, "image/png", desc
    picture.data = b"\x89PNG\r\n\x1a\n" + desc.encode() * 8
    return picture


EDITED = ("TITLE", "ARTIST", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_PEAK", "REPLAYGAIN_ALBUM_GAIN",
          "REPLAYGAIN_ALBUM_PEAK", "PLAY_COUNT")


def _player_edits(path):
    """What foobar2000-style tagging does: rename, add artist, scan ReplayGain (track+album), add art, play."""
    audio = FLAC(path)
    audio["TITLE"] = "利用者が付けた曲名"
    audio["ARTIST"] = "利用者が入力した作曲者"
    audio["REPLAYGAIN_TRACK_GAIN"] = "-3.10 dB"
    audio["REPLAYGAIN_TRACK_PEAK"] = "0.500000"
    audio["REPLAYGAIN_ALBUM_GAIN"] = "-2.00 dB"
    audio["REPLAYGAIN_ALBUM_PEAK"] = "0.900000"
    audio["PLAY_COUNT"] = "3"
    audio.add_picture(_picture("front"))
    audio.add_picture(_picture("back", 4))
    audio.save()
    written = FLAC(path)
    return {key: written.get(key) for key in EDITED}


def test_foreign_edits_are_replaced_by_lss_tags_and_unwritten_tags_are_left_alone(tmp_path):
    path = _cue(tmp_path)
    root = tmp_path / "state"
    accepted = _accepted(path)
    first = listening.finish(path, root, accepted=accepted)
    own = {key: FLAC(path).get(key) for key in ("TITLE", "ARTIST", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_PEAK")}
    assert "仮題" in own["TITLE"][0]
    pcm = first["audio"]["pcm_sha256"]
    _player_edits(path)
    tampered = FLAC(path)
    tampered["LSS_BGM_CUE_KEY"] = "edited-key"
    tampered.save()
    assert listening.current([path], root) is False

    repaired = listening.finish(path, root, accepted=accepted)

    audio = FLAC(path)
    for key, value in own.items():
        assert audio.get(key) == value, key
    assert audio.pictures == []
    assert audio["PLAY_COUNT"] == ["3"]  # a tag LSS never writes is not LSS's to remove
    assert audio["LSS_BGM_CUE_KEY"] == [accepted["key"]]
    assert audio_identity.audio_identity(path)["pcm_sha256"] == pcm
    record = finalizer.receipt(path)
    assert record is not None and "user_owned" not in record and "artwork_owner" not in record
    assert "user_owned" not in repaired and repaired["title"] == own["TITLE"][0]
    before = path.read_bytes()
    assert listening.finish(path, root, accepted=accepted)["reused"]
    assert path.read_bytes() == before
    assert len(list(tmp_path.glob("*.flac"))) == 1


def test_official_name_replaces_the_lss_title(tmp_path):
    path = _cue(tmp_path)
    listening.finish(path, path.parent / "state")
    asset = FLAC(path)["LSS_NAMING_ASSET_ID"][0]
    official = {"official": {"asset_id": asset, "title": "確認済みの公式名", "verified": True,
                             "relation": "exact_version", "evidence_ref": "test:verified"}}
    assert listening.finish(path, path.parent / "state", extra_evidence=official)["naming"]["kind"] == "official"
    assert FLAC(path)["TITLE"] == ["確認済みの公式名"]
    assert FLAC(path)["LSS_NAMING_OFFICIAL"] == ["1"]


def test_a_deleted_lss_value_is_written_again(tmp_path):
    path = _cue(tmp_path)
    listening.finish(path, tmp_path / "state")
    title = FLAC(path)["TITLE"]
    audio = FLAC(path)
    del audio["TITLE"]
    audio.save()
    listening.finish(path, tmp_path / "state")
    assert FLAC(path)["TITLE"] == title


@pytest.mark.parametrize("edited", [False, True])
def test_legacy_publication_without_receipt_gets_lss_title_whatever_the_file_held(tmp_path, edited):
    path = _cue(tmp_path)
    audio = FLAC(path)
    audio.pop(finalizer.RECEIPT_TAG, None)  # r4の公開は受領票を持たない
    if edited:
        audio["TITLE"] = "別のソフトが付けた名前"
    audio.save()
    listening.finish(path, tmp_path / "state", accepted=_accepted(path))
    assert "仮題" in FLAC(path)["TITLE"][0]


def test_instrumental_legacy_output_gets_lss_replaygain_and_keeps_its_source_cover(tmp_path):
    path = tmp_path / "song - Instrumental.flac"
    t = np.arange(44100 * 7) / 44100
    sf.write(path, np.column_stack((0.3 * np.sin(2 * np.pi * 330 * t),) * 2), 44100, subtype="PCM_24")
    audio = FLAC(path)
    audio.update({"TITLE": "song", "ARTIST": "artist", "LSS_STEM": "Instrumental", "LSS_SOURCE_TYPE": "separated",
                  "LSS_RECIPE_ID": "separation-v1-x", "LSS_INPUT_SHA256": "c" * 64,
                  "REPLAYGAIN_TRACK_GAIN": "-3.10 dB", "REPLAYGAIN_TRACK_PEAK": "0.500000"})
    audio.add_picture(_picture("source cover"))
    audio.add_picture(_picture("added later", 4))
    audio.save()
    from lss.core import worker
    tags, picture = worker._listening_request(path)
    finalizer.ensure(path, tags=tags, picture=picture)
    audio = FLAC(path)
    assert audio["REPLAYGAIN_TRACK_GAIN"] != ["-3.10 dB"] and audio["REPLAYGAIN_TRACK_PEAK"] != ["0.500000"]
    assert [p.desc for p in audio.pictures] == ["source cover"]
    assert audio["TITLE"] == ["song"] and audio["ARTIST"] == ["artist"]
    before = path.read_bytes()
    assert finalizer.ensure(path, tags=tags, picture=picture)["reused"]
    assert path.read_bytes() == before


def test_receipt_backed_lss_replaygain_can_follow_a_new_listening_policy(tmp_path, monkeypatch):
    path = tmp_path / "song - Instrumental.flac"
    t = np.arange(44100 * 7) / 44100
    sf.write(path, np.column_stack((0.1 * np.sin(2 * np.pi * 330 * t),) * 2), 44100, subtype="PCM_24")
    tags = {"TITLE": "song", "LSS_STEM": "Instrumental", "LSS_SOURCE_TYPE": "separated"}
    first = finalizer.ensure(path, tags=tags)
    gain = FLAC(path)["REPLAYGAIN_TRACK_GAIN"]
    monkeypatch.setattr(finalizer, "TARGET_LUFS", -19.0)
    monkeypatch.setattr(finalizer, "POLICY", {**finalizer.POLICY, "target_lufs": -19.0})
    second = finalizer.ensure(path, tags=tags)
    assert FLAC(path)["REPLAYGAIN_TRACK_GAIN"] != gain
    assert second["audio"] == first["audio"]


def test_instrumental_repair_takes_provenance_and_presentation_from_the_previous_receipt_not_from_edits(tmp_path):
    path = tmp_path / "song - Instrumental.flac"
    t = np.arange(44100 * 7) / 44100
    sf.write(path, np.column_stack((0.3 * np.sin(2 * np.pi * 330 * t),) * 2), 44100, subtype="PCM_24")
    finalizer.ensure(path, tags={"TITLE": "song", "VOCAL_TYPE": "ON VOCAL", "LSS_MODEL": "model-a",
                                 "LSS_RECIPE_ID": "separation-v1-x"})
    audio = FLAC(path)
    audio["LSS_MODEL"] = "edited-model"
    audio["VOCAL_TYPE"] = "他者が足した値"
    audio["TITLE"] = "他者の曲名"
    audio.save()
    from lss.core import worker
    tags, picture = worker._listening_request(path)
    assert tags["LSS_MODEL"] == ["model-a"] and tags["VOCAL_TYPE"] == ["ON VOCAL"] and tags["TITLE"] == ["song"]
    finalizer.ensure(path, tags=tags, picture=picture)
    audio = FLAC(path)
    assert audio["LSS_MODEL"] == ["model-a"] and audio["LSS_SOURCE_VOCAL_TYPE"] == ["ON VOCAL"]
    assert "VOCAL_TYPE" not in audio and audio["TITLE"] == ["song"]


def test_a_replacement_file_is_written_with_lss_full_current_tag_set(tmp_path):
    old, new = _cue(tmp_path / "old"), _cue(tmp_path / "new", title="replacing cue")
    listening.finish(old, tmp_path / "old" / "state")
    foreign = FLAC(old)
    foreign["COMMENT"] = "note"
    foreign.add_picture(_picture("front"))
    foreign.save()
    result = listening.finish(new, tmp_path / "new" / "state")
    tags, record = FLAC(new), finalizer.receipt(new)
    assert record is not None and result["reused"] is False
    for key, value in record["projected_tags"].items():
        assert tags[key] == value, key
    for key in ("TITLE", "ALBUM", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_PEAK", "DYNAMIC RANGE",
                "LSS_NAMING_REVISION", "LSS_NAMING_ASSET_ID", "LSS_BGM_CUE_KEY", "LSS_LISTENING_POLICY"):
        assert tags.get(key), key
    assert "COMMENT" not in tags and tags.pictures == []


def test_an_old_record_with_user_owned_fields_is_readable_and_is_converged_not_crashed_on(tmp_path):
    path = _cue(tmp_path)
    listening.finish(path, tmp_path / "state")
    audio = FLAC(path)
    record = json.loads(audio[finalizer.RECEIPT_TAG][0])
    del record["projected_tags"]["TITLE"]
    record.update(user_owned=["TITLE"], artwork_owner="user", artwork=None)
    audio[finalizer.RECEIPT_TAG] = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    audio["TITLE"] = "古い版で利用者の物とされた名前"
    audio.add_picture(_picture("old user cover"))
    audio.save()
    assert finalizer.previous_record(path)["user_owned"] == ["TITLE"]   # reading never crashes
    assert finalizer.receipt(path) is None                              # but it is no longer current
    with pytest.raises(finalizer.FinalizationError):
        listening.describe([path])
    assert listening.current([path], tmp_path / "state") is False
    listening.finish(path, tmp_path / "state")
    audio = FLAC(path)
    assert "仮題" in audio["TITLE"][0] and audio.pictures == []
    fresh = finalizer.receipt(path)
    assert fresh is not None and "user_owned" not in fresh
    # Result projection tolerates old listening rows that carry the removed fields.
    row = {"path": str(path), "audio": fresh["audio"], "finalization": {**fresh, "user_owned": ["TITLE"]},
           "user_owned": ["TITLE"], "title": "x"}
    asset = results.project(disposition="PUBLISHED", outputs=[str(path)], listening=[row])["assets"][0]
    assert "title_owner" not in asset and "user_owned_tags" not in asset


def test_metadata_only_repair_reuses_the_measurement_of_the_same_pcm(tmp_path, monkeypatch):
    path = _cue(tmp_path)
    first = listening.finish(path, tmp_path / "state")
    album = FLAC(path)["ALBUM"]
    audio = FLAC(path)
    audio["LSS_BGM_CUE_KEY"] = "edited"
    audio["ALBUM"] = "player album"
    audio.save()
    monkeypatch.setattr(finalizer, "scan_track_file", lambda *a, **k: pytest.fail("同じPCMを再走査しない"))
    second = listening.finish(path, tmp_path / "state")
    assert not second["reused"]
    assert second["finalization"]["measurement"] == first["finalization"]["measurement"]
    assert FLAC(path)["ALBUM"] == album


def test_rebuild_is_projected_distinctly_from_reuse():
    projection = results.project(disposition="PUBLISHED", outputs=["a.flac", "b.flac"], reused=True, rebuilt=["b.flac"])
    assert projection["state"] == "REBUILT" and projection["rebuilt_count"] == 1
    assert results.project(disposition="PUBLISHED", outputs=["a.flac"], reused=True)["state"] == "REUSED"
    assert "作り直しました" in result_report.LABELS["REBUILT"]


@pytest.mark.parametrize("disposition", ["ZERO", "HELD", "UNSUPPORTED", "FAILED"])
def test_no_result_file_or_folder_without_saved_audio(tmp_path, disposition):
    projection = results.project(disposition=disposition, outputs=[])
    saved = result_report.save(tmp_path / "source.wav", "inst", projection)
    assert "report_path" not in saved and "report_error" not in saved
    assert not (tmp_path / "LSS").exists() and list(tmp_path.iterdir()) == []


def test_provisional_title_carries_the_episode_time_and_no_identity_token(tmp_path):
    path = _cue(tmp_path, start_seconds=3723.4)  # 62分03秒
    first = listening.finish(path, tmp_path / "state")
    title = FLAC(path)["TITLE"][0]
    assert title.startswith("62:03 ") and title.endswith("（仮題）")
    assert first["naming"]["asset_id"][-10:] not in title and "仮題・" not in title
    official = {"official": {"asset_id": first["naming"]["asset_id"], "title": "公式曲名", "verified": True,
                             "relation": "exact_version", "evidence_ref": "test:verified"}}
    listening.finish(path, tmp_path / "state", extra_evidence=official)
    assert FLAC(path)["TITLE"] == ["公式曲名"]


def test_first_bgm_finish_writes_lss_tags_over_unreceipted_listener_fields(tmp_path):
    path = _cue(tmp_path)
    audio = FLAC(path)
    audio.update({"GENRE": ["listener genre"], "ARTIST": ["listener composer"], "DATE": ["2024"],
                  "COMMENT": ["listener note"]})
    audio.save()
    pcm = audio_identity.audio_identity(path)["pcm_sha256"]
    assert finalizer.previous_record(path) is None
    for _ in range(2):
        listening.finish(path, tmp_path / "state")
        current = FLAC(path)
        # LSS owns the whole presentation set of its own file; a tag it never writes is left alone.
        assert current.get("GENRE") is None and current.get("ARTIST") is None and current.get("DATE") is None
        assert current["COMMENT"] == ["listener note"]
        assert audio_identity.audio_identity(path)["pcm_sha256"] == pcm
    assert "user_owned" not in finalizer.receipt(path)
