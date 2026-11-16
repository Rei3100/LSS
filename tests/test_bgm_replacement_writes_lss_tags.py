"""A cue replaced by a better CurrentBest cue is written with LSS's complete current tag set.

IDENTITY_AND_CURRENTBEST: tags, covers, ReplayGain and library information of an LSS file are LSS-owned Product
state. Nothing is carried from the replaced file; the archived history copy keeps the bytes it had.
"""
from __future__ import annotations

from mutagen.flac import FLAC, Picture

from lss import application
from lss.assets.naming import NamingStore
from lss.bgm import current_best, listening
from lss.output import finalizer
from tests.test_application_bgm_candidates import _run, gpu  # noqa: F401
from tests.test_application_bgm_current_best import _history_files, _older_and_improved, _public


def _picture():
    picture = Picture()
    picture.type, picture.mime, picture.desc, picture.data = 3, "image/png", "foreign", b"\x89PNG\r\n\x1a\nxx" * 4
    return picture


def _foreign_edits(path):
    audio = FLAC(path)
    audio["TITLE"] = "他者の曲名"
    audio["COMMENT"] = "my note"
    audio["MY_CUSTOM_TAG"] = "kept?"
    audio["REPLAYGAIN_TRACK_GAIN"] = "-3.10 dB"
    audio.add_picture(_picture())
    audio.save()


def test_replacing_cue_carries_lss_full_tag_set_and_nothing_from_the_replaced_file(tmp_path, gpu, monkeypatch):
    source, old, old_run = _older_and_improved(tmp_path, monkeypatch, gpu)
    _foreign_edits(old)
    edited = old.read_bytes()
    outcome = _run(tmp_path, source)
    [new] = outcome.outputs
    assert _public(tmp_path) == ["episode - BGM 00m05s.flac"]
    tags = FLAC(new)
    record = finalizer.receipt(new)
    assert record is not None and "user_owned" not in record and "artwork_owner" not in record
    # The file carries exactly what LSS projected: every projected tag, the naming revision of the store, no cover.
    assert all(tags.get(key) == value for key, value in record["projected_tags"].items())
    state = NamingStore(tmp_path / "state" / listening.DIRECTORY).read(tags["LSS_NAMING_ASSET_ID"][0])
    assert state is not None and tags["LSS_NAMING_REVISION"] == [state["current_revision_id"]]
    assert tags["TITLE"] != ["他者の曲名"] and "COMMENT" not in tags and "MY_CUSTOM_TAG" not in tags
    assert list(tags.pictures) == []
    assert tags.get("REPLAYGAIN_TRACK_GAIN") != ["-3.10 dB"] and tags.get("REPLAYGAIN_TRACK_GAIN")
    old_tags = FLAC(_history_files(tmp_path)[0])
    assert tags["LSS_BGM_CUE_KEY"] != old_tags["LSS_BGM_CUE_KEY"]
    [archived] = _history_files(tmp_path)
    assert archived.read_bytes() == edited
    current = current_best.read(tmp_path / "state", old_run["source"]["sha256"])
    [decision] = current["current"]["decisions"]["old"]
    assert decision["decision"] == "replace_with_better"
    assert not [key for key in decision if key.startswith("user_tag")]
    before = new.read_bytes()
    again = _run(tmp_path, source)
    assert again.reused is True and new.read_bytes() == before


def test_the_user_tag_carrying_step_no_longer_exists():
    assert not hasattr(application, "_carry_user_tags_to_replacements")
    assert not hasattr(finalizer, "user_payload") and not hasattr(finalizer, "apply_user_payload")
    assert not hasattr(finalizer, "ownership")
