from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC, Picture

from lss.core import audio_identity
from lss.output import finalizer


def _asset(tmp_path, *, seconds=2, level=0.2):
    path = tmp_path / "accepted.flac"
    t = np.arange(int(44100 * seconds)) / 44100
    sf.write(path, np.column_stack((level * np.sin(2 * np.pi * 440 * t),
                                   level * np.sin(2 * np.pi * 630 * t))), 44100, subtype="PCM_24")
    return path


def _tags():
    return {"TITLE": "仮の表示名", "LSS_SOURCE_TYPE": "bgm_cue", "LSS_INPUT_SHA256": "a" * 64,
            "LSS_BGM_CUE_KEY": "same-logical-cue"}


def test_finalization_measures_encoded_audio_preserves_pcm_and_reuses_without_scan(tmp_path, monkeypatch):
    path = _asset(tmp_path)
    before = audio_identity.audio_identity(path)
    result = finalizer.ensure(path, tags=_tags())
    assert result["audio"] == before
    assert finalizer.receipt(path, tags=_tags()) is not None
    tags = FLAC(path)
    assert "ARTIST" not in tags and "ALBUM" not in tags
    assert tags["REPLAYGAIN_TRACK_GAIN"]
    assert float(tags["LSS_TRUE_PEAK_DBTP"][0]) == pytest.approx(result["finalization"]["measurement"]["true_peak_dbtp"], abs=1e-6)
    original_bytes, mtime = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(finalizer, "scan_track_file", lambda *a, **k: pytest.fail("再走査不要"))
    assert finalizer.ensure(path, tags=_tags())["reused"]
    assert path.read_bytes() == original_bytes and path.stat().st_mtime_ns == mtime


def test_metadata_update_is_copy_on_write_and_does_not_modify_hardlinked_history(tmp_path):
    path = _asset(tmp_path)
    finalizer.ensure(path, tags=_tags())
    history = tmp_path / "history.flac"
    os.link(path, history)
    original = history.read_bytes()
    changed = {**_tags(), "TITLE": "根拠のある公式名"}
    result = finalizer.ensure(path, tags=changed)
    assert not result["reused"]
    assert history.read_bytes() == original
    assert audio_identity.audio_identity(path) == audio_identity.audio_identity(history)
    assert FLAC(path)["TITLE"] == ["根拠のある公式名"]


@pytest.mark.parametrize("failure", ["scan", "save", "late_input", "replace"])
def test_failure_preserves_published_asset_and_cleans_stage(tmp_path, monkeypatch, failure):
    path = _asset(tmp_path)
    original = path.read_bytes()
    def refuse(*a, **k):
        raise OSError("injected publication failure")
    options = {}
    if failure == "scan":
        monkeypatch.setattr(finalizer, "scan_track_file", refuse)
    elif failure == "save":
        monkeypatch.setattr(FLAC, "save", refuse)
    elif failure == "replace":
        monkeypatch.setattr(finalizer.os, "replace", refuse)
    else:
        options["validate_before_publish"] = refuse
    with pytest.raises(OSError):
        finalizer.ensure(path, tags=_tags(), **options)
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".lss-listening-*.tmp")) == []
    assert finalizer.receipt(path) is None


def test_peak_limited_playback_metadata_does_not_limit_signal(tmp_path):
    path = _asset(tmp_path, level=0.9)
    original = audio_identity.audio_identity(path)
    from lss.output.scan import scan_track_file
    def scanner(path):
        scan = scan_track_file(path)
        scan.update(loudness_lufs=-30.0, track_gain_db=12.0, true_peak_dbtp=-0.1)
        return scan
    record = finalizer.finalize_staged(path, tags=_tags(), scanner=scanner)
    assert record["requested_gain_db"] == 12
    assert record["playback_gain_db"] <= -0.9
    assert FLAC(path)["LSS_LISTENING_GAIN_LIMITED"] == ["1"]
    assert audio_identity.audio_identity(path) == original


@pytest.mark.parametrize("seconds,level", [(0.1, 0.2), (2, 0.0)])
def test_unmeasurable_loudness_remains_unknown(tmp_path, seconds, level):
    path = _asset(tmp_path, seconds=seconds, level=level)
    record = finalizer.finalize_staged(path, tags=_tags())
    assert record["measurement"]["loudness_lufs"] is None
    assert "REPLAYGAIN_TRACK_GAIN" not in FLAC(path)
    assert record["requested_gain_db"] is None


def test_receipt_detects_tampered_tags_artwork_and_pcm(tmp_path):
    path = _asset(tmp_path)
    picture = Picture()
    picture.data, picture.mime, picture.type = b"trusted-cover-bytes", "image/png", 3
    finalizer.ensure(path, tags=_tags(), picture=picture)
    audio = FLAC(path)
    audio["TITLE"] = "changed"
    audio.save()
    assert finalizer.receipt(path, tags=_tags(), picture=picture) is None
    finalizer.ensure(path, tags=_tags(), picture=picture)
    audio = FLAC(path)
    audio.clear_pictures()
    audio.save()
    assert finalizer.receipt(path) is None
