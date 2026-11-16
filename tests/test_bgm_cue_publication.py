"""公開cueは保留候補の音声の連結で作り、別の内容を無言で上書きしない（`lss.bgm.cue_publication`）。"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.bgm import cue_publication
from lss.core import audio_identity

RATE = 1000


def _part(tmp_path, name, first, values, shift=0):
    path = tmp_path / f"{name}.flac"
    sf.write(str(path), values * 2.0 ** -shift, RATE, format="FLAC", subtype="PCM_24")
    record = {"path": str(path), "sha256": audio_identity.file_digest(path),
              "audio": audio_identity.audio_identity(path), "source_start_frame": first,
              "source_end_frame": first + len(values), "sample_rate": RATE,
              "serialization": {"headroom_shift": shift}}
    return record


def _tone(frames, level=0.5):
    t = np.arange(frames) / RATE
    return np.stack([level * np.sin(2 * np.pi * 5 * t), 0.5 * level * np.sin(2 * np.pi * 7 * t)], axis=1)


def test_compose_joins_candidates_and_undoes_their_declared_headroom(tmp_path):
    signal = _tone(3000, level=1.4)
    parts = [_part(tmp_path, "a", 0, signal[:1000], shift=1), _part(tmp_path, "b", 1000, signal[1000:3000], shift=1)]
    audio = cue_publication.compose(parts, 500, 2500)
    np.testing.assert_allclose(audio, signal[500:2500], atol=2 ** -21)


def test_compose_refuses_a_gap_or_a_changed_candidate(tmp_path):
    signal = _tone(3000)
    parts = [_part(tmp_path, "a", 0, signal[:1000]), _part(tmp_path, "b", 1200, signal[1200:3000])]
    with pytest.raises(cue_publication.CuePublicationError):
        cue_publication.compose(parts, 500, 2500)
    whole = _part(tmp_path, "c", 0, signal)
    sf.write(whole["path"], signal * 0.5, RATE, format="FLAC", subtype="PCM_24")
    with pytest.raises(cue_publication.CuePublicationError):
        cue_publication.compose([whole], 0, 3000)


def test_publish_keeps_values_reuses_the_same_cue_and_never_overwrites_another(tmp_path):
    audio = _tone(2000, level=1.2)
    target = tmp_path / "out" / "ep - BGM 01 (00m00s).flac"
    kwargs = dict(title="ep BGM 01", album="ep BGM", provenance={"type": "bgm_published_cue"},
                  input_sha256="a" * 64, recipe_id="recipe", version="0.1.0")
    first = cue_publication.publish(audio, RATE, target, key="k1", **kwargs)
    written, _ = sf.read(first["path"], dtype="float64", always_2d=True)
    assert first["headroom_shift"] == 1 and FLAC(first["path"])["LSS_OUTPUT_HEADROOM_SHIFT"] == ["1"]
    np.testing.assert_allclose(written * 2.0, audio, atol=2 ** -22)
    again = cue_publication.publish(audio, RATE, target, key="k1", **kwargs)
    assert again["path"] == first["path"] and len(list(target.parent.iterdir())) == 1
    other = cue_publication.publish(audio * 0.5, RATE, target, key="k2", **kwargs)
    assert other["path"].endswith("(r2).flac") and audio_identity.matches_record(target, first)


@pytest.mark.parametrize("change", ["first", "boundary", "last", "short", "long", "rate", "mono"])
def test_saved_audio_verification_rejects_changes_across_blocks(tmp_path, monkeypatch, change):
    monkeypatch.setattr(cue_publication, "_BLOCK", 128)
    audio = _tone(300)
    written, rate = audio.copy(), RATE
    if change in ("first", "boundary", "last"):
        index = {"first": 0, "boundary": 128, "last": 299}[change]
        written[index, 1] += .01
    elif change == "short":
        written = written[:-1]
    elif change == "long":
        written = np.concatenate((written, written[:1]))
    elif change == "rate":
        rate += 1
    else:
        written = written[:, 0]
    path = tmp_path / "changed.flac"
    sf.write(path, written, rate, subtype="PCM_24")
    with pytest.raises(cue_publication.CuePublicationError):
        cue_publication._verify_saved_audio(path, audio, RATE, 1.)


@pytest.mark.parametrize("frames,level", [(128, .5), (129, 1.4), (300, .5)])
def test_saved_audio_verification_preserves_quantization_and_headroom(tmp_path, monkeypatch, frames, level):
    monkeypatch.setattr(cue_publication, "_BLOCK", 128)
    audio = _tone(frames, level)
    scaled, _, scale = cue_publication.serialization_audio(audio)
    path = tmp_path / "valid.flac"
    sf.write(path, scaled, RATE, subtype="PCM_24")
    cue_publication._verify_saved_audio(path, audio, RATE, scale)


def test_saved_audio_verification_bounds_each_decode(tmp_path, monkeypatch):
    audio = _tone(300)
    path = tmp_path / "blocks.flac"
    sf.write(path, audio, RATE, subtype="PCM_24")
    monkeypatch.setattr(cue_publication, "_BLOCK", 128)
    read = sf.SoundFile.read
    counts = []

    def bounded(stream, frames=-1, **kwargs):
        assert 0 < frames <= 128
        counts.append(frames)
        return read(stream, frames, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", bounded)
    cue_publication._verify_saved_audio(path, audio, RATE, 1.)
    assert counts == [128, 128, 44]
