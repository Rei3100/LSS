"""Key reading used for the music's direction words (bright/dark) of scene names; never a title by itself."""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm import music_description, scene_job

RATE = 44100


def _tone(pitches, seconds, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    out = np.zeros_like(t)
    for pitch in pitches:
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        for harmonic in range(1, 6):
            out += np.sin(2 * np.pi * f0 * harmonic * t) / harmonic
    return out


def _cadence(tonic: int, minor: bool, repeats=4, rate=RATE):
    third = 3 if minor else 4
    chords = [(0, third, 7), (5, 5 + (3 if minor else 4), 12), (7, 11, 14), (0, third, 7)]
    scale = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    parts = []
    for _ in range(repeats):
        for number, chord in enumerate(chords):
            notes = [48 + tonic + step for step in chord] + [60 + tonic + scale[(number * 2) % 7]]
            parts.append(_tone(notes, 2.0, rate))
    signal = np.concatenate(parts)
    return (0.2 * signal / np.max(np.abs(signal))).astype(np.float32)


@pytest.mark.parametrize("tonic", range(0, 12, 3))
@pytest.mark.parametrize("minor", [False, True])
def test_key_of_a_cadence_names_the_tonic_and_mode(tonic, minor):
    found = music_description.key(_cadence(tonic, minor), RATE)
    assert found["status"] == "measured", found
    assert (found["tonic"], found["mode"]) == (tonic, "minor" if minor else "major")


def test_key_abstains_on_noise_and_short_audio():
    rng = np.random.default_rng(3)
    assert music_description.key(rng.standard_normal(RATE * 20).astype(np.float32) * 0.1, RATE)["status"] == "abstained"
    assert music_description.key(_cadence(2, True)[: RATE * 4], RATE)["reason"] == "too_short"


def test_sound_words_say_dark_for_a_minor_cue_and_bright_for_a_major_one(tmp_path):
    import soundfile as sf

    for minor, word in ((True, "短調"), (False, "長調")):
        mono = _cadence(2, minor)
        path = tmp_path / f"{word}.flac"
        sf.write(path, np.column_stack((mono, mono)), RATE)
        assert word in scene_job.sound_words(path)
