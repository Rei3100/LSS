from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import music_facts

RATE = 44100


def clicks(bpm: float, seconds: float = 20.0) -> np.ndarray:
    out = np.zeros(int(seconds * RATE), dtype=np.float32)
    period = 60.0 / bpm
    burst = (np.hanning(400) * np.sin(2 * np.pi * 1500 * np.arange(400) / RATE)).astype(np.float32)
    for k in range(int(seconds / period)):
        start = int(k * period * RATE)
        out[start:start + 400] += 0.5 * burst[: len(out) - start]
    return out


@pytest.mark.parametrize("bpm", [80, 100])
def test_a_clear_beat_states_its_tempo(bpm):
    found = music_facts.tempo(clicks(bpm), RATE)
    assert found["status"] == "measured"
    assert abs(found["bpm"] - bpm) / bpm < 0.03


@pytest.mark.parametrize("bpm", [70, 90, 128, 150])
def test_a_stated_tempo_is_right_or_nothing_is_stated(bpm):
    # A strictly periodic click is as strong at half its tempo: the reading may abstain, never state the half.
    found = music_facts.tempo(clicks(bpm), RATE)
    assert found["status"] == "abstained" or abs(found["bpm"] - bpm) / bpm < 0.03


def test_noise_states_no_tempo_and_no_bpm_tag(tmp_path):
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(RATE * 20) * 0.05).astype(np.float32)
    assert music_facts.tempo(noise, RATE)["status"] == "abstained"
    path = tmp_path / "noise.flac"
    sf.write(path, np.column_stack((noise, noise)), RATE)
    facts = music_facts.measure(path)
    assert "BPM" not in music_facts.tags(facts) and "INITIALKEY" not in music_facts.tags(facts)
    assert "拍は自由に揺れる" in music_facts.words(facts)


def test_words_and_tags_say_only_what_was_measured(tmp_path):
    path = tmp_path / "clicks.flac"
    mono = clicks(100)
    sf.write(path, np.column_stack((mono, mono)), RATE)
    facts = music_facts.measure(path)
    assert music_facts.tags(facts)["BPM"] == "100"
    assert "テンポ約100" in music_facts.words(facts) and "編成" not in music_facts.words(facts)
