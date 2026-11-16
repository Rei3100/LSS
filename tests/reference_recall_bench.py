"""Seeded synthetic benchmark for released-reference identification under dialogue beds and TV edits.

Nothing is written to disk: every "released recording", episode usage, dialogue bed and negative is
synthesized from seeds at the fingerprint rate (`identity.SAMPLE_RATE`, mono analysis of a stereo
synthesis). The benchmark measures two layers of the existing matcher chain on the same population:

- `identity.match` (landmark votes + waveform verification): does a genuine released piece get a
  `known_exact` match at the true reference/offset, and does any `known_exact` claim land where the
  truth has no such recording (false EXACT)?
- `reference_relations.relate_segments` (region accounting on a simulated Music stem): which relation
  state each usage receives, and whether any EXACT covers material the release does not contain.

Run `python -m tests.reference_recall_bench` for the full tables (`--quick` for a smaller population).
The test module `tests/test_reference_recall_benchmark.py` keeps the negatives as hard falsifiers and
a recall floor on a small slice. Synthetic tone and speech models are harsher and simpler than real
media; numbers here rank methods, they are not Product quality claims.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sps

from lss.bgm import cue_semantics, identity, reference_relations as rr

RATE = identity.SAMPLE_RATE
SNRS = (10.0, 5.0, 0.0, -5.0, -10.0)
EDITS = ("clean", "cut", "loop", "fade_level", "eq_bandlimit", "speed", "balance")
# Resampled playback faster/slower than the waveform verification tolerates (lag drifts by several samples
# inside one 2 s block). Still the same recording, so an EXACT there is not false, but no recall is expected:
# such usages belong to `stretch_match`-style lag tracking. Reported separately, outside the "all" row.
INFO_EDITS = ("speed_beyond_tolerance",)
NEGATIVES = ("unrelated_same_key_tempo", "reorchestrated_same_melody", "fragment_in_unknown_body")
OFFSET_TOLERANCE = 0.1


# ----------------------------------------------------------------------------------------------
# Music
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Score:
    """Notes of one piece: (onset seconds, duration seconds, midi, velocity, voice)."""

    notes: tuple
    seconds: float
    tempo: float
    key: int
    drums: tuple  # (onset seconds, kind, velocity)


_MAJOR = np.array([0, 2, 4, 5, 7, 9, 11])
_PROGRESSIONS = ((0, 5, 3, 4), (0, 3, 4, 4), (5, 3, 0, 4), (0, 4, 5, 3), (1, 4, 0, 0))


def compose(seed: int, seconds: float, *, key: int | None = None, tempo: float | None = None) -> Score:
    """A tonal piece with chords, bass, melody and drums; every bar differs (no exact internal repeat)."""
    rng = np.random.default_rng(seed)
    key = int(rng.integers(0, 12)) if key is None else key
    tempo = float(rng.uniform(84, 136)) if tempo is None else tempo
    beat = 60.0 / tempo
    bar = 4 * beat
    progression = _PROGRESSIONS[int(rng.integers(len(_PROGRESSIONS)))]
    notes, drums = [], []
    bars = int(np.ceil(seconds / bar))
    melody_degree = int(rng.integers(0, 7))
    for number in range(bars):
        t0 = number * bar
        degree = progression[number % len(progression)] + (int(rng.integers(0, 2)) if number % 8 == 7 else 0)
        root = 48 + key + _MAJOR[degree % 7]
        chord = [root + _MAJOR[(degree + k) % 7] - _MAJOR[degree % 7] + (12 if (degree + k) >= 7 else 0)
                 for k in (0, 2, 4)]
        # pad: sustained chord, re-voiced every bar
        inversion = int(rng.integers(0, 3))
        for k, midi in enumerate(chord):
            notes.append((t0, bar * 0.98, midi + 12 + (12 if k < inversion else 0), 0.25, "pad"))
        # bass: rhythm varies by bar
        patterns = ((0, 2), (0, 1.5, 2, 3), (0, 0.75, 2, 2.5, 3.5), (0, 1, 2, 3))
        pattern = patterns[int(rng.integers(len(patterns)))]
        for position in pattern:
            notes.append((t0 + position * beat, beat * 0.45, root - 12 + int(rng.choice([0, 0, 7, 12])),
                          float(rng.uniform(0.5, 0.8)), "bass"))
        # arpeggio
        if rng.random() < 0.7:
            for step in range(8):
                notes.append((t0 + step * beat / 2, beat * 0.4, chord[step % 3] + 24,
                              float(rng.uniform(0.2, 0.4)), "arp"))
        # melody: random walk on the scale with varied rhythm
        position = 0.0
        while position < 4.0:
            length = float(rng.choice([0.5, 0.5, 1.0, 1.0, 1.5, 2.0, 0.25]))
            if rng.random() > 0.12:
                melody_degree = int(np.clip(melody_degree + rng.integers(-2, 3), -3, 12))
                midi = 72 + key + _MAJOR[melody_degree % 7] + 12 * (melody_degree // 7)
                notes.append((t0 + position * beat, min(length, 4.0 - position) * beat * 0.92, midi,
                              float(rng.uniform(0.5, 0.9)), "lead"))
            position += length
        for step in range(8):
            when = t0 + step * beat / 2
            if step % 4 == 0:
                drums.append((when, "kick", float(rng.uniform(0.7, 1.0))))
            if step % 4 == 2:
                drums.append((when, "snare", float(rng.uniform(0.6, 0.9))))
            if rng.random() < 0.85:
                drums.append((when + float(rng.normal(0, 0.004)), "hat", float(rng.uniform(0.2, 0.45))))
    notes = tuple(n for n in notes if n[0] < seconds)
    drums = tuple(d for d in drums if d[0] < seconds)
    return Score(notes, seconds, tempo, key, drums)


# Timbres: harmonic amplitudes, attack, decay, vibrato depth, stereo pan.
ORCHESTRA = {
    "pad": ((1.0, 0.6, 0.4, 0.25, 0.15, 0.1), 0.08, 0.6, 0.004, -0.4),
    "bass": ((1.0, 0.5, 0.2, 0.1), 0.005, 4.0, 0.0, 0.0),
    "arp": ((1.0, 0.3, 0.5, 0.1, 0.2), 0.002, 6.0, 0.0, 0.5),
    "lead": ((1.0, 0.8, 0.5, 0.35, 0.2, 0.12, 0.08), 0.02, 1.2, 0.006, 0.15),
}
# A different orchestration of the same score: other instruments, other envelopes.
REORCHESTRA = {
    "pad": ((1.0, 0.1, 0.5, 0.05, 0.3), 0.2, 0.3, 0.0, 0.3),
    "bass": ((1.0, 0.9, 0.6, 0.4, 0.2), 0.01, 2.0, 0.0, 0.1),
    "arp": ((1.0, 0.05, 0.2, 0.0, 0.1), 0.004, 3.0, 0.0, -0.5),
    "lead": ((1.0, 0.2, 0.6, 0.1, 0.4, 0.05), 0.04, 0.8, 0.012, -0.2),
}


def _drum(kind: str, velocity: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    if kind == "kick":
        t = np.arange(int(0.25 * RATE)) / RATE
        f = 50 + 90 * np.exp(-t * 30)
        return velocity * np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t * 14), 0.0
    if kind == "snare":
        t = np.arange(int(0.18 * RATE)) / RATE
        noise = sps.lfilter(*sps.butter(2, [900 / (RATE / 2), 4500 / (RATE / 2)], "band"), rng.standard_normal(t.size))
        return velocity * (0.6 * noise + 0.5 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 22), 0.1
    t = np.arange(int(0.05 * RATE)) / RATE
    noise = sps.lfilter(*sps.butter(2, 4000 / (RATE / 2), "high"), rng.standard_normal(t.size))
    return velocity * 0.5 * noise * np.exp(-t * 80), -0.3


def render(score: Score, *, orchestra=ORCHESTRA, performance: int = 0, drums: bool = True,
           section_seconds: float = 12.0) -> np.ndarray:
    """Stereo (n, 2) float32 rendering. `performance` seeds phases, timing and timbre drift."""
    rng = np.random.default_rng(10_000 + performance)
    size = int(score.seconds * RATE)
    out = np.zeros((size + RATE * 3, 2), np.float64)
    for onset, duration, midi, velocity, voice in score.notes:
        harmonics, attack, decay, vibrato, pan = orchestra[voice]
        onset = max(0.0, onset + float(rng.normal(0, 0.006)))
        # timbre variation by section (brightness drifts)
        section = int(onset // section_seconds)
        bright = 0.7 + 0.6 * ((np.sin(section * 1.7 + performance) + 1) / 2)
        length = int((duration + 0.25) * RATE)
        t = np.arange(length) / RATE
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        phase_mod = vibrato * np.sin(2 * np.pi * 5.3 * t) * (t > 0.15)
        tone = np.zeros(length)
        for order, gain in enumerate(harmonics, start=1):
            if frequency * order >= RATE / 2 - 200:
                break
            g = gain * bright ** (order - 1)
            tone += g * np.sin(2 * np.pi * frequency * order * t * (1 + phase_mod) + rng.uniform(0, 2 * np.pi))
        envelope = np.minimum(1.0, t / attack) * np.exp(-t * decay / max(duration, 0.1))
        envelope *= np.clip((duration + 0.25 - t) / 0.25, 0, 1)
        tone *= envelope * velocity
        begin = int(onset * RATE)
        left, right = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
        out[begin:begin + length, 0] += left * tone
        out[begin:begin + length, 1] += right * tone
    if drums:
        for onset, kind, velocity in score.drums:
            hit, pan = _drum(kind, velocity, rng)
            begin = int(max(0.0, onset) * RATE)
            out[begin:begin + hit.size, 0] += np.sqrt((1 - pan) / 2) * hit
            out[begin:begin + hit.size, 1] += np.sqrt((1 + pan) / 2) * hit
    # light room: decorrelated short echoes per channel
    for channel, delays in ((0, (0.031, 0.067)), (1, (0.043, 0.079))):
        dry = out[:, channel].copy()
        for k, delay in enumerate(delays):
            shift = int(delay * RATE)
            out[shift:, channel] += (0.25 / (k + 1)) * dry[:-shift]
    out = out[:size]
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.6).astype(np.float32)


def mono(stereo: np.ndarray) -> np.ndarray:
    """Same fold as the analysis decode (`-ac 1`)."""
    return stereo.mean(axis=1).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Dialogue bed and SFX
# ----------------------------------------------------------------------------------------------

def _speech_shaped_noise(size: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(size)
    # long-term speech spectrum: peak near 400-600 Hz, -6 dB/oct above ~1 kHz
    b, a = sps.butter(2, [150 / (RATE / 2), 900 / (RATE / 2)], "band")
    low = sps.lfilter(b, a, white)
    b, a = sps.butter(1, 3000 / (RATE / 2), "low")
    return low + 0.3 * sps.lfilter(b, a, rng.standard_normal(size))


def _formant_filter(x: np.ndarray, formants: list[tuple[float, float]]) -> np.ndarray:
    out = np.zeros_like(x)
    for frequency, bandwidth in formants:
        r = np.exp(-np.pi * bandwidth / RATE)
        theta = 2 * np.pi * frequency / RATE
        out += sps.lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], x)
    return out


_VOWELS = ((730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410))


def dialogue(seconds: float, seed: int, *, sfx: bool = True) -> np.ndarray:
    """Dialogue-like bed: voiced syllables (F0 contour through moving formants), unvoiced
    speech-shaped noise, syllabic AM at 3-6 Hz with phrase pauses, plus SFX bursts."""
    rng = np.random.default_rng(seed)
    size = int(seconds * RATE)
    out = np.zeros(size)
    position = int(rng.uniform(0, 0.5) * RATE)
    while position < size:
        phrase = int(rng.uniform(1.0, 4.0) * RATE)
        end = min(size, position + phrase)
        f0 = rng.uniform(95, 260)
        cursor = position
        while cursor < end:
            syllable = int(rng.uniform(0.12, 0.3) * RATE)
            n = min(syllable, end - cursor)
            if n <= 32:
                break
            t = np.arange(n) / RATE
            contour = f0 * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t + rng.uniform(0, 6)))
            phase = 2 * np.pi * np.cumsum(contour) / RATE
            source = sum(np.sin(k * phase) / k for k in range(1, int(RATE / 2 / (f0 * 1.2))))
            vowel = _VOWELS[int(rng.integers(len(_VOWELS)))]
            voiced = _formant_filter(source, [(vowel[0], 80), (vowel[1], 100), (vowel[2], 140)])
            noise = _speech_shaped_noise(n, rng) * (0.6 if rng.random() < 0.3 else 0.15)
            am = np.sin(np.pi * np.arange(n) / n) ** 0.7
            out[cursor:cursor + n] += (voiced / (np.std(voiced) + 1e-9) + noise) * am
            cursor += n + int(rng.uniform(0.0, 0.06) * RATE)
        position = end + int(rng.uniform(0.2, 1.5) * RATE)
    if sfx:
        position = int(rng.uniform(0.5, 4.0) * RATE)
        while position < size:
            kind = rng.integers(3)
            if kind == 0:  # impact
                n = int(0.3 * RATE)
                t = np.arange(n) / RATE
                burst = (rng.standard_normal(n) * 0.7 + np.sin(2 * np.pi * 70 * t)) * np.exp(-t * 18) * 3
            elif kind == 1:  # whoosh
                n = int(rng.uniform(0.5, 1.2) * RATE)
                b, a = sps.butter(2, [600 / (RATE / 2), 3000 / (RATE / 2)], "band")
                burst = sps.lfilter(b, a, rng.standard_normal(n)) * np.hanning(n) * 4
            else:  # clicks / footsteps
                n = int(0.6 * RATE)
                burst = np.zeros(n)
                for k in range(4):
                    at = int(k * 0.15 * RATE)
                    burst[at:at + 60] += rng.standard_normal(60) * 3
            n = min(n, size - position)
            out[position:position + n] += burst[:n]
            position += int(rng.uniform(2.0, 7.0) * RATE)
    return (out / (np.std(out) + 1e-9) * 0.1).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Edits
# ----------------------------------------------------------------------------------------------

@dataclass
class Piece:
    """Truth: episode [start, end) is the release `reference` at `offset` (reference - episode)."""

    start: float
    end: float
    reference: str
    offset: float


@dataclass
class Usage:
    name: str
    edit: str
    kind: str              # "positive" or a NEGATIVES name
    stereo: np.ndarray     # music only, before the bed
    pieces: list           # Piece in usage-local time
    # Seconds of the usage that are not any released recording (negatives / unknown body).
    unknown: list = field(default_factory=list)
    expected: str = "exact"  # "exact" | "released_edit" | "not_exact"
    rate: float = 1.0
    # Re-performances only: where the same score sits in the release (diagnosis of the verification margin).
    score_alignment: Piece | None = None


def _xfade_join(parts: list[np.ndarray], fade: int = int(0.02 * RATE)) -> np.ndarray:
    out = parts[0].astype(np.float64)
    for part in parts[1:]:
        part = part.astype(np.float64)
        ramp = np.linspace(0, 1, fade)[:, None]
        mixed = out[-fade:] * (1 - ramp) + part[:fade] * ramp
        out = np.concatenate([out[:-fade], mixed, part[fade:]])
    return out.astype(np.float32)


def _excerpt(release: np.ndarray, start: float, seconds: float) -> np.ndarray:
    return release[int(start * RATE):int(start * RATE) + int(seconds * RATE)]


def make_usage(edit: str, name: str, release_name: str, release: np.ndarray, rng: np.random.Generator) -> Usage:
    duration = release.shape[0] / RATE
    if edit == "cut":
        # two or three pieces from different places of the release, joined with short crossfades
        count = int(rng.integers(2, 4))
        parts, pieces, cursor = [], [], 0.0
        starts = np.sort(rng.choice(np.arange(0.0, duration - 9.0, 1.0), size=count, replace=False))
        if rng.random() < 0.5:
            starts = starts[::-1]
        for start in starts:
            seconds = float(rng.uniform(7.0, 11.0))
            seconds = min(seconds, duration - start)
            parts.append(_excerpt(release, start, seconds))
            pieces.append(Piece(cursor, cursor + seconds, release_name, float(start) - cursor))
            cursor += seconds - 0.02
        return Usage(name, edit, "positive", _xfade_join(parts), _fix_join(pieces), expected="released_edit")
    if edit == "loop":
        start = float(rng.uniform(0, duration - 26.0))
        first = float(rng.uniform(10.0, 14.0))
        back = float(rng.uniform(4.0, 8.0))
        second = float(rng.uniform(8.0, 12.0))
        parts = [_excerpt(release, start, first), _excerpt(release, start + first - back, second)]
        pieces = [Piece(0.0, first, release_name, start),
                  Piece(first - 0.02, first - 0.02 + second, release_name, start + first - back - (first - 0.02))]
        return Usage(name, edit, "positive", _xfade_join(parts), _fix_join(pieces), expected="released_edit")
    seconds = float(rng.uniform(18.0, min(36.0, duration - 2.0)))
    start = float(rng.uniform(0, duration - seconds))
    clip = _excerpt(release, start, seconds).astype(np.float64)
    rate = 1.0
    if edit == "fade_level":
        n = clip.shape[0]
        gain = np.ones(n) * 10 ** (rng.uniform(-8, -2) / 20)
        fade_in, fade_out = int(rng.uniform(1.5, 4.0) * RATE), int(rng.uniform(2.0, 5.0) * RATE)
        gain[:fade_in] *= np.linspace(0, 1, fade_in) ** 2
        gain[-fade_out:] *= np.linspace(1, 0, fade_out) ** 2
        # a level duck in the middle (as under a line), 6-10 dB for 3 s
        duck = int(rng.uniform(0.35, 0.6) * n)
        width = int(3 * RATE)
        gain[duck:duck + width] *= 10 ** (-rng.uniform(6, 10) / 20)
        clip *= gain[:, None]
        # the faded ends are still the release but quiet; the truth keeps them
    elif edit == "eq_bandlimit":
        low = float(rng.uniform(120, 300))
        high = float(rng.uniform(3000, 4500))
        b, a = sps.butter(3, [low / (RATE / 2), high / (RATE / 2)], "band")
        clip = sps.lfilter(b, a, clip, axis=0)
        # a broad presence bump (+4 dB around 2 kHz)
        b, a = sps.iirpeak(2000 / (RATE / 2), 1.0)
        clip = clip + 0.6 * sps.lfilter(b, a, clip, axis=0)
    elif edit in ("speed", "speed_beyond_tolerance"):
        # Resampled playback (a clock difference in the broadcast chain). Within tolerance the lag moves by
        # less than one sample per 2 s verification block (< 0.0045 %), though it accumulates to more than
        # the 16-tap FIR spans over a 20-36 s use; beyond tolerance it moves 4-13 samples per block.
        bounds = (0.00001, 0.00004) if edit == "speed" else (0.0002, 0.0006)
        rate = 1.0 + float(rng.choice([-1, 1])) * float(rng.uniform(*bounds))
        n = clip.shape[0]
        positions = np.arange(int(n / rate)) * rate
        positions = positions[positions < n - 1]
        clip = np.stack([np.interp(positions, np.arange(n), clip[:, c]) for c in range(2)], axis=1)
        seconds = clip.shape[0] / RATE
    elif edit == "balance":
        # stereo collapse / balance: one channel dominates, the other partly inverted
        a = float(rng.uniform(0.75, 0.95))
        b = float(rng.uniform(-0.2, 0.3))
        folded = a * clip[:, 0] + b * clip[:, 1]
        clip = np.stack([folded, folded], axis=1)
    piece = Piece(0.0, seconds, release_name, start)
    return Usage(name, edit, "positive", clip.astype(np.float32), [piece], rate=rate)


def _fix_join(pieces: list[Piece]) -> list[Piece]:
    """Crossfaded joins: do not hold the 20 ms seam against either side."""
    return pieces


def make_negatives(rng: np.random.Generator, releases: dict[str, tuple[Score, np.ndarray]], seed: int) -> list[Usage]:
    usages = []
    names = sorted(releases)
    # 1. unrelated music in the same key and tempo as a release (not registered)
    for k in range(2):
        score, _ = releases[names[k]]
        other = compose(seed * 100 + 51 + k, 30.0, key=score.key, tempo=score.tempo)
        usages.append(Usage(f"neg-unrelated-{k}", "clean", "unrelated_same_key_tempo", render(other), [],
                            unknown=[(0.0, 30.0)], expected="not_exact"))
    # 2. the same score re-orchestrated / re-performed (must never be the released recording)
    for k in range(2):
        score, _ = releases[names[k + 2]]
        variant = render(score, orchestra=REORCHESTRA if k == 0 else ORCHESTRA, performance=7 + k)
        start = float(rng.uniform(0, score.seconds - 28))
        usages.append(Usage(f"neg-reorch-{k}", "clean", "reorchestrated_same_melody", _excerpt(variant, start, 26.0),
                            [], unknown=[(0.0, 26.0)], expected="not_exact",
                            score_alignment=Piece(0.0, 26.0, names[k + 2], start)))
    # 3. a fragment of a release inside a longer unknown body in the same key/tempo
    for k in range(2):
        name = names[k + 4]
        score, release = releases[name]
        body = render(compose(seed * 100 + 71 + k, 36.0, key=score.key, tempo=score.tempo))
        fragment_start = float(rng.uniform(0, score.seconds - 9))
        fragment = _excerpt(release, fragment_start, 7.0)
        at = float(rng.uniform(12.0, 18.0))
        clip = body.copy()
        clip[int(at * RATE):int(at * RATE) + fragment.shape[0]] = fragment
        usages.append(Usage(f"neg-fragment-{k}", "clean", "fragment_in_unknown_body", clip,
                            [Piece(at, at + 7.0, name, fragment_start - at)],
                            unknown=[(0.0, at), (at + 7.0, 36.0)], expected="not_exact"))
    return usages


# ----------------------------------------------------------------------------------------------
# Population
# ----------------------------------------------------------------------------------------------

@dataclass
class Population:
    releases: dict          # name -> (Score, stereo)
    usages: list            # Usage


def population(seed: int = 1, *, releases: int = 8, per_edit: int = 2) -> Population:
    rng = np.random.default_rng(seed)
    table = {}
    for k in range(releases):
        seconds = float(rng.uniform(30, 120))
        score = compose(seed * 1000 + k, seconds)
        table[f"R{k:02d}"] = (score, render(score))
    usages = []
    names = sorted(table)
    for edit in EDITS + INFO_EDITS:
        for k in range(per_edit):
            name = names[int(rng.integers(len(names)))]
            usages.append(make_usage(edit, f"{edit}-{k}", name, table[name][1], rng))
    usages.extend(make_negatives(rng, table, seed))
    return Population(table, usages)


@dataclass
class Episode:
    """Usages laid out with gaps of dialogue; mixture, simulated Music stem and dialogue stem."""

    mixture: np.ndarray
    music: np.ndarray
    dialogue: np.ndarray
    placements: list  # (usage, episode start seconds)


def episode(pop: Population, snr_db: float, *, seed: int = 5, leak: float = 0.05, gap: float = 6.0) -> Episode:
    cursor = gap
    placements = []
    for usage in pop.usages:
        placements.append((usage, cursor))
        cursor += usage.stereo.shape[0] / RATE + gap
    size = int(cursor * RATE)
    music = np.random.default_rng(seed).normal(0, 1e-4, size).astype(np.float32)
    bed = dialogue(cursor, seed=seed + 1)
    dialogue_track = np.zeros(size, np.float32)
    for usage, at in placements:
        clip = mono(usage.stereo)
        begin = int(at * RATE)
        music[begin:begin + clip.size] += clip
        # SNR per usage: music RMS over active music vs the bed under it
        bed_part = bed[begin:begin + clip.size]
        active = np.abs(clip) > 1e-3
        music_rms = float(np.sqrt(np.mean(clip[active].astype(np.float64) ** 2))) if active.any() else 1e-3
        bed_rms = float(np.sqrt(np.mean(bed_part.astype(np.float64) ** 2))) + 1e-9
        dialogue_track[begin:begin + clip.size] = bed_part * (music_rms / bed_rms) * 10 ** (-snr_db / 20)
    # dialogue continues in the gaps at the -0 dB reference level of the last usage
    gaps = dialogue_track == 0
    dialogue_track[gaps] = bed[gaps]
    mixture = music + dialogue_track
    return Episode(mixture, music + leak * dialogue_track, (1 - leak) * dialogue_track, placements)


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------

def library_for(pop: Population, tmp: Path | None = None) -> rr.ReferenceLibrary:
    import tempfile

    from tests.release_receipts import distribution_receipt

    signals = {name: mono(stereo) for name, (_, stereo) in pop.releases.items()}
    directory = Path(tmp or tempfile.mkdtemp(prefix="lss-recall-"))
    paths = []
    for name in sorted(signals):
        path = directory / f"{name}.flac"
        path.write_bytes(name.encode())  # identity only; content comes from the loader
        paths.append(path)
    library = rr.ReferenceLibrary.build(paths, loader=lambda p: signals[Path(p).stem])
    for reference in library.index.references:
        library.metadata[reference.sha256]["independent_distribution"] = distribution_receipt(reference.sha256)
    return library


def _truth(ep: Episode):
    """Episode-time truth pieces and unknown spans."""
    pieces, unknown = [], []
    for usage, at in ep.placements:
        for p in usage.pieces:
            pieces.append((usage, Piece(at + p.start, at + p.end, p.reference, p.offset - at)))
        for a, b in usage.unknown:
            unknown.append((usage, at + a, at + b))
    return pieces, unknown


def _correct(match, piece: Piece, rate: float = 1.0) -> bool:
    if Path(match.reference_path).stem != piece.reference:
        return False
    # a resampled usage drifts; accept the offset anywhere along its path
    drift = abs(rate - 1.0) * (piece.end - piece.start)
    return abs(match.offset_seconds - piece.offset) <= OFFSET_TOLERANCE + drift


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def score_identity(ep: Episode, matches: list) -> dict:
    """Per-piece recall and false-EXACT accounting for `identity.match` results."""
    pieces, unknown = _truth(ep)
    exact = [m for m in matches if m.identity_state == "known_exact"]
    rows = []
    for usage, piece in pieces:
        covered = 0.0
        for m in exact:
            if _correct(m, piece, usage.rate):
                covered += _overlap(m.query_start_seconds, m.query_end_seconds, piece.start, piece.end)
        length = piece.end - piece.start
        # diagnosis: the strongest candidate at the true alignment, whatever its state
        near = [m for m in matches if _correct(m, piece, usage.rate)
                and _overlap(m.query_start_seconds, m.query_end_seconds, piece.start, piece.end) > 0]
        best = max(near, key=lambda m: (m.identity_state == "known_exact", m.votes), default=None)
        rows.append({"usage": usage.name, "edit": usage.edit, "kind": usage.kind,
                     "seconds": round(length, 2), "covered": round(min(covered, length), 2),
                     "best": None if best is None else {
                         "votes": best.votes, "windows": best.measured_windows,
                         "drift": best.max_window_drift_seconds, "explained": best.waveform_explained,
                         "null_max": best.waveform_null_max, "blocks": best.waveform_blocks,
                         "state": best.identity_state}})
    false = []
    correct_seconds = 0.0
    total_seconds = 0.0
    for m in exact:
        span = m.query_end_seconds - m.query_start_seconds
        total_seconds += span
        good = sum(_overlap(m.query_start_seconds, m.query_end_seconds, p.start, p.end)
                   for u, p in pieces if _correct(m, p, u.rate))
        correct_seconds += min(good, span)
        # false EXACT: more than one second of the claim lies outside every true piece at its offset
        if span - good > 1.0:
            where = [u.name for u, a, b in unknown if _overlap(m.query_start_seconds, m.query_end_seconds, a, b) > 0]
            false.append({"reference": Path(m.reference_path).stem, "start": m.query_start_seconds,
                          "end": m.query_end_seconds, "offset": m.offset_seconds, "unsupported_seconds": round(span - good, 2),
                          "inside": where})
    return {"pieces": rows, "false_exact": false,
            "precision_seconds": round(correct_seconds / total_seconds, 4) if total_seconds else None}


def _segment(start: float, end: float) -> dict:
    return {"start_seconds": start, "end_seconds": end, "core_seconds": [start, end],
            "start_edge": {"kind": cue_semantics.EDGE_SILENCE}, "end_edge": {"kind": cue_semantics.EDGE_SILENCE},
            "stretch": [start, end], "candidates": [[start, end]]}


def score_relations(ep: Episode, library: rr.ReferenceLibrary, matches: list) -> list[dict]:
    rows = []
    for usage, at in ep.placements:
        # The relation map measures whole one-second cells. A segment that ends inside a second also measures
        # what follows the use (silence while the reference keeps playing), which is a boundary question of
        # cue formation, not of identification. Score the whole seconds inside the use.
        start = float(np.ceil(at))
        end = float(np.floor(at + usage.stereo.shape[0] / RATE))
        pieces = rr.relate_segments([_segment(start, end)], query=ep.mixture, library=library, matches=matches,
                                    music=ep.music, interference=[ep.dialogue])
        states = [(p["start_seconds"], p["end_seconds"], p["reference_relation"]["state"],
                   rr.released_use(p["reference_relation"]),
                   (p["reference_relation"].get("reference") or {}).get("file_name")) for p in pieces]
        exact_seconds = sum(b - a for a, b, s, _, _ in states if s == rr.EXACT)
        released_seconds = sum(b - a for a, b, _, r, _ in states if r)
        # false EXACT/released: any released-use piece overlapping unknown material by > 1 s
        false = 0.0
        for a, b, s, released, _ in states:
            if not released:
                continue
            for u0, u1 in usage.unknown:
                false += _overlap(a, b, at + u0, at + u1)
        length = end - start
        truth_seconds = sum(_overlap(at + p.start, at + p.end, start, end) for p in usage.pieces)
        rows.append({"usage": usage.name, "edit": usage.edit, "kind": usage.kind, "expected": usage.expected,
                     "seconds": round(length, 2), "truth_seconds": round(truth_seconds, 2),
                     "exact_seconds": round(exact_seconds, 2), "released_seconds": round(released_seconds, 2),
                     "false_released_seconds": round(false, 2),
                     "states": [[round(a - at, 2), round(b - at, 2), s] for a, b, s, _, _ in states]})
    return rows


def run(seed: int = 1, *, snrs=SNRS, per_edit: int = 2, relations: bool = True, releases: int = 8) -> dict:
    pop = population(seed, releases=releases, per_edit=per_edit)
    library = library_for(pop)
    result = {"seed": seed, "per_edit": per_edit, "snr": {}}
    for snr in snrs:
        ep = episode(pop, snr, seed=seed * 7 + 5)
        clock = time.perf_counter()
        matches = identity.match(ep.mixture, library.index, collapse=False)
        entry = {"identity": score_identity(ep, matches), "match_seconds": round(time.perf_counter() - clock, 1)}
        if relations:
            clock = time.perf_counter()
            entry["relations"] = score_relations(ep, library, matches)
            entry["relation_seconds"] = round(time.perf_counter() - clock, 1)
        result["snr"][str(snr)] = entry
    return result


def tables(results: list[dict]) -> str:
    """Markdown tables aggregated over seeds."""
    lines = []
    snrs = list(results[0]["snr"])
    # identity layer, piece-level recall (>= 50 % of the piece covered at the true offset) and seconds recall
    lines.append("identity.match: piece recall (>=50% covered at true ref/offset) / seconds recall")
    lines.append("| edit | " + " | ".join(f"{s} dB" for s in snrs) + " |")
    lines.append("|---|" + "---|" * len(snrs))
    for edit in EDITS + INFO_EDITS:
        cells = []
        for snr in snrs:
            rows = [r for res in results for r in res["snr"][snr]["identity"]["pieces"]
                    if r["edit"] == edit and r["kind"] == "positive"]
            hit = sum(r["covered"] >= 0.5 * r["seconds"] for r in rows)
            sec = sum(r["covered"] for r in rows) / max(sum(r["seconds"] for r in rows), 1e-9)
            cells.append(f"{hit}/{len(rows)} · {sec:.2f}")
        lines.append(f"| {edit} | " + " | ".join(cells) + " |")
    cells = []
    for snr in snrs:
        rows = [r for res in results for r in res["snr"][snr]["identity"]["pieces"]
                if r["kind"] == "positive" and r["edit"] in EDITS]
        hit = sum(r["covered"] >= 0.5 * r["seconds"] for r in rows)
        sec = sum(r["covered"] for r in rows) / max(sum(r["seconds"] for r in rows), 1e-9)
        cells.append(f"**{hit}/{len(rows)} · {sec:.2f}**")
    lines.append("| all | " + " | ".join(cells) + " |")
    precision = []
    false = []
    for snr in snrs:
        values = [res["snr"][snr]["identity"]["precision_seconds"] for res in results]
        values = [v for v in values if v is not None]
        precision.append(f"{np.mean(values):.3f}" if values else "-")
        false.append(str(sum(len(res["snr"][snr]["identity"]["false_exact"]) for res in results)))
    lines.append("| precision (s) | " + " | ".join(precision) + " |")
    lines.append("| false EXACT | " + " | ".join(false) + " |")
    if "relations" in results[0]["snr"][snrs[0]]:
        lines.append("")
        lines.append("relate_segments: usages whose truth is covered >=50% by released use (EXACT or complete released EDIT)")
        lines.append("| edit | " + " | ".join(f"{s} dB" for s in snrs) + " |")
        lines.append("|---|" + "---|" * len(snrs))
        for edit in EDITS + INFO_EDITS:
            cells = []
            for snr in snrs:
                rows = [r for res in results for r in res["snr"][snr]["relations"]
                        if r["edit"] == edit and r["kind"] == "positive"]
                hit = sum(r["released_seconds"] >= 0.5 * r["truth_seconds"] for r in rows)
                exact = sum(r["exact_seconds"] >= 0.5 * r["truth_seconds"] for r in rows)
                cells.append(f"{hit}/{len(rows)} (EXACT {exact})")
            lines.append(f"| {edit} | " + " | ".join(cells) + " |")
        cells = []
        for snr in snrs:
            rows = [r for res in results for r in res["snr"][snr]["relations"]
                    if r["kind"] == "positive" and r["edit"] in EDITS]
            hit = sum(r["released_seconds"] >= 0.5 * r["truth_seconds"] for r in rows)
            exact = sum(r["exact_seconds"] >= 0.5 * r["truth_seconds"] for r in rows)
            cells.append(f"**{hit}/{len(rows)} (EXACT {exact})**")
        lines.append("| all | " + " | ".join(cells) + " |")
        for negative in NEGATIVES:
            cells = []
            for snr in snrs:
                rows = [r for res in results for r in res["snr"][snr]["relations"] if r["kind"] == negative]
                bad = sum(r["false_released_seconds"] > 1.0 for r in rows)
                cells.append(f"false {bad}/{len(rows)}")
            lines.append(f"| neg: {negative} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--per-edit", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-relations", action="store_true")
    parser.add_argument("--snrs", type=float, nargs="+", default=list(SNRS))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--merge", type=Path, nargs="+", help="print tables for saved --json results")
    args = parser.parse_args(argv)
    if args.merge:
        results = [item for path in args.merge for item in json.loads(path.read_text())]
        print(tables(results))
        return
    seeds = args.seeds[:1] if args.quick else args.seeds
    results = [run(seed, snrs=args.snrs, per_edit=args.per_edit, relations=not args.no_relations)
               for seed in seeds]
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))
    print(tables(results))
    for res in results:
        for snr, entry in res["snr"].items():
            for item in entry["identity"]["false_exact"]:
                print("FALSE EXACT identity", res["seed"], snr, item)
            for row in entry.get("relations", []):
                if row["false_released_seconds"] > 1.0:
                    print("FALSE RELEASED relation", res["seed"], snr, row["usage"], row["states"])


if __name__ == "__main__":
    main()
