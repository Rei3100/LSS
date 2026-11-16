"""同じ楽曲の素材を共有する別の録音。キー・テンポ・音色・ミックスの違いを許し、別の曲とは区別する。"""
from __future__ import annotations

import numpy as np

from lss.bgm import versions

SR = versions.SAMPLE_RATE


def _song(seed: int, *, seconds: float = 60.0, transpose: int = 0, tempo: float = 1.0,
          harmonics: int = 3, noise: float = 0.0, noise_seed: int = 0) -> np.ndarray:
    """seedごとの和音と旋律の並び。tempo>1で速く演奏する。"""
    rng = np.random.default_rng(seed)
    beat = 0.5 / tempo
    steps = int(seconds / 0.5)
    roots = rng.integers(0, 12, size=steps // 4 + 1)
    melody = rng.integers(0, 12, size=steps + 1)
    length = int(seconds / tempo * SR)
    out = np.zeros(length, dtype=np.float32)
    t = np.arange(int(beat * SR)) / SR
    envelope = np.minimum(1.0, t * 40) * np.exp(-t * 2.0)

    def tone(midi: float, amplitude: float) -> np.ndarray:
        freq = 440.0 * 2 ** ((midi - 69) / 12)
        return (amplitude * sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1)) * envelope
                ).astype(np.float32)

    for step in range(steps):
        begin = int(step * beat * SR)
        if begin >= length:
            break
        end = min(length, begin + t.size)
        root = 48 + roots[step // 4] + transpose
        chord = tone(root, 0.3) + tone(root + 4, 0.25) + tone(root + 7, 0.25)
        out[begin:end] += (chord + tone(72 + melody[step] + transpose, 0.35))[: end - begin]
    if noise:
        out += (np.random.default_rng(noise_seed).standard_normal(length) * noise).astype(np.float32)
    return out


def test_an_arrangement_in_another_key_tempo_and_timbre_shares_material_within_a_library():
    features = {f"other-{seed}": versions.chroma(_song(seed)) for seed in range(100, 124)}
    features["original"] = versions.chroma(_song(7))
    features["arrangement"] = versions.chroma(_song(7, transpose=3, tempo=0.9, harmonics=8, noise=0.05))
    relations = versions.library_relations(features)
    shared = [r for r in relations if r.state == "shared_material_candidate"]
    assert [(r.first_id, r.second_id) for r in shared] == [("arrangement", "original")]
    assert all(r.state == "not_shared" for r in relations if r not in shared)


def test_a_segment_of_an_arrangement_points_to_the_original_only():
    references = {f"other-{seed}": versions.chroma(_song(seed)) for seed in range(200, 224)}
    references["original"] = versions.chroma(_song(11))
    segment = versions.chroma(_song(11, transpose=-2, tempo=1.1, harmonics=6, noise=0.05)[int(10 * SR):int(45 * SR)])
    found = versions.segment_relations(segment, references)
    assert [item["reference_id"] for item in found] == ["original"]
    unrelated = versions.chroma(_song(12)[int(10 * SR):int(45 * SR)])
    assert versions.segment_relations(unrelated, references) == []


def test_small_libraries_short_segments_and_silence_are_not_judged():
    small = {name: versions.chroma(_song(seed)) for name, seed in (("a", 1), ("b", 1), ("c", 2))}
    assert {r.state for r in versions.library_relations(small)} == {"undetermined"}
    references = {f"r{seed}": versions.chroma(_song(seed)) for seed in range(300, 324)}
    assert versions.segment_relations(versions.chroma(_song(300, seconds=8.0)), references) == []
    assert versions.chroma(np.zeros(100, dtype=np.float32)).shape == (0, 12)
    silent = versions.chroma(np.zeros(SR * 30, dtype=np.float32))
    assert versions.qmax(silent, references["r300"]) == 0.0
