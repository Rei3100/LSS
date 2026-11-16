"""Reference identification under dialogue beds and TV edits, on the seeded synthetic benchmark.

Hard falsifiers (must hold at every SNR): unrelated music in the same key and tempo, the same score
re-orchestrated or re-performed, and a released fragment inside a longer unknown body never receive a
`known_exact` claim or a released-use relation beyond what the release really contains. Recall floors on a
small slice keep the dialogue-bed verification from silently regressing; the full tables come from
`python -m tests.reference_recall_bench` (`cloud-reports/reference-recall.md`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lss.bgm import identity, reference_relations as rr
from tests import reference_recall_bench as bench

RATE = bench.RATE


@pytest.fixture(scope="module")
def small():
    """Five releases, every edit once and all negatives (seeded, built in memory)."""
    pop = bench.population(11, releases=6, per_edit=1)
    return pop, bench.library_for(pop)


@pytest.fixture(scope="module")
def episodes(small):
    pop, library = small
    found = {}
    for snr in (10.0, -10.0):
        ep = bench.episode(pop, snr, seed=17)
        found[snr] = (ep, identity.match(ep.mixture, library.index, collapse=False))
    return found


@pytest.mark.parametrize("snr", [10.0, -10.0])
def test_no_known_exact_claim_reaches_material_the_release_does_not_contain(episodes, snr):
    ep, matches = episodes[snr]
    scored = bench.score_identity(ep, matches)
    assert scored["false_exact"] == []
    # the negatives' unknown bodies get no exact claim at all
    _, unknown = bench._truth(ep)
    for m in matches:
        if m.identity_state != "known_exact":
            continue
        for usage, a, b in unknown:
            assert bench._overlap(m.query_start_seconds, m.query_end_seconds, a, b) <= 1.0, usage.name


@pytest.mark.parametrize("snr", [10.0, -10.0])
def test_negatives_never_become_a_released_use(small, episodes, snr):
    _, library = small
    ep, matches = episodes[snr]
    ep.placements = [(u, at) for u, at in ep.placements if u.kind != "positive"]
    rows = bench.score_relations(ep, library, matches)
    assert {row["kind"] for row in rows} == set(bench.NEGATIVES)
    for row in rows:
        assert row["false_released_seconds"] <= 1.0, row
        if row["kind"] != "fragment_in_unknown_body":
            assert all(state != rr.EXACT for _, _, state in row["states"]), row


def test_dialogue_bed_recall_floor_at_minus_ten_db(episodes):
    """At -10 dB music-to-dialogue the absolute waveform threshold cannot be reached (the mixture is ~9 %
    music); the bed verification still identifies contiguous, cut and looped uses."""
    ep, matches = episodes[-10.0]
    pieces = bench.score_identity(ep, matches)["pieces"]
    wanted = [p for p in pieces if p["kind"] == "positive" and p["edit"] in ("clean", "cut", "loop", "balance")]
    found = [p for p in wanted if p["covered"] >= 0.5 * p["seconds"]]
    assert len(found) >= 0.75 * len(wanted), pieces
    assert any(m.evidence_kind == "landmark_and_bed_waveform" for m in matches if m.identity_state == "known_exact")


def test_bed_verification_separates_the_recording_from_another_performance_of_its_score():
    rng = np.random.default_rng(4)
    score = bench.compose(901, 30.0)
    release = bench.mono(bench.render(score))
    other = bench.mono(bench.render(score, performance=5))
    reorchestrated = bench.mono(bench.render(score, orchestra=bench.REORCHESTRA, performance=6))
    bed = bench.dialogue(30.0, seed=3)
    size = 20 * RATE

    def mixed(music):
        m = music[2 * RATE:2 * RATE + size].astype(np.float64)
        d = bed[2 * RATE:2 * RATE + size].astype(np.float64)
        return (m + d * np.std(m) / np.std(d) * 10 ** (10 / 20)).astype(np.float32)  # -10 dB

    reference = release.copy()
    nulls = [int(c) for c in rng.integers(300, 420, size=8)]  # frames, far from the true 86
    same = identity.bed_run(mixed(release), reference, 0, size, 2 * RATE, nulls)
    assert same is not None and same["blocks"] >= 8 and same["remaining"] <= identity.BED_MAX_RUN_REMAINING
    for music in (other, reorchestrated):
        assert identity.bed_run(mixed(music), reference, 0, size, 2 * RATE, nulls) is None
        blocks = identity.bed_blocks(mixed(music), reference, 0, size, 2 * RATE)
        assert np.median(blocks["remaining"]) > identity.BED_MAX_REMAINING


def test_bed_thresholds_enter_the_matching_settings_identity(monkeypatch):
    before = identity.matching_settings_id()
    monkeypatch.setattr(identity, "BED_MAX_REMAINING", identity.BED_MAX_REMAINING + 0.01)
    assert identity.matching_settings_id() != before


def _cut_episode(clip_of, release_seconds: float = 70.0, seed: int = 21):
    """Three pieces of one recording joined at whole seconds, under a 0 dB dialogue bed."""
    score = bench.compose(seed, release_seconds)
    release = bench.mono(bench.render(score))
    episode_music = clip_of(score)
    parts = [(40.0, 9.0), (5.0, 8.0), (22.0, 10.0)]
    clip = np.concatenate([episode_music[int(a * RATE):int((a + n) * RATE)] for a, n in parts])
    at = 4.0
    size = int((at + clip.size / RATE + 4.0) * RATE)
    music = np.random.default_rng(1).normal(0, 1e-4, size).astype(np.float32)
    music[int(at * RATE):int(at * RATE) + clip.size] += clip
    bed = bench.dialogue(size / RATE, seed=seed)
    bed *= np.std(clip) / np.std(bed[int(at * RATE):int(at * RATE) + clip.size])
    library = bench.library_for(bench.Population({"R": (score, bench.render(score))}, []))
    mixture = music + bed
    matches = identity.match(mixture, library.index, collapse=False)
    segment = bench._segment(at, at + 27.0)
    [piece] = rr.relate_segments([segment], query=mixture, library=library, matches=matches,
                                 music=music + 0.05 * bed, interference=[0.95 * bed])
    return piece["reference_relation"], release


def test_a_cut_of_one_released_recording_is_its_released_edit_not_a_component_variant():
    relation, _ = _cut_episode(lambda score: bench.mono(bench.render(score)))
    assert relation["state"] == rr.EDIT and rr.released_use(relation), relation.get("basis")


def test_a_cut_of_a_version_without_a_layer_of_the_release_never_becomes_released_use():
    relation, _ = _cut_episode(lambda score: bench.mono(bench.render(score, drums=False)))
    assert not rr.released_use(relation), relation["state"]
