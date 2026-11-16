"""Scoped local search must improve observations without granting release authority."""
from dataclasses import replace

import numpy as np
import pytest

from lss.bgm import cue_semantics, identity, reference_reconstruction, reference_relations as rr
from tests.test_bgm_reference_relations import Episode, RATE, _library, _segment
from tests.test_bgm_identity import _music


@pytest.mark.parametrize("variant", ["clean", "foreign_insert", "overlay", "missing_component"])
def test_autonomous_fragment_candidates_keep_local_authority(tmp_path, variant):
    reference = _music(1, 80.0)
    starts = [4, 12, 20, 28, 36, 44, 52, 60]
    music = np.concatenate([reference[s * RATE:(s + 2) * RATE] for s in starts])
    if variant == "missing_component":
        reference = reference + _music(3, 80.0, harmonics=(1.0,), transpose=-12.0)
    if variant == "foreign_insert":
        music[6 * RATE:8 * RATE] = _music(77, 2.0)
    elif variant == "overlay":
        music += 0.5 * _music(77, 16.0)
    episode = Episode(16.0, [(0, music)], dialogue_gain=0.05)
    library = _library(tmp_path, {"A": reference})
    matches = identity.match(episode.mixture, library.index, collapse=False)
    assert matches, "The real fingerprint path must nominate its own reference"
    kwargs = dict(query=episode.mixture, library=library, matches=matches,
                  music=episode.music, interference=[episode.dialogue])
    baseline = rr.relate_segments([_segment(0, 16)],
                                 recipe=replace(rr.default_recipe(), nomination_max_windows=0), **kwargs)
    pieces = rr.relate_segments([_segment(0, 16)], **kwargs)
    relations = [p["reference_relation"] for p in pieces]
    assert any(r.get("local_nomination", {}).get("accepted_scopes", 0) for r in relations), relations
    assert all(r["state"] != rr.EXACT for r in relations)
    assert all(not r.get("released_cover", {}).get("complete", False) for r in relations)
    assert all(not r.get("donor_reconstructable", False) for r in relations)
    assert all(not reference_reconstruction.plan(p["reference_relation"],
               p["start_seconds"], p["end_seconds"])["eligible"] for p in pieces)
    assert all(not rr.released_use(r) for r in relations)
    decided = [{**p, "semantic_class": cue_semantics.WANTED, "reasons": []} for p in pieces]
    assert all(p["semantic_class"] == cue_semantics.WANTED
               for p in rr.apply(decided, cue_semantics.KNOWN_RELEASED))
    assert rr.occurrences(decided) == []
    for relation in relations:
        compact = rr.compact(relation)
        assert compact["local_nomination"] == relation["local_nomination"]
        assert not reference_reconstruction.plan(compact, 0, 16)["eligible"]
    for related in rr.related_occurrences(decided):
        assert related["authority_hold_reasons"] == ["scoped_local_observations_require_full_accounting"]
        assert related["local_nomination"]["accepted_scopes"] > 0
    assert sum(p["end_seconds"] - p["start_seconds"] for p in pieces) == 16
    assert all(r["reference_relation"]["state"] != rr.EXACT for r in baseline)
    local_regions = [region for r in relations for region in r.get("regions", [])
                     if region.get("origin") == "scoped_local_nomination"]
    # This missing-layer mixture can remain unresolved rather than positively
    # classified as missing; a nomination must not turn it into supported music.
    if variant != "missing_component":
        assert local_regions
    assert all(region["end_seconds"] - region["start_seconds"] <= 2 for region in local_regions)
    for relation in relations:
        report = relation.get("local_nomination")
        if report:
            assert report["searched_windows"] <= 4 * 32
            assert report["waveform_checks"] <= 4 * 32 * 3
    if variant == "foreign_insert":
        assert not any(region["kind"] == rr.CELL_SUPPORTED
                       and region["start_seconds"] <= 6 and region["end_seconds"] >= 8
                       for r in relations for region in r.get("regions", []))
    if variant == "missing_component":
        assert not any(region["kind"] == rr.CELL_SUPPORTED
                       for r in relations for region in r.get("regions", []))


def test_local_window_budget_leaves_unobserved_music_unknown(tmp_path):
    reference = _music(1, 80.0)
    music = np.concatenate([reference[s * RATE:(s + 2) * RATE]
                            for s in [4, 12, 20, 28, 36, 44, 52, 60]])
    episode = Episode(16.0, [(0, music)])
    library = _library(tmp_path, {"A": reference})
    matches = identity.match(episode.mixture, library.index, collapse=False)
    [piece] = rr.relate_segments([_segment(0, 16)], query=episode.mixture,
        library=library, matches=matches, music=episode.music,
        interference=[episode.dialogue],
        recipe=replace(rr.default_recipe(), nomination_max_windows=1))
    relation = piece["reference_relation"]
    report = relation["local_nomination"]
    assert report["budget_exhausted"] and report["searched_windows"] == 1
    assert report["accepted_scopes"] == 1
    local = [r for r in relation.get("regions", [])
             if r.get("origin") == "scoped_local_nomination"]
    assert all(0 <= r["start_seconds"] < r["end_seconds"] <= 2 for r in local)
    assert relation["state"] != rr.EXACT
    assert not relation.get("released_cover", {}).get("complete", False)


def test_presence_on_one_alignment_does_not_hide_unexplained_music(tmp_path):
    """Real fingerprints for one strong section must not mask later short fragments."""
    reference = _music(1, 80.0)
    foreign = _music(77, 4.0)
    # First 4 s are a stable A offset. The next 4 s retain A under a foreign
    # overlay; the final 8 s jump among unrelated A offsets, defeating one
    # whole-span alignment while leaving local A opportunities.
    music = np.concatenate([
        reference[20 * RATE:24 * RATE],
        reference[24 * RATE:28 * RATE] + 0.5 * foreign,
        reference[42 * RATE:44 * RATE],
        reference[8 * RATE:10 * RATE],
        reference[60 * RATE:62 * RATE],
        reference[32 * RATE:34 * RATE],
    ])
    episode = Episode(16.0, [(0, music)])
    library = _library(tmp_path, {"A": reference})
    matches = identity.match(episode.mixture, library.index, collapse=False)
    assert matches, "The production fingerprint path must nominate A"
    [piece] = rr.relate_segments([_segment(0, 16)], query=episode.mixture,
        library=library, matches=matches, music=episode.music,
        interference=[episode.dialogue])
    relation = piece["reference_relation"]
    assert relation["local_nomination"]["accepted_scopes"] > 0
    assert relation["state"] != rr.EXACT
    assert not relation.get("released_cover", {}).get("complete", False)
    assert not relation.get("donor_reconstructable", False)


def test_scoped_missing_at_disjoint_offset_does_not_mask_measured_support(tmp_path, monkeypatch):
    """Real fingerprints/accounting: distant same-SHA passages are independent evidence."""
    from scipy.signal import lfilter

    excerpt = _music(31, 16.0)
    layer = _music(73, 16.0, harmonics=(1.0,), transpose=12.0)
    reference = _music(9, 80.0)
    filtered = lfilter([1.0, -0.92], [1.0], excerpt).astype(np.float32)
    reference[8 * RATE:24 * RATE] = filtered
    reference[40 * RATE:56 * RATE] = excerpt + 0.8 * layer
    episode = Episode(32.0, [(0.0, excerpt), (16.0, _music(88, 16.0))])
    library = _library(tmp_path, {"A": reference})
    matches = identity.match(episode.mixture, library.index, collapse=False)
    offsets = [m.offset_seconds for m in matches
               if m.reference_sha256 == library.index.references[0].sha256]
    assert any(abs(offset - 8.0) < 1.0 for offset in offsets), offsets
    assert any(abs(offset - 40.0) < 1.0 for offset in offsets), offsets

    observed_scopes = []
    original_nominate = rr._Relator._nominate

    def capture_scopes(relator, cells, candidates, quiet):
        local, report = original_nominate(relator, cells, candidates, quiet)
        observed_scopes.extend((alignment.offset, [relator._label(evidence, i)
                                                  for i in range(len(cells))], evidence.reference_ranges)
                               for alignment, evidence in local)
        return local, report

    monkeypatch.setattr(rr._Relator, "_nominate", capture_scopes)
    # Exercise the conflict policy even if the independent donor-retention
    # policy would otherwise preserve the baseline map without local search.
    monkeypatch.setattr(reference_reconstruction, "clean_donor_regions",
                        lambda *args, **kwargs: {"donors": [], "rejected_regions": [], "protected_spans": []})
    [piece] = rr.relate_segments([_segment(0, 32)], query=episode.mixture,
        library=library, matches=matches, music=episode.music,
        interference=[episode.dialogue])
    relation = piece["reference_relation"]
    assert relation.get("local_nomination", {}).get("accepted_scopes", 0) > 0
    missing = [(offset, i, ranges[i]) for offset, labels, ranges in observed_scopes
               for i, label in enumerate(labels) if label == rr.CELL_MISSING]
    assert any(abs(offset - 40.0) < 1.0 for offset, _, _ in missing), missing
    assert any(region["kind"] == rr.CELL_SUPPORTED
               and abs(region.get("offset_seconds", -100) - 8.0) < 1.0
               and any(region["start_seconds"] <= cell < region["end_seconds"]
                       for _, cell, _ in missing)
               for region in relation.get("regions", [])), relation.get("regions")
    assert relation["state"] != rr.EXACT
    assert not relation.get("released_cover", {}).get("complete", False)


def test_overlapping_scoped_missing_footprint_still_vetoes_support():
    """Actual measured 40s/40.251s views overlap and retain their contradiction."""
    period = 2767
    motif = _music(211, period / RATE).astype(np.float32)
    query = np.tile(motif, int(np.ceil(2 * RATE / period)))[:2 * RATE].copy()
    reference = np.zeros(80 * RATE, dtype=np.float32)
    reference[40 * RATE:42 * RATE] = np.tile(motif, int(np.ceil(2 * RATE / period)))[:2 * RATE]
    border_start, border_end = int(round(41.5 * RATE)), int(round(41.751 * RATE))
    layer = _music(223, (border_end - border_start) / RATE, harmonics=(1.0,), transpose=7.0)
    reference[border_start:border_end] += 0.75 * layer[:border_end - border_start]
    evidence = rr._Evidence(query, [])
    support = evidence.measure(reference, 40.0, [0, 1])
    missing = evidence.measure(reference, 40.251, [0, 1])
    relator = object.__new__(rr._Relator)
    relator.recipe = rr.default_recipe()
    assert relator._label(support, 0) == rr.CELL_SUPPORTED
    assert relator._label(missing, 0) == rr.CELL_MISSING
    support_range = support.reference_ranges[0]
    missing_range = missing.reference_ranges[0]
    assert support_range and missing_range
    assert max(support_range[0], missing_range[0]) < min(support_range[1], missing_range[1])
    assert rr._reference_ranges_conflict(support_range, missing_range)
    assert rr._reference_ranges_conflict(None, support_range)
    assert rr._reference_ranges_conflict(missing_range, None)


def test_scoped_missing_without_footprint_falls_back_to_sha_wide_veto(tmp_path, monkeypatch):
    """Legacy/custom measurements without bounds remain conservative and do not crash scatter."""
    from scipy.signal import lfilter

    excerpt = _music(31, 16.0)
    layer = _music(73, 16.0, harmonics=(1.0,), transpose=12.0)
    reference = _music(9, 80.0)
    reference[8 * RATE:24 * RATE] = lfilter([1.0, -0.92], [1.0], excerpt).astype(np.float32)
    reference[40 * RATE:56 * RATE] = excerpt + 0.8 * layer
    episode = Episode(32.0, [(0.0, excerpt), (16.0, _music(88, 16.0))])
    library = _library(tmp_path, {"A": reference})
    matches = identity.match(episode.mixture, library.index, collapse=False)
    assert matches

    original_measure = rr._Evidence.measure
    observed_missing = []
    original_nominate = rr._Relator._nominate

    def omit_two_cell_ranges(evidence, reference_samples, offset, cells):
        measured = original_measure(evidence, reference_samples, offset, cells)
        if len(cells) == 2:
            measured.reference_ranges = None
        return measured

    def capture_missing(relator, cells, candidates, quiet):
        local, report = original_nominate(relator, cells, candidates, quiet)
        observed_missing.extend((alignment.offset, i, relator._label(evidence, i))
                                for alignment, evidence in local
                                for i in range(len(cells))
                                if relator._label(evidence, i) == rr.CELL_MISSING)
        return local, report

    monkeypatch.setattr(rr._Evidence, "measure", omit_two_cell_ranges)
    monkeypatch.setattr(rr._Relator, "_nominate", capture_missing)
    monkeypatch.setattr(reference_reconstruction, "clean_donor_regions",
                        lambda *args, **kwargs: {"donors": [], "rejected_regions": [], "protected_spans": []})
    [piece] = rr.relate_segments([_segment(0, 32)], query=episode.mixture,
        library=library, matches=matches, music=episode.music,
        interference=[episode.dialogue])
    relation = piece["reference_relation"]
    assert relation.get("local_nomination", {}).get("accepted_scopes", 0) > 0
    assert any(abs(offset - 40.0) < 1.0 for offset, _, _ in observed_missing), observed_missing
    assert not any(region["kind"] == rr.CELL_SUPPORTED
                   and abs(region.get("offset_seconds", -100) - 8.0) < 1.0
                   and any(region["start_seconds"] <= cell < region["end_seconds"]
                           for _, cell, _ in observed_missing)
                   for region in relation.get("regions", [])), relation.get("regions")
    assert relation["state"] != rr.EXACT
    assert not relation.get("released_cover", {}).get("complete", False)


def test_existing_short_clean_donor_is_not_held_by_local_nomination(tmp_path):
    """A pre-existing four-second clean donor remains usable beside foreign music."""
    released = _music(101, 80.0)
    missing_layer = _music(109, 4.0, harmonics=(1.0,), transpose=-12.0)
    reference = released.copy()
    reference[10 * RATE:14 * RATE] += 0.4 * missing_layer
    foreign = _music(103, 2.0)
    episode = Episode(10.0, [(0.0, released[10 * RATE:18 * RATE]), (8.0, foreign)])
    library = _library(tmp_path, {"A": reference, "B": _music(107, 80.0)})
    matches = identity.match(episode.mixture, library.index, collapse=False)
    pieces = rr.relate_segments([_segment(0, 10)], query=episode.mixture,
        library=library, matches=matches, music=episode.music,
        interference=[episode.dialogue])
    relations = [p["reference_relation"] for p in pieces]
    assert not any(r.get("local_nomination") for r in relations), relations
    plans = [reference_reconstruction.plan(r, p["start_seconds"], p["end_seconds"])
             for p, r in zip(pieces, relations)]
    assert any(plan["eligible"] and any(d["end_seconds"] - d["start_seconds"] <= 5.0
                                         for d in plan["donors"])
               for plan in plans), plans


def test_positive_missing_measurement_stays_local_and_is_not_smoothed(tmp_path, monkeypatch):
    """Measurement/classification boundary; injected values are not an audio detector claim."""
    reference = _music(1, 80.0)
    music = np.concatenate([reference[s * RATE:(s + 2) * RATE]
                            for s in [4, 12, 20, 28, 36, 44, 52, 60]])
    episode = Episode(16.0, [(0, music)])
    library = _library(tmp_path, {"A": reference})
    matches = identity.match(episode.mixture, library.index, collapse=False)
    original = rr._Evidence.measure

    def measure(evidence, samples, offset, cells):
        observed = original(evidence, samples, offset, cells)
        if len(cells) == 2:
            # The one allowed local window has a positively measured missing
            # component. Ordinary full-span measurements remain actual values.
            observed.joint[:] = 1.0
            observed.novelty[:] = 0.0
            observed.absence[:] = 1.0
            observed.inside[:] = True
        return observed

    monkeypatch.setattr(rr._Evidence, "measure", measure)
    [piece] = rr.relate_segments([_segment(0, 16)], query=episode.mixture,
        library=library, matches=matches, music=episode.music,
        interference=[episode.dialogue],
        recipe=replace(rr.default_recipe(), nomination_max_windows=1))
    relation = piece["reference_relation"]
    assert relation["local_nomination"]["accepted_scopes"] == 1
    regions = relation["regions"]
    missing = [r for r in regions if r["kind"] == rr.CELL_MISSING]
    assert [(r["start_seconds"], r["end_seconds"]) for r in missing] == [(0, 2)]
    assert all(r["end_seconds"] <= 2 or r["kind"] != rr.CELL_MISSING for r in regions)
    assert relation["state"] != rr.EXACT
    assert not reference_reconstruction.plan(relation, 0, 16)["eligible"]
