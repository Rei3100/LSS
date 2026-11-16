"""A mono reference match cannot discard wanted music hidden in the stereo difference."""
import numpy as np
import pytest
import soundfile as sf

from lss.bgm import cue_semantics, identity, reference_reconstruction, reference_relations as rr, reference_stereo
from tests.release_receipts import distribution_receipt
from tests.test_bgm_identity import _music
from tests.test_bgm_reference_relations import _segment

RATE = identity.SAMPLE_RATE


def _case(tmp_path, overlay=False, stereo=False):
    a, b = _music(1, 40.), _music(3, 40.)
    reference = np.column_stack((a + .3 * b, .8 * a - .2 * b)) if stereo else np.column_stack((a, a))
    query = reference[10 * RATE:30 * RATE].copy()
    if overlay:
        foreign = .25 * _music(77, 20.)
        query += np.column_stack((foreign, -foreign))
    ref, source = tmp_path / "reference.wav", tmp_path / "music.wav"
    sf.write(ref, reference, RATE, subtype="FLOAT")
    sf.write(source, query, RATE, subtype="FLOAT")
    library = rr.ReferenceLibrary.build([ref])
    for item in library.index.references:
        library.metadata[item.sha256]["independent_distribution"] = distribution_receipt(item.sha256)
    gate = rr.Gate.prepare(source, cache_dir=None, music=source, library=library)
    return gate, source


@pytest.mark.parametrize("stereo", [False, True])
def test_antiphase_unregistered_music_prevents_release_suppression_and_donor_use(tmp_path, stereo):
    gate, _ = _case(tmp_path, overlay=True, stereo=stereo)
    pieces = gate.relate([_segment(0., 20.)])
    assert pieces
    for piece in pieces:
        relation = piece["reference_relation"]
        assert not rr.released_use(relation)
        assert not reference_reconstruction.plan(relation, piece["start_seconds"], piece["end_seconds"])["eligible"]
        piece.update(semantic_class=cue_semantics.WANTED, reasons=[])
    assert all(item["semantic_class"] == cue_semantics.WANTED for item in rr.apply(pieces, cue_semantics.KNOWN_RELEASED))


@pytest.mark.parametrize("stereo", [False, True])
def test_unmodified_stereo_reference_still_avoids_redundant_extraction(tmp_path, stereo):
    gate, _ = _case(tmp_path, stereo=stereo)
    pieces = gate.relate([_segment(0., 20.)])
    assert pieces and all(rr.released_use(piece["reference_relation"]) for piece in pieces)


def _nominated(gate, edited=False):
    sha = gate.library.index.references[0].sha256
    relation = {"state": rr.EDIT if edited else rr.EXACT,
                "reference": {"reference_sha256": sha, "offset_seconds": 10.,
                              "independent_distribution": distribution_receipt(sha)},
                "regions": [{"start_seconds": 0., "end_seconds": 10. if edited else 20.,
                             "kind": "supported", "reference_sha256": sha, "offset_seconds": 10.}],
                "released_cover": {"complete": True}}
    if edited:
        relation["regions"].append({"start_seconds": 10., "end_seconds": 20., "kind": "supported",
                                    "reference_sha256": sha, "offset_seconds": 15.})
    return {**_segment(0., 20.), "reference_relation": relation}


def _guard(gate, source, item):
    [result] = reference_stereo.guard([item], music=source, library=gate.library, recipe=gate.recipe)
    return result["reference_relation"]


@pytest.mark.parametrize("change", ["gain", "swap", "short_eq", "channel_mix"])
def test_joint_accounting_preserves_linear_stereo_changes(tmp_path, change):
    gate, source = _case(tmp_path, stereo=True)
    query, rate = sf.read(source, always_2d=True)
    if change == "gain":
        query *= 1.3
    elif change == "swap":
        query = query[:, ::-1]
    elif change == "short_eq":
        query[1:] = .9 * query[1:] + .1 * query[:-1]
    else:
        query = query @ np.array([[1., .1], [-.1, .8]])
    sf.write(source, query, rate, subtype="FLOAT")
    relation = _guard(gate, source, _nominated(gate))
    assert relation["state"] == rr.EXACT
    assert relation["stereo_accounting"]["state"] == "verified"


@pytest.mark.parametrize("start", [0., 10., 19.75])
def test_quarter_second_antiphase_overlay_is_not_averaged_away(tmp_path, start):
    gate, source = _case(tmp_path, stereo=True)
    query, rate = sf.read(source, always_2d=True)
    foreign = .125 * _music(77, .25)
    a = round(start * rate)
    query[a:a + len(foreign)] += np.column_stack((foreign, -foreign))
    sf.write(source, query, rate, subtype="FLOAT")
    relation = _guard(gate, source, _nominated(gate))
    assert relation["state"] == rr.UNKNOWN
    assert relation["stereo_accounting"]["state"] == "unexplained_stereo_material"
    assert not reference_reconstruction.plan(relation, 0., 20.)["eligible"]


@pytest.mark.parametrize("overlay", [False, True])
def test_edit_verifies_each_reference_offset_and_blocks_stereo_overlay(tmp_path, overlay):
    gate, source = _case(tmp_path, stereo=True)
    ref, rate = sf.read(gate.library.index.references[0].path, always_2d=True)
    query = np.concatenate((ref[10 * rate:20 * rate], ref[25 * rate:35 * rate]))
    if overlay:
        foreign = .125 * _music(77, .25)
        query[10 * rate:10 * rate + len(foreign)] += np.column_stack((foreign, -foreign))
    sf.write(source, query, rate, subtype="FLOAT")
    relation = _guard(gate, source, _nominated(gate, edited=True))
    assert rr.released_use(relation) is (not overlay)
    if overlay:
        assert not relation["released_cover"]["complete"]
        assert not reference_reconstruction.plan(relation, 0., 20.)["eligible"]
    else:
        assert [row["offset_seconds"] for row in relation["stereo_accounting"]["pieces"]] == [10., 15.]


@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("failure", ["decode", "short", "nonfinite", "changed_reference"])
def test_unverified_stereo_has_no_release_or_donor_authority(tmp_path, monkeypatch, edited, failure):
    gate, source = _case(tmp_path, stereo=True)
    item = _nominated(gate, edited=edited)
    real = identity.decode_stereo_window

    def broken(path, start, frames):
        if failure == "decode":
            raise identity.DecodeError("controlled decode failure")
        values = real(path, start, frames)
        if failure == "short":
            return values[:-1]
        if failure == "nonfinite":
            return np.full_like(values, np.nan)
        return values

    if failure == "changed_reference":
        with open(gate.library.index.references[0].path, "ab") as stream:
            stream.write(b"changed")
    else:
        monkeypatch.setattr(identity, "decode_stereo_window", broken)
    relation = _guard(gate, source, item)
    assert relation["state"] == rr.UNKNOWN
    assert relation["stereo_accounting"]["state"] == "unverified"
    assert not rr.released_use(relation)
    assert not reference_reconstruction.plan(relation, 0., 20.)["eligible"]
    assert rr.compact(relation)["mono_state"] == item["reference_relation"]["state"]
    assert rr.compact(relation)["stereo_accounting"] == relation["stereo_accounting"]


def test_stereo_guard_is_part_of_the_relation_recipe():
    recipe = rr.default_recipe()
    assert recipe.to_dict()["stereo_accounting"] == reference_stereo.policy()


def test_tail_is_not_fitted_as_an_unconstrained_few_sample_problem():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, .1, (5512 + 20, 2))
    query = reference.copy()
    query[-20:] += .1
    result = reference_stereo._measure(query, reference, 0)
    # The small perturbation is below the Product veto threshold; it must still
    # leave a measurable residual instead of an unconstrained exact fit.
    assert result["max_residual"] > .001
    with pytest.raises(ValueError, match="too short"):
        reference_stereo._measure(query[:20], reference[:20], 0)


def _partial(gate):
    item = _nominated(gate, edited=True)
    relation = item["reference_relation"]
    relation["released_cover"]["complete"] = False
    relation["regions"] = [
        {**relation["regions"][0], "median_joint": 1., "median_novelty": 0., "max_novelty": 0.,
         "median_absence": 0., "max_absence": 0.},
        {"start_seconds": 10., "end_seconds": 20., "kind": "unexplained_music"}]
    return item


@pytest.mark.parametrize("overlay", [False, True])
def test_partial_donor_checks_only_the_final_replacement_span(tmp_path, overlay):
    gate, source = _case(tmp_path, stereo=True)
    query, rate = sf.read(source, always_2d=True)
    foreign = _music(77, 10.)
    query[10 * rate:] = np.column_stack((foreign, -foreign))
    if overlay:
        addition = .25 * foreign[:rate]
        query[:rate] += np.column_stack((addition, -addition))
    sf.write(source, query, rate, subtype="FLOAT")
    item = _partial(gate)
    before = reference_reconstruction.plan(item["reference_relation"], 0., 20.)
    assert before["eligible"]
    relation = _guard(gate, source, item)
    after = reference_reconstruction.plan(relation, 0., 20.)
    assert after["eligible"] is (not overlay)
    assert not rr.released_use(relation)
    if not overlay:
        assert after["donors"] == before["donors"]
        assert relation["regions"][-1]["kind"] == "unexplained_music"
        assert relation["stereo_accounting"]["scope"] == "final_donor_ranges"
        assert all(p["span_seconds"][1] <= 10. for p in relation["stereo_accounting"]["pieces"])


@pytest.mark.parametrize("state", [rr.EDIT, rr.COMPONENT])
def test_descriptive_relation_without_suppression_or_donors_is_retained(tmp_path, state):
    relation = {"state": state, "basis": "partial_observation", "released_cover": {"complete": False},
                "regions": [{"start_seconds": 0., "end_seconds": 20., "kind": "unexplained_music"}]}
    item = {**_segment(0., 20.), "reference_relation": relation}
    [after] = reference_stereo.guard([item], music=None, library=None, recipe=rr.default_recipe())
    assert after["reference_relation"]["state"] == state
    assert after["reference_relation"]["regions"] == relation["regions"]
    assert not rr.released_use(after["reference_relation"])
    assert not reference_reconstruction.plan(after["reference_relation"], 0., 20.)["eligible"]
