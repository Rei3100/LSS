"""Short assets require measured outside boundaries, never duration alone."""
from dataclasses import replace
import numpy as np
import pytest
import soundfile as sf

from lss.bgm import cue_semantics as cs, cue_scope, short_cues as sc

RATE = 8000
SOURCE = {"path": "episode.wav", "sha256": "a"*64}


def setup_case(tmp_path, *, singing=False):
    t = np.arange(10*RATE)/RATE
    x = .1*np.sin(2*np.pi*220*t)
    x[(t < 3.1) | (t >= 6.9)] = 0
    path = tmp_path/"music.wav"
    sf.write(path, np.column_stack([x, .8*x]), RATE, subtype="FLOAT")
    p = np.zeros((10, 527), np.float32)
    p[3:7, cs.MUSIC_CLASS] = .9
    if singing:
        p[3:7, 27] = .8
    db = np.full(10, -100.)
    db[3:7] = -30.
    e = cs.WindowEvidence.from_arrays(db, p, np.ones((10, 8)))
    segment = {"start_seconds": 3.2, "end_seconds": 6.8, "core_seconds": [3.2, 6.8],
               "candidates": [[0., 10.]], "stretch": [0., 10.],
               "start_edge": {"kind": cs.EDGE_SILENCE}, "end_edge": {"kind": cs.EDGE_SILENCE}}
    [segment] = cs.decide([segment], e, cs.default_recipe(), source_seconds=10.)
    return path, e, p, segment


def decide(path, e, p, segment):
    [item] = sc.apply([segment], e, p, path, cs.default_recipe(), float(e.count))
    cue_scope.decide([item], SOURCE, mixture=p)
    return item


@pytest.mark.parametrize("singing", [False, True])
def test_bounded_short_music_and_singing_pass_without_cutting_the_body(tmp_path, singing):
    path, e, p, segment = setup_case(tmp_path, singing=singing)
    item = decide(path, e, p, segment)
    assert cue_scope.publishable(item)
    assert sc.confirmed(item)
    assert item["start_seconds"] <= segment["start_seconds"]
    assert item["end_seconds"] >= segment["end_seconds"]
    assert item["features"]["singing_present"] is singing
    assert item["short_music"]["original_mixture_authority"] == "none"
    measured = cue_scope.integrity(sf.read(path, always_2d=True)[0][3*RATE:7*RATE], RATE)
    assert cue_scope.final(item, 0, SOURCE, measured)["disposition"] == "PUBLISHED"


def test_interior_notches_do_not_establish_an_outside_boundary(tmp_path):
    path, e, p, segment = setup_case(tmp_path)
    t = np.arange(10*RATE)/RATE
    x = .1*np.sin(2*np.pi*220*t)
    x[((t > 3.3) & (t < 3.4)) | ((t > 6.6) & (t < 6.7))] = 0
    sf.write(path, np.column_stack([x, x]), RATE, subtype="FLOAT")
    item = decide(path, e, p, segment)
    assert not cue_scope.publishable(item)
    assert not item["short_music"]["confirmed"]
    assert item["start_seconds"] == 3.2 and item["end_seconds"] == 6.8


def test_one_quiet_channel_cannot_prove_a_boundary(tmp_path):
    path, e, p, segment = setup_case(tmp_path)
    x, _ = sf.read(path, always_2d=True)
    t = np.arange(len(x))/RATE
    x[:, 1] = .1*np.sin(2*np.pi*230*t)
    sf.write(path, x, RATE, subtype="FLOAT")
    assert not cue_scope.publishable(decide(path, e, p, segment))


@pytest.mark.parametrize("failure", ["source_music", "music", "event", "candidate_edge", "reference", "partial", "insert", "missing", "truncated", "candidate_hole"])
def test_short_route_keeps_independent_vetoes(tmp_path, failure):
    path, e, p, segment = setup_case(tmp_path)
    if failure == "source_music":
        p[4, cs.MUSIC_CLASS] = .49
    elif failure == "music":
        e = replace(e, music_probability=e.music_probability.copy())
        e.music_probability[4] = .49
    elif failure == "event":
        p[3:7, 72] = .9
    elif failure == "candidate_edge":
        segment["start_edge"] = {"kind": cs.EDGE_CANDIDATE}
        segment["reasons"].append("start_edge_music_continues_outside_candidate")
    elif failure == "reference":
        segment["semantic_class"] = cs.KNOWN_RELEASED
    elif failure == "partial":
        segment["reasons"].append("partly_registered_reference")
    elif failure == "insert":
        item = decide(path, e, p, segment)
        cue_scope.decide([item], SOURCE, mixture=np.tile(p, (100, 1)), boundaries=[(2.5, 2.8), (7.2, 7.5)])
        assert not cue_scope.publishable(item)
        return
    elif failure == "missing":
        path = tmp_path/"missing.wav"
    elif failure == "truncated":
        x, _ = sf.read(path, always_2d=True)
        sf.write(path, x[:round(6.9*RATE)], RATE, subtype="FLOAT")
    elif failure == "candidate_hole":
        segment["candidates"] = [[3.2, 6.8]]
    assert not cue_scope.publishable(decide(path, e, p, segment))


def test_short_proof_is_bound_to_exact_span_and_policy(tmp_path):
    path, e, p, segment = setup_case(tmp_path)
    item = decide(path, e, p, segment)
    assert cue_scope.publishable(item)
    item["end_seconds"] -= .1
    item["core_seconds"] = [item["start_seconds"], item["end_seconds"]]
    cue_scope.decide([item], SOURCE, mixture=p)
    assert not cue_scope.publishable(item)


def test_source_continuation_vetoes_a_separator_only_quiet_boundary(tmp_path):
    path, e, p, segment = setup_case(tmp_path)
    p[2, cs.MUSIC_CLASS] = .9
    item = decide(path, e, p, segment)
    assert item["short_music"]["reason"] == "source_music_continues_outside_short"
    assert not cue_scope.publishable(item)


def test_source_edit_silence_can_resolve_overlapping_tag_context(tmp_path):
    path, e, p, segment = setup_case(tmp_path)
    p[7, cs.MUSIC_CLASS] = .9
    segment["end_edge"] = {"kind": "edit_boundary"}
    item = decide(path, e, p, segment)
    assert cue_scope.publishable(item)
    assert item["short_music"]["source_boundaries"]["end"]["authority"] == "source_digital_silence"


def test_edit_child_remeasures_parent_extent_without_promoting_long_parent(tmp_path):
    path, e, p, segment = setup_case(tmp_path)
    segment["reasons"] = ["possible_change_inside"]
    segment["features"].update(seconds=100., inner_split_distance=.17)
    item = decide(path, e, p, segment)
    assert cue_scope.publishable(item)
    assert item["features"]["seconds"] < 10.
    assert item["features"]["inner_split_distance"] == 0.


def test_ordinary_length_segment_is_not_changed(tmp_path):
    path, e, p, segment = setup_case(tmp_path)
    segment.update(start_seconds=0., end_seconds=10.)
    [item] = sc.apply([segment], e, p, path, cs.default_recipe(), 10.)
    assert item is segment


def test_historical_scope_ids_are_unchanged_when_short_route_is_off():
    import json, hashlib
    scope = cue_scope.product_scope()
    legacy = replace(scope, short_cue_minimum_seconds=None, short_cue_required_evidence=())
    payload = legacy.to_dict()
    assert "short_cue_minimum_seconds" not in payload
    assert legacy.scope_id.endswith(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16])


@pytest.mark.parametrize("short", [False, True])
def test_scope_roundtrip_preserves_json_array_evidence_requirements(tmp_path, short):
    import json
    from lss.bgm.supported_scope import SupportedScope, classify_candidate
    path, e, p, segment = setup_case(tmp_path)
    item = decide(path, e, p, segment)
    mapped = cue_scope.candidate(item, 0, SOURCE, {"integrity_passed": 1.})
    if not short:
        mapped.update(start_frame=0, end_frame=30000)
    scope = cue_scope.product_scope()
    restored = SupportedScope(**json.loads(json.dumps(scope.to_dict())))
    assert restored.scope_id == scope.scope_id
    assert classify_candidate(mapped, restored)["disposition"] == "PUBLISHED"


@pytest.mark.parametrize("minimum,required", [(0, ("proof",)), (10., ("proof",)), (2., ()), (None, ("proof",))])
def test_invalid_short_scope_declarations_are_rejected(minimum, required):
    from lss.bgm.supported_scope import ScopeError
    with pytest.raises(ScopeError):
        replace(cue_scope.product_scope(), short_cue_minimum_seconds=minimum, short_cue_required_evidence=required).to_dict()


def off_grid_case(tmp_path):
    """Music from 3.8 s to 7.2 s: the windows at 3 s and 7 s mostly hold the silence around it."""
    t = np.arange(10*RATE)/RATE
    x = .1*np.sin(2*np.pi*220*t)
    x[(t < 3.8) | (t >= 7.2)] = 0
    path = tmp_path/"music.wav"
    sf.write(path, np.column_stack([x, .8*x]), RATE, subtype="FLOAT")
    p = np.zeros((10, 527), np.float32)
    p[4:7, cs.MUSIC_CLASS] = .9
    p[3, cs.MUSIC_CLASS] = p[7, cs.MUSIC_CLASS] = .1
    p[3, 0] = p[7, 0] = 0.
    db = np.full(10, -100.)
    db[3:8] = -30.
    e = cs.WindowEvidence.from_arrays(db, p, np.ones((10, 8)))
    segment = {"start_seconds": 3.9, "end_seconds": 7.1, "core_seconds": [3.9, 7.1],
               "candidates": [[0., 10.]], "stretch": [0., 10.],
               "start_edge": {"kind": cs.EDGE_SILENCE}, "end_edge": {"kind": cs.EDGE_SILENCE}}
    [segment] = cs.decide([segment], e, cs.default_recipe(), source_seconds=10.)
    return path, e, p, segment


def test_off_grid_short_music_is_judged_on_its_whole_windows(tmp_path):
    # Synthetic short music between two longer foreground spans.
    # Music 0.09 and held a complete short cue as "not continuous".
    path, e, p, segment = off_grid_case(tmp_path)
    assert segment["reasons"] == ["shorter_than_wanted_minimum"]
    item = sc.apply([segment], e, p, path, cs.default_recipe(), float(e.count))[0]
    assert item["short_music"]["reason"] != "short_music_not_continuous"
    assert sc.confirmed(item)
    # The re-measured body must not fall back to the straddling windows either.
    assert item["semantic_class"] == cs.WANTED, item["reasons"]
    assert item["core_seconds"] == [item["start_seconds"], item["end_seconds"]]
    cue_scope.decide([item], SOURCE, mixture=p)
    assert cue_scope.publishable(item)


def test_a_gap_inside_whole_windows_still_breaks_continuity(tmp_path):
    path, e, p, segment = off_grid_case(tmp_path)
    p[5, cs.MUSIC_CLASS] = .2
    item = sc.apply([segment], e, p, path, cs.default_recipe(), float(e.count))[0]
    assert item["short_music"]["reason"] == "short_music_not_continuous"
    assert not sc.confirmed(item)


@pytest.mark.parametrize("state, routed", [("unknown", True), ("released_recording_edit", False),
                                           ("related_version_candidate", False)])
def test_reference_checked_short_cue_keeps_the_route_only_when_unexplained(tmp_path, state, routed):
    # With references checked every segment carries a relation; "unknown" (no reference explains it)
    # must not block the short route, any reference relation must.
    path, e, p, segment = setup_case(tmp_path)
    segment["reference_relation"] = {"state": state}
    [item] = sc.apply([segment], e, p, path, cs.default_recipe(), float(e.count))
    assert ("short_music" in item) is routed
    assert sc.confirmed(item) is routed


def test_straddle_flagged_short_cue_is_re_measured_not_excluded(tmp_path):
    # Real ep1 with references: the first decision already said music_not_continuous because of the
    # onset window; the short route must re-measure on whole windows instead of skipping the cue.
    path, e, p, segment = off_grid_case(tmp_path)
    segment = {**segment, "semantic_class": cs.UNCERTAIN,
               "reasons": ["shorter_than_wanted_minimum", "music_not_continuous"]}
    [item] = sc.apply([segment], e, p, path, cs.default_recipe(), float(e.count))
    assert sc.confirmed(item)
    assert item["semantic_class"] == cs.WANTED


def test_straddle_admission_still_rejects_a_real_gap(tmp_path):
    path, e, p, segment = off_grid_case(tmp_path)
    p[5, cs.MUSIC_CLASS] = .2
    segment = {**segment, "semantic_class": cs.UNCERTAIN,
               "reasons": ["shorter_than_wanted_minimum", "music_not_continuous"]}
    [item] = sc.apply([segment], e, p, path, cs.default_recipe(), float(e.count))
    assert item["short_music"]["reason"] == "short_music_not_continuous"
    assert not sc.confirmed(item) and item["semantic_class"] == cs.UNCERTAIN
