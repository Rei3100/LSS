"""Published-cue memory as local recording support: a recording LSS published for another episode measures, in
this episode's own source and music, the part the classifier missed; it never uses this episode itself, an
unrelated recording, or the member's audio as output."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from lss.bgm import cue_formation, cue_memory, cue_semantics as cs, identity, memory_support

from tests.test_bgm_cue_memory import OTHER, RATE, RECORDING, _cue, _observe, _residue

SR = identity.SAMPLE_RATE


def _down(x: np.ndarray) -> np.ndarray:
    return resample_poly(x, SR, RATE).astype(np.float32)


def _episode(seconds: float, placements: list[tuple[float, np.ndarray]], seed: int):
    """(source mix, first-pass music) of an episode: music placed at positions, dialogue on top of the mix."""
    music = np.zeros(int(seconds * RATE))
    for at, part in placements:
        music[int(at * RATE):int(at * RATE) + part.size] += part
    mix = music + 6 * _residue(seconds, seed)
    stem = music + 0.5 * _residue(seconds, seed + 1)
    return _down(mix), _down(stem)


def _memory(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    # Another episode published the whole recording (0-60 s of RECORDING) and an unrelated cue.
    paths = [_cue(tmp_path, "ep01", 100.0, RECORDING, seed=21),
             _cue(tmp_path, "ep01", 400.0, OTHER, seed=22)]
    _observe(state, *paths)
    from lss.bgm import current_best

    items = [cue_memory.published_item(path) for path in paths]
    sha = items[0]["source_sha256"]
    current_best._write_json_atomic(current_best.record_path(state, sha), {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION, "source_sha256": sha,
        "current": {"files": [{"public_path": item["path"], "pcm_sha256": item["pcm_sha256"],
                                "cue_key": item["cue_key"], "start_seconds": item["start_seconds"],
                                "end_seconds": item["start_seconds"] + item["duration_seconds"],
                                "run_key": "fixture-run", "recipe_id": "fixture-recipe"} for item in items],
                    "pinned": False, "evaluated_runs": ["fixture-run"]}, "history": []})
    return state


def test_memory_measures_the_head_the_classifier_missed(tmp_path):
    state = _memory(tmp_path)
    # This episode plays RECORDING 10-50 s at 200 s; the classifier heard only from 206 s (a quiet intro missed).
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 206.0, "end_seconds": 240.0}], state, "f" * 64)
    observed = [r for r in receipt["records"] if r["state"] == "component_observed"]
    assert receipt["state"] == "checked" and len(observed) == 1
    record = observed[0]
    assert record["reference_type"] == "published_cue_memory" and record["donor_authority"] == "none"
    spans = record["spans"]
    assert min(s["start_seconds"] for s in spans) < 201.0          # presence measured from the true start
    assert max(s["end_seconds"] for s in spans) > 238.0
    assert abs(record["onset"]["predicted_seconds"] - 200.0) < 1.5
    assert record["render_trace"]["fraction"]                       # this episode's own music corroborates


def test_an_episode_never_supports_itself(tmp_path):
    state = _memory(tmp_path)
    sha = cue_memory.published_item(next((tmp_path / "ep01 - BGM").glob("*.flac")))["source_sha256"]
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 206.0, "end_seconds": 240.0}], state, sha)
    assert (receipt["records"] == [] and receipt["peer_query_component_measurements"] == []
            and receipt["state"] == "no_published_cues_of_other_sources")


def test_unrelated_music_gets_no_support(tmp_path):
    state = _memory(tmp_path)
    fresh = np.random.default_rng(5).standard_normal(int(40 * RATE)) * 0.05
    mix, stem = _episode(300.0, [(200.0, fresh)], seed=33)
    loaded = []
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 200.0, "end_seconds": 240.0}], state,
                                      "f" * 64, query_interference_loader=lambda: loaded.append(True) or {})
    assert [r for r in receipt["records"] if r["state"] == "component_observed"] == []
    assert receipt["peer_query_component_measurements"] == [] and loaded == []


def test_current_best_pcm_mismatch_holds_the_peer_diagnostic(tmp_path):
    state = _memory(tmp_path)
    from lss.bgm import current_best

    item = cue_memory.read(state)["members"]
    member = next(value for value in item.values() if value["source_name"] == "ep01"
                  and value["duration_seconds"] == 60.0)
    record = current_best.read(state, member["source_sha256"])
    row = next(row for row in record["current"]["files"] if row["cue_key"] == member["cue_key"])
    row["pcm_sha256"] = "0" * 64
    current_best._write_json_atomic(current_best.record_path(state, member["source_sha256"]), record)
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 206.0, "end_seconds": 240.0}], state,
                                      "f" * 64, query_interference={"speech": np.zeros_like(stem),
                                                                    "effects": np.zeros_like(stem)})
    peer = next(row for row in receipt["peer_query_component_measurements"]
                if row["peer_member"]["asset_id"] == member["asset_id"])
    assert peer["state"] == "held_not_current_best_member"
    assert peer["current_best_membership"] is None and peer["authority"]["donor"] == "none"


def test_current_best_binding_selects_each_of_two_peer_cues_from_one_source(tmp_path):
    state = _memory(tmp_path)
    members = [value for value in cue_memory.read(state)["members"].values() if value["source_name"] == "ep01"]
    assert len(members) == 2 and members[0]["source_sha256"] == members[1]["source_sha256"]
    cache = {}
    rows = [memory_support._current_best_member(state, member, cache) for member in members]
    assert [row["cue_key"] for row in rows] == [member["cue_key"] for member in members]
    assert [row["pcm_sha256"] for row in rows] == [member["pcm_sha256"] for member in members]


def test_retired_peer_audio_is_not_nominated_for_component_measurement(tmp_path):
    state = _memory(tmp_path)
    member = next(value for value in cue_memory.read(state)["members"].values()
                  if value["source_name"] == "ep01")
    Path(member["path"]).unlink()
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 206.0, "end_seconds": 240.0}], state,
                                      "f" * 64, query_interference={"speech": np.zeros_like(stem),
                                                                    "effects": np.zeros_like(stem)})
    assert all(row["peer_member"]["asset_id"] != member["asset_id"]
               for row in receipt["peer_query_component_measurements"])


def test_verified_peer_exposes_typed_non_authoritative_full_window_component_measurement(tmp_path):
    """A real autonomous peer match gets a query-component diagnostic, never release/donor authority."""
    state = _memory(tmp_path)
    mix, music = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    speech = np.zeros_like(music)
    effects = np.zeros_like(music)
    receipt = memory_support.nominate(
        mix, music, [{"start_seconds": 206.0, "end_seconds": 240.0}], state, "f" * 64,
        query_interference={"speech": speech, "effects": effects})

    measured = receipt["peer_query_component_measurements"]
    assert measured and measured[0]["state"] == "measured_peer_query_components_held_no_quality_certificate"
    assert measured[0]["candidate_status"] == "HELD"
    assert measured[0]["candidate_kind"] == "current_best_peer"
    assert measured[0]["query_source_sha256"] == "f" * 64
    assert measured[0]["requested_query_window_seconds"] == [206.0, 240.0]
    assert len(measured[0]["query_analysis_pcm_sha256"]) == 64
    assert measured[0]["peer_member"]["source_sha256"] != "f" * 64
    assert measured[0]["peer_member"]["pcm_sha256"]
    assert measured[0]["query_components"] == ["music", "speech", "effects"]
    assert measured[0]["authority"] == {
        "identity": "none", "availability": "none", "absence": "none", "donor": "none"}
    assert measured[0]["peer_source_quality_certificate"] is None
    binding = measured[0]["peer_waveform_binding"]
    assert binding["validated"] and binding["state"] == "bound_to_currentbest_original_pcm"
    assert binding["original_pcm_recheck"]["pcm_sha256"] == measured[0]["peer_member"]["pcm_sha256"]
    mono = np.asarray(identity.decode_mono(Path(measured[0]["peer_member"]["path"])), dtype=np.float32)
    assert binding["decoded_measurement_reference"] == {
        "sample_rate": SR, "dtype": "float32", "sample_count": mono.size,
        "sha256": hashlib.sha256(np.ascontiguousarray(mono).tobytes()).hexdigest()}
    assert measured[0]["cells"] and all("joint" in cell and "novelty" in cell for cell in measured[0]["cells"])
    first_reference_sample = measured[0]["cells"][0]["reference_input_samples_11025"][0]
    assert abs(first_reference_sample / SR - 16.0) < 0.01
    assert all(cell["inside"] for cell in measured[0]["cells"])
    # Existing component-support path remains the same authority and data shape.
    observed = [row for row in receipt["records"] if row["state"] == "component_observed"]
    assert observed and observed[0]["reference_type"] == "published_cue_memory"
    assert observed[0]["donor_authority"] == "none"


def test_peer_measurement_holds_if_source_file_changes_during_decode_without_removing_local_result(tmp_path):
    state = _memory(tmp_path)
    mix, music = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    member_path = next(Path(value["path"]) for value in cue_memory.read(state)["members"].values()
                       if value["source_name"] == "ep01" and value["duration_seconds"] == 60.0)
    original_loader = identity.decode_mono
    changed = False
    target_loads = 0

    def mutating_loader(path):
        nonlocal changed, target_loads
        samples = original_loader(Path(path))
        if Path(path) == member_path:
            target_loads += 1
        if Path(path) == member_path and target_loads == 2:
            changed = True
            stat = member_path.stat()
            import os

            os.utime(member_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        return samples

    receipt = memory_support.nominate(
        mix, music, [{"start_seconds": 206.0, "end_seconds": 240.0}], state, "f" * 64,
        loader=mutating_loader,
        query_interference={"speech": np.zeros_like(music), "effects": np.zeros_like(music)})
    peers = receipt["peer_query_component_measurements"]
    assert changed
    assert peers and peers[0]["state"] == "held_peer_waveform_identity_unverified"
    assert peers[0]["peer_waveform_binding"]["state"] == "held_file_or_currentbest_changed"
    assert "cells" not in peers[0]
    # The optional diagnostic cannot erase the already existing local component-support result.
    assert any(row["state"] == "component_observed" for row in receipt["records"])


def test_peer_component_measurement_holds_layered_audio_and_missing_components(tmp_path):
    state = _memory(tmp_path)
    music_native = np.zeros(int(300 * RATE), np.float32)
    part = RECORDING[int(10 * RATE):int(50 * RATE)]
    music_native[int(200 * RATE):int(200 * RATE) + part.size] = part
    # An independent wanted layer remains in Music; the measurement is evidence only and cannot call the peer clean.
    music_native[int(200 * RATE):int(200 * RATE) + part.size] += np.random.default_rng(94).standard_normal(part.size).astype(np.float32) * .025
    mix = music_native + 6 * _residue(300.0, 41)
    speech = np.zeros_like(music_native)
    effects = np.zeros_like(music_native)
    receipt = memory_support.nominate(
        _down(mix), _down(music_native), [{"start_seconds": 206.0, "end_seconds": 240.0}], state, "b" * 64,
        query_interference={"speech": _down(speech), "effects": _down(effects)})
    assert receipt["peer_query_component_measurements"][0]["state"] == \
        "measured_peer_query_components_held_no_quality_certificate"
    assert receipt["peer_query_component_measurements"][0]["authority"]["donor"] == "none"

    missing = memory_support.nominate(
        _down(mix), _down(music_native), [{"start_seconds": 206.0, "end_seconds": 240.0}], state, "b" * 64,
        query_interference={"speech": _down(speech)})
    assert missing["peer_query_component_measurements"][0]["state"] == "held_missing_query_component"
    assert missing["peer_query_component_measurements"][0]["missing_components"] == ["effects"]


def test_peer_measurement_covers_full_fractional_candidate_and_marks_unisolated_edges(tmp_path):
    state = _memory(tmp_path)
    mix, music = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, music, [{"start_seconds": 205.25, "end_seconds": 240.5}], state,
                                      "e" * 64, query_interference={"speech": np.zeros_like(music),
                                                                    "effects": np.zeros_like(music)})
    peer = receipt["peer_query_component_measurements"][0]
    assert peer["requested_query_window_seconds"] == [205.25, 240.5]
    assert peer["requested_measurement_scope"]["cell_seconds"] == [205, 241]
    assert peer["requested_measurement_scope"]["unknown_query_edge_intervals_seconds"] == [[205.0, 205.25],
                                                                                           [240.5, 241.0]]
    assert peer["requested_measurement_scope"]["edge_cells_without_fractional_isolation"] == [205, 240]
    assert [row["query_cell_seconds"] for row in peer["cells"]] == list(range(205, 241))
    assert peer["matched_alignment"]["matcher_fragment_seconds"][1] < 240.5
    assert any(not row["matched_fragment_covers_cell"] for row in peer["cells"])
    for component, bounds in peer["component_window_sample_ranges"].items():
        samples = {"query": mix, "music": music, "speech": np.zeros_like(music),
                   "effects": np.zeros_like(music)}[component]
        first, last = bounds
        digest = hashlib.sha256(np.ascontiguousarray(samples[first:last]).tobytes()).hexdigest()
        assert digest == peer["component_window_pcm_sha256"][component]


def test_peer_measurement_marks_short_reference_fir_edges_unknown(tmp_path):
    state = tmp_path / "short-state"
    peer = _cue(tmp_path, "ep-short", 0.0, RECORDING[:int(20 * RATE)], seed=23)
    _observe(state, peer)
    from lss.bgm import current_best

    item = cue_memory.published_item(peer)
    current_best._write_json_atomic(current_best.record_path(state, item["source_sha256"]), {
        "type": current_best.RECORD_TYPE, "schema_version": current_best.SCHEMA_VERSION,
        "source_sha256": item["source_sha256"],
        "current": {"files": [{"public_path": item["path"], "pcm_sha256": item["pcm_sha256"],
                                "cue_key": item["cue_key"], "start_seconds": 0.0,
                                "end_seconds": 20.0, "run_key": "short-run", "recipe_id": "fixture"}],
                    "pinned": False, "evaluated_runs": ["short-run"]}, "history": []})
    mix, music = _episode(300.0, [(200.0, RECORDING[:int(30 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, music, [{"start_seconds": 200.0, "end_seconds": 230.0}], state,
                                      "d" * 64, query_interference={"speech": np.zeros_like(music),
                                                                    "effects": np.zeros_like(music)})
    peer_rows = receipt["peer_query_component_measurements"]
    assert peer_rows and peer_rows[0]["matched_alignment"]["matcher_fragment_seconds"][1] < 230.0
    outside = [row for row in peer_rows[0]["cells"] if row["state"] == "unknown_reference_fir_outside_peer"]
    assert outside and all(row["joint"] is None and row["inside"] is False for row in outside)


def test_standard_application_memory_boundary_loads_interference_lazily_and_preserves_components(tmp_path,
                                                                                                  monkeypatch):
    from lss import application
    from lss.harness.fetch import file_sha256

    state = _memory(tmp_path)
    run_dir = state / "bgm-candidates" / "query" / "run"
    run_dir.mkdir(parents=True)
    source = tmp_path / "query.mkv"
    source.write_bytes(b"finished source identity")
    paths = {name: tmp_path / f"{name}.wav" for name in ("analysis", "music", "speech", "effects")}
    for path in paths.values():
        path.write_bytes(b"test decoder handle")
    mix, music = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    speech, effects = np.zeros_like(music), np.zeros_like(music)
    source_audio = {member["path"]: _down(RECORDING) for member in cue_memory.read(state)["members"].values()}
    source_audio.update({str(paths["analysis"]): mix, str(paths["music"]): music,
                         str(paths["speech"]): speech, str(paths["effects"]): effects})
    decoded = []
    from lss.bgm import identity

    monkeypatch.setattr(identity, "decode_mono", lambda path: decoded.append(str(Path(path))) or
                        source_audio[str(Path(path))])
    monkeypatch.setattr(identity, "decode_stereo_window",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("optional stereo unavailable")))
    sha = file_sha256(source)
    baseline = memory_support.nominate(mix, music, [{"start_seconds": 206.0, "end_seconds": 240.0}], state, sha,
                                       loader=lambda path: source_audio[str(Path(path))], source_name="query",
                                       query_interference={"speech": speech, "effects": effects})
    decoded.clear()
    normal = application._bgm_memory_support(paths["analysis"], paths["music"],
                                             [{"start_seconds": 206.0, "end_seconds": 240.0}], run_dir,
                                             source, [(195.0, 245.0)], speech=paths["speech"],
                                             effects=paths["effects"])
    assert normal["peer_query_component_measurements"][0]["authority"]["availability"] == "none"
    assert normal["peer_query_component_measurements"][0]["state"] == \
        "measured_peer_query_components_held_no_quality_certificate"
    assert normal["records"] == baseline["records"]
    assert decoded.count(str(paths["speech"])) == decoded.count(str(paths["effects"])) == 1


def test_no_memory_is_the_single_episode_result(tmp_path):
    mix, stem = _episode(60.0, [], seed=34)
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 0.0, "end_seconds": 30.0}], tmp_path, "f" * 64)
    assert receipt["records"] == []


def test_cue_formation_recovers_the_head_and_hole_from_memory_records(tmp_path):
    """The records drive the existing component rules: head to the measured start, classifier hole reconnected."""
    state = _memory(tmp_path)
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    seconds = 300
    # Classifier: music 206-221 and 224-240 (hole 221-224), nothing before 206.
    level = np.full(seconds, -80.0, np.float32)
    probability = np.zeros((seconds, 527), np.float32)
    for low, high in ((206, 221), (224, 240)):
        level[low:high] = -20.0
        probability[low:high, cs.MUSIC_CLASS] = 0.9
    rng = np.random.default_rng(0)
    embeddings = np.tile(rng.standard_normal(16).astype(np.float32), (seconds, 1))
    evidence = cs.WindowEvidence.from_arrays(level, probability, embeddings)
    recipe = cs.default_recipe()
    candidates = [(195.0, 245.0)]
    base, _ = cue_formation.segment(candidates, evidence, recipe)
    receipt = memory_support.nominate(mix, stem, base, state, "f" * 64)
    components = {"records": receipt["records"], "onsets": []}
    parts, record = cue_formation.segment(candidates, evidence, recipe, components=components)
    states = [row["state"] for row in record["observations"]]
    assert "recording_component_recovers_head" in states
    assert len(parts) == 1 and parts[0]["start_seconds"] < 201.0 and parts[0]["end_seconds"] >= 239.0
    assert len(base) >= 1 and min(p["start_seconds"] for p in base) >= 205.0


def _classifier(seconds: int, runs):
    level = np.full(seconds, -80.0, np.float32)
    probability = np.zeros((seconds, 527), np.float32)
    for low, high in runs:
        level[low:high] = -20.0
        probability[low:high, cs.MUSIC_CLASS] = 0.9
    embeddings = np.tile(np.random.default_rng(0).standard_normal(16).astype(np.float32), (seconds, 1))
    return cs.WindowEvidence.from_arrays(level, probability, embeddings)


def test_presence_before_the_materialized_stretch_never_holds_the_cue(tmp_path):
    """Music that began before the candidate stretch: the head is recovered to the stretch start with a silence
    edge, never a candidate edge (that would hold a cue published without memory)."""
    state = _memory(tmp_path)
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    evidence = _classifier(300, [(206, 240)])
    recipe = cs.default_recipe()
    candidates = [(203.0, 245.0)]
    base, _ = cue_formation.segment(candidates, evidence, recipe)
    receipt = memory_support.nominate(mix, stem, base, state, "f" * 64, stretches=[(203.0, 245.0)])
    observed = [r for r in receipt["records"] if r["state"] == "component_observed"]
    assert observed and observed[0]["presence_before_materialized_stretch"]
    parts, _ = cue_formation.segment(candidates, evidence, recipe, components={"records": receipt["records"],
                                                                               "onsets": []})
    assert len(parts) == 1 and 203.0 <= parts[0]["start_seconds"] < 204.0
    assert parts[0]["start_edge"]["kind"] == cs.EDGE_SILENCE
    decided = cs.decide(parts, evidence, recipe, source_seconds=300.0)
    assert decided[0]["semantic_class"] == cs.WANTED, decided[0]["reasons"]


def test_another_file_of_the_same_episode_never_supports_it(tmp_path):
    state = _memory(tmp_path)
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 206.0, "end_seconds": 240.0}], state, "f" * 64,
                                      source_name="ep01")
    assert receipt["records"] == []


def _record(identity: str, low: float, high: float, onset: float | None = None) -> dict:
    return {"state": "component_observed", "reference_type": "published_cue_memory", "music_corroborated": True,
            "reference_sha256": identity, "onset": {"track": f"memory:{identity}", "predicted_seconds": onset or low},
            "spans": [{"start_seconds": low, "end_seconds": high}]}


def _parts(*bounds, stretch=(90.0, 420.0)):
    edge = {"kind": cs.EDGE_CHANGE, "distance": 0.3}
    return [{"start_seconds": a, "end_seconds": b, "core_seconds": [a, b], "start_edge": dict(edge),
             "end_edge": dict(edge), "stretch": list(stretch), "candidates": [stretch]} for a, b in bounds]


def test_one_recording_on_one_clock_rejoins_a_split_it_plays_through():
    evidence = _classifier(450, [(100, 160)])
    observations = []
    joined = memory_support.join_covered(_parts((100.0, 130.0), (130.0, 160.0)), [_record("g1", 100.0, 160.0)],
                                         evidence, cs.default_recipe(), observations=observations)
    assert [(p["start_seconds"], p["end_seconds"]) for p in joined] == [(100.0, 160.0)]
    assert observations[0]["state"] == "memory_continuous_recording_joined"


def test_no_join_when_another_recording_comes_in_or_the_span_does_not_reach_both_parts():
    evidence = _classifier(450, [(100, 160)])
    recipe = cs.default_recipe()
    incoming = [_record("g1", 100.0, 160.0), _record("g2", 129.5, 160.0)]
    assert len(memory_support.join_covered(_parts((100.0, 130.0), (130.0, 160.0)), incoming, evidence, recipe)) == 2
    short = [_record("g1", 100.0, 131.0)]
    assert len(memory_support.join_covered(_parts((100.0, 130.0), (130.0, 160.0)), short, evidence, recipe)) == 2


def test_a_join_that_would_hold_a_wanted_cue_keeps_the_split():
    evidence = _classifier(450, [(100, 400)])
    observations = []
    parts = _parts((100.0, 250.0), (250.0, 400.0))
    kept = memory_support.join_covered(parts, [_record("g1", 100.0, 400.0)], evidence, cs.default_recipe(),
                                       observations=observations)
    assert len(kept) == 2 and observations[0]["state"] == "memory_join_kept_split_without_lowering"


def test_copies_of_one_memory_group_are_one_recording(tmp_path):
    """Two published copies of the same recording (one group) give records with one recording identity, so they
    never compete with each other in cue formation."""
    state = tmp_path / "state"
    _observe(state, _cue(tmp_path, "ep01", 100.0, RECORDING, seed=21))
    _observe(state, _cue(tmp_path, "ep02", 50.0, RECORDING[int(5 * RATE):int(55 * RATE)], seed=23))
    mix, stem = _episode(300.0, [(200.0, RECORDING[int(10 * RATE):int(50 * RATE)])], seed=31)
    receipt = memory_support.nominate(mix, stem, [{"start_seconds": 206.0, "end_seconds": 240.0}], state, "f" * 64)
    identities = {r["reference_sha256"] for r in receipt["records"] if r["state"] == "component_observed"}
    members = {r["member"]["source_name"] for r in receipt["records"] if r["state"] == "component_observed"}
    assert members == {"ep01", "ep02"} and len(identities) == 1 and next(iter(identities)).startswith("cue-memory-group:")


def test_centred_dialogue_that_masks_mono_is_cancelled_by_the_stereo_difference(tmp_path):
    state = _memory(tmp_path)
    seconds = 300.0
    music = np.zeros(int(seconds * RATE))
    part = RECORDING[int(10 * RATE):int(50 * RATE)]
    music[int(200 * RATE):int(200 * RATE) + part.size] = part
    dialogue = 40 * _residue(seconds, 41)                 # loud, identical in both channels (centred)
    source = np.stack([music + dialogue, 0.4 * music + dialogue], axis=1)
    stem = np.stack([music, 0.4 * music], axis=1) + 0.2 * np.stack([dialogue, dialogue], axis=1)
    down = lambda x: np.stack([_down(x[:, 0]), _down(x[:, 1])], axis=1)
    source_sr, stem_sr = down(source), down(stem)
    member = next((tmp_path / "ep01 - BGM").glob("*01m40s*.flac"))

    def stereo(start, frames):
        return source_sr[start:start + frames], stem_sr[start:start + frames]

    receipt = memory_support.nominate(source_sr.mean(1), stem_sr.mean(1), [{"start_seconds": 206.0,
                                      "end_seconds": 240.0}], state, "f" * 64, stereo=stereo,
                                      stereo_loader=lambda path, frames: identity.decode_stereo_window(member, 0, frames))
    side = [r for r in receipt["records"] if r.get("projection") == "stereo_difference"]
    assert side and min(s["start_seconds"] for s in side[0]["spans"]) < 202.0


def test_never_lower_keeps_untouched_parts_as_they_were():
    evidence = _classifier(450, [(100, 160), (200, 260), (262, 270)])
    recipe = cs.default_recipe()
    first = _parts((100.0, 160.0), (200.0, 260.0), (262.0, 270.0))
    first[0]["core_seconds"] = [101.0, 159.0]
    # Shared component rules widened the untouched part's core; memory joined 200-260 and 262-270 (still WANTED).
    formed = [dict(first[0], core_seconds=[100.0, 160.0]), dict(first[1], end_seconds=270.0, core_seconds=[200.0, 270.0])]
    kept = memory_support.never_lower(first, formed, evidence, recipe)
    assert kept[0] is first[0]
    assert [(p["start_seconds"], p["end_seconds"]) for p in kept] == [(100.0, 160.0), (200.0, 270.0)]


def test_never_lower_reverts_a_region_that_would_lose_wanted_music():
    evidence = _classifier(450, [(200, 260), (268, 275)])
    recipe = cs.default_recipe()
    first = _parts((200.0, 260.0), (268.0, 275.0))
    assert cs.decide(first, evidence, recipe, source_seconds=450.0)[0]["semantic_class"] == cs.WANTED
    formed = [dict(first[0], end_seconds=275.0, core_seconds=[200.0, 275.0])]     # music_not_continuous
    observations = []
    kept = memory_support.never_lower(first, formed, evidence, recipe, observations=observations)
    assert [(p["start_seconds"], p["end_seconds"]) for p in kept] == [(200.0, 260.0), (268.0, 275.0)]
    assert observations[0]["state"] == "memory_change_reverted_without_lowering"


def test_memory_music_windows_never_reach_parts_memory_did_not_move():
    evidence = _classifier(450, [(100, 160), (200, 260)])
    recipe = cs.default_recipe()
    first = _parts((100.0, 160.0), (200.0, 260.0))
    moved = [first[0], dict(first[1], start_seconds=198.0, core_seconds=[198.0, 260.0])]
    windows = np.zeros(450, bool)
    windows[90:270] = True
    parts, kept = memory_support.final_check(first, moved, windows, evidence, recipe)
    assert not kept[100:160].any() and kept[198:260].all()


def test_a_final_decision_that_loses_wanted_music_returns_the_first_pass_without_windows(monkeypatch):
    evidence = _classifier(450, [(100, 160)])
    recipe = cs.default_recipe()
    first = _parts((100.0, 160.0))
    moved = [dict(first[0], start_seconds=95.0, core_seconds=[95.0, 160.0])]
    real = cs.decide

    def demoting(items, *args, additional_music_windows=None, **kwargs):
        out = real(items, *args, additional_music_windows=additional_music_windows, **kwargs)
        if additional_music_windows is not None:
            out = [dict(item, semantic_class=cs.UNCERTAIN, reasons=["singing_present"]) for item in out]
        return out

    monkeypatch.setattr(cs, "decide", demoting)
    observations = []
    parts, kept = memory_support.final_check(first, moved, np.ones(450, bool), evidence, recipe,
                                             observations=observations)
    assert parts == first and kept is None
    assert observations[0]["state"] == "memory_result_reverted_to_first_pass"


def test_memory_guards_compare_against_stereo_supported_first_pass():
    evidence = _classifier(450, [(200, 210), (214, 230)])
    recipe = cs.default_recipe()
    first = _parts((200.0, 230.0))
    stereo = np.zeros(450, bool)
    stereo[210:214] = True
    assert cs.decide(first, evidence, recipe, source_seconds=450.)[0]["semantic_class"] != cs.WANTED
    assert cs.decide(first, evidence, recipe, source_seconds=450.,
                     additional_music_windows=stereo)[0]["semantic_class"] == cs.WANTED
    # The optional memory moved the cue into a much longer unconfirmed region.
    moved = [dict(first[0], start_seconds=180., core_seconds=[180., 230.])]
    observations = []
    kept = memory_support.never_lower(first, moved, evidence, recipe, additional_music_windows=stereo,
                                      baseline_music_windows=stereo, observations=observations)
    assert kept == first
    assert observations[0]["state"] == "memory_change_reverted_without_lowering"
    observations = []
    parts, windows = memory_support.final_check(first, moved, np.zeros(450, bool), evidence, recipe,
                                               baseline_music_windows=stereo, observations=observations)
    assert parts == first and np.array_equal(windows, stereo)
    assert observations[0]["state"] == "memory_result_reverted_to_first_pass"
    # Scoping memory windows away from an unchanged part still retains its independent stereo observations.
    parts, windows = memory_support.final_check(first, first, np.ones(450, bool), evidence, recipe,
                                               baseline_music_windows=stereo)
    assert parts == first and np.array_equal(windows, stereo)
