"""分けた子候補から、元候補の同一性の状態を決める。一部だけの一致で既知にしない。"""
from __future__ import annotations

import pytest

from lss.bgm.identity_candidates import candidate_identity_states


def view(*candidates: dict) -> dict:
    return {"activity_runs": [{"recipe_id": "renderer", "candidates": list(candidates)}]}


def child(name: str, start: float, end: float, state: str = "unresolved", *,
          parent: str | None = None, version: str | None = None) -> dict:
    record = {"candidate_id": name, "start_seconds": start, "end_seconds": end,
              "identity_state": state}
    if parent is not None:
        record["derived_from"] = parent
    if version is not None:
        record["audio_version_id"] = version
    return record


def test_a_candidate_fully_covered_by_one_known_recording_is_known():
    states = candidate_identity_states(view(child("cue", 10.0, 40.0, "known_exact", version="sha256:a")))
    assert states["cue"]["identity_state"] == "known_exact"
    assert states["cue"]["known_exact_fraction"] == 1.0
    assert states["cue"]["audio_version_ids"] == ["sha256:a"]


def test_a_candidate_covered_by_consecutive_known_children_is_known():
    states = candidate_identity_states(view(
        child("cue/00", 10.0, 25.0, "known_exact", parent="cue", version="sha256:a"),
        child("cue/01", 25.0, 40.0, "known_exact", parent="cue", version="sha256:b")))
    assert states["cue"]["identity_state"] == "known_exact"
    assert states["cue"]["audio_version_ids"] == ["sha256:a", "sha256:b"]
    assert states["cue"]["children"] == ["cue/00", "cue/01"]


@pytest.mark.parametrize("known_end,fraction", [(25.0, 0.5), (37.0, 0.9), (10.2, 0.0067)])
def test_a_partly_matched_candidate_stays_unresolved_with_its_covered_fraction(known_end, fraction):
    states = candidate_identity_states(view(
        child("cue/00", 10.0, known_end, "known_exact", parent="cue", version="sha256:a"),
        child("cue/01", known_end, 40.0, "unresolved", parent="cue")))
    row = states["cue"]
    assert row["identity_state"] == "unresolved"
    assert row["known_exact_fraction"] == pytest.approx(fraction, abs=0.001)
    assert row["candidate_seconds"] == 30.0
    # 既知の裏付けは残す。候補を捨てる理由にも、まとめる理由にもしない。
    assert row["audio_version_ids"] == ["sha256:a"]


def test_a_gap_just_inside_the_tolerance_is_still_known():
    states = candidate_identity_states(view(
        child("cue/00", 0.0, 9.98, "known_exact", parent="cue"),
        child("cue/01", 9.98, 10.0, "unresolved", parent="cue")))
    assert states["cue"]["identity_state"] == "known_exact"
    strict = candidate_identity_states(view(
        child("cue/00", 0.0, 9.98, "known_exact", parent="cue"),
        child("cue/01", 9.98, 10.0, "unresolved", parent="cue")), tolerance_seconds=0.0)
    assert strict["cue"]["identity_state"] == "unresolved"


def test_overlapping_children_are_not_counted_twice():
    states = candidate_identity_states(view(
        child("cue/00", 0.0, 20.0, "known_exact", parent="cue"),
        child("cue/01", 10.0, 20.0, "known_exact", parent="cue"),
        child("cue/02", 20.0, 30.0, "unresolved", parent="cue")))
    row = states["cue"]
    assert row["candidate_seconds"] == 30.0 and row["known_exact_seconds"] == 20.0
    assert row["identity_state"] == "unresolved"


def test_candidates_without_any_recording_match_stay_unresolved():
    states = candidate_identity_states(view(child("cue", 0.0, 30.0)))
    assert states["cue"] == {"identity_state": "unresolved", "known_exact_seconds": 0.0,
                             "candidate_seconds": 30.0, "known_exact_fraction": 0.0,
                             "audio_version_ids": [], "children": ["cue"]}
