"""録音一致の前後にある短い未知区間を消さず、偽のゼロ件を返さない。"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from lss.bgm import identity, identity_run
from tests.test_bgm_identity import _exact


@pytest.mark.parametrize("start,end", [(9.0, 30.0), (10.0, 31.0), (9.9999, 30.0001)])
def test_short_edges_stay_pending_and_prevent_a_no_novel_asset_result(start, end):
    candidate = {"candidate_id": "cue", "start_seconds": start, "end_seconds": end,
                 "identity_state": "unresolved", "audio_version_id": None}
    before = copy.deepcopy(candidate)
    pieces = identity.split_candidates_at_matches([candidate], [_exact("A", 10.0, 30.0)])
    pending = [p for p in pieces if p["identity_state"] == "unresolved"]
    assert pending and all(p["short_unresolved_remainder"] for p in pending)
    assert all(p["end_seconds"] > p["start_seconds"] for p in pieces)
    assert pieces[0]["start_seconds"] == start and pieces[-1]["end_seconds"] == end
    assert all(a["end_seconds"] == b["start_seconds"] for a, b in zip(pieces, pieces[1:]))
    decisions = identity.novelty_decisions(pieces)
    assert {d["decision"] for d in decisions} == {"known_recording_exists", "pending"}
    assert identity.episode_zero_reason(decisions) is None
    assert candidate == before


def test_exact_coverage_can_still_produce_a_valid_no_novel_asset_result():
    pieces = identity.split_candidates_at_matches(
        [{"candidate_id": "cue", "start_seconds": 10.0, "end_seconds": 30.0}], [_exact("A", 10.0, 30.0)])
    assert len(pieces) == 1 and "short_unresolved_remainder" not in pieces[0]
    assert identity.episode_zero_reason(identity.novelty_decisions(pieces)) == "no_novel_asset"


def test_split_preserves_all_original_time_even_with_overlapping_known_and_recurring_regions():
    rng = np.random.default_rng(7)
    for _ in range(40):
        matches = [_exact("A", float(s), float(s + length))
                   for s, length in zip(rng.uniform(0, 80, 8), rng.uniform(6, 25, 8))]
        recurring = [(float(s), float(s + length), {"other_episode": "ep2"})
                     for s, length in zip(rng.uniform(0, 80, 5), rng.uniform(6, 25, 5))]
        pieces = identity.split_candidates_at_matches(
            [{"candidate_id": "cue", "start_seconds": 0.0, "end_seconds": 100.0}],
            matches, recurring=recurring)
        assert pieces[0]["start_seconds"] == 0 and pieces[-1]["end_seconds"] == 100
        assert all(a["end_seconds"] == b["start_seconds"] for a, b in zip(pieces, pieces[1:]))
        assert all(p["end_seconds"] > p["start_seconds"] for p in pieces)


def test_episode_annotation_keeps_short_unknown_edges_on_the_media_timeline(tmp_path, monkeypatch):
    signal = tmp_path / "signal.wav"
    signal.write_bytes(b"verified signal")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"verified reference")
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({
        "status": identity_run.ROUTED_STATUS,
        "source": {"sha256": "a" * 64, "audio_timeline_origin": {"offset_seconds": 0.007}},
        "rendering": {"signal": {"path": str(signal), "sha256": identity_run._sha256(signal)}},
        "candidates": [{"asset": {"candidate_asset_id": "cue"}, "origin": {
            "separation_recipe_id": "renderer", "activity_recipe_id": "activity",
            "start_frame": 9 * 11025, "end_frame": 31 * 11025, "sample_rate": 11025}}],
    }), encoding="utf-8")
    original = run_path.read_bytes()
    monkeypatch.setattr(identity.ReferenceIndex, "build", lambda *a, **k: object())
    monkeypatch.setattr(identity, "match", lambda *a, **k: [_exact("A", 10.007, 30.007)])
    record = json.loads(identity_run.annotate_run(
        run_path, [reference], loader=lambda p: np.ones(11025, dtype=np.float32)).read_text("utf-8"))
    [activity] = record["activity_runs"]
    assert activity["zero_reason"] is None
    assert [d["decision"] for d in activity["novelty_decisions"]] == [
        "pending", "known_recording_exists", "pending"]
    assert activity["candidates"][0]["start_seconds"] == 9.007
    assert activity["candidates"][-1]["end_seconds"] == 31.007
    assert run_path.read_bytes() == original
