import json
from pathlib import Path

import pytest

from scripts.probe_bgm_identity_boundary import (
    IdentityBoundaryProbeError,
    _implied_boundaries,
    build,
)


def _match(reference_id, start, end):
    return {"reference_id": reference_id, "query_start_seconds": start,
            "query_end_seconds": end}


def test_a_boundary_is_read_only_between_two_different_recordings():
    rows = [_match("a", 0.0, 10.0), _match("a", 9.5, 20.0), _match("b", 20.5, 30.0)]

    boundaries = _implied_boundaries(rows)

    assert len(boundaries) == 1
    assert boundaries[0]["before_reference_id"] == "a"
    assert boundaries[0]["after_reference_id"] == "b"
    assert boundaries[0]["earliest_seconds"] == 20.0
    assert boundaries[0]["latest_seconds"] == 20.5
    assert boundaries[0]["seconds"] == pytest.approx(20.25)


def test_the_same_recording_repeated_implies_no_boundary():
    assert _implied_boundaries([_match("a", 0.0, 10.0), _match("a", 30.0, 40.0)]) == []


def test_a_single_match_implies_no_boundary():
    assert _implied_boundaries([_match("a", 0.0, 10.0)]) == []


def _recipe_file(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps({
        "n_fft": 4096, "stft_hop_frames": 1024, "feature_seconds": 1.0,
        "feature_hop_seconds": 0.5, "band_count": 24, "low_hz": 55.0, "high_hz": 8000.0,
        "kernel_seconds": 4.0, "minimum_boundary_gap_seconds": 5.0,
        "novelty_threshold": 0.2, "chroma_weight": 0.5,
        "schema_version": 1, "method": "spectral-chroma-novelty-checkerboard",
    }), encoding="utf-8")
    return path


def _episode(tmp_path, *, publication_state="not_published"):
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "run.json").write_text(json.dumps({
        "promotion_allowed": False, "publication_state": publication_state,
        "source": {"source_id": "test/episode-1"}, "candidates": [],
    }), encoding="utf-8")
    (episode / "identity.json").write_text(json.dumps({
        "type": "episode_identity_annotation", "matches": [],
    }), encoding="utf-8")
    return episode


def test_refuses_to_read_back_a_published_result(tmp_path):
    episode = _episode(tmp_path, publication_state="published")

    with pytest.raises(IdentityBoundaryProbeError, match="固定Development記録"):
        build(episode, _recipe_file(tmp_path), tmp_path / "out",
              tolerance_seconds=2.0, minimum_similarity=0.7)


def test_refuses_an_annotation_of_another_kind(tmp_path):
    episode = _episode(tmp_path)
    (episode / "identity.json").write_text(json.dumps({"type": "something_else"}),
                                           encoding="utf-8")

    with pytest.raises(IdentityBoundaryProbeError, match="一致記録"):
        build(episode, _recipe_file(tmp_path), tmp_path / "out",
              tolerance_seconds=2.0, minimum_similarity=0.7)


def test_the_frozen_plan_keeps_the_declared_limits(tmp_path):
    plan_path = build(_episode(tmp_path), _recipe_file(tmp_path), tmp_path / "out",
                      tolerance_seconds=2.0, minimum_similarity=0.7)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))

    assert plan["promotion_allowed"] is False
    assert plan["holdout_consumed"] is False
    assert plan["tolerance_seconds"] == 2.0
    assert "novelty_is_an_observation_not_a_semantic_cue_boundary" in plan["limits"]
