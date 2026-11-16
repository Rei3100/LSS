"""注釈Stageが、録音一致から抽出元を選ぶ順位付けを持つ。音声は使わない。"""
from __future__ import annotations

import pytest

from lss.bgm import identity, identity_run, occurrence_quality


def _match(reference_sha256, start, end, explained, state="known_exact"):
    return identity.RecordingMatch(
        reference_id="ref", reference_path=f"/refs/{reference_sha256}.flac",
        reference_sha256=reference_sha256,
        query_start_seconds=start, query_end_seconds=end, reference_start_seconds=0.0,
        offset_seconds=0.0, votes=40, similarity=0.9, null_mean=0.1, null_std=0.01, z=9.0,
        measured_windows=4, max_window_drift_seconds=0.0,
        waveform_explained=explained, waveform_null_max=0.05, waveform_blocks=5,
        lag_samples=0, alternative_alignments=0, identity_state=state)


def test_view_ranks_occurrences_of_the_same_recording():
    view = identity_run.occurrence_quality_view(
        [_match("cue", 10.0, 24.0, 0.26), _match("cue", 100.0, 106.0, 0.70)], "ep1", None)
    group = view["groups"][0]
    assert group["selected_occurrence_id"] == "ep1@100.000"
    assert group["explained_spread"] == pytest.approx(0.44)
    assert group["occurrences"][0]["evidence_source"] == occurrence_quality.EVIDENCE_RELEASED


def test_view_excludes_theme_recordings_given_by_the_caller():
    view = identity_run.occurrence_quality_view(
        [_match("opening", 0.0, 90.0, 0.95), _match("cue", 200.0, 220.0, 0.40)],
        "ep1", {"opening"})
    themes = [g for g in view["groups"] if g["theme_excluded"]]
    assert len(themes) == 1 and themes[0]["recording_key"] == "opening"
    assert view["theme_excluded_count"] == 1 and view["selected_count"] == 1


def test_view_ignores_unresolved_and_empty_matches():
    view = identity_run.occurrence_quality_view([
        _match("cue", 10.0, 20.0, 0.5, state="unresolved"),
        _match("cue", 30.0, 30.0, 0.5),
        _match("cue", 40.0, 50.0, None),
    ], "ep1", None)
    assert view["recording_count"] == 0
    assert view["groups"] == []


def test_settings_record_the_ranking_recipe():
    assert identity_run.evidence_settings()["occurrence_quality_recipe_id"] == \
        identity_run.OCCURRENCE_QUALITY_RECIPE.recipe_id
