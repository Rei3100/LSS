from pathlib import Path
from types import SimpleNamespace

import pytest

from lss.validation.bgm_corpus import BgmCase, ReferenceOccurrence
from lss.validation.occurrence_metrics import (
    DetectedOccurrence,
    InvalidDetectionError,
    detection_from_dict,
    evaluate_activity_discovery,
    evaluate_occurrences,
    summarize_occurrence_results,
)


def _reference(
    occurrence_id="ref-1", start=100, end=400, *, wanted=True,
    role="dramatic_score", identity="unknown", cue_id=None, version_id=None,
):
    return ReferenceOccurrence(
        occurrence_id=occurrence_id, start_frame=start, end_frame=end,
        semantic_role=role, wanted=wanted, novel=True if wanted else None,
        identity_state=identity, cue_identity_id=cue_id, audio_version_id=version_id,
    )


def _case(*references, frames=1000, outcome="wanted_present"):
    return BgmCase(
        case_id="case-1", description="test", source_path=Path("source.wav"),
        source_sha256="0" * 64, sample_rate=1000, frames=frames,
        expected_outcome=outcome, attributes=("quiet_score",),
        occurrences=tuple(references or (_reference(),)),
    )


def _det(candidate_id="candidate-1", start=100, end=400, **kwargs):
    kwargs.setdefault("semantic_role", "dramatic_score")
    return DetectedOccurrence(candidate_id, start, end, **kwargs)


def test_exact_occurrence_has_full_coverage_precision_and_no_failures():
    result = evaluate_occurrences(_case(), [_det()])

    assert result["wanted_coverage"] == 1.0
    assert result["candidate_precision"] == 1.0
    assert result["missed_frames"] == 0
    assert result["outside_wanted_frames"] == 0
    assert result["worst_boundary_error_frames"] == 0
    assert result["hard_failures"] == []


def test_partial_detection_keeps_raw_missed_outside_and_boundary_values():
    result = evaluate_occurrences(_case(), [_det(start=50, end=250)])

    assert result["overlap_frames"] == 150
    assert result["missed_frames"] == 150
    assert result["outside_wanted_frames"] == 50
    assert result["wanted_coverage"] == 0.5
    assert result["candidate_precision"] == 0.75
    assert result["worst_boundary_error_frames"] == 150


def test_target_absent_candidate_is_hard_failure():
    case = _case(_reference(wanted=False, role="musical_sfx"), outcome="music_not_wanted")

    result = evaluate_occurrences(case, [_det(start=100, end=200)])

    assert result["wanted_frames"] == 0
    assert result["wanted_coverage"] == 1.0
    assert result["candidate_precision"] == 0.0
    assert result["hard_failures"] == ["target_absent_false_positive"]


def test_complete_miss_is_hard_failure():
    result = evaluate_occurrences(_case(), [])

    assert result["wanted_coverage"] == 0.0
    assert result["missed_frames"] == 300
    assert result["hard_failures"] == ["wanted_target_completely_missed"]


def test_overlapping_candidates_do_not_inflate_coverage():
    result = evaluate_occurrences(
        _case(),
        [_det("one", 100, 300), _det("two", 200, 400)],
    )

    assert result["detected_frames"] == 300
    assert result["overlap_frames"] == 300
    assert result["wanted_coverage"] == 1.0
    assert result["fragmentation_count"] == 1


def test_multiple_reference_occurrences_are_evaluated_independently():
    case = _case(_reference("first", 100, 200), _reference("second", 500, 700))
    result = evaluate_occurrences(case, [_det("one", 100, 200), _det("two", 550, 650)])

    assert result["wanted_frames"] == 300
    assert result["overlap_frames"] == 200
    assert [item["occurrence_id"] for item in result["references"]] == ["first", "second"]
    assert result["references"][1]["start_error_frames"] == 50


def test_distinct_audio_versions_are_preserved_even_when_intervals_overlap():
    detections = [
        _det(
            "version-a", 100, 400, identity_state="known_exact",
            cue_identity_id="cue-1", audio_version_id="version-a",
        ),
        _det(
            "version-b", 100, 400, identity_state="known_related",
            cue_identity_id="cue-1", audio_version_id="version-b",
        ),
    ]

    result = evaluate_occurrences(_case(), detections)

    assert [item["audio_version_id"] for item in result["candidates"]] == ["version-a", "version-b"]
    assert result["detected_frames"] == 300


def test_duplicate_candidate_ids_are_rejected():
    with pytest.raises(InvalidDetectionError, match="candidate_idが重複"):
        evaluate_occurrences(_case(), [_det("same"), _det("same", 500, 600)])


def test_detection_parser_rejects_out_of_bounds_and_unknown_keys():
    with pytest.raises(InvalidDetectionError, match="source範囲外"):
        detection_from_dict(
            {"candidate_id": "x", "start_frame": 0, "end_frame": 1001}, source_frames=1000,
        )
    with pytest.raises(InvalidDetectionError, match="未対応のキー"):
        detection_from_dict(
            {"candidate_id": "x", "start_frame": 0, "end_frame": 1, "typo": True},
            source_frames=1000,
        )


def test_activity_discovery_counts_unwanted_source_music_as_positive_activity():
    case = _case(_reference(wanted=False, role="source_music"), outcome="music_not_wanted")

    result = evaluate_activity_discovery(
        case, [_det(start=100, end=400, semantic_role="unresolved_music_like")],
    )

    assert result["evaluation_layer"] == "activity_discovery"
    assert result["activity_coverage"] == 1.0
    assert result["candidate_precision"] == 1.0
    assert result["hard_failures"] == []


def test_unresolved_activity_candidate_cannot_be_scored_as_semantic_bgm():
    with pytest.raises(InvalidDetectionError, match="未解決activity候補"):
        evaluate_occurrences(
            _case(), [_det(semantic_role="unresolved_music_like")],
        )


def test_missing_detection_record_is_unevaluated_not_valid_zero():
    result = evaluate_activity_discovery(_case(), None)

    assert result["evaluated"] is False
    assert result["hard_failures"] == ["unevaluated_case"]
    assert result["activity_coverage"] is None


def test_shared_summary_excludes_target_absent_cases_from_coverage_denominator():
    present = _case()
    absent = BgmCase(
        case_id="absent", description="absent", source_path=Path("absent.wav"),
        source_sha256="1" * 64, sample_rate=1000, frames=1000,
        expected_outcome="no_wanted_score", attributes=(), occurrences=(),
    )
    summary = summarize_occurrence_results(
        SimpleNamespace(cases=(present, absent), corpus_id="test", revision="r1"),
        [evaluate_occurrences(present, [_det()]), evaluate_occurrences(absent, [])],
    )

    assert summary["wanted_reference_case_count"] == 1
    assert summary["macro_wanted_coverage"] == 1.0
