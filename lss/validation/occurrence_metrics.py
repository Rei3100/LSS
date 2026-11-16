"""BGM activity候補と正解区間を、重複で水増しせずケース単位で評価する。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from lss.validation.bgm_corpus import BgmCase, IDENTITY_STATES, SEMANTIC_ROLES


METRIC_VERSION = 2
WANTED_BGM_ROLES = frozenset({"dramatic_score", "singing_score"})
MUSIC_ACTIVITY_ROLES = SEMANTIC_ROLES - {"vocalization"}


class InvalidDetectionError(ValueError):
    """activity discovery出力が評価契約に合わない。"""


@dataclass(frozen=True)
class DetectedOccurrence:
    candidate_id: str
    start_frame: int
    end_frame: int
    semantic_role: str = "unresolved_music_like"
    identity_state: str = "unresolved"
    cue_identity_id: str | None = None
    audio_version_id: str | None = None


def detection_from_dict(raw: Any, *, source_frames: int) -> DetectedOccurrence:
    if not isinstance(raw, dict):
        raise InvalidDetectionError("candidateはオブジェクトである必要があります")
    allowed = {
        "candidate_id", "start_frame", "end_frame", "semantic_role", "identity_state",
        "cue_identity_id", "audio_version_id",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise InvalidDetectionError(f"candidateに未対応のキーがあります: {', '.join(unknown)}")
    candidate_id = raw.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise InvalidDetectionError("candidate_idが未指定です")
    start = raw.get("start_frame")
    end = raw.get("end_frame")
    if (
        not isinstance(start, int) or isinstance(start, bool) or start < 0
        or not isinstance(end, int) or isinstance(end, bool) or end <= start or end > source_frames
    ):
        raise InvalidDetectionError(
            f"candidate {candidate_id}の区間がsource範囲外です: "
            f"start={start}, end={end}, frames={source_frames}"
        )
    role = raw.get("semantic_role", "unresolved_music_like")
    identity = raw.get("identity_state", "unresolved")
    if role not in SEMANTIC_ROLES:
        raise InvalidDetectionError(f"candidate {candidate_id}のsemantic_roleが不正です")
    if identity not in IDENTITY_STATES:
        raise InvalidDetectionError(f"candidate {candidate_id}のidentity_stateが不正です")
    cue_id = raw.get("cue_identity_id")
    version_id = raw.get("audio_version_id")
    if cue_id is not None and (not isinstance(cue_id, str) or not cue_id.strip()):
        raise InvalidDetectionError(f"candidate {candidate_id}のcue_identity_idが不正です")
    if version_id is not None and (not isinstance(version_id, str) or not version_id.strip()):
        raise InvalidDetectionError(f"candidate {candidate_id}のaudio_version_idが不正です")
    return DetectedOccurrence(
        candidate_id=candidate_id.strip(),
        start_frame=start,
        end_frame=end,
        semantic_role=role,
        identity_state=identity,
        cue_identity_id=cue_id,
        audio_version_id=version_id,
    )


def _union(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_length(
    left: list[tuple[int, int]], right: list[tuple[int, int]],
) -> int:
    total = 0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total


def _validated_candidates(case: BgmCase, detections: Iterable[DetectedOccurrence]) -> tuple[DetectedOccurrence, ...]:
    candidates = tuple(detections)
    candidate_ids = [item.candidate_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise InvalidDetectionError(f"case {case.case_id}のcandidate_idが重複しています")
    for item in candidates:
        if item.start_frame < 0 or item.end_frame <= item.start_frame or item.end_frame > case.frames:
            raise InvalidDetectionError(f"candidate {item.candidate_id}がsource範囲外です")
    return candidates


def _evaluate_intervals(
    case: BgmCase,
    candidates: tuple[DetectedOccurrence, ...],
    references,
    *,
    layer: str,
    absent_failure: str,
    miss_failure: str,
) -> dict[str, Any]:
    reference_intervals = _union((item.start_frame, item.end_frame) for item in references)
    detected_intervals = _union((item.start_frame, item.end_frame) for item in candidates)
    reference_frames = _length(reference_intervals)
    detected_frames = _length(detected_intervals)
    overlap_frames = _intersection_length(reference_intervals, detected_intervals)
    missed_frames = reference_frames - overlap_frames
    outside_frames = detected_frames - overlap_frames

    per_reference = []
    fragmentation_count = 0
    boundary_errors: list[int] = []
    for reference in references:
        overlaps = []
        for candidate in candidates:
            overlap = max(
                0,
                min(reference.end_frame, candidate.end_frame)
                - max(reference.start_frame, candidate.start_frame),
            )
            if overlap:
                overlaps.append((overlap, candidate))
        fragmentation_count += max(0, len(overlaps) - 1)
        if overlaps:
            _, best = max(overlaps, key=lambda pair: (pair[0], -pair[1].start_frame, pair[1].candidate_id))
            start_error = abs(best.start_frame - reference.start_frame)
            end_error = abs(best.end_frame - reference.end_frame)
            boundary_errors.extend((start_error, end_error))
            best_id = best.candidate_id
        else:
            start_error = end_error = None
            best_id = None
        per_reference.append({
            "occurrence_id": reference.occurrence_id,
            "matched_candidate_id": best_id,
            "overlapping_candidate_count": len(overlaps),
            "start_error_frames": start_error,
            "end_error_frames": end_error,
        })

    hard_failures = []
    if reference_frames == 0 and detected_frames > 0:
        hard_failures.append(absent_failure)
    if reference_frames > 0 and overlap_frames == 0:
        hard_failures.append(miss_failure)

    common = {
        "type": "case_result",
        "metric_version": METRIC_VERSION,
        "evaluation_layer": layer,
        "evaluated": True,
        "case_id": case.case_id,
        "expected_outcome": case.expected_outcome,
        "attributes": list(case.attributes),
        "source_frames": case.frames,
        "detected_frames": detected_frames,
        "overlap_frames": overlap_frames,
        "candidate_precision": overlap_frames / detected_frames if detected_frames else 1.0,
        "fragmentation_count": fragmentation_count,
        "worst_boundary_error_frames": max(boundary_errors, default=None),
        "hard_failures": hard_failures,
        "references": per_reference,
        # 区間coverageのunion化とは別に候補identityを全件保持し、別versionを破壊的に潰さない。
        "candidates": [asdict(item) for item in candidates],
    }
    if layer == "activity_discovery":
        return {
            **common,
            "activity_reference_frames": reference_frames,
            "missed_activity_frames": missed_frames,
            "outside_activity_frames": outside_frames,
            "activity_coverage": overlap_frames / reference_frames if reference_frames else 1.0,
            "outside_activity_ratio": outside_frames / case.frames,
        }
    return {
        **common,
        "wanted_frames": reference_frames,
        "missed_frames": missed_frames,
        "outside_wanted_frames": outside_frames,
        "wanted_coverage": overlap_frames / reference_frames if reference_frames else 1.0,
        "outside_wanted_ratio": outside_frames / case.frames,
    }


def evaluate_activity_discovery(
    case: BgmCase,
    detections: Iterable[DetectedOccurrence] | None,
) -> dict[str, Any]:
    """L1候補発見を全music-like区間に対して評価する。意味roleの採否は扱わない。"""
    references = tuple(item for item in case.occurrences if item.semantic_role in MUSIC_ACTIVITY_ROLES)
    if detections is None:
        reference_frames = _length(_union((item.start_frame, item.end_frame) for item in references))
        return {
            "type": "case_result",
            "metric_version": METRIC_VERSION,
            "evaluation_layer": "activity_discovery",
            "evaluated": False,
            "case_id": case.case_id,
            "expected_outcome": case.expected_outcome,
            "attributes": list(case.attributes),
            "source_frames": case.frames,
            "activity_reference_frames": reference_frames,
            "detected_frames": None,
            "overlap_frames": None,
            "missed_activity_frames": None,
            "outside_activity_frames": None,
            "activity_coverage": None,
            "candidate_precision": None,
            "outside_activity_ratio": None,
            "fragmentation_count": None,
            "worst_boundary_error_frames": None,
            "hard_failures": ["unevaluated_case"],
            "references": [],
            "candidates": [],
        }
    candidates = _validated_candidates(case, detections)
    return _evaluate_intervals(
        case, candidates, references, layer="activity_discovery",
        absent_failure="music_activity_absent_false_positive",
        miss_failure="music_activity_completely_missed",
    )


def evaluate_occurrences(
    case: BgmCase,
    detections: Iterable[DetectedOccurrence],
) -> dict[str, Any]:
    """L4意味判定済み候補だけをwanted BGMとして評価する。"""
    all_candidates = _validated_candidates(case, detections)
    if any(item.semantic_role == "unresolved_music_like" for item in all_candidates):
        raise InvalidDetectionError("L4 semantic評価へ未解決activity候補を渡せません")
    candidates = tuple(item for item in all_candidates if item.semantic_role in WANTED_BGM_ROLES)
    references = tuple(item for item in case.occurrences if item.wanted)
    return _evaluate_intervals(
        case, candidates, references, layer="semantic_role",
        absent_failure="target_absent_false_positive",
        miss_failure="wanted_target_completely_missed",
    )


def summarize_occurrence_results(corpus, results: list[dict[str, Any]]) -> dict[str, Any]:
    """全評価経路で同じ分母とmetric versionを使う共有summary。"""
    if len(results) != len(corpus.cases):
        raise ValueError("summary対象のcase数がcorpusと一致しません")
    layers = {item.get("evaluation_layer") for item in results}
    if len(layers) != 1:
        raise ValueError("異なる評価layerを同じsummaryへ混在できません")
    layer = layers.pop()
    evaluated = [item for item in results if item.get("evaluated") is True]
    boundary_values = [
        item["worst_boundary_error_frames"] for item in evaluated
        if item["worst_boundary_error_frames"] is not None
    ]
    summary: dict[str, Any] = {
        "type": "evaluation_summary",
        "schema_version": 1,
        "metric_version": METRIC_VERSION,
        "evaluation_layer": layer,
        "corpus_id": corpus.corpus_id,
        "corpus_revision": corpus.revision,
        "case_count": len(results),
        "evaluated_case_count": len(evaluated),
        "unevaluated_case_count": len(results) - len(evaluated),
        "hard_failure_count": sum(len(item["hard_failures"]) for item in results),
        "cases_with_hard_failures": sum(bool(item["hard_failures"]) for item in results),
        "worst_boundary_error_frames": max(boundary_values, default=None),
    }
    if layer == "activity_discovery":
        coverage = [item for item in evaluated if item["activity_reference_frames"] > 0]
        precision = [item for item in evaluated if item["detected_frames"] > 0]
        summary.update({
            "activity_reference_case_count": len(coverage),
            "candidate_precision_case_count": len(precision),
            "macro_activity_coverage": (
                sum(item["activity_coverage"] for item in coverage) / len(coverage)
                if coverage else 1.0
            ),
            "macro_candidate_precision": (
                sum(item["candidate_precision"] for item in precision) / len(precision)
                if precision else 1.0
            ),
            "total_missed_activity_frames": sum(item["missed_activity_frames"] for item in evaluated),
            "total_outside_activity_frames": sum(item["outside_activity_frames"] for item in evaluated),
        })
    elif layer == "semantic_role":
        coverage = [item for item in evaluated if item["wanted_frames"] > 0]
        precision = [item for item in evaluated if item["detected_frames"] > 0]
        summary.update({
            "wanted_reference_case_count": len(coverage),
            "candidate_precision_case_count": len(precision),
            "macro_wanted_coverage": (
                sum(item["wanted_coverage"] for item in coverage) / len(coverage)
                if coverage else 1.0
            ),
            "macro_candidate_precision": (
                sum(item["candidate_precision"] for item in precision) / len(precision)
                if precision else 1.0
            ),
            "total_missed_frames": sum(item["missed_frames"] for item in evaluated),
            "total_outside_wanted_frames": sum(item["outside_wanted_frames"] for item in evaluated),
        })
    else:
        raise ValueError(f"未対応の評価layerです: {layer}")
    return summary
