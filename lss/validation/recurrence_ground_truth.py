"""制御corpusの構築記録から、再演の正解と継ぎ目を読み、報告と突き合わせる。

分離・照合の実装は持たない。`lss/bgm/identity.py` が返した共通録音の区間を、
構築時に決まっている正解（どの録音のどの部分が、どの話のどこに置かれたか）と
比べるためだけの部品。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# 照合Stageが返し得る最短の共通区間。これ未満しか共有しない組は正解にしない。
MINIMUM_SHARED_SECONDS = 6.0
# 報告区間と正解区間を同じものとみなす重なりの割合（両方の話で満たす）。
MINIMUM_OVERLAP_RATIO = 0.5
# 継ぎ目を「見つけた」とする境界のずれ。
SEAM_TOLERANCE_SECONDS = 1.0


class RecurrenceGroundTruthError(ValueError):
    """構築記録が再演の正解として読めない。"""


@dataclass(frozen=True)
class Occurrence:
    """ある話の中に置かれた、ある録音の一区間。"""

    episode: str
    component_id: str
    cue_id: str
    start_seconds: float
    end_seconds: float
    source_start_seconds: float
    source_end_seconds: float
    gain_db: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.episode}/{self.component_id}"

    def at_source(self, source_seconds: float) -> float:
        """録音の中の時刻を、この話の中の時刻へ直す。"""
        return self.start_seconds + (source_seconds - self.source_start_seconds)


@dataclass(frozen=True)
class Placement:
    """話の中へ置かれた素材の一片。音楽に限らず台詞・効果音も含む。"""

    episode: str
    component_id: str
    role: str
    source_sha256: str
    gain_db: float
    start_seconds: float
    end_seconds: float
    source_start_seconds: float
    source_end_seconds: float

    def source_at(self, seconds: float) -> float:
        """この話の中の時刻を、素材の中の時刻へ直す。"""
        return self.source_start_seconds + (seconds - self.start_seconds)


@dataclass(frozen=True)
class ExpectedPair:
    """同じ録音の同じ部分を共有する2つの登場。話の中の正解区間を持つ。"""

    cue_id: str
    first: Occurrence
    second: Occurrence
    shared_seconds: float
    first_span: tuple[float, float]
    second_span: tuple[float, float]

    @property
    def episodes(self) -> tuple[str, str]:
        return (self.first.episode, self.second.episode)

    @property
    def within_episode(self) -> bool:
        return self.first.episode == self.second.episode


@dataclass(frozen=True)
class Seam:
    """同じ話の中で、別の録音が間を空けずに続く境目。"""

    episode: str
    seconds: float
    before_cue_id: str
    after_cue_id: str


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrenceGroundTruthError(f"{context}が数値ではありません")
    return float(value)


def load_occurrences(construction_path: Path) -> list[Occurrence]:
    """controlled mixのconstruction.jsonから、正解の登場一覧を読む。"""
    path = Path(construction_path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrenceGroundTruthError(f"構築記録を読めません: {path}") from exc
    cases = record.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RecurrenceGroundTruthError("構築記録にcasesがありません")
    occurrences: list[Occurrence] = []
    for case in cases:
        episode = case.get("case_id")
        rate = case.get("sample_rate")
        if not isinstance(episode, str) or not episode or not isinstance(rate, int) or rate <= 0:
            raise RecurrenceGroundTruthError("caseのcase_id/sample_rateが不正です")
        for component in case.get("components", []):
            occurrence = component.get("occurrence")
            if occurrence is None:
                continue
            cue_id = occurrence.get("cue_identity_id")
            if not isinstance(cue_id, str) or not cue_id:
                raise RecurrenceGroundTruthError(f"{episode}の登場にcue_identity_idがありません")
            source_rate = component.get("source_sample_rate")
            if not isinstance(source_rate, int) or source_rate <= 0:
                raise RecurrenceGroundTruthError(f"{episode}の登場にsource_sample_rateがありません")
            occurrences.append(Occurrence(
                episode=episode,
                component_id=str(component["component_id"]),
                cue_id=cue_id,
                start_seconds=_number(component["output_start_frame"], "output_start_frame") / rate,
                end_seconds=_number(component["output_end_frame"], "output_end_frame") / rate,
                source_start_seconds=_number(component["source_start_frame"], "source_start_frame") / source_rate,
                source_end_seconds=_number(component["source_end_frame"], "source_end_frame") / source_rate,
                gain_db=_number(component.get("gain_db", 0.0), "gain_db"),
            ))
    if not occurrences:
        raise RecurrenceGroundTruthError("構築記録に録音の登場がありません")
    return occurrences


def load_placements(construction_path: Path) -> list[Placement]:
    """構築記録から、音楽以外も含めた素材の配置を読む。誤りの正体を見分けるために使う。"""
    path = Path(construction_path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrenceGroundTruthError(f"構築記録を読めません: {path}") from exc
    placements: list[Placement] = []
    for case in record.get("cases", []):
        episode = case.get("case_id")
        rate = case.get("sample_rate")
        if not isinstance(episode, str) or not episode or not isinstance(rate, int) or rate <= 0:
            raise RecurrenceGroundTruthError("caseのcase_id/sample_rateが不正です")
        for component in case.get("components", []):
            source_rate = component.get("source_sample_rate")
            source_sha256 = component.get("source_sha256")
            if not isinstance(source_rate, int) or source_rate <= 0 or not isinstance(source_sha256, str):
                raise RecurrenceGroundTruthError(f"{episode}のcomponentに素材の同一性がありません")
            placements.append(Placement(
                episode=episode,
                component_id=str(component["component_id"]),
                role=str(component.get("reference_role") or "unknown"),
                source_sha256=source_sha256,
                gain_db=_number(component.get("gain_db", 0.0), "gain_db"),
                start_seconds=_number(component["output_start_frame"], "output_start_frame") / rate,
                end_seconds=_number(component["output_end_frame"], "output_end_frame") / rate,
                source_start_seconds=_number(component["source_start_frame"], "source_start_frame") / source_rate,
                source_end_seconds=_number(component["source_end_frame"], "source_end_frame") / source_rate,
            ))
    if not placements:
        raise RecurrenceGroundTruthError("構築記録に素材の配置がありません")
    return placements


def _covering(placements: Iterable[Placement], episode: str, span: tuple[float, float]) -> list[Placement]:
    """その区間に鳴っている素材を全て返す。台詞は音楽へ重なるため一つに絞らず、中央だけも見ない。"""
    return [placement for placement in placements
            if placement.episode == episode
            and _overlap((placement.start_seconds, placement.end_seconds), span) > 0]


def classify_report(
    report: dict, *, first_episode: str, second_episode: str, placements: Iterable[Placement],
) -> dict:
    """正解の組に対応しなかった報告が、同じ素材の別整列か、別素材どうしの誤りかを見分ける。

    報告区間に重なる素材を両側で集め、同じ実体（同じsha256）の組があれば同じ録音として
    数える。整列した素材内の時刻が重なれば同じ部分、重ならなければ別の部分とする。
    """
    items = list(placements)
    first_span = (report["first_start_seconds"], report["first_end_seconds"])
    second_span = (report["second_start_seconds"], report["second_end_seconds"])
    first_covering = _covering(items, first_episode, first_span)
    second_covering = _covering(items, second_episode, second_span)
    if not first_covering or not second_covering:
        return {
            "category": "outside_placed_material",
            "first_roles": sorted({item.role for item in first_covering}),
            "second_roles": sorted({item.role for item in second_covering}),
        }
    best = None
    for first in first_covering:
        for second in second_covering:
            if first.source_sha256 != second.source_sha256:
                continue
            first_source = (first.source_at(report["first_start_seconds"]),
                            first.source_at(report["first_end_seconds"]))
            second_source = (second.source_at(report["second_start_seconds"]),
                             second.source_at(report["second_end_seconds"]))
            shared = _overlap(first_source, second_source)
            candidate = {
                "category": "same_recording_same_part" if shared > 0 else "same_recording_other_part",
                "first_roles": [first.role],
                "second_roles": [second.role],
                "first_source_span": [round(value, 3) for value in first_source],
                "second_source_span": [round(value, 3) for value in second_source],
            }
            if best is None or (shared > 0 and best["category"] != "same_recording_same_part"):
                best = candidate
    if best is not None:
        return best
    return {
        "category": "cross_recording",
        "first_roles": sorted({item.role for item in first_covering}),
        "second_roles": sorted({item.role for item in second_covering}),
    }


def occurrence_counts(occurrences: Iterable[Occurrence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for occurrence in occurrences:
        counts[occurrence.cue_id] = counts.get(occurrence.cue_id, 0) + 1
    return counts


def expected_pairs(
    occurrences: Iterable[Occurrence], *, minimum_seconds: float = MINIMUM_SHARED_SECONDS,
) -> list[ExpectedPair]:
    """同じ録音の同じ部分を十分な長さで共有する登場の組を、話の中の正解区間付きで返す。"""
    items = sorted(occurrences, key=lambda item: (item.episode, item.start_seconds, item.component_id))
    pairs: list[ExpectedPair] = []
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            if first.cue_id != second.cue_id:
                continue
            start = max(first.source_start_seconds, second.source_start_seconds)
            end = min(first.source_end_seconds, second.source_end_seconds)
            shared = end - start
            if shared < minimum_seconds:
                continue
            pairs.append(ExpectedPair(
                cue_id=first.cue_id, first=first, second=second, shared_seconds=shared,
                first_span=(first.at_source(start), first.at_source(end)),
                second_span=(second.at_source(start), second.at_source(end)),
            ))
    return pairs


def seams(occurrences: Iterable[Occurrence], *, gap_tolerance_seconds: float = 0.001) -> list[Seam]:
    """別の録音が間を空けずに続く境目を返す。"""
    found: list[Seam] = []
    by_episode: dict[str, list[Occurrence]] = {}
    for occurrence in occurrences:
        by_episode.setdefault(occurrence.episode, []).append(occurrence)
    for episode in sorted(by_episode):
        ordered = sorted(by_episode[episode], key=lambda item: item.start_seconds)
        for before, after in zip(ordered, ordered[1:]):
            if abs(after.start_seconds - before.end_seconds) > gap_tolerance_seconds:
                continue
            if before.cue_id == after.cue_id:
                continue
            found.append(Seam(episode=episode, seconds=after.start_seconds,
                              before_cue_id=before.cue_id, after_cue_id=after.cue_id))
    return found


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _covers(reported: tuple[float, float], expected: tuple[float, float], ratio: float) -> bool:
    shared = _overlap(reported, expected)
    shortest = min(reported[1] - reported[0], expected[1] - expected[0])
    return shortest > 0 and shared / shortest >= ratio


@dataclass(frozen=True)
class PairAssessment:
    """ある組について、正解の何が見つかり、報告の何が正解と対応しなかったか。"""

    detected: list[dict]
    missed: list[dict]
    unmatched_reports: list[dict]


def assess_pair(
    expected: list[ExpectedPair], reported: list[dict], *,
    first_episode: str, second_episode: str,
    minimum_overlap_ratio: float = MINIMUM_OVERLAP_RATIO,
) -> PairAssessment:
    """1組の話に対する報告区間を正解と突き合わせる。

    `reported` は `first_start_seconds`/`first_end_seconds`/`second_start_seconds`/
    `second_end_seconds` を持つ辞書の並び。`first_episode` 側が `first_*` に対応する。
    """
    matched_reports: set[int] = set()
    detected: list[dict] = []
    missed: list[dict] = []
    for pair in expected:
        if (pair.first.episode, pair.second.episode) == (first_episode, second_episode):
            forward = True
        elif (pair.second.episode, pair.first.episode) == (first_episode, second_episode):
            forward = False
        else:
            continue
        want_first, want_second = (
            (pair.first_span, pair.second_span) if forward else (pair.second_span, pair.first_span))
        best = None
        for position, report in enumerate(reported):
            got_first = (report["first_start_seconds"], report["first_end_seconds"])
            got_second = (report["second_start_seconds"], report["second_end_seconds"])
            if not (_covers(got_first, want_first, minimum_overlap_ratio)
                    and _covers(got_second, want_second, minimum_overlap_ratio)):
                continue
            covered = _overlap(got_first, want_first)
            if best is None or covered > best[1]:
                best = (position, covered, got_first, got_second)
        record = {
            "cue_id": pair.cue_id,
            "first": pair.first.key if forward else pair.second.key,
            "second": pair.second.key if forward else pair.first.key,
            "shared_seconds": round(pair.shared_seconds, 3),
            "expected_first": [round(value, 3) for value in want_first],
            "expected_second": [round(value, 3) for value in want_second],
        }
        if best is None:
            missed.append(record)
            continue
        position, covered, got_first, got_second = best
        matched_reports.add(position)
        detected.append({
            **record,
            "reported_first": [round(value, 3) for value in got_first],
            "reported_second": [round(value, 3) for value in got_second],
            "start_error_seconds": round(got_first[0] - want_first[0], 3),
            "end_error_seconds": round(got_first[1] - want_first[1], 3),
            "covered_ratio": round(covered / (want_first[1] - want_first[0]), 4),
        })
    unmatched = [report for position, report in enumerate(reported) if position not in matched_reports]
    return PairAssessment(detected=detected, missed=missed, unmatched_reports=unmatched)


def boundary_proposals(reports_by_episode: dict[str, list[float]]) -> dict[str, list[float]]:
    """話ごとの境界候補を、重複を除いて昇順で返す。"""
    return {episode: sorted({round(value, 3) for value in values})
            for episode, values in reports_by_episode.items()}


def seam_findings(
    seam_list: Iterable[Seam], proposals: dict[str, list[float]], *,
    tolerance_seconds: float = SEAM_TOLERANCE_SECONDS,
) -> list[dict]:
    """正解の継ぎ目ごとに、最も近い境界候補とのずれを返す。"""
    findings = []
    for seam in seam_list:
        candidates = proposals.get(seam.episode, [])
        nearest = min(candidates, key=lambda value: abs(value - seam.seconds), default=None)
        distance = None if nearest is None else round(abs(nearest - seam.seconds), 3)
        findings.append({
            "episode": seam.episode,
            "seam_seconds": round(seam.seconds, 3),
            "before_cue_id": seam.before_cue_id,
            "after_cue_id": seam.after_cue_id,
            "nearest_boundary_seconds": nearest,
            "distance_seconds": distance,
            "found": distance is not None and distance <= tolerance_seconds,
        })
    return findings
