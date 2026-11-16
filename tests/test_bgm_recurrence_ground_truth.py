"""再演の正解の読み取りと突き合わせ。実音声は使わない。"""
from __future__ import annotations

import json

import pytest

from lss.validation import recurrence_ground_truth as truth


def _component(component_id, cue_id, source_start, source_end, start, *, rate=48000, out_rate=44100):
    occurrence = None if cue_id is None else {
        "occurrence_id": f"{component_id}", "semantic_role": "dramatic_score", "wanted": True,
        "novel": True, "identity_state": "known_exact", "cue_identity_id": cue_id,
        "audio_version_id": "version-1",
    }
    return {
        "component_id": component_id,
        "source_sample_rate": rate,
        "source_start_frame": int(source_start * rate),
        "source_end_frame": int(source_end * rate),
        "output_start_frame": int(start * out_rate),
        "output_end_frame": int((start + (source_end - source_start)) * out_rate),
        "occurrence": occurrence,
    }


def _construction(cases):
    return {"schema_version": 1, "cases": [
        {"case_id": case_id, "sample_rate": 44100, "components": components}
        for case_id, components in cases
    ]}


def _write(tmp_path, cases):
    path = tmp_path / "construction.json"
    path.write_text(json.dumps(_construction(cases), ensure_ascii=False), encoding="utf-8")
    return path


def test_load_occurrences_converts_frames_to_seconds(tmp_path):
    path = _write(tmp_path, [("ep1", [_component("cue-a-1", "cue-a", 10, 50, 20)])])
    occurrences = truth.load_occurrences(path)
    assert len(occurrences) == 1
    occurrence = occurrences[0]
    assert occurrence.episode == "ep1"
    assert occurrence.cue_id == "cue-a"
    assert occurrence.start_seconds == pytest.approx(20.0)
    assert occurrence.end_seconds == pytest.approx(60.0)
    assert occurrence.source_start_seconds == pytest.approx(10.0)
    assert occurrence.at_source(25.0) == pytest.approx(35.0)


def test_load_occurrences_skips_components_without_occurrence(tmp_path):
    path = _write(tmp_path, [("ep1", [
        _component("cue-a-1", "cue-a", 0, 30, 10),
        _component("dialogue-1", None, 0, 5, 40, rate=22050),
    ])])
    assert [item.component_id for item in truth.load_occurrences(path)] == ["cue-a-1"]


def test_load_occurrences_rejects_records_without_cue_identity(tmp_path):
    path = _write(tmp_path, [("ep1", [_component("cue-a-1", "cue-a", 0, 30, 10)])])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["cases"][0]["components"][0]["occurrence"]["cue_identity_id"] = ""
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(truth.RecurrenceGroundTruthError):
        truth.load_occurrences(path)


def test_expected_pairs_uses_shared_source_span_only(tmp_path):
    path = _write(tmp_path, [
        ("ep1", [_component("cue-a-1", "cue-a", 0, 40, 10)]),
        ("ep2", [_component("cue-a-1", "cue-a", 20, 60, 100)]),
        ("ep3", [_component("cue-a-1", "cue-a", 45, 80, 5)]),
    ])
    occurrences = truth.load_occurrences(path)
    pairs = truth.expected_pairs(occurrences)
    # ep1とep3は同じ録音でも別の部分しか使っていない。組にしない。
    assert {(pair.first.episode, pair.second.episode) for pair in pairs} == {("ep1", "ep2"), ("ep2", "ep3")}
    first = next(pair for pair in pairs if pair.episodes == ("ep1", "ep2"))
    assert first.shared_seconds == pytest.approx(20.0)
    assert first.first_span == pytest.approx((30.0, 50.0))
    assert first.second_span == pytest.approx((100.0, 120.0))


def test_expected_pairs_ignores_short_overlap(tmp_path):
    path = _write(tmp_path, [
        ("ep1", [_component("cue-a-1", "cue-a", 0, 30, 10)]),
        ("ep2", [_component("cue-a-1", "cue-a", 25, 60, 10)]),
    ])
    assert truth.expected_pairs(truth.load_occurrences(path)) == []


def test_expected_pairs_finds_repeat_inside_one_episode(tmp_path):
    path = _write(tmp_path, [("ep1", [
        _component("cue-a-1", "cue-a", 0, 40, 10),
        _component("cue-a-2", "cue-a", 10, 50, 100),
    ])])
    pairs = truth.expected_pairs(truth.load_occurrences(path))
    assert len(pairs) == 1 and pairs[0].within_episode


def test_seams_only_count_touching_different_recordings(tmp_path):
    path = _write(tmp_path, [("ep1", [
        _component("cue-a-1", "cue-a", 0, 30, 10),   # 10-40
        _component("cue-b-1", "cue-b", 0, 20, 40),   # 40-60 継ぎ目
        _component("cue-b-2", "cue-b", 40, 60, 60),  # 同じ録音どうしは継ぎ目にしない
        _component("cue-c-1", "cue-c", 0, 10, 100),  # 間が空くので継ぎ目にしない
    ])])
    found = truth.seams(truth.load_occurrences(path))
    assert [(item.episode, item.seconds, item.before_cue_id, item.after_cue_id)
            for item in found] == [("ep1", 40.0, "cue-a", "cue-b")]


def _report(first_start, first_end, second_start, second_end):
    return {"first_start_seconds": first_start, "first_end_seconds": first_end,
            "second_start_seconds": second_start, "second_end_seconds": second_end,
            "waveform_explained": 0.5, "blocks": 10}


def test_assess_pair_matches_overlapping_report(tmp_path):
    path = _write(tmp_path, [
        ("ep1", [_component("cue-a-1", "cue-a", 0, 40, 10)]),
        ("ep2", [_component("cue-a-1", "cue-a", 10, 50, 100)]),
    ])
    expected = truth.expected_pairs(truth.load_occurrences(path))
    assessment = truth.assess_pair(expected, [_report(19.5, 50.0, 99.5, 130.0)],
                                   first_episode="ep1", second_episode="ep2")
    assert assessment.missed == [] and assessment.unmatched_reports == []
    assert assessment.detected[0]["start_error_seconds"] == pytest.approx(-0.5)
    assert assessment.detected[0]["end_error_seconds"] == pytest.approx(0.0)


def test_assess_pair_reports_unmatched_and_missed(tmp_path):
    path = _write(tmp_path, [
        ("ep1", [_component("cue-a-1", "cue-a", 0, 40, 10)]),
        ("ep2", [_component("cue-a-1", "cue-a", 10, 50, 100)]),
    ])
    expected = truth.expected_pairs(truth.load_occurrences(path))
    assessment = truth.assess_pair(expected, [_report(0.0, 5.0, 200.0, 205.0)],
                                   first_episode="ep1", second_episode="ep2")
    assert len(assessment.missed) == 1 and len(assessment.unmatched_reports) == 1
    assert assessment.detected == []


def test_assess_pair_swaps_sides_when_pair_order_differs(tmp_path):
    path = _write(tmp_path, [
        ("ep1", [_component("cue-a-1", "cue-a", 0, 40, 10)]),
        ("ep2", [_component("cue-a-1", "cue-a", 10, 50, 100)]),
    ])
    expected = truth.expected_pairs(truth.load_occurrences(path))
    assessment = truth.assess_pair(expected, [_report(100.0, 130.0, 20.0, 50.0)],
                                   first_episode="ep2", second_episode="ep1")
    assert len(assessment.detected) == 1
    assert assessment.detected[0]["first"] == "ep2/cue-a-1"
    assert assessment.detected[0]["second"] == "ep1/cue-a-1"


def test_seam_findings_use_nearest_boundary(tmp_path):
    path = _write(tmp_path, [("ep1", [
        _component("cue-a-1", "cue-a", 0, 30, 10),
        _component("cue-b-1", "cue-b", 0, 20, 40),
    ])])
    seam_list = truth.seams(truth.load_occurrences(path))
    proposals = truth.boundary_proposals({"ep1": [10.0, 39.4, 60.0, 39.4]})
    assert proposals == {"ep1": [10.0, 39.4, 60.0]}
    findings = truth.seam_findings(seam_list, proposals)
    assert findings[0]["found"] is True
    assert findings[0]["distance_seconds"] == pytest.approx(0.6)
    assert truth.seam_findings(seam_list, {"ep1": [10.0]})[0]["found"] is False
    assert truth.seam_findings(seam_list, {})[0]["nearest_boundary_seconds"] is None


def _with_sha(cases, shas):
    record = _construction(cases)
    for case in record["cases"]:
        for component in case["components"]:
            component["source_sha256"] = shas[component["component_id"]]
            component["reference_role"] = "dialogue" if component["component_id"].startswith("dialogue") else "music"
    return record


def _write_placements(tmp_path, cases, shas):
    path = tmp_path / "construction.json"
    path.write_text(json.dumps(_with_sha(cases, shas), ensure_ascii=False), encoding="utf-8")
    return path


def test_load_placements_includes_material_without_occurrence(tmp_path):
    path = _write_placements(tmp_path, [("ep1", [
        _component("cue-a-1", "cue-a", 0, 30, 10),
        _component("dialogue-1", None, 0, 5, 50, rate=22050),
    ])], {"cue-a-1": "a" * 64, "dialogue-1": "d" * 64})
    placements = truth.load_placements(path)
    assert [(item.component_id, item.role) for item in placements] == [
        ("cue-a-1", "music"), ("dialogue-1", "dialogue")]


def test_load_placements_rejects_missing_source_identity(tmp_path):
    path = _write_placements(tmp_path, [("ep1", [_component("cue-a-1", "cue-a", 0, 30, 10)])],
                             {"cue-a-1": "a" * 64})
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["cases"][0]["components"][0]["source_sha256"]
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(truth.RecurrenceGroundTruthError):
        truth.load_placements(path)


def test_classify_report_separates_recording_and_alignment(tmp_path):
    path = _write_placements(tmp_path, [
        ("ep1", [_component("cue-a-1", "cue-a", 0, 60, 0),
                 _component("dialogue-1", None, 0, 5, 100, rate=22050)]),
        ("ep2", [_component("cue-a-1", "cue-a", 0, 60, 0),
                 _component("cue-b-1", "cue-b", 0, 30, 100)]),
    ], {"cue-a-1": "a" * 64, "cue-b-1": "b" * 64, "dialogue-1": "d" * 64})
    placements = truth.load_placements(path)

    same_part = truth.classify_report(_report(10.0, 20.0, 10.0, 20.0),
                                      first_episode="ep1", second_episode="ep2", placements=placements)
    assert same_part["category"] == "same_recording_same_part"

    other_part = truth.classify_report(_report(5.0, 15.0, 40.0, 50.0),
                                       first_episode="ep1", second_episode="ep2", placements=placements)
    assert other_part["category"] == "same_recording_other_part"

    cross = truth.classify_report(_report(10.0, 20.0, 105.0, 115.0),
                                  first_episode="ep1", second_episode="ep2", placements=placements)
    assert cross["category"] == "cross_recording"
    assert cross["first_roles"] == ["music"] and cross["second_roles"] == ["music"]

    outside = truth.classify_report(_report(200.0, 210.0, 10.0, 20.0),
                                    first_episode="ep1", second_episode="ep2", placements=placements)
    assert outside["category"] == "outside_placed_material"


def test_classify_report_prefers_the_overlapping_material(tmp_path):
    """台詞が音楽へ重なる時刻では、同じ台詞どうしの一致を別録音の誤りにしない。"""
    path = _write_placements(tmp_path, [
        ("ep1", [_component("cue-a-1", "cue-a", 0, 60, 0),
                 _component("dialogue-1", None, 0, 8, 10, rate=22050)]),
        ("ep2", [_component("cue-b-1", "cue-b", 0, 60, 0),
                 _component("dialogue-1", None, 0, 8, 30, rate=22050)]),
    ], {"cue-a-1": "a" * 64, "cue-b-1": "b" * 64, "dialogue-1": "d" * 64})
    placements = truth.load_placements(path)
    found = truth.classify_report(_report(10.0, 18.0, 30.0, 38.0),
                                  first_episode="ep1", second_episode="ep2", placements=placements)
    assert found["category"] == "same_recording_same_part"
    assert found["first_roles"] == ["dialogue"] and found["second_roles"] == ["dialogue"]
