import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.validation.bgm_corpus import load_bgm_corpus
from lss.bgm.activity import ActivityRecipe, discover_activity
from lss.validation.activity_evidence import build_activity_detection_record
from lss.validation.occurrence_metrics import InvalidDetectionError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_bgm_discovery.py"
SPEC = importlib.util.spec_from_file_location("evaluate_bgm_discovery", SCRIPT)
evaluate_bgm_discovery = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_bgm_discovery)


def _corpus(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((1000, 2), dtype=np.float32), 1000, subtype="FLOAT")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records = [
        {
            "type": "manifest", "schema_version": 1, "corpus_id": "bgm.test",
            "revision": "r1", "description": "test",
        },
        {
            "type": "case", "case_id": "present", "description": "present",
            "source": {"path": source.name, "sha256": digest, "sample_rate": 1000, "frames": 1000},
            "expected_outcome": "wanted_present", "attributes": [],
            "occurrences": [{
                "occurrence_id": "ref", "start_frame": 100, "end_frame": 400,
                "semantic_role": "dramatic_score", "wanted": True, "novel": True,
                "identity_state": "unknown", "cue_identity_id": None, "audio_version_id": None,
            }],
        },
        {
            "type": "case", "case_id": "absent", "description": "absent",
            "source": {"path": source.name, "sha256": digest, "sample_rate": 1000, "frames": 1000},
            "expected_outcome": "no_wanted_score", "attributes": ["target_absent"],
            "occurrences": [],
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    return manifest


def _record(corpus, case_id, candidates):
    case = next(item for item in corpus.cases if item.case_id == case_id)
    recipe = ActivityRecipe(
        window_frames=100, hop_frames=50, onset_rms=0.08, offset_rms=0.04,
        bridge_gap_frames=0, minimum_active_frames=1, padding_frames=0,
    )
    samples, sample_rate = sf.read(case.source_path, dtype="float32", always_2d=True)
    result = discover_activity(samples.T, sample_rate, recipe)
    record = build_activity_detection_record(case, case.source_path, recipe, result)
    record["candidates"] = candidates
    return record


def test_evaluator_writes_case_vectors_summary_and_worst_tail(tmp_path):
    manifest = _corpus(tmp_path)
    corpus = load_bgm_corpus(manifest)
    detections = tmp_path / "detections.jsonl"
    detections.write_text("\n".join(json.dumps(item) for item in [
        _record(corpus, "present", [{
            "candidate_id": "partial", "start_frame": 100, "end_frame": 250,
            "semantic_role": "unresolved_music_like", "identity_state": "unresolved",
            "cue_identity_id": None, "audio_version_id": None,
        }]),
        _record(corpus, "absent", []),
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "report.jsonl"

    assert evaluate_bgm_discovery.main([str(manifest), str(detections), str(output)]) == 0

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary, present, absent = records
    assert summary["type"] == "evaluation_summary"
    assert summary["case_count"] == 2
    assert summary["worst_activity_coverage_case"] == "present"
    assert summary["total_missed_activity_frames"] == 150
    assert summary["activity_reference_case_count"] == 1
    assert present["missed_activity_frames"] == 150
    assert absent["hard_failures"] == []
    assert list(tmp_path.glob(f".{output.name}-*.tmp")) == []


def test_missing_case_detection_is_unevaluated_not_valid_zero(tmp_path):
    corpus = load_bgm_corpus(_corpus(tmp_path))
    detections = tmp_path / "empty.jsonl"
    detections.write_text("", encoding="utf-8")

    results = evaluate_bgm_discovery.evaluate(
        corpus, evaluate_bgm_discovery.load_detections(detections, corpus),
    )

    assert results[0]["hard_failures"] == ["unevaluated_case"]
    assert results[1]["hard_failures"] == ["unevaluated_case"]
    summary = evaluate_bgm_discovery.build_summary(corpus, results)
    assert summary["unevaluated_case_count"] == 2
    assert summary["hard_failure_count"] == 2


def test_unknown_and_duplicate_detection_cases_are_rejected(tmp_path):
    corpus = load_bgm_corpus(_corpus(tmp_path))
    detections = tmp_path / "detections.jsonl"
    unknown = _record(corpus, "present", [])
    unknown["case_id"] = "missing"
    detections.write_text(json.dumps(unknown) + "\n", encoding="utf-8")
    with pytest.raises(InvalidDetectionError, match="corpusにないcase_id"):
        evaluate_bgm_discovery.load_detections(detections, corpus)

    duplicate = json.dumps(_record(corpus, "present", []))
    detections.write_text(duplicate + "\n" + duplicate + "\n", encoding="utf-8")
    with pytest.raises(InvalidDetectionError, match="case_idが重複"):
        evaluate_bgm_discovery.load_detections(detections, corpus)


def test_duplicate_json_object_key_is_rejected(tmp_path):
    corpus = load_bgm_corpus(_corpus(tmp_path))
    record = _record(corpus, "present", [])
    encoded = json.dumps(record)
    duplicate_key = encoded[:-1] + ', "case_id": "present"}'
    detections = tmp_path / "detections.jsonl"
    detections.write_text(duplicate_key + "\n", encoding="utf-8")

    with pytest.raises(InvalidDetectionError, match="JSONが不正"):
        evaluate_bgm_discovery.load_detections(detections, corpus)


def test_semantic_layer_measures_wanted_selection_and_refuses_unresolved(tmp_path):
    manifest = _corpus(tmp_path)
    corpus = load_bgm_corpus(manifest)
    detections = tmp_path / "detections.jsonl"
    detections.write_text("\n".join(json.dumps(item) for item in [
        _record(corpus, "present", [{
            "candidate_id": "kept", "start_frame": 100, "end_frame": 400,
            "semantic_role": "dramatic_score", "identity_state": "unresolved",
            "cue_identity_id": None, "audio_version_id": None,
        }, {
            "candidate_id": "dropped", "start_frame": 600, "end_frame": 700,
            "semantic_role": "musical_sfx", "identity_state": "unresolved",
            "cue_identity_id": None, "audio_version_id": None,
        }]),
        _record(corpus, "absent", []),
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "semantic.jsonl"
    assert evaluate_bgm_discovery.main(
        [str(manifest), str(detections), str(output), "--layer", "semantic-role"]) == 0
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary, results = lines[0], lines[1:]
    assert summary["evaluation_layer"] == "semantic_role"
    assert summary["macro_wanted_coverage"] == 1.0
    # 同じcoverageならcase_idの順で決まる。対象なしのcaseは1.0として扱われる。
    assert summary["worst_wanted_coverage"] == 1.0
    assert summary["worst_wanted_coverage_case"] in {"absent", "present"}
    present = next(item for item in results if item["case_id"] == "present")
    # 欲しくないroleは選ばれていないので、欲しい区間の外へはみ出していない。
    assert present["wanted_coverage"] == 1.0 and present["outside_wanted_frames"] == 0

    unresolved = tmp_path / "unresolved.jsonl"
    unresolved.write_text("\n".join(json.dumps(item) for item in [
        _record(corpus, "present", [{
            "candidate_id": "unknown", "start_frame": 100, "end_frame": 400,
            "semantic_role": "unresolved_music_like", "identity_state": "unresolved",
            "cue_identity_id": None, "audio_version_id": None,
        }]),
        _record(corpus, "absent", []),
    ]) + "\n", encoding="utf-8")
    with pytest.raises(InvalidDetectionError, match="未解決"):
        evaluate_bgm_discovery.main(
            [str(manifest), str(unresolved), str(tmp_path / "x.jsonl"), "--layer", "semantic-role"])


def test_the_default_layer_stays_activity_discovery(tmp_path):
    manifest = _corpus(tmp_path)
    corpus = load_bgm_corpus(manifest)
    detections = tmp_path / "detections.jsonl"
    detections.write_text("\n".join(json.dumps(item) for item in [
        _record(corpus, "present", [{
            "candidate_id": "partial", "start_frame": 100, "end_frame": 250,
            "semantic_role": "unresolved_music_like", "identity_state": "unresolved",
            "cue_identity_id": None, "audio_version_id": None,
        }]),
        _record(corpus, "absent", []),
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "default.jsonl"
    assert evaluate_bgm_discovery.main([str(manifest), str(detections), str(output)]) == 0
    summary = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert summary["evaluation_layer"] == "activity_discovery"
    assert "worst_activity_coverage" in summary
