"""照合による子候補を、未公開のまま安全に切り出し、中断から再利用する。"""
from __future__ import annotations

import json
from pathlib import Path

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss.bgm import candidate_store, identity, identity_candidates as export, identity_run
from tests.test_bgm_identity import _exact

RATE = 11025
OFFSET = 0.007


@pytest.fixture
def annotated(tmp_path):
    source = tmp_path / "source.mkv"
    source.write_bytes(b"original stereo program")
    signal = tmp_path / "signal.wav"
    values = np.random.default_rng(3).uniform(-0.5, 0.5, (40 * RATE, 2))
    sf.write(signal, values, RATE, subtype="PCM_24")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference recording")
    origin = candidate_store.CandidateOrigin(
        identity_run._sha256(source), identity_run._sha256(signal), "renderer", "runtime", "activity",
        9 * RATE, 31 * RATE, RATE, 7, 1000)
    run = {"status": identity_run.ROUTED_STATUS,
           "source": {"path": str(source), "sha256": origin.source_sha256, "source_id": "ep",
                      "audio_timeline_origin": {"offset_seconds": OFFSET}},
           "rendering": {"signal": {"path": str(signal), "sha256": origin.signal_sha256}},
           "candidates": [{"origin": origin.to_dict(), "asset": {"candidate_asset_id": origin.candidate_id}}]}
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    raw = identity_run.candidate_sets(run)["renderer+activity"]
    pieces = identity.split_candidates_at_matches(raw, [_exact("A", 10.007, 30.007)])
    record = {"type": "episode_identity_annotation", "schema_version": identity_run.RECORD_SCHEMA,
              "intent": {**identity_run.evidence_settings(), "run_sha256": identity_run._sha256(run_path),
                         "references": [{"path": str(reference), "sha256": identity_run._sha256(reference)}]},
              "query_signal": {"kind": "routed_signal", "sha256": origin.signal_sha256},
              "activity_runs": [{"recipe_id": "renderer+activity", "candidates": pieces}]}
    (tmp_path / "identity.json").write_text(json.dumps(record), encoding="utf-8")
    return run_path, signal, reference


def _record(path):
    return json.loads(Path(path).read_text("utf-8"))


def _edit(path, transform):
    data = _record(path)
    transform(data)
    Path(path).write_text(json.dumps(data), encoding="utf-8")


def test_short_unknown_children_are_saved_with_exact_frames_and_portable_unresolved_provenance(annotated, tmp_path):
    run_path, signal, reference = annotated
    inputs = {p: p.read_bytes() for p in (run_path, signal, reference, run_path.parent / "identity.json")}
    path = export.materialize(run_path, tmp_path / "export")
    record = _record(path)
    assert record["zero_reason"] is None and record["promotion_allowed"] is False
    assert record["candidate_basis"] == "identity_split_candidates"
    assert [c["status"] for c in record["candidates"]] == [
        "materialized_unresolved_development_candidate", "known_recording_not_materialized",
        "materialized_unresolved_development_candidate"]
    original, _ = sf.read(signal, dtype="float64", always_2d=True)
    for row, start in zip([c for c in record["candidates"] if "asset" in c], [9 * RATE, 30 * RATE]):
        saved = Path(row["asset"]["path"])
        values, rate = sf.read(saved, dtype="float64", always_2d=True)
        assert rate == RATE and values.shape == (RATE, 2)
        np.testing.assert_array_equal(values, original[start:start + RATE])
        provenance = json.loads(FLAC(saved)["LSS_CANDIDATE_PROVENANCE"][0])
        assert provenance["publication_state"] == "not_published" and provenance["identity_state"] == "unresolved"
        assert provenance["evidence"]["short_unresolved_remainder"] is True
        assert provenance["origin"]["time_offset_numerator"] == 7
        assert provenance["origin"]["start_frame"] == start
    assert inputs == {p: p.read_bytes() for p in inputs}
    stamp = path.stat().st_mtime_ns
    assert export.materialize(run_path, tmp_path / "export") == path
    assert path.stat().st_mtime_ns == stamp


def test_resume_reuses_finished_children_after_an_interrupted_later_crop(annotated, tmp_path, monkeypatch):
    run_path, *_ = annotated
    original = export.candidate_store.materialize_candidate
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("保存中断")
        return original(*args, **kwargs)

    monkeypatch.setattr(export.candidate_store, "materialize_candidate", interrupted)
    output = tmp_path / "export"
    with pytest.raises(export.IdentityCandidatesError, match="保存中断"):
        export.materialize(run_path, output)
    assert not (output / export.RECORD_NAME).exists()
    [first] = list((output / "candidates").glob("*.flac"))
    before = first.read_bytes(), first.stat().st_mtime_ns
    monkeypatch.setattr(export.candidate_store, "materialize_candidate", original)
    path = export.materialize(run_path, output)
    assert len([r for r in _record(path)["candidates"] if "asset" in r]) == 2
    assert (first.read_bytes(), first.stat().st_mtime_ns) == before
    assert not list(output.rglob("*.tmp"))


@pytest.mark.parametrize("change", ["annotation", "signal", "reference", "settings"])
def test_changes_during_export_do_not_replace_a_previous_complete_record(annotated, tmp_path, monkeypatch, change):
    run_path, signal, reference = annotated
    output = tmp_path / "export"
    path = export.materialize(run_path, output)
    before = path.read_bytes()
    original = export.candidate_store.materialize_candidate

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "settings":
            monkeypatch.setattr(identity, "MIN_VOTES", 10**9)
        else:
            target = {"annotation": run_path.parent / "identity.json", "signal": signal, "reference": reference}[change]
            target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(export.candidate_store, "materialize_candidate", changed)
    with pytest.raises(export.IdentityCandidatesError):
        export.materialize(run_path, output)
    assert path.read_bytes() == before
    assert not list(output.rglob("*.tmp"))


@pytest.mark.parametrize("change", ["old_settings", "missing_piece", "duplicate_piece", "missing_reference"])
def test_invalid_annotations_are_rejected_before_any_output(annotated, tmp_path, change):
    run_path, _, reference = annotated
    annotation = run_path.parent / "identity.json"
    if change == "old_settings":
        _edit(annotation, lambda r: r["intent"].update(implementation_sha256="old"))
    elif change == "missing_piece":
        _edit(annotation, lambda r: r["activity_runs"][0]["candidates"].pop(0))
    elif change == "duplicate_piece":
        _edit(annotation, lambda r: r["activity_runs"][0]["candidates"].append(r["activity_runs"][0]["candidates"][0]))
    else:
        reference.unlink()
    with pytest.raises(export.IdentityCandidatesError):
        export.materialize(run_path, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_unknown_single_occurrence_is_kept_when_there_are_no_references(annotated, tmp_path):
    run_path, *_ = annotated
    _edit(run_path.parent / "identity.json", lambda r: r.update(
        intent={**r["intent"], "references": []}, query_signal=None, activity_runs=[]))
    record = _record(export.materialize(run_path, tmp_path / "export"))
    assert record["candidate_basis"] == "raw_unmatched_candidates"
    [row] = record["candidates"]
    assert row["decision"] == "pending" and row["asset"]["frames"] == 22 * RATE


def test_no_activity_is_a_valid_empty_observation_even_without_a_separated_signal(annotated, tmp_path):
    run_path, signal, _ = annotated
    _edit(run_path, lambda r: r.update(candidates=[]))
    _edit(run_path.parent / "identity.json", lambda r: r.update(
        intent={**r["intent"], "run_sha256": identity_run._sha256(run_path), "references": []},
        query_signal=None, activity_runs=[]))
    signal.unlink()
    record = _record(export.materialize(run_path, tmp_path / "export"))
    assert record["candidates"] == [] and record["zero_reason"] is None
    assert record["signal"] is None
    assert not (tmp_path / "export" / "candidates").exists()


def test_all_known_regions_are_recorded_without_an_extra_audio_asset(annotated, tmp_path):
    run_path, *_ = annotated
    raw = identity_run.candidate_sets(_record(run_path))["renderer+activity"]
    pieces = identity.split_candidates_at_matches(raw, [_exact("A", 9.007, 31.007)])
    _edit(run_path.parent / "identity.json", lambda r: r["activity_runs"][0].update(candidates=pieces))
    record = _record(export.materialize(run_path, tmp_path / "export"))
    assert record["zero_reason"] == "no_novel_asset" and len(record["candidates"]) == 1
    assert not (tmp_path / "export" / "candidates").exists()


def test_standard_episode_candidates_use_the_original_separation_and_activity_revision(annotated, tmp_path):
    run_path, signal, _ = annotated
    run = _record(run_path)
    raw = identity_run.candidate_sets(run)["renderer+activity"]
    activity_path = tmp_path / "activity.json"
    activity_path.write_text(json.dumps({"candidates": raw}), encoding="utf-8")
    run.update(type="episode_bgm_discovery_run", status=identity_run.EPISODE_STATUS,
               separation={"recipe_id": "renderer", "runtime_id": "runtime", "stem_target": "Music",
                           "artifacts": [{"path": str(signal), "sha256": identity_run._sha256(signal), "stem": "Music"}]},
               activity_runs=[{"recipe_id": "activity", "record_path": str(activity_path),
                               "record_sha256": identity_run._sha256(activity_path)}])
    run.pop("rendering")
    run.pop("candidates")
    run_path.write_text(json.dumps(run), encoding="utf-8")
    _edit(run_path.parent / "identity.json", lambda r: r.update(
        intent={**r["intent"], "run_sha256": identity_run._sha256(run_path)},
        query_signal={"kind": "separation_artifact", "sha256": identity_run._sha256(signal)},
        activity_runs=[{**r["activity_runs"][0], "recipe_id": "activity"}]))
    record = _record(export.materialize(run_path, tmp_path / "export"))
    assert len([c for c in record["candidates"] if "asset" in c]) == 2
    [first, _, _] = record["candidates"]
    provenance = json.loads(FLAC(first["asset"]["path"])["LSS_CANDIDATE_PROVENANCE"][0])
    assert provenance["origin"]["separation_recipe_id"] == "renderer"
    assert provenance["origin"]["activity_recipe_id"] == "activity"


def test_positive_subsample_gap_is_kept_pending_without_an_empty_audio_file(annotated, tmp_path):
    run_path, *_ = annotated
    raw = identity_run.candidate_sets(_record(run_path))["renderer+activity"]
    pieces = identity.split_candidates_at_matches(raw, [_exact("A", 9.0070001, 31.007)])
    _edit(run_path.parent / "identity.json", lambda r: r["activity_runs"][0].update(candidates=pieces))
    record = _record(export.materialize(run_path, tmp_path / "export"))
    assert record["zero_reason"] is None
    assert record["candidates"][0]["status"] == "pending_subsample_interval_not_materializable"
    assert not (tmp_path / "export" / "candidates").exists()


def test_corrupt_saved_child_is_not_reused_as_completed(annotated, tmp_path):
    run_path, *_ = annotated
    path = export.materialize(run_path, tmp_path / "export")
    record = _record(path)
    child = Path(record["candidates"][0]["asset"]["path"])
    tags = dict(FLAC(child))
    sf.write(child, np.zeros((RATE, 2)), RATE, subtype="PCM_24")
    saved = FLAC(child)
    saved.update(tags)
    saved.save()
    before = path.read_bytes(), child.read_bytes()
    with pytest.raises(export.IdentityCandidatesError, match="音声"):
        export.materialize(run_path, tmp_path / "export")
    assert (path.read_bytes(), child.read_bytes()) == before


@pytest.mark.parametrize("field,value", [("sample_rate", 44100), ("source_sha256", "b" * 64),
                                        ("time_offset_numerator", 8)])
def test_inconsistent_routed_parent_provenance_is_not_cut(annotated, tmp_path, field, value):
    run_path, *_ = annotated
    _edit(run_path, lambda r: r["candidates"][0]["origin"].update({field: value}))
    run = _record(run_path)
    raw = identity_run.candidate_sets(run)["renderer+activity"]
    _edit(run_path.parent / "identity.json", lambda r: r.update(
        intent={**r["intent"], "run_sha256": identity_run._sha256(run_path)},
        activity_runs=[{"recipe_id": "renderer+activity", "candidates": raw}]))
    with pytest.raises(export.IdentityCandidatesError):
        export.materialize(run_path, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("payload", ["[]", "null", '{"intent": null}', '{"intent": []}', "{incomplete"])
def test_malformed_optional_annotation_has_a_structured_failure(annotated, tmp_path, payload):
    run_path, *_ = annotated
    (run_path.parent / "identity.json").write_text(payload, encoding="utf-8")
    with pytest.raises(export.IdentityCandidatesError):
        export.materialize(run_path, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_loaded_implementation_is_not_relabelled_with_a_changed_source_file(annotated, tmp_path, monkeypatch):
    run_path, *_ = annotated
    changed = tmp_path / "identity_candidates.py"
    changed.write_bytes(Path(export.__file__).read_bytes() + b"\n# changed after import\n")
    monkeypatch.setattr(export, "__file__", str(changed))
    with pytest.raises(export.IdentityCandidatesError, match="実装"):
        export.materialize(run_path, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("previous", ['{"type": "unrelated", "data": "keep"}', "old user file", "[]",
    '{"type": "identity_development_candidates", "schema_version": 1, "source": ["bad"]}',
    '{"type": "identity_development_candidates", "schema_version": 1, "source": {"sha256": "other"}}'])
def test_another_source_or_unrelated_destination_record_is_kept_before_cropping(annotated, tmp_path, previous):
    run_path, *_ = annotated
    output = tmp_path / "export"
    output.mkdir()
    path = output / export.RECORD_NAME
    path.write_text(previous, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(export.IdentityCandidatesError, match="保存先"):
        export.materialize(run_path, output)
    assert path.read_bytes() == before
    assert not (output / "candidates").exists()


def test_another_source_written_during_cropping_is_not_replaced(annotated, tmp_path, monkeypatch):
    run_path, *_ = annotated
    output = tmp_path / "export"
    original = export.candidate_store.materialize_candidate
    foreign = '{"type": "identity_development_candidates", "schema_version": 1, "source": {"sha256": "other"}}'

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (output / export.RECORD_NAME).write_text(foreign, encoding="utf-8")
        return result

    monkeypatch.setattr(export.candidate_store, "materialize_candidate", changed)
    with pytest.raises(export.IdentityCandidatesError, match="別の出典"):
        export.materialize(run_path, output)
    assert (output / export.RECORD_NAME).read_text("utf-8") == foreign
    assert not list(output.rglob("*.tmp"))


def test_record_competition_after_validation_is_not_overwritten(annotated, tmp_path, monkeypatch):
    run_path, *_ = annotated
    output = tmp_path / "export"
    original_rename = Path.rename
    foreign = '{"type": "identity_development_candidates", "schema_version": 1, "source": {"sha256": "other"}}'

    def compete(source, target):
        if Path(target).name == export.RECORD_NAME:
            Path(target).write_text(foreign, encoding="utf-8")
            raise FileExistsError("record appeared")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", compete)
    with pytest.raises(export.IdentityCandidatesError, match="競合"):
        export.materialize(run_path, output)
    assert (output / export.RECORD_NAME).read_text("utf-8") == foreign
    assert not list(output.rglob("*.tmp"))
