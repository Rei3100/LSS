"""実番組の時刻境界・非破壊保留保存・読戻し・中断と遅い入力変更を確認する。"""
from dataclasses import replace
import json
from pathlib import Path

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss.bgm import identity, identity_run
from lss.bgm.recording_recovery import RecordingRecoveryRecipe
from lss.bgm.reference_residual import ReferenceResidualRecipe
from lss.bgm.reference_alignment import ReferenceAlignmentRecipe
from lss.harness.fetch import file_sha256
from lss.validation import reference_residual_episode as stage
from tests.test_bgm_recording_recovery import _match
from tests.test_held_bgm_alternatives import upstream
from tests.test_routed_episode_discovery import write


@pytest.fixture
def case(tmp_path, monkeypatch, request):
    # 完成mixの音声開始位置を持つrun。照合の時刻も元メディアの時刻。
    timeline = {"offset_numerator": 7, "offset_denominator": 1000, "offset_seconds": 0.007,
                "method": "fixture_decoded_audio_pts"}
    from lss.validation import routed_episode_discovery
    from types import SimpleNamespace
    if getattr(request, "param", None) == "FLOAT":
        write_audio = sf.write
        def float_music(path, *args, **kwargs):
            if Path(path).name == "music.wav":
                kwargs["subtype"] = "FLOAT"
            return write_audio(path, *args, **kwargs)
        monkeypatch.setattr(sf, "write", float_music)
    monkeypatch.setattr(routed_episode_discovery, "probe_audio_timeline",
                        lambda path: SimpleNamespace(to_dict=lambda: timeline))
    run = upstream(tmp_path)
    result = json.loads(run.read_bytes())
    reference = tmp_path / "reference.wav"
    source, rate = sf.read(tmp_path / "source.wav", always_2d=True)
    sf.write(reference, source, rate, subtype="DOUBLE")
    match = replace(_match(0.007 + 128 / rate, 0.007 + 896 / rate, 128 / rate),
                    reference_path=str(reference), reference_sha256=file_sha256(reference))
    query, kind, query_hash = identity_run._query_signal(result)
    annotation = {"type": "episode_identity_annotation", "schema_version": 1,
                  "intent": {**identity_run.evidence_settings(), "run_sha256": file_sha256(run),
                             "references": [stage.artifact(reference)]},
                  "query_signal": {"kind": kind, "sha256": query_hash}, "matches": [match.to_dict()],
                  "activity_runs": [{"recipe_id": key,
                                     "candidates": identity.split_candidates_at_matches(candidates, [match])}
                                    for key, candidates in identity_run.candidate_sets(result).items()]}
    write(run.parent / "identity.json", annotation)
    recipe = ReferenceResidualRecipe(fit_recipe=RecordingRecoveryRecipe(fir_taps=3,
        fit_window_frames=256, fit_hop_frames=128, boundary_fade_frames=0,
        maximum_alignment_refinement_frames=0), boundary_fade_frames=20)
    return run, tmp_path / "music-run.json", recipe


def _freeze(case, tmp_path):
    return stage.freeze(case[0], case[1], tmp_path / "residual", recipe=case[2])


def test_optional_alignment_recipe_is_frozen_and_read_back_without_reexecution(case, tmp_path, monkeypatch):
    recipe = replace(case[2], alignment_recipe=ReferenceAlignmentRecipe(probe_frames=128, minimum_probe_frames=16))
    path = stage.freeze(case[0], case[1], tmp_path / "aligned", recipe=recipe)
    plan = json.loads(path.read_bytes())
    assert stage._recipe(plan["recipe"]).to_dict() == recipe.to_dict()
    alignment_path = Path(stage.reference_alignment.__file__).resolve()
    assert plan["implementation"][str(alignment_path.relative_to(stage.ROOT))] == file_sha256(alignment_path)
    result = stage.run(path)
    measured = json.loads(result.read_bytes())["matches"][0]["observations"]
    assert measured["speed_alignment"]["speed_correction_used"] is False
    monkeypatch.setattr(stage, "_render", lambda *a: pytest.fail("読戻しに再実行は不要"))
    assert stage.read_result(case[0], result)["publication_allowed"] is False


def test_media_offset_is_subtracted_and_unmatched_music_and_parent_cues_are_preserved(case, tmp_path):
    plan = _freeze(case, tmp_path)
    fixed = json.loads(plan.read_bytes())
    assert fixed["cases"][0]["start_frame"] == 128 and fixed["cases"][0]["end_frame"] == 896
    parents = json.loads(case[0].read_bytes())["candidates"]
    before = case[0].read_bytes(), [Path(p["asset"]["path"]).read_bytes() for p in parents]
    report_path = stage.run(plan)
    report = json.loads(report_path.read_bytes())
    candidate, _ = sf.read(report["signal"]["path"], always_2d=True)
    source, _ = sf.read(tmp_path / "source.wav", always_2d=True)
    base, _ = sf.read(tmp_path / "music.wav", always_2d=True)
    np.testing.assert_allclose(candidate[160:850], source[160:850], rtol=0, atol=0.002)
    assert np.array_equal(candidate[:128], base[:128]) and np.array_equal(candidate[896:], base[896:])
    view = stage.read_result(case[0], report_path)
    assert len(view["artifacts"]) == 1 and view["publication_allowed"] is False
    assert view["artifacts"][0]["primary_candidate_asset_id"] == parents[0]["asset"]["candidate_asset_id"]
    assert report["semantic_ground_truth"] is None and report["new_inference_performed"] is False
    assert report["matches"][0]["observations"]["alignment_from_original_only"]
    assert (case[0].read_bytes(), [Path(p["asset"]["path"]).read_bytes() for p in parents]) == before


def test_readback_uses_no_renderer_and_no_record_update(case, tmp_path, monkeypatch):
    report = stage.run(_freeze(case, tmp_path))
    before = report.read_bytes()
    monkeypatch.setattr(stage, "_render", lambda *a: pytest.fail("読戻しで描画しない"))
    monkeypatch.setattr(stage.reference_residual, "recover_reference_residual", lambda *a: pytest.fail("再推定しない"))
    assert stage.read_result(case[0], report)["publication_allowed"] is False
    assert report.read_bytes() == before


def test_audio_changed_during_the_final_input_recheck_is_not_returned(case, tmp_path, monkeypatch):
    report = stage.run(_freeze(case, tmp_path))
    candidate = Path(json.loads(report.read_bytes())["candidates"][0]["asset"]["path"])
    verify = stage._verify
    calls = 0
    def late(*args, **kwargs):
        nonlocal calls
        verify(*args, **kwargs)
        calls += 1
        if calls == 2:
            candidate.write_bytes(candidate.read_bytes() + b"late")
    monkeypatch.setattr(stage, "_verify", late)
    with pytest.raises(stage.ReferenceResidualResultError, match="音声内容"):
        stage.read_result(case[0], report)


@pytest.mark.parametrize("damage", ["source_audio", "music", "reference", "identity", "run", "primary"])
def test_changed_inputs_stop_before_final_record(case, tmp_path, damage):
    plan = _freeze(case, tmp_path)
    target = {"source_audio": tmp_path / "source.wav", "music": tmp_path / "music.wav",
              "reference": tmp_path / "reference.wav", "identity": case[0].parent / "identity.json",
              "run": case[0], "primary": Path(json.loads(case[0].read_bytes())["candidates"][0]["asset"]["path"])}[damage]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="内容"):
        stage.run(plan)
    assert not (plan.parent / "residuals.json").exists()


def test_interrupted_candidate_save_resumes_and_retains_originals(case, tmp_path):
    plan = _freeze(case, tmp_path)
    def stop(saved):
        raise InterruptedError("中断")
    with pytest.raises(InterruptedError):
        stage.run(plan, on_candidate=stop)
    assert not (plan.parent / "residuals.json").exists()
    candidate = next((plan.parent / "candidates").glob("*.flac"))
    before = candidate.read_bytes()
    report = stage.run(plan)
    assert candidate.read_bytes() == before
    assert stage.read_result(case[0], report)["artifacts"]


@pytest.mark.parametrize("case", ["FLOAT"], indirect=True)
def test_float_signal_is_reused_after_interrupt_without_rerender_or_duplicate_candidates(case, tmp_path, monkeypatch):
    plan = _freeze(case, tmp_path)
    with pytest.raises(InterruptedError):
        stage.run(plan, on_candidate=lambda saved: (_ for _ in ()).throw(InterruptedError("中断")))
    rendering = plan.parent / "rendering.json"
    receipt_before = rendering.read_bytes()
    candidate = next((plan.parent / "candidates").glob("*.flac")); before = candidate.read_bytes()
    monkeypatch.setattr(stage, "_render_fresh", lambda *a: pytest.fail("FLOAT信号を再描画しない"))
    report = stage.run(plan)
    assert rendering.read_bytes() == receipt_before and candidate.read_bytes() == before
    assert len(list((plan.parent / "candidates").glob("*.flac"))) == 1
    assert len(stage.read_result(case[0], report)["artifacts"]) == 1


@pytest.mark.parametrize("damage", ["signal", "receipt", "plan_hash", "missing_span"])
def test_invalid_cached_rendering_cannot_replace_previous_completion(case, tmp_path, damage):
    plan = _freeze(case, tmp_path); report = stage.run(plan); before = report.read_bytes()
    receipt_path = plan.parent / "rendering.json"; receipt = json.loads(receipt_path.read_bytes())
    if damage == "signal":
        signal = Path(receipt["signal"]["path"]); signal.write_bytes(signal.read_bytes() + b"changed")
    elif damage == "receipt":
        receipt_path.write_bytes(b"null")
    else:
        if damage == "plan_hash":
            receipt["plan_sha256"] = "other"
        else:
            receipt["matches"] = []
        write(receipt_path, receipt)
    with pytest.raises(ValueError):
        stage.run(plan)
    assert report.read_bytes() == before


def test_late_input_change_preserves_previous_receipt_and_audio(case, tmp_path, monkeypatch):
    plan = _freeze(case, tmp_path)
    report = stage.run(plan)
    before = report.read_bytes()
    candidate = Path(json.loads(before)["candidates"][0]["asset"]["path"])
    candidate_before = candidate.read_bytes()
    writer = stage._write_record
    def late(path, payload, validate):
        def check():
            reference = tmp_path / "reference.wav"
            reference.write_bytes(reference.read_bytes() + b"late change")
            validate()
        writer(path, payload, check)
    monkeypatch.setattr(stage, "_write_record", late)
    with pytest.raises(ValueError, match="内容"):
        stage.run(plan)
    assert report.read_bytes() == before and candidate.read_bytes() == candidate_before


def test_changed_audio_with_replaced_hash_still_fails_crop_verification(case, tmp_path):
    report_path = stage.run(_freeze(case, tmp_path))
    report = json.loads(report_path.read_bytes())
    asset = report["candidates"][0]["asset"]
    path = Path(asset["path"])
    tags = dict(FLAC(path))
    sf.write(path, np.zeros((1024, 2)), 44100, subtype="PCM_24")
    flac = FLAC(path); flac.update(tags); flac.save()
    asset["sha256"] = file_sha256(path)
    write(report_path, report)
    with pytest.raises(ValueError, match="音声"):
        stage.read_result(case[0], report_path)


def test_no_positive_match_is_unresolved_and_keeps_the_entire_music(case, tmp_path):
    annotation_path = case[0].parent / "identity.json"
    annotation = json.loads(annotation_path.read_bytes()); annotation["matches"] = []
    write(annotation_path, annotation)
    report_path = stage.run(_freeze(case, tmp_path))
    report = json.loads(report_path.read_bytes())
    assert report["outcome"] == "no_positive_recording_spans" and report["candidates"] == []
    assert report["semantic_ground_truth"] is None and report["selection_state"] == "unresolved"
    music, _ = sf.read(tmp_path / "music.wav", always_2d=True)
    signal, _ = sf.read(report["signal"]["path"], always_2d=True)
    assert np.array_equal(signal, music)


def test_overlapping_alignments_and_unresolved_matches_keep_parent_music(case, tmp_path):
    annotation_path = case[0].parent / "identity.json"
    annotation = json.loads(annotation_path.read_bytes())
    original = annotation["matches"][0]
    overlapping = {**original, "query_start_seconds": 0.007 + 512 / 44100,
                   "query_end_seconds": 0.007 + 960 / 44100, "reference_start_seconds": 512 / 44100}
    unresolved = {**original, "identity_state": "unresolved"}
    annotation["matches"].extend([overlapping, unresolved])
    write(annotation_path, annotation)
    plan_path = _freeze(case, tmp_path)
    plan = json.loads(plan_path.read_bytes())
    assert [(c["start_frame"], c["end_frame"]) for c in plan["cases"]] == [(128, 512), (896, 960)]
    assert plan["retained_ambiguous_spans"][0]["start_frame"] == 512
    assert plan["unresolved_match_count"] == 1
    report = json.loads(stage.run(plan_path).read_bytes())
    signal, _ = sf.read(report["signal"]["path"], always_2d=True)
    parent, _ = sf.read(tmp_path / "music.wav", always_2d=True)
    assert np.array_equal(signal[512:896], parent[512:896])


def test_reference_rate_conversion_is_fixed_and_keeps_channels_and_seconds(tmp_path):
    from scipy.signal import resample_poly
    reference = np.zeros((4800, 2)); reference[1600, 0] = 0.4; reference[3200, 1] = 0.8
    path = tmp_path / "reference-48k.wav"
    sf.write(path, reference, 48000, subtype="DOUBLE")
    converted = stage._read_reference(path, {"sample_rate": 48000, "frames": 4800})
    assert converted.shape == (2, 4410)
    np.testing.assert_array_equal(converted, resample_poly(reference, 147, 160, axis=0,
                                                         window=("kaiser", 5.0), padtype="constant").T)
    assert np.argmax(converted[0]) == 1470 and np.argmax(converted[1]) == 2940


@pytest.mark.parametrize("damage", ["overlap", "time_outside", "wrong_reference", "stale_settings", "other_source"])
def test_ambiguous_or_stale_identity_and_preparation_cannot_freeze(case, tmp_path, damage):
    annotation_path = case[0].parent / "identity.json"
    annotation = json.loads(annotation_path.read_bytes())
    if damage == "overlap":
        annotation["matches"] *= 2
    elif damage == "time_outside":
        annotation["matches"][0]["query_end_seconds"] = 100
    elif damage == "wrong_reference":
        annotation["matches"][0]["reference_sha256"] = "b" * 64
    elif damage == "stale_settings":
        annotation["intent"]["matching_settings_id"] = "old"
    else:
        prep = json.loads(case[1].read_bytes()); prep["source"]["sha256"] = "b" * 64; write(case[1], prep)
    write(annotation_path, annotation)
    with pytest.raises(ValueError):
        _freeze(case, tmp_path)
    assert not (tmp_path / "residual").exists()
