import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lss.validation.bgm_corpus import load_bgm_corpus
from lss.validation.controlled_mix import InvalidControlledMixError, build_controlled_corpus
from lss.validation import controlled_mix


def _source(path, *, sample_rate=8000, amplitude=0.2):
    t = np.arange(sample_rate) / sample_rate
    sf.write(path, amplitude * np.sin(2 * np.pi * 220 * t), sample_rate, subtype="FLOAT")


def _plan(source, **case_overrides):
    case = {
        "case_id": "known-score",
        "description": "既知scoreとdialogue代替信号のcontrolled mix",
        "duration_frames": 2000,
        "expected_outcome": "wanted_present",
        "attributes": ["target_present"],
        "components": [{
            "component_id": "score-1",
            "path": str(source),
            "source_start_frame": 100,
            "source_end_frame": 500,
            "start_frame": 100,
            "gain_db": -6.0,
            "fade_frames": 10,
            "occurrence": {
                "occurrence_id": "score-occ-1",
                "semantic_role": "dramatic_score",
                "wanted": True,
                "novel": False,
                "identity_state": "known_exact",
                "cue_identity_id": "score-1",
                "audio_version_id": "score-1-file",
            },
        }],
    }
    case.update(case_overrides)
    return {
        "schema_version": 1,
        "corpus_id": "controlled-test",
        "revision": "r1",
        "description": "test corpus",
        "sample_rate": 16000,
        "channels": 2,
        "cases": [case],
    }


def test_builds_verified_corpus_with_exact_resampled_occurrence(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(source)), encoding="utf-8")

    manifest, construction = build_controlled_corpus(plan, tmp_path / "corpus")

    corpus = load_bgm_corpus(manifest)
    case = corpus.cases[0]
    assert (case.sample_rate, case.frames) == (16000, 2000)
    assert [(item.start_frame, item.end_frame) for item in case.occurrences] == [(100, 900)]
    assert sf.info(case.source_path).channels == 2
    construction_data = json.loads(construction.read_text(encoding="utf-8"))
    assert construction_data["cases"][0]["components"][0]["source_sha256"]
    assert construction_data["runtime"]["resampler"] == "scipy.signal.resample_poly-default-v1"
    assert any(item.startswith("construction_sha256=") for item in case.attributes)
    assert not list(tmp_path.glob(".corpus-*.tmp"))


def test_target_absent_case_with_unlabelled_dialogue_component_is_valid(tmp_path):
    source = tmp_path / "dialogue.wav"
    _source(source, sample_rate=16000, amplitude=0.1)
    case = _plan(source)["cases"][0]
    case.update({
        "case_id": "target-absent",
        "expected_outcome": "no_wanted_score",
        "attributes": ["target_absent"],
    })
    case["components"][0]["occurrence"] = None
    plan_data = _plan(source)
    plan_data["cases"] = [case]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    manifest, _ = build_controlled_corpus(plan, tmp_path / "corpus")

    assert load_bgm_corpus(manifest).cases[0].occurrences == ()


def test_builds_complete_clean_role_references_that_reconstruct_mix(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    plan_data = _plan(source)
    plan_data["cases"][0]["components"][0]["reference_role"] = "music"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    manifest, construction = build_controlled_corpus(plan, tmp_path / "corpus")

    case = load_bgm_corpus(manifest).cases[0]
    record = json.loads(construction.read_text(encoding="utf-8"))["cases"][0]
    assert record["reference_sum_max_abs_error"] == 0.0
    assert set(record["references"]) == {"music", "dialogue", "sfx"}
    mix, _ = sf.read(case.source_path, dtype="float64", always_2d=True)
    reference_sum = np.zeros_like(mix)
    for item in record["references"].values():
        data, _ = sf.read(
            construction.parent / item["path"], dtype="float64", always_2d=True,
        )
        reference_sum += data
    np.testing.assert_allclose(reference_sum, mix, atol=2.0 / 2**23)


def test_rejects_partial_clean_reference_assignment(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    plan_data = _plan(source)
    first = plan_data["cases"][0]["components"][0]
    first["reference_role"] = "music"
    second = dict(first, component_id="dialogue-1", start_frame=1000)
    second.pop("reference_role")
    second["occurrence"] = None
    plan_data["cases"][0]["components"].append(second)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(InvalidControlledMixError, match="全componentへreference_role"):
        build_controlled_corpus(plan, tmp_path / "corpus")


def test_rejects_unknown_clean_reference_role(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    plan_data = _plan(source)
    plan_data["cases"][0]["components"][0]["reference_role"] = "ambience"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(InvalidControlledMixError, match="reference_roleが不正"):
        build_controlled_corpus(plan, tmp_path / "corpus")


def test_clipping_rejects_entire_corpus_without_partial_publication(tmp_path):
    source = tmp_path / "source.wav"
    _source(source, amplitude=0.8)
    plan_data = _plan(source)
    component = plan_data["cases"][0]["components"][0]
    component.update({"source_start_frame": 0, "source_end_frame": 8000, "gain_db": 6.0})
    plan_data["cases"][0]["duration_frames"] = 17000
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_data), encoding="utf-8")

    with pytest.raises(InvalidControlledMixError, match="clipping"):
        build_controlled_corpus(plan, tmp_path / "corpus")

    assert not (tmp_path / "corpus").exists()
    assert not list(tmp_path.glob(".corpus-*.tmp"))


def test_existing_revision_is_never_overwritten(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(source)), encoding="utf-8")
    output = tmp_path / "corpus"
    build_controlled_corpus(plan, output)

    with pytest.raises(FileExistsError):
        build_controlled_corpus(plan, output)


def test_source_change_during_pinning_rejects_entire_corpus(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    _source(source)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_plan(source)), encoding="utf-8")
    real_fstat = os.fstat
    calls = 0

    def changed_fstat(fd):
        nonlocal calls
        value = real_fstat(fd)
        calls += 1
        if calls == 2:
            return SimpleNamespace(
                st_dev=value.st_dev, st_ino=value.st_ino, st_size=value.st_size,
                st_mtime_ns=value.st_mtime_ns + 1, st_mode=value.st_mode,
            )
        return value

    monkeypatch.setattr(controlled_mix.os, "fstat", changed_fstat)

    with pytest.raises(InvalidControlledMixError, match="コピー中に変更"):
        build_controlled_corpus(plan, tmp_path / "corpus")

    assert not (tmp_path / "corpus").exists()
    assert not list(tmp_path.glob(".corpus-*.tmp"))


def _ducked_plan(source, windows):
    plan = _plan(source)
    plan["cases"][0]["components"][0]["ducking"] = windows
    plan["cases"][0]["components"][0]["fade_frames"] = 0
    return plan


def _built_audio(tmp_path, plan, name="ducked"):
    plan_path = tmp_path / f"{name}-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    manifest, construction = build_controlled_corpus(plan_path, tmp_path / name)
    audio, _ = sf.read(manifest.parent / "audio" / "known-score.flac", always_2d=True)
    record = json.loads(construction.read_text(encoding="utf-8"))
    return audio, record


def test_ducking_lowers_only_the_named_window(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    plain_plan = _plan(source)
    plain_plan["cases"][0]["components"][0]["fade_frames"] = 0
    plain, _ = _built_audio(tmp_path, plain_plan, name="plain")
    ducked, record = _built_audio(tmp_path, _ducked_plan(source, [
        {"start_frame": 200, "end_frame": 400, "depth_db": -12.0, "ramp_frames": 0},
    ]), name="ducked")
    # componentは出力の100frameから始まる。下げた区間だけが元と変わる。
    assert np.max(np.abs(ducked[100:300])) > 0
    assert np.allclose(ducked[100:300], plain[100:300], atol=1e-6)
    assert np.allclose(ducked[500:600], plain[500:600], atol=1e-6)
    lowered = np.max(np.abs(ducked[300:500]))
    assert lowered <= np.max(np.abs(plain[300:500])) * 10 ** (-12.0 / 20.0) + 1e-9
    assert lowered > 0
    assert record["cases"][0]["components"][0]["ducking"] == [
        {"start_frame": 200, "end_frame": 400, "depth_db": -12.0, "ramp_frames": 0}]


def test_ducking_ramps_between_full_and_lowered_level(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    ducked, _ = _built_audio(tmp_path, _ducked_plan(source, [
        {"start_frame": 200, "end_frame": 300, "depth_db": -20.0, "ramp_frames": 50},
    ]), name="ramped")
    # 傾斜の途中は、下げきった値と1.0の間にある。
    lowered = 10 ** (-20.0 / 20.0)
    middle = np.max(np.abs(ducked[275:285])) / max(np.max(np.abs(ducked[100:140])), 1e-12)
    assert lowered < middle < 1.0


def test_ducking_rejects_broken_windows(tmp_path):
    source = tmp_path / "source.wav"
    _source(source)
    for windows in (
        [],
        [{"start_frame": 200, "end_frame": 200, "depth_db": -6.0, "ramp_frames": 0}],
        [{"start_frame": 200, "end_frame": 300, "depth_db": 3.0, "ramp_frames": 0}],
        [{"start_frame": 200, "end_frame": 300, "depth_db": -6.0, "ramp_frames": 0},
         {"start_frame": 250, "end_frame": 350, "depth_db": -6.0, "ramp_frames": 0}],
        [{"start_frame": 0, "end_frame": 100000, "depth_db": -6.0, "ramp_frames": 0}],
        [{"start_frame": 200, "end_frame": 300, "depth_db": -6.0}],
    ):
        plan_path = tmp_path / "broken-plan.json"
        plan_path.write_text(json.dumps(_ducked_plan(source, windows), ensure_ascii=False), encoding="utf-8")
        with pytest.raises(InvalidControlledMixError):
            build_controlled_corpus(plan_path, tmp_path / "broken")
        plan_path.unlink()
