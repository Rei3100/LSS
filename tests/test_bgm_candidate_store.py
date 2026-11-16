"""BGM候補の単独来歴・区間整合性・冪等保存・失敗境界を確認する。"""
from dataclasses import replace
import json
from pathlib import Path

from mutagen.flac import FLAC
import numpy as np
import pytest
import soundfile as sf

from lss.bgm import candidate_store as store
from lss.core.session import input_sha256_of


def _setup(tmp_path, subtype="PCM_24"):
    signal = tmp_path / "signal.wav"
    samples = np.random.default_rng(12).uniform(-0.9, 0.9, (1100, 2))
    sf.write(signal, samples, 44100, subtype=subtype)
    origin = store.CandidateOrigin(
        "a" * 64, input_sha256_of(signal), "separation-r1", "runtime-r1",
        "activity-r1", 203, 1077, 44100,
    )
    destination = tmp_path / "candidates" / "candidate.flac"
    return signal, origin, destination


def _write(signal, origin, destination, **kwargs):
    return store.materialize_candidate(
        signal, destination, origin=origin, source_id="episode-1",
        source_path=signal.parent / "original.mp4", evidence={"peak_rms": 0.2}, **kwargs,
    )


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "FLOAT"])
def test_candidate_preserves_exact_crop_and_portable_provenance(tmp_path, monkeypatch, subtype):
    signal, origin, destination = _setup(tmp_path, subtype)
    monkeypatch.setattr(store, "COPY_BLOCK_FRAMES", 71)
    result = _write(signal, origin, destination)
    assert result["frames"] == 874
    assert result["publication_state"] == "not_published"
    saved, sr = sf.read(destination, dtype="float64", always_2d=True)
    original, _ = sf.read(signal, dtype="float64", always_2d=True)
    assert sr == 44100 and saved.shape == (874, 2)
    np.testing.assert_allclose(saved, original[203:1077], rtol=0, atol=2 ** -23)
    moved = tmp_path / "moved.flac"
    destination.rename(moved)
    tags = FLAC(moved)
    assert tags["LSS_SOURCE_TYPE"] == ["bgm_candidate"]
    record = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    assert record["origin"]["source_sha256"] == "a" * 64
    assert record["origin"]["start_frame"] == 203
    assert record["origin"]["end_frame"] == 1077
    assert record["start_seconds"] == 203 / 44100
    assert record["end_seconds"] == 1077 / 44100
    assert record["semantic_role"] == "unresolved_music_like"
    assert record["identity_state"] == "unresolved"
    assert record["publication_state"] == "not_published"


def test_repeated_save_reuses_valid_candidate_without_rewriting(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    first = _write(signal, origin, destination)
    before = destination.read_bytes()
    assert _write(signal, origin, destination) == first
    assert destination.read_bytes() == before
    assert not list(destination.parent.glob("*.tmp"))


@pytest.mark.parametrize("subtype", ["PCM_24", "FLOAT"])
def test_lower_precision_audio_with_matching_crop_and_tags_is_not_reused(tmp_path, subtype):
    signal, origin, destination = _setup(tmp_path, subtype)
    _write(signal, origin, destination)
    tags = dict(FLAC(destination))
    values = sf.read(destination, always_2d=True)[0]
    sf.write(destination, values, 44100, subtype="PCM_16")
    saved = FLAC(destination)
    saved.update(tags)
    saved.save()
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="保存精度"):
        _write(signal, origin, destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("channels", [1, 6])
def test_candidate_rejects_non_stereo_before_creating_output(tmp_path, channels):
    signal, origin, destination = _setup(tmp_path)
    sf.write(signal, np.ones((1100, channels)) * 0.1, 44100, subtype="FLOAT")
    origin = replace(origin, signal_sha256=input_sha256_of(signal))
    with pytest.raises(ValueError, match="完成ステレオ"):
        _write(signal, origin, destination)
    assert not destination.parent.exists()


def test_candidate_is_excluded_from_folder_inputs_and_can_be_explicitly_added(tmp_path):
    from lss.core.session import iter_input_files

    signal, origin, destination = _setup(tmp_path)
    _write(signal, origin, destination)
    assert destination not in iter_input_files([tmp_path])
    assert iter_input_files([destination]) == [destination]


@pytest.mark.parametrize("change", [
    {"source_sha256": "b" * 64}, {"signal_sha256": "b" * 64},
    {"separation_recipe_id": "r2"}, {"runtime_id": "r2"},
    {"activity_recipe_id": "r2"}, {"start_frame": 204}, {"end_frame": 1078},
    {"sample_rate": 48000},
    {"time_offset_numerator": 7, "time_offset_denominator": 1000},
])
def test_candidate_identity_includes_every_output_choice(tmp_path, change):
    _, origin, _ = _setup(tmp_path)
    assert replace(origin, **change).candidate_id != origin.candidate_id


def test_source_media_offset_is_preserved_in_candidate_timestamps(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    origin = replace(origin, time_offset_numerator=7, time_offset_denominator=1000)
    _write(signal, origin, destination)
    provenance = json.loads(FLAC(destination)["LSS_CANDIDATE_PROVENANCE"][0])
    assert provenance["start_seconds"] == 203 / 44100 + 0.007
    assert provenance["end_seconds"] == 1077 / 44100 + 0.007
    assert provenance["origin"]["time_offset_numerator"] == 7
    assert provenance["origin"]["time_offset_denominator"] == 1000


@pytest.mark.parametrize("boundary", ["read", "verify", "flush", "rename"])
def test_interrupted_save_leaves_no_final_or_partial_candidate(tmp_path, monkeypatch, boundary):
    signal, origin, destination = _setup(tmp_path)
    before = signal.read_bytes()
    read = sf.SoundFile.read
    rename = Path.rename

    def fail(*args, **kwargs):
        raise OSError("保存中断")

    if boundary == "read":
        def fail_read(audio, *args, **kwargs):
            if Path(audio.name) == signal:
                fail()
            return read(audio, *args, **kwargs)
        monkeypatch.setattr(sf.SoundFile, "read", fail_read)
    elif boundary == "verify":
        monkeypatch.setattr(store, "_verify_crop", fail)
    elif boundary == "flush":
        monkeypatch.setattr(store.os, "fsync", fail)
    else:
        monkeypatch.setattr(Path, "rename", lambda path, target: fail() if Path(target) == destination else rename(path, target))
    with pytest.raises(OSError, match="保存中断"):
        _write(signal, origin, destination)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))
    assert signal.read_bytes() == before


def test_different_existing_candidate_is_never_overwritten(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    _write(signal, origin, destination)
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        _write(signal, replace(origin, source_sha256="b" * 64), destination)
    assert destination.read_bytes() == before


def test_corrupt_audio_with_correct_tags_is_not_reused(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    _write(signal, origin, destination)
    tags = dict(FLAC(destination))
    sf.write(destination, np.zeros((874, 2)), 44100, subtype="PCM_24")
    saved = FLAC(destination)
    saved.update(tags)
    saved.save()
    with pytest.raises(ValueError, match="音声"):
        _write(signal, origin, destination)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 1.2, -1.2])
def test_unsafe_float_signal_is_not_published(tmp_path, invalid):
    signal, origin, destination = _setup(tmp_path, "FLOAT")
    samples, sr = sf.read(signal, always_2d=True)
    samples[500, 1] = invalid
    sf.write(signal, samples, sr, subtype="FLOAT")
    origin = replace(origin, signal_sha256=input_sha256_of(signal))
    with pytest.raises(ValueError, match="クリップ"):
        _write(signal, origin, destination)
    assert not destination.exists()


@pytest.mark.parametrize("peak", [1.0, -1.0, 1.2, -1.2, 2.0, -2.0, 4.1])
def test_explicit_headroom_candidate_preserves_scaled_crop_and_provenance(tmp_path, peak):
    signal, origin, destination = _setup(tmp_path, "FLOAT")
    values, rate = sf.read(signal, always_2d=True)
    values[500, 1] = peak
    sf.write(signal, values, rate, subtype="FLOAT")
    origin = replace(origin, signal_sha256=input_sha256_of(signal),
                     serialization_policy="declared_power_of_two_headroom_v1")
    before = signal.read_bytes()
    record = _write(signal, origin, destination)
    saved, _ = sf.read(destination, dtype="float64", always_2d=True)
    actual_source, _ = sf.read(signal, dtype="float64", always_2d=True)
    serialization = record["serialization"]
    scale = serialization["gain"]
    assert scale == 2.0 ** -serialization["headroom_shift"]
    np.testing.assert_allclose(saved, actual_source[203:1077] * scale, rtol=0, atol=2 ** -23)
    assert np.max(saved) < 1 and np.min(saved) >= -1
    tags = FLAC(destination)
    metadata = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    assert metadata["schema_version"] == 3
    assert metadata["serialization"] == serialization
    assert metadata["origin"]["serialization_policy"] == origin.serialization_policy
    assert tags["LSS_OUTPUT_HEADROOM_SHIFT"] == [str(serialization["headroom_shift"])]
    assert record["candidate_asset_id"].startswith("bgm-candidate-v3-")
    assert signal.read_bytes() == before
    assert _write(signal, origin, destination) == record


def test_headroom_policy_is_in_identity_and_native_v2_identity_is_compatible(tmp_path):
    import hashlib
    from dataclasses import asdict

    _, origin, _ = _setup(tmp_path)
    legacy = asdict(origin)
    legacy.pop("serialization_policy")
    payload = {"format_revision": 2, **legacy}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert origin.candidate_id == f"bgm-candidate-v2-{expected}"
    assert origin.to_dict() == legacy
    changed = replace(origin, serialization_policy="declared_power_of_two_headroom_v1")
    assert changed.candidate_id != origin.candidate_id


def test_tampered_headroom_cannot_reuse_candidate_with_valid_id(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    origin = replace(origin, serialization_policy="declared_power_of_two_headroom_v1")
    _write(signal, origin, destination)
    tags = FLAC(destination)
    tags["LSS_OUTPUT_HEADROOM_SHIFT"] = "1"
    tags.save()
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="headroom"):
        _write(signal, origin, destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_explicit_headroom_still_refuses_nonfinite_audio(tmp_path, invalid):
    signal, origin, destination = _setup(tmp_path, "FLOAT")
    values, rate = sf.read(signal, always_2d=True)
    values[500, 1] = invalid
    sf.write(signal, values, rate, subtype="FLOAT")
    origin = replace(origin, signal_sha256=input_sha256_of(signal),
                     serialization_policy="declared_power_of_two_headroom_v1")
    with pytest.raises(ValueError, match="クリップ"):
        _write(signal, origin, destination)
    assert not destination.exists()


@pytest.mark.parametrize("key,value", [
    ("schema_version", 99), ("candidate_id", "another-candidate"),
    ("start_seconds", 0), ("end_seconds", 999), ("timebase", "another-timeline"),
    ("semantic_role", "dramatic_score"), ("identity_state", "known_exact"),
    ("publication_state", "published"), ("source_id", ""), ("source_path", None),
    ("evidence", None), ("evidence", {"invalid": float("nan")}),
])
def test_same_audio_and_id_cannot_hide_corrupt_portable_provenance(tmp_path, key, value):
    signal, origin, destination = _setup(tmp_path)
    _write(signal, origin, destination)
    tags = FLAC(destination)
    stored = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    stored[key] = value
    tags["LSS_CANDIDATE_PROVENANCE"] = json.dumps(stored)
    tags.save()
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="来歴"):
        _write(signal, origin, destination)
    assert destination.read_bytes() == before
    assert not list(destination.parent.glob("*.tmp"))


@pytest.mark.parametrize("key,value", [("LSS_SOURCE_TYPE", "separation"),
                                       ("LSS_INPUT_SHA256", "b" * 64)])
def test_same_audio_and_id_cannot_hide_changed_source_tags(tmp_path, key, value):
    signal, origin, destination = _setup(tmp_path)
    _write(signal, origin, destination)
    tags = FLAC(destination)
    tags[key] = value
    tags.save()
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="出典タグ"):
        _write(signal, origin, destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize("values", [[], ["{}", "{}"], ["[]"], ["null"], ["invalid-json"]])
def test_missing_ambiguous_and_malformed_portable_provenance_is_rejected(tmp_path, values):
    signal, origin, destination = _setup(tmp_path)
    _write(signal, origin, destination)
    tags = FLAC(destination)
    if values:
        tags["LSS_CANDIDATE_PROVENANCE"] = values
    else:
        del tags["LSS_CANDIDATE_PROVENANCE"]
    tags.save()
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="来歴"):
        _write(signal, origin, destination)
    assert destination.read_bytes() == before


def test_a_boolean_origin_frame_cannot_impersonate_integer_one(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    origin = replace(origin, start_frame=1)
    _write(signal, origin, destination)
    tags = FLAC(destination)
    stored = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    stored["origin"]["start_frame"] = True
    tags["LSS_CANDIDATE_PROVENANCE"] = json.dumps(stored)
    tags.save()
    with pytest.raises(ValueError, match="来歴"):
        _write(signal, origin, destination)


def test_relocating_files_or_renaming_source_does_not_rewrite_valid_provenance(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    first = _write(signal, origin, destination)
    before = destination.read_bytes()
    relocated = tmp_path / "relocated.wav"
    signal.rename(relocated)
    reused = store.materialize_candidate(
        relocated, destination, origin=origin, source_id="renamed-episode",
        source_path=tmp_path / "moved-original.mp4", evidence={"new_observation": True})
    assert reused == first
    assert destination.read_bytes() == before
    stored = json.loads(FLAC(destination)["LSS_CANDIDATE_PROVENANCE"][0])
    assert stored["source_id"] == "episode-1" and stored["evidence"] == {"peak_rms": 0.2}


@pytest.mark.parametrize("evidence", [None, [], "invalid", {"metric": float("nan")}])
def test_invalid_new_evidence_cannot_leave_a_final_candidate(tmp_path, evidence):
    signal, origin, destination = _setup(tmp_path)
    before = signal.read_bytes()
    with pytest.raises(ValueError):
        store.materialize_candidate(signal, destination, origin=origin, source_id="episode-1",
                                    source_path=tmp_path / "original.mp4", evidence=evidence)
    assert not destination.exists()
    assert signal.read_bytes() == before
    assert not list(destination.parent.glob("*.tmp"))


def test_false_timestamp_cannot_impersonate_zero_seconds(tmp_path):
    signal, origin, destination = _setup(tmp_path)
    origin = replace(origin, start_frame=0)
    _write(signal, origin, destination)
    tags = FLAC(destination)
    stored = json.loads(tags["LSS_CANDIDATE_PROVENANCE"][0])
    stored["start_seconds"] = False
    tags["LSS_CANDIDATE_PROVENANCE"] = json.dumps(stored)
    tags.save()
    with pytest.raises(ValueError, match="来歴"):
        _write(signal, origin, destination)


def test_fresh_candidate_is_crop_verified_once_and_record_equals_the_double_verified_one(tmp_path, monkeypatch):
    signal, origin, destination = _setup(tmp_path)
    verified = []
    real = store._verify_crop
    monkeypatch.setattr(store, "_verify_crop", lambda s, c, o: verified.append(Path(c).name) or real(s, c, o))
    record = _write(signal, origin, destination)
    assert len(verified) == 1 and verified[0].startswith(".lss-candidate-")
    verified.clear()
    # reuse of an existing file still verifies it
    assert _write(signal, origin, destination) == record
    assert verified == ["candidate.flac"]
    store._verify_crop(signal, destination, origin)
