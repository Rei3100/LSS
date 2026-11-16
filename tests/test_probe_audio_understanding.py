import hashlib
import json
import os
import struct
import types
from pathlib import Path

import pytest

from scripts import probe_audio_understanding as probe


def test_standalone_thinker_retains_its_stop_tokens_without_mutating_outer_defaults():
    outer = {"_from_model_config": True, "transformers_version": "4.45.2"}
    tokens = {"bos_token_id": 151644, "eos_token_id": 151645, "pad_token_id": 151643}
    derived = probe.thinker_generation_settings(tokens, outer)
    assert derived["eos_token_id"] == 151645
    assert derived["pad_token_id"] == 151643
    assert "eos_token_id" not in outer


def test_missing_thinker_stop_token_cannot_silently_use_outer_generation_file():
    with pytest.raises(probe.ProbeError, match="thinker_generation_token_missing:eos_token_id"):
        probe.thinker_generation_settings({"bos_token_id": 151644}, {})


def test_checkpoint_preparation_can_repeat_after_deriving_metadata(tmp_path):
    source = _model_source(tmp_path / "source")
    destination = tmp_path / "prepared"
    original_config = (source / "config.json").read_bytes()
    original_generation = (source / "generation_config.json").read_bytes()
    try:
        first = probe.build_thinker_checkpoint(source, destination)
    except probe.ProbeError as exc:
        if "hardlink_failed" in str(exc):
            pytest.skip("OS sandbox denies same-volume hardlink creation")
        raise
    second = probe.build_thinker_checkpoint(source, destination)
    assert first["derived_generation_tokens"] == second["derived_generation_tokens"]
    assert json.loads((destination / "generation_config.json").read_text())["eos_token_id"] == 151645
    assert (source / "config.json").read_bytes() == original_config
    assert (source / "generation_config.json").read_bytes() == original_generation
    assert not (source / "config.json").samefile(destination / "config.json")


@pytest.mark.parametrize("control,category", [("silence", "SILENCE"), ("white_noise", "NOISE"), ("harmonic_music", "MUSIC")])
def test_bounded_category_is_checked_against_construction(control, category):
    assert probe.classify_control_output(json.dumps({"category": category, "caption": "diagnostic"}), control)["exact_control_category"]
    assert not probe.classify_control_output(json.dumps({"category": "SPEECH", "caption": "diagnostic"}), control)["exact_control_category"]


def test_invalid_or_conflicting_category_is_never_repaired_by_caption_prose():
    assert not probe.classify_control_output('{"category":"SILENCE","caption":"music"}', "harmonic_music")["exact_control_category"]
    assert not probe.classify_control_output('Human: inventing a dialogue', "silence")["valid_json"]


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    header = {}
    payload = bytearray()
    offset = 0
    for name, data in tensors.items():
        header[name] = {"dtype": "F32", "shape": [len(data) // 4], "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _model_source(root: Path) -> Path:
    source = root / "source"
    (source / ".cache" / "huggingface" / "trees").mkdir(parents=True)
    config = {
        "architectures": ["Qwen2_5OmniModel"], "model_type": "qwen2_5_omni",
        "thinker_config": {"architectures": ["Qwen2_5OmniNaViTThinkerForConditionalGeneration"],
                           "model_type": "qwen2_5_omni_thinker", "audio_config": {}, "vision_config": {},
                           "text_config": {}, "bos_token_id": 151644, "eos_token_id": 151645, "pad_token_id": 151643},
    }
    index = {"metadata": {"total_size": 24}, "weight_map": {
        "thinker.audio_tower.weight": probe.SHARDS[0],
        "thinker.model.weight": probe.SHARDS[1],
        "talker.model.weight": probe.SHARDS[1],
        "talker.codec_head.weight": "model-00003-of-00003.safetensors",
        "token2wav.weight": "model-00003-of-00003.safetensors",
    }}
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (source / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    for name in probe.REQUIRED_FILES:
        path = source / name
        if name == "config.json":
            continue
        path.write_text("{}\n", encoding="utf-8")
    _write_safetensors(source / probe.SHARDS[0], {"thinker.audio_tower.weight": b"a" * 8})
    _write_safetensors(source / probe.SHARDS[1], {
        "thinker.model.weight": b"b" * 4,
        "talker.model.weight": b"c" * 8,
    })
    rows = {}
    for name in probe.REQUIRED_FILES + ("model.safetensors.index.json",):
        path = source / name
        rows[name] = {"size": path.stat().st_size, "blob_id": _git_blob_sha1(path)}
    for shard in probe.SHARDS:
        path = source / shard
        rows[shard] = {"size": path.stat().st_size, "lfs_sha256": probe.sha256_file(path)}
    rows["model-00003-of-00003.safetensors"] = {"size": 16, "lfs_sha256": "0" * 64}
    tree = source / ".cache" / "huggingface" / "trees" / f"{probe.MODEL_REVISION}.json"
    tree.write_text(json.dumps({"files": rows}), encoding="utf-8")
    return source


def test_synthetic_controls_are_repeatable_stereo_and_opaque(tmp_path):
    first = probe.make_controls(tmp_path / "one")
    second = probe.make_controls(tmp_path / "two")
    assert [row["input_file"] for row in first] == ["input-001.wav", "input-002.wav", "input-003.wav"]
    assert all(row["channels"] == 2 and row["sample_rate"] == 24_000 for row in first)
    assert [row["sha256"] for row in first] == [row["sha256"] for row in second]
    assert [row["truth"] for row in first] == ["no_music", "no_music", "harmonic_music"]
    assert not any(row["control"] in row["input_file"] for row in first)


def test_natural_caption_prompt_matches_cpu_trial_contract_without_runtime_import():
    rows = [{"caption": f"caption {letter}"} for letter in "ABC"]
    prompt, expected = probe._caption_prompt(rows, 1)
    assert expected == "C"
    assert json.loads(prompt.split("\n", 1)[1]) == {"candidates": [
        {"caption_id": "A", "caption": "caption C"},
        {"caption_id": "B", "caption": "caption A"},
        {"caption_id": "C", "caption": "caption B"},
        {"caption_id": "UNKNOWN"}]}


def test_natural_manifest_uses_only_first_three_prepared_rows(tmp_path):
    manifest = tmp_path / "acquisition.json"
    manifest.write_text(json.dumps({"rows": [
        {"ytid": "held", "status": "held_input_shape"},
        *[{"ytid": str(i), "caption": f"caption {i}", "status": "prepared"} for i in range(4)]
    ]}), encoding="utf-8")
    rows = probe._natural_rows(manifest)
    assert [row["ytid"] for row in rows] == ["0", "1", "2"]


def test_natural_fixture_sha_and_stereo_are_verified(tmp_path, monkeypatch):
    wav = tmp_path / "fixture.wav"
    probe._write_stereo(wav, ((0.1, -0.1),) * 100)
    row = {"fixture_path": str(wav), "fixture_sha256": probe.sha256_file(wav), "channels": 2}
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "storage_policy.json").write_text('{"hot_volume":"C:"}', encoding="utf-8")
    monkeypatch.setitem(__import__("sys").modules, "dev_hot_paths", types.SimpleNamespace(require_hot=lambda *_args: None))
    paths = probe._verify_natural_rows([row])
    assert paths == [wav.resolve()]
    audio, metadata = probe._resample_audio(wav)
    assert len(audio) > 0 and metadata["source_channels"] == 2
    row["fixture_sha256"] = "0" * 64
    with pytest.raises(probe.ProbeError, match="natural_audio_fixture_sha256_mismatch"):
        probe._verify_natural_rows([row])


def test_prepare_prunes_talker_and_token2wav_from_index(tmp_path, monkeypatch):
    source = _model_source(tmp_path)
    destination = tmp_path / "evidence" / "prepared" / "thinker"
    linked = []

    def record_link(src, dst):
        # The managed Windows sandbox denies hard-link creation, so exercise
        # the preparation/index logic with a content copy while recording each
        # link request. Production keeps the implementation hardlink-only.
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        linked.append((src.name, dst.name))

    monkeypatch.setattr(probe, "_hardlink", record_link)
    manifest = probe.build_thinker_checkpoint(source, destination)
    derived = json.loads((destination / "model.safetensors.index.json").read_text())
    config = json.loads((destination / "config.json").read_text())
    assert config["model_type"] == "qwen2_5_omni_thinker"
    assert config["architectures"] == ["Qwen2_5OmniThinkerForConditionalGeneration"]
    assert set(derived["weight_map"]) == {"thinker.audio_tower.weight", "thinker.model.weight"}
    assert derived["metadata"]["total_size"] == 12  # excludes talker tensors in mixed shard 2
    assert all((source_name, source_name) in linked for source_name in probe.SHARDS)
    derived_metadata = {"config.json", "generation_config.json"}
    assert all((name, name) in linked for name in probe.REQUIRED_FILES if name not in derived_metadata)
    assert not any((name, name) in linked for name in derived_metadata)
    assert json.loads((destination / "generation_config.json").read_text())["eos_token_id"] == 151645
    assert manifest["thinker_tensor_count"] == 2
    assert manifest["thinker_parameter_count"] == 3
    assert manifest["logical_bfloat16_parameter_bytes"] == 6
    assert manifest["bfloat16_vram_preflight_bytes_with_headroom"] == 6 + probe.MIN_FREE_VRAM
    assert manifest["source_revision"] == probe.MODEL_REVISION


def test_hardlink_failure_never_falls_back_to_model_copy(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"model bytes")

    def denied(*_args):
        raise PermissionError("hard links disabled")

    monkeypatch.setattr(probe.os, "link", denied)
    with pytest.raises(probe.ProbeError, match="hardlink_failed_no_copy_fallback"):
        probe._hardlink(source, destination)
    assert not destination.exists()


def test_prepare_rejects_thinker_weights_in_omitted_third_shard(tmp_path):
    source = _model_source(tmp_path)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["thinker.extra.weight"] = "model-00003-of-00003.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    tree_path = source / ".cache" / "huggingface" / "trees" / f"{probe.MODEL_REVISION}.json"
    tree = json.loads(tree_path.read_text())
    tree["files"]["model.safetensors.index.json"] = {
        "size": index_path.stat().st_size, "blob_id": _git_blob_sha1(index_path)}
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    with pytest.raises(probe.ProbeError, match="thinker_tensor_outside_first_two_shards"):
        probe.verify_source_metadata(source, verify_shards=True)



def test_caption_trie_shared_prefix_unknown_eos_and_fail_closed():
    class Tokenizer:
        eos_token_id = 300
        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return list(text.encode('utf-8'))
    tokenizer = Tokenizer()
    callback = probe._caption_prefix_allowed_tokens(tokenizer, 2)
    sequences = [tokenizer.encode(json.dumps({'caption_id': label}), add_special_tokens=False)
                 for label in ('A', 'B', 'C', 'UNKNOWN')]
    prefix = [99, 98]
    assert callback(0, prefix) == [sequences[0][0]]
    for sequence in sequences:
        for index, token in enumerate(sequence):
            assert token in callback(0, prefix + sequence[:index])
            assert tokenizer.eos_token_id not in callback(0, prefix + sequence[:index])
        assert callback(0, prefix + sequence) == [tokenizer.eos_token_id]
    with pytest.raises(probe.ProbeError, match='invalid_prefix'):
        callback(0, prefix + [0])
    with pytest.raises(probe.ProbeError, match='short_input'):
        callback(0, [99])


def test_caption_constraint_requires_manifest():
    with pytest.raises(probe.ProbeError, match='requires_natural_audio_manifest'):
        probe.run_gpu(Path('.'), Path('.'), constrained_captions=True)
