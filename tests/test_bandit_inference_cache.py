import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.harness.fetch import file_sha256
from lss.validation import bandit_inference_cache
from lss.validation.bandit_inference_cache import InferenceCache, exclusive_output, validate_plan


@pytest.fixture
def inputs(tmp_path):
    cases = []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.wav"
        sf.write(path, np.arange(40).reshape(20, 2) / 100, 44100, subtype="FLOAT")
        cases.append({"case_id": name, "stems": {},
                      "source": {"path": str(path), "sha256": file_sha256(path), "frames": 20}})
    plan = {"status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False, "holdout_consumed": False,
            "sample_rate": 44100, "cases": cases}
    identity = {"plan_sha256": "a" * 64, "execution_device": "cuda:0", "cpu_fallback": False,
                "runtime": {"torch": "test-CUDA-only"}, "adapter_sha256": "b" * 64}
    return tmp_path / "output", plan, identity


def values(case):
    return {role: np.full((case["source"]["frames"], 2), index * 0.1)
            for index, role in enumerate(("music", "speech", "effects"), 1)}


def save(cache, case, **kwargs):
    return cache.save(case, values(case), seconds=0.25, peak_cuda_bytes=1234, **kwargs)


FOUR_ROLES = ["music_instrumental", "music_vocals", "speech", "sfx"]


def test_four_stems_resume_requires_singing_and_preserves_completed_audio(inputs):
    output, plan, identity = inputs
    plan["output_roles"] = FOUR_ROLES.copy()
    identity["output_roles"] = FOUR_ROLES.copy()
    case = plan["cases"][0]
    samples = {role: np.full((case["source"]["frames"], 2), index * 0.1)
               for index, role in enumerate(FOUR_ROLES, 1)}
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        first = cache.save(case, samples, seconds=0.25, peak_cuda_bytes=1234)
    singing = output / "first" / "music_vocals.wav"
    snapshot = singing.read_bytes(), singing.stat().st_mtime_ns
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        assert cache.load(case) == first
        assert set(first["stems"]) == set(FOUR_ROLES)
        assert snapshot == (singing.read_bytes(), singing.stat().st_mtime_ns)
        singing.unlink()
        with pytest.raises(FileNotFoundError):
            cache.load(case)
        with pytest.raises(FileNotFoundError):
            cache.complete(verify_identity=lambda: None)
        assert not (output / "receipt.json").exists()


@pytest.mark.parametrize("roles", [[], "music", ["../music"], ["music", "music"], [True], ["m"] * 33])
def test_unsafe_or_ambiguous_output_roles_cannot_form_a_frozen_plan(inputs, roles):
    _, plan, _ = inputs
    plan["output_roles"] = roles
    with pytest.raises(ValueError, match="output_roles"):
        validate_plan(plan)


def test_four_stem_identity_cannot_be_reused_as_three_stem_or_reordered_plan(inputs):
    output, plan, identity = inputs
    plan["output_roles"] = FOUR_ROLES.copy()
    with exclusive_output(output, resume=False):
        with pytest.raises(ValueError, match="output_roles"):
            InferenceCache(output, plan, identity, resume=False)
        identity["output_roles"] = FOUR_ROLES.copy()
        InferenceCache(output, plan, identity, resume=False)
    changed = copy.deepcopy(plan)
    changed["output_roles"].reverse()
    with exclusive_output(output, resume=True):
        with pytest.raises(ValueError, match="output_roles"):
            InferenceCache(output, changed, identity, resume=True)
        changed_identity = {**identity, "output_roles": changed["output_roles"]}
        with pytest.raises(ValueError, match="再開する推論"):
            InferenceCache(output, changed, changed_identity, resume=True)


def test_incomplete_four_stem_model_output_does_not_publish_a_case(inputs):
    output, plan, identity = inputs
    plan["output_roles"] = FOUR_ROLES.copy()
    identity["output_roles"] = FOUR_ROLES.copy()
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        with pytest.raises(ValueError, match="固定stem"):
            save(cache, plan["cases"][0])
        assert not (output / "first").exists()
        assert not list(output.glob(".partial-*"))


def test_variant_definition_and_identity_survive_case_completion_and_resume(inputs):
    from lss.bgm.inference_variants import InferenceVariantRecipe

    output, plan, identity = inputs
    case = plan["cases"][0]
    case["inference_variant"] = InferenceVariantRecipe(-6.0, 11025).to_dict()
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        record = save(cache, case)
        assert record["inference_variant"] == case["inference_variant"]
        assert record["inference_variant_id"] == InferenceVariantRecipe(-6.0, 11025).recipe_id
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        assert cache.load(case) == record
        metadata = output / "first" / "case.json"
        corrupt = json.loads(metadata.read_text())
        corrupt["inference_variant"] = InferenceVariantRecipe(6.0, 11025).to_dict()
        metadata.write_text(json.dumps(corrupt))
        with pytest.raises(ValueError, match="variant"):
            cache.load(case)


@pytest.mark.parametrize("variant", [None, [], {"input_gain_db": float("nan")},
                                      {"leading_silence_frames": True}, {"unknown": 3}])
def test_invalid_optional_variant_cannot_become_a_recovery_plan(inputs, variant):
    _, plan, _ = inputs
    plan["cases"][0]["inference_variant"] = variant
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_partial_completion_resumes_without_rewriting_verified_case(inputs):
    output, plan, identity = inputs
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        first = save(cache, plan["cases"][0])
        first_file = output / "first" / "music.wav"
        snapshot = (first_file.read_bytes(), first_file.stat().st_mtime_ns)
        with pytest.raises(ValueError, match="全固定ケース"):
            cache.complete(verify_identity=lambda: None)
        assert not (output / "receipt.json").exists()
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        assert cache.load(plan["cases"][0]) == first
        assert cache.load(plan["cases"][1]) is None
        save(cache, plan["cases"][1])
        receipt = cache.complete(verify_identity=lambda: None)
        stamp = receipt.stat().st_mtime_ns
        assert len(json.loads(receipt.read_text())["cases"]) == 2
        assert cache.complete(verify_identity=lambda: None).stat().st_mtime_ns == stamp
    assert snapshot == (first_file.read_bytes(), first_file.stat().st_mtime_ns)


def test_interruption_before_publish_is_not_a_completed_case(inputs):
    output, plan, identity = inputs
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        def interrupt():
            raise InterruptedError("power loss proxy")
        with pytest.raises(InterruptedError):
            save(cache, plan["cases"][0], before_publish=interrupt)
        assert cache.load(plan["cases"][0]) is None
        assert not list(output.glob(".partial-*"))
        # OS終了で残った未完成stagingは、完了ケースとして読み込まない。
        (output / ".partial-second-crash").mkdir()
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        save(cache, plan["cases"][0])
        assert cache.load(plan["cases"][0]) is not None


@pytest.mark.parametrize("mutation", ["content", "timeline", "nan", "missing", "role"])
def test_completed_cache_cannot_hide_bad_stems(inputs, mutation):
    output, plan, identity = inputs
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        save(cache, plan["cases"][0])
        path = output / "first" / "music.wav"
        metadata = output / "first" / "case.json"
        record = json.loads(metadata.read_text())
        if mutation == "content":
            sf.write(path, np.zeros((20, 2)), 44100, subtype="FLOAT")
        elif mutation == "timeline":
            sf.write(path, np.zeros((19, 2)), 44100, subtype="FLOAT")
            record["stems"]["music"]["sha256"] = file_sha256(path)
        elif mutation == "nan":
            sf.write(path, np.full((20, 2), np.nan), 44100, subtype="FLOAT")
            record["stems"]["music"]["sha256"] = file_sha256(path)
        elif mutation == "missing":
            path.unlink()
        else:
            del record["stems"]["effects"]
        metadata.write_text(json.dumps(record))
        with pytest.raises((ValueError, FileNotFoundError)):
            cache.load(plan["cases"][0])
        assert not (output / "receipt.json").exists()


@pytest.mark.parametrize("key", ["runtime", "adapter_sha256", "plan_sha256"])
def test_changed_runtime_adapter_or_plan_rejects_resume(inputs, key):
    output, plan, identity = inputs
    with exclusive_output(output, resume=False):
        InferenceCache(output, plan, identity, resume=False)
    changed = {**identity, key: "changed"}
    with exclusive_output(output, resume=True):
        with pytest.raises(ValueError, match="再開する推論"):
            InferenceCache(output, plan, changed, resume=True)


def test_source_changes_during_serialization_prevent_publish(inputs):
    output, plan, identity = inputs
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        def change():
            sf.write(plan["cases"][0]["source"]["path"], np.zeros((20, 2)), 44100, subtype="FLOAT")
        with pytest.raises(ValueError, match="sourceの内容"):
            save(cache, plan["cases"][0], before_publish=change)
        assert not (output / "first").exists()


def test_code_change_prevents_final_receipt(inputs):
    output, plan, identity = inputs
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        for case in plan["cases"]:
            save(cache, case)
        def changed():
            raise ValueError("adapter changed")
        with pytest.raises(ValueError, match="adapter changed"):
            cache.complete(verify_identity=changed)
        assert not (output / "receipt.json").exists()


def test_competing_owner_is_rejected_and_released_lock_can_be_reused(inputs):
    output, _, _ = inputs
    with exclusive_output(output, resume=False):
        with pytest.raises(RuntimeError, match="別プロセス"):
            with exclusive_output(output, resume=True):
                pytest.fail("must not acquire")
    with exclusive_output(output, resume=True):
        pass


@pytest.mark.parametrize("id", ["../escape", "absolute/path", "", "C:bad", ".", "first/../../outside"])
def test_case_id_cannot_escape_output(inputs, id):
    _, plan, _ = inputs
    changed = copy.deepcopy(plan)
    changed["cases"][0]["case_id"] = id
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_no_duplicate_cases_or_preexisting_stem_relabeling(inputs):
    _, plan, _ = inputs
    changed = copy.deepcopy(plan)
    changed["cases"].append(changed["cases"][0])
    with pytest.raises(ValueError, match="一意"):
        validate_plan(changed)
    changed = copy.deepcopy(plan)
    changed["cases"][0]["stems"] = {"music": "old-unverified"}
    with pytest.raises(ValueError):
        validate_plan(changed)


def test_identity_cannot_be_attached_to_legacy_unrecorded_output(inputs):
    output, plan, identity = inputs
    output.mkdir()
    (output / "first").mkdir()
    with exclusive_output(output, resume=True):
        with pytest.raises(ValueError, match="記録がない"):
            InferenceCache(output, plan, identity, resume=True)


@pytest.mark.parametrize("device,fallback", [("cpu", False), ("cuda:0", True), ("cuda:1", False)])
def test_cpu_fallback_is_not_a_valid_cache_identity(inputs, device, fallback):
    output, plan, identity = inputs
    output.mkdir()
    with pytest.raises(ValueError, match="CUDA専用"):
        InferenceCache(output, plan, {**identity, "execution_device": device, "cpu_fallback": fallback}, resume=False)


def test_provider_transitive_code_changes_or_additions_invalidate_frozen_run(inputs, tmp_path):
    from scripts.infer_bgm_bandit_resumable import freeze, verify_fixed
    _, parent, _ = inputs
    provider = tmp_path / "provider"
    (provider / "core").mkdir(parents=True)
    checkpoint = provider / "model.ckpt"
    checkpoint.write_bytes(b"observed-fixed-checkpoint")
    code = provider / "core" / "model.py"
    code.write_text("fixed provider code")
    parent["provider"] = {"root": str(provider), "artifacts": {
        "checkpoint": {"path": "model.ckpt", "sha256": file_sha256(checkpoint)}}}
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(parent))
    frozen_path = freeze(path, tmp_path / "frozen", case_ids=["first"])
    frozen = json.loads(frozen_path.read_text())
    assert len(frozen["cases"]) == 1
    assert frozen["provider_files"]["core/model.py"] == file_sha256(code)
    verify_fixed(frozen_path, frozen, file_sha256(frozen_path))
    code.write_text("changed provider code")
    with pytest.raises(ValueError, match="provider"):
        verify_fixed(frozen_path, frozen, file_sha256(frozen_path))
    code.write_text("fixed provider code")
    (provider / "core" / "unexpected.py").write_text("new import")
    with pytest.raises(ValueError, match="provider"):
        verify_fixed(frozen_path, frozen, file_sha256(frozen_path))


PLAYBACK_METADATA = b"LIST\x1a\x00\x00\x00INFOIsft\x0e\x00\x00\x00play_count=2\x00\x00"


def extra_case(tmp_path, name, divisor=200):
    path = tmp_path / f"{name}.wav"
    sf.write(path, np.arange(40).reshape(20, 2) / divisor, 44100, subtype="FLOAT")
    return {"case_id": name, "stems": {},
            "source": {"path": str(path), "sha256": file_sha256(path), "frames": 20}}


def append_external_playback_history(path):
    """容器の後ろへ正しいchunkとして再生履歴が書き戻される。音は変わらない。"""
    with path.open("ab") as stream:
        stream.write(PLAYBACK_METADATA)


def damage_container(path):
    """chunkとして解釈できないbytesが末尾に残る。metadataだけの変更と言えない。"""
    with path.open("ab") as stream:
        stream.write(b"partial")


def test_container_damaged_case_is_quarantined_and_reinferred_alone(inputs, tmp_path):
    output, plan, identity = inputs
    plan["cases"].append(extra_case(tmp_path, "third"))
    quarantine = tmp_path / "quarantine-external-metadata-r1"
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        for case in plan["cases"]:
            save(cache, case)
        receipt_path = cache.complete(verify_identity=lambda: None)
        old_receipt = receipt_path.read_bytes()
        # 外部の再生履歴が完了済みWAVへ追記され、厳密なhash検査が正しく止まる。
        damage_container(output / "second" / "music.wav")
        kept = output / "first" / "music.wav"
        snapshot = kept.read_bytes(), kept.stat().st_mtime_ns
        with pytest.raises(ValueError, match="推論stemの所在・内容"):
            cache.load(plan["cases"][1])
        record = cache.quarantine_invalid(quarantine, reason="容器末尾の不明bytes")
        assert record["status"] == "QUARANTINED_WITHOUT_DELETION" and record["deletes_nothing"] is True
        assert [row["case_id"] for row in record["cases"]] == ["second"]
        assert record["reusable_cases"] == ["first", "third"]
        assert record["reinference_required_cases"] == ["second"]
        assert not (output / "second").exists() and not receipt_path.exists()
        # 削除ではなく移動。隔離先の内容hashは記録と一致する。
        moved = quarantine / "second"
        assert record["cases"][0]["files"] == {
            name: file_sha256(moved / name) for name in ("case.json", "music.wav", "speech.wav", "effects.wav")}
        assert (quarantine / "receipt.json").read_bytes() == old_receipt
        assert cache.load(plan["cases"][1]) is None
        assert cache.load(plan["cases"][0]) is not None
        save(cache, plan["cases"][1])
        again = cache.complete(verify_identity=lambda: None)
        assert [row["case_id"] for row in json.loads(again.read_text())["cases"]] == [
            "first", "second", "third"]
    assert snapshot == (kept.read_bytes(), kept.stat().st_mtime_ns)
    assert (moved / "music.wav").read_bytes().endswith(b"partial")


def test_quarantine_keeps_uninferred_cases_and_leaves_valid_results_in_place(inputs, tmp_path):
    output, plan, identity = inputs
    plan["cases"].append(extra_case(tmp_path, "third"))
    quarantine = tmp_path / "quarantine-r1"
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        save(cache, plan["cases"][0])
        save(cache, plan["cases"][1])
        (output / "second" / "effects.wav").unlink()
        record = cache.quarantine_invalid(quarantine, reason="停電で欠けたstem")
        assert record["receipt"] is None and not (quarantine / "receipt.json").exists()
        assert record["missing_cases"] == ["third"]
        assert record["reinference_required_cases"] == ["second", "third"]
        assert sorted(path.name for path in quarantine.iterdir()) == ["quarantine.json", "second"]
        assert [row["state"] for row in cache.inspect()] == ["REUSABLE", "MISSING", "MISSING"]


def test_changed_fixed_source_is_not_recoverable_by_quarantine(inputs, tmp_path):
    output, plan, identity = inputs
    quarantine = tmp_path / "quarantine-r1"
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        save(cache, plan["cases"][0])
        append_external_playback_history(Path(plan["cases"][0]["source"]["path"]))
        assert [row["state"] for row in cache.inspect()] == ["INVALID_SOURCE", "MISSING"]
        with pytest.raises(ValueError, match="固定sourceが変わった"):
            cache.quarantine_invalid(quarantine, reason="固定sourceの変更")
        assert not quarantine.exists() and (output / "first" / "music.wav").exists()


@pytest.mark.parametrize("destination,reason,message", [
    ("outside", "", "隔離の理由"),
    ("inside", "正当な理由", "隔離先は推論出力の外"),
    ("outside", "隔離する対象がない", "隔離する不正な推論出力がありません")])
def test_quarantine_refuses_unsafe_destination_reason_or_healthy_output(
        inputs, tmp_path, destination, reason, message):
    output, plan, identity = inputs
    target = output / "aside" if destination == "inside" else tmp_path / "quarantine-r1"
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        save(cache, plan["cases"][0])
        with pytest.raises(ValueError, match=message):
            cache.quarantine_invalid(target, reason=reason)
        assert not target.exists()
        assert cache.load(plan["cases"][0]) is not None


def test_interrupted_quarantine_resumes_and_rejects_a_different_condition(inputs, tmp_path, monkeypatch):
    output, plan, identity = inputs
    quarantine = tmp_path / "quarantine-r1"
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        for case in plan["cases"]:
            save(cache, case)
        for name in ("first", "second"):
            damage_container(output / name / "music.wav")
        moves, rename = [], os.rename
        def interrupt(source, target):
            if moves:
                raise InterruptedError("power loss proxy")
            moves.append(rename(source, target))
        monkeypatch.setattr(bandit_inference_cache.os, "rename", interrupt)
        with pytest.raises(InterruptedError):
            cache.quarantine_invalid(quarantine, reason="容器末尾の不明bytes")
        frozen = json.loads((quarantine / "quarantine.json").read_text(encoding="utf-8"))
        assert frozen["status"] == "FROZEN_BEFORE_MOVE"
        assert (quarantine / "first").exists() and (output / "second").exists()
        monkeypatch.undo()
        with pytest.raises(ValueError, match="中断した同じ隔離"):
            cache.quarantine_invalid(quarantine, reason="別の理由")
        record = cache.quarantine_invalid(quarantine, reason="容器末尾の不明bytes")
        assert record["status"] == "QUARANTINED_WITHOUT_DELETION"
        assert record["reinference_required_cases"] == ["first", "second"]
        assert cache.quarantine_invalid(quarantine, reason="容器末尾の不明bytes") == record
