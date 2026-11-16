import json

import numpy as np
import pytest
import soundfile as sf

from lss.harness.fetch import file_sha256
from lss.validation import facing_music
from lss.validation.bandit_inference_cache import InferenceCache, exclusive_output
from scripts import probe_bgm_facing_music as probe

PLAYBACK_METADATA = b"LIST" + bytes([12, 0, 0, 0]) + b"INFOIsftplay"
DAMAGED_TAIL = b"partial"


def build(tmp_path, complete=True):
    provider = tmp_path / "provider.py"
    provider.write_text("# 固定した実装\n", encoding="utf-8")
    cases = []
    for index, name in enumerate(("first", "second"), 1):
        path = tmp_path / f"{name}.wav"
        sf.write(path, np.arange(40).reshape(20, 2) / (100 * index), 44100, subtype="FLOAT")
        cases.append({"case_id": name, "stems": {},
                      "source": {"path": str(path), "sha256": file_sha256(path), "frames": 20}})
    plan = {"status": "FROZEN_BEFORE_RESULTS", "promotion_allowed": False, "holdout_consumed": False,
            "sample_rate": 44100, "output_roles": list(facing_music.ROLES),
            "provider_revision": "a" * 40,
            "checkpoint": {"path": str(provider), "sha256": file_sha256(provider)},
            "model_config": dict(facing_music.MODEL_CONFIG), "fader": dict(probe.FADER),
            "provider_and_inputs": {str(provider): file_sha256(provider)}, "cases": cases}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    identity = {"plan_sha256": file_sha256(plan_path), "execution_device": "cuda:0",
                "cpu_fallback": False, **{key: plan[key] for key in probe.IDENTITY_KEYS},
                "runtime": {"torch": "test-CUDA-only"}}
    output = tmp_path / "output"
    with exclusive_output(output, resume=False):
        cache = InferenceCache(output, plan, identity, resume=False)
        for case in cases:
            cache.save(case, {role: np.full((20, 2), number * 0.1) for number, role
                              in enumerate(facing_music.ROLES, 1)}, seconds=0.5, peak_cuda_bytes=99)
        if complete:
            cache.complete(verify_identity=lambda: None)
    return plan_path, output, plan, identity, provider


def test_recover_quarantines_only_the_invalid_case_and_leaves_reinference_to_resume(tmp_path):
    plan_path, output, plan, identity, _ = build(tmp_path)
    kept = output / "first" / "music_instrumental.wav"
    snapshot = kept.read_bytes(), kept.stat().st_mtime_ns
    # chunkとして解釈できないbytesが末尾に残り、metadataだけの変更と言えなくなる。
    with (output / "second" / "sfx.wav").open("ab") as stream:
        stream.write(DAMAGED_TAIL)
    quarantine = tmp_path / "quarantine-r1"
    result = json.loads(probe.recover(plan_path, output, quarantine, "容器末尾の不明bytes"))
    assert result["status"] == "QUARANTINED_WITHOUT_DELETION" and result["deleted"] == []
    assert result["quarantined_cases"] == ["second"]
    assert result["reinference_required_cases"] == ["second"]
    assert result["reusable_cases"] == ["first"]
    assert not (output / "second").exists() and not (output / "receipt.json").exists()
    assert (quarantine / "second" / "sfx.wav").read_bytes().endswith(DAMAGED_TAIL)
    assert (quarantine / "receipt.json").exists()
    assert snapshot == (kept.read_bytes(), kept.stat().st_mtime_ns)
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        assert [row["state"] for row in cache.inspect()] == ["REUSABLE", "MISSING"]


def test_recover_refuses_output_that_is_not_this_frozen_plan(tmp_path):
    plan_path, output, plan, _, _ = build(tmp_path)
    (output / "second" / "sfx.wav").unlink()
    plan["limits"] = ["再凍結でplan hashが変わる"]
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    quarantine = tmp_path / "quarantine-r1"
    with pytest.raises(ValueError, match="この固定planのCUDA推論出力ではありません"):
        probe.recover(plan_path, output, quarantine, "欠けたstem")
    assert not quarantine.exists() and (output / "receipt.json").exists()


def test_recover_refuses_after_the_frozen_implementation_changed(tmp_path):
    plan_path, output, _, _, provider = build(tmp_path)
    (output / "second" / "sfx.wav").unlink()
    provider.write_text("# 変更した実装\n", encoding="utf-8")
    quarantine = tmp_path / "quarantine-r1"
    with pytest.raises(ValueError, match="固定provider・実装・入力の変更"):
        probe.recover(plan_path, output, quarantine, "欠けたstem")
    assert not quarantine.exists()


def test_recover_does_not_touch_a_healthy_inference_output(tmp_path):
    plan_path, output, _, _, _ = build(tmp_path)
    before = {path.name: file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    quarantine = tmp_path / "quarantine-r1"
    with pytest.raises(ValueError, match="隔離する不正な推論出力がありません"):
        probe.recover(plan_path, output, quarantine, "念のため")
    assert not quarantine.exists()
    assert before == {path.name: file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}


def test_resume_stops_with_the_recovery_entry_instead_of_reusing_invalid_output(tmp_path):
    plan_path, output, plan, identity, _ = build(tmp_path, complete=False)
    with (output / "second" / "speech.wav").open("ab") as stream:
        stream.write(DAMAGED_TAIL)
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        with pytest.raises(ValueError, match="recoverで隔離してから再開"):
            probe.pending_cases(cache, plan)
    quarantine = tmp_path / "quarantine-r1"
    assert json.loads(probe.recover(plan_path, output, quarantine,
                                    "容器末尾の不明bytes"))["quarantined_cases"] == ["second"]
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        assert [case["case_id"] for case in probe.pending_cases(cache, plan)] == ["second"]


def test_a_played_stem_needs_no_quarantine_and_no_reinference(tmp_path):
    plan_path, output, plan, identity, _ = build(tmp_path)
    played = output / "second" / "sfx.wav"
    before = played.read_bytes()
    with played.open("ab") as stream:
        stream.write(PLAYBACK_METADATA)
    assert played.read_bytes() != before
    with exclusive_output(output, resume=True):
        cache = InferenceCache(output, plan, identity, resume=True)
        assert [row["state"] for row in cache.inspect()] == ["REUSABLE", "REUSABLE"]
        assert probe.pending_cases(cache, plan) == []
    with pytest.raises(ValueError, match="隔離する不正な推論出力がありません"):
        probe.recover(plan_path, output, tmp_path / "quarantine-r1", "念のため")
    assert played.read_bytes().endswith(PLAYBACK_METADATA)
