"""入力検証をworkerへつなぎ、保存と再利用まで確認する。"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.core import recipe_registry, session, models_store, worker
from lss.harness import infer
from tests.test_worker import _setup, _fake_msst_separate


def _worker_setup(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test-model")
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **kw: (checkpoint, tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    return {"mix_path": str(tmp_path / "source.wav"), "model_id": "model_x", "stem_target": "Instrumental"}


def test_worker_records_and_reuses_silence_without_placeholder_or_inference(tmp_path, monkeypatch):
    job = _worker_setup(tmp_path, monkeypatch)
    source = Path(job["mix_path"])
    sf.write(source, np.zeros((44100, 2)), 44100, subtype="FLOAT")
    before = source.read_bytes()
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("完全無音に分離を実行しない"))
    monkeypatch.setattr(models_store, "ensure_model_cached",
                        lambda *a, **k: pytest.fail("完全無音ではmodelを準備しない"))
    phases = []
    first = worker.handle_job(job, progress=phases.append)
    assert first["status"] == "ok", first
    assert first["outputs"] == []
    assert first["zero_result"]["reason"] == "no_wanted_target"
    assert phases[-1] == {"phase": "completed", "output_count": 0, "zero_reason": "no_wanted_target"}
    second = worker.handle_job(job)
    assert second["status"] == "skipped"
    assert second["zero_result"] == first["zero_result"]
    assert second["recipe_id"] == first["recipe_id"]
    assert recipe_registry.revision_for(first["recipe_id"]) == first["recipe_revision"]
    assert not (tmp_path / "LSS").exists()
    assert source.read_bytes() == before


def test_silence_zero_does_not_suppress_replacement_audio(tmp_path, monkeypatch):
    job = _worker_setup(tmp_path, monkeypatch)
    source = Path(job["mix_path"])
    sf.write(source, np.zeros((44100, 2)), 44100, subtype="FLOAT")
    assert worker.handle_job(job)["outputs"] == []
    sf.write(source, np.full((44100, 2), 0.1), 44100, subtype="FLOAT")
    result = worker.handle_job(job)
    assert result["status"] == "ok", result
    assert len(result["outputs"]) == 1
    assert "zero_result" not in result


def test_nonfinite_worker_input_fails_before_model_preparation(tmp_path, monkeypatch):
    job = _worker_setup(tmp_path, monkeypatch)
    samples = np.full((1000, 2), 0.1)
    samples[-1, 1] = float("nan")
    sf.write(job["mix_path"], samples, 44100, subtype="FLOAT")
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: pytest.fail("不正入力ではモデルを準備しない"))
    result = worker.handle_job(job)
    assert result["status"] == "error"
    assert result["error_code"] == "invalid_input_signal"
    assert not session.completion_ledger_path().exists()
