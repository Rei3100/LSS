"""最終ファイルが入るだけの空き容量では、内部分離を開始しない。"""
from types import SimpleNamespace

import pytest

from lss.core import models_store, session, storage, worker
from lss.harness import infer, vram
from tests.test_worker import _setup, _write_wav, _fake_msst_separate


def prepare(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **kw: (checkpoint, tmp_path / "config"))
    source = tmp_path / "source.wav"
    _write_wav(source)
    return source, {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}


def test_insufficient_work_and_output_capacity_fails_before_inference_and_can_retry(tmp_path, monkeypatch):
    source, job = prepare(tmp_path, monkeypatch)
    original = source.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=3_000_000))
        patch.setattr(vram, "reset_peak_vram", lambda: pytest.fail("作業容量の検証が先"))
        patch.setattr(infer, "msst_separate", lambda *a, **kw: pytest.fail("分離を開始しない"))
        failed = worker.handle_job(job)
    assert failed["status"] == "error" and failed["error_code"] == "insufficient_output_space", failed
    assert "分離・保存" in failed["message"]
    assert source.read_bytes() == original
    assert not list((tmp_path / "LSS").glob("*.flac"))
    assert not session.completion_ledger_path().exists()
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    assert worker.handle_job(job)["status"] == "ok"
    assert source.read_bytes() == original


def test_completed_outputs_are_reusable_when_no_new_files_fit(tmp_path, monkeypatch):
    source, job = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    first = worker.handle_job(job)
    assert first["status"] == "ok", first
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **kw: pytest.fail("完成済み成果を再利用"))
    reused = worker.handle_job(job)
    assert reused["status"] == "skipped" and reused["outputs"] == first["outputs"]
