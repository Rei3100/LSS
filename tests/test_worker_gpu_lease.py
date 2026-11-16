"""完成済み成果はGPU待ちせず、未処理入力は所有権の取得後だけ分離する。"""
import pytest

from lss.core import gpu_lease, worker
from lss.harness import infer, vram
from tests.test_worker_processing_capacity import prepare
from tests.test_worker import _fake_msst_separate


def test_ownership_failure_stops_before_inference_without_changing_source(tmp_path, monkeypatch):
    source, job = prepare(tmp_path, monkeypatch)
    before = source.read_bytes()

    def fail(**kwargs):
        raise gpu_lease.GpuLeaseUnavailableError("排他所有を確認できない")

    monkeypatch.setattr(gpu_lease, "acquire", fail)
    monkeypatch.setattr(vram, "reset_peak_vram", lambda: pytest.fail("所有権取得が先"))
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("分離を開始しない"))
    failed = worker.handle_job(job)
    assert failed["status"] == "error" and failed["error_code"] == "gpu_unavailable", failed
    assert source.read_bytes() == before


def test_gpu_cleanup_finishes_before_lease_release_and_reuse_needs_no_lease(tmp_path, monkeypatch):
    source, job = prepare(tmp_path, monkeypatch)
    events = []

    class Lease:
        def close(self):
            events.append("released")

    def acquire(**kwargs):
        events.append("acquired")
        return Lease()

    monkeypatch.setattr(gpu_lease, "acquire", acquire)
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    monkeypatch.setattr(vram, "release_unused_vram", lambda: events.append("gpu_cleanup"))
    assert worker.handle_job(job)["status"] == "ok"
    assert events == ["acquired", "gpu_cleanup", "released"]
    monkeypatch.setattr(gpu_lease, "acquire", lambda **kw: pytest.fail("再利用はGPUの順番待ち不要"))
    assert worker.handle_job(job)["status"] == "skipped"


def test_result_completed_while_waiting_is_reused_without_second_inference(tmp_path, monkeypatch):
    source, job = prepare(tmp_path, monkeypatch)
    inferences = []

    class Lease:
        def close(self):
            pass

    def infer_once(*args, **kwargs):
        inferences.append(1)
        return _fake_msst_separate(*args, **kwargs)

    def acquire_after_another_worker_finishes(**kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(gpu_lease, "acquire", lambda **kw: Lease())
            assert worker.handle_job(job)["status"] == "ok"
        return Lease()

    monkeypatch.setattr(infer, "msst_separate", infer_once)
    monkeypatch.setattr(gpu_lease, "acquire", acquire_after_another_worker_finishes)
    result = worker.handle_job(job)
    assert result["status"] == "skipped", result
    assert len(inferences) == 1


def test_source_changed_during_gpu_wait_fails_before_inference(tmp_path, monkeypatch):
    source, job = prepare(tmp_path, monkeypatch)
    released = []

    class Lease:
        def close(self):
            released.append(True)

    def acquire(**kwargs):
        source.write_bytes(source.read_bytes() + b"changed while waiting")
        return Lease()

    monkeypatch.setattr(gpu_lease, "acquire", acquire)
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **kw: pytest.fail("変更された原本は分離しない"))
    result = worker.handle_job(job)
    assert result["status"] == "error" and result["error_code"] == "input_changed", result
    assert released == [True]
