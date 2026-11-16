"""キュー保存の容量不足・使用中エラーから、同じキューで安全に再開する。"""
import errno
import json
from pathlib import Path

import pytest

from lss.core import jobs
from lss.core.jobs import SeparationQueue


@pytest.mark.parametrize("operation", ["add", "retry", "start"])
@pytest.mark.parametrize("boundary", ["flush", "replace"])
def test_queue_mutation_is_retryable_after_storage_failure(tmp_path, monkeypatch, operation, boundary):
    calls = []
    state_path = tmp_path / "queue.json"
    queue = SeparationQueue(
        lambda job: calls.append(job) or {
            "status": "error", "error_code": "file_not_found", "message": "入力がありません",
        }, state_path=state_path,
    )
    ids = queue.add([tmp_path / "first.wav"], "model", "Instrumental")
    if operation == "retry":
        queue.run()
    before = queue.snapshot()
    durable_before = state_path.read_bytes()
    action = {
        "add": lambda: queue.add([tmp_path / "second.wav"], "model", "Instrumental"),
        "remove": lambda: queue.remove(ids[0]),
        "retry": lambda: queue.retry(ids[0]),
        "start": queue.run,
    }[operation]
    replace = Path.replace

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "保存先の空き容量が不足しています")

    with monkeypatch.context() as patch:
        if boundary == "flush":
            patch.setattr(jobs.os, "fsync", fail)
        else:
            def fail_queue_replace(path, target):
                if Path(target) == state_path:
                    raise PermissionError("キューの保存先が使用中です")
                return replace(path, target)
            patch.setattr(Path, "replace", fail_queue_replace)
        with pytest.raises(OSError):
            action()

    assert state_path.read_bytes() == durable_before
    assert queue.snapshot() == before
    assert not list(tmp_path.glob(".queue.json-*.tmp"))
    if operation == "start":
        assert calls == [], "開始状態を保存できるまで実処理へ進まない"
    action()
    assert json.loads(state_path.read_text(encoding="utf-8"))["items"] == [
        {**vars(item), "path": str(item.path)} for item in queue.snapshot()
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_progress_does_not_enter_durable_queue(tmp_path, monkeypatch, value):
    def dispatch(job, on_progress):
        on_progress({"phase": "separating", "fraction": value})
        return {"status": "ok", "outputs": ["result.flac"]}

    monkeypatch.setattr(jobs, "dispatch_job", dispatch)
    observed = []
    path = tmp_path / "queue.json"
    queue = SeparationQueue(state_path=path, on_progress=lambda _, event: observed.append(event))
    queue.add([tmp_path / "source.wav"], "model", "Instrumental")
    queue.run()
    assert queue.snapshot()[0].status == "completed"
    assert observed == []
    assert "NaN" not in path.read_text(encoding="utf-8")
    assert "Infinity" not in path.read_text(encoding="utf-8")
