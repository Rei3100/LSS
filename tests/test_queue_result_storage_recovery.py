"""処理後のキュー保存が失敗しても、同じキューで結果を回収できる。"""
import errno
from pathlib import Path

import pytest

from lss.core import jobs
from lss.core.jobs import SeparationQueue


@pytest.mark.parametrize("boundary", ["flush", "replace"])
@pytest.mark.parametrize("outcome", ["ok", "skipped", "error"])
def test_result_storage_failure_replays_before_next_input(tmp_path, monkeypatch, boundary, outcome):
    path = tmp_path / "queue.json"
    completed = tmp_path / "completed.flac"
    calls = []
    durable_before = None
    expensive_runs = {}
    failed_once = False
    fsync, replace = jobs.os.fsync, Path.replace

    def dispatch(job):
        nonlocal durable_before
        calls.append(Path(job["mix_path"]).name)
        if len(calls) == 1:
            # workerはすでに公開と完了記録を済ませたが、queueはrunningを保存している。
            durable_before = path.read_bytes()
        name = calls[-1]
        if name not in expensive_runs:
            completed.write_bytes(b"validated worker output")
            expensive_runs[name] = 1
        if outcome == "error":
            return {"status": "error", "error_code": "input_decode_failed", "message": "読込失敗"}
        return {"status": "ok" if len(calls) == 1 and outcome == "ok" else "skipped", "outputs": [str(completed)]}

    def fail_after_dispatch():
        nonlocal failed_once
        if calls and not failed_once:
            failed_once = True
            raise OSError(errno.ENOSPC, "処理完了直後にキュー保存が失敗")

    def flush(fd):
        fail_after_dispatch()
        return fsync(fd)

    def publish(temporary, target):
        if Path(target) == path:
            fail_after_dispatch()
        return replace(temporary, target)

    queue = SeparationQueue(dispatch, state_path=path)
    queue.add([tmp_path / "first.wav", tmp_path / "second.wav"], "model", "Instrumental")
    with monkeypatch.context() as patch:
        patch.setattr(jobs.os, "fsync", flush) if boundary == "flush" else patch.setattr(Path, "replace", publish)
        with pytest.raises(OSError, match="処理完了直後"):
            queue.run()
    assert calls == ["first.wav"], "結果を保存できるまで次の入力を開始しない"
    assert path.read_bytes() == durable_before
    assert [item.status for item in queue.snapshot()] == ["queued", "queued"]
    assert queue.snapshot()[0].result is None and queue.snapshot()[0].progress is None
    assert completed.read_bytes() == b"validated worker output"
    assert not list(tmp_path.glob(".queue.json-*.tmp"))

    queue.run()
    assert calls == ["first.wav", "first.wav", "second.wav"]
    assert expensive_runs == {"first.wav": 1, "second.wav": 1}, "完成済み成果の再回収で分離を繰り返さない"
    assert [item.status for item in SeparationQueue(dispatch, state_path=path).snapshot()] == (
        ["failed", "failed"] if outcome == "error" else ["skipped", "skipped"]
    )
