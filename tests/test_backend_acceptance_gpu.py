"""実曲・別プロセス・実GPUで、保存と再起動後の再利用を確認する。

LSS_RUN_BACKEND_ACCEPTANCE=1で実行する。音楽品質の採用評価ではなく、
現行の既定経路の処理・通知・保存・回復を検証する受入試験。
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.core import catalog, client, jobs, models_store, recipe, session
from lss.core.jobs import SeparationQueue
from lss.harness import fetch, infer, tsv_io


pytestmark = pytest.mark.skipif(
    os.environ.get("LSS_RUN_BACKEND_ACCEPTANCE") != "1",
    reason="実曲のバックエンド受入は明示実行",
)


def _start_worker(root: Path, address: str) -> subprocess.Popen:
    # 実モデルを使い、測定の書き込み先だけを隔離する。ユーザーの台帳は変更しない。
    code = (
        "import sys; from pathlib import Path; "
        "from lss.core import models_store, worker; "
        "models_store.MODELS_TSV = Path(sys.argv[1]); "
        "worker.main(['--address', sys.argv[2]])"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(root / "models.tsv"), address],
        cwd=client.REPO_ROOT,
        env={**os.environ, "LSS_STATE_DIR": str(root / "state")},
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if client.is_worker_running(address):
                return process
            if process.poll() is not None:
                raise AssertionError(f"ワーカー起動失敗: {process.returncode}")
            time.sleep(0.1)
        raise AssertionError("ワーカー起動が時間内に完了しません")
    except BaseException:
        process.terminate()
        process.wait(timeout=10)
        raise


def _stop_worker(process: subprocess.Popen, address: str) -> None:
    try:
        response = client.request_worker_shutdown(address)
        assert response is not None and response["status"] == "ok"
        process.wait(timeout=10)
        assert process.returncode == 0
        assert not client.is_worker_running(address)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def _verify_result(result: dict, source: Path, config: Path) -> Path:
    assert result["status"] in {"ok", "skipped"}, result
    assert len(result["outputs"]) == 1
    saved = Path(result["outputs"][0])
    assert saved.parent == source.parent / "LSS"
    assert "Instrumental" in saved.stem
    input_info = sf.info(source)
    output_info = sf.info(saved)
    _, rate = infer.expected_output_layout(config, input_info.samplerate)
    assert output_info.frames == math.ceil(input_info.frames * rate / input_info.samplerate)
    assert output_info.samplerate == rate
    assert output_info.channels == 2
    for block in sf.blocks(saved, blocksize=65536, always_2d=True):
        assert np.isfinite(block).all()
    tags = FLAC(saved)
    assert tags["LSS_INPUT_SHA256"] == [session.input_sha256_of(source)]
    assert tags["LSS_RUNTIME_ID"] == [result["runtime_id"]]
    assert tags["LSS_RECIPE_ID"] == [result["recipe_id"]]
    revision = recipe.validate_recipe_revision(result["recipe_revision"])
    assert revision["publication"]["stems"] == ["Instrumental"]
    assert session.is_completed(
        result["input_sha256"], result["recipe_id"], source,
        state_root=source.parent.parent / "state",
    )
    return saved


def test_real_song_queue_progress_pause_restart_reuse_and_corruption(tmp_path, monkeypatch):
    header, rows = tsv_io.load_tsv(models_store.MODELS_TSV)
    tsv_io.save_tsv(tmp_path / "models.tsv", header, rows)
    default = catalog.default_model()
    cache = fetch.model_cache_dir() / default["model_id"]
    config = cache / default["config_url"].rsplit("/", 1)[-1]
    checkpoint = cache / default["distributor_url"].rsplit("/", 1)[-1]
    assert config.is_file() and checkpoint.is_file(), "取得済み既定モデルが必要です"
    configured_source = os.environ.get("LSS_BACKEND_ACCEPTANCE_SOURCE")
    if configured_source:
        original = Path(configured_source)
    else:
        _, corpus = tsv_io.load_tsv(dev_evidence_root(client.REPO_ROOT) / "ledger" / "test_set.tsv")
        original = Path(corpus[0]["mix_path"])
    assert sf.info(original).channels == 2, "受入対象は完成済みステレオ音源です"
    original_digest = session.input_sha256_of(original)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "実曲.flac"
    # FLAC以外の任意入力でも拡張子を偽装しない。
    source = source.with_suffix(original.suffix)
    shutil.copyfile(original, source)
    missing = inputs / "再試行.wav"
    state_path = tmp_path / "queue.json"
    address = fr"\\.\pipe\lss-backend-acceptance-{uuid4().hex}"
    dispatch = client.dispatch_job
    monkeypatch.setattr(jobs, "dispatch_job", lambda job, **kw: dispatch(job, address=address, **kw))
    events = []
    results = []

    def on_result(item_id, result):
        results.append((item_id, result))
        queue.pause()

    queue = SeparationQueue(
        state_path=state_path, on_progress=lambda item_id, event: events.append((item_id, event)),
        on_result=on_result,
    )
    ids = queue.add([source, missing])
    process = _start_worker(tmp_path, address)
    try:
        queue.run()
        assert [item.status for item in queue.snapshot()] == ["completed", "queued"]
        saved = _verify_result(queue.snapshot()[0].result, source, config)
        assert {event["phase"] for _, event in events} >= {
            "preparing_input", "checking_model", "separating", "completed",
        }
        assert results[0][0] == ids[0]
        before = session.input_sha256_of(saved)
    finally:
        _stop_worker(process, address)

    # 保存済みキューと新しい実ワーカーを使う。失敗後も結果を再利用できる。
    queue = SeparationQueue(state_path=state_path)
    process = _start_worker(tmp_path, address)
    try:
        queue.run()
        assert [item.status for item in queue.snapshot()] == ["completed", "failed"]
        assert queue.snapshot()[1].result["error_code"] == "file_not_found"
        queue.add([source])
        queue.run()
        assert queue.snapshot()[-1].status == "skipped"
        assert _verify_result(queue.snapshot()[-1].result, source, config) == saved
        assert session.input_sha256_of(saved) == before

        with sf.SoundFile(source) as audio:
            excerpt = audio.read(frames=audio.samplerate * 15, always_2d=True)
            sf.write(missing, excerpt, audio.samplerate, subtype="PCM_24")
        queue.retry(ids[1])
        queue.run()
        assert queue.snapshot()[1].status == "completed"
        _verify_result(queue.snapshot()[1].result, missing, config)

        saved.write_bytes(b"corrupt output")
        queue.add([source])
        queue.run()
        assert queue.snapshot()[-1].status == "completed"
        regenerated = _verify_result(queue.snapshot()[-1].result, source, config)
        assert regenerated.is_file()
        assert session.input_sha256_of(original) == original_digest
        assert session.input_sha256_of(source) == original_digest
        assert not list((tmp_path / "state" / "publications").glob("*.json"))
        assert not list((tmp_path / "state" / "work").iterdir())
        print(json.dumps({
            "source_sha256": original_digest,
            "source_seconds": sf.info(source).duration,
            "recipe_id": queue.snapshot()[-1].result["recipe_id"],
            "runtime_id": queue.snapshot()[-1].result["runtime_id"],
            "verified": ["real_gpu", "progress", "pause", "restart", "reuse", "retry", "corruption"],
        }, ensure_ascii=False))
    finally:
        _stop_worker(process, address)
