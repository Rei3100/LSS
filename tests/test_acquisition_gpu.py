"""URLから取得した素材が、ローカル入力と同じキュー・実ワーカー・実GPUで伴奏になることを確認する。

LSS_RUN_ACQUISITION_GPU=1で明示実行。外部サービスへは接続せず、実曲の冒頭をYouTubeの最良音声と
同じWebM/Opusにしてローカルで配信し、実際のyt-dlpで取得する。音楽品質の評価ではない。
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import math
import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from functools import partial

from lss import application
from lss.acquisition import service, store
from lss.acquisition.ytdlp import YtDlpBackend
from lss.core import catalog, client, jobs, models_store, session
from lss.core.audio_input import prepare_input
from lss.core.jobs import SeparationQueue
from lss.harness import fetch, infer, tsv_io
from lss.paths import resource_root
from tests.test_acquisition import NO_WAIT, serve  # noqa: F401  fixture
from tests.test_backend_acceptance_gpu import _start_worker, _stop_worker

pytestmark = pytest.mark.skipif(
    os.environ.get("LSS_RUN_ACQUISITION_GPU") != "1",
    reason="取得物の実GPU分離は明示実行",
)


def test_acquired_webm_opus_becomes_a_verified_instrumental(
    tmp_path, monkeypatch, serve, record_testsuite_property,
):  # noqa: F811
    header, rows = tsv_io.load_tsv(models_store.MODELS_TSV)
    tsv_io.save_tsv(tmp_path / "models.tsv", header, rows)
    default = catalog.default_model()
    checkpoint, config = fetch.model_paths(default["model_id"], default["distributor_url"], default["config_url"])
    assert checkpoint.is_file() and config.is_file(), "取得済みの既定モデルが必要です"
    monkeypatch.setattr(models_store, "MODELS_TSV", tmp_path / "models.tsv")
    _, corpus = tsv_io.load_tsv(dev_evidence_root(client.REPO_ROOT) / "ledger" / "test_set.tsv")
    original = Path(os.environ.get("LSS_BACKEND_ACCEPTANCE_SOURCE") or corpus[0]["mix_path"])
    original_digest = session.input_sha256_of(original)

    webm = tmp_path / "served.webm"
    subprocess.run([str(resource_root() / "bin" / "ffmpeg.exe"), "-nostdin", "-v", "error", "-t", "30",
                    "-i", str(original), "-map", "0:a:0", "-ar", "48000", "-c:a", "libopus", "-b:a", "160k",
                    str(webm)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    url, handler = serve(webm.read_bytes(), "clip.webm", "audio/webm")

    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv(store.LIBRARY_ENV, str(tmp_path / "library"))
    address = fr"\\.\pipe\lss-acquisition-gpu-{uuid4().hex}"
    dispatch = client.dispatch_job
    monkeypatch.setattr(jobs, "dispatch_job", lambda job, **kw: dispatch(job, address=address, **kw))
    queue = SeparationQueue(state_path=tmp_path / "queue.json")

    request = application.ProcessingRequest.from_urls([url], application.UserIntent.EXTRACT_ACCOMPANIMENT)
    acquirer = partial(service.acquire, backend=YtDlpBackend(), pacing=NO_WAIT)
    intake = application.submit(request, queue, acquirer=acquirer)
    [outcome] = intake.acquisitions
    assert outcome.source_path is not None, outcome.message
    acquired = outcome.source_path
    assert acquired.read_bytes() == handler.payload and len(intake.queue_item_ids) == 1

    process = _start_worker(tmp_path, address)
    try:
        queue.run()
        [item] = queue.snapshot()
        published = application.outcome_from_item(item)
        assert published.disposition == application.ResultDisposition.PUBLISHED, item.result
        saved_before = published.outputs[0]
        digest_before = store.file_sha256(saved_before)
        mtime_before = saved_before.stat().st_mtime_ns
        # 同じ製品要求の再投入は、取得物と検証済み成果物を両方再利用する。
        again = application.submit(request, queue, acquirer=acquirer)
        assert again.acquisitions[0].reused and len(again.queue_item_ids) == 1
        queue.run()
        reused_item = next(entry for entry in queue.snapshot() if entry.id == again.queue_item_ids[0])
        reused = application.outcome_from_item(reused_item)
        assert reused.disposition == application.ResultDisposition.PUBLISHED and reused.reused
        assert reused.outputs == published.outputs
        assert store.file_sha256(saved_before) == digest_before and saved_before.stat().st_mtime_ns == mtime_before
    finally:
        _stop_worker(process, address)

    item = queue.snapshot()[0]
    assert item.status == "completed", item.result
    result = item.result
    assert result["runtime_revision"]["device"] == "cuda:0"
    assert len(result["outputs"]) == 1
    saved = Path(result["outputs"][0])
    assert saved.parent == acquired.parent / "LSS" and "Instrumental" in saved.stem

    with prepare_input(acquired, work_root=tmp_path / "decode") as prepared:
        decoded = sf.info(prepared)
    _, rate = infer.expected_output_layout(config, decoded.samplerate)
    output = sf.info(saved)
    assert output.channels == 2 and output.samplerate == rate
    assert output.frames == math.ceil(decoded.frames * rate / decoded.samplerate)
    for block in sf.blocks(saved, blocksize=65536, always_2d=True):
        assert np.isfinite(block).all()

    # 分離の完了記録と成果物のタグから、取得の来歴へ入力SHA-256でたどれる。
    assert FLAC(saved)["LSS_INPUT_SHA256"] == [outcome.source_sha256]
    assert result["input_sha256"] == outcome.source_sha256
    [record] = store.records_for_sha256(result["input_sha256"])
    assert record["location"]["requested_url"] == url
    assert store.file_sha256(acquired) == outcome.source_sha256
    assert session.input_sha256_of(original) == original_digest
    record_testsuite_property("url_product_acceptance", json.dumps({
        "intent": request.intent.value,
        "original_source_sha256": original_digest,
        "source_sha256": outcome.source_sha256,
        "source_unchanged": True,
        "acquisition_tool": record["tool"],
        "acquisition_auth": record["auth"],
        "acquisition_format": record["format"],
        "recipe_id": result["recipe_id"],
        "recipe_revision": result["recipe_revision"],
        "runtime_revision": result["runtime_revision"],
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_vram_mb": result["peak_vram_mb"],
        "output_sha256": digest_before,
        "output_frames": output.frames,
        "output_channels": output.channels,
        "output_sample_rate": output.samplerate,
        "all_output_samples_finite": True,
        "acquired_source_reused": True,
        "validated_output_reused": True,
        "validated_output_mtime_unchanged": True,
        "fallback_used": False,
        "external_service_workflow_verified": False,
        "music_quality_evaluated": False,
    }, ensure_ascii=False, allow_nan=False))
