"""PyInstaller onedirの実ワーカー確認。LSS_RUN_PACKAGED_TEST=1で明示実行する。"""
from __future__ import annotations

import os
import subprocess
import time
from multiprocessing import connection
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.core.protocol import AUTHKEY
from lss.harness import tsv_io


@pytest.mark.skipif(
    os.environ.get("LSS_RUN_PACKAGED_TEST") != "1",
    reason="配布ビルドと実GPUモデルが必要なため明示実行",
)
def test_packaged_worker_separates_with_bundled_resources(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    executable = repository / "dist" / "LSS" / "LSS.exe"
    model_dir = repository / ".models"
    assert executable.is_file(), "先にbuild.ps1を実行してください"
    assert model_dir.is_dir(), "取得済みモデルが必要です"
    bundled = executable.parent / "_internal"
    for unused in ("onnx", "onnxruntime", "sklearn", "torchvision"):
        assert not (bundled / unused).exists(), f"未使用依存を配布しない: {unused}"

    sample_rate = 44100
    seconds = 3
    timeline = np.arange(sample_rate * seconds) / sample_rate
    waveform = 0.1 * np.sin(2 * np.pi * 220 * timeline)
    source = tmp_path / "packaged.flac"
    sf.write(source, np.column_stack([waveform, waveform]), sample_rate, subtype="PCM_16")
    audio = FLAC(source)
    audio["TITLE"] = "配布版の分離確認"
    audio.save()

    address = fr"\\.\pipe\lss-packaged-test-{uuid4().hex}"
    environment = os.environ.copy()
    environment.update({
        "LSS_MODEL_DIR": str(model_dir),
        "LSS_STATE_DIR": str(tmp_path / "state"),
        "LSS_MIN_FREE_BYTES": "0",
    })
    process = subprocess.Popen(
        [str(executable), "--worker", "--address", address],
        cwd=executable.parent,
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    client = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"配布版ワーカーが終了しました: {process.returncode}")
            try:
                client = connection.Client(address, authkey=AUTHKEY)
                break
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.2)
        assert client is not None, "配布版ワーカーへ接続できません"
        client.send({
            "mix_path": str(source),
            "model_id": "becruily_deux_inst",
            "stem_target": "Instrumental",
        })
        assert client.poll(120), "配布版ワーカーの分離が120秒以内に完了しません"
        result = client.recv()
    finally:
        if client is not None:
            client.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

    assert result["status"] == "ok", result
    outputs = [Path(item) for item in result["outputs"]]
    assert len(outputs) == 2 and all(item.is_file() for item in outputs)
    for output in outputs:
        tags = FLAC(output)
        assert tags["TITLE"] == ["配布版の分離確認"]
        assert tags["LSS_MODEL_SHA256"] == [
            "10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa"
        ]
        assert len(tags["LSS_CONFIG_SHA256"][0]) == 64
        assert tags["LSS_RUNTIME_ID"] == [result["runtime_id"]]
        assert '"provider":"msst"' in tags["LSS_RUNTIME_REVISION"][0]

    runtime_ledger = executable.parent / "ledger" / "models.tsv"
    _, rows = tsv_io.load_tsv(runtime_ledger)
    default = next(row for row in rows if row["model_id"] == "becruily_deux_inst")
    assert default["sha256"] == "10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa"
    work_root = tmp_path / "state" / "work"
    assert not work_root.exists() or list(work_root.iterdir()) == []
