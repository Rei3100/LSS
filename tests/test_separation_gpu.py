"""取得済みの既定モデルによる実機確認。LSS_RUN_GPU_TEST=1で明示実行する。"""
import os
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mutagen.flac import FLAC

from lss.core import audio_input, models_store, session, worker
from lss.core.jobs import SeparationQueue
from lss.harness import fetch, infer, tsv_io
from tests.test_audio_input import transcode


@pytest.mark.skipif(os.environ.get("LSS_RUN_GPU_TEST") != "1", reason="実GPU分離は明示実行")
@pytest.mark.parametrize("input_format", ["flac", "m4a", "wma"])
@pytest.mark.parametrize("input_layout", [(44100, 2), (48000, 1)])
def test_cached_default_model_separates_one_input_and_writes_flac(tmp_path, monkeypatch, input_format, input_layout):
    import torch

    assert torch.cuda.is_available(), "CUDA GPUが必要です"
    header, rows = tsv_io.load_tsv(models_store.MODELS_TSV)
    row = next(r for r in rows if r["状態"] == "既定")
    model_dir = fetch.model_cache_dir() / row["model_id"]
    ckpt = model_dir / row["distributor_url"].rsplit("/", 1)[-1]
    config = model_dir / row["config_url"].rsplit("/", 1)[-1]
    assert ckpt.is_file() and config.is_file(), "取得済みモデルと設定が必要です"

    # 実際のモデル台帳・完了履歴・音声ライブラリは変更しない。
    ledger = tmp_path / "models.tsv"
    tsv_io.save_tsv(ledger, header, rows)
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (ckpt, config))
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    sr, channels = input_layout
    t = np.arange(sr * 3) / sr
    wave = (0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 660 * t))
    source = inputs / "a_対象.flac"
    sf.write(source, np.column_stack([wave, wave]) if channels == 2 else wave, sr, subtype="PCM_16")
    audio = FLAC(source)
    audio["TITLE"] = "分離確認用の音声"
    audio.save()
    if input_format in {"m4a", "wma"}:
        encoded = source.with_suffix(f".{input_format}")
        transcode(source, encoded, codec={"m4a": "aac", "wma": "wmav2"}[input_format])
        source = encoded
    # 旧実装では後からこの別ファイルを読み、対象の結果を上書きしていた。
    sf.write(inputs / "z_対象外.wav", np.zeros((sr, 2)), sr)
    with audio_input.prepare_input(source, work_root=tmp_path / "probe") as decoded:
        decoded_info = sf.info(decoded)
        _, output_rate = infer.expected_output_layout(config, decoded_info.samplerate)
        source_frames = math.ceil(decoded_info.frames * (output_rate / decoded_info.samplerate))
    source_hash = session.input_sha256_of(source)
    job = {"mix_path": str(source), "model_id": row["model_id"], "stem_target": "Instrumental"}

    queue = SeparationQueue(worker.handle_job, state_path=tmp_path / "queue.json")
    queue.add([source])
    queue.run()
    item = queue.snapshot()[0]
    assert item.status == "completed", item.result
    result = item.result
    assert result["status"] == "ok", result
    # 要求した意味的な結果は伴奏だけ。既定モデルが相補stemも作るが、公開しない。
    assert [Path(p).stem.rsplit(" - ", 1)[-1] for p in result["outputs"]] == ["Instrumental"]
    saved = sorted(p.name for p in Path(result["outputs"][0]).parent.iterdir())
    assert saved == [Path(result["outputs"][0]).name], saved
    infos = [sf.info(p) for p in result["outputs"]]
    assert len({(i.frames, i.samplerate, i.channels) for i in infos}) == 1
    assert infos[0].frames == source_frames
    assert infos[0].samplerate == output_rate
    assert infos[0].channels == 2  # 既定モデルは設定でmono入力のstereo複製を宣言する。
    assert result["peak_vram_mb"] > 0
    assert torch.cuda.memory_allocated() < 64 * 1024**2, "モデル本体のVRAM割当を保持しない"
    assert torch.cuda.memory_reserved() < 64 * 1024**2, "モデル本体の予約領域を保持しない"
    for path in result["outputs"]:
        tags = FLAC(path)
        assert tags["TITLE"] == (
            ["入力の情報"] if input_format in {"m4a", "wma"} else ["分離確認用の音声"]
        )
        assert tags["LSS_INPUT_SHA256"] == [source_hash]
        assert tags["LSS_MODEL"] == [row["model_id"]]
        assert len(tags["LSS_MODEL_SHA256"][0]) == 64
        assert len(tags["LSS_CONFIG_SHA256"][0]) == 64
        assert tags["LSS_RUNTIME_ID"] == [result["runtime_id"]]
        assert '"device":"cuda:0"' in tags["LSS_RUNTIME_REVISION"][0]
    before = {p: Path(p).read_bytes() for p in result["outputs"]}
    assert worker.handle_job(job)["status"] == "skipped"
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    missing = Path(result["outputs"][0])
    missing.unlink()
    retry = worker.handle_job(job)
    assert retry["status"] == "ok", retry
    assert retry["outputs"] == result["outputs"]
    assert missing.exists()
    assert all(Path(p).read_bytes() == data for p, data in before.items() if Path(p) != missing)
    assert worker.handle_job(job)["status"] == "skipped"
    assert session.input_sha256_of(source) == source_hash
    print(f"実GPU確認: 出力{len(infos)}件, {infos[0].frames} samples, "
          f"RTF={result['rtf']:.3f}, peak VRAM={result['peak_vram_mb']:.1f} MB")
