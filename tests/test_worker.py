import threading
import shutil
import time
from contextlib import contextmanager
from multiprocessing import connection
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from lss.core import audio_input, gpu_lease, models_store, session, worker
from lss.core.runtime import RuntimeRevision
from lss.harness import fetch, infer, tsv_io, vram
from lss.output import writer


def _write_wav(path: Path, channels: int = 2, sr: int = 44100, seconds: float = 1.0) -> None:
    frames = int(sr * seconds)
    wave = (0.01 * np.random.randn(frames, channels)).astype("float32")
    sf.write(path, wave, sr)


def _write_models_tsv(
    tmp_path: Path, model_id: str = "model_x", size_bytes: str = "", state: str = "候補",
) -> Path:
    header = [
        "model_id", "arch", "distributor_url", "config_url", "size_bytes",
        "状態", "last_used_at", "rtf", "peak_vram_mb", "note",
    ]
    rows = [{
        "model_id": model_id,
        "arch": "mel_band_roformer",
        "distributor_url": "https://example.com/ckpt",
        "config_url": "https://example.com/config",
        "size_bytes": size_bytes,
        "状態": state,
        "last_used_at": "",
        "rtf": "",
        "peak_vram_mb": "",
        "note": "",
    }]
    path = tmp_path / "models.tsv"
    tsv_io.save_tsv(path, header, rows)
    return path


def _fake_msst_separate(
    arch, mix_path, ckpt_path, config_path, stem_target, store_dir, **kwargs,
):
    observer = kwargs.get("runtime_observer")
    if observer is not None:
        observer(kwargs["runtime_revision"])
    data, sr = sf.read(mix_path, always_2d=True)
    sf.write(store_dir / "Instrumental.wav", data * 0.5, sr)
    sf.write(store_dir / "Vocals.wav", data * 0.5, sr)
    return (data * 0.5).T.astype("float32"), sr


def _setup(tmp_path, monkeypatch, model_id="model_x", state="候補"):
    # このfixtureは推論もRuntimeも模擬する。実CUDAの所有待ちには入らない。
    # 所有権の振る舞いはtest_worker_gpu_leaseと実mutex試験で別に検証する。
    monkeypatch.setattr(gpu_lease, "acquire", lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    models_tsv = _write_models_tsv(tmp_path, model_id=model_id, state=state)
    monkeypatch.setattr(models_store, "MODELS_TSV", models_tsv)
    (tmp_path / "config").write_text(
        "audio:\n  sample_rate: 44100\ntraining:\n  instruments: [Vocals, Instrumental]\n",
        encoding="utf-8",
    )
    runtime_revision = RuntimeRevision(
        provider="msst", device="cuda:0", python_version="3.11", platform="test",
        torch_version="test", cuda_build="test", cudnn_version="test",
        gpu_name="test gpu", gpu_capability="12.0", adapter_artifacts=(("provider.py", "a" * 64),),
        dependency_versions=(("torch", "test"),),
    )
    monkeypatch.setattr(infer, "capture_msst_runtime", lambda: runtime_revision)
    return models_tsv


def test_handle_job_rejects_multichannel_before_loading_model(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *a, **k: calls.append(1) or (tmp_path / "ckpt", tmp_path / "config"),
    )
    mix_path = tmp_path / "mix6ch.wav"
    _write_wav(mix_path, channels=6)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}

    result = worker.handle_job(job)

    assert result["status"] == "error"
    assert result["error_code"] == "unsupported_channels"
    # 表示は「Nch は分離できません」を実測値で書く。文字列から掘り返させない。
    assert result["channels"] == 6
    assert calls == []


@pytest.mark.parametrize("channels", [1, 2])
def test_handle_job_accepts_mono_and_stereo_channel_layouts(tmp_path, monkeypatch, channels):
    """実機で観測した誤判定の再発防止。上限内のチャンネル構成を弾かない。"""
    _setup(tmp_path, monkeypatch)
    reached = []
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *a, **k: reached.append(1) or (tmp_path / "ckpt", tmp_path / "config"),
    )
    mix_path = tmp_path / f"mix{channels}ch.wav"
    _write_wav(mix_path, channels=channels)
    job = {
        "cmd": "separate", "mix_path": str(mix_path),
        "model_id": "model_x", "stem_target": "Instrumental",
    }

    result = worker.handle_job(job)

    # モデル取得まで到達していれば、チャンネル判定では止まっていない。
    assert reached == [1]
    assert result.get("error_code") != "unsupported_channels"


@pytest.mark.parametrize("job", [
    {},
    {"mix_path": "input.wav", "model_id": "model_x", "stem_target": ""},
    {"cmd": "delete", "mix_path": "input.wav", "model_id": "model_x", "stem_target": "Vocals"},
])
def test_invalid_job_is_rejected_with_machine_readable_code(tmp_path, monkeypatch, job):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: pytest.fail("モデルへ到達しない"))

    result = worker.handle_job(job)

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_job"


def test_unknown_model_has_machine_readable_code(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "missing", "stem_target": "Vocals",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "unknown_model"


@pytest.mark.parametrize(("state", "error_code"), [
    ("除外", "model_unavailable"),
    ("保留", "model_unavailable"),
    ("", "invalid_model_ledger"),
])
def test_unavailable_model_is_rejected_before_loading_input(tmp_path, monkeypatch, state, error_code):
    _setup(tmp_path, monkeypatch, state=state)
    monkeypatch.setattr(audio_input, "prepare_input", lambda *a, **k: pytest.fail("入力を読み込まない"))

    result = worker.handle_job({
        "mix_path": str(tmp_path / "missing.wav"),
        "model_id": "model_x", "stem_target": "Vocals",
    })

    assert result["status"] == "error"
    assert result["error_code"] == error_code


def test_invalid_model_source_has_machine_readable_code(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)

    def reject_source(*args, **kwargs):
        raise fetch.InvalidModelSourceError("HTTPSが必要です")

    monkeypatch.setattr(models_store, "ensure_model_cached", reject_source)
    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Vocals",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_model_source"


def test_model_download_failure_has_machine_readable_code(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)

    def fail_download(*args, **kwargs):
        raise fetch.ModelDownloadError("接続できません")

    monkeypatch.setattr(models_store, "ensure_model_cached", fail_download)
    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Vocals",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "model_download_failed"


def test_unavailable_gpu_has_machine_readable_code(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *args, **kwargs: (tmp_path / "ckpt", tmp_path / "config"),
    )
    monkeypatch.setattr(
        infer, "expected_output_layout",
        lambda *args, **kwargs: (["Vocals", "Instrumental"], 44100),
    )
    monkeypatch.setattr(
        infer, "capture_msst_runtime",
        lambda: (_ for _ in ()).throw(infer.RuntimeUnavailableError("CUDA GPUを利用できません")),
    )

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "gpu_unavailable"


def test_success_persists_exact_runtime_and_config_provenance(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *args, **kwargs: (tmp_path / "ckpt", tmp_path / "config"),
    )
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "ok"
    assert len(result["runtime_id"]) == 64
    assert len(result["config_sha256"]) == 64
    for output in result["outputs"]:
        output_tags = writer.read_source_tags(Path(output))
        assert output_tags["LSS_RUNTIME_ID"] == result["runtime_id"]
        assert output_tags["LSS_CONFIG_SHA256"] == result["config_sha256"]
        assert '"device":"cuda:0"' in output_tags["LSS_RUNTIME_REVISION"]


def test_handle_job_can_publish_outside_read_only_source_tree(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    source_root = tmp_path / "source"
    source_root.mkdir()
    mix_path = source_root / "mix.wav"
    _write_wav(mix_path)
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *args, **kwargs: (tmp_path / "ckpt", tmp_path / "config"),
    )
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    destination = tmp_path / "validation" / "artifacts"

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental",
    }, destination_dir=destination)

    assert result["status"] == "ok"
    assert all(Path(path).parent == destination.resolve() for path in result["outputs"])
    assert not (source_root / "LSS").exists()


@pytest.mark.parametrize(
    "failure,error_code",
    [
        (fetch.DownloadTooLargeError("上限超過"), "model_download_too_large"),
        (fetch.DownloadIntegrityError("ハッシュ不一致"), "model_download_invalid"),
    ],
)
def test_model_download_size_failures_have_precise_codes(
    tmp_path, monkeypatch, failure, error_code,
):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Vocals",
    })

    assert result["status"] == "error"
    assert result["error_code"] == error_code


def test_invalid_model_ledger_has_machine_readable_code(tmp_path, monkeypatch):
    models_tsv = _write_models_tsv(tmp_path)
    header, rows = tsv_io.load_tsv(models_tsv)
    rows.append(dict(rows[0]))
    tsv_io.save_tsv(models_tsv, header, rows)
    monkeypatch.setattr(models_store, "MODELS_TSV", models_tsv)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Vocals",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_model_ledger"


@pytest.mark.parametrize(("failure", "error_code"), [
    (audio_input.DecoderUnavailableError("ffmpeg missing"), "decoder_unavailable"),
    (audio_input.InputDecodeTimeoutError("decode timed out"), "input_decode_timeout"),
    (audio_input.InputDecodeError("decode failed"), "input_decode_failed"),
])
def test_input_decode_failure_has_machine_readable_code(tmp_path, monkeypatch, failure, error_code):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    monkeypatch.setattr(
        audio_input, "prepare_input", lambda *a, **k: (_ for _ in ()).throw(failure),
    )

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Vocals",
    })

    assert result["status"] == "error"
    assert result["error_code"] == error_code


def test_unknown_stem_is_rejected_before_inference(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"),
    )
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("推論しない"))

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Drums",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_stem_target"


def test_handle_job_returns_ok_and_records_metrics(tmp_path, monkeypatch):
    models_tsv = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}

    result = worker.handle_job(job)

    assert result["status"] == "ok"
    # 要求したのはInstrumentalだけ。モデルが作る相補stemは成果物にならない。
    assert [Path(p).name for p in result["outputs"]] == ["mix - Instrumental.flac"]
    assert sorted(p.name for p in (mix_path.parent / "LSS").iterdir()) == [
        "mix - Instrumental.flac",
    ]
    assert result["input_sha256"] == session.input_sha256_of(mix_path)
    assert result["recipe_id"].startswith("separation-v1-")
    assert result["recipe_revision"]["stage_graph"][0]["model_id"] == "model_x"
    assert result["elapsed_seconds"] >= 0
    assert all(Path(p).is_file() for p in result["outputs"])
    assert list((tmp_path / "state" / "work").iterdir()) == []
    _, rows = tsv_io.load_tsv(models_tsv)
    assert rows[0]["rtf"] != ""
    assert rows[0]["peak_vram_mb"] != ""


def test_stem_above_normalized_range_still_publishes(tmp_path, monkeypatch):
    """実機で全曲が失敗した経路。1.0超は正常な分離結果で、成果物になるまで通す。

    実測: 通常の音源13件すべてで instrumental の sample peak が 1.04〜1.40 になり、
    ここが `unsafe_output_serialization` で止まっていた。
    """
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))

    def loud_separate(arch, mix_path, ckpt_path, config_path, stem_target, store_dir, **kwargs):
        observer = kwargs.get("runtime_observer")
        if observer is not None:
            observer(kwargs["runtime_revision"])
        data, sr = sf.read(mix_path, always_2d=True)
        loud = np.clip(data * 200.0, -1.4, 1.4).astype("float32")
        sf.write(store_dir / "Instrumental.wav", loud, sr, subtype="FLOAT")
        sf.write(store_dir / "Vocals.wav", loud * 0.25, sr, subtype="FLOAT")
        return loud.T, sr

    monkeypatch.setattr(infer, "msst_separate", loud_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x",
           "stem_target": "Instrumental"}

    result = worker.handle_job(job)

    assert result["status"] == "ok", result
    outputs = [Path(path) for path in result["outputs"]]
    assert [p.name for p in outputs] == ["mix - Instrumental.flac"]
    shifts = set()
    for output in outputs:
        assert output.is_file()
        from mutagen.flac import FLAC

        shifts.add(FLAC(output)["LSS_OUTPUT_HEADROOM_SHIFT"][0])
        samples, rate = sf.read(output, dtype="float64", always_2d=True)
        assert rate == 44100
        assert np.isfinite(samples).all()
        assert float(np.max(np.abs(samples))) <= 1.0
    # 公開単位で1つの倍率。破棄する内部生成物のピークでは決めない。
    assert shifts == {"1"}


def test_handle_job_reports_processing_phases_without_affecting_result(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    events = []

    result = worker.handle_job(
        {"mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"},
        progress=events.append,
    )

    assert result["status"] == "ok"
    phases = [event["phase"] for event in events]
    assert phases[0] == "preparing_input"
    assert phases[-1] == "completed"
    assert phases.index("hashing_input") < phases.index("checking_model") < phases.index("checking_runtime") < phases.index("separating")
    assert phases.index("analyzing_outputs") < phases.index("writing_outputs")
    assert phases.index("writing_outputs") < phases.index("recording_completion")
    writing = [event for event in events if event["phase"] == "writing_outputs"]
    assert [event["completed_stems"] for event in writing] == [0, 1]
    assert {
        event["checkpoint"] for event in events if event["phase"] == "verifying_input"
    } == {"after_decode", "before_reuse", "after_metadata", "after_separation", "before_publish"}


def test_progress_consumer_failure_does_not_abort_processing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)

    result = worker.handle_job(
        {"mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"},
        progress=lambda event: (_ for _ in ()).throw(RuntimeError("表示失敗")),
    )

    assert result["status"] == "ok"


def test_insufficient_output_space_stops_before_inference(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(writer.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("推論前に停止する必要があります"))
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "insufficient_output_space"
    assert "出力先の空き容量が不足しています" in result["message"]


def test_handle_job_skips_when_already_completed(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *a, **k: calls.append(1) or (tmp_path / "ckpt", tmp_path / "config"),
    )
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    first = worker.handle_job(job)
    assert first["status"] == "ok"
    calls.clear()

    result = worker.handle_job(job)

    assert result["status"] == "skipped"
    assert result["outputs"] == first["outputs"]
    assert calls == [1]


def test_config_revision_change_creates_new_recipe_instead_of_reusing_old_output(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"),
    )
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    source = tmp_path / "mix.wav"
    _write_wav(source)
    job = {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}

    first = worker.handle_job(job)
    (tmp_path / "config").write_text(
        (tmp_path / "config").read_text(encoding="utf-8") + "# output-affecting revision\n",
        encoding="utf-8",
    )
    second = worker.handle_job(job)

    assert first["status"] == second["status"] == "ok"
    assert first["recipe_id"] != second["recipe_id"]
    assert first["outputs"] != second["outputs"]
    assert all("(r2)" in Path(path).name for path in second["outputs"])


def test_changing_requested_part_never_reuses_another_parts_completion(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    calls = []

    def separate(*args, **kwargs):
        calls.append(args[4])
        return _fake_msst_separate(*args, **kwargs)

    monkeypatch.setattr(infer, "msst_separate", separate)
    source = tmp_path / "source.wav"
    _write_wav(source)
    job = {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}
    instrumental = worker.handle_job(job)
    vocals = worker.handle_job({**job, "stem_target": "Vocals"})

    assert instrumental["status"] == vocals["status"] == "ok"
    assert instrumental["recipe_id"] != vocals["recipe_id"]
    assert [Path(p).name for p in vocals["outputs"]] == ["source - Vocals.flac"]
    assert calls == ["Instrumental", "Vocals"]
    # 同じ意味の要求は大小文字が違っても、正しいパートの既存成果物を再利用する。
    for target, original in [("instrumental", instrumental), ("VOCALS", vocals)]:
        reused = worker.handle_job({**job, "stem_target": target})
        assert reused["status"] == "skipped"
        assert reused["outputs"] == original["outputs"]
    assert calls == ["Instrumental", "Vocals"]


def test_zero_result_for_one_part_does_not_suppress_another_part(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    source = tmp_path / "source.wav"
    _write_wav(source)
    job = {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}
    first = worker.handle_job(job)
    assert first["status"] == "ok"
    session.mark_zero_result(
        first["input_sha256"], first["recipe_id"], source_path=source,
        zero_result={"reason": "music_not_wanted", "validation_result": {
            "validator_id": "test-semantic-validator", "validator_revision": "1",
            "hard_gates_passed": True,
        }},
    )

    vocals = worker.handle_job({**job, "stem_target": "Vocals"})

    assert vocals["status"] == "ok"
    assert [Path(p).name for p in vocals["outputs"]] == ["source - Vocals.flac"]
    assert "zero_result" not in vocals


def test_truncated_provider_output_never_reaches_publication(tmp_path, monkeypatch):
    import sys

    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    source = tmp_path / "source.wav"
    _write_wav(source)
    original = source.read_bytes()

    def proc_folder(args):
        for stem in ["Vocals", "Instrumental"]:
            sf.write(Path(args["store_dir"]) / f"{stem}.wav", np.zeros((22050, 2)), 44100)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    monkeypatch.setattr(writer, "write_recipe_output", lambda *a, **k: pytest.fail("不完全な出力を公開しない"))
    result = worker.handle_job({
        "mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "invalid_separation_output"
    assert not session.completion_ledger_path().exists()
    assert not (tmp_path / "LSS").exists()
    assert not list(session.work_dir().iterdir())
    assert source.read_bytes() == original


def test_legacy_model_id_completion_is_not_reused_as_strict_recipe(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    source = tmp_path / "source.wav"
    _write_wav(source)
    output = tmp_path / "output.flac"
    output.write_bytes(b"output")
    source_sha256 = session.input_sha256_of(source)
    session.mark_completed(
        source_sha256, "model_x", source_path=source, outputs=[output],
    )
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)

    result = worker.handle_job({
        "mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "ok"
    assert result["recipe_id"] != "model_x"
    assert str(output) not in result["outputs"]


def test_legacy_model_id_zero_result_is_not_reused_as_strict_recipe(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    source = tmp_path / "source.wav"
    _write_wav(source)
    source_sha256 = session.input_sha256_of(source)
    zero_result = {
        "reason": "no_novel_asset",
        "validation_result": {
            "validator_id": "bgm-novelty-gate", "validator_revision": "r1",
            "hard_gates_passed": True, "evidence": {"known_matches": 2},
        },
    }
    session.mark_zero_result(
        source_sha256, "model_x", source_path=source, zero_result=zero_result,
    )
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)

    result = worker.handle_job({
        "mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "ok"
    assert result["recipe_id"] != "model_x"


def test_input_changed_during_decode_is_rejected_before_model_loading(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    source = tmp_path / "source.m4a"
    source.write_bytes(b"original source")
    decoded = tmp_path / "decoded.wav"
    _write_wav(decoded)

    @contextmanager
    def replace_during_decode(path, **kwargs):
        path.write_bytes(b"replacement source")
        yield decoded

    monkeypatch.setattr(audio_input, "prepare_input", replace_during_decode)
    monkeypatch.setattr(
        models_store, "ensure_model_cached", lambda *a, **k: pytest.fail("モデルを読み込まない"),
    )

    result = worker.handle_job({
        "mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "input_changed"


def test_short_input_reports_limit_breach_without_excluding_model(tmp_path, monkeypatch):
    models_tsv = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path, seconds=1.0)
    # 所要時間の時計だけを制御し、scratchの保持時刻や他moduleの時計を変えない。
    values = iter([0.0, 100.0])
    monkeypatch.setattr(worker, "time", SimpleNamespace(time=lambda: next(values)))
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}

    result = worker.handle_job(job)

    assert result["status"] == "ok"
    assert result["limits_exceeded"] is True
    assert result["excluded"] is False
    _, rows = tsv_io.load_tsv(models_tsv)
    assert rows[0]["状態"] == "候補"


@pytest.mark.parametrize("change", ["delete", "modify", "copy_source", "legacy"])
def test_retry_recovers_outputs_and_preserves_existing_files(tmp_path, monkeypatch, change):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    source = tmp_path / "入力.wav"
    _write_wav(source)
    job = {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}
    first = worker.handle_job(job)
    assert first["status"] == "ok", first
    paths = [Path(p) for p in first["outputs"]]
    if change == "delete":
        paths[0].unlink()
    elif change == "modify":
        # タグが読めるまま内容が変わったファイルも無言で再利用・上書きしない。
        with paths[0].open("ab") as f:
            f.write(b"modified")
    elif change == "copy_source":
        copied = tmp_path / "別の場所"
        copied.mkdir()
        source = copied / source.name
        shutil.copyfile(job["mix_path"], source)
        job["mix_path"] = str(source)
    else:
        legacy = {"input_sha256": session.input_sha256_of(source), "recipe_id": "model_x", "completed_at": "2026-08-14"}
        tsv_io.save_tsv(session.completion_ledger_path(), list(legacy), [legacy])
    existing = {p: p.read_bytes() for p in paths if p.exists()}

    retry = worker.handle_job(job)
    assert retry["status"] == "ok", retry
    assert len(retry["outputs"]) == 1
    assert all(p.read_bytes() == data for p, data in existing.items())
    if change == "modify":
        assert any("(r2)" in Path(p).name for p in retry["outputs"])
    elif change == "copy_source":
        assert all(Path(p).is_relative_to(source.parent) for p in retry["outputs"])
    else:
        assert retry["outputs"] == first["outputs"]
    skipped = worker.handle_job(job)
    assert skipped["status"] == "skipped"
    assert skipped["outputs"] == retry["outputs"]


def test_serve_ipc_roundtrip(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}

    address = fr"\\.\pipe\lss-worker-test-{uuid4().hex[:8]}"
    thread = threading.Thread(target=worker.serve, kwargs={"address": address}, daemon=True)
    thread.start()

    conn = None
    deadline = time.time() + 5
    last_exc = None
    while conn is None and time.time() < deadline:
        try:
            conn = connection.Client(address, authkey=worker.AUTHKEY)
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            last_exc = exc
            time.sleep(0.05)
    assert conn is not None, f"IPC接続に失敗: {last_exc}"

    conn.send(job)
    response = conn.recv()
    conn.close()

    assert response["status"] == "ok"


def test_empty_inference_is_error_and_retry_uses_fresh_work_dir(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}
    seen = []

    def empty_inference(*args, **kwargs):
        seen.append(args[-1])
        return np.zeros((2, 100)), 44100

    monkeypatch.setattr(infer, "msst_separate", empty_inference)
    assert worker.handle_job(job)["status"] == "error"
    assert not session.is_completed(session.input_sha256_of(mix_path), "model_x", mix_path)
    assert not seen[0].exists()

    def successful_inference(*args, **kwargs):
        seen.append(args[-1])
        return _fake_msst_separate(*args, **kwargs)

    monkeypatch.setattr(infer, "msst_separate", successful_inference)
    result = worker.handle_job(job)
    assert result["status"] == "ok"
    assert seen[1] != seen[0]
    assert not seen[1].exists()
    assert len(result["outputs"]) == 1


def test_inference_failure_releases_unused_vram(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"),
    )
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("失敗")))
    released = []
    monkeypatch.setattr(vram, "release_unused_vram", lambda: released.append(True))
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)

    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "error"
    assert released == [True]


def test_input_changed_during_inference_is_rejected_without_outputs(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    original_sha256 = session.input_sha256_of(mix_path)

    def separate_and_replace_input(*args, **kwargs):
        result = _fake_msst_separate(*args, **kwargs)
        with mix_path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(infer, "msst_separate", separate_and_replace_input)
    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "input_changed"
    assert "入力ファイルが変更されました" in result["message"]
    assert not (tmp_path / "LSS").exists()
    assert not session.is_completed(original_sha256, "model_x", mix_path)


def test_input_changed_during_output_staging_is_rejected_before_publish(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        models_store, "ensure_model_cached",
        lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"),
    )
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    original_sha256 = session.input_sha256_of(mix_path)
    original_stage = writer._stage_flac_from_file
    changed = False

    def stage_and_replace_input(*args, **kwargs):
        nonlocal changed
        temporary_path = original_stage(*args, **kwargs)
        if not changed:
            with mix_path.open("ab") as stream:
                stream.write(b"changed during output")
            changed = True
        return temporary_path

    monkeypatch.setattr(writer, "_stage_flac_from_file", stage_and_replace_input)
    result = worker.handle_job({
        "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental",
    })

    assert result["status"] == "error"
    assert result["error_code"] == "input_changed"
    assert list((tmp_path / "LSS").iterdir()) == []
    assert not session.is_completed(original_sha256, "model_x", mix_path)


def test_serve_survives_probe_connection_without_job(tmp_path, monkeypatch):
    """is_worker_running()相当(ハンドシェイクのみで即切断)を受けてもserve()ループが死なない回帰テスト。

    実E2E実測(AC7)で発見: 接続してjobを送らず切断する生存確認だけの接続でconn.recv()が
    EOFErrorを出し、それがserve()のwhileループを丸ごと落としていた(dispatch_jobは常に
    ensure_worker_running→is_worker_runningの生存確認を先に行うため、実運用で確実に踏む)。
    """
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}

    address = fr"\\.\pipe\lss-worker-test-{uuid4().hex[:8]}"
    thread = threading.Thread(target=worker.serve, kwargs={"address": address}, daemon=True)
    thread.start()

    def _connect():
        deadline = time.time() + 5
        last_exc = None
        while time.time() < deadline:
            try:
                return connection.Client(address, authkey=worker.AUTHKEY)
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last_exc = exc
                time.sleep(0.05)
        raise AssertionError(f"IPC接続に失敗: {last_exc}")

    probe_conn = _connect()
    probe_conn.close()  # jobを送らず即切断(is_worker_running()と同じ挙動)

    real_conn = _connect()
    real_conn.send(job)
    response = real_conn.recv()
    real_conn.close()

    assert response["status"] == "ok"


def test_skipping_installs_sample_identity_so_a_played_output_stays_reusable(tmp_path, monkeypatch):
    import json

    from mutagen.flac import FLAC

    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    source = tmp_path / "入力.wav"
    _write_wav(source)
    job = {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}
    first = worker.handle_job(job)
    assert first["status"] == "ok", first
    output = Path(first["outputs"][0])

    ledger = session.completion_ledger_path()
    header, rows = tsv_io.load_tsv(ledger)
    for row in rows:
        row["outputs"] = json.dumps([{key: value for key, value in item.items() if key != "audio"}
                                     for item in json.loads(row["outputs"])], ensure_ascii=False)
    tsv_io.save_tsv(ledger, header, rows)

    assert worker.handle_job(job)["status"] == "skipped"
    stored = json.loads(tsv_io.load_tsv(ledger)[1][-1]["outputs"])
    assert stored[0]["audio"]["frames"] > 0

    tags = FLAC(output)
    tags["LSS_TEST_PLAY_COUNT"] = ["1"]
    tags.save()
    played = output.read_bytes()
    assert session.input_sha256_of(output) != stored[0]["sha256"]
    assert worker.handle_job(job)["status"] == "skipped"
    assert output.read_bytes() == played


def _runtime(platform="test", **changes):
    return RuntimeRevision(**{**dict(
        provider="msst", device="cuda:0", python_version="3.11", platform=platform,
        torch_version="test", cuda_build="test", cudnn_version="test",
        gpu_name="test gpu", gpu_capability="12.0", adapter_artifacts=(("provider.py", "a" * 64),),
        dependency_versions=(("torch", "test"),)), **changes})


def test_a_runtime_only_change_reuses_the_completed_output_instead_of_a_second_copy(tmp_path, monkeypatch):
    """同じモデルの処理は、実行環境だけの更新で複製しない。

    実行環境だけの違いは品質の更新ではない（`lss.assets.reprocess`）。推論し直して「(r2)」を作らない。
    """
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}
    first = worker.handle_job(job)
    assert first["status"] == "ok"

    monkeypatch.setattr(infer, "capture_msst_runtime", lambda: _runtime("Windows-10-10.0.26300-SP0"))
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("実行環境だけの違いで推論し直さない"))
    again = worker.handle_job(job)

    assert again["status"] == "skipped" and again["outputs"] == first["outputs"]
    assert again["recipe_id"] == first["recipe_id"]  # 出力の来歴は作った時のRecipeのまま
    assert again["equivalent_to"]["basis"] == "runtime_only_difference"
    assert again["equivalent_to"]["recipe_id"] != first["recipe_id"]
    assert sorted(p.name for p in (mix_path.parent / "LSS").iterdir()) == ["mix - Instrumental.flac"]
    assert again["product_result"]["state"] == "REUSED"


def test_a_different_model_file_is_not_treated_as_the_same_processing(tmp_path, monkeypatch):
    models_tsv = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "mix.wav"
    _write_wav(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}
    first = worker.handle_job(job)
    (tmp_path / "config").write_text(
        "audio:\n  sample_rate: 44100\ntraining:\n  instruments: [Vocals, Instrumental]\n# changed\n", encoding="utf-8")
    monkeypatch.setattr(infer, "capture_msst_runtime", lambda: _runtime("other-os"))
    again = worker.handle_job(job)
    assert again["status"] == "ok" and again["recipe_id"] != first["recipe_id"]
    assert "equivalent_to" not in again


def _flac_source(path, seed=0):
    rng = np.random.default_rng(seed)
    sf.write(path, (rng.standard_normal((44100 * 3, 2)) * 0.05).astype("float32"), 44100, subtype="PCM_16")


def test_a_source_retagged_by_a_player_is_the_same_input_and_is_not_separated_again(tmp_path, monkeypatch):
    """合成音源のタグだけが変化しても、音声の同一性は保たれる。"""
    from mutagen.flac import FLAC

    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "song.flac"
    _flac_source(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}
    first = worker.handle_job(job)
    assert first["status"] == "ok"
    tags = FLAC(mix_path)
    tags["PLAY_COUNT"] = "7"
    tags["REPLAYGAIN_TRACK_GAIN"] = "-6.00 dB"
    tags.save()
    retagged = session.input_sha256_of(mix_path)
    assert retagged != first["input_sha256"]

    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("tagだけ違う原本を推論し直さない"))
    again = worker.handle_job(job)

    assert again["status"] == "skipped" and again["outputs"] == first["outputs"]
    assert again["input_sha256"] == retagged and again["provenance_input_sha256"] == first["input_sha256"]
    assert again["equivalent_to"]["basis"] == "same_decoded_input_audio"
    assert sorted(p.name for p in (mix_path.parent / "LSS").iterdir()) == ["song - Instrumental.flac"]
    assert FLAC(first["outputs"][0])["LSS_INPUT_SHA256"] == [first["input_sha256"]]  # 来歴は作った時の入力
    # 次からは今のbytesでも完了として見つかる（照会もr4も推論し直さない）。
    assert session.check_completed_outputs(retagged, first["recipe_id"], mix_path)
    assert worker.handle_job(job)["status"] == "skipped"


def test_different_audio_at_the_same_place_is_separated_again(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "song.flac"
    _flac_source(mix_path, seed=0)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}
    worker.handle_job(job)
    _flac_source(mix_path, seed=1)
    again = worker.handle_job(job)
    assert again["status"] == "ok" and "equivalent_to" not in again


def test_a_completion_without_recorded_input_audio_is_not_trusted_for_a_changed_source(tmp_path, monkeypatch):
    from mutagen.flac import FLAC
    from lss.harness import tsv_io

    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    mix_path = tmp_path / "song.flac"
    _flac_source(mix_path)
    job = {"cmd": "separate", "mix_path": str(mix_path), "model_id": "model_x", "stem_target": "Instrumental"}
    worker.handle_job(job)
    ledger = session.completion_ledger_path()
    header, rows = tsv_io.load_tsv(ledger)
    for row in rows:
        row["input_audio"] = ""  # 記録を始める前の完了
    tsv_io.save_tsv(ledger, header, rows)
    tags = FLAC(mix_path)
    tags["PLAY_COUNT"] = "1"
    tags.save()
    assert worker.handle_job(job)["status"] == "ok"  # 確かめられないので推論する（記録の無い完了を同じと見なさない）
