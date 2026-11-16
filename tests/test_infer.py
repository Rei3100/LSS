from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from lss.harness.infer import (
    _runtime_adapter_paths,
    expected_output_layout,
    msst_separate,
    resolve_target_filename,
)
from lss.core.runtime import RuntimeRevision

import pytest


@pytest.fixture(autouse=True)
def _fake_cuda_runtime(monkeypatch):
    revision = RuntimeRevision(
        provider="msst", device="cuda:0", python_version="3.11", platform="test",
        torch_version="test", cuda_build="test", cudnn_version="test",
        gpu_name="test gpu", gpu_capability="12.0", adapter_artifacts=(),
    )
    monkeypatch.setattr("lss.harness.infer.capture_cuda_runtime", lambda *a, **k: revision)
    return revision


def _write_config(tmp_path: Path, training_yaml: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(f"training:\n{training_yaml}\n", encoding="utf-8")
    return path


def test_frozen_runtime_adapter_uses_bundled_source(tmp_path, monkeypatch):
    bundled = tmp_path / "bundle"
    monkeypatch.setattr("lss.harness.infer.__file__", str(tmp_path / "missing" / "infer.py"))
    monkeypatch.setattr("lss.harness.infer.resource_root", lambda: bundled)

    paths = _runtime_adapter_paths(bundled / "vendor" / "msst")

    assert paths == (
        bundled / "runtime-sources" / "lss" / "harness" / "infer.py",
        bundled / "vendor" / "msst",
    )


def test_resolve_target_filename_uses_target_instrument_when_present(tmp_path):
    config_path = _write_config(tmp_path, "  instruments: ['vocals', 'other']\n  target_instrument: other\n")

    assert resolve_target_filename(config_path, "Instrumental") == "other"


def test_resolve_target_filename_matches_instruments_case_insensitively_when_no_target(tmp_path):
    config_path = _write_config(tmp_path, "  instruments:\n  - Vocals\n  - Instrumental\n")

    assert resolve_target_filename(config_path, "instrumental") == "Instrumental"


def test_resolve_target_filename_raises_when_stem_not_found(tmp_path):
    config_path = _write_config(tmp_path, "  instruments:\n  - Vocals\n  - Drums\n")

    with pytest.raises(ValueError):
        resolve_target_filename(config_path, "Instrumental")


def test_expected_output_layout_uses_config_samplerate(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "audio:\n  sample_rate: 48000\ntraining:\n  instruments: [Vocals, Instrumental]\n",
        encoding="utf-8",
    )

    assert expected_output_layout(config_path, 44100) == (["Vocals", "Instrumental"], 48000)


def _separation_setup(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    mix_path = source / "選択した音声.wav"
    sf.write(mix_path, np.full((44100, 2), 0.125), 44100)
    sf.write(source / "別の音声.wav", np.full((44100, 2), 0.75), 44100)
    nested = source / "sub"
    nested.mkdir()
    sf.write(nested / "さらに別の音声.wav", np.full((44100, 2), 0.5), 44100)
    config = _write_config(tmp_path, "  instruments: [Vocals, Instrumental]")
    return mix_path, config, tmp_path / "output"


def test_separate_only_passes_selected_file_to_folder_backend(tmp_path, monkeypatch):
    mix_path, config, store = _separation_setup(tmp_path)
    original = {p: p.read_bytes() for p in mix_path.parent.rglob("*.wav")}
    seen = []

    def proc_folder(args):
        folder = Path(args["input_folder"])
        seen.append(folder)
        files = list(folder.rglob("*"))
        assert len(files) == 1
        assert files[0].read_bytes() == original[mix_path]
        data, sr = sf.read(files[0])
        for stem in ["Vocals", "Instrumental"]:
            sf.write(Path(args["store_dir"]) / f"{stem}.wav", data, sr)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    estimate, sr = msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store)
    assert sr == 44100
    np.testing.assert_allclose(estimate, 0.125)
    assert not seen[0].exists()
    assert seen[0].parent == store
    assert all(p.read_bytes() == data for p, data in original.items())


def test_worker_mode_skips_loading_return_waveform_and_reports_input_copy(tmp_path, monkeypatch):
    mix_path, config, store = _separation_setup(tmp_path)
    events = []

    def proc_folder(args):
        for stem in ["Vocals", "Instrumental"]:
            sf.write(Path(args["store_dir"]) / f"{stem}.wav", np.zeros((44100, 2)), 44100)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    monkeypatch.setattr(
        "lss.harness.infer.sf.read",
        lambda *a, **k: pytest.fail("製品ワーカーでは出力波形を再読込しない"),
    )

    result = msst_separate(
        "arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store,
        progress=events.append, load_estimate=False,
    )

    assert result is None
    assert events[0]["phase"] == "runtime_ready"
    assert events[0]["device"] == "cuda:0"
    assert len(events[0]["runtime_id"]) == 64
    assert events[1] == {
        "phase": "staging_input", "completed_bytes": 0,
        "total_bytes": mix_path.stat().st_size,
    }
    assert events[-1]["completed_bytes"] == events[-1]["total_bytes"]
    assert not any(path.is_dir() for path in store.iterdir())


def test_separate_pins_cuda_device_and_reports_runtime(tmp_path, monkeypatch, _fake_cuda_runtime):
    mix_path, config, store = _separation_setup(tmp_path)
    seen = []
    observed = []

    def proc_folder(args):
        seen.append(args)
        for stem in ["Vocals", "Instrumental"]:
            sf.write(Path(args["store_dir"]) / f"{stem}.wav", np.zeros((44100, 2)), 44100)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    msst_separate(
        "arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store,
        device_index=0, runtime_observer=observed.append, load_estimate=False,
    )

    assert seen[0]["device_ids"] == [0]
    assert seen[0]["force_cpu"] is False
    assert observed == [_fake_cuda_runtime]


def test_separate_rejects_stale_output_before_backend(tmp_path, monkeypatch):
    mix_path, config, store = _separation_setup(tmp_path)
    store.mkdir()
    stale = store / "Instrumental.wav"
    stale.write_bytes(b"stale")
    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=lambda args: pytest.fail("実行されてはいけない")))
    with pytest.raises(ValueError, match="空ではありません"):
        msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store)
    assert stale.read_bytes() == b"stale"


@pytest.mark.parametrize("stems", [[], ["Instrumental"]])
def test_missing_any_expected_stem_is_error(tmp_path, monkeypatch, stems):
    mix_path, config, store = _separation_setup(tmp_path)
    seen = []

    def proc_folder(args):
        seen.append(Path(args["input_folder"]))
        for stem in stems:
            sf.write(store / f"{stem}.wav", np.zeros((44100, 2)), 44100)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    with pytest.raises(FileNotFoundError, match="揃っていません"):
        msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store)
    assert not seen[0].exists()


def test_input_copy_is_removed_when_backend_raises(tmp_path, monkeypatch):
    mix_path, config, store = _separation_setup(tmp_path)
    seen = []

    def proc_folder(args):
        seen.append(Path(args["input_folder"]))
        raise RuntimeError("推論失敗")

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    with pytest.raises(RuntimeError, match="推論失敗"):
        msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store)
    assert not seen[0].exists()


@pytest.mark.parametrize("frames,channels,samplerate", [
    (22050, 2, 44100),  # 両方のstemが同じ長さでも、入力の途中で切れている。
    (44099, 2, 44100),
    (44101, 2, 44100),
    (44100, 1, 44100),
    (44100, 2, 48000),
    (0, 2, 44100),
])
@pytest.mark.parametrize("load_estimate", [False, True])
def test_malformed_provider_layout_is_rejected_before_return(
    tmp_path, monkeypatch, frames, channels, samplerate, load_estimate,
):
    mix_path, config, store = _separation_setup(tmp_path)

    def proc_folder(args):
        for stem in ["Vocals", "Instrumental"]:
            sf.write(store / f"{stem}.wav", np.zeros((frames, channels)), samplerate)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    with pytest.raises(ValueError, match="分離出力.*一致"):
        msst_separate(
            "arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store,
            load_estimate=load_estimate,
        )


@pytest.mark.parametrize("source_rate,output_rate,frames", [
    (48000, 44100, 48001), (44100, 48000, 44101), (48000, 48000, 48001),
])
def test_declared_resampling_preserves_exact_provider_frame_count(
    tmp_path, monkeypatch, source_rate, output_rate, frames,
):
    mix_path, config, store = _separation_setup(tmp_path)
    sf.write(mix_path, np.zeros((frames, 2)), source_rate)
    config.write_text(
        f"audio:\n  sample_rate: {output_rate}\ntraining:\n  instruments: [Vocals, Instrumental]\n",
        encoding="utf-8",
    )
    expected_frames = (frames * output_rate + source_rate - 1) // source_rate

    def proc_folder(args):
        for stem in ["Vocals", "Instrumental"]:
            sf.write(store / f"{stem}.wav", np.zeros((expected_frames, 2)), output_rate)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    estimate, sr = msst_separate(
        "arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store,
    )
    assert estimate.shape == (2, expected_frames)
    assert sr == output_rate


@pytest.mark.parametrize("configured_channels,output_channels", [(None, 1), (2, 2)])
def test_mono_layout_follows_declared_provider_channel_policy(
    tmp_path, monkeypatch, configured_channels, output_channels,
):
    mix_path, config, store = _separation_setup(tmp_path)
    sf.write(mix_path, np.zeros(44100), 44100)
    audio_yaml = "" if configured_channels is None else "audio:\n  num_channels: 2\n"
    config.write_text(audio_yaml + "training:\n  instruments: [Vocals, Instrumental]\n", encoding="utf-8")

    def proc_folder(args):
        for stem in ["Vocals", "Instrumental"]:
            sf.write(store / f"{stem}.wav", np.zeros((44100, output_channels)), 44100)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(proc_folder=proc_folder))
    estimate, _ = msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", store)
    assert estimate.shape == (output_channels, 44100)


def _fake_loading_inference(monkeypatch, calls):
    """Stand-in for vendor inference: load_model counts loads; proc_folder writes a value derived from the model."""
    def load_model(args):
        calls.append("load")
        return {"model": 3.0, "key": _key(args)}

    def _key(args):
        return tuple(str(args.get(k)) for k in ("model_type", "config_path", "start_check_point", "force_cpu", "device_ids"))

    def proc_folder(args, loaded=None):
        if loaded is None:
            loaded = load_model(args)
        data, sr = sf.read(next(Path(args["input_folder"]).iterdir()))
        for stem in ["Vocals", "Instrumental"]:
            sf.write(Path(args["store_dir"]) / f"{stem}.wav", data * loaded["model"] / 3.0, sr)

    monkeypatch.setitem(sys.modules, "inference", SimpleNamespace(
        proc_folder=proc_folder, load_model=load_model, _loaded_key=_key))


def test_resident_model_loads_once_and_outputs_equal_fresh_loads(tmp_path, monkeypatch):
    from lss.harness.infer import ResidentModel
    mix_path, config, _ = _separation_setup(tmp_path)
    fresh_calls, resident_calls = [], []
    _fake_loading_inference(monkeypatch, fresh_calls)
    fresh = [msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", tmp_path / f"f{i}")
             for i in range(3)]
    _fake_loading_inference(monkeypatch, resident_calls)
    resident = ResidentModel()
    kept = [msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", tmp_path / f"r{i}",
                          resident_model=resident) for i in range(3)]
    assert fresh_calls == ["load"] * 3
    assert resident_calls == ["load"] and resident.load_count == 1
    for (a, ra), (b, rb) in zip(fresh, kept):
        assert ra == rb and a.tobytes() == b.tobytes()


def test_resident_model_refuses_a_different_checkpoint(tmp_path, monkeypatch):
    from lss.harness.infer import ResidentModel
    mix_path, config, _ = _separation_setup(tmp_path)
    _fake_loading_inference(monkeypatch, [])
    resident = ResidentModel()
    msst_separate("arch", mix_path, tmp_path / "ckpt", config, "Instrumental", tmp_path / "a", resident_model=resident)
    with pytest.raises(ValueError):
        msst_separate("arch", mix_path, tmp_path / "other", config, "Instrumental", tmp_path / "b",
                      resident_model=resident)


def test_vendor_proc_folder_reuses_loaded_model_for_run_folder(tmp_path, monkeypatch):
    msst = Path(__file__).resolve().parents[1] / "vendor" / "msst"
    monkeypatch.syspath_prepend(str(msst))
    monkeypatch.delitem(sys.modules, "inference", raising=False)
    import torch
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    import inference as vendor

    loads, runs = [], []
    class Attr(dict):
        __getattr__ = dict.__getitem__

    config = SimpleNamespace(training=Attr(instruments=["vocals"]))
    model = torch.nn.Linear(1, 1)

    def get_model(model_type, config_path):
        loads.append(model_type)
        return model, config

    monkeypatch.setattr(vendor, "get_model_from_config", get_model)
    monkeypatch.setattr(vendor, "run_folder", lambda m, a, c, d, verbose=False: runs.append((m, a.input_folder, a.model_type, d)))
    base = {"model_type": "mel_band_roformer", "config_path": "c.yaml", "start_check_point": "",
            "store_dir": str(tmp_path), "device_ids": [0], "force_cpu": True}
    handle = vendor.load_model({**base, "input_folder": "x"})
    vendor.proc_folder({**base, "input_folder": "a"}, loaded=handle)
    vendor.proc_folder({**base, "input_folder": "b"}, loaded=handle)
    assert loads == ["mel_band_roformer"]
    assert [r[0] for r in runs] == [model, model] and [r[1] for r in runs] == ["a", "b"]
    with pytest.raises(ValueError):
        vendor.proc_folder({**base, "config_path": "other.yaml", "input_folder": "c"}, loaded=handle)
    monkeypatch.delitem(sys.modules, "inference", raising=False)
