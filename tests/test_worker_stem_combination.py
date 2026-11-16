"""分離ジョブでのステムの組み合わせ（模擬分離器・合成信号）。

組み合わせの公開ファイルは宣言した利得での正確な和（直列化のheadroom倍率と量子化だけが違う）で、
来歴とタグを持ち、単一ステムの成果物のbytesとRecipe IDを変えない。部品が欠ければ何も公開しない。
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from lss.core import flac_frames, gpu_lease, models_store, session, worker
from lss.core.runtime import RuntimeRevision
from lss.paths import resource_root
from lss.harness import infer, tsv_io
from lss.output import writer

FOUR = ["drums", "bass", "other", "vocals"]
_GAINS = {"drums": 0.9, "bass": 0.5, "other": 0.3, "vocals": 0.9}


def _setup(tmp_path, monkeypatch, state_name="state"):
    monkeypatch.setattr(gpu_lease, "acquire", lambda **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / state_name))
    ledger = tmp_path / "models.tsv"
    header = ["model_id", "arch", "distributor_url", "config_url", "size_bytes", "状態",
              "last_used_at", "rtf", "peak_vram_mb", "note"]
    tsv_io.save_tsv(ledger, header, [{
        "model_id": "four", "arch": "scnet", "distributor_url": "https://example.com/ckpt",
        "config_url": "https://example.com/config", "size_bytes": "", "状態": "候補",
        "last_used_at": "", "rtf": "", "peak_vram_mb": "", "note": "",
    }])
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    config = tmp_path / "config"
    config.write_text("audio:\n  sample_rate: 44100\ntraining:\n  instruments: [drums, bass, other, vocals]\n",
                      encoding="utf-8")
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", config))
    monkeypatch.setattr(infer, "capture_msst_runtime", lambda: RuntimeRevision(
        provider="msst", device="cuda:0", python_version="3.11", platform="test",
        torch_version="test", cuda_build="test", cudnn_version="test",
        gpu_name="test gpu", gpu_capability="12.0", adapter_artifacts=(("provider.py", "a" * 64),),
        dependency_versions=(("torch", "test"),),
    ))

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 29, tzinfo=timezone.utc)

    if not (resource_root() / "bin" / "ffmpeg.exe").is_file():
        # FLAC streamの完全性検査は同梱のffmpeg.exeで全frameを復号する（Windows配布物だけにある）。
        # ここで確かめるのは和・来歴・識別なので、その検査だけを「完全」とみなす。
        monkeypatch.setattr(flac_frames, "stream_is_exact", lambda path: True)
    # 処理時刻のタグだけを固定し、同じ入力の成果物をbytesで比べられるようにする。
    monkeypatch.setattr(writer, "datetime", _Frozen)


def _source(tmp_path) -> Path:
    path = tmp_path / "source.wav"
    if not path.exists():
        t = np.arange(44100, dtype=np.float64) / 44100
        left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 3100 * t)
        right = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.1 * np.sign(np.sin(2 * np.pi * 4 * t))
        sf.write(path, np.stack([left, right], axis=1).astype(np.float32), 44100, subtype="PCM_24")
    return path


def _component(name: str, data: np.ndarray) -> np.ndarray:
    # 部品ごとに別の、決定的な合成信号（入力に依存し、和が1.0を超え得る）。
    rng = np.random.default_rng(FOUR.index(name) + 1)
    noise = rng.uniform(-1.0, 1.0, data.shape)
    return (_GAINS[name] * (0.5 * noise + 0.5 * np.sign(data))).astype(np.float32)


def _separator(calls, skip=(), shorten=()):
    def separate(arch, mix_path, ckpt_path, config_path, stem_target, store_dir, **kwargs):
        calls.append(stem_target)
        kwargs["runtime_observer"](kwargs["runtime_revision"])
        data, sr = sf.read(mix_path, always_2d=True, dtype="float32")
        for name in FOUR:
            if name in skip:
                continue
            values = _component(name, data)
            sf.write(store_dir / f"{name}.wav", values[:-10] if name in shorten else values, sr, subtype="FLOAT")
        return None
    return separate


def _decoded(path: Path) -> tuple[np.ndarray, int]:
    data, _ = sf.read(path, dtype="float64", always_2d=True)
    tags = writer.read_source_tags(path)
    return data, int(tags["LSS_OUTPUT_HEADROOM_SHIFT"])


def test_combination_publishes_the_exact_sum_with_provenance(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(infer, "msst_separate", _separator(calls))
    source = _source(tmp_path)

    result = worker.handle_job({"mix_path": str(source), "model_id": "four", "stem_target": "drums+vocals"})

    assert result["status"] == "ok", result
    assert [Path(p).name for p in result["outputs"]] == ["source - drums + vocals.flac"]
    assert result["recipe_id"].startswith("stem-combination-v1-")
    assert calls == ["drums"]  # 1回の分離passだけ
    published = Path(result["outputs"][0])
    data, shift = _decoded(published)
    mix, _ = sf.read(source, always_2d=True, dtype="float32")
    exact = (_component("drums", mix).astype(np.float64)
             + _component("vocals", mix).astype(np.float64)).astype(np.float32).astype(np.float64)
    assert np.max(np.abs(exact)) > 1.0 and shift >= 1  # 和は1.0を超え、切らずにheadroomで収める
    bits = {"PCM_16": 16, "PCM_24": 24}[sf.info(published).subtype]
    # 公開値 = 和 / 2**shift を整数PCMへ量子化したもの。差は量子化の半ステップ以内。
    assert np.max(np.abs(data * 2.0 ** shift - exact)) <= 2.0 ** shift * 2.0 ** -(bits - 1)
    tags = writer.read_source_tags(published)
    assert tags["LSS_STEM"] == "drums + vocals"
    assert tags["LSS_RECIPE_ID"] == result["recipe_id"]
    receipt = json.loads(tags["LSS_STEM_COMBINATION"])
    assert receipt["components"] == ["drums", "vocals"] and receipt["gains"] == [1.0, 1.0]
    assert receipt["route"] == "sum_of_stems" and set(receipt["component_sha256"]) == {"drums", "vocals"}
    assert result["stem_combination"]["components"] == ["drums", "vocals"]
    revision = json.loads(tags["LSS_RECIPE_REVISION"])
    assert revision["capability"] == "stem_combination"
    assert revision["stage_graph"][1]["components"] == ["drums", "vocals"]

    # 同じ要求（順序・大小文字違い）は同じRecipeで、完了済みを再利用する。
    again = worker.handle_job({"mix_path": str(source), "model_id": "four", "stem_target": "Vocals+DRUMS"})
    assert again["status"] == "skipped" and again["recipe_id"] == result["recipe_id"]
    assert again["outputs"] == result["outputs"] and calls == ["drums"]


def test_combination_has_its_own_identity_and_leaves_single_stem_bytes_unchanged(tmp_path, monkeypatch):
    calls = []
    # A: 単一ステムだけ。
    (tmp_path / "a").mkdir()
    _setup(tmp_path / "a", monkeypatch)
    monkeypatch.setattr(infer, "msst_separate", _separator(calls))
    source_a = _source(tmp_path / "a")
    single_a = worker.handle_job({"mix_path": str(source_a), "model_id": "four", "stem_target": "vocals"})
    # B: 先に組み合わせ、その後に同じ単一ステム。
    (tmp_path / "b").mkdir()
    _setup(tmp_path / "b", monkeypatch)
    monkeypatch.setattr(infer, "msst_separate", _separator(calls))
    source_b = _source(tmp_path / "b")
    combined = worker.handle_job({"mix_path": str(source_b), "model_id": "four", "stem_target": "drums+vocals"})
    single_b = worker.handle_job({"mix_path": str(source_b), "model_id": "four", "stem_target": "vocals"})

    assert single_a["status"] == single_b["status"] == combined["status"] == "ok"
    assert single_a["recipe_id"] == single_b["recipe_id"]
    assert single_a["recipe_id"].startswith("separation-v1-")
    assert combined["recipe_id"] not in {single_a["recipe_id"], single_b["recipe_id"]}
    assert [Path(p).name for p in single_b["outputs"]] == ["source - vocals.flac"]
    assert Path(single_a["outputs"][0]).read_bytes() == Path(single_b["outputs"][0]).read_bytes()
    assert "LSS_STEM_COMBINATION" not in writer.read_source_tags(Path(single_b["outputs"][0]))
    # 組み合わせと単一ステムは別の完了記録（それぞれの系譜）を持つ。
    combined_record = session.check_completed_outputs(combined["input_sha256"], combined["recipe_id"], source_b)
    single_record = session.check_completed_outputs(single_b["input_sha256"], single_b["recipe_id"], source_b)
    assert [p.name for p, valid in combined_record.items() if valid] == ["source - drums + vocals.flac"]
    assert [p.name for p, valid in single_record.items() if valid] == ["source - vocals.flac"]


@pytest.mark.parametrize("failure", [{"skip": ("drums",)}, {"shorten": ("vocals",)}])
def test_failed_component_publishes_nothing(tmp_path, monkeypatch, failure):
    _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(infer, "msst_separate", _separator(calls, **failure))
    source = _source(tmp_path)

    result = worker.handle_job({"mix_path": str(source), "model_id": "four", "stem_target": "drums+vocals"})

    assert result["status"] == "error"
    assert result["error_code"] == "stem_combination_incomplete"
    assert "outputs" not in result
    assert not [p for p in tmp_path.rglob("*.flac")]
    # 失敗は完了として残らない。部品が揃えば同じ要求が改めて分離して公開する。
    monkeypatch.setattr(infer, "msst_separate", _separator(calls))
    retried = worker.handle_job({"mix_path": str(source), "model_id": "four", "stem_target": "drums+vocals"})
    assert retried["status"] == "ok" and len(calls) == 2


@pytest.mark.parametrize("target,code", [
    ("drums+bass+other+vocals", "stem_combination_unsupported"),   # 原本の全部
    ("piano+vocals", "stem_combination_unsupported"),             # 出力の一部
    ("drums+drums", "stem_combination_unsupported"),              # 異なる部品が2つ未満
])
def test_refused_combination_never_reaches_the_separator(tmp_path, monkeypatch, target, code):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(infer, "msst_separate", lambda *a, **k: pytest.fail("分離しない"))
    result = worker.handle_job({"mix_path": str(_source(tmp_path)), "model_id": "four", "stem_target": target})
    assert result["status"] == "error" and result["error_code"] == code
