import pytest
import yaml

from lss.core import catalog, models_store
from lss.core.jobs import SeparationQueue
from lss.harness import fetch, tsv_io


def _catalog(tmp_path, monkeypatch, states=("候補", "既定", "除外", "保留")):
    rows = [{"model_id": f"model{i}", "状態": state, "config_url": "https://example.com/config.yaml"}
            for i, state in enumerate(states)]
    ledger = tmp_path / "models.tsv"
    tsv_io.save_tsv(ledger, ["model_id", "状態", "config_url"], rows)
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    monkeypatch.setenv("LSS_MODEL_DIR", str(tmp_path / "models"))
    return rows


def test_default_is_first_and_unavailable_models_are_hidden(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    assert [row["model_id"] for row in catalog.list_models()] == ["model1", "model0"]
    assert catalog.default_model()["model_id"] == "model1"
    with pytest.raises(ValueError, match="利用可能"):
        catalog.get_model("model2")


@pytest.mark.parametrize("states", [("候補",), ("既定", "既定")])
def test_missing_or_ambiguous_default_is_not_guessed(tmp_path, monkeypatch, states):
    _catalog(tmp_path, monkeypatch, states)
    with pytest.raises(ValueError, match="既定モデル"):
        catalog.default_model()


def test_queue_uses_default_and_cached_config_without_weight_download(tmp_path, monkeypatch):
    _catalog(tmp_path, monkeypatch)
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("training:\n  instruments: [Vocals, Instrumental]\n", encoding="utf-8")
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("取得済み設定は通信しない"))
    calls = []
    queue = SeparationQueue(
        lambda job: calls.append(job) or {"status": "ok", "outputs": ["result.flac"]},
        state_path=tmp_path / "queue.json",
    )
    queue.add([tmp_path / "input.wav"])
    queue.run()
    assert calls[0]["model_id"] == "model1"
    # 要求するのはモデルの先頭パートではなく、製品が満たす意味的な結果。
    assert calls[0]["stem_target"] == "Instrumental"
    assert list(config.parent.iterdir()) == [config]


def test_requested_stem_is_the_semantic_result_not_the_first_model_output(tmp_path, monkeypatch):
    row = _catalog(tmp_path, monkeypatch)[1]
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        'training:\n  instruments: [Vocals, Instrumental]\n', encoding="utf-8",
    )
    assert catalog.stem_targets(row)[0] == "Vocals"
    assert catalog.requested_stem(row) == "Instrumental"


def test_requested_stem_is_refused_when_the_model_cannot_produce_it(tmp_path, monkeypatch):
    row = _catalog(tmp_path, monkeypatch)[1]
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text('training:\n  target_instrument: drums\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Instrumental"):
        catalog.requested_stem(row)


@pytest.mark.parametrize("training", ["instruments: []", "instruments: Vocals", "instruments: [null]", "target_instrument: 3"])
def test_invalid_config_does_not_invent_stems(tmp_path, monkeypatch, training):
    row = _catalog(tmp_path, monkeypatch)[1]
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(f"training:\n  {training}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="出力パート"):
        catalog.stem_targets(row)


def test_tuple_values_in_model_config_do_not_prevent_stem_lookup(tmp_path, monkeypatch):
    row = _catalog(tmp_path, monkeypatch)[1]
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "model:\n  resolutions: !!python/tuple [4096, 2048]\n"
        "training:\n  instruments: [Vocals, Instrumental]\n",
        encoding="utf-8",
    )
    assert catalog.stem_targets(row) == ["Vocals", "Instrumental"]


def test_corrupt_cached_config_is_redownloaded_once(tmp_path, monkeypatch):
    row = _catalog(tmp_path, monkeypatch)[1]
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("training: [", encoding="utf-8")
    calls = []

    def download(url, dest, **kwargs):
        calls.append(url)
        if not dest.exists():
            dest.write_text(
                "training:\n  instruments: [Vocals, Instrumental]\n",
                encoding="utf-8",
            )
        return dest

    monkeypatch.setattr(fetch, "download", download)

    assert catalog.stem_targets(row) == ["Vocals", "Instrumental"]
    assert len(calls) == 2


def test_invalid_new_config_is_removed_without_retry_loop(tmp_path, monkeypatch):
    row = _catalog(tmp_path, monkeypatch)[1]
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    calls = []

    def download(url, dest, **kwargs):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("training: [", encoding="utf-8")
        return dest

    monkeypatch.setattr(fetch, "download", download)

    with pytest.raises(yaml.parser.ParserError):
        catalog.stem_targets(row)
    assert calls == [row["config_url"]]
    assert not config.exists()


def test_model_config_rejects_python_object_tags(tmp_path, monkeypatch):
    row = _catalog(tmp_path, monkeypatch)[1]
    config = fetch.model_cache_dir() / "model1" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("training: !!python/name:os.system ''\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        catalog.stem_targets(row)
    assert not config.exists()
