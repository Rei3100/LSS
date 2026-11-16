"""既定設定の内容固定と、キャッシュ・初回取得の両方での改変拒否。"""
import hashlib

import pytest

from lss.core import catalog, models_store
from lss.harness import fetch
from lss.harness.tsv_io import save_tsv


CONFIG = b"training:\n  instruments: [Vocals, Instrumental]\n"
WEIGHT = b"known weight"


class Response:
    def __init__(self, data, url):
        self.data, self.url = data, url
        self.headers, self.history = {}, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data


def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LSS_MIN_FREE_BYTES", "0")
    row = {
        "model_id": "pinned", "状態": "既定",
        "distributor_url": "https://example.com/weight.ckpt",
        "config_url": "https://example.com/config.yaml",
        "size_bytes": str(len(WEIGHT)), "sha256": hashlib.sha256(WEIGHT).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG).hexdigest(), "last_used_at": "",
    }
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, list(row), [row])
    monkeypatch.setattr(models_store, "MODELS_TSV", ledger)
    weight, config = fetch.model_paths(row["model_id"], row["distributor_url"], row["config_url"])
    weight.parent.mkdir(parents=True)
    return row, ledger, weight, config


def ensure(row, ledger):
    return models_store.ensure_model_cached(
        row["model_id"], row["distributor_url"], row["config_url"],
        len(WEIGHT), models_tsv_path=ledger,
    )


@pytest.mark.parametrize("entry", ["catalog", "worker"])
def test_verified_cached_config_uses_no_network(tmp_path, monkeypatch, entry):
    row, ledger, weight, config = setup(tmp_path, monkeypatch)
    weight.write_bytes(WEIGHT)
    config.write_bytes(CONFIG)
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("固定済み設定は通信しない"))
    if entry == "catalog":
        assert catalog.stem_targets(row) == ["Vocals", "Instrumental"]
    else:
        assert ensure(row, ledger) == (weight, config)


@pytest.mark.parametrize("cached_weight", [False, True])
def test_bad_remote_config_is_not_activated_or_recorded(tmp_path, monkeypatch, cached_weight):
    row, ledger, weight, config = setup(tmp_path, monkeypatch)
    if cached_weight:
        weight.write_bytes(WEIGHT)
        config.write_bytes(b"training:\n  instruments: [drums]\n")
    original_ledger = ledger.read_bytes()
    old_config = config.read_bytes() if config.exists() else None
    monkeypatch.setattr(fetch.requests, "get", lambda url, **k: Response(
        WEIGHT if url == row["distributor_url"] else b"training:\n  instruments: [drums]\n", url,
    ))
    with pytest.raises(ValueError, match="SHA-256"):
        ensure(row, ledger)
    assert ledger.read_bytes() == original_ledger
    assert config.read_bytes() == old_config if old_config is not None else not config.exists()
    assert not list(weight.parent.glob("*.part"))


def test_valid_yaml_with_wrong_hash_is_repaired_before_catalog_parse(tmp_path, monkeypatch):
    row, _, _, config = setup(tmp_path, monkeypatch)
    config.write_bytes(b"training:\n  instruments: [drums]\n")
    calls = []
    monkeypatch.setattr(fetch.requests, "get", lambda url, **k: calls.append(url) or Response(CONFIG, url))
    assert catalog.stem_targets(row) == ["Vocals", "Instrumental"]
    assert calls == [row["config_url"]]
    assert config.read_bytes() == CONFIG


def test_catalog_refuses_valid_yaml_from_changed_remote(tmp_path, monkeypatch):
    row, _, _, config = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(fetch.requests, "get", lambda url, **k: Response(
        b"training:\n  instruments: [Vocals, Instrumental, drums]\n", url,
    ))
    with pytest.raises(ValueError, match="SHA-256"):
        catalog.stem_targets(row)
    assert not config.exists()


@pytest.mark.parametrize("invalid", ["short", "g" * 64])
def test_bad_config_hash_in_ledger_is_rejected_before_fetch(tmp_path, monkeypatch, invalid):
    row, ledger, _, _ = setup(tmp_path, monkeypatch)
    row["config_sha256"] = invalid
    save_tsv(ledger, list(row), [row])
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("壊れた台帳で取得しない"))
    with pytest.raises(models_store.InvalidModelLedgerError, match="設定SHA-256"):
        catalog.default_model()



