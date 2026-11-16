from datetime import datetime
import hashlib

import pytest

from lss.core import models_store
from lss.harness import fetch
from lss.harness.tsv_io import load_tsv, save_tsv

HEADER = ["model_id", "状態", "last_used_at", "rtf", "peak_vram_mb", "sha256", "note"]


def _row(
    model_id, 状態="候補", last_used_at="2026-01-01", rtf="", peak_vram_mb="",
    sha256="", note="",
):
    return {
        "model_id": model_id,
        "状態": 状態,
        "last_used_at": last_used_at,
        "rtf": rtf,
        "peak_vram_mb": peak_vram_mb,
        "sha256": sha256,
        "note": note,
    }


def _make_cache_file(cache_dir, model_id, size_bytes):
    d = cache_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "weight.ckpt").write_bytes(b"0" * size_bytes)


def _setenv(monkeypatch, tmp_path, cap_bytes=None, min_free_bytes=None):
    monkeypatch.setenv("LSS_MODEL_DIR", str(tmp_path / "models"))
    if cap_bytes is not None:
        monkeypatch.setenv("LSS_MODEL_CACHE_CAP_BYTES", str(cap_bytes))
    if min_free_bytes is not None:
        monkeypatch.setenv("LSS_MIN_FREE_BYTES", str(min_free_bytes))


def test_evict_lru_once_deletes_oldest_by_last_used_at(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    cache_dir = models_store.cache_dir()
    _make_cache_file(cache_dir, "model_a", 100)
    _make_cache_file(cache_dir, "model_b", 100)
    header = HEADER
    rows = [_row("model_a", last_used_at="2026-01-01"), _row("model_b", last_used_at="2026-02-01")]

    evicted = models_store.evict_lru_once(header, rows)

    assert evicted["model_id"] == "model_a"
    assert not (cache_dir / "model_a").exists()
    assert (cache_dir / "model_b").exists()


def test_ensure_cache_headroom_evicts_until_cap_satisfied(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=0)
    cache_dir = models_store.cache_dir()
    _make_cache_file(cache_dir, "model_a", 100)
    _make_cache_file(cache_dir, "model_b", 100)
    header = HEADER
    rows = [_row("model_a", last_used_at="2026-01-01"), _row("model_b", last_used_at="2026-02-01")]
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, header, rows)

    models_store.ensure_cache_headroom(0, models_tsv_path=models_tsv_path)

    assert not (cache_dir / "model_a").exists()
    assert (cache_dir / "model_b").exists()
    assert models_store.cache_size_bytes() <= 150


def test_zero_byte_cache_hit_does_not_require_download_reserve(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=200)
    _make_cache_file(models_store.cache_dir(), "requested", 100)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("requested")])
    monkeypatch.setattr(models_store, "free_disk_bytes", lambda: 100)

    models_store.ensure_cache_headroom(
        0, ledger, protected_model_ids=frozenset({"requested"}),
    )

    assert (models_store.cache_dir() / "requested" / "weight.ckpt").is_file()


def test_new_cache_bytes_still_require_download_reserve(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=100)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [])
    monkeypatch.setattr(models_store, "free_disk_bytes", lambda: 100)

    with pytest.raises(models_store.InsufficientCacheSpaceError):
        models_store.ensure_cache_headroom(1, ledger)


def test_ensure_cache_headroom_raises_when_nothing_left_to_evict(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=0, min_free_bytes=0)
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, HEADER, [])

    try:
        models_store.ensure_cache_headroom(100, models_tsv_path=models_tsv_path)
        assert False, "InsufficientCacheSpaceErrorが送出されるはず"
    except models_store.InsufficientCacheSpaceError as exc:
        assert "100" in str(exc)


def test_evicted_row_stays_in_ledger(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=50, min_free_bytes=0)
    cache_dir = models_store.cache_dir()
    _make_cache_file(cache_dir, "model_a", 100)
    header = HEADER
    rows = [_row("model_a", last_used_at="2026-01-01")]
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, header, rows)

    models_store.ensure_cache_headroom(0, models_tsv_path=models_tsv_path)

    _, reloaded_rows = load_tsv(models_tsv_path)
    assert len(reloaded_rows) == 1
    assert reloaded_rows[0]["model_id"] == "model_a"


def test_touch_last_used_records_time_precise_enough_for_same_day_lru(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, HEADER, [_row("model_x")])

    models_store.touch_last_used("model_x", models_tsv_path=models_tsv_path)

    value = load_tsv(models_tsv_path)[1][0]["last_used_at"]
    recorded = datetime.fromisoformat(value)
    assert "T" in value
    assert recorded.tzinfo is not None


def test_same_day_lru_evicts_earlier_timestamp(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    cache_dir = models_store.cache_dir()
    _make_cache_file(cache_dir, "model_a", 100)
    _make_cache_file(cache_dir, "model_b", 100)
    rows = [
        _row("model_a", last_used_at="2026-09-06T12:00:00+00:00"),
        _row("model_b", last_used_at="2026-09-06T08:00:00+00:00"),
    ]

    evicted = models_store.evict_lru_once(HEADER, rows)

    assert evicted["model_id"] == "model_b"
    assert (cache_dir / "model_a").exists()
    assert not (cache_dir / "model_b").exists()


def test_record_recipe_metrics_records_rtf_without_excluding_quality_candidate(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, HEADER, [_row("model_x")])

    excluded = models_store.record_recipe_metrics(
        "model_x", rtf=3.5, peak_vram_mb=100.0, models_tsv_path=models_tsv_path
    )

    assert excluded is False
    _, rows = load_tsv(models_tsv_path)
    assert rows[0]["状態"] == "候補"
    assert rows[0]["rtf"] == "3.5"


def test_record_recipe_metrics_records_vram_without_excluding_quality_candidate(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, HEADER, [_row("model_x")])

    excluded = models_store.record_recipe_metrics(
        "model_x", rtf=0.5, peak_vram_mb=13000.0, models_tsv_path=models_tsv_path
    )

    assert excluded is False
    _, rows = load_tsv(models_tsv_path)
    assert rows[0]["状態"] == "候補"
    assert rows[0]["peak_vram_mb"] == "13000.0"


def test_record_recipe_metrics_keeps_state_when_within_limits(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, HEADER, [_row("model_x")])

    excluded = models_store.record_recipe_metrics(
        "model_x", rtf=0.5, peak_vram_mb=100.0, models_tsv_path=models_tsv_path
    )

    assert excluded is False
    _, rows = load_tsv(models_tsv_path)
    assert rows[0]["状態"] == "候補"


def test_short_runtime_sample_does_not_exclude_candidate(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, HEADER, [_row("model_x")])

    excluded = models_store.record_recipe_metrics(
        "model_x", rtf=9.0, peak_vram_mb=100.0, models_tsv_path=models_tsv_path,
        audio_seconds=models_store.MIN_RUNTIME_SAMPLE_AUDIO_SECONDS - 0.1,
    )

    assert excluded is False
    _, rows = load_tsv(models_tsv_path)
    assert rows[0]["状態"] == "候補"
    assert rows[0]["rtf"] == "9.0"


def test_adopted_default_is_not_removed_by_one_runtime_outlier(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    models_tsv_path = tmp_path / "models.tsv"
    save_tsv(models_tsv_path, HEADER, [_row("model_x", 状態="既定")])

    excluded = models_store.record_recipe_metrics(
        "model_x", rtf=9.0, peak_vram_mb=13000.0, models_tsv_path=models_tsv_path,
        audio_seconds=300.0,
    )

    assert excluded is False
    assert load_tsv(models_tsv_path)[1][0]["状態"] == "既定"


def test_cached_model_is_not_counted_twice_or_downloaded_again(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=0)
    root = models_store.cache_dir()
    _make_cache_file(root, "requested", 100)
    config = root / "requested" / "config.yaml"
    config.write_bytes(b"training: {}")
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("requested")])
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("取得済みなので通信しない"))

    ckpt, actual_config = models_store.ensure_model_cached(
        "requested", "https://example.com/weight.ckpt", "https://example.com/config.yaml", 100, ledger,
    )
    assert ckpt.read_bytes() == b"0" * 100
    assert actual_config == config
    assert load_tsv(ledger)[1][0]["last_used_at"] != "2026-01-01"


def test_model_without_registered_size_is_rejected_before_network_access(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=0)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("requested")])
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("通信してはいけない"))

    with pytest.raises(fetch.InvalidModelSourceError, match="予定サイズ"):
        models_store.ensure_model_cached(
            "requested", "https://example.com/weight.ckpt",
            "https://example.com/config.yaml", None, ledger,
        )


def test_partial_cache_keeps_requested_weight_while_fetching_config(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=0)
    root = models_store.cache_dir()
    _make_cache_file(root, "requested", 100)
    _make_cache_file(root, "other", 100)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("requested"), _row("other", last_used_at="2026-02-01")])
    real_download = fetch.download

    def download(url, dest, **kwargs):
        if dest.name == "config.yaml":
            assert (root / "requested" / "weight.ckpt").exists()
            dest.write_bytes(b"training: {}")
            return dest
        return real_download(url, dest, **kwargs)

    monkeypatch.setattr(fetch, "download", download)
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: pytest.fail("重みは再取得しない"))
    models_store.ensure_model_cached("requested", "https://example.com/weight.ckpt", "https://example.com/config.yaml", 100, ledger)
    assert not (root / "other").exists()
    assert (root / "requested" / "weight.ckpt").stat().st_size == 100


def test_oversized_download_preserves_existing_cache(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=0)
    _make_cache_file(models_store.cache_dir(), "other", 100)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("other")])
    with pytest.raises(models_store.InsufficientCacheSpaceError):
        models_store.ensure_cache_headroom(200, ledger)
    assert (models_store.cache_dir() / "other" / "weight.ckpt").exists()


def _free_disk_that_grows_with_eviction(monkeypatch, free_with_full_cache):
    initial = models_store.cache_size_bytes()
    monkeypatch.setattr(
        models_store, "free_disk_bytes",
        lambda: free_with_full_cache + initial - models_store.cache_size_bytes(),
    )


def _two_cached_models(tmp_path):
    _make_cache_file(models_store.cache_dir(), "model_a", 100)
    _make_cache_file(models_store.cache_dir(), "model_b", 100)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("model_a", last_used_at="2026-01-01"), _row("model_b", last_used_at="2026-02-01")])
    return ledger


def test_unreachable_free_floor_keeps_every_cached_model(tmp_path, monkeypatch):
    # 2件とも退避しても 50 + 200 - 10 = 240 < 400 で下限に届かない。
    _setenv(monkeypatch, tmp_path, cap_bytes=1000, min_free_bytes=400)
    ledger = _two_cached_models(tmp_path)
    _free_disk_that_grows_with_eviction(monkeypatch, 50)

    with pytest.raises(models_store.InsufficientCacheSpaceError, match="evictable_bytes=200"):
        models_store.ensure_cache_headroom(10, ledger)

    assert (models_store.cache_dir() / "model_a" / "weight.ckpt").is_file()
    assert (models_store.cache_dir() / "model_b" / "weight.ckpt").is_file()


def test_unreachable_cap_keeps_unprotected_cached_models(tmp_path, monkeypatch):
    # 保護モデル(100)と取得予定(60)だけで上限150を超え、何を退避しても満たせない。
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=0)
    ledger = _two_cached_models(tmp_path)

    with pytest.raises(models_store.InsufficientCacheSpaceError):
        models_store.ensure_cache_headroom(60, ledger, protected_model_ids=frozenset({"model_b"}))

    assert (models_store.cache_dir() / "model_a" / "weight.ckpt").is_file()


def test_reachable_free_floor_evicts_only_what_it_needs(tmp_path, monkeypatch):
    # 1件退避で 250 + 100 - 10 = 340 >= 300。新しい方は残る。
    _setenv(monkeypatch, tmp_path, cap_bytes=1000, min_free_bytes=300)
    ledger = _two_cached_models(tmp_path)
    _free_disk_that_grows_with_eviction(monkeypatch, 250)

    models_store.ensure_cache_headroom(10, ledger)

    assert not (models_store.cache_dir() / "model_a").exists()
    assert (models_store.cache_dir() / "model_b" / "weight.ckpt").is_file()


def test_eviction_error_is_reported_instead_of_retrying_forever(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=50, min_free_bytes=0)
    _make_cache_file(models_store.cache_dir(), "other", 100)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("other")])

    def fail_delete(*args, **kwargs):
        assert not kwargs.get("ignore_errors", False)
        raise PermissionError("使用中")

    monkeypatch.setattr(models_store.shutil, "rmtree", fail_delete)
    with pytest.raises(PermissionError, match="使用中"):
        models_store.ensure_cache_headroom(0, ledger)


def test_eviction_never_follows_model_path_outside_cache(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    models_store.cache_dir().mkdir()
    with pytest.raises(ValueError, match="直下ではありません"):
        models_store.evict_lru_once(HEADER, [_row("../outside")])
    assert outside.exists()


def test_frozen_models_ledger_is_initialized_once_in_writable_app_root(tmp_path, monkeypatch):
    resources = tmp_path / "internal"
    app = tmp_path / "app"
    bundled = resources / "ledger" / "models.tsv"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("model_id\t状態\nmodel-a\t既定\n", encoding="utf-8")
    monkeypatch.setattr(models_store, "is_frozen", lambda: True)
    monkeypatch.setattr(models_store, "resource_root", lambda: resources)
    monkeypatch.setattr(models_store, "app_root", lambda: app)

    runtime = models_store.models_tsv_path()

    assert runtime == app / "ledger" / "models.tsv"
    assert runtime.read_text(encoding="utf-8") == bundled.read_text(encoding="utf-8")
    save_tsv(
        runtime,
        ["model_id", "状態", "rtf", "peak_vram_mb", "last_used_at", "sha256"],
        [{
            "model_id": "model-a", "状態": "既定", "rtf": "0.25",
            "peak_vram_mb": "2000", "last_used_at": "2026-09-06T00:00:00+00:00",
            "sha256": "a" * 64,
        }],
    )
    bundled.write_text(
        "model_id\t状態\tnote\nmodel-a\t既定\tupdated definition\nmodel-b\t候補\tnew model\n",
        encoding="utf-8",
    )

    merged_path = models_store.models_tsv_path()
    header, rows = load_tsv(merged_path)

    assert "note" in header
    assert [row["model_id"] for row in rows] == ["model-a", "model-b"]
    assert rows[0]["note"] == "updated definition"
    assert rows[0]["rtf"] == "0.25"
    assert rows[0]["peak_vram_mb"] == "2000"
    assert rows[0]["last_used_at"] == "2026-09-06T00:00:00+00:00"
    assert rows[0]["sha256"] == "a" * 64


def test_obsolete_runtime_auto_exclusion_does_not_override_bundle_candidate(tmp_path):
    bundled = tmp_path / "bundled.tsv"
    runtime = tmp_path / "runtime.tsv"
    header = ["model_id", "状態", "rtf", "note"]
    save_tsv(bundled, header, [
        {"model_id": "candidate", "状態": "候補", "rtf": "", "note": "new definition"},
    ])
    save_tsv(runtime, header, [
        {"model_id": "candidate", "状態": "除外", "rtf": "4.0", "note": "T320自動除外: slow"},
        {"model_id": "retired", "状態": "既定", "rtf": "0.2", "note": "old definition"},
    ])

    models_store._merge_runtime_ledger(bundled, runtime)
    _, rows = load_tsv(runtime)

    assert rows[0]["状態"] == "候補"
    assert rows[0]["rtf"] == "4.0"
    assert rows[0]["note"] == "new definition"
    assert rows[1]["model_id"] == "retired"
    assert rows[1]["状態"] == "保留"


def test_first_model_use_records_actual_sha256_in_ledger(tmp_path, monkeypatch):
    _setenv(monkeypatch, tmp_path, cap_bytes=150, min_free_bytes=0)
    root = models_store.cache_dir()
    _make_cache_file(root, "requested", 100)
    (root / "requested" / "config.yaml").write_bytes(b"training: {}")
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("requested")])

    ckpt, _ = models_store.ensure_model_cached(
        "requested", "https://example.com/weight.ckpt", "https://example.com/config.yaml", 100, ledger,
    )

    expected = hashlib.sha256(b"0" * 100).hexdigest()
    assert models_store.model_sha256("requested", ckpt, ledger) == expected
    assert load_tsv(ledger)[1][0]["sha256"] == expected


def test_corrupt_weight_can_self_repair_without_double_counting_cache_space(tmp_path, monkeypatch):
    valid_weight = b"new weight"
    config = b"training: {}"
    expected_sha256 = hashlib.sha256(valid_weight).hexdigest()
    _setenv(
        monkeypatch, tmp_path,
        cap_bytes=len(valid_weight) + len(config), min_free_bytes=0,
    )
    root = models_store.cache_dir()
    model_dir = root / "requested"
    model_dir.mkdir(parents=True)
    (model_dir / "weight.ckpt").write_bytes(b"bad weight")
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("requested", sha256=expected_sha256)])

    def download(url, dest, **kwargs):
        dest.write_bytes(config if dest.suffix == ".yaml" else valid_weight)
        return dest

    monkeypatch.setattr(fetch, "download", download)

    ckpt, actual_config = models_store.ensure_model_cached(
        "requested", "https://example.com/weight.ckpt", "https://example.com/config.yaml",
        len(valid_weight), ledger,
    )

    assert ckpt.read_bytes() == valid_weight
    assert actual_config.read_bytes() == config
    assert models_store.cache_size_bytes() == len(valid_weight) + len(config)


def test_model_download_progress_identifies_weight_and_config(tmp_path, monkeypatch):
    weight = b"weight"
    config = b"training: {}"
    _setenv(monkeypatch, tmp_path, cap_bytes=100, min_free_bytes=0)
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, HEADER, [_row("requested", sha256=hashlib.sha256(weight).hexdigest())])

    def download(url, dest, **kwargs):
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        data = config if dest.suffix == ".yaml" else weight
        callback = kwargs.get("progress")
        if callback is not None:
            callback(0, len(data))
            callback(len(data), len(data))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    monkeypatch.setattr(fetch, "download", download)
    events = []

    models_store.ensure_model_cached(
        "requested", "https://example.com/weight.ckpt", "https://example.com/config.yaml",
        len(weight), ledger, progress=events.append,
    )

    assert [(event["asset"], event["completed_bytes"], event["total_bytes"]) for event in events] == [
        ("weight", 0, len(weight)),
        ("weight", len(weight), len(weight)),
        ("config", 0, len(config)),
        ("config", len(config), len(config)),
    ]


@pytest.mark.parametrize(("header", "rows", "message"), [
    (["状態"], [{"状態": "候補"}], "model_id列"),
    (HEADER, [_row("same"), _row("same")], "重複"),
    (HEADER, [_row("model", 状態="謎")], "状態が不正"),
    (HEADER, [_row("CON")], "model_idが不正"),
    (HEADER + ["size_bytes"], [{**_row("model"), "size_bytes": "large"}], "size_bytes"),
    (HEADER, [_row("model", sha256="not-a-hash")], "SHA-256"),
])
def test_invalid_model_ledger_is_rejected(header, rows, message, tmp_path):
    ledger = tmp_path / "models.tsv"
    save_tsv(ledger, header, rows)

    with pytest.raises(models_store.InvalidModelLedgerError, match=message):
        models_store.load_model_ledger(ledger)


@pytest.mark.parametrize("field", ["distributor_url", "config_url"])
def test_available_model_requires_safe_https_sources(tmp_path, field):
    ledger = tmp_path / "models.tsv"
    header = HEADER + ["distributor_url", "config_url"]
    row = {
        **_row("model"),
        "distributor_url": "https://example.com/weight.ckpt",
        "config_url": "https://example.com/config.yaml",
    }
    row[field] = "http://unsafe.example.com/asset"
    save_tsv(ledger, header, [row])

    with pytest.raises(models_store.InvalidModelLedgerError, match=field):
        models_store.load_model_ledger(ledger)


def test_the_development_tree_keeps_runtime_values_out_of_the_tracked_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("LSS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(models_store, "is_frozen", lambda: False)
    bundled = models_store.resource_root() / "ledger" / "models.tsv"
    before = bundled.read_bytes()
    runtime = models_store.models_tsv_path()
    assert runtime == tmp_path / "state" / "models.tsv"
    _, rows = load_tsv(runtime)
    model_id = rows[0]["model_id"]
    models_store.touch_last_used(model_id, models_tsv_path=runtime)
    assert bundled.read_bytes() == before
    # 次の起動でも実測値は残り、同梱台帳の更新は取り込まれる。
    again = models_store.models_tsv_path()
    assert next(row for row in load_tsv(again)[1] if row["model_id"] == model_id)["last_used_at"]
