"""再登場の記録でも、変わった参照・区間cache・保存競合から古い判断を引き継がない。"""
import json

import pytest

from lss.bgm import identity, series_identity
from tests.test_bgm_identity_run import _mono
from tests.test_bgm_series_identity import two_runs


@pytest.fixture
def prepared(two_runs, tmp_path, monkeypatch):
    first, second = two_runs
    # 参照照合は実物。再登場の証拠だけを固定し、ここでは保存・再利用の境界を検証する。
    match = identity.SharedOccurrence(30.0, 50.0, 140.0, 160.0, 0.9, 10)
    monkeypatch.setattr(series_identity.identity, "match", lambda *args: [])
    episodes = [series_identity._episode(path) for path in (first, second)]
    series_identity._write_pair(tmp_path / "cache", *episodes, [match])
    record = series_identity.annotate_series([first, second], tmp_path / "series",
                                            loader=_mono, cache_dir=tmp_path / "cache")
    return first, second, record


@pytest.mark.parametrize("change", ["changed", "missing"])
def test_changed_reference_drops_known_label_and_preserves_raw_candidates(prepared, tmp_path, change):
    first, second, record_path = prepared
    annotation_path = first.parent / "identity.json"
    annotation_before = annotation_path.read_bytes()
    reference = tmp_path / "known.flac"
    if change == "missing":
        reference.unlink()
    else:
        reference.write_bytes(b"changed reference")

    series_identity.annotate_series([first, second], record_path.parent, loader=_mono,
                                   cache_dir=tmp_path / "cache")
    record = json.loads(record_path.read_bytes())
    candidates = record["activity_recipes"][0]["candidates"]["ep1"]
    assert not any(c["identity_state"] == "known_exact" for c in candidates)
    assert any(c["start_seconds"] <= 120 < c["end_seconds"] for c in candidates)
    assert len(record["shared_recordings"][0]["occurrences"]) == 1
    assert annotation_path.read_bytes() == annotation_before


def test_corrupt_series_body_is_rebuilt_using_valid_pair_cache(prepared, tmp_path, monkeypatch):
    first, second, path = prepared
    record = json.loads(path.read_bytes())
    record["activity_recipes"] = []
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(series_identity.identity, "match", lambda *args: pytest.fail("有効な組cacheを保持"))

    series_identity.annotate_series([first, second], path.parent, loader=_mono, cache_dir=tmp_path / "cache")
    repaired = json.loads(path.read_bytes())
    assert repaired["activity_recipes"] and series_identity._intact(repaired)


@pytest.mark.parametrize("damage", ["checksum", "wrong_signal", "invalid_span", "nonfinite"])
def test_bad_pair_cache_is_recomputed_instead_of_linking_candidates(two_runs, tmp_path, monkeypatch, damage):
    first, second = [series_identity._episode(path) for path in two_runs]
    cache = tmp_path / "cache"
    series_identity._write_pair(cache, first, second, [identity.SharedOccurrence(30, 50, 140, 160, 0.9, 10)])
    path = series_identity._pair_cache_path(cache, first, second)
    payload = json.loads(path.read_bytes())
    if damage == "wrong_signal":
        payload["intent"]["first_signal_sha256"] = "0" * 64
    elif damage == "invalid_span":
        payload["occurrences"][0]["first_end_seconds"] = 29
    elif damage == "nonfinite":
        payload["occurrences"][0]["waveform_explained"] = float("nan")
    else:
        payload["occurrences"] = []
    if damage in {"wrong_signal", "invalid_span"}:
        payload["payload_sha256"] = series_identity._payload_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    monkeypatch.setattr(series_identity.identity, "match", lambda *args: calls.append(True) or [])

    shared = series_identity._shared_pairs([first, second], cache_dir=cache, loader=_mono)
    assert len(calls) == 2 and shared[(first["key"], second["key"])] == []
    assert series_identity._read_pair(cache, first, second) == []


def test_reference_change_during_cache_reuse_preserves_previous_series(prepared, tmp_path, monkeypatch):
    first, second, path = prepared
    before = path.read_bytes()
    load = series_identity._episode
    def load_then_change(run_path):
        result = load(run_path)
        if run_path == second:
            (tmp_path / "known.flac").write_bytes(b"late reference change")
        return result
    monkeypatch.setattr(series_identity, "_episode", load_then_change)

    with pytest.raises(series_identity.SeriesIdentityError):
        series_identity.annotate_series([first, second], path.parent, loader=_mono)
    assert path.read_bytes() == before


def test_reference_change_during_series_replace_preserves_previous_series(prepared, tmp_path, monkeypatch):
    first, second, path = prepared
    before = path.read_bytes()
    # 新しい出力を必要にする。組cacheは再計算せずに使える。
    changed = json.loads(path.read_bytes())
    changed["intent"]["schema_version"] = 0
    path.write_text(json.dumps(changed), encoding="utf-8")
    before = path.read_bytes()
    write = series_identity.identity_run._write_annotation
    def write_then_change(record_path, payload, validate):
        if record_path == path:
            (tmp_path / "known.flac").write_bytes(b"late reference change")
        return write(record_path, payload, validate)
    monkeypatch.setattr(series_identity.identity_run, "_write_annotation", write_then_change)

    with pytest.raises(series_identity.SeriesIdentityError):
        series_identity.annotate_series([first, second], path.parent, loader=_mono, cache_dir=tmp_path / "cache")
    assert path.read_bytes() == before and not list(path.parent.glob(".*.tmp"))


def test_nested_pair_writes_use_distinct_staging_files(two_runs, tmp_path, monkeypatch):
    first, second = [series_identity._episode(path) for path in two_runs]
    cache = tmp_path / "cache"
    found = [identity.SharedOccurrence(30, 50, 140, 160, 0.9, 10)]
    verify = series_identity._verify_episodes
    staging = []
    nested = False
    def verify_with_nested_write(episodes):
        nonlocal nested
        if not nested:
            nested = True
            series_identity._write_pair(cache, first, second, found)
        else:
            staging.extend(cache.glob(".*.tmp"))
        verify(episodes)
    monkeypatch.setattr(series_identity, "_verify_episodes", verify_with_nested_write)

    series_identity._write_pair(cache, first, second, found)
    assert len(staging) == 2 and len(set(staging)) == 2
    assert series_identity._read_pair(cache, first, second) == found and not list(cache.glob(".*.tmp"))
