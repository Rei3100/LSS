"""古い一致の付加記録で候補を既知曲扱いせず、再利用前にも記録と信号を確認する。"""
import json
from pathlib import Path

import pytest

from lss.bgm import identity, identity_run, series_identity
from tests.test_bgm_identity_run import _mono, completed_run
from tests.test_bgm_series_identity import two_runs


@pytest.fixture
def annotated_episode(tmp_path):
    signal = tmp_path / "signal.wav"
    signal.write_bytes(b"verified separated signal")
    source = tmp_path / "source.wav"
    source.write_bytes(b"verified original source")
    reference = tmp_path / "annotated-reference.wav"
    reference.write_bytes(b"verified reference recording")
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({
        "status": identity_run.ROUTED_STATUS,
        "source": {"source_id": "episode", "path": str(source), "sha256": identity._sha256(source)},
        "rendering": {"signal": {"path": str(signal), "sha256": identity._sha256(signal)}},
        "candidates": [],
    }), encoding="utf-8")
    record_path = tmp_path / identity_run.RECORD_NAME
    record_path.write_text(json.dumps({
        "type": "episode_identity_annotation", "schema_version": identity_run.RECORD_SCHEMA,
        "intent": {**identity_run.evidence_settings(), "run_sha256": identity._sha256(run_path),
                   "references": [{"path": str(reference), "sha256": identity._sha256(reference)}]},
        "query_signal": {"kind": "routed_signal", "sha256": identity._sha256(signal)},
        "matches": [],
    }), encoding="utf-8")
    return run_path, record_path, signal


@pytest.mark.parametrize("change", ["threshold", "matching_code", "annotation_code", "missing_revision",
                                   "signal_hash", "signal_kind", "record_type", "schema"])
def test_series_ignores_annotation_from_other_conditions(annotated_episode, tmp_path, monkeypatch, change):
    run_path, record_path, _ = annotated_episode
    assert series_identity._episode(run_path)["identity_sha256"] == identity._sha256(record_path)
    record = json.loads(record_path.read_text("utf-8"))
    if change == "threshold":
        monkeypatch.setattr(identity, "MIN_VOTES", identity.MIN_VOTES + 1)
    elif change in {"matching_code", "annotation_code"}:
        module = identity if change == "matching_code" else identity_run
        changed_code = tmp_path / f"changed_{Path(module.__file__).name}"
        changed_code.write_bytes(Path(module.__file__).read_bytes() + b"\n# changed evidence\n")
        monkeypatch.setattr(module, "__file__", str(changed_code))
    elif change == "missing_revision":
        del record["intent"]["matching_settings_id"]
    elif change == "signal_hash":
        record["query_signal"]["sha256"] = "old signal"
    elif change == "signal_kind":
        record["query_signal"]["kind"] = "source"
    elif change == "record_type":
        record["type"] = "other_annotation"
    else:
        record["schema_version"] += 1
    record_path.write_text(json.dumps(record), encoding="utf-8")
    before = record_path.read_bytes()
    episode = series_identity._episode(run_path)
    assert episode["identity_sha256"] is None and episode["matches"] == []
    assert record_path.read_bytes() == before


@pytest.mark.parametrize("bad_record", ["{incomplete", "[]", '{"intent": null}'])
def test_broken_optional_annotation_preserves_episode(annotated_episode, bad_record):
    run_path, record_path, _ = annotated_episode
    record_path.write_text(bad_record, encoding="utf-8")
    episode = series_identity._episode(run_path)
    assert episode["key"] == "episode" and episode["identity_sha256"] is None


def test_series_ignores_matches_from_a_missing_separation_signal(annotated_episode):
    run_path, record_path, signal = annotated_episode
    before = record_path.read_bytes()
    signal.unlink()
    episode = series_identity._episode(run_path)
    assert episode["signal_kind"] == "source" and episode["identity_sha256"] is None
    assert record_path.read_bytes() == before


def test_series_rebuild_does_not_keep_known_matches_from_a_previous_threshold(two_runs, tmp_path, monkeypatch):
    first, second = two_runs
    monkeypatch.setattr(series_identity, "_shared_pairs", lambda *a, **k: {("ep1", "ep2"): []})
    path = series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono)
    before = {p: p.read_bytes() for p in (first, second, first.parent / identity_run.RECORD_NAME)}
    record = json.loads(path.read_text("utf-8"))
    assert any(c["identity_state"] == "known_exact" for c in record["activity_recipes"][0]["candidates"]["ep1"])

    monkeypatch.setattr(identity, "MIN_VOTES", 10**9)
    series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono)
    rebuilt = json.loads(path.read_text("utf-8"))
    candidates = rebuilt["activity_recipes"][0]["candidates"]["ep1"]
    assert candidates and all(c["identity_state"] == "unresolved" for c in candidates)
    assert all(d["decision"] == "pending" for d in rebuilt["activity_recipes"][0]["novelty_decisions"]["ep1"])
    assert {p: p.read_bytes() for p in before} == before
    stamp = path.stat().st_mtime_ns
    series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono)
    assert path.stat().st_mtime_ns == stamp


def test_cached_episode_rechecks_signal_and_does_not_return_a_missing_signal(completed_run):
    run_path, references = completed_run
    path = identity_run.annotate_run(run_path, references, loader=_mono)
    run = json.loads(run_path.read_text("utf-8"))
    artifact = Path(run["separation"]["artifacts"][0]["path"])
    artifact.unlink()
    identity_run.annotate_run(run_path, references, loader=_mono)
    record = json.loads(path.read_text("utf-8"))
    assert record["query_signal"]["kind"] == "source"
    previous = path.read_bytes()
    Path(run["source"]["path"]).unlink()
    with pytest.raises(identity_run.IdentityRunError):
        identity_run.annotate_run(run_path, references, loader=_mono)
    assert path.read_bytes() == previous


def test_cached_episode_refuses_a_changed_activity_record(completed_run):
    run_path, references = completed_run
    path = identity_run.annotate_run(run_path, references, loader=_mono)
    before = path.read_bytes()
    run = json.loads(run_path.read_text("utf-8"))
    activity = Path(run["activity_runs"][0]["record_path"])
    activity.write_bytes(activity.read_bytes() + b"\n")
    with pytest.raises(identity_run.IdentityRunError):
        identity_run.annotate_run(run_path, references, loader=_mono)
    assert path.read_bytes() == before


def test_cached_series_refuses_a_changed_activity_record(two_runs, tmp_path, monkeypatch):
    first, second = two_runs
    monkeypatch.setattr(series_identity, "_shared_pairs", lambda *a, **k: {("ep1", "ep2"): []})
    path = series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono)
    before = path.read_bytes()
    run = json.loads(first.read_text("utf-8"))
    activity = Path(run["activity_runs"][0]["record_path"])
    activity.write_bytes(activity.read_bytes() + b"\n")
    with pytest.raises(series_identity.SeriesIdentityError):
        series_identity.annotate_series([first, second], tmp_path / "series", loader=_mono)
    assert path.read_bytes() == before
