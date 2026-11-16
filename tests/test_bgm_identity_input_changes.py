"""照合中の変更で古い一致を確定せず、以前の記録と未解決候補を守る。"""
import json

import numpy as np
import pytest

from lss.bgm import identity, identity_run, series_identity
from tests.test_bgm_identity_evidence_reuse import annotated_episode


@pytest.mark.parametrize("change", ["run", "signal", "reference", "settings", "activity", "missing_signal"])
def test_episode_changes_during_matching_preserve_previous_annotation(
        annotated_episode, tmp_path, monkeypatch, change):
    run_path, record_path, signal = annotated_episode
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference recording")
    run = json.loads(run_path.read_text("utf-8"))
    activity = tmp_path / "activity.json"
    activity.write_text('{"candidates": []}', encoding="utf-8")
    run.update(type="episode_bgm_discovery_run", status=identity_run.EPISODE_STATUS,
               activity_runs=[{"recipe_id": "activity", "record_path": str(activity),
                               "record_sha256": identity._sha256(activity)}])
    run_path.write_text(json.dumps(run), encoding="utf-8")
    before = record_path.read_bytes()
    monkeypatch.setattr(identity.ReferenceIndex, "build", lambda *a, **k: None)

    def match(*a, **k):
        if change == "settings":
            monkeypatch.setattr(identity, "MIN_VOTES", identity.MIN_VOTES + 1)
        elif change == "missing_signal":
            # 照合は番組の中の再演の分も呼ばれる。二度目の呼出で消し直さない。
            signal.unlink(missing_ok=True)
        else:
            path = {"run": run_path, "signal": signal, "reference": reference, "activity": activity}[change]
            path.write_bytes(path.read_bytes() + b"\n")
        return []

    monkeypatch.setattr(identity, "match", match)
    with pytest.raises(identity_run.IdentityRunError):
        identity_run.annotate_run(run_path, [reference], loader=lambda p: np.zeros(100))
    assert record_path.read_bytes() == before
    assert not record_path.with_name(f".{identity_run.RECORD_NAME}.tmp").exists()


@pytest.mark.parametrize("change", ["run", "signal", "annotation", "settings"])
def test_series_changes_during_matching_preserve_previous_record_and_pair_cache(
        annotated_episode, tmp_path, monkeypatch, change):
    run_path, record_path, signal = annotated_episode
    other = tmp_path / "other"
    other.mkdir()
    second = other / "run.json"
    run = json.loads(run_path.read_text("utf-8"))
    run["source"]["source_id"] = "other"
    second.write_text(json.dumps(run), encoding="utf-8")
    out = tmp_path / "series"
    out.mkdir()
    previous = out / series_identity.RECORD_NAME
    previous.write_bytes(b"previous valid series record")
    cache = tmp_path / "cache"
    monkeypatch.setattr(series_identity, "_chunk_fingerprints", lambda *a, **k: [])

    def match(*a, **k):
        if change == "settings":
            monkeypatch.setattr(identity, "MIN_VOTES", identity.MIN_VOTES + 1)
        else:
            path = {"run": run_path, "signal": signal, "annotation": record_path}[change]
            path.write_bytes(path.read_bytes() + b"\n")
        return []

    monkeypatch.setattr(identity, "match", match)
    with pytest.raises(series_identity.SeriesIdentityError):
        series_identity.annotate_series([run_path, second], out, cache_dir=cache,
                                        loader=lambda p: np.zeros(100))
    assert previous.read_bytes() == b"previous valid series record"
    assert not list(cache.glob("shared-*.json"))
    assert not previous.with_name(f".{series_identity.RECORD_NAME}.tmp").exists()
