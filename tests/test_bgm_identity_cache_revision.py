"""録音一致の保存物を、異なる実装・判定条件で作った証拠として再利用しない。"""
import json
from pathlib import Path

import numpy as np
import pytest

from lss.bgm import identity, identity_run, series_identity
from tests.test_bgm_identity_run import _mono, completed_run
from tests.test_bgm_identity_evidence_reuse import annotated_episode


def test_fingerprints_are_recomputed_when_the_extraction_implementation_changes(tmp_path, monkeypatch):
    fingerprint = (np.array([1], np.uint32), np.array([0], np.int32), np.zeros((2, 64), np.float32), 2)
    identity._save_fingerprint(tmp_path, "signal-sha", fingerprint)
    changed_code = tmp_path / "changed_identity.py"
    changed_code.write_bytes(Path(identity.__file__).read_bytes() + b"\n# changed extraction\n")
    monkeypatch.setattr(identity, "__file__", str(changed_code))

    assert identity._load_fingerprint(tmp_path, "signal-sha") is None


@pytest.mark.parametrize("change", ["matching_implementation", "series_implementation", "decision_threshold"])
def test_shared_recording_pairs_are_recomputed_after_code_or_decision_changes(
    tmp_path, monkeypatch, annotated_episode, change,
):
    first = series_identity._episode(annotated_episode[0])
    second = {**first, "key": "second"}
    series_identity._write_pair(tmp_path, first, second, [])
    assert series_identity._read_pair(tmp_path, first, second) == []  # 完了したゼロ件も再利用できる
    if change == "decision_threshold":
        monkeypatch.setattr(identity, "MIN_BLOCK_EXPLAINED", identity.MIN_BLOCK_EXPLAINED + 0.1)
    else:
        module = identity if change == "matching_implementation" else series_identity
        changed_code = tmp_path / f"changed_{Path(module.__file__).name}"
        changed_code.write_bytes(Path(module.__file__).read_bytes() + b"\n# changed matching\n")
        monkeypatch.setattr(module, "__file__", str(changed_code))

    assert series_identity._read_pair(tmp_path, first, second) is None


def test_completed_series_is_rechecked_after_a_decision_threshold_changes(tmp_path, monkeypatch):
    for name in ("first", "second"):
        (tmp_path / name).write_bytes(name.encode("utf-8"))
    episodes = {
        name: {"key": name, "run_path": tmp_path / name, "run": {"source": {}},
               "run_sha256": identity._sha256(tmp_path / name), "signal_path": tmp_path / name,
               "signal_sha256": identity._sha256(tmp_path / name),
               "signal_kind": "separation_artifact", "identity_sha256": None,
               "offset": 0.0, "matches": []}
        for name in ("first", "second")
    }
    monkeypatch.setattr(series_identity, "_episode", lambda path: episodes[path.name])
    monkeypatch.setattr(series_identity.identity_run, "candidate_sets", lambda run: {})
    calls = []

    def shared(*args, **kwargs):
        calls.append(1)
        return {("first", "second"): []}

    monkeypatch.setattr(series_identity, "_shared_pairs", shared)
    paths = [tmp_path / name for name in episodes]
    series_identity.annotate_series(paths, tmp_path / "out")
    series_identity.annotate_series(paths, tmp_path / "out")
    assert len(calls) == 1
    monkeypatch.setattr(identity, "MIN_BLOCK_EXPLAINED", identity.MIN_BLOCK_EXPLAINED + 0.1)
    series_identity.annotate_series(paths, tmp_path / "out")
    assert len(calls) == 2


def test_completed_episode_is_rechecked_when_the_vote_threshold_changes(completed_run, monkeypatch):
    run_path, references = completed_run
    calls = []
    original = identity.match

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(identity, "match", counted)
    path = identity_run.annotate_run(run_path, references, loader=_mono)
    before = json.loads(path.read_text("utf-8"))
    assert any(match["identity_state"] == "known_exact" for match in before["matches"])
    annotated = len(calls)
    identity_run.annotate_run(run_path, references, loader=_mono)
    assert len(calls) == annotated  # 同じ条件なら照合し直さない

    monkeypatch.setattr(identity, "MIN_VOTES", 10**9)
    identity_run.annotate_run(run_path, references, loader=_mono)
    assert len(calls) == annotated * 2
    assert json.loads(path.read_text("utf-8"))["matches"] == []
