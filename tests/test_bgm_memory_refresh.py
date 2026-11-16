from __future__ import annotations

import json
from pathlib import Path

import pytest

from lss.bgm import cue_memory, memory_refresh as refresh


def _setup(tmp_path: Path, monkeypatch, *, pinned=False, rate=None, overlap=8.0, extension=5.5,
           same_source=False, donor_present=True):
    state = tmp_path / "state"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    source = audio_dir / "episode.mkv"
    source.write_bytes(b"source")
    target_path, witness_path = audio_dir / "target.flac", audio_dir / "witness.flac"
    target_path.write_bytes(b"t")
    witness_path.write_bytes(b"w")
    src_sha, donor_sha = "a" * 64, ("a" * 64 if same_source else "b" * 64)
    target = {"asset_id": "bgm-cue:target", "cue_key": "target", "source_sha256": src_sha,
              "path": str(target_path), "pcm_sha256": "1" * 64, "duration_seconds": 20.0,
              "unresolved_extent": True, "source_name": "episode"}
    witness = {"asset_id": "bgm-cue:witness", "cue_key": "witness", "source_sha256": donor_sha,
               "path": str(witness_path), "pcm_sha256": "2" * 64, "duration_seconds": 30.0,
               "unresolved_extent": False, "source_name": "other"}
    monkeypatch.setattr(cue_memory, "_still_present", lambda member: member["path"] != str(witness_path) or donor_present)
    mem_path = cue_memory.root(state) / cue_memory.RECORD_NAME
    mem_path.parent.mkdir(parents=True)
    mem_path.write_text(json.dumps({"type": "bgm_cue_memory", "schema_version": cue_memory.SCHEMA_VERSION,
                                    "recipe_id": cue_memory.recipe_id(), "recipe": cue_memory.recipe(),
                                    "members": {target["asset_id"]: target, witness["asset_id"]: witness},
                                    "links": [{"a": target["asset_id"], "b": witness["asset_id"], "paired": True, "waveform_explained": .8,
                                               "a_ranges": [[0.0, overlap]],
                                               "b_ranges": [[0.0, overlap]], "rate": rate}],
                                    "labels": {"1": {"name": "anything"}}, "retired": []}), encoding="utf-8")
    # The verified passage maps witness [0, overlap] to target [0, overlap]. Its full 30s clock
    # therefore extends 10s past the target. Clamp via a tunable witness duration for extension tests.
    witness["duration_seconds"] = 20.0 + extension
    mem_path.write_text(json.dumps({"type": "bgm_cue_memory", "schema_version": cue_memory.SCHEMA_VERSION,
                                    "recipe_id": cue_memory.recipe_id(), "recipe": cue_memory.recipe(),
                                    "members": {target["asset_id"]: target, witness["asset_id"]: witness},
                                    "links": [{"a": target["asset_id"], "b": witness["asset_id"], "paired": True, "waveform_explained": .8,
                                               "a_ranges": [[0.0, overlap]], "b_ranges": [[0.0, overlap]], "rate": rate}],
                                    "labels": {"1": {"label": "anything"}}, "retired": []}), encoding="utf-8")
    for sha, row, pin in ((src_sha, target, pinned), (donor_sha, witness, False)):
        if sha == donor_sha and same_source:
            continue
        run_path = state / "runs" / f"{sha}.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps({"source": {"path": str(source), "sha256": sha}}), encoding="utf-8")
        row.update(run_path=str(run_path), public_path=row["path"])
        cb = state / "bgm-current-best" / f"{sha}.json"
        cb.parent.mkdir(parents=True, exist_ok=True)
        cb.write_text(json.dumps({"type": "bgm_current_best", "schema_version": 2, "source_sha256": sha,
                                  "current": {"pinned": pin, "files": [row]}}), encoding="utf-8")
    return state


def test_only_unresolved_target_with_present_external_constant_offset_donor(tmp_path, monkeypatch):
    state = _setup(tmp_path, monkeypatch)
    (plan,) = refresh.pending(state)
    assert plan["source_path"].endswith("episode.mkv")
    assert plan["source_sha256"] == "a" * 64
    assert plan["target_member"] == "bgm-cue:target"
    assert plan["witness_member"] == "bgm-cue:witness"


@pytest.mark.parametrize("bad", [None, [], {"paired": True, "waveform_explained": "broken"},
                                 {"paired": True, "waveform_explained": .8, "a": [], "b": {}}])
def test_malformed_link_does_not_hide_valid_refresh(tmp_path, monkeypatch, bad):
    state = _setup(tmp_path, monkeypatch)
    expected = refresh.pending(state)
    assert len(expected) == 1
    path = cue_memory.root(state) / cue_memory.RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["links"].insert(0, bad)
    path.write_text(json.dumps(record), encoding="utf-8")
    assert refresh.pending(state) == expected
    record["links"] = [bad]
    path.write_text(json.dumps(record), encoding="utf-8")
    assert refresh.pending(state) == []


@pytest.mark.parametrize("kwargs", [
    {"pinned": True}, {"rate": 1.02}, {"overlap": 5.9}, {"extension": 1.9},
    {"same_source": True}, {"donor_present": False}, {"extension": 4.9},
])
def test_false_triggers_return_no_plan(tmp_path, monkeypatch, kwargs):
    state = _setup(tmp_path, monkeypatch, **kwargs)
    assert refresh.pending(state) == []


def test_token_ignores_labels_and_presentation_metadata(tmp_path, monkeypatch):
    state = _setup(tmp_path, monkeypatch)
    token = refresh.pending(state)[0]["token"]
    path = cue_memory.root(state) / cue_memory.RECORD_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["labels"] = {"42": {"label": "renamed"}}
    data["members"]["bgm-cue:target"]["source_name"] = "a different display name"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert refresh.pending(state)[0]["token"] == token


@pytest.mark.parametrize("outcome,status", [({"ok": True}, "complete"), ({"ok": False}, "failed")])
def test_claim_and_terminal_receipt_prevent_repeat(tmp_path, monkeypatch, outcome, status):
    state = _setup(tmp_path, monkeypatch)
    plan = refresh.pending(state)[0]
    assert refresh.claim(state, plan) is True
    receipt = state / refresh.DIRECTORY / f"{plan['token']}.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "running"
    assert refresh.claim(state, plan) is False
    refresh.finish(state, plan, outcome)
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == status
    assert refresh.pending(state) == []
    assert refresh.claim(state, plan) is False


def test_interrupted_claim_can_be_retried_by_next_process(tmp_path, monkeypatch):
    state = _setup(tmp_path, monkeypatch)
    plan = refresh.pending(state)[0]
    assert refresh.claim(state, plan) is True
    receipt = state / refresh.DIRECTORY / f"{plan['token']}.json"
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["pid"] = -1
    receipt.write_text(json.dumps(value), encoding="utf-8")
    assert refresh.pending(state) == [plan]
    assert refresh.claim(state, plan) is True


def test_repeated_three_second_phrase_cannot_count_as_six_second_constant_clock():
    target = {"duration_seconds": 20}
    witness = {"asset_id": "witness", "duration_seconds": 30}
    link = {"a": "target", "b": "witness", "paired": True, "waveform_explained": .8,
            "a_ranges": [[0, 3], [10, 13]], "b_ranges": [[0, 3], [0, 3]]}
    assert refresh._link_geometry(link, "target", target, witness) is None


def test_offset_maps_witness_into_target_clock_in_both_directions():
    target = {"duration_seconds": 20}
    witness = {"asset_id": "witness", "duration_seconds": 30}
    link = {"a": "target", "b": "witness", "paired": True, "waveform_explained": .8,
            "a_ranges": [[10, 18]], "b_ranges": [[0, 8]]}
    assert refresh._link_geometry(link, "target", target, witness) == (8, 20)
    link.update(a_ranges=[[0, 8]], b_ranges=[[10, 18]])
    assert refresh._link_geometry(link, "target", target, witness) == (8, 10)


def test_windows_owner_query_does_not_use_process_signals(monkeypatch):
    import os
    if os.name != "nt":
        pytest.skip("Windows process query")
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("process signal used as liveness query"))
    assert refresh._pid_alive(os.getpid())
    assert not refresh._pid_alive(-1)


def test_fixed_match_window_counts_only_audio_present_on_both_clocks():
    # A synthetic link extends beyond the short reference; count only shared support.
    target = {"duration_seconds": 20}
    witness = {"asset_id": "witness", "duration_seconds": 55.5}
    link = {"a": "target", "b": "witness", "paired": True, "waveform_explained": .8,
            "a_ranges": [[.5, 20.5]], "b_ranges": [[0, 20]]}
    overlap, extension = refresh._link_geometry(link, "target", target, witness)
    assert overlap == pytest.approx(19.5)
    assert extension == pytest.approx(36.0)
    link.update(a_ranges=[[19, 39]], b_ranges=[[0, 20]])
    assert refresh._link_geometry(link, "target", target, witness)[0] == 1
    link.update(a_ranges=[[0, 6.10]], b_ranges=[[0, 5.99]])
    assert refresh._link_geometry(link, "target", target, witness)[0] == pytest.approx(5.99)


def test_full_source_attempt_covers_alternate_cues_and_unrelated_growth(tmp_path, monkeypatch):
    state = _setup(tmp_path, monkeypatch)
    path = cue_memory.root(state) / cue_memory.RECORD_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    other = {**data["members"]["bgm-cue:target"], "asset_id": "bgm-cue:other-target", "cue_key": "other-target"}
    data["members"][other["asset_id"]] = other
    data["links"].append({**data["links"][0], "a": other["asset_id"]})
    path.write_text(json.dumps(data), encoding="utf-8")
    cb = state / "bgm-current-best" / ("a" * 64 + ".json")
    current = json.loads(cb.read_text(encoding="utf-8"))
    current["current"]["files"].append({**current["current"]["files"][0], "cue_key": "other-target"})
    cb.write_text(json.dumps(current), encoding="utf-8")
    plans = refresh.pending(state, recipe_id="one")
    assert len(plans) == 1
    plan = plans[0]
    assert refresh.claim(state, plan)
    refresh.finish(state, plan, {"ok": True})
    assert not refresh.pending(state, recipe_id="one")
    # A new accepted external file without a verified link does not justify repeating the old source.
    unrelated = {**data["members"]["bgm-cue:witness"], "asset_id": "bgm-cue:unrelated",
                 "cue_key": "unrelated", "pcm_sha256": "3" * 64}
    data["members"][unrelated["asset_id"]] = unrelated
    path.write_text(json.dumps(data), encoding="utf-8")
    donor_cb = state / "bgm-current-best" / ("b" * 64 + ".json")
    donor = json.loads(donor_cb.read_text(encoding="utf-8"))
    donor["current"]["files"].append({**donor["current"]["files"][0], "cue_key": "unrelated", "pcm_sha256": "3" * 64})
    donor_cb.write_text(json.dumps(donor), encoding="utf-8")
    assert not refresh.pending(state, recipe_id="one")
    data["links"].append({**data["links"][0], "b": unrelated["asset_id"]})
    path.write_text(json.dumps(data), encoding="utf-8")
    assert refresh.pending(state, recipe_id="one")
    assert refresh.pending(state, recipe_id="two")
