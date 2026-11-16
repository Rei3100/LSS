"""Integration falsifiers for the merged distinct-fixed-name rule.

`cloud/invariant-audit` and `cloud/video-metadata` each made a later recording's fixed scene name distinct from
the names other recordings of the same work hold. The merged rule keeps both: names given in one settle are
adjusted oldest first (whatever order the store lists them in), names compare after NFKC/case/space folding, and
a stale ``asset:`` row of a cue that has joined a recurring group never blocks a name.
"""
from __future__ import annotations

from lss.bgm import scene_names


def _cue(data: dict, asset_id: str, name: str, at: str, *, work: str = "work-1") -> None:
    data["cues"][asset_id] = {"asset_id": asset_id, "work_key": work, "name": name, "recorded_at": at,
                              "pcm_sha256": "p" + asset_id, "source_sha256": "s", "episode": 1}


def _settle(tmp_path, monkeypatch, data: dict, memory: dict | None = None) -> dict:
    monkeypatch.setattr(scene_names, "update", lambda state_root, change: change(data) or data)
    monkeypatch.setattr(scene_names, "_m_numbers", lambda data, by_recording: {})
    scene_names.settle(tmp_path, memory)
    return {key: value["name"] for key, value in data["recordings"].items()}


def test_the_oldest_new_name_keeps_it_even_when_the_store_lists_it_last(tmp_path, monkeypatch):
    data = {"cues": {}, "recordings": {}, "works": {}}
    _cue(data, "bgm-cue:late", "ミズウオ大作戦", "2026-01-02T00:00:00+00:00")
    _cue(data, "bgm-cue:early", "ミズウオ大作戦", "2026-01-01T00:00:00+00:00")
    names = _settle(tmp_path, monkeypatch, data)
    assert names["asset:bgm-cue:early"] == "ミズウオ大作戦"
    assert names["asset:bgm-cue:late"] == "ミズウオ大作戦 その2"


def test_width_case_and_space_variants_are_the_same_name(tmp_path, monkeypatch):
    data = {"cues": {}, "recordings": {}, "works": {}}
    _cue(data, "bgm-cue:1", "ＢＧＭの朝", "2026-01-01T00:00:00+00:00")
    _cue(data, "bgm-cue:2", "bgm の朝", "2026-01-02T00:00:00+00:00")
    names = _settle(tmp_path, monkeypatch, data)
    assert names["asset:bgm-cue:1"] == "ＢＧＭの朝"
    assert names["asset:bgm-cue:2"] == "bgm の朝 その2"


def test_a_stale_row_of_a_regrouped_cue_never_blocks_a_name(tmp_path, monkeypatch):
    data = {"cues": {}, "recordings": {}, "works": {}}
    _cue(data, "bgm-cue:1", "はじめての挑戦", "2026-01-01T00:00:00+00:00")
    _cue(data, "bgm-cue:2", "はじめての挑戦", "2026-01-03T00:00:00+00:00")
    # bgm-cue:1 was named alone, then joined the recurring group label:7 with bgm-cue:2 (one recording).
    data["recordings"]["asset:bgm-cue:1"] = {"name": "はじめての挑戦", "first_asset": "bgm-cue:1",
                                            "named_at": "2026-01-01T00:00:00+00:00"}
    _cue(data, "bgm-cue:3", "夜明けの道", "2026-01-04T00:00:00+00:00")
    _cue(data, "bgm-cue:4", "夜明けの道", "2026-01-05T00:00:00+00:00")
    memory = {"labels": {"7": {"members": ["bgm-cue:1", "bgm-cue:2"]}}}
    names = _settle(tmp_path, monkeypatch, data, memory)
    assert names["label:7"] == "はじめての挑戦"          # the group keeps the oldest member's name
    assert "asset:bgm-cue:1" not in names
    assert names["asset:bgm-cue:3"] == "夜明けの道"
    assert names["asset:bgm-cue:4"] == "夜明けの道 その2"
    assert _settle(tmp_path, monkeypatch, data, memory) == names   # settling again renames nothing
