"""共通録音の重なりを整理しても、観測した出現範囲と別の登場を失わない。"""
from __future__ import annotations

import copy
import json
import sqlite3

from lss.assets import catalog


def _pair(first, second, *spans):
    return {"first": first, "second": second, "occurrences": [
        {"first_start_seconds": a, "first_end_seconds": b, "second_start_seconds": c,
         "second_end_seconds": d, "waveform_explained": 0.8, "waveform_blocks": 6} for a, b, c, d in spans]}


def _build(tmp_path, pairs, name):
    record = {"type": "series_identity_annotation",
              "episodes": [{"key": e, "source_sha256": e * 64} for e in "abc"], "shared_recordings": pairs}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    database = catalog.build(state_root=tmp_path / "state", identity_records=[path], database=tmp_path / f"{name}.sqlite")
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT recording_sha256,source_sha256,start_seconds,end_seconds FROM occurrences"
                                  " ORDER BY recording_sha256,source_sha256,start_seconds,end_seconds").fetchall()
    return catalog.summary(database=database), rows


def test_overlapping_occurrence_observations_keep_the_longer_tail_and_are_order_independent(tmp_path):
    pairs = [_pair("a", "b", (10.0, 30.0, 100.0, 120.0)), _pair("a", "c", (20.0, 40.0, 200.0, 220.0))]
    summary, rows = _build(tmp_path, pairs, "forward")
    assert summary["recordings"] == 1 and summary["occurrences"] == 3
    assert [(r[1][0], r[2], r[3]) for r in rows] == [("a", 10.0, 40.0), ("b", 100.0, 120.0), ("c", 200.0, 220.0)]
    assert _build(tmp_path, list(reversed(pairs)), "reverse") == (summary, rows)


def test_a_late_bridge_links_both_old_fragments_without_losing_either_source(tmp_path):
    pairs = [_pair("a", "b", (10.0, 30.0, 100.0, 120.0), (40.0, 60.0, 200.0, 220.0)),
             _pair("a", "c", (20.0, 50.0, 0.0, 30.0))]
    summary, rows = _build(tmp_path, pairs, "late")
    assert summary["recordings"] == 1 and summary["occurrences"] == 4
    assert [(r[2], r[3]) for r in rows if r[1] == "a" * 64] == [(10.0, 60.0)]
    reordered = copy.deepcopy(pairs[::-1])
    reordered[1]["occurrences"].reverse()
    assert _build(tmp_path, reordered, "early") == (summary, rows)


def test_separate_repeats_remain_separate_occurrences_with_an_unobserved_gap(tmp_path):
    summary, rows = _build(tmp_path, [_pair("a", "b", (10.0, 30.0, 100.0, 120.0),
                                                               (50.0, 70.0, 100.0, 120.0))], "repeats")
    assert summary["recordings"] == 1 and summary["occurrences"] == 3
    assert [(r[2], r[3]) for r in rows if r[1] == "a" * 64] == [(10.0, 30.0), (50.0, 70.0)]


def test_weak_overlap_does_not_turn_two_recordings_into_one(tmp_path):
    summary, rows = _build(tmp_path, [_pair("a", "b", (10.0, 30.0, 100.0, 120.0)),
                                    _pair("a", "c", (25.0, 45.0, 200.0, 220.0))], "different")
    assert summary["recordings"] == 2 and summary["occurrences"] == 4
    assert len({row[0] for row in rows}) == 2
