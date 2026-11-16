"""ep001-truth/score_usages.py: clock offset choice, unreleased usage classes and miss causes."""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root


import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "score_usages", dev_evidence_root(Path(__file__).resolve().parents[1]) / "ep001-truth" / "score_usages.py")
su = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(su)


def usage(seconds, cd, title):
    return {"seconds": seconds, "time": f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}", "cd_id": cd, "title": title}


def write_run(tmp: Path, published, segments=None, relations=None) -> Path:
    run_dir = tmp / "run"
    (run_dir / "semantics").mkdir(parents=True)
    (run_dir / "semantics" / "cue-semantics.json").write_text(json.dumps(
        {"segments": segments or [], "reference_relations": relations or {}}), encoding="utf-8")
    run = {"published": [{"start_seconds": a, "end_seconds": b, "title": t} for a, b, t in published],
           "semantics": {"record": {"name": "cue-semantics.json"}}}
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir / "run.json"


@pytest.fixture
def listed(tmp_path):
    usages = [usage(100, 375, "A"), usage(200, 375, "B"), usage(300, 375, "C"), usage(400, 375, "D"),
              usage(500, 576, "Released"), usage(600, 676, "ED"), usage(700, 375, "Preview")]
    path = tmp_path / "usages.json"
    path.write_text(json.dumps({"usages": usages}), encoding="utf-8")
    return path


def test_offset_override_shifts_every_usage(tmp_path, listed):
    run = write_run(tmp_path, [(113, 190, "a"), (213, 290, "b")])
    result = su.score(150, run, offset=13.0, usages_path=listed, offsets_path=tmp_path / "none.json")
    classes = {r["title"]: r["class"] for r in result["usages"] if r["kind"] == "unreleased"}
    assert classes["A"] == "ok" and classes["B"] == "ok"
    unshifted = su.score(150, run, offset=0.0, usages_path=listed, offsets_path=tmp_path / "none.json")
    assert {r["title"]: r["class"] for r in unshifted["usages"]}["A"] == "late"


def test_offset_comes_from_offsets_file_before_relations(tmp_path, listed):
    offsets = tmp_path / "offsets.json"
    offsets.write_text(json.dumps({"episodes": {"150": {"seconds": 13.1, "method": "ending_theme_alignment"}}}), encoding="utf-8")
    run = write_run(tmp_path, [], relations={"related_occurrences": [
        {"display_title": "Released", "source_start_seconds": 520.0, "source_end_seconds": 560.0}]})
    assert su.score(150, run, usages_path=listed, offsets_path=offsets)["clock_offset"] == {
        "seconds": 13.1, "method": "ending_theme_alignment"}
    by_relation = su.score(150, run, usages_path=listed, offsets_path=tmp_path / "none.json")["clock_offset"]
    assert by_relation["method"] == "relation_anchors" and by_relation["seconds"] == 20.0


def test_relation_offset_ignores_a_far_anchor():
    usages = [usage(100, 576, "Theme"), usage(300, 576, "Other")]
    occurrences = [{"display_title": "Theme", "source_start_seconds": 113.0},
                   {"display_title": "Other", "source_start_seconds": 313.6},
                   {"display_title": "Other", "source_start_seconds": 313.1},
                   {"display_title": "Theme", "source_start_seconds": 133.0}]
    value, anchors = su.relation_offset(usages, occurrences)
    assert len(anchors) == 4 and value == 13.1


def test_classes_merge_and_after_ending(tmp_path, listed):
    published = [(100, 130, "a1"), (140, 190, "a2"),   # A split
                 (210, 290, "b"),                      # B starts late
                 (300, 480, "cd")]                     # C covers D's start too: possible merge; D is split-free ok
    run = write_run(tmp_path, published)
    result = su.score(150, run, offset=0.0, usages_path=listed, offsets_path=tmp_path / "none.json")
    rows = {r["title"]: r for r in result["usages"]}
    assert rows["A"]["class"] == "split"
    assert rows["B"]["class"] == "late"
    assert rows["C"]["class"] == "ok" and rows["C"]["possible_merge"]
    assert rows["Preview"]["class"] == "none" and rows["Preview"]["after_ending"]
    tally = result["tally"]["unreleased"]
    assert tally["classes"]["none"] == 1 and tally["none_after_ending"] == 1
    assert tally["clean"] == tally["classes"]["ok"] - tally["possible_merge"]
    assert result["tally"]["unreleased"]["correct"] == tally["classes"]["ok"]


def test_miss_cause_distinguishes_hold_from_nothing_detected(tmp_path, listed):
    segments = [{"start_seconds": 95, "end_seconds": 195, "semantic_class": "UNCERTAIN",
                 "reasons": ["longer_than_one_cue"], "scope": {"disposition": "HELD", "reason": "above_maximum_cue_length"}}]
    run = write_run(tmp_path, [], segments=segments)
    result = su.score(150, run, offset=0.0, usages_path=listed, offsets_path=tmp_path / "none.json")
    rows = {r["title"]: r for r in result["usages"]}
    assert rows["A"]["cause"] == "held" and rows["A"]["segments"][0]["scope_reason"] == "above_maximum_cue_length"
    assert rows["C"]["cause"] == "no_segment"
    assert result["tally"]["unreleased"]["causes"]["held"] >= 1


def test_unlisted_published_cue_is_counted(tmp_path, listed):
    run = write_run(tmp_path, [(100, 190, "a"), (900, 960, "extra")])
    assert su.score(150, run, offset=0.0, usages_path=listed, offsets_path=tmp_path / "none.json")["unlisted_published_cues"] == 1
