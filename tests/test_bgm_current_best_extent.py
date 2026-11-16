"""CurrentBest: a fragment whose one-cue extent was unresolved is replaced by a new cue that covers it with a
decided extent (more complete asset); nothing else about the adoption rules changes."""
from __future__ import annotations

from lss.bgm import current_best


def _file(path, start, end, *, extent, pcm, workflow="episode-sfc-routed-cues-v13"):
    return {"public_path": path, "start_seconds": start, "end_seconds": end, "extent": extent, "pcm_sha256": pcm,
            "workflow": workflow, "run_key": path, "improvements": []}


VALID = {"state": "VALID", "basis": []}


def test_an_unresolved_fragment_is_replaced_by_the_resolved_cue_that_covers_it():
    old = _file("old 05m09s (区切り未確定).flac", 309.0, 329.0, extent="unresolved", pcm="a")
    new = _file("new 05m09s.flac", 309.0, 380.37, extent="single_cue", pcm="b")
    chosen = current_best.select([old], [VALID], [new])
    assert [item["public_path"] for item in chosen["place"]] == [new["public_path"]]
    [decision] = chosen["decisions"]
    assert decision["decision"] == "replace_with_better"
    assert decision["improvements"] == [current_best.EXTENT_IMPROVEMENT]


def test_no_replacement_when_the_new_cue_does_not_cover_or_is_not_resolved_or_not_longer():
    old = _file("old.flac", 309.0, 329.0, extent="unresolved", pcm="a")
    partial = _file("partial.flac", 315.0, 380.0, extent="single_cue", pcm="b")         # misses the head
    still_unresolved = _file("unresolved.flac", 309.0, 380.0, extent="unresolved", pcm="c")
    same_length = _file("same.flac", 309.0, 329.5, extent="single_cue", pcm="d")
    for new in (partial, still_unresolved, same_length):
        chosen = current_best.select([old], [VALID], [new])
        assert chosen["decisions"][0]["decision"] == "keep", new["public_path"]
        assert chosen["place"] == []


def test_a_resolved_old_cue_or_an_unknown_extent_is_never_replaced_by_extent():
    new = _file("new.flac", 300.0, 400.0, extent="single_cue", pcm="b")
    for extent in ("single_cue", "multiple_cues", None):
        old = _file("old.flac", 309.0, 329.0, extent=extent, pcm="a")
        assert current_best.better(new, old) == []
