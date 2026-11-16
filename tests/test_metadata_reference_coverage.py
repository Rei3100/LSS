"""Overlapping reference witnesses cannot turn a short quote into official cue metadata."""
from pathlib import Path

import pytest

from lss.metadata import library, project
from tests.test_metadata_project import album


@pytest.mark.parametrize("spans,official,contained_count", [
    ([(100, 120), (100, 120), (105, 125)], False, 1),
    ([(100, 120), (110, 140)], False, 1),
    ([(150, 175), (100, 125)], True, 0),
    ([(90, 130), (120, 155), (140, 160)], True, 0),
    ([(80, 110), (190, 230)], False, 2),
    ([(130, 110), (100, 120), (100, 120)], False, 1),
])
def test_unique_clipped_coverage_controls_official_tags(tmp_path, monkeypatch, spans, official, contained_count):
    monkeypatch.setattr(library, "_reference_info", lambda *args: {
        "sha256": "ref", "title": "メインテーマ", "length_seconds": 60,
    })
    relation = {"state": "released_recording_edit", "reference": {"reference_sha256": "ref"},
                "regions": [{"reference_sha256": "ref", "kind": "supported",
                             "start_seconds": start, "end_seconds": end} for start, end in spans]}
    provenance = {"sample_rate": 44100, "start_frame": 100 * 44100, "end_frame": 200 * 44100,
                  "start_seconds": 100, "end_seconds": 200, "reference_relation": relation}
    member = library.member(Path("cue.flac"), provenance, "cue", {"album": album()}, None, tmp_path)
    piece = {**member, "occurrences": [member]}
    tags = project.tags_for("cue", album(), {"cue": piece}, None)
    assert (member["official"] is not None) is official
    assert tags["DISCNUMBER"] == ("1" if official else "3")
    assert ("EDIT_OF" in tags) is official
    assert ("COMPOSER" in tags) is official
    if not official:
        assert tags["CONTAINS_RELEASED"] == ["メインテーマ"] * contained_count
