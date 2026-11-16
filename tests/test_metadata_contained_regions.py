"""Released fragments retain their observed gaps and the cue's source-time bounds."""
import json
import pytest
from mutagen.flac import FLAC

from lss.bgm import listening
from lss.core import audio_identity
from lss.metadata import library
from tests.test_bgm_collection_numbering import cue, library_for
from tests.test_metadata_project import album


def region(sha, start, end, kind="supported"):
    return {"reference_sha256": sha, "start_seconds": start, "end_seconds": end, "kind": kind}


@pytest.mark.parametrize("version_of", [None, "a"])
def test_contained_ranges_are_clipped_unioned_and_kept_separate_per_recording(tmp_path, monkeypatch, version_of):
    monkeypatch.setattr(library, "_reference_info", lambda sha, *args: {
        "title": "メインテーマ" if sha == "a" else "決戦"})
    provenance = {"sample_rate": 44100, "start_frame": 100 * 44100,
                  "end_frame": 200 * 44100, "start_seconds": 130.}
    relation = {"regions": [region("a", 190, 220), region("a", 105, 120), region("a", 80, 110),
                            region("a", 118, 120), region("a", 140, 150), region("a", 150, 160),
                            region("a", 180, 170), region("b", 125, 130),
                            region("a", 120, 140, "unexplained"),
                            region("a", 120, 140, "additional_material"),
                            region("a", 120, 140, "missing_component")]}
    expected = [{"title": "メインテーマ", "start_seconds": 130., "end_seconds": 150.},
                {"title": "決戦", "start_seconds": 155., "end_seconds": 160.},
                {"title": "メインテーマ", "start_seconds": 170., "end_seconds": 190.},
                {"title": "メインテーマ", "start_seconds": 220., "end_seconds": 230.}]
    if version_of:
        expected = [row for row in expected if row["title"] == "決戦"]
    assert library._contained(relation, provenance, {"album": album()}, tmp_path, version_of) == expected


def test_disjoint_quotes_reach_saved_flac_and_survive_restart_without_pcm_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "_reference_info", lambda *args: {"title": "メインテーマ"})
    path = cue(tmp_path, "unreleased with quotes", 20)
    tags = FLAC(path)
    provenance = json.loads(tags["LSS_BGM_PROVENANCE"][0])
    provenance.update(start_seconds=120., end_seconds=125., reference_relation={
        "state": "released_recording_with_additional_material", "reference": {"reference_sha256": "a"},
        "regions": [region("a", 19, 20.75), region("a", 20.5, 21),
                    region("a", 23, 23.5), region("a", 24.5, 27)]})
    tags["LSS_BGM_PROVENANCE"] = json.dumps(provenance)
    tags.save()
    before = audio_identity.audio_identity(path)
    root = tmp_path / "state"
    summary = library_for("第1話", 1, album())
    listening.finish_outputs([path], root, library=summary)
    tags = FLAC(path)
    assert tags["DISCNUMBER"] == ["3"] and "EDIT_OF" not in tags
    assert tags["CONTAINS_RELEASED"] == ["メインテーマ"] * 3
    assert tags["CONTAINS_RELEASED_START"] == ["02:00", "02:03", "02:04"]
    assert tags["CONTAINS_RELEASED_END"] == ["02:01", "02:03", "02:05"]
    member, = library.read(root, summary["key"])["members"].values()
    assert member["contains_released"] == [
        {"title": "メインテーマ", "start_seconds": 120., "end_seconds": 121.},
        {"title": "メインテーマ", "start_seconds": 123., "end_seconds": 123.5},
        {"title": "メインテーマ", "start_seconds": 124.5, "end_seconds": 125.}]
    finished = path.read_bytes()
    # Clear process-local memoization; the saved ledger/receipts remain the authority on restart.
    audio_identity._IDENTITIES.clear()
    audio_identity._BY_CONTENT.clear()
    listening.finish_outputs([path], root, library=summary)
    assert path.read_bytes() == finished
    assert audio_identity.audio_identity(path) == before
