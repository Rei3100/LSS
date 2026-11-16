"""Reference relation -> listening name seam (integrated successor).

A published cue that the relation engine verified as a version of one released recording is named after that
recording ("<title> — 関連版"). Harmony-only candidates, file-name titles, short quotes and unknown relations
never name a cue.
"""
from pathlib import Path

from lss.assets import naming
from lss.bgm import listening

SHA = "a" * 64
RATE = 48000


def _provenance(state="released_recording_edit", regions=None, *, title="戦闘のテーマ",
                title_source="reference_file_tags", start=100.0, end=140.0):
    if regions is None:
        regions = [{"start_seconds": start, "end_seconds": start + 30.0, "kind": "supported",
                    "reference_sha256": SHA, "offset_seconds": -40.0}]
    return {"start_frame": int(start * RATE), "end_frame": int(end * RATE), "sample_rate": RATE,
            # Source media time carries an offset; regions are on the analysed clock.
            "start_seconds": start + 7.5, "end_seconds": end + 7.5,
            "reference_relation": {"state": state, "basis": "x",
                                   "reference": {"reference_sha256": SHA, "display_title": title,
                                                 "title_source": title_source, "offset_seconds": -40.0},
                                   "regions": regions}}


def test_verified_edit_names_cue_after_released_recording():
    evidence = listening.related_naming_evidence("bgm-cue:k", _provenance())
    assert evidence["related"]["title"] == "戦闘のテーマ"
    assert evidence["related"]["present_share"] == 0.75
    proposal = naming.propose("bgm-cue:k", evidence)
    assert proposal["kind"] == "provisional_related"
    assert proposal["title"] == "戦闘のテーマ — 関連版"
    assert proposal["is_official"] is False


def test_component_variant_and_additional_material_also_name():
    for state, kind in (("released_recording_component_variant", "missing_component"),
                        ("released_recording_with_additional_material", "additional_material")):
        regions = [{"start_seconds": 100.0, "end_seconds": 125.0, "kind": kind, "reference_sha256": SHA}]
        assert listening.related_naming_evidence("bgm-cue:k", _provenance(state, regions)) is not None


def test_short_quote_does_not_name_cue():
    regions = [{"start_seconds": 100.0, "end_seconds": 110.0, "kind": "supported", "reference_sha256": SHA},
               {"start_seconds": 110.0, "end_seconds": 140.0, "kind": "unexplained_music", "reference_sha256": None}]
    assert listening.related_naming_evidence("bgm-cue:k", _provenance(regions=regions)) is None


def test_other_recording_regions_do_not_count():
    regions = [{"start_seconds": 100.0, "end_seconds": 140.0, "kind": "supported", "reference_sha256": "b" * 64}]
    assert listening.related_naming_evidence("bgm-cue:k", _provenance(regions=regions)) is None


def test_harmony_only_unknown_exact_and_untagged_never_name():
    for state in ("related_version_candidate", "unknown", "exact_released_recording", None):
        assert listening.related_naming_evidence("bgm-cue:k", _provenance(state)) is None
    assert listening.related_naming_evidence("bgm-cue:k", _provenance(title_source=None)) is None
    assert listening.related_naming_evidence("bgm-cue:k", _provenance(title=None)) is None
    assert listening.related_naming_evidence("bgm-cue:k", {"start_frame": 0, "end_frame": RATE,
                                                          "sample_rate": RATE}) is None


def test_regions_are_read_on_the_analysed_clock_not_source_time():
    # With source time (start+7.5) the overlap would shrink to 22.5/40; on the analysed clock it is 30/40.
    evidence = listening.related_naming_evidence("bgm-cue:k", _provenance())
    assert evidence["related"]["present_share"] == 0.75


def test_complete_projection_uses_relation_decisions_not_fingerprint_hits():
    from lss.output import result_report, results

    occurrence = {"start_seconds": 10.0, "end_seconds": 40.0, "source_start_seconds": 70.0,
                  "source_end_seconds": 100.0, "relation": "exact_released_recording", "reference_sha256": SHA,
                  "display_title": "戦闘のテーマ", "reference_range_seconds": [0.0, 30.0], "tags": {"x": "y"}}
    record = {"references": {"known_spans": [{"start_seconds": 0.0, "end_seconds": 50.0, "reference": "a.flac"}]},
              "reference_relations": {"state": "checked", "occurrences": [occurrence]}}
    projection = results.project(disposition="PUBLISHED", semantic_record=record)
    assert projection["reference_check"] == "checked"
    assert projection["available_count"] == 1
    [known] = projection["known"]
    assert known["state"] == "ALREADY_AVAILABLE" and known["basis"] == "reference_relation_decision"
    assert known["display_title"] == "戦闘のテーマ" and "tags" not in known
    page = result_report.render(Path("episode.mkv"), "bgm", projection)
    assert "手元の既発売の録音と同じだった区間：1件" in page and "01:10–01:40" in page and "戦闘のテーマ" in page


def test_unchecked_or_absent_references_keep_previous_projection():
    from lss.output import result_report, results

    legacy = {"references": {"known_spans": [{"start_seconds": 0.0, "end_seconds": 5.0}]}}
    projection = results.project(disposition="PUBLISHED", semantic_record=legacy)
    assert [k["state"] for k in projection["known"]] == ["KNOWN"] and projection["available_count"] is None
    failed = {"references": {}, "reference_relations": {"state": "not_checked", "error": "OSError"}}
    projection = results.project(disposition="PUBLISHED", semantic_record=failed)
    assert projection["known"] == [] and projection["reference_check"] == "not_checked"
    assert "照合はできませんでした" in result_report.render(Path("e.mkv"), "bgm", projection)
