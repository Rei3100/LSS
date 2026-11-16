"""最初の対応領域を先に固定し、公開しない理由を必ず名前で残す。"""
from __future__ import annotations

import pytest

from lss.bgm.supported_scope import (
    ScopeError, SupportedScope, classify_candidate, classify_result,
)

RATE = 44100


def scope(**changes) -> SupportedScope:
    base = {"source_domains": ("finished_stereo_episode",),
            "published_semantic_roles": ("dramatic_score",),
            "covered_semantic_roles": ("dramatic_score", "singing_score"),
            "minimum_cue_seconds": 20.0, "maximum_cue_seconds": 300.0,
            "required_evidence": ("source_provenance", "occurrence_span"),
            "quality_gates": {"snr_db": 6.0},
            "singing_observation_policy": "exclude_from_publication",
            "derived_role_policy": "accept_declared_ambiguity",
            "cue_extent_policy": "allow_unresolved_cue_extent",
            "publication_allowed": True}
    return SupportedScope(**{**base, **changes})


def cue(**changes) -> dict:
    base = {"candidate_id": "cue-1", "sample_rate": RATE, "start_frame": 0,
            "end_frame": RATE * 30, "source_domain": "finished_stereo_episode",
            "semantic_role": "dramatic_score", "identity_state": "unknown",
            "evidence": {"source_provenance": {"sha256": "a" * 64}, "occurrence_span": [0, RATE * 30]},
            "quality": {"snr_db": 9.0}}
    return {**base, **changes}


def test_a_cue_meeting_every_declared_condition_is_published():
    decided = classify_candidate(cue(), scope())
    assert decided["disposition"] == "PUBLISHED" and decided["reason"] == "within_supported_scope"
    assert decided["scope_id"] == scope().scope_id and decided["cue_seconds"] == 30.0
    # 単独出現・未知の同一性・未発売であることだけでは除外しない。
    assert classify_candidate(cue(identity_state="unresolved"), scope())["disposition"] == "PUBLISHED"


@pytest.mark.parametrize("changes,disposition,reason", [
    ({"source_domain": "live_recording"}, "UNSUPPORTED", "source_domain_out_of_scope"),
    ({"semantic_role": "unresolved_music_like"}, "HELD", "semantic_role_unresolved"),
    ({"semantic_role": "unresolved"}, "HELD", "semantic_role_unresolved"),
    ({"semantic_role": "source_music"}, "UNSUPPORTED", "semantic_role_not_wanted_bgm"),
    ({"semantic_role": "musical_sfx"}, "UNSUPPORTED", "semantic_role_not_wanted_bgm"),
    ({"semantic_role": "vocalization"}, "UNSUPPORTED", "semantic_role_not_wanted_bgm"),
    ({"semantic_role": "singing_score"}, "UNSUPPORTED", "semantic_role_covered_but_not_published"),
    ({"end_frame": RATE * 19}, "HELD", "below_minimum_cue_usefulness"),
    ({"end_frame": RATE * 301}, "HELD", "above_maximum_cue_length"),
    ({"evidence": {"source_provenance": {"sha256": "a" * 64}}}, "HELD", "required_evidence_missing"),
    ({"evidence": {"source_provenance": {}, "occurrence_span": []}}, "HELD", "required_evidence_missing"),
    ({"quality": None}, "HELD", "quality_not_measured"),
    ({"quality": {"snr_db": 5.9}}, "HELD", "below_quality_gate"),
    ({"quality": {"snr_db": True}}, "HELD", "below_quality_gate"),
    ({"quality": {}}, "HELD", "below_quality_gate"),
    ({"duplicate_of_published_cue_id": "cue-earlier"}, "HELD", "duplicate_of_published_cue"),
])
def test_every_reason_for_not_publishing_has_a_name(changes, disposition, reason):
    decided = classify_candidate(cue(**changes), scope())
    assert (decided["disposition"], decided["reason"]) == (disposition, reason)


def test_a_wanted_role_outside_the_declared_scope_is_unsupported_not_unwanted():
    narrow = scope(covered_semantic_roles=("dramatic_score",))
    decided = classify_candidate(cue(semantic_role="singing_score"), narrow)
    assert (decided["disposition"], decided["reason"]) == ("UNSUPPORTED", "semantic_role_out_of_scope")


def test_missing_evidence_and_failed_gates_are_named_individually():
    decided = classify_candidate(cue(evidence={"source_provenance": None}), scope())
    assert decided["missing_evidence"] == ["source_provenance", "occurrence_span"]
    narrow = scope(quality_gates={"snr_db": 6.0, "stereo_match": 0.9})
    failed = classify_candidate(cue(quality={"snr_db": 9.0, "stereo_match": 0.5}), narrow)
    assert failed["failed_gates"] == ["stereo_match"]


def test_a_development_scope_holds_even_a_fully_qualified_cue():
    development = scope(publication_allowed=False)
    decided = classify_candidate(cue(), development)
    assert (decided["disposition"], decided["reason"]) == ("HELD", "publication_not_allowed_by_scope")
    assert development.scope_id != scope().scope_id


@pytest.mark.parametrize("changes", [
    {"published_semantic_roles": ()},
    {"published_semantic_roles": ("unresolved_music_like",)},
    {"published_semantic_roles": ("dramatic_score", "dramatic_score")},
    {"published_semantic_roles": ("wanted_score",)},
    {"covered_semantic_roles": ("singing_score",)},
    {"source_domains": ()},
    {"source_domains": ("",)},
    {"minimum_cue_seconds": 0},
    {"minimum_cue_seconds": float("inf")},
    {"maximum_cue_seconds": 10.0},
    {"required_evidence": ()},
    {"quality_gates": {}},
    {"quality_gates": {"snr_db": float("nan")}},
    {"quality_gates": {"snr_db": True}},
    {"publication_allowed": "yes"}, {"singing_observation_policy": "drop"},
    {"singing_observation_policy": None}, {"cue_extent_policy": "split"},
    {"cue_extent_policy": None},
    {"schema_version": 2},
    {"method": "guessed"},
])
def test_an_incomplete_scope_can_never_decide_anything(changes):
    incomplete = scope(**changes)
    with pytest.raises(ScopeError):
        incomplete.to_dict()
    with pytest.raises(ScopeError):
        classify_candidate(cue(), incomplete)
    with pytest.raises(ScopeError):
        classify_result([cue()], incomplete, music_activity_found=True)


@pytest.mark.parametrize("changes", [
    {"candidate_id": None}, {"sample_rate": 0}, {"end_frame": 0}, {"start_frame": -1},
    {"end_frame": 44100.5}, {"identity_state": "maybe"}, {"semantic_role": "score"},
    {"semantic_role": None},
])
def test_a_cue_without_a_readable_record_is_refused_not_guessed(changes):
    with pytest.raises(ScopeError):
        classify_candidate(cue(**changes), scope())


def test_evidence_that_is_not_a_record_is_refused():
    with pytest.raises(ScopeError):
        classify_candidate(cue(evidence=["source_provenance"]), scope())


def test_a_result_reports_published_held_and_unsupported_without_hiding_any():
    decided = classify_result([
        cue(candidate_id="a"),
        cue(candidate_id="b", semantic_role="source_music"),
        cue(candidate_id="c", quality={"snr_db": 1.0}),
    ], scope(), music_activity_found=True)
    assert decided["disposition"] == "PUBLISHED" and decided["reason"] == "published_cues_within_scope"
    assert decided["counts"] == {"HELD": 1, "PUBLISHED": 1, "UNSUPPORTED": 1}
    assert [row["candidate_id"] for row in decided["candidates"]] == ["a", "b", "c"]


def test_held_candidates_keep_the_result_from_being_reported_as_out_of_scope():
    held = classify_result([cue(candidate_id="a", semantic_role="source_music"),
                            cue(candidate_id="b", semantic_role="unresolved")],
                           scope(), music_activity_found=True)
    assert held["disposition"] == "HELD"
    unsupported = classify_result([cue(candidate_id="a", semantic_role="singing_score")],
                                  scope(), music_activity_found=True)
    assert (unsupported["disposition"], unsupported["reason"]) == ("UNSUPPORTED", "all_candidates_out_of_scope")


def test_music_that_is_not_wanted_is_a_legitimate_zero_not_an_unsupported_source():
    decided = classify_result([cue(candidate_id="a", semantic_role="source_music"),
                               cue(candidate_id="b", semantic_role="musical_sfx")],
                              scope(), music_activity_found=True)
    assert (decided["disposition"], decided["reason"]) == ("ZERO", "music_not_wanted")
    assert decided["counts"]["UNSUPPORTED"] == 2
    mixed = classify_result([cue(candidate_id="a", semantic_role="source_music"),
                             cue(candidate_id="b", semantic_role="singing_score")],
                            scope(), music_activity_found=True)
    assert mixed["disposition"] == "UNSUPPORTED"


def test_wanted_cues_that_only_duplicate_published_ones_are_a_legitimate_zero():
    decided = classify_result([cue(candidate_id="a", duplicate_of_published_cue_id="cue-x"),
                               cue(candidate_id="b", duplicate_of_published_cue_id="cue-y")],
                              scope(), music_activity_found=True)
    assert (decided["disposition"], decided["reason"]) == ("ZERO", "no_novel_asset")
    with_new = classify_result([cue(candidate_id="a", duplicate_of_published_cue_id="cue-x"),
                                cue(candidate_id="b", quality={"snr_db": 1.0})],
                               scope(), music_activity_found=True)
    assert with_new["disposition"] == "HELD"


@pytest.mark.parametrize("found,reason", [(True, "music_not_wanted"), (False, "no_wanted_score")])
def test_no_candidate_is_a_legitimate_zero_that_names_what_was_observed(found, reason):
    decided = classify_result([], scope(), music_activity_found=found)
    assert (decided["disposition"], decided["reason"]) == ("ZERO", reason)
    assert decided["counts"] == {"HELD": 0, "PUBLISHED": 0, "UNSUPPORTED": 0}


def test_duplicate_candidate_identifiers_are_refused():
    with pytest.raises(ScopeError):
        classify_result([cue(candidate_id="a"), cue(candidate_id="a")], scope(), music_activity_found=True)
    with pytest.raises(ScopeError):
        classify_result([], scope(), music_activity_found="yes")


@pytest.mark.parametrize("observed,policy,disposition,reason", [
    ("present", "exclude_from_publication", "UNSUPPORTED", "observed_singing_out_of_scope"),
    ("present", "ignore", "PUBLISHED", "within_supported_scope"),
    ("absent", "exclude_from_publication", "PUBLISHED", "within_supported_scope"),
    ("unresolved", "exclude_from_publication", "PUBLISHED", "within_supported_scope"),
    (None, "exclude_from_publication", "PUBLISHED", "within_supported_scope"),
])
def test_observed_singing_only_changes_coverage_never_the_role(observed, policy, disposition, reason):
    decided = classify_candidate(cue(observations={"singing": observed} if observed else {}),
                                 scope(singing_observation_policy=policy))
    assert (decided["disposition"], decided["reason"]) == (disposition, reason)
    # 観測はroleを書き換えない。
    assert decided["semantic_role"] == "dramatic_score"


def test_an_unknown_singing_observation_is_refused_not_guessed():
    with pytest.raises(ScopeError):
        classify_candidate(cue(observations={"singing": "maybe"}), scope())
    with pytest.raises(ScopeError):
        classify_candidate(cue(observations=["singing"]), scope())


def test_singing_exclusion_is_reported_as_coverage_not_as_a_zero_result():
    decided = classify_result([cue(candidate_id="a", observations={"singing": "present"}),
                               cue(candidate_id="b", observations={"singing": "present"})],
                              scope(), music_activity_found=True)
    # 欲しい劇伴ではない音楽（ZERO）ではなく、現在の対応範囲外として残す。
    assert (decided["disposition"], decided["reason"]) == ("UNSUPPORTED", "all_candidates_out_of_scope")
    assert decided["counts"]["UNSUPPORTED"] == 2


def test_a_candidate_known_to_hold_more_than_one_cue_is_never_published():
    """複数曲と分かった候補は、方針によらず一曲のcueとして公開しない。"""
    for policy in ("allow_unresolved_cue_extent", "require_single_cue_evidence"):
        decided = classify_candidate(
            cue(cue_extent={"state": "multiple_cues",
                            "evidence": {"released_reference_count": 2}}),
            scope(cue_extent_policy=policy),
        )
        assert decided["disposition"] == "HELD"
        assert decided["reason"] == "contains_more_than_one_cue"
        assert decided["cue_extent_state"] == "multiple_cues"


def test_unresolved_cue_extent_blocks_only_where_the_scope_requires_the_evidence():
    strict = scope(cue_extent_policy="require_single_cue_evidence")

    assert classify_candidate(cue(), strict)["reason"] == "cue_extent_unresolved"
    assert classify_candidate(cue(cue_extent={"state": "unresolved"}), strict)["reason"] == (
        "cue_extent_unresolved")
    assert classify_candidate(cue(cue_extent={"state": "single_cue"}), strict)["disposition"] == (
        "PUBLISHED")
    assert classify_candidate(cue(), scope())["disposition"] == "PUBLISHED"


def test_the_cue_extent_is_read_before_the_declared_cue_length():
    """複数曲を含む候補の長さを、一曲の長さとして判断しない。"""
    decided = classify_candidate(
        cue(end_frame=RATE * 301, cue_extent={"state": "multiple_cues"}), scope())

    assert decided["reason"] == "contains_more_than_one_cue"


def test_rejects_a_cue_extent_that_is_not_a_declared_state():
    for extent in ({"state": "probably"}, {"evidence": {}}, "single_cue", []):
        with pytest.raises(ScopeError, match="cueの範囲"):
            classify_candidate(cue(cue_extent=extent), scope())


def test_the_cue_extent_policy_changes_the_scope_identity():
    assert scope().scope_id != scope(cue_extent_policy="require_single_cue_evidence").scope_id
    assert "cue_extent_policy" in scope().to_dict()


def _derivation(**changes) -> dict:
    base = {"semantic_role": "dramatic_score", "reason": "no_observed_singing",
            "policy_id": "bgm-role-policy-v1-test", "derived_by_policy": True,
            "indistinguishable_roles": ["source_music", "musical_sfx"]}
    return {**base, **changes}


def test_a_derived_role_that_cannot_exclude_unpublished_roles_is_held():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score", semantic_role_derivation=_derivation()),
        scope(derived_role_policy="require_unambiguous_role"))

    assert decided["disposition"] == "HELD"
    assert decided["reason"] == "derived_role_cannot_exclude_other_roles"
    assert decided["ambiguous_roles"] == ["musical_sfx", "source_music"]
    assert decided["role_policy_id"] == "bgm-role-policy-v1-test"


def test_the_declared_ambiguity_can_be_accepted_by_the_scope():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score", semantic_role_derivation=_derivation()),
        scope(derived_role_policy="accept_declared_ambiguity"))

    assert decided["reason"] != "derived_role_cannot_exclude_other_roles"


def test_a_role_taken_from_the_record_is_not_gated_by_any_derivation():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score"),
        scope(derived_role_policy="require_unambiguous_role"))

    assert decided["reason"] != "derived_role_cannot_exclude_other_roles"


def test_an_ambiguity_that_stays_inside_the_published_roles_does_not_hold():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score",
            semantic_role_derivation=_derivation(indistinguishable_roles=["dramatic_score"])),
        scope(derived_role_policy="require_unambiguous_role"))

    assert decided["reason"] != "derived_role_cannot_exclude_other_roles"


@pytest.mark.parametrize("derivation", ["derived", {"indistinguishable_roles": [1]},
                                        {"indistinguishable_roles": "source_music"}])
def test_a_malformed_derivation_is_refused(derivation):
    with pytest.raises(ScopeError):
        classify_candidate(
            cue(semantic_role="dramatic_score", semantic_role_derivation=derivation),
            scope(derived_role_policy="require_unambiguous_role"))


def test_the_derived_role_policy_changes_the_scope_identity():
    assert (scope(derived_role_policy="require_unambiguous_role").scope_id
            != scope(derived_role_policy="accept_declared_ambiguity").scope_id)


def test_several_occurrences_of_one_recording_are_held_until_one_is_chosen():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score",
            occurrence_group={"group_id": "g1", "members": ["cue-1", "cue-2"],
                              "chosen_candidate_id": None}),
        scope())

    assert decided["disposition"] == "HELD"
    assert decided["reason"] == "occurrence_choice_not_made"
    assert decided["other_occurrences"] == ["cue-2"]


def test_the_chosen_occurrence_is_not_held_by_the_duplicate_rule():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score",
            occurrence_group={"group_id": "g1", "members": ["cue-1", "cue-2"],
                              "chosen_candidate_id": "cue-1"}),
        scope())

    assert decided["reason"] not in ("occurrence_choice_not_made", "another_occurrence_is_chosen")


def test_an_occurrence_that_was_not_chosen_is_held_with_its_own_reason():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score",
            occurrence_group={"group_id": "g1", "members": ["cue-1", "cue-2"],
                              "chosen_candidate_id": "cue-2"}),
        scope())

    assert decided["reason"] == "another_occurrence_is_chosen"
    assert decided["chosen_candidate_id"] == "cue-2"


def test_a_group_with_only_this_candidate_does_not_hold_it():
    decided = classify_candidate(
        cue(semantic_role="dramatic_score",
            occurrence_group={"group_id": "g1", "members": ["cue-1"],
                              "chosen_candidate_id": None}),
        scope())

    assert decided["reason"] not in ("occurrence_choice_not_made", "another_occurrence_is_chosen")


def test_a_malformed_occurrence_group_is_refused():
    with pytest.raises(ScopeError):
        classify_candidate(
            cue(semantic_role="dramatic_score", occurrence_group={"group_id": "g1"}), scope())


@pytest.mark.parametrize("changes,disposition,reason", [
    ({"known_release": "exact"}, "HELD", "known_released_recording"),
    ({"known_release": "partial"}, "HELD", "partly_known_released_recording"),
    ({"excluded_non_target": "theme_song_shape_with_singing"}, "UNSUPPORTED", "excluded_non_target_music"),
])
def test_known_releases_and_policy_exclusions_are_named(changes, disposition, reason):
    decided = classify_candidate(cue(**changes), scope())
    assert (decided["disposition"], decided["reason"]) == (disposition, reason)


@pytest.mark.parametrize("changes", [{"known_release": "maybe"}, {"excluded_non_target": ""}])
def test_malformed_identity_or_exclusion_is_rejected(changes):
    with pytest.raises(ScopeError):
        classify_candidate(cue(**changes), scope())


def test_only_known_releases_and_duplicates_are_no_novel_asset():
    decided = classify_result([cue(known_release="exact"),
                               cue(candidate_id="cue-2", duplicate_of_published_cue_id="cue-earlier")],
                              scope(), music_activity_found=True)
    assert (decided["disposition"], decided["reason"]) == ("ZERO", "no_novel_asset")
    excluded = classify_result([cue(excluded_non_target="theme_song_shape_with_singing")], scope(),
                               music_activity_found=True)
    assert (excluded["disposition"], excluded["reason"]) == ("ZERO", "music_not_wanted")
