"""保存済みの保留結果を固定した対応領域で読み直し、公開しない理由を返す。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lss import application
from lss.bgm.supported_scope import ScopeError, SupportedScope
from tests.test_application_bgm_identity import routed_result, _execute  # noqa: F401
from tests.test_bgm_held_candidate_audition import auditionable_result  # noqa: F401


def scope(**changes) -> SupportedScope:
    base = {"source_domains": ("finished_stereo_episode",),
            "published_semantic_roles": ("dramatic_score",),
            "covered_semantic_roles": ("dramatic_score", "singing_score"),
            "minimum_cue_seconds": 5.0, "maximum_cue_seconds": 600.0,
            "required_evidence": ("source_provenance", "occurrence_span"),
            "quality_gates": {"snr_db": 6.0},
            "singing_observation_policy": "exclude_from_publication",
            "derived_role_policy": "accept_declared_ambiguity",
            "cue_extent_policy": "allow_unresolved_cue_extent",
            "publication_allowed": True}
    return SupportedScope(**{**base, **changes})


def review(case, **kwargs):
    outcome = _execute(case)
    return outcome, application.review_bgm_scope_coverage(
        outcome.record_path, scope(**kwargs.pop("scope_changes", {})),
        source_domain=kwargs.pop("source_domain", "finished_stereo_episode"), **kwargs)


def test_unresolved_candidates_are_held_and_name_the_missing_decision(routed_result, tmp_path):
    outcome, decided = review(routed_result, state_root=tmp_path / "state")
    assert decided["status"] == "REVIEWED_WITHOUT_PUBLISHING"
    assert decided["publication_performed"] is False
    assert decided["disposition"] == "HELD"
    assert {row["reason"] for row in decided["candidates"]} == {"semantic_role_unresolved"}
    assert decided["counts"] == {"HELD": 2, "PUBLISHED": 0, "UNSUPPORTED": 0}
    assert decided["scope"]["publication_allowed"] is True
    assert decided["evidence_path"] == str(outcome.evidence_path)


def test_a_source_outside_the_declared_domain_is_unsupported_not_silently_dropped(routed_result, tmp_path):
    _, decided = review(routed_result, state_root=tmp_path / "state", source_domain="game_capture")
    assert decided["disposition"] == "UNSUPPORTED"
    assert {row["reason"] for row in decided["candidates"]} == {"source_domain_out_of_scope"}


def test_reviewing_changes_nothing_on_disk(routed_result, tmp_path):
    outcome = _execute(routed_result)
    before = {path: path.read_bytes() for path in
              [outcome.record_path, outcome.evidence_path, *outcome.held_artifacts]}
    application.review_bgm_scope_coverage(outcome.record_path, scope(),
                                          source_domain="finished_stereo_episode",
                                          state_root=tmp_path / "state")
    assert {path: path.read_bytes() for path in before} == before


def test_an_unreadable_held_result_is_reported_without_a_scope_decision(routed_result, tmp_path):
    outcome = _execute(routed_result)
    Path(outcome.held_artifacts[0]).unlink()
    decided = application.review_bgm_scope_coverage(outcome.record_path, scope(),
                                                    source_domain="finished_stereo_episode",
                                                    state_root=tmp_path / "state")
    assert decided["status"] == "NO_READABLE_HELD_RESULT"
    assert decided["outcome_disposition"] == "FAILED" and "candidates" not in decided
    assert decided["publication_performed"] is False


@pytest.mark.parametrize("domain", ["", "   ", None, 3])
def test_a_review_without_a_declared_source_domain_is_refused(routed_result, tmp_path, domain):
    outcome = _execute(routed_result)
    with pytest.raises(ScopeError):
        application.review_bgm_scope_coverage(outcome.record_path, scope(), source_domain=domain,
                                              state_root=tmp_path / "state")


def test_every_row_carries_the_scope_it_was_judged_under(routed_result, tmp_path):
    _, decided = review(routed_result, state_root=tmp_path / "state")
    assert {row["scope_id"] for row in decided["candidates"]} == {scope().scope_id}
    assert decided["scope_id"] == scope().scope_id
    development = application.review_bgm_scope_coverage(
        _execute(routed_result).record_path, scope(publication_allowed=False),
        source_domain="finished_stereo_episode", state_root=tmp_path / "state")
    assert development["scope_id"] != decided["scope_id"]
    assert development["publication_allowed"] is False


def test_a_source_reached_through_another_spelling_is_the_same_result(routed_result, tmp_path,
                                                                      monkeypatch):
    """素材を移し、旧所在から別名で辿れる状態でも、保存済み結果は同じものとして読める。"""
    outcome = _execute(routed_result)
    record = json.loads(Path(outcome.record_path).read_bytes())
    stored = Path(record["source"]["path"])
    moved = tmp_path / "moved" / stored.name
    moved.parent.mkdir(parents=True)
    shutil.copyfile(stored, moved)
    real = Path(application.__file__)

    original = Path.resolve

    def resolve(self, strict=False):
        if str(self) == str(stored):
            return original(moved, strict)
        return original(self, strict)

    monkeypatch.setattr(Path, "resolve", resolve)
    assert real.is_file()
    decided = application.review_bgm_scope_coverage(
        outcome.record_path, scope(), source_domain="finished_stereo_episode",
        state_root=tmp_path / "state")
    assert decided["status"] == "REVIEWED_WITHOUT_PUBLISHING"


def readiness_recipe(**changes):
    from lss.bgm.cue_readiness import CueReadinessRecipe

    base = {"required_channels": 2, "minimum_loudness_lufs": -45.0,
            "maximum_true_peak_dbtp": -1.0, "maximum_clipped_samples": 0,
            "minimum_spectral_edge_hz": 4000.0, "spectral_edge_drop_db": 60.0,
            "minimum_side_to_mid_db": -60.0}
    return CueReadinessRecipe(**{**base, **changes})


def test_without_a_readiness_recipe_quality_stays_unmeasured(auditionable_result, tmp_path):
    decided = application.review_bgm_scope_coverage(
        auditionable_result.record_path, scope(published_semantic_roles=("dramatic_score",)),
        source_domain="finished_stereo_episode", state_root=tmp_path / "state")
    assert decided["cue_readiness_recipe_id"] is None
    assert all("readiness" not in row for row in decided["candidates"])


@pytest.mark.parametrize("method", ["cue-publication-readiness-v1", "cue-publication-readiness-v2"])
def test_a_declared_readiness_recipe_measures_every_cue(auditionable_result, tmp_path, method):
    recipe = readiness_recipe(method=method)
    decided = application.review_bgm_scope_coverage(
        auditionable_result.record_path, scope(), source_domain="finished_stereo_episode",
        readiness=recipe, state_root=tmp_path / "state")
    assert decided["cue_readiness_recipe_id"] == recipe.recipe_id
    rows = decided["candidates"]
    assert len(rows) == 2 and all("readiness" in row for row in rows)
    for row in rows:
        assert row["readiness"]["recipe_id"] == recipe.recipe_id
        assert row["readiness"]["method"] == method
        assert isinstance(row["readiness"]["passed"], bool)
        assert any("L4" in item for item in row["readiness"]["not_checked"])
    # 意味的roleが未解決である限り、検査を通っても公開されない。
    assert {row["reason"] for row in rows} == {"semantic_role_unresolved"}


def test_a_quality_gate_can_require_the_readiness_result(auditionable_result, tmp_path, monkeypatch):
    from lss.bgm import supported_scope

    original = supported_scope.classify_candidate
    seen = {}

    def capture(candidate, used):
        seen[candidate["candidate_id"]] = candidate.get("quality")
        return original(candidate, used)

    monkeypatch.setattr(supported_scope, "classify_candidate", capture)
    application.review_bgm_scope_coverage(
        auditionable_result.record_path, scope(quality_gates={"readiness_passed": 1.0}),
        source_domain="finished_stereo_episode", readiness=readiness_recipe(),
        state_root=tmp_path / "state")
    assert seen and all(value is not None and "readiness_passed" in value
                        for value in seen.values())
    assert all(isinstance(value["loudness_lufs"], float) for value in seen.values())


def test_readiness_file_boundary_uses_the_declared_spectral_method(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import soundfile as sf

    rate = 44100
    time = np.arange(rate * 3) / rate
    low = 0.12 * np.sin(2 * np.pi * 440 * time)
    high = 0.08 * np.sin(2 * np.pi * 16000 * time)
    path = tmp_path / "wanted-stereo.wav"
    sf.write(path, np.stack([low + high, low - high], axis=1), rate, subtype="DOUBLE")
    original = path.read_bytes()
    outcome = SimpleNamespace(held_artifacts=[path])
    report = {"candidates": [{"asset": {"candidate_asset_id": "wanted"}}]}
    for method, expected in (("cue-publication-readiness-v1", False),
                             ("cue-publication-readiness-v2", True)):
        recipe = readiness_recipe(method=method, minimum_spectral_edge_hz=8000.0,
                                  minimum_side_to_mid_db=None)
        result = application._cue_readiness(outcome, report, recipe)["wanted"]
        assert result["readiness"]["method"] == method
        assert result["readiness"]["passed"] is expected
        assert result["quality"]["readiness_passed"] == float(expected)
    assert path.read_bytes() == original


def test_measured_observations_move_a_singing_cue_out_of_scope(auditionable_result, tmp_path):
    original = auditionable_result
    first = original.held_artifacts[0].name.split(".")[0]
    ids = [path.name.split(".")[0] for path in original.held_artifacts]
    decided = application.review_bgm_scope_coverage(
        original.record_path, scope(), source_domain="finished_stereo_episode",
        observations={ids[0]: {"singing": "present"}}, state_root=tmp_path / "state")
    assert decided["observed_candidate_count"] == 1
    rows = {row["candidate_id"]: row for row in decided["candidates"]}
    assert rows[ids[0]]["reason"] == "observed_singing_out_of_scope"
    assert rows[ids[0]]["disposition"] == "UNSUPPORTED"
    # 観測のない候補はこれまでどおり意味的roleの未解決で保留のまま。
    assert rows[ids[1]]["reason"] == "semantic_role_unresolved"
    assert first == ids[0]


def test_without_observations_nothing_is_assumed(auditionable_result, tmp_path):
    decided = application.review_bgm_scope_coverage(
        auditionable_result.record_path, scope(), source_domain="finished_stereo_episode",
        state_root=tmp_path / "state")
    assert decided["observed_candidate_count"] == 0
    assert {row["reason"] for row in decided["candidates"]} == {"semantic_role_unresolved"}


def _annotation(tmp_path: Path, children: list[dict]) -> Path:
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({
        "type": "episode_identity_annotation",
        "activity_runs": [{"recipe_id": "renderer+activity", "candidates": children}],
    }, ensure_ascii=False), encoding="utf-8")
    return path


def _report(start_frame: int = 0, end_frame: int = 20 * 11025) -> dict:
    return {"candidates": [{"asset": {"candidate_asset_id": "cue-a"},
                            "origin": {"sample_rate": 11025, "start_frame": start_frame,
                                       "end_frame": end_frame}}]}


def test_a_covering_child_reading_is_lifted_to_the_parent_candidate(tmp_path):
    path = _annotation(tmp_path, [
        {"candidate_id": "child", "start_seconds": 0.0, "end_seconds": 20.0,
         "cue_extent": {"state": "single_cue",
                        "evidence": {"reason": "one_continuous_recurring_recording"}}}])

    extents = application._cue_extents_from_annotation(_report(), path)

    assert extents["cue-a"]["state"] == "single_cue"


def test_a_child_that_does_not_cover_the_parent_decides_nothing(tmp_path):
    path = _annotation(tmp_path, [
        {"candidate_id": "child", "start_seconds": 0.0, "end_seconds": 8.0,
         "cue_extent": {"state": "single_cue", "evidence": {"reason": "x"}}}])

    assert application._cue_extents_from_annotation(_report(), path) is None


def test_two_decided_children_say_the_parent_holds_more_than_one_cue(tmp_path):
    path = _annotation(tmp_path, [
        {"candidate_id": "a", "start_seconds": 0.0, "end_seconds": 9.0,
         "cue_extent": {"state": "single_cue", "evidence": {"reason": "x"}}},
        {"candidate_id": "b", "start_seconds": 9.0, "end_seconds": 20.0,
         "cue_extent": {"state": "single_cue", "evidence": {"reason": "x"}}}])

    extents = application._cue_extents_from_annotation(_report(), path)

    assert extents["cue-a"]["state"] == "multiple_cues"


def test_an_unreadable_annotation_decides_nothing(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text("broken", encoding="utf-8")

    assert application._cue_extents_from_annotation(_report(), path) is None


def test_a_caller_supplied_reading_is_not_replaced(routed_result, tmp_path):
    outcome = _execute(routed_result)

    decided = application.review_bgm_scope_coverage(
        outcome.record_path, scope(), source_domain="finished_stereo_episode",
        cue_extents={}, state_root=tmp_path / "state")

    assert decided["cue_extent_candidate_count"] == 0


def _role_policy():
    from lss.bgm.role_policy import RolePolicy

    return RolePolicy(singing_present_role="singing_score",
                      singing_absent_role="dramatic_score",
                      unresolved_role="unresolved_music_like",
                      indistinguishable_roles=("source_music", "musical_sfx"))


def test_the_role_is_derived_from_observations_without_any_human_label(routed_result, tmp_path):
    outcome = _execute(routed_result)

    decided = application.review_bgm_scope_coverage(
        outcome.record_path, scope(), source_domain="finished_stereo_episode",
        observations={"cue-a": {"singing": "absent"}, "cue-b": {"singing": "present"}},
        role_policy=_role_policy(), state_root=tmp_path / "state")

    roles = {row["candidate_id"]: row["semantic_role"] for row in decided["candidates"]}
    assert roles["cue-a"] == "dramatic_score"
    assert decided["derived_role_candidate_count"] == 2
    assert decided["role_policy_id"].startswith("bgm-role-policy-v1-")
    reasons = {row["candidate_id"]: row["reason"] for row in decided["candidates"]}
    # 役割が決まったので、保留の理由はroleの未解決ではなくなる。
    assert reasons["cue-a"] != "semantic_role_unresolved"
    # 歌唱ありの観測は、役割より先に対応範囲の外へ出る。
    assert reasons["cue-b"] == "observed_singing_out_of_scope"


def test_without_an_observation_the_derived_role_stays_unresolved(routed_result, tmp_path):
    outcome = _execute(routed_result)

    decided = application.review_bgm_scope_coverage(
        outcome.record_path, scope(), source_domain="finished_stereo_episode",
        role_policy=_role_policy(), state_root=tmp_path / "state")

    assert decided["derived_role_candidate_count"] == 0
    reasons = {row["candidate_id"]: row["reason"] for row in decided["candidates"]}
    assert set(reasons.values()) == {"semantic_role_unresolved"}


def test_no_policy_means_no_derived_role(routed_result, tmp_path):
    outcome = _execute(routed_result)

    decided = application.review_bgm_scope_coverage(
        outcome.record_path, scope(), source_domain="finished_stereo_episode",
        observations={"cue-a": {"singing": "absent"}}, state_root=tmp_path / "state")

    assert decided["role_policy_id"] is None
    assert decided["derived_role_candidate_count"] == 0
    reasons = {row["candidate_id"]: row["reason"] for row in decided["candidates"]}
    assert reasons["cue-a"] == "semantic_role_unresolved"


def _annotation_with_groups(tmp_path: Path, children: list[dict], groups: list[dict]) -> Path:
    path = tmp_path / "identity.json"
    path.write_text(json.dumps({
        "type": "episode_identity_annotation",
        "activity_runs": [{"recipe_id": "renderer+activity", "candidates": children,
                           "occurrence_groups": groups}],
    }, ensure_ascii=False), encoding="utf-8")
    return path


def _two_candidate_report() -> dict:
    return {"candidates": [
        {"asset": {"candidate_asset_id": "cue-a"},
         "origin": {"sample_rate": 11025, "start_frame": 0, "end_frame": 20 * 11025}},
        {"asset": {"candidate_asset_id": "cue-b"},
         "origin": {"sample_rate": 11025, "start_frame": 25 * 11025,
                    "end_frame": 40 * 11025}}]}


def test_a_group_across_two_candidates_is_lifted_to_them(tmp_path):
    path = _annotation_with_groups(
        tmp_path,
        [{"candidate_id": "a0", "start_seconds": 0.0, "end_seconds": 20.0},
         {"candidate_id": "b0", "start_seconds": 25.0, "end_seconds": 40.0}],
        [{"occurrence_group_id": "g1", "members": [["ep", "a0"], ["ep", "b0"]],
          "evidence": [{"kind": "shared_recording"}]}])

    groups = application._occurrence_groups_from_annotation(_two_candidate_report(), path)

    assert set(groups) == {"cue-a", "cue-b"}
    assert groups["cue-a"]["members"] == ["cue-a", "cue-b"]
    assert groups["cue-a"]["chosen_candidate_id"] is None


def test_a_group_inside_one_candidate_is_not_a_duplicate(tmp_path):
    path = _annotation_with_groups(
        tmp_path,
        [{"candidate_id": "a0", "start_seconds": 0.0, "end_seconds": 9.0},
         {"candidate_id": "a1", "start_seconds": 9.0, "end_seconds": 20.0}],
        [{"occurrence_group_id": "g1", "members": [["ep", "a0"], ["ep", "a1"]],
          "evidence": [{"kind": "shared_recording"}]}])

    assert application._occurrence_groups_from_annotation(_two_candidate_report(), path) is None


def test_an_unreadable_annotation_reports_no_group(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text("broken", encoding="utf-8")

    assert application._occurrence_groups_from_annotation(_two_candidate_report(), path) is None
