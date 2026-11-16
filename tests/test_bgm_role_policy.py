"""観測から役割を導く採用policyを、人手のラベル無しで固定する。"""
from __future__ import annotations

import pytest

from lss.bgm.role_policy import RolePolicy, RolePolicyError, decide_role, decide_roles


def policy(**changes) -> RolePolicy:
    base = {"singing_present_role": "singing_score",
            "singing_absent_role": "dramatic_score",
            "unresolved_role": "unresolved_music_like",
            "indistinguishable_roles": ("source_music", "musical_sfx")}
    return RolePolicy(**{**base, **changes})


def test_observed_singing_is_taken_as_a_singing_score():
    read = decide_role({"singing": "present"}, policy())

    assert read["semantic_role"] == "singing_score"
    assert read["reason"] == "observed_singing"
    assert read["derived_by_policy"] is True
    assert read["policy_id"].startswith("bgm-role-policy-v1-")


def test_no_observed_singing_is_taken_as_a_dramatic_score():
    read = decide_role({"singing": "absent"}, policy())

    assert read["semantic_role"] == "dramatic_score"
    assert read["reason"] == "no_observed_singing"
    # 区別できない役割を隠さない。
    assert read["indistinguishable_roles"] == ["source_music", "musical_sfx"]


@pytest.mark.parametrize("observations,reason", [
    ({}, "singing_not_observed"),
    (None, "singing_not_observed"),
    ({"singing": "unresolved"}, "singing_observation_unresolved"),
])
def test_without_a_decided_observation_the_role_stays_unresolved(observations, reason):
    read = decide_role(observations, policy())

    assert read["semantic_role"] == "unresolved_music_like"
    assert read["reason"] == reason


def test_every_candidate_gets_its_own_reading():
    decided = decide_roles({"a": {"singing": "absent"}, "b": {"singing": "present"},
                            "c": {}}, policy())

    assert {key: row["semantic_role"] for key, row in decided.items()} == {
        "a": "dramatic_score", "b": "singing_score", "c": "unresolved_music_like"}


@pytest.mark.parametrize("changes", [
    {"singing_present_role": "unresolved_music_like"},
    {"singing_absent_role": "not_a_role"},
    {"unresolved_role": "dramatic_score"},
    {"indistinguishable_roles": ()},
    {"indistinguishable_roles": ("source_music", "source_music")},
    {"indistinguishable_roles": ("unresolved_music_like",)},
    {"schema_version": 2},
    {"method": "other"},
])
def test_an_unsupported_declaration_is_refused(changes):
    with pytest.raises(RolePolicyError):
        policy(**changes).validate()


@pytest.mark.parametrize("observations", ["singing", {"singing": "maybe"}])
def test_a_malformed_observation_is_refused(observations):
    with pytest.raises(RolePolicyError):
        decide_role(observations, policy())


def test_candidate_ids_are_required():
    with pytest.raises(RolePolicyError):
        decide_roles({" ": {"singing": "absent"}}, policy())
