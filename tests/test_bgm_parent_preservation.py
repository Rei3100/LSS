import numpy as np
import pytest

from lss.bgm.parent_preservation import preserve_parent
from tests.test_bgm_dominance_routing import stems


def test_weakening_side_or_right_preserves_all_parent_stems_without_mutation():
    parent = stems()
    original = {role: value.copy() for role, value in parent.items()}
    candidate = stems(0.2, 0.3, 0.4)
    candidate["music"][1] = parent["music"][1] * 0.01
    chosen, evidence = preserve_parent(parent, candidate)
    assert not evidence["enabled"]
    assert any(row["component"] == "right" and not row["accepted"] for row in evidence["observations"])
    for role in parent:
        np.testing.assert_array_equal(chosen[role], original[role])
        np.testing.assert_array_equal(parent[role], original[role])
    chosen["music"][0, 0] = 10
    np.testing.assert_array_equal(parent["music"], original["music"])


def test_preserving_all_music_components_can_use_candidate_but_does_not_prove_quality():
    parent, candidate = stems(), stems(0.11, 0.03, 0.02)
    chosen, evidence = preserve_parent(parent, candidate)
    assert evidence["enabled"]
    assert "not_target_preservation" in evidence["claim_boundary"]
    for role in parent:
        np.testing.assert_array_equal(chosen[role], candidate[role])


def test_zero_parent_component_does_not_accept_new_phantom_component():
    parent, candidate = stems(0), stems(0.1)
    chosen, evidence = preserve_parent(parent, candidate)
    assert not evidence["enabled"]
    np.testing.assert_array_equal(chosen["music"], 0)


@pytest.mark.parametrize("candidate", [stems(float("nan")), stems(frames=499), stems(1e200)])
def test_bad_or_unrepresentable_candidate_is_rejected(candidate):
    with pytest.raises(ValueError):
        preserve_parent(stems(), candidate)


def test_infinite_diagnostic_ratio_is_not_saved_as_acceptance():
    with pytest.raises(ValueError, match="相対診断"):
        preserve_parent(stems(1e-154), stems(1e150))


def test_parent_consistency_can_still_hide_component_loss_and_requires_reference_gates():
    from lss.validation.song_component_comparison import component_regressions, wanted_components
    from tests.test_song_component_comparison import signals

    accompaniment, singing, noise = signals()
    parent_music = accompaniment + singing + noise
    candidate_music = accompaniment * 0.01 + singing + noise * 1.02
    parent = {role: parent_music if role == "music" else np.zeros_like(parent_music) for role in ("music", "speech", "effects")}
    candidate = {role: candidate_music if role == "music" else np.zeros_like(candidate_music) for role in parent}
    chosen, evidence = preserve_parent(parent, candidate)
    assert evidence["enabled"]
    failures = component_regressions(wanted_components(parent_music, accompaniment, singing),
                                     wanted_components(chosen["music"], accompaniment, singing))
    assert failures[0]["component"] == "accompaniment"
