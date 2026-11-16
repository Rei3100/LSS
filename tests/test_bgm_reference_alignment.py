"""参照の速度だけを原本から推定し、根拠が弱いときは既存処理を保つ。"""
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import resample_poly

from lss.bgm.reference_alignment import ReferenceAlignmentRecipe, align_reference
from tests.test_bgm_recording_recovery import _match


@pytest.fixture
def alignment_case():
    random = np.random.default_rng(942)
    reference = random.normal(size=(2, 10000)) * 0.05
    source = reference * np.array([0.7, -1.2])[:, None]
    recipe = ReferenceAlignmentRecipe(sample_rate=1000, probe_frames=1000, minimum_probe_frames=100,
                                      maximum_offset_frames=5)
    return source, reference, _match(0, 10, 0), recipe


@pytest.mark.parametrize("up,down,numerator", [(1001, 1000, 1998), (999, 1000, 2002)])
def test_source_alone_selects_the_inverse_speed_with_one_shared_stereo_timeline(alignment_case, up, down, numerator):
    source, reference, match, recipe = alignment_case
    warped = resample_poly(reference, up, down, axis=1)
    snapshots = source.copy(), warped.copy()
    aligned, corrected, observations = align_reference(source, warped, match, recipe)
    assert observations["speed_correction_used"]
    assert observations["speed_numerator"] == numerator
    assert observations["alignment_from_original_only"] and observations["publication_allowed"] is False
    length = min(source.shape[1], aligned.shape[1], warped.shape[1])
    assert np.mean((aligned[:, :length] - reference[:, :length]) ** 2) < np.mean((warped[:, :length] - reference[:, :length]) ** 2) * 0.1
    assert corrected.reference_start_seconds == 0
    assert corrected.query_start_seconds == 0 and corrected.query_end_seconds <= 10
    assert np.array_equal(source, snapshots[0]) and np.array_equal(warped, snapshots[1])


def test_same_speed_keeps_reference_bytes_and_the_original_match(alignment_case):
    source, reference, match, recipe = alignment_case
    aligned, corrected, observations = align_reference(source, reference, match, recipe)
    assert aligned is reference and corrected is match
    assert observations["speed_correction_used"] is False


@pytest.mark.parametrize("kind", ["silence", "unrelated", "short"])
def test_weak_or_short_evidence_does_not_invent_a_reference_speed(alignment_case, kind):
    source, reference, match, recipe = alignment_case
    if kind == "silence":
        source = np.zeros_like(source)
    elif kind == "unrelated":
        source = np.random.default_rng(17).normal(size=source.shape)
    else:
        source, reference, match = source[:, :100], reference[:, :100], _match(0, 0.1, 0)
    aligned, corrected, observations = align_reference(source, reference, match, recipe)
    assert aligned is reference and corrected is match
    assert observations["speed_correction_used"] is False


def test_reference_start_is_transformed_about_its_own_origin(alignment_case):
    source, reference, _, recipe = alignment_case
    warped = resample_poly(reference, 1001, 1000, axis=1)
    aligned, match, observations = align_reference(source, warped, _match(2, 9, 2.002), recipe)
    assert observations["speed_correction_used"] and match.reference_start_seconds == 2
    assert match.query_start_seconds == 2 and match.query_end_seconds == 9
    assert aligned.shape[0] == 2


def test_long_silent_ends_do_not_hide_the_distributed_nonzero_source_probes(alignment_case):
    source, reference, match, recipe = alignment_case
    source, reference = np.pad(source, ((0, 0), (10000, 20000))), np.pad(reference, ((0, 0), (10000, 20000)))
    warped = resample_poly(reference, 1001, 1000, axis=1)
    _, _, observations = align_reference(source, warped, _match(0, 40, 0), recipe)
    assert observations["speed_correction_used"]
    assert len(observations["probe_start_frames"]) == 5
    assert all(10000 <= p < 20000 for p in observations["probe_start_frames"])


@pytest.mark.parametrize("change", [{"speed_numerators": (1998, 2002)}, {"speed_numerators": (2000, 2000)},
                                    {"speed_numerators": (2000, True)}, {"speed_denominator": True},
                                    {"probe_count": 2}, {"maximum_offset_frames": -1},
                                    {"minimum_explained_gain": float("nan")}, {"schema_version": True}])
def test_invalid_alignment_rules_are_rejected(alignment_case, change):
    with pytest.raises(ValueError):
        replace(alignment_case[3], **change).to_dict()


def test_unknown_recording_and_invalid_time_are_rejected(alignment_case):
    source, reference, match, recipe = alignment_case
    for bad in (replace(match, identity_state="unresolved"), _match(0, 11, 0), _match(float("nan"), 1, 0)):
        with pytest.raises(ValueError):
            align_reference(source, reference, bad, recipe)
