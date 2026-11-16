"""Brief effects require independent, coincident support rather than a shorter global gate."""
import numpy as np
import pytest

from lss.sound import audioset, verify


def decide(*, parent_start=5, output_indices=(5, 6, 7), classes=("Glass",), family="sound", audible=True,
           allow_brief_extraction=True):
    rate = 1000
    audio = np.zeros((rate * 2, 2), dtype=np.float32)
    if audible:
        audio[100:300] = .1
    tags = np.zeros((2, len(audioset.LABELS)), dtype=np.float32)
    if classes:
        tags[0, audioset.INDEX[classes[0]]] = .5
    output = np.zeros(50, dtype=np.float32)
    parent = np.zeros(50, dtype=np.float32)
    output[list(output_indices)] = .9
    parent[parent_start:parent_start + 3] = .9
    return verify.decide(audio, audio, rate, classes=classes, family=family, tags=tags if classes else None,
                         frames=verify.Frames(output=output, parent=parent),
                         allow_brief_extraction=allow_brief_extraction)


def test_short_effect_needs_coincident_parent_output_and_class_support():
    result = decide()
    assert result.present
    assert result.reasons == ["transient_corroborated_in_parent_and_output"]
    assert result.keep[0] and not result.keep[1]


@pytest.mark.parametrize("kwargs", [
    {"parent_start": 25},
    {"output_indices": (5, 7, 9)},
    {"output_indices": (5, 6)},
    {"classes": ()},
    {"family": "speech"},
    {"family": "singing"},
    {"family": "music_part"},
    {"audible": False},
    {"allow_brief_extraction": False},
])
def test_short_effect_cannot_bypass_other_evidence_or_semantic_boundaries(kwargs):
    assert not decide(**kwargs).present


def test_short_probability_array_cannot_be_lengthened_by_convolution():
    frames = verify.Frames(output=np.ones(2), parent=np.ones(2))
    assert not verify._transient_heard(frames, np.array([True]))


def test_transient_rule_is_bound_into_recipe_policy():
    policy = verify.policy_revision()
    assert policy["frames"]["transient_present_frames"] == 3
    assert policy["frames"]["transient_authority"] == "extraction_only"
