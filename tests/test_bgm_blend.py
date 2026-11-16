import numpy as np
import pytest

from lss.bgm.blend import StereoBlendRecipe, blend_stereo


def test_common_strength_preserves_timeline_inputs_and_channel_relation():
    parent = np.array([[0.1, 0.2, 0.4], [-0.2, -0.4, -0.8]])
    child = parent * 0.5
    before = parent.copy(), child.copy()
    blended = blend_stereo(parent, child, StereoBlendRecipe(0.5))
    np.testing.assert_allclose(blended, parent * 0.75)
    np.testing.assert_allclose(blended[1], -2 * blended[0])
    np.testing.assert_array_equal(parent, before[0])
    np.testing.assert_array_equal(child, before[1])


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_zero_target_remains_zero_and_strength_is_part_of_identity(strength):
    zero = np.zeros((2, 17))
    np.testing.assert_array_equal(blend_stereo(zero, zero, StereoBlendRecipe(strength)), zero)
    if strength != 0.5:
        assert StereoBlendRecipe(strength).recipe_id != StereoBlendRecipe(0.5).recipe_id


@pytest.mark.parametrize("strength", [-0.1, 1.1, True, float("inf"), float("nan")])
def test_invalid_blend_is_rejected(strength):
    with pytest.raises(ValueError):
        StereoBlendRecipe(strength).validate()


def test_blend_rejects_different_timelines_and_nonfinite_audio():
    for child in (np.zeros((2, 16)), np.full((2, 17), np.nan)):
        with pytest.raises(ValueError):
            blend_stereo(np.zeros((2, 17)), child, StereoBlendRecipe(0.5))
