import numpy as np
import pytest

from lss.bgm.dominance_routing import DominanceRoutingRecipe, route_stereo


def stems(music=0.1, speech=0.0, effects=0.0, frames=500):
    return {name: np.tile([[value], [value / 2]], (1, frames))
            for name, value in (("music", music), ("speech", speech), ("effects", effects))}


def test_music_dominance_routes_all_stems_and_channels_with_same_weight():
    parent = stems()
    child = stems(0.2, 0.01, 0.02)
    original = {name: values.copy() for name, values in parent.items()}
    result = route_stereo(parent, child, stems(0.01), DominanceRoutingRecipe(9))
    for name in parent:
        np.testing.assert_array_equal(result.stems[name], child[name])
        np.testing.assert_array_equal(parent[name], original[name])
    np.testing.assert_array_equal(result.dominant_weight, 1)
    np.testing.assert_array_equal(result.mixed_weight, 0)


def test_mixed_scene_routes_to_mixed_child_and_missing_branch_preserves_parent():
    parent = stems(0.01, 0.1, 0.1)
    mixed = stems(0.02, 0.09, 0.09)
    result = route_stereo(parent, stems(0.3), mixed, DominanceRoutingRecipe(9))
    np.testing.assert_array_equal(result.mixed_weight, 1)
    for name in parent:
        np.testing.assert_array_equal(result.stems[name], mixed[name])
    result = route_stereo(parent, stems(0.3), None, DominanceRoutingRecipe(9))
    np.testing.assert_array_equal(result.stems['music'], parent['music'])


def test_absent_or_inaudibly_small_music_preserves_parent_even_with_phantom_children():
    for parent in (stems(0, 0, 0), stems(0.00001, 0.1, 0.1)):
        result = route_stereo(parent, stems(0.5), stems(0.5), DominanceRoutingRecipe(9))
        np.testing.assert_array_equal(result.parent_weight, 1)
        np.testing.assert_array_equal(result.stems['music'], parent['music'])


def test_transition_uses_convex_weights_and_preserves_common_timing():
    parent = stems(0.1, 0.04)
    parent['speech'][:, 250:] *= 10
    dominant, mixed = stems(0.2), stems(-0.1)
    result = route_stereo(parent, dominant, mixed, DominanceRoutingRecipe(51))
    np.testing.assert_allclose(result.parent_weight + result.dominant_weight + result.mixed_weight, 1, atol=1e-15)
    assert np.all(result.parent_weight >= -1e-15)
    assert np.any((result.dominant_weight > 0) & (result.dominant_weight < 1))
    np.testing.assert_allclose(result.stems['music'][1], result.stems['music'][0] / 2)
    assert result.stems['music'].shape == (2, 500)


@pytest.mark.parametrize('definition', [{'energy_window_frames': 2}, {'lower_music_share': 0.99},
                                        {'minimum_music_rms': 1}, {'upper_music_share': float('nan')},
                                        {'energy_window_frames': True}])
def test_invalid_recipe_is_rejected(definition):
    with pytest.raises(ValueError):
        DominanceRoutingRecipe(**definition).validate()


def test_child_layout_nonfinite_or_incomplete_output_is_rejected():
    parent = stems()
    for child in (stems(frames=499), {'music': parent['music']}, stems(music=float('nan'))):
        with pytest.raises(ValueError):
            route_stereo(parent, child, None, DominanceRoutingRecipe(9))


def test_identity_includes_routing_thresholds_and_window():
    assert len({DominanceRoutingRecipe().recipe_id, DominanceRoutingRecipe(9).recipe_id,
                DominanceRoutingRecipe(lower_music_share=0.75).recipe_id}) == 3


def test_finite_input_with_unrepresentable_energy_is_rejected_before_invalid_output():
    parent = stems(music=1e200)
    assert all(np.isfinite(values).all() for values in parent.values())
    with pytest.raises(ValueError, match="energy集計"):
        route_stereo(parent, stems(), None, DominanceRoutingRecipe(9))
