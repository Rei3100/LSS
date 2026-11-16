import numpy as np
import pytest

from lss.bgm.four_stem_residual import FourStemResidualRecipe, recover_four_stem_residual


def stems(shape, **changes):
    result = {role:np.zeros(shape) for role in
              ("music_instrumental", "music_vocals", "speech", "sfx")}
    result.update(changes)
    return result


def test_positive_vocal_residual_is_half_recovered_with_one_common_stereo_gain():
    vocal = np.array([[1., -1., 0.5, -0.5], [0.25, -0.25, 0.125, -0.125]])
    base = np.full((2, 4), 0.1)
    result, observed = recover_four_stem_residual(base + vocal, base,
        stems(base.shape, music_vocals=vocal),
        FourStemResidualRecipe("music_vocals", window_frames=4, hop_frames=2))
    assert np.allclose(result, base + 0.5 * vocal)
    assert observed["accepted_window_count"] == 1
    assert observed["source_error_after"] < observed["source_error_before"]
    assert observed["publication_allowed"] is False


def test_absent_or_opposite_basis_preserves_parent_exactly():
    source = np.array([[1., 2., 3.], [-1., -2., -3.]])
    base = np.zeros_like(source)
    result, observed = recover_four_stem_residual(source, base,
        stems(source.shape, music_vocals=-source),
        FourStemResidualRecipe("music_vocals", window_frames=3, hop_frames=2))
    assert np.array_equal(result, base)
    assert observed["accepted_window_count"] == 0


def test_explained_fraction_gate_rejects_small_aligned_leak():
    source = np.vstack([np.ones(8), np.zeros(8)])
    basis = np.vstack([np.full(8, 0.01), np.zeros(8)])
    result, observed = recover_four_stem_residual(source, np.zeros_like(source),
        stems(source.shape, music_vocals=basis),
        FourStemResidualRecipe("music_vocals", window_frames=8, hop_frames=4,
                               minimum_explained_fraction=0.1))
    assert np.array_equal(result, np.zeros_like(source))
    assert observed["windows"][0]["alpha"] == 1
    assert observed["windows"][0]["explained_fraction"] < 0.1


def test_combined_basis_and_overlaps_keep_timeline_finite():
    rng = np.random.default_rng(23)
    source = rng.normal(size=(2, 17))
    estimates = stems(source.shape, music_instrumental=source * 0.4,
                      music_vocals=source * 0.2)
    result, observed = recover_four_stem_residual(source, np.zeros_like(source), estimates,
        FourStemResidualRecipe("combined_music", window_frames=7, hop_frames=3))
    assert result.shape == source.shape and np.isfinite(result).all()
    assert observed["window_count"] == 5
    assert observed["source_error_after"] <= observed["source_error_before"]


def test_channel_gate_rejects_basis_that_only_explains_left_residual():
    source = np.array([[1., -1., 1., -1.], [-0.2, 0.2, -0.2, 0.2]])
    basis = np.array([[1., -1., 1., -1.], [0.2, -0.2, 0.2, -0.2]])
    parent = np.zeros_like(source)
    ungated, before = recover_four_stem_residual(source, parent,
        stems(source.shape, music_vocals=basis),
        FourStemResidualRecipe("music_vocals", blend=0.25, window_frames=4, hop_frames=2))
    gated, after = recover_four_stem_residual(source, parent,
        stems(source.shape, music_vocals=basis), FourStemResidualRecipe(
            "music_vocals", blend=0.25, window_frames=4, hop_frames=2,
            minimum_channel_explained_fraction=0.1))
    assert before["windows"][0]["accepted"] is True
    assert before["windows"][0]["channel_explained_fraction"][1] == 0
    assert not np.array_equal(ungated, parent)
    assert np.array_equal(gated, parent)
    assert after["accepted_window_count"] == 0


@pytest.mark.parametrize("recipe", [
    FourStemResidualRecipe("unknown"), FourStemResidualRecipe("music_vocals", blend=0),
    FourStemResidualRecipe("music_vocals", hop_frames=0),
    FourStemResidualRecipe("music_vocals", minimum_explained_fraction=1.1),
    FourStemResidualRecipe("music_vocals", minimum_channel_explained_fraction=1.1)])
def test_invalid_recipe_is_rejected(recipe):
    with pytest.raises(ValueError, match="Recipe"):
        recipe.to_dict()


def test_missing_stem_and_nonfinite_input_are_rejected():
    value = np.zeros((2, 4))
    with pytest.raises(ValueError, match="4stem"):
        recover_four_stem_residual(value, value, {"music_vocals":value},
                                   FourStemResidualRecipe("music_vocals"))
    broken = value.copy(); broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        recover_four_stem_residual(broken, value, stems(value.shape),
                                   FourStemResidualRecipe("music_vocals"))
