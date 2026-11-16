"""参照録音の中の繰り返しを、その録音自身の無関係な組と比べて測る。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm.reference_repetition import (
    ReferenceRepetitionError,
    ReferenceRepetitionRecipe,
    compare_positions,
)
from lss.bgm.spectral_features import (
    SpectralFeatureError,
    SpectralFeatureRecipe,
    feature_count,
    timbre_and_chroma,
)

RATE = 22050
SECTION_SECONDS = 8.0
# 8つの楽節のうち、2番目だけが6番目でもう一度鳴る。
FUNDAMENTALS = (220.0, 277.2, 330.0, 415.3, 494.0, 587.3, 277.2, 698.5)


def feature(**changes) -> SpectralFeatureRecipe:
    base = {"n_fft": 2048, "stft_hop_frames": 512, "feature_seconds": 1.0,
            "feature_hop_seconds": 0.5, "band_count": 16, "low_hz": 55.0, "high_hz": 8000.0}
    return SpectralFeatureRecipe(**{**base, **changes})


def recipe(**changes) -> ReferenceRepetitionRecipe:
    base = {"feature": feature(), "chroma_weight": 0.5, "comparison_seconds": 4.0,
            "null_pair_count": 200, "null_percentile": 95.0, "random_seed": 11}
    return ReferenceRepetitionRecipe(**{**base, **changes})


def _reference() -> np.ndarray:
    time = np.arange(int(SECTION_SECONDS * RATE)) / RATE
    blocks = [0.2 * (np.sin(2 * np.pi * base * time)
                     + 0.4 * np.sin(2 * np.pi * base * 3 * time))
              for base in FUNDAMENTALS]
    mono = np.concatenate(blocks)
    return np.stack([mono, mono])


def _at(section: int) -> float:
    return section * SECTION_SECONDS + 1.0


def test_a_repeated_section_stands_out_against_the_recording_itself():
    read = compare_positions(_reference(), RATE, [
        {"label": "repeat", "first_seconds": _at(1), "second_seconds": _at(6)},
        {"label": "different", "first_seconds": _at(0), "second_seconds": _at(3)},
    ], recipe())

    repeat, different = read["pairs"]
    assert repeat["above_null_threshold"] is True
    assert different["above_null_threshold"] is False
    assert repeat["similarity"] > different["similarity"]
    assert read["recipe_id"].startswith("bgm-reference-repetition-v1-")


def test_the_null_comes_from_the_same_recording_and_is_reproducible():
    first = compare_positions(_reference(), RATE, [], recipe())
    again = compare_positions(_reference(), RATE, [], recipe())
    other = compare_positions(_reference(), RATE, [], recipe(random_seed=12))

    assert first["null"] == again["null"]
    assert first["null"]["pair_count"] == 200
    assert first["null"]["minimum"] <= first["null"]["median"] <= first["null"]["maximum"]
    assert other["null"]["threshold"] == pytest.approx(first["null"]["threshold"], abs=0.2)


def test_a_stricter_percentile_makes_it_harder_to_call_it_a_repeat():
    pairs = [{"label": "different", "first_seconds": _at(0), "second_seconds": _at(3)}]

    loose = compare_positions(_reference(), RATE, pairs, recipe(null_percentile=5.0))
    strict = compare_positions(_reference(), RATE, pairs, recipe(null_percentile=99.0))

    assert loose["null"]["threshold"] < strict["null"]["threshold"]
    assert strict["pairs"][0]["above_null_threshold"] is False


def test_rejects_a_position_outside_the_reference():
    with pytest.raises(ReferenceRepetitionError, match="参照録音の外"):
        compare_positions(_reference(), RATE,
                          [{"first_seconds": 1.0, "second_seconds": 10_000.0}], recipe())


def test_rejects_a_reference_shorter_than_the_comparison():
    short = _reference()[:, :int(3.0 * RATE)]

    with pytest.raises(ReferenceRepetitionError, match="短すぎます"):
        compare_positions(short, RATE, [], recipe())


@pytest.mark.parametrize("changes", [
    {"chroma_weight": 1.5},
    {"comparison_seconds": 0.0},
    {"null_pair_count": 9},
    {"null_percentile": 100.0},
    {"random_seed": -1},
    {"method": "reference-self-similarity"},
])
def test_rejects_unusable_declarations(changes):
    with pytest.raises(ReferenceRepetitionError):
        recipe(**changes).validate()


def test_rejects_a_feature_declaration_of_another_kind():
    with pytest.raises(ReferenceRepetitionError, match="共通の特徴"):
        recipe(feature={"n_fft": 2048}).validate()


def test_rejects_a_pair_that_cannot_be_read():
    for pair in ("not-a-pair", {"first_seconds": None, "second_seconds": 1.0},
                 {"first_seconds": 1.0}):
        with pytest.raises(ReferenceRepetitionError):
            compare_positions(_reference(), RATE, [pair], recipe())


def test_the_feature_count_matches_what_the_features_actually_produce():
    """区間の端を決める側が、音を作らずに最後の位置を知れる。"""
    values = _reference()
    recipe = feature()

    timbre, chroma = timbre_and_chroma(values, RATE, recipe)

    assert feature_count(values.shape[1], RATE, recipe) == timbre.shape[0] == chroma.shape[0]


def test_the_feature_count_is_zero_below_one_window():
    assert feature_count(100, RATE, feature()) == 0


@pytest.mark.parametrize("frames,rate", [(0, RATE), (-1, RATE), (1000, 0)])
def test_the_feature_count_rejects_unusable_lengths(frames, rate):
    with pytest.raises(SpectralFeatureError):
        feature_count(frames, rate, feature())
