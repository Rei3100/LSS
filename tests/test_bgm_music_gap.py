"""候補の内側で音楽が止まる箇所を、宣言した長さと下限だけで読む。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm.music_gap import (
    MusicDipRecipe,
    MusicGapError,
    MusicGapRecipe,
    extent_from_gaps,
    find_interior_gaps,
    find_relative_dips,
)

RATE = 8000


def recipe(**changes) -> MusicGapRecipe:
    base = {"window_seconds": 0.10, "hop_seconds": 0.025, "floor_dbfs": -60.0,
            "minimum_gap_seconds": 1.5, "edge_margin_seconds": 0.5}
    return MusicGapRecipe(**{**base, **changes})


def signal(*blocks) -> np.ndarray:
    mono = np.concatenate([np.full(int(seconds * RATE), level, dtype=np.float64)
                           for seconds, level in blocks])
    return np.stack([mono, mono])


def test_a_stop_inside_the_candidate_is_reported_with_its_span():
    read = find_interior_gaps(signal((10.0, 0.2), (2.0, 0.0), (10.0, 0.2)), RATE, recipe())

    assert len(read["interior_gaps"]) == 1
    gap = read["interior_gaps"][0]
    assert gap["start_seconds"] == pytest.approx(10.0, abs=0.1)
    assert gap["end_seconds"] == pytest.approx(12.0, abs=0.1)
    assert gap["minimum_dbfs"] == -200.0
    assert read["recipe_id"].startswith("bgm-music-gap-v1-")


def test_a_stop_shorter_than_the_declared_length_is_not_reported():
    read = find_interior_gaps(signal((10.0, 0.2), (1.0, 0.0), (10.0, 0.2)), RATE, recipe())

    assert read["interior_gaps"] == []


def test_the_quiet_edges_of_a_candidate_are_not_stops():
    read = find_interior_gaps(signal((2.0, 0.0), (10.0, 0.2), (2.0, 0.0)), RATE, recipe())

    assert read["interior_gaps"] == []


def test_a_quiet_passage_above_the_declared_floor_is_not_a_stop():
    # -46 dBFSは宣言した下限-60より上、-40より下。
    quiet = signal((10.0, 0.2), (2.0, 0.005), (10.0, 0.2))

    assert find_interior_gaps(quiet, RATE, recipe())["interior_gaps"] == []
    assert find_interior_gaps(quiet, RATE, recipe(floor_dbfs=-40.0))["interior_gaps"]


def test_every_stop_is_kept_not_only_the_first():
    read = find_interior_gaps(
        signal((5.0, 0.2), (2.0, 0.0), (5.0, 0.2), (2.0, 0.0), (5.0, 0.2)), RATE, recipe())

    assert len(read["interior_gaps"]) == 2


def test_a_stop_makes_the_candidate_more_than_one_cue():
    read = find_interior_gaps(signal((10.0, 0.2), (2.0, 0.0), (10.0, 0.2)), RATE, recipe())
    extent = extent_from_gaps(read)

    assert extent["state"] == "multiple_cues"
    assert extent["evidence"]["reason"] == "music_stops_inside"
    assert extent["evidence"]["interior_gap_count"] == 1


def test_no_stop_leaves_the_extent_unresolved_rather_than_one_cue():
    """止まりが無いことは、一曲であることの証明ではない。"""
    extent = extent_from_gaps(find_interior_gaps(signal((20.0, 0.2)), RATE, recipe()))

    assert extent["state"] == "unresolved"
    assert extent["evidence"]["reason"] == "no_interior_music_gap"


@pytest.mark.parametrize("changes", [
    {"window_seconds": 0.0},
    {"hop_seconds": 1.0},
    {"floor_dbfs": 0.0},
    {"minimum_gap_seconds": -1.0},
    {"edge_margin_seconds": -0.1},
    {"method": "music-estimate-interior-gap"},
])
def test_rejects_unusable_declarations(changes):
    with pytest.raises(MusicGapError):
        recipe(**changes).validate()


def test_rejects_audio_that_cannot_be_read():
    with pytest.raises(MusicGapError, match="有限"):
        find_interior_gaps(np.array([[0.1, np.nan, 0.1] * 4000]), RATE, recipe())
    with pytest.raises(MusicGapError, match="窓より長い"):
        find_interior_gaps(signal((0.05, 0.2)), RATE, recipe())


def test_rejects_an_observation_without_gaps_recorded():
    with pytest.raises(MusicGapError, match="止まりの観測"):
        extent_from_gaps({"recipe_id": "x"})


def dip_recipe(**changes) -> MusicDipRecipe:
    base = {"window_seconds": 0.10, "hop_seconds": 0.025, "depth_db": 20.0, "rise_db": 10.0,
            "rise_window_seconds": 3.0, "minimum_separation_seconds": 5.0,
            "edge_margin_seconds": 1.0}
    return MusicDipRecipe(**{**base, **changes})


def _decay(seconds, high, low):
    return np.geomspace(high, low, int(seconds * RATE))


def _dip_signal(low=0.004):
    mono = np.concatenate([np.full(10 * RATE, 0.2), _decay(1.0, 0.2, low),
                           np.full(int(0.2 * RATE), low), np.full(10 * RATE, 0.2)])
    return np.stack([mono, mono])


def test_a_decay_into_an_onset_is_found_without_any_silence():
    read = find_relative_dips(_dip_signal(), RATE, dip_recipe())

    assert len(read["dips"]) == 1
    assert read["dips"][0]["seconds"] == pytest.approx(11.0, abs=0.3)
    assert read["dips"][0]["depth_db"] > 20.0


def test_a_shallow_dip_is_not_reported_at_the_declared_depth():
    shallow = _dip_signal(low=0.05)

    assert find_relative_dips(shallow, RATE, dip_recipe())["dips"] == []
    assert find_relative_dips(shallow, RATE, dip_recipe(depth_db=6.0))["dips"]


def test_a_decay_that_never_comes_back_is_not_a_boundary():
    mono = np.concatenate([np.full(10 * RATE, 0.2), _decay(1.0, 0.2, 0.004),
                           np.full(10 * RATE, 0.004)])

    assert find_relative_dips(np.stack([mono, mono]), RATE, dip_recipe())["dips"] == []


def test_dips_keep_the_declared_separation():
    mono = np.concatenate([
        np.full(6 * RATE, 0.2), np.full(int(0.2 * RATE), 0.004), np.full(2 * RATE, 0.2),
        np.full(int(0.2 * RATE), 0.004), np.full(6 * RATE, 0.2)])
    signal = np.stack([mono, mono])

    assert len(find_relative_dips(signal, RATE, dip_recipe(
        minimum_separation_seconds=1.0))["dips"]) == 2
    assert len(find_relative_dips(signal, RATE, dip_recipe(
        minimum_separation_seconds=5.0))["dips"]) == 1


@pytest.mark.parametrize("changes", [
    {"depth_db": 0.0},
    {"rise_db": -1.0},
    {"rise_window_seconds": 0.0},
    {"minimum_separation_seconds": 0.0},
    {"edge_margin_seconds": -0.1},
    {"method": "music-estimate-relative-dip"},
])
def test_rejects_unusable_dip_declarations(changes):
    with pytest.raises(MusicGapError):
        dip_recipe(**changes).validate()
