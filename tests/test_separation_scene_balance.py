"""二つのstemの釣り合いから場面の種類を読むことを固定する。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.separation.scene_balance import (
    SceneBalanceError,
    SceneBalanceRecipe,
    read_balance,
)


def recipe(**changes) -> SceneBalanceRecipe:
    base = {"absent_below_db": -40.0, "quiet_below_db": -3.0, "dominant_above_db": 3.0}
    return SceneBalanceRecipe(**{**base, **changes})


def _pair(target_db: float, frames: int = 4410):
    rng = np.random.default_rng(0)
    other = rng.standard_normal((2, frames))
    target = rng.standard_normal((2, frames)) * (10.0 ** (target_db / 20.0))
    return target, other


@pytest.mark.parametrize("target_db,state", [
    (20.0, "target_dominant"),
    (0.0, "balanced"),
    (-12.0, "target_quiet"),
    (-60.0, "target_absent"),
])
def test_the_balance_is_placed_in_the_declared_band(target_db, state):
    target, other = _pair(target_db)

    read = read_balance(target, other, recipe())

    assert read["state"] == state
    assert read["is_observation_not_quality"] is True
    assert read["recipe_id"].startswith("scene-balance-v1-")


def test_a_silent_other_stem_makes_the_target_dominant():
    target, _ = _pair(0.0)

    read = read_balance(target, np.zeros_like(target), recipe())

    assert read["state"] == "target_dominant"
    assert read["reason"] == "the_other_stem_is_silent"


def test_a_silent_target_is_absent_not_quiet():
    _, other = _pair(0.0)

    read = read_balance(np.zeros_like(other), other, recipe())

    assert read["state"] == "target_absent"
    assert read["reason"] == "the_target_stem_is_silent"


def test_two_silent_stems_decide_nothing():
    silence = np.zeros((2, 100))

    read = read_balance(silence, silence, recipe())

    assert read["state"] == "unresolved"
    assert read["balance_db"] is None


@pytest.mark.parametrize("changes", [
    {"absent_below_db": 0.0}, {"quiet_below_db": -50.0}, {"dominant_above_db": -10.0},
    {"schema_version": 2}, {"method": "other"},
])
def test_an_unsupported_declaration_is_refused(changes):
    with pytest.raises(SceneBalanceError):
        recipe(**changes).validate()


@pytest.mark.parametrize("target,other", [
    (np.zeros((2, 10)), np.zeros((2, 5))),
    (np.zeros((0, 10)), np.zeros((0, 10))),
    (np.full((2, 10), np.inf), np.zeros((2, 10))),
])
def test_a_malformed_reading_is_refused(target, other):
    with pytest.raises(SceneBalanceError):
        read_balance(target, other, recipe())


def test_the_worker_records_the_scene_without_extra_inference(tmp_path):
    import soundfile as sf

    from lss.core import worker

    rng = np.random.default_rng(3)
    frames = 44100
    music = rng.standard_normal((frames, 2)) * 0.02
    voice = rng.standard_normal((frames, 2)) * 0.4
    mix = tmp_path / "mix.wav"
    estimate = tmp_path / "estimate.wav"
    sf.write(mix, music + voice, 44100, subtype="FLOAT")
    sf.write(estimate, music, 44100, subtype="FLOAT")

    read = worker._scene_balance(mix, [estimate])

    assert read["state"] == "target_quiet"
    assert read["recipe_id"] == worker.SCENE_BALANCE_RECIPE.recipe_id
    # 観測のための読み込み条件を記録に残す。
    assert read["read"]["channels"] == 2 and read["read"]["sample_rate"] == 44100


def test_the_worker_leaves_no_observation_when_it_cannot_read(tmp_path):
    from lss.core import worker

    missing = tmp_path / "missing.wav"

    assert worker._scene_balance(missing, [missing]) is None
    assert worker._scene_balance(missing, []) is None


def test_the_product_result_carries_the_scene_observation():
    from lss import application

    outcome = application.outcome_from_worker_result({
        "status": "ok", "outputs": ["out.flac"], "recipe_id": "r1",
        "scene_balance": {"state": "target_quiet", "reason": "below_the_declared_quiet_boundary",
                          "recipe_id": "scene-balance-v1-test", "balance_db": -12.0}})

    assert outcome.scene_balance["state"] == "target_quiet"
    assert outcome.scene_balance["balance_db"] == -12.0


@pytest.mark.parametrize("value", [
    None, "target_quiet", {"state": "unknown_state", "recipe_id": "x"},
    {"state": "target_quiet"}, {"state": "target_quiet", "recipe_id": "x", "balance_db": "low"},
])
def test_a_malformed_scene_observation_is_dropped(value):
    from lss import application

    outcome = application.outcome_from_worker_result({
        "status": "ok", "outputs": ["out.flac"], "recipe_id": "r1", "scene_balance": value})

    assert outcome.scene_balance is None


def timed(**changes):
    from lss.separation.scene_balance import TimedSceneBalanceRecipe

    base = {"window_seconds": 1.0, "hop_seconds": 0.5, "absent_below_db": -40.0,
            "quiet_below_db": -3.0, "dominant_above_db": 3.0}
    return TimedSceneBalanceRecipe(**{**base, **changes})


def test_a_long_source_reports_how_long_it_stays_in_each_band():
    from lss.separation.scene_balance import read_balance_over_time

    rng = np.random.default_rng(5)
    frames = 44100 * 4
    other = rng.standard_normal((2, frames)) * 0.2
    target = np.concatenate([rng.standard_normal((2, frames // 2)) * 0.01,
                             rng.standard_normal((2, frames // 2)) * 1.0], axis=1)

    read = read_balance_over_time(target, other, 44100, timed())

    assert read["window_count"] > 1
    assert set(read["state_shares"]) == {"target_quiet", "target_dominant"}
    assert abs(sum(read["state_shares"].values()) - 1.0) < 1e-9
    assert read["recipe_id"].startswith("scene-balance-timed-v1-")


def test_a_steady_source_stays_in_one_band():
    from lss.separation.scene_balance import read_balance_over_time

    rng = np.random.default_rng(6)
    frames = 44100 * 3
    other = rng.standard_normal((2, frames)) * 0.2
    target = rng.standard_normal((2, frames)) * 0.01

    read = read_balance_over_time(target, other, 44100, timed())

    assert read["state_shares"] == {"target_quiet": 1.0}


@pytest.mark.parametrize("changes", [
    {"window_seconds": 0.0}, {"hop_seconds": 0.0}, {"hop_seconds": 2.0},
    {"quiet_below_db": -50.0}, {"schema_version": 2}, {"method": "other"},
])
def test_an_unsupported_timed_declaration_is_refused(changes):
    with pytest.raises(SceneBalanceError):
        timed(**changes).validate()


def test_a_source_shorter_than_the_window_is_refused():
    from lss.separation.scene_balance import read_balance_over_time

    short = np.zeros((2, 100))
    with pytest.raises(SceneBalanceError):
        read_balance_over_time(short, short, 44100, timed())


def test_a_long_job_also_records_the_time_shares(tmp_path):
    import soundfile as sf

    from lss.core import worker

    rng = np.random.default_rng(11)
    frames = 44100 * 30
    voice = rng.standard_normal((frames, 2)) * 0.3
    music = np.concatenate([rng.standard_normal((frames // 2, 2)) * 0.01,
                            rng.standard_normal((frames // 2, 2)) * 1.0])
    mix, estimate = tmp_path / "mix.wav", tmp_path / "estimate.wav"
    sf.write(mix, music + voice, 44100, subtype="FLOAT")
    sf.write(estimate, music, 44100, subtype="FLOAT")

    read = worker._scene_balance(mix, [estimate])

    assert set(read["over_time"]["state_shares"]) <= set(
        ("target_quiet", "balanced", "target_dominant", "target_absent", "unresolved"))
    assert read["over_time"]["window_count"] > 1
    assert read["over_time"]["recipe_id"] == worker.SCENE_BALANCE_TIMED_RECIPE.recipe_id


def test_a_short_job_keeps_the_file_observation_without_time_shares(tmp_path):
    import soundfile as sf

    from lss.core import worker

    rng = np.random.default_rng(12)
    frames = 44100
    voice = rng.standard_normal((frames, 2)) * 0.3
    music = rng.standard_normal((frames, 2)) * 0.01
    mix, estimate = tmp_path / "mix.wav", tmp_path / "estimate.wav"
    sf.write(mix, music + voice, 44100, subtype="FLOAT")
    sf.write(estimate, music, 44100, subtype="FLOAT")

    read = worker._scene_balance(mix, [estimate])

    assert read["state"] == "target_quiet"


def test_the_product_result_carries_the_time_shares():
    from lss import application

    outcome = application.outcome_from_worker_result({
        "status": "ok", "outputs": ["out.flac"], "recipe_id": "r1",
        "scene_balance": {"state": "target_quiet", "reason": "x",
                          "recipe_id": "scene-balance-v1-test", "balance_db": -12.0,
                          "over_time": {"recipe_id": "scene-balance-timed-v1-test",
                                        "window_count": 4,
                                        "state_shares": {"target_quiet": 0.75,
                                                         "balanced": 0.25},
                                        "median_balance_db": -9.0}}})

    assert outcome.scene_balance["over_time"]["state_shares"]["target_quiet"] == 0.75


@pytest.mark.parametrize("timed", [
    {"recipe_id": "x"},
    {"recipe_id": "x", "state_shares": {"unknown": 1.0}},
    {"state_shares": {"target_quiet": 1.0}},
    "over_time",
])
def test_a_malformed_time_share_is_dropped(timed):
    from lss import application

    outcome = application.outcome_from_worker_result({
        "status": "ok", "outputs": ["out.flac"], "recipe_id": "r1",
        "scene_balance": {"state": "target_quiet", "reason": "x",
                          "recipe_id": "scene-balance-v1-test", "balance_db": -12.0,
                          "over_time": timed}})

    assert "over_time" not in outcome.scene_balance


def test_a_48k_input_is_observed_without_a_rate_branch(tmp_path):
    import soundfile as sf

    from lss.core import worker

    rng = np.random.default_rng(21)
    frames = 48000 * 2
    voice = rng.standard_normal((frames, 2)) * 0.3
    music = rng.standard_normal((frames, 2)) * 0.01
    mix = tmp_path / "mix.wav"
    sf.write(mix, music + voice, 48000, subtype="FLOAT")
    estimate = tmp_path / "estimate.wav"
    sf.write(estimate, music[: 44100 * 2], 44100, subtype="FLOAT")

    read = worker._scene_balance(mix, [estimate])

    assert read["state"] == "target_quiet"
    assert read["read"]["mixture_resampled"]["from_rate"] == 48000


def test_the_saved_headroom_is_restored_before_comparing(tmp_path):
    import soundfile as sf
    from mutagen.flac import FLAC

    from lss.separation import observation_audio

    rng = np.random.default_rng(31)
    frames = 44100
    music = rng.standard_normal((frames, 2)) * 0.2
    voice = rng.standard_normal((frames, 2)) * 0.2
    mix = tmp_path / "mix.wav"
    sf.write(mix, music + voice, 44100, subtype="FLOAT")
    saved = tmp_path / "estimate.flac"
    # 保存時に半分へ落とした出力。倍率を戻さないと小さく見える。
    sf.write(saved, music / 2.0, 44100, subtype="PCM_24")
    tags = FLAC(saved)
    tags["LSS_OUTPUT_HEADROOM_SHIFT"] = ["1"]
    tags.save()

    plain = observation_audio.segment_energies(mix, saved, segment_seconds=0.5)
    restored = observation_audio.segment_energies(mix, saved, segment_seconds=0.5,
                                                  estimate_gain=2.0)

    plain_energy = sum(row["estimate_energy"] for row in plain["segments"])
    restored_energy = sum(row["estimate_energy"] for row in restored["segments"])
    assert restored_energy == pytest.approx(plain_energy * 4.0, rel=1e-9)
    assert restored["revision"]["estimate_gain"] == 2.0


def test_a_video_input_is_observed_through_the_prepared_signal(tmp_path):
    import soundfile as sf

    from lss.core import audio_input, worker

    rng = np.random.default_rng(32)
    frames = 44100 * 2
    music = rng.standard_normal((frames, 2)) * 0.01
    voice = rng.standard_normal((frames, 2)) * 0.3
    mix = tmp_path / "mix.wav"
    sf.write(mix, music + voice, 44100, subtype="FLOAT")
    estimate = tmp_path / "estimate.wav"
    sf.write(estimate, music, 44100, subtype="FLOAT")

    with audio_input.prepare_input(mix) as prepared:
        read = worker._scene_balance(prepared, [estimate])

    assert read["state"] == "target_quiet"
