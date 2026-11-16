import json

import numpy as np
import pytest
import soundfile as sf

from lss.bgm.activity import (
    ActivityRecipe,
    InvalidActivityRecipeError,
    RelativeHysteresisRecipe,
    RelativeStemEnergyShareRecipe,
    StemEnergyShareRecipe,
    discover_activity,
    discover_activity_file,
    discover_stem_energy_share_files,
    load_activity_recipe,
    load_discovery_recipe,
)


def _recipe(**overrides):
    values = {
        "window_frames": 100,
        "hop_frames": 50,
        "onset_rms": 0.08,
        "offset_rms": 0.04,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }
    values.update(overrides)
    return ActivityRecipe(**values)


def _stem_recipe(**overrides):
    values = {
        "onset_share": 0.2,
        "offset_share": 0.1,
        "onset_music_rms": 0.05,
        "offset_music_rms": 0.025,
        "window_frames": 100,
        "hop_frames": 50,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }
    values.update(overrides)
    return StemEnergyShareRecipe(**values)


def test_discovers_activity_without_assigning_semantic_identity():
    samples = np.zeros((2, 1000), dtype=np.float32)
    samples[:, 200:500] = 0.1

    result = discover_activity(samples, 1000, _recipe())

    # 150..250 の窓は半分だけ有音で onset 閾値未満。最初の完全有音窓から開始する。
    assert [(item.start_frame, item.end_frame) for item in result.candidates] == [(200, 550)]
    detections = result.as_detections()
    assert detections[0].semantic_role == "unresolved_music_like"
    assert detections[0].identity_state == "unresolved"
    assert detections[0].candidate_id.startswith(result.recipe_id)


def test_hysteresis_keeps_below_onset_signal_until_offset_threshold():
    samples = np.zeros((1, 500), dtype=np.float32)
    samples[:, 100:200] = 0.1
    samples[:, 200:300] = 0.03
    samples[:, 300:400] = 0.1

    result = discover_activity(
        samples, 1000, _recipe(window_frames=100, hop_frames=100, offset_rms=0.02),
    )

    assert [(item.start_frame, item.end_frame) for item in result.candidates] == [(100, 400)]


def test_short_gap_is_bridged_before_minimum_duration_gate():
    samples = np.zeros((1, 500), dtype=np.float32)
    samples[:, 100:200] = 0.1
    samples[:, 250:350] = 0.1

    result = discover_activity(
        samples, 1000,
        _recipe(
            window_frames=50, hop_frames=50, bridge_gap_frames=50,
            minimum_active_frames=200,
        ),
    )

    assert [(item.start_frame, item.end_frame) for item in result.candidates] == [(100, 350)]


def test_short_activity_is_rejected_and_silence_is_valid_zero():
    samples = np.zeros((1, 500), dtype=np.float32)
    samples[:, 100:150] = 0.1
    recipe = _recipe(window_frames=50, hop_frames=50, minimum_active_frames=100)

    assert discover_activity(samples, 1000, recipe).candidates == ()
    assert discover_activity(np.zeros_like(samples), 1000, recipe).candidates == ()


def test_padding_is_clamped_to_source_and_overlaps_are_remerged():
    samples = np.zeros((1, 500), dtype=np.float32)
    samples[:, 0:50] = 0.1
    samples[:, 100:150] = 0.1

    result = discover_activity(
        samples, 1000,
        _recipe(window_frames=50, hop_frames=50, padding_frames=30, bridge_gap_frames=20),
    )

    assert [(item.start_frame, item.end_frame) for item in result.candidates] == [(0, 180)]


def test_recipe_identity_covers_every_output_affecting_parameter():
    baseline = _recipe()
    variants = [
        _recipe(window_frames=101), _recipe(hop_frames=51), _recipe(onset_rms=0.081),
        _recipe(offset_rms=0.039), _recipe(bridge_gap_frames=1),
        _recipe(minimum_active_frames=2), _recipe(padding_frames=1),
    ]

    assert len(baseline.recipe_id) > 20
    assert len({baseline.recipe_id, *(item.recipe_id for item in variants)}) == len(variants) + 1


@pytest.mark.parametrize(
    "recipe,message",
    [
        (_recipe(window_frames=0), "window_frames"),
        (_recipe(onset_rms=float("nan")), "onset_rms"),
        (_recipe(onset_rms=0.01, offset_rms=0.02), "offset_rms"),
        (_recipe(bridge_gap_frames=-1), "bridge_gap_frames"),
    ],
)
def test_invalid_recipe_is_rejected(recipe, message):
    with pytest.raises(InvalidActivityRecipeError, match=message):
        recipe.validate()


def test_recipe_file_requires_complete_exact_schema(tmp_path):
    path = tmp_path / "recipe.json"
    payload = _recipe().to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_activity_recipe(path) == _recipe()

    payload["typo"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InvalidActivityRecipeError, match="extra"):
        load_activity_recipe(path)


@pytest.mark.parametrize(
    "samples",
    [np.empty((1, 0)), np.zeros((1, 2, 3)), np.array([[np.nan, 0.0]])],
)
def test_invalid_audio_never_produces_candidates(samples):
    with pytest.raises(ValueError):
        discover_activity(samples, 1000, _recipe())


def test_streaming_file_discovery_matches_in_memory_result(tmp_path):
    samples = np.zeros((2, 1053), dtype=np.float32)
    samples[:, 173:497] = 0.1
    samples[:, 701:991] = 0.06
    path = tmp_path / "long.wav"
    sf.write(path, samples.T, 1000, subtype="FLOAT")
    recipe = _recipe(
        window_frames=97, hop_frames=43, onset_rms=0.08, offset_rms=0.04,
        bridge_gap_frames=80, minimum_active_frames=50, padding_frames=23,
    )

    streamed = discover_activity_file(path, recipe)
    in_memory = discover_activity(samples, 1000, recipe)

    assert streamed == in_memory


def test_stem_energy_share_recipe_uses_all_normalized_stems(tmp_path):
    signals = {
        "Music": np.zeros((1000, 1), dtype=np.float32),
        "Speech": np.full((1000, 1), 0.03, dtype=np.float32),
        "Effects": np.zeros((1000, 1), dtype=np.float32),
    }
    signals["Music"][200:600] = 0.1
    paths = {}
    for stem, samples in signals.items():
        path = tmp_path / f"{stem}.wav"
        sf.write(path, samples, 1000, subtype="FLOAT")
        paths[stem] = path

    result = discover_stem_energy_share_files(paths, _stem_recipe())

    assert [(item.start_frame, item.end_frame) for item in result.candidates] == [(150, 650)]
    assert result.candidates[0].candidate_id.startswith(result.recipe_id)
    assert result.candidates[0].peak_rms == pytest.approx(0.1)


def test_stem_energy_share_recipe_rejects_music_residual_below_absolute_gate(tmp_path):
    paths = {}
    for stem, level in {"Music": 0.001, "Speech": 0.0, "Effects": 0.0}.items():
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.full((500, 1), level, dtype=np.float32), 1000, subtype="FLOAT")
        paths[stem] = path

    assert discover_stem_energy_share_files(paths, _stem_recipe()).candidates == ()


def test_stem_energy_share_recipe_loader_and_frozen_identity(tmp_path):
    recipe = _stem_recipe(
        onset_music_rms=0.001,
        offset_music_rms=0.0005,
        window_frames=4410,
        hop_frames=2205,
        bridge_gap_frames=88200,
        minimum_active_frames=22050,
        padding_frames=4410,
    )
    path = tmp_path / "stem-energy-share.json"
    path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")

    loaded = load_discovery_recipe(path)

    assert loaded == recipe
    assert loaded.recipe_id == "bgm-stem-energy-share-v1-465b8613c1eebcea"


def test_stem_energy_share_recipe_requires_matching_layout(tmp_path):
    paths = {}
    for stem, frames in {"Music": 500, "Speech": 400, "Effects": 500}.items():
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.zeros((frames, 1), dtype=np.float32), 1000, subtype="FLOAT")
        paths[stem] = path

    with pytest.raises(ValueError, match="layout"):
        discover_stem_energy_share_files(paths, _stem_recipe())


def _relative_recipe(**overrides):
    values = {
        "window_frames": 100,
        "hop_frames": 50,
        "onset_rms": 0.08,
        "offset_ratio_db": 20.0,
        "offset_floor_rms": 0.001,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }
    values.update(overrides)
    return RelativeHysteresisRecipe(**values)


def test_relative_hysteresis_keeps_one_quiet_cue_together():
    """絶対offsetでは切れる静かな中間部を、区間自身のpeakからの相対量で保つ。"""
    samples = np.zeros((2, 1000), dtype=np.float32)
    samples[:, 100:300] = 0.2
    samples[:, 300:600] = 0.03
    samples[:, 600:800] = 0.2

    absolute = discover_activity(samples, 1000, _recipe())
    relative = discover_activity(samples, 1000, _relative_recipe())

    assert len(absolute.candidates) == 2
    assert len(relative.candidates) == 1
    assert relative.candidates[0].start_frame <= 100
    assert relative.candidates[0].end_frame >= 800


def test_relative_hysteresis_does_not_start_below_the_absolute_onset():
    """相対規則でも、区間の開始は宣言した絶対gateのままにする。"""
    samples = np.full((2, 1000), 0.03, dtype=np.float32)

    assert discover_activity(samples, 1000, _relative_recipe()).candidates == ()


def test_relative_hysteresis_stops_at_the_declared_absolute_floor():
    """相対しきい値が下限を割る場合は、絶対の下限で止める。"""
    samples = np.zeros((2, 1000), dtype=np.float32)
    samples[:, 400:600] = 0.2
    samples[:, 600:900] = 0.0005

    result = discover_activity(samples, 1000, _relative_recipe(offset_ratio_db=60.0))

    assert len(result.candidates) == 1
    assert result.candidates[0].end_frame <= 700


def test_relative_hysteresis_expansion_does_not_depend_on_scan_order():
    """先に種を決めてから広げるので、強い区間が後ろでも同じ結果になる。"""
    forward = np.zeros((2, 1200), dtype=np.float32)
    forward[:, 100:200] = 0.4
    forward[:, 800:900] = 0.1
    backward = forward[:, ::-1].copy()

    left = discover_activity(forward, 1000, _relative_recipe()).candidates
    right = discover_activity(backward, 1000, _relative_recipe()).candidates

    assert len(left) == len(right) == 2
    assert [item.end_frame - item.start_frame for item in left] == [
        item.end_frame - item.start_frame for item in reversed(right)
    ]


def test_relative_recipe_loader_and_frozen_identity(tmp_path):
    recipe = _relative_recipe(
        window_frames=4410,
        hop_frames=2205,
        onset_rms=0.005,
        offset_ratio_db=30.0,
        offset_floor_rms=0.0005,
        bridge_gap_frames=8820,
        minimum_active_frames=22050,
        padding_frames=4410,
    )
    path = tmp_path / "relative.json"
    path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")

    loaded = load_discovery_recipe(path)

    assert loaded == recipe
    assert loaded.recipe_id.startswith("bgm-activity-relative-rms-v1-")
    assert loaded.offset_rms_for(0.05) == pytest.approx(0.05 / 10.0 ** 1.5)
    assert loaded.offset_rms_for(0.001) == 0.0005


@pytest.mark.parametrize("overrides", [
    {"offset_ratio_db": 0.0},
    {"offset_ratio_db": float("inf")},
    {"offset_floor_rms": 0.0},
    {"offset_floor_rms": 0.5},
    {"method": "windowed_rms_hysteresis"},
])
def test_relative_recipe_rejects_unusable_declarations(overrides):
    with pytest.raises(InvalidActivityRecipeError):
        _relative_recipe(**overrides).validate()


def test_relative_recipe_loader_rejects_absolute_keys(tmp_path):
    path = tmp_path / "mixed.json"
    payload = _relative_recipe().to_dict()
    payload["offset_rms"] = 0.04
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidActivityRecipeError, match="キーが不正"):
        load_discovery_recipe(path)


def _relative_stem_recipe(**overrides):
    values = {
        "onset_share": 0.2,
        "onset_music_rms": 0.05,
        "offset_ratio_db": 20.0,
        "offset_floor_rms": 0.001,
        "window_frames": 100,
        "hop_frames": 50,
        "bridge_gap_frames": 0,
        "minimum_active_frames": 1,
        "padding_frames": 0,
    }
    values.update(overrides)
    return RelativeStemEnergyShareRecipe(**values)


def _write_stems(tmp_path, music, speech, effects):
    paths = {}
    for stem, values in {"Music": music, "Speech": speech, "Effects": effects}.items():
        path = tmp_path / f"{stem}.wav"
        sf.write(path, np.asarray(values, dtype=np.float32)[:, np.newaxis], 1000, subtype="FLOAT")
        paths[stem] = path
    return paths


def test_relative_share_keeps_a_cue_through_loud_dialogue(tmp_path):
    """台詞で取り分が落ちても、Musicが残っていれば同じ一曲として保つ。"""
    music = np.full(600, 0.2, dtype=np.float32)
    speech = np.zeros(600, dtype=np.float32)
    speech[200:400] = 0.9
    effects = np.zeros(600, dtype=np.float32)
    paths = _write_stems(tmp_path, music, speech, effects)

    absolute = discover_stem_energy_share_files(paths, _stem_recipe(
        onset_music_rms=0.05, offset_music_rms=0.001))
    relative = discover_stem_energy_share_files(paths, _relative_stem_recipe())

    assert len(absolute.candidates) == 2
    assert len(relative.candidates) == 1
    assert relative.candidates[0].start_frame == 0
    assert relative.candidates[0].end_frame == 600


def test_relative_share_still_needs_the_music_share_to_start(tmp_path):
    """続きの判断から取り分を外しても、開始の取り分gateは残す。"""
    music = np.full(600, 0.06, dtype=np.float32)
    speech = np.full(600, 0.9, dtype=np.float32)
    effects = np.zeros(600, dtype=np.float32)

    paths = _write_stems(tmp_path, music, speech, effects)

    assert discover_stem_energy_share_files(paths, _relative_stem_recipe()).candidates == ()


def test_relative_share_stops_at_the_declared_absolute_floor(tmp_path):
    music = np.zeros(900, dtype=np.float32)
    music[100:300] = 0.2
    music[300:900] = 0.0005
    paths = _write_stems(tmp_path, music, np.zeros(900), np.zeros(900))

    result = discover_stem_energy_share_files(
        paths, _relative_stem_recipe(offset_ratio_db=60.0))

    assert len(result.candidates) == 1
    assert result.candidates[0].end_frame <= 400


def test_relative_stem_recipe_loader_and_frozen_identity(tmp_path):
    recipe = _relative_stem_recipe(
        onset_music_rms=0.001,
        offset_ratio_db=30.0,
        offset_floor_rms=0.0005,
        window_frames=4410,
        hop_frames=2205,
        bridge_gap_frames=88200,
        minimum_active_frames=22050,
        padding_frames=4410,
    )
    path = tmp_path / "relative-share.json"
    path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")

    loaded = load_discovery_recipe(path)

    assert loaded == recipe
    assert loaded.recipe_id == "bgm-stem-energy-share-relative-v1-7c171a7a1d4adcac"
    assert loaded.required_stems == ("Music", "Speech", "Effects")


@pytest.mark.parametrize("overrides", [
    {"offset_ratio_db": 0.0},
    {"offset_floor_rms": 0.5},
    {"onset_share": 1.5},
    {"method": "stem_energy_share_hysteresis"},
])
def test_relative_stem_recipe_rejects_unusable_declarations(overrides):
    with pytest.raises(InvalidActivityRecipeError):
        _relative_stem_recipe(**overrides).validate()
