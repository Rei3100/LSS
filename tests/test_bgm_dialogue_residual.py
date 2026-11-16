"""公開cueに残る台詞を伴奏との共通成分で取るStageの契約。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import dialogue_residual as dr

RATE = 44100


def _tone(seconds: float, freq: float = 330.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    tone = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([tone, 0.7 * tone], axis=1)


def _sdr(estimate, truth):
    return 10 * np.log10(np.sum(truth ** 2) / np.sum((estimate - truth) ** 2))


def test_recipe_identity_changes_with_the_declaration():
    base = dr.default_recipe()
    assert base.recipe_id == dr.default_recipe().recipe_id
    changed = dr.DialogueResidualRecipe(**{**base.to_dict(), "speech_gate": 0.4})
    assert changed.recipe_id != base.recipe_id
    with pytest.raises(dr.DialogueResidualError):
        dr.DialogueResidualRecipe(**{**base.to_dict(), "hop": 4096}).validate()


def test_eligibility_averages_speech_on_the_cue_windows():
    probabilities = np.zeros((10, 527), dtype=np.float16)
    probabilities[2:6, dr.SPEECH_CLASS] = 0.6
    recipe = dr.default_recipe()
    got = dr.eligibility(probabilities, 2.0, 6.0, recipe)
    assert got["eligible"] is True and got["speech_on_music"] == pytest.approx(0.6, abs=1e-3)
    assert dr.eligibility(probabilities, 6.0, 10.0, recipe)["eligible"] is False
    # 半分だけ台詞が残る区間は平均0.3で該当する（gate以上）。
    assert dr.eligibility(probabilities, 2.0, 10.0, recipe)["speech_on_music"] == pytest.approx(0.3, abs=1e-3)
    assert dr.eligibility(probabilities, 12.0, 14.0, recipe) == {"speech_on_music": None, "eligible": False}


def test_consensus_removes_what_either_estimate_removed_and_keeps_the_shared_music():
    rng = np.random.default_rng(0)
    music = _tone(4.0)
    dialogue = np.zeros_like(music)
    dialogue[RATE:3 * RATE] = rng.normal(0, 0.1, (2 * RATE, 1))
    effects = np.zeros_like(music)
    for second in range(4):
        effects[second * RATE:second * RATE + RATE // 20] = rng.normal(0, 0.3, (RATE // 20, 1))
    sfc_music = music + dialogue  # SFCは台詞を残す
    instrumental = music + effects  # 伴奏modelは効果音を残す
    out = dr.consensus(sfc_music, instrumental, dr.default_recipe())
    assert out.shape == music.shape and np.isfinite(out).all()
    assert _sdr(out, music) > _sdr(sfc_music, music) + 6
    assert _sdr(out, music) > _sdr(instrumental, music) + 6
    # 同じ推定どうしなら値を変えない（STFTの往復の誤差だけ）。
    assert np.max(np.abs(dr.consensus(music, music, dr.default_recipe()) - music)) < 1e-6


def test_consensus_is_the_same_operation_as_the_measured_probe():
    """probeで測った演算（`scripts/probe_bgm_single_occurrence_bench.combinations`）と数値が一致する。"""
    from scripts.probe_bgm_single_occurrence_bench import combinations

    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 0.1, (RATE, 2)), rng.normal(0, 0.1, (RATE, 2))
    measured = combinations(np.zeros_like(a), {"inst": b, "sfc_music": a}, RATE)["consensus"]
    assert np.max(np.abs(dr.consensus(a, b, dr.default_recipe()) - measured)) < 1e-9


def test_consensus_refuses_mismatched_or_non_finite_estimates():
    recipe = dr.default_recipe()
    with pytest.raises(dr.DialogueResidualError):
        dr.consensus(np.zeros((100, 2)), np.zeros((99, 2)), recipe)
    bad = np.zeros((RATE, 2))
    bad[5, 0] = np.nan
    with pytest.raises(dr.DialogueResidualError):
        dr.consensus(bad, np.zeros((RATE, 2)), recipe)


def test_saved_span_round_trips_with_headroom_and_is_verified(tmp_path):
    values = _tone(1.0, amp=1.6)  # 1を超える
    item = dr.write_span(values, RATE, tmp_path / "span.flac")
    assert item["headroom_shift"] >= 1
    assert np.max(np.abs(dr.read_span(item) - values)) < 1e-5
    with open(tmp_path / "span.flac", "ab") as stream:
        stream.write(b"x")
    with pytest.raises(dr.DialogueResidualError):
        dr.read_span(item)


def test_instrumental_spans_use_context_and_keep_the_cue_part(tmp_path):
    signal = _tone(40.0)
    signal[:, 1] += np.linspace(0, 0.01, signal.shape[0])  # 位置が分かる成分
    path = tmp_path / "analysis.wav"
    sf.write(str(path), signal.astype(np.float32), RATE, subtype="FLOAT")
    seen = []

    def separate(crop_path: Path, store: Path):
        crop, rate = sf.read(str(crop_path), dtype="float32", always_2d=True)
        seen.append(crop.shape[0])
        return 0.5 * crop.T, rate

    spans = [[20 * RATE, 25 * RATE], [2 * RATE, 5 * RATE]]
    result = dr.run_instrumental_spans(path, tmp_path / "out", spans, separate=separate,
                                       recipe=dr.default_recipe())
    assert result["spans"] == sorted(spans)
    # 文脈は前後15秒、信号の内側に収める。
    assert seen == [40 * RATE]
    first = result["instrumental"][0]
    assert first["context"] == [0, 40 * RATE]
    assert result["inference_frames"] == 40 * RATE
    got = dr.read_span(first)
    stored = sf.read(str(path), dtype="float64", always_2d=True)[0]
    assert np.max(np.abs(got - 0.5 * stored[2 * RATE:5 * RATE])) < 1e-5
    assert not list((tmp_path / "out").glob(".work-*"))


def test_instrumental_spans_write_in_order_across_groups_and_surface_write_errors(tmp_path, monkeypatch):
    path = tmp_path / "analysis.wav"
    sf.write(str(path), _tone(120.0).astype(np.float32), RATE, subtype="FLOAT")
    spans = [[2 * RATE, 4 * RATE], [100 * RATE, 103 * RATE]]  # 文脈が重ならない2群

    def separate(crop_path: Path, store: Path):
        crop, rate = sf.read(str(crop_path), dtype="float32", always_2d=True)
        return 0.5 * crop.T, rate

    result = dr.run_instrumental_spans(path, tmp_path / "out", spans, separate=separate, recipe=dr.default_recipe())
    assert len(result["inference_groups"]) == 2
    assert [item["span"] for item in result["instrumental"]] == spans
    stored = sf.read(str(path), dtype="float64", always_2d=True)[0]
    for item, (a, b) in zip(result["instrumental"], spans):
        assert np.max(np.abs(dr.read_span(item) - 0.5 * stored[a:b])) < 1e-5

    def broken(values, rate, target):
        raise OSError("disk full")

    monkeypatch.setattr(dr, "write_span", broken)
    with pytest.raises(OSError):
        dr.run_instrumental_spans(path, tmp_path / "out2", spans, separate=separate, recipe=dr.default_recipe())


def test_instrumental_spans_refuse_output_on_another_timeline(tmp_path):
    path = tmp_path / "analysis.wav"
    sf.write(str(path), _tone(10.0).astype(np.float32), RATE, subtype="FLOAT")
    with pytest.raises(dr.DialogueResidualError):
        dr.run_instrumental_spans(path, tmp_path / "out", [[RATE, 2 * RATE]],
                                  separate=lambda crop, store: (np.zeros((2, 10)), RATE),
                                  recipe=dr.default_recipe())


def test_local_regions_dilate_speech_windows_and_stay_inside_the_cue():
    probabilities = np.zeros((40, 527), dtype=np.float16)
    probabilities[[12, 13, 20], dr.SPEECH_CLASS] = 0.5
    probabilities[30, dr.SPEECH_CLASS] = 0.29  # gate未満
    recipe = dr.default_recipe()
    start, end = int(10.5 * RATE), int(21.5 * RATE)
    assert dr.local_regions(probabilities, start, end, RATE, recipe) == [
        [int(12 * RATE) - start, int(14 * RATE) - start], [int(20 * RATE) - start, int(21 * RATE) - start]]
    regions = dr.local_regions(probabilities, start, end, RATE, recipe, dilation_windows=1)
    # 広げると 窓12,13 → [11,15)秒、窓20 → [19,22)秒をcue [10.5,21.5) に切る
    assert regions == [[int(11 * RATE) - start, int(15 * RATE) - start], [int(19 * RATE) - start, end - start]]
    assert dr.local_regions(probabilities, int(25 * RATE), int(35 * RATE), RATE, recipe) == []


def test_local_consensus_is_bit_identical_outside_regions_and_ramps():
    rng = np.random.default_rng(0)
    music = rng.standard_normal((6 * RATE, 2)) * 0.1
    instrumental = music * 0.5
    regions = [[2 * RATE, 3 * RATE]]
    out = dr.local_consensus(music, instrumental, regions, RATE, dr.default_recipe())
    ramp = dr.default_recipe().local_ramp_seconds
    weight = dr.local_weight(len(music), regions, RATE, ramp)
    assert weight[2 * RATE] == 1.0 and weight[3 * RATE - 1] == 1.0
    assert 0.0 < weight[2 * RATE - 1] < 1.0 and weight[int(2 * RATE - ramp * RATE) - 2] == 0.0
    assert np.array_equal(out[weight == 0], music[weight == 0])
    inside = slice(2 * RATE + 4096, 3 * RATE - 4096)
    assert np.allclose(out[inside], instrumental[inside], atol=1e-6)
    assert np.array_equal(dr.local_consensus(music, instrumental, [], RATE, dr.default_recipe()), music)


def test_context_span_adds_the_margin_and_stays_inside_the_signal():
    recipe = dr.default_recipe()

    assert dr.context_span(20 * RATE, 25 * RATE, 100 * RATE, RATE, recipe) == (5 * RATE, 40 * RATE)
    assert dr.context_span(2 * RATE, 5 * RATE, 100 * RATE, RATE, recipe) == (0, 20 * RATE)
    assert dr.context_span(90 * RATE, 99 * RATE, 100 * RATE, RATE, recipe) == (75 * RATE, 100 * RATE)

    with pytest.raises(dr.DialogueResidualError):
        dr.context_span(5 * RATE, 5 * RATE, 100 * RATE, RATE, recipe)

    with pytest.raises(dr.DialogueResidualError):
        dr.context_span(50 * RATE, 101 * RATE, 100 * RATE, RATE, recipe)


def test_recipe_validate_rejects_out_of_range_declarations():
    import dataclasses

    base = dr.default_recipe()

    with pytest.raises(dr.DialogueResidualError):
        dataclasses.replace(base, speech_gate=1.5).validate()

    with pytest.raises(dr.DialogueResidualError):
        dataclasses.replace(base, context_seconds=200.0).validate()

    with pytest.raises(dr.DialogueResidualError):
        dataclasses.replace(base, local_dilation_windows=11).validate()

    with pytest.raises(dr.DialogueResidualError):
        dataclasses.replace(base, hop=4096).validate()

    with pytest.raises(dr.DialogueResidualError):
        dataclasses.replace(base, n_fft=8).validate()

    assert base.validate() is None
    assert base.recipe_id.startswith('bgm-dialogue-residual-v')
