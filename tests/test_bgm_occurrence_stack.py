"""同じ録音の複数の回を重ねて音楽を推定する。分離modelは使わない。"""
from __future__ import annotations

import numpy as np
import pytest

from lss.bgm import occurrence_stack as stack

RATE = 8000


def _recipe(*, taps=33, block=0.5, combine=stack.COMBINE_MEDIAN):
    return stack.OccurrenceStackRecipe(fir_taps=taps, block_seconds=block, combine=combine)


def _tone(seed, frames, channels=2):
    """帯域の限られた、繰り返しのある信号。白色雑音より実際の録音に近い。"""
    rng = np.random.default_rng(seed)
    kernel = np.hanning(33)
    kernel /= kernel.sum()
    raw = rng.standard_normal((frames + 64, channels))
    filtered = np.stack([np.convolve(raw[:, c], kernel, mode="same") for c in range(channels)],
                        axis=1)
    return filtered[:frames]


@pytest.fixture
def occurrences():
    """同じ音楽が3回鳴り、回ごとに音量と上に乗るものが違う。"""
    frames = RATE * 4
    music = _tone(1, frames)
    return {
        "music": music,
        "a": music * 0.8 + _tone(2, frames) * 0.3,
        "b": music * 0.5 + _tone(3, frames) * 0.3,
        "c": music * 0.9 + _tone(4, frames) * 0.3,
        "truth_a": music * 0.8,
    }


def _snr(truth, estimate):
    return 10.0 * np.log10(np.sum(truth ** 2) / np.sum((truth - estimate) ** 2))


def test_recipe_id_depends_on_the_declaration():
    assert _recipe().recipe_id == _recipe().recipe_id
    assert _recipe().recipe_id != _recipe(taps=65).recipe_id
    assert _recipe().recipe_id != _recipe(combine=stack.COMBINE_MEAN).recipe_id
    assert _recipe().recipe_id.startswith("bgm-occurrence-stack-v1-")


@pytest.mark.parametrize("kwargs", [
    {"taps": 32}, {"taps": 0}, {"block": 0.0}, {"block": float("nan")},
    {"combine": "average"},
])
def test_recipe_rejects_invalid_declarations(kwargs):
    with pytest.raises(stack.OccurrenceStackError):
        _recipe(**kwargs).validate()


def test_recipe_rejects_an_unknown_schema():
    with pytest.raises(stack.OccurrenceStackError):
        stack.OccurrenceStackRecipe(
            fir_taps=33, block_seconds=0.5, combine=stack.COMBINE_MEDIAN,
            method="other").validate()


def test_two_occurrences_recover_the_shared_music_better_than_the_mix(occurrences):
    result = stack.estimate_music(
        occurrences["a"], [occurrences["b"], occurrences["c"]], _recipe(), sample_rate=RATE)
    truth = occurrences["truth_a"]
    assert _snr(truth, result["music"]) > _snr(truth, occurrences["a"])
    assert result["occurrence_count"] == 3
    assert len(result["used"]) == 2


def test_the_estimate_keeps_the_shape_and_time_base_of_its_own_occurrence(occurrences):
    result = stack.estimate_music(
        occurrences["a"], [occurrences["b"]], _recipe(), sample_rate=RATE)
    assert result["music"].shape == occurrences["a"].shape
    assert result["residual"].shape == occurrences["a"].shape
    assert np.allclose(result["music"] + result["residual"], occurrences["a"])


def test_a_shifted_occurrence_is_aligned_before_fitting(occurrences):
    shift = 137
    shifted = np.roll(occurrences["b"], shift, axis=0)
    result = stack.estimate_music(occurrences["a"], [shifted], _recipe(), sample_rate=RATE)
    assert result["used"][0]["lag_samples"] == -shift


def test_an_unrelated_occurrence_explains_almost_nothing(occurrences):
    unrelated = _tone(99, occurrences["a"].shape[0])
    related = stack.estimate_music(
        occurrences["a"], [occurrences["b"]], _recipe(), sample_rate=RATE)
    apart = stack.estimate_music(occurrences["a"], [unrelated], _recipe(), sample_rate=RATE)
    assert apart["used"][0]["explained"] < related["used"][0]["explained"]


def test_mono_and_stereo_occurrences_can_be_mixed(occurrences):
    mono = occurrences["b"].mean(axis=1)
    result = stack.estimate_music(occurrences["a"], [mono], _recipe(), sample_rate=RATE)
    assert result["music"].shape == occurrences["a"].shape


@pytest.mark.parametrize("combine", sorted(stack.COMBINES))
def test_every_declared_combination_produces_an_estimate(occurrences, combine):
    result = stack.estimate_music(
        occurrences["a"], [occurrences["b"], occurrences["c"]],
        _recipe(combine=combine), sample_rate=RATE)
    assert result["music"].shape == occurrences["a"].shape
    assert 0.0 <= result["combined_explained"] <= 1.0


def test_best_single_needs_the_target():
    with pytest.raises(stack.OccurrenceStackError):
        stack.combine_estimates([np.zeros((4, 1))], stack.COMBINE_BEST)


def test_no_other_occurrence_is_refused(occurrences):
    with pytest.raises(stack.OccurrenceStackError):
        stack.estimate_music(occurrences["a"], [], _recipe(), sample_rate=RATE)


def test_a_silent_other_occurrence_is_skipped(occurrences):
    silent = np.zeros_like(occurrences["b"])
    result = stack.estimate_music(
        occurrences["a"], [silent, occurrences["b"]], _recipe(), sample_rate=RATE)
    assert [item["index"] for item in result["used"]] == [1]


@pytest.mark.parametrize("bad", [
    np.zeros((1, 2)), np.zeros((4, 0)), np.zeros((2, 2, 2)),
    np.array([[np.nan, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
])
def test_audio_that_is_not_a_usable_occurrence_is_refused(occurrences, bad):
    with pytest.raises(stack.OccurrenceStackError):
        stack.estimate_music(occurrences["a"], [bad], _recipe(), sample_rate=RATE)


def test_a_nonpositive_sample_rate_is_refused(occurrences):
    with pytest.raises(stack.OccurrenceStackError):
        stack.estimate_music(occurrences["a"], [occurrences["b"]], _recipe(), sample_rate=0)


def test_the_result_states_what_it_does_not_claim(occurrences):
    result = stack.estimate_music(
        occurrences["a"], [occurrences["b"]], _recipe(), sample_rate=RATE)
    assert any("欲しい劇伴かどうかは判定しない" in line for line in result["claim_boundary"])
    assert any("残差へ落ちる" in line for line in result["claim_boundary"])


def test_joint_fitting_uses_every_occurrence_at_once(occurrences):
    """回ごとに当てはめてから束ねるより、同時に当てはめるほうが説明できる。"""
    recipe = _recipe(combine=stack.COMBINE_JOINT)
    joint = stack.estimate_music(
        occurrences["a"], [occurrences["b"], occurrences["c"]], recipe, sample_rate=RATE)
    apart = stack.estimate_music(
        occurrences["a"], [occurrences["b"], occurrences["c"]],
        _recipe(combine=stack.COMBINE_MEDIAN), sample_rate=RATE)
    truth = occurrences["truth_a"]
    assert _snr(truth, joint["music"]) > _snr(truth, apart["music"])
    assert _snr(truth, joint["music"]) > _snr(truth, occurrences["a"])


def test_joint_fitting_needs_at_least_one_occurrence(occurrences):
    with pytest.raises(stack.OccurrenceStackError):
        stack.fit_explained_jointly(occurrences["a"], [], _recipe(), sample_rate=RATE)


def test_the_attenuation_of_each_fit_can_be_measured_without_the_truth(occurrences):
    """最小二乗の当てはめは音楽より小さく出る。その縮み率は回どうしから測れる。"""
    recipe = _recipe()
    fits = []
    for key in ("b", "c"):
        aligned, _ = stack.align_to(occurrences["a"], occurrences[key], sample_rate=RATE)
        fits.append(stack.fit_explained(occurrences["a"], aligned, recipe, sample_rate=RATE))
    truth = occurrences["truth_a"]
    actual = [float(np.sum(fit * truth) / np.sum(truth ** 2)) for fit in fits]
    estimated = stack.attenuations(fits, occurrences["a"])
    assert all(0.0 < value <= 1.0 for value in estimated)
    for measured, guessed in zip(actual, estimated):
        assert abs(measured - guessed) < 0.1


def test_a_single_occurrence_cannot_be_debiased(occurrences):
    recipe = _recipe(combine=stack.COMBINE_DEBIASED)
    aligned, _ = stack.align_to(occurrences["a"], occurrences["b"], sample_rate=RATE)
    fit = stack.fit_explained(occurrences["a"], aligned, recipe, sample_rate=RATE)
    combined = stack.combine_estimates([fit], stack.COMBINE_DEBIASED, target=occurrences["a"])
    assert np.allclose(combined, fit)


def test_two_stage_estimation_builds_a_reference_before_fitting(occurrences):
    """一度推定した音楽を寄せ集めて参照を作り、それを当てはめ直す。"""
    result = stack.estimate_music(
        occurrences["a"], [occurrences["b"], occurrences["c"]],
        _recipe(combine=stack.COMBINE_TWO_STAGE), sample_rate=RATE)
    truth = occurrences["truth_a"]
    assert _snr(truth, result["music"]) > _snr(truth, occurrences["a"])
    assert result["music"].shape == occurrences["a"].shape


def test_two_stage_falls_back_to_one_stage_with_a_single_occurrence(occurrences):
    recipe = _recipe(combine=stack.COMBINE_TWO_STAGE)
    aligned, _ = stack.align_to(occurrences["a"], occurrences["b"], sample_rate=RATE)
    two = stack.two_stage_estimate(occurrences["a"], [aligned], recipe, sample_rate=RATE)
    one = stack.fit_explained_jointly(occurrences["a"], [aligned], recipe, sample_rate=RATE)
    assert np.allclose(two, one)


def test_two_stage_needs_at_least_one_occurrence(occurrences):
    with pytest.raises(stack.OccurrenceStackError):
        stack.two_stage_estimate(occurrences["a"], [], _recipe(), sample_rate=RATE)


def test_the_estimate_reports_how_much_of_the_mix_it_kept(occurrences):
    result = stack.estimate_music(
        occurrences["a"], [occurrences["b"], occurrences["c"]], _recipe(), sample_rate=RATE)
    share = np.sum(result["music"] ** 2) / np.sum(np.asarray(occurrences["a"]) ** 2)
    assert result["energy_share"] == pytest.approx(share, rel=1e-6)
    assert result["meets_minimum_energy_share"] is True  # 下限0なら常に満たす


def test_a_minimum_energy_share_marks_an_untrustworthy_estimate(occurrences):
    # 実番組の既発売cueで、残した割合が小さい推定は録音へ近づかなかった。
    strict = stack.OccurrenceStackRecipe(
        fir_taps=33, block_seconds=0.5, combine=stack.COMBINE_MEDIAN,
        minimum_energy_share=0.99)
    result = stack.estimate_music(
        occurrences["a"], [occurrences["b"]], strict, sample_rate=RATE)
    assert result["meets_minimum_energy_share"] is False
    assert 0.0 < result["energy_share"] < 0.99
    assert any("残りかす" in line for line in result["claim_boundary"])


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_an_impossible_minimum_energy_share_is_refused(value):
    with pytest.raises(stack.OccurrenceStackError):
        stack.OccurrenceStackRecipe(
            fir_taps=9, block_seconds=0.1, combine=stack.COMBINE_JOINT,
            minimum_energy_share=value).validate()


def test_the_minimum_is_part_of_the_declaration():
    base = stack.OccurrenceStackRecipe(
        fir_taps=9, block_seconds=0.1, combine=stack.COMBINE_JOINT)
    strict = stack.OccurrenceStackRecipe(
        fir_taps=9, block_seconds=0.1, combine=stack.COMBINE_JOINT,
        minimum_energy_share=0.10)
    assert base.recipe_id != strict.recipe_id
