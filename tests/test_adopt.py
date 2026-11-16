from lss.harness.adopt import is_significant, passes_adoption

BASELINE = {
    "sdr": 10.0,
    "l1_freq": 50.0,
    "aura_mrstft": 50.0,
    "rtf": 1.0,
    "peak_vram_mb": 4000,
    "f1_ok": True,
    "f2_ok": True,
    "f3_ok": True,
    "f4_ok": True,
}


def test_ac9_single_metric_spike_with_others_regressing_is_rejected():
    """AC9必須の合成テスト: 1指標だけ大幅に高く他2指標が僅かに低い候補は不採用（指標ブースト耐性）。"""
    candidate = dict(BASELINE)
    candidate.update({"sdr": 20.0, "l1_freq": 49.5, "aura_mrstft": 49.5})

    assert passes_adoption(candidate, BASELINE) is False


def test_all_three_metrics_improved_above_threshold_is_significant():
    candidate = {"sdr": 10.31, "l1_freq": 50.31, "aura_mrstft": 50.31}
    baseline = {"sdr": 10.0, "l1_freq": 50.0, "aura_mrstft": 50.0}
    assert is_significant(candidate, baseline) is True


def test_all_three_metrics_improved_below_threshold_is_not_significant():
    candidate = {"sdr": 10.1, "l1_freq": 50.1, "aura_mrstft": 50.1}
    baseline = {"sdr": 10.0, "l1_freq": 50.0, "aura_mrstft": 50.0}
    assert is_significant(candidate, baseline) is False


def test_single_metric_improved_above_threshold_is_significant():
    candidate = {"sdr": 11.5, "l1_freq": 50.0, "aura_mrstft": 50.0}
    baseline = {"sdr": 10.0, "l1_freq": 50.0, "aura_mrstft": 50.0}
    assert is_significant(candidate, baseline) is True


def test_single_metric_improved_below_threshold_is_not_significant():
    candidate = {"sdr": 10.5, "l1_freq": 50.0, "aura_mrstft": 50.0}
    baseline = {"sdr": 10.0, "l1_freq": 50.0, "aura_mrstft": 50.0}
    assert is_significant(candidate, baseline) is False


def test_exactly_two_metrics_improved_is_not_significant():
    candidate = {"sdr": 12.0, "l1_freq": 52.0, "aura_mrstft": 50.0}
    baseline = {"sdr": 10.0, "l1_freq": 50.0, "aura_mrstft": 50.0}
    assert is_significant(candidate, baseline) is False


def test_passes_adoption_true_for_clean_improvement():
    candidate = dict(BASELINE)
    candidate.update({"sdr": 11.5})
    assert passes_adoption(candidate, BASELINE) is True


def test_passes_adoption_does_not_trade_quality_away_for_rtf():
    candidate = dict(BASELINE)
    candidate.update({"sdr": 11.5, "rtf": 3.5})
    assert passes_adoption(candidate, BASELINE) is True


def test_passes_adoption_does_not_trade_quality_away_for_vram():
    candidate = dict(BASELINE)
    candidate.update({"sdr": 11.5, "peak_vram_mb": 13000})
    assert passes_adoption(candidate, BASELINE) is True


def test_passes_adoption_allows_authenticated_acquisition_after_runtime_validation():
    candidate = dict(BASELINE)
    candidate.update({"sdr": 11.5, "f1_ok": False})
    assert passes_adoption(candidate, BASELINE) is True


def test_passes_adoption_false_when_exact_runtime_validation_fails():
    candidate = dict(BASELINE)
    candidate.update({"sdr": 11.5, "f4_ok": False})
    assert passes_adoption(candidate, BASELINE) is False


def test_passes_adoption_false_when_no_significant_improvement():
    candidate = dict(BASELINE)
    candidate.update({"sdr": 10.2})
    assert passes_adoption(candidate, BASELINE) is False
