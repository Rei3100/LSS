import numpy as np

from lss.harness.metrics import compute_decision_metrics


def _sine(freq: float, seconds: float = 1.0, sr: int = 44100) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wave = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([wave, wave])  # (channels, samples)


def test_identical_signal_scores_near_ceiling():
    ref = _sine(440.0)
    result = compute_decision_metrics(ref, ref.copy())

    assert result["sdr"] > 100.0  # 差が実質ゼロなのでSDRは非常に大きくなる
    assert 0.0 <= result["l1_freq"] <= 100.0
    assert result["l1_freq"] > 90.0
    assert 0.0 <= result["aura_mrstft"] <= 100.0
    assert result["aura_mrstft"] > 90.0


def test_noisy_estimate_scores_lower_than_identical():
    ref = _sine(440.0)
    rng = np.random.default_rng(0)
    noisy = ref + rng.normal(scale=0.05, size=ref.shape).astype(np.float32)

    identical_result = compute_decision_metrics(ref, ref.copy())
    noisy_result = compute_decision_metrics(ref, noisy)

    assert noisy_result["sdr"] < identical_result["sdr"]
    assert noisy_result["l1_freq"] < identical_result["l1_freq"]
    assert noisy_result["aura_mrstft"] < identical_result["aura_mrstft"]


def test_returns_all_five_keys():
    ref = _sine(440.0)
    result = compute_decision_metrics(ref, ref.copy())
    assert set(result.keys()) == {"sdr", "l1_freq", "aura_mrstft", "bleedless", "fullness"}
