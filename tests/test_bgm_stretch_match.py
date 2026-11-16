import numpy as np

from lss.bgm import stretch_match as sm

RATE = sm.SAMPLE_RATE


def _music(seconds: float, seed: int) -> np.ndarray:
    """Notes with harmonics and a changing melody: enough structure for waveform matching."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * RATE))
    position = 0
    while position < out.size:
        length = int(rng.uniform(0.15, 0.5) * RATE)
        t = np.arange(min(length, out.size - position)) / RATE
        base = rng.choice([196.0, 220.0, 247.0, 262.0, 294.0, 330.0, 349.0, 392.0, 440.0])
        note = sum(np.sin(2 * np.pi * base * h * t + rng.uniform(0, 6.28)) / h for h in (1, 2, 3, 4))
        out[position:position + t.size] += note * np.exp(-3 * t)
        position += t.size
    return out / np.max(np.abs(out))


def _stretch(signal: np.ndarray, rate: float, segment_seconds: float = 0.25, fade_seconds: float = 0.01) -> np.ndarray:
    """Pitch-preserving segment copy: output time u plays input time u * rate, segments joined by short fades."""
    segment, fade = int(segment_seconds * RATE), int(fade_seconds * RATE)
    length = int(signal.size / rate)
    out = np.zeros(length + segment + fade)
    norm = np.zeros_like(out)
    window = np.ones(segment + fade)
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    for start in range(0, length, segment):
        source = int(start * rate)
        piece = signal[source:source + segment + fade]
        if piece.size < segment + fade:
            break
        out[start:start + piece.size] += piece * window
        norm[start:start + piece.size] += window
    return (out / np.maximum(norm, 1e-3))[:length]


def _speech(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(int(seconds * RATE))
    envelope = (np.sin(2 * np.pi * 3 * np.arange(noise.size) / RATE) > 0.3).astype(float)
    return noise * envelope * 0.15


def test_tempo_changed_copy_is_found_with_its_ratio():
    recording = _music(40, 1)
    first = recording[:int(30 * RATE)] + _speech(30, 2)
    second = _stretch(recording, 1.0 / 1.05)[:int(30 * RATE)] + _speech(30, 3)
    # second plays the recording 5 % slower: second time = first time * 1.05
    found = sm.stretched_matches(first, second, [(5.0, 0.2)])
    assert found, "stretched copy not found"
    assert all(abs(match.rate - 1.05) < 0.005 for match in found)
    assert sum(match.query_end_seconds - match.query_start_seconds for match in found) >= 12
    assert all(match.mean_explained >= match.null_max + sm.MIN_MARGIN for match in found)


def test_diagnostic_sink_preserves_matches_and_rng_while_tracing_positive_path():
    recording = _music(30, 101)
    query = recording[:int(24 * RATE)] + _speech(24, 102)
    other = _stretch(recording, 1.0 / 1.05)[:int(24 * RATE)] + _speech(24, 103)
    seeds = [(5.0, 0.2)]
    ordinary_rng = np.random.default_rng(34)
    traced_rng = np.random.default_rng(34)

    ordinary = sm.stretched_matches(query, other, seeds, rng=ordinary_rng)
    trace = []
    traced = sm.stretched_matches(query, other, seeds, rng=traced_rng, diagnostics=trace)

    assert [item.to_dict() for item in traced] == [item.to_dict() for item in ordinary]
    assert traced_rng.bit_generator.state == ordinary_rng.bit_generator.state
    assert trace[0]["schema"] == "stretch-lag-trace-r1"
    assert trace[0]["authority"] == {"publication": False, "exact": False, "donor": False,
                                      "reconstruction": False}
    assert trace[0]["attempts"]
    accepted = [run for attempt in trace[0]["attempts"] for run in attempt["runs"]
                if run["decision"] == "accepted"]
    assert accepted
    run = accepted[0]
    assert round(run["mean_explained"], 4) == traced[0].mean_explained
    assert round(run["null_max"], 4) == traced[0].null_max
    assert abs(run["query_sample_bounds"][0] / RATE - traced[0].query_start_seconds) <= 0.0005
    assert abs(run["reference_sample_bounds"][0] / RATE - traced[0].other_start_seconds) <= 0.0005
    null_trials = [trial for trial in run["null_trials"] if trial["status"] != "excluded_near_true_lag"]
    assert null_trials and all("rows" in trial and "score" in trial for trial in null_trials)


def test_trace_distinguishes_off_line_explained_window_from_fit_inlier(monkeypatch):
    width = int(sm.WINDOW_SECONDS * RATE)
    hop = int(sm.HOP_SECONDS * RATE)
    query = np.ones(width + hop * 11)
    other = np.ones(query.size + RATE * 2)
    rows = []
    for index in range(12):
        start = index * hop
        lag = int(0.03 * start) + (int(0.20 * RATE) if index == 5 else 0)
        rows.append((start, lag, 0.8))
    monkeypatch.setattr(sm, "track", lambda *args, **kwargs: list(rows))

    def fixed_null(query, other, seed_seconds, true_lag, rate, rng, diagnostics=None):
        if diagnostics is not None:
            diagnostics.append({"trial_index": 0, "seed_lag_samples": 12345, "rows": [], "score": 0.1,
                                "status": "measured"})
        return 0.1

    monkeypatch.setattr(sm, "_null_level", fixed_null)
    trace = []
    found = sm.stretched_matches(query, other, [(0.0, 0.0)], rate=RATE,
                                 rng=np.random.default_rng(0), diagnostics=trace)

    assert found
    run = next(run for attempt in trace[0]["attempts"] for run in attempt["runs"]
               if run["decision"] == "accepted")
    outlier = next(row for row in run["windows"] if row["query_start_sample"] == 5 * hop)
    assert outlier["inside_run"] and outlier["mean_included"]
    assert outlier["explained_threshold_pass"] and not outlier["line_inlier"]


def test_foreign_music_trace_records_rejection_without_changing_empty_result():
    query = _music(24, 107) + _speech(24, 108)
    other = _music(24, 109) + _speech(24, 110)
    ordinary_rng = np.random.default_rng(11)
    traced_rng = np.random.default_rng(11)
    ordinary = sm.stretched_matches(query, other, [(5.0, 0.2)], rng=ordinary_rng)
    trace = []
    traced = sm.stretched_matches(query, other, [(5.0, 0.2)], rng=traced_rng, diagnostics=trace)

    assert ordinary == traced == []
    assert ordinary_rng.bit_generator.state == traced_rng.bit_generator.state
    assert trace[0]["attempts"]
    assert not any(run["decision"] == "accepted" for attempt in trace[0]["attempts"]
                   for run in attempt["runs"])


def test_same_tempo_copy_is_found_with_unit_ratio():
    recording = _music(30, 4)
    found = sm.stretched_matches(recording + _speech(30, 5), recording + _speech(30, 6), [(10.0, 0.0)])
    assert found and abs(found[0].rate - 1.0) < 0.002


def test_different_music_is_not_a_match():
    found = sm.stretched_matches(_music(30, 7) + _speech(30, 8), _music(30, 9) + _speech(30, 10),
                                 [(5.0, 0.0), (15.0, 1.0), (20.0, -2.0)])
    assert found == []


def test_speed_change_is_not_claimed_as_stretch():
    """Resampling changes pitch: not the same waveform, so this tracker does not claim it."""
    from scipy.signal import resample_poly

    recording = _music(30, 11)
    faster = resample_poly(recording, 20, 21)
    assert sm.stretched_matches(recording, faster, [(5.0, 0.0)]) == []


def test_tempo_ratio_beyond_the_limit_is_rejected():
    recording = _music(40, 12)
    second = _stretch(recording, 1.0 / 1.2)[:int(30 * RATE)]
    found = sm.stretched_matches(recording[:int(30 * RATE)], second, [(5.0, 1.0)])
    assert all(abs(match.rate - 1.0) <= sm.MAX_RATE_DEVIATION + 1e-6 for match in found)
