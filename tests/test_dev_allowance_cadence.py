import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts/dev_allowance_cadence.py"
SPEC = importlib.util.spec_from_file_location("allowance_cadence", MODULE)
cadence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cadence)


NOW = 1_767_225_600
RESET = NOW + 10_000


def snapshot(used=40, *, stamp=NOW, reset=RESET, status="fresh", secondary=None):
    windows = {"primary": {"used_percent": used, "resets_at": reset}}
    if secondary is not None:
        windows["secondary"] = secondary
    return {"status": status, "observed_at": stamp, "read_at": NOW,
            "windows": windows}


def history(*points):
    return [{"observed_at": stamp,
             "windows": {"primary": {"used_percent": used, "resets_at": reset}}}
            for stamp, used, reset in points]


def test_five_percent_remaining_holds_for_one_reset_wakeup():
    decision = cadence.decide(snapshot(95))
    assert decision["action"] == cadence.HOLD_UNTIL_RESET
    assert decision["dynamic_reserve_percent"] == 5
    assert decision["reset_deadline"] == RESET
    assert decision["wake_at"] == RESET


def test_unknown_stale_null_and_expired_primary_fail_closed_without_inferred_reset():
    unknown = cadence.decide({"status": "unknown", "read_at": NOW})
    null = cadence.decide(snapshot(None))
    stale = cadence.decide(snapshot(20, status="stale"))
    expired = cadence.decide(snapshot(20, reset=NOW - 1))
    for decision in (unknown, null, stale, expired):
        assert decision["action"] == cadence.HOLD_FOR_FRESH_PRIMARY
        assert decision["dynamic_reserve_percent"] is None
        assert decision["wake_at"] is None
    assert expired["known_reset_deadline"] is None


def test_different_reset_history_does_not_supply_a_jump():
    old = RESET - 1
    decision = cadence.decide(snapshot(90), history((NOW - 60, 10, old)))
    assert decision["action"] == cadence.WORK
    assert decision["measured_interval_jump_percent"] == 0
    assert decision["dynamic_reserve_percent"] == 5


def test_recent_large_same_reset_jump_expands_closeout_reserve():
    prior = history((NOW - 180, 94, RESET), (NOW - 120, 96, RESET),
                    (NOW - 60, 99, RESET))
    decision = cadence.decide(snapshot(99, stamp=NOW), prior)
    assert decision["measured_interval_jump_percent"] == 3
    assert decision["dynamic_reserve_percent"] == 6
    assert decision["action"] == cadence.HOLD_UNTIL_RESET
    assert "not guaranteed per-request bounds" in decision["reserve_basis"]


def test_long_gap_jump_is_diagnostic_and_does_not_expand_closeout_reserve():
    prior = history((NOW - 40 * 60, 38, RESET))
    decision = cadence.decide(snapshot(62), prior)
    assert decision["measured_interval_jump_percent"] == 0
    assert decision["dynamic_reserve_percent"] == 5
    assert decision["largest_ignored_gap_increase_percent"] == 24
    gap = decision["ignored_gap_diagnostics"][0]
    assert gap["elapsed_seconds"] == 40 * 60
    assert gap["cumulative_increase_percent"] == 24
    assert gap["rate_percent_per_hour"] == 36
    assert "not a bounded closeout cost" in gap["reason"]
    assert "base reserve only" in decision["measurement_uncertainty"]


def test_short_three_point_jump_still_expands_reserve_to_six():
    prior = history((NOW - 120, 57, RESET), (NOW - 60, 60, RESET))
    decision = cadence.decide(snapshot(60, stamp=NOW), prior)
    assert decision["measured_interval_jump_percent"] == 3
    assert decision["dynamic_reserve_percent"] == 6
    assert decision["ignored_gap_diagnostics"] == []
    assert "do not bound per-request" in decision["measurement_uncertainty"]


def test_long_gap_then_short_increase_keeps_five_percent_reserve_near_reset():
    reset = NOW + 31 * 60
    prior = history((NOW - 40 * 60, 38, reset), (NOW - 60, 62, reset))
    current = snapshot(63, reset=reset)
    decision = cadence.decide(current, prior, primary_window_seconds=5 * 60 * 60)
    assert decision["largest_ignored_gap_increase_percent"] == 24
    assert decision["measured_interval_jump_percent"] == 1
    assert decision["dynamic_reserve_percent"] == 5
    assert decision["action"] == cadence.WORK


def test_duplicate_and_older_reset_samples_cannot_create_reserve_jumps():
    other_reset = RESET - 1
    prior = history((NOW - 60, 20, RESET), (NOW - 60, 99, RESET),
                    (NOW - 30, 10, other_reset))
    decision = cadence.decide(snapshot(20, stamp=NOW), prior)
    assert decision["measured_interval_jump_percent"] == 0
    assert decision["dynamic_reserve_percent"] == 5
    assert decision["ignored_gap_diagnostics"] == []
    assert decision["duplicate_timestamp_count"] == 1


def test_weekly_spend_window_never_time_throttles_primary_work():
    secondary = {"used_percent": 100, "resets_at": NOW + 900_000,
                 "policy": "spend", "hard_exhausted": True}
    decision = cadence.decide(snapshot(40, secondary=secondary))
    assert decision["action"] == cadence.WORK
    assert decision["weekly_policy"] == "report_only"
    assert decision["secondary_hard_exhausted"] is True
    assert decision["reset_deadline"] is None


def test_primary_hard_exhaustion_is_separate_from_dynamic_reserve():
    decision = cadence.decide(snapshot(100))
    assert decision["action"] == cadence.EXHAUSTED
    assert decision["reset_deadline"] == RESET


def test_configured_five_hour_window_quiets_early_34_percent_burst():
    window = 5 * 60 * 60
    start = NOW - 78 * 60
    reset = start + window
    current = snapshot(34, reset=reset)
    current["read_at"] = NOW
    decision = cadence.decide(current, now=NOW, primary_window_seconds=window)
    assert decision["action"] == cadence.PACED_QUIET
    assert decision["pacing_envelope_percent"] == 26.7
    assert decision["wake_at"] == start + (34 - 2) / 95 * window
    assert decision["wake_at"] < reset
    assert decision["weekly_policy"] == "report_only"


def test_linear_envelope_supports_zero_or_two_percent_burst():
    window = 5 * 60 * 60
    start = NOW - 78 * 60
    reset = start + window
    current = snapshot(25, reset=reset)
    current["read_at"] = NOW
    no_burst = cadence.decide(current, now=NOW, primary_window_seconds=window,
                              burst_allowance_percent=0)
    default_burst = cadence.decide(current, now=NOW, primary_window_seconds=window)
    assert no_burst["action"] == cadence.PACED_QUIET
    assert no_burst["wake_at"] == start + 25 / 95 * window
    assert default_burst["action"] == cadence.WORK
    assert default_burst["pacing_envelope_percent"] == 26.7


def test_pacing_deadline_near_reset_is_capped_at_reset():
    window = 5 * 60 * 60
    start = NOW - 4 * 60 * 60
    reset = start + window
    current = snapshot(90, reset=reset)
    current["read_at"] = NOW
    decision = cadence.decide(current, now=NOW, primary_window_seconds=window)
    assert decision["action"] == cadence.PACED_QUIET
    assert decision["wake_at"] <= reset
    assert decision["wake_at"] == start + 88 / 95 * window


def test_configured_pacing_keeps_unknown_primary_fail_closed():
    decision = cadence.decide({"status": "unknown", "read_at": NOW},
                              now=NOW, primary_window_seconds=18_000)
    assert decision["action"] == cadence.HOLD_FOR_FRESH_PRIMARY
    assert decision["wake_at"] is None


def test_pacing_parameters_must_be_finite_and_valid():
    for value in (0, -1, float("inf"), float("nan")):
        try:
            cadence.decide(snapshot(), primary_window_seconds=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid window {value!r}")
    for value in (0, -1, float("inf"), float("nan")):
        try:
            cadence.decide(snapshot(), max_measurement_interval_seconds=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid interval limit {value!r}")
