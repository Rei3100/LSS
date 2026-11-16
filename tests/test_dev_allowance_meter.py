import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts/dev_allowance_meter.py"
SPEC = importlib.util.spec_from_file_location("allowance_meter", MODULE)
meter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(meter)


def write(log, percent, timestamp="2026-01-01T00:00:00Z", reset=1767240000):
    row = {"timestamp": timestamp, "type": "event_msg", "payload": {"type": "token_count",
           "rate_limits": {"primary": {"used_percent": percent, "resets_at": reset}}}}
    with log.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row) + "\n")


def write_windows(log, timestamp, primary, secondary):
    row = {"timestamp": timestamp, "type": "event_msg", "payload": {"type": "token_count",
           "rate_limits": {"primary": primary, "secondary": secondary}}}
    with log.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row) + "\n")


def write_limits(log, timestamp, limits):
    row = {"timestamp": timestamp, "type": "event_msg", "payload":
           {"type": "token_count", "rate_limits": limits}}
    with log.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row) + "\n")


def test_stale_or_missing_data_never_assumes_a_fresh_allowance(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{}\n', encoding="utf8")
    assert not meter.inspect(log, now=1767225600)["within_reserve"]
    write(log, 10)
    assert meter.inspect(log, now=1767225601)["within_reserve"]
    assert not meter.inspect(log, now=1767226000)["within_reserve"]


def test_reserve_and_history_reduce_relative_wallclock_multiplier(tmp_path):
    log = tmp_path / "log.jsonl"
    write(log, 10)
    write(log, 30, "2026-01-01T00:10:00Z")
    data = meter.inspect(log, now=1767226200)
    assert 0 < data["relative_wallclock_rate_multiplier"] < 1
    write(log, 80, "2026-01-01T00:10:00Z")
    high_use = meter.inspect(log, now=1767226200)
    assert high_use["relative_wallclock_rate_multiplier"] < data["relative_wallclock_rate_multiplier"]
    write(log, 95, "2026-01-01T00:10:00Z")
    assert not meter.inspect(log, now=1767226200)["within_reserve"]


def test_reset_does_not_mix_history_from_the_previous_window(tmp_path):
    log = tmp_path / "log.jsonl"
    write(log, 70, reset=1767225900)
    write(log, 1, "2026-01-01T00:10:00Z")
    data = meter.inspect(log, now=1767226200)
    assert data["windows"]["primary"]["recent_percent_per_minute"] is None
    assert data["windows"]["primary"]["used_percent"] == 1


def test_default_spend_policy_does_not_throttle_daily_work_by_weekly_remaining(tmp_path):
    log = tmp_path / "log.jsonl"
    now = 1767225600
    logs = []
    for index, weekly_used in enumerate((46, 80, 100)):
        path = tmp_path / f"log-{index}.jsonl"
        write_windows(path, "2026-01-01T00:00:00Z",
                      {"used_percent": 30, "resets_at": now + 17931.6},
                      {"used_percent": 40, "resets_at": now + 549914.6})
        current = now + 600
        write_windows(path, "2026-01-01T00:10:00Z",
                      {"used_percent": 40, "resets_at": now + 17931.6},
                      {"used_percent": weekly_used, "resets_at": now + 549914.6})
        logs.append(meter.inspect(path, now=current))
    fractions = [item["relative_wallclock_rate_multiplier"] for item in logs]
    assert fractions[0] == fractions[1] == fractions[2]
    assert logs[0]["windows"]["secondary"]["budget_before_reserve_percent"] == 54
    assert logs[1]["windows"]["secondary"]["budget_before_reserve_percent"] == 20
    assert logs[2]["secondary_hard_exhausted"] is True
    assert "cannot guarantee future turns" in logs[2]["continuity_warning"]
    assert logs[2]["relative_wallclock_rate_multiplier"] > 0
    assert logs[0]["policy"]["secondary_policy"] == "spend"


def test_primary_reserve_stops_advice_at_ninety_five_percent(tmp_path):
    log = tmp_path / "log.jsonl"
    now = 1767225600
    write_windows(log, "2026-01-01T00:00:00Z",
                  {"used_percent": 94, "resets_at": now + 10000},
                  {"used_percent": 46, "resets_at": now + 500000})
    data = meter.inspect(log, now=now)
    assert data["windows"]["primary"]["reserve_percent"] == 5
    assert data["windows"]["primary"]["budget_before_reserve_percent"] == 1
    assert data["configured_unknown_rate_probe_fraction"] == .05
    write_windows(log, "2026-01-01T00:01:00Z",
                  {"used_percent": 95, "resets_at": now + 10000},
                  {"used_percent": 46, "resets_at": now + 500000})
    stopped = meter.inspect(log, now=now + 60)
    assert stopped["windows"]["primary"]["budget_before_reserve_percent"] == 0
    assert stopped["configured_unknown_rate_probe_fraction"] == 0
    assert stopped["relative_wallclock_rate_multiplier"] is None
    assert not stopped["within_reserve"]


def test_explicit_weekly_paced_policy_keeps_its_own_horizon(tmp_path):
    log = tmp_path / "log.jsonl"
    now = 1767225600
    write_windows(log, "2026-01-01T00:00:00Z",
                  {"used_percent": 0, "resets_at": now + 17931.6},
                  {"used_percent": 41, "resets_at": now + 549914.6})
    data = meter.inspect(log, now=now, secondary_policy="paced")
    primary = data["windows"]["primary"]
    secondary = data["windows"]["secondary"]
    assert abs(primary["planning_horizon_seconds"] - 17931.6) < 1e-6
    assert abs(secondary["planning_horizon_seconds"] - 549914.6) < 1e-6
    assert secondary["budget_before_reserve_percent"] == 39
    assert abs(secondary["permitted_percent_per_minute"] - 39 * 60 / 549914.6) < 1e-12
    assert secondary["permitted_percent_per_minute"] < .005


def test_within_reserve_reflects_explicit_weekly_constraint(tmp_path):
    log = tmp_path / "log.jsonl"
    now = 1767225600
    write_windows(log, "2026-01-01T00:00:00Z",
                  {"used_percent": 30, "resets_at": now + 10000},
                  {"used_percent": 80, "resets_at": now + 500000})
    data = meter.inspect(log, now=now, secondary_policy="paced")
    assert data["windows"]["primary"]["unknown_rate_probe_fraction"] == .05
    assert data["windows"]["secondary"]["relative_wallclock_rate_multiplier"] is None
    assert data["windows"]["secondary"]["unknown_rate_probe_fraction"] == 0
    assert data["relative_wallclock_rate_multiplier"] is None
    assert data["configured_unknown_rate_probe_fraction"] == 0
    assert data["within_reserve"] is False


def test_report_only_weekly_expiry_does_not_stale_primary_advice(tmp_path):
    now = 1767225600
    log = tmp_path / "weekly-expired.jsonl"
    write_windows(log, "2026-01-01T00:00:00Z",
                  {"used_percent": 30, "resets_at": now + 10000},
                  {"used_percent": 46, "resets_at": now - 1})
    data = meter.inspect(log, now=now)
    assert data["status"] == "fresh"
    assert data["windows"]["secondary"]["status"] == "expired_snapshot"
    assert data["within_reserve"] is True
    assert data["configured_unknown_rate_probe_fraction"] == .05

    primary_expired = tmp_path / "primary-expired.jsonl"
    write_windows(primary_expired, "2026-01-01T00:00:00Z",
                  {"used_percent": 30, "resets_at": now - 1},
                  {"used_percent": 46, "resets_at": now + 500000})
    blocked = meter.inspect(primary_expired, now=now)
    assert blocked["status"] == "stale"
    assert blocked["within_reserve"] is False


def test_empty_latest_window_keeps_usage_and_original_freshness(tmp_path):
    now = 1767225600
    log = tmp_path / "empty-latest.jsonl"
    write_windows(log, "2026-01-01T00:00:00Z",
                  {"used_percent": 96, "resets_at": now + 10000},
                  {"used_percent": 46, "resets_at": now + 500000})
    write_limits(log, "2026-01-01T00:04:59Z", {"primary": None, "secondary": None})
    data = meter.inspect(log, now=now + 301)
    assert data["status"] == "stale"
    assert data["observed_at"] == now
    assert data["windows"]["primary"]["used_percent"] == 96
    assert data["windows"]["primary"]["budget_before_reserve_percent"] == 0
    assert data["configured_unknown_rate_probe_fraction"] == 0
    assert data["within_reserve"] is False


def test_partial_null_secondary_preserves_primary_observation(tmp_path):
    now = 1767225600
    log = tmp_path / "partial-null.jsonl"
    write_windows(log, "2026-01-01T00:00:00Z",
                  {"used_percent": 96, "resets_at": now + 10000},
                  {"used_percent": 46, "resets_at": now + 500000})
    write_limits(log, "2026-01-01T00:01:00Z", {
        "primary": None,
        "secondary": {"used_percent": 80, "resets_at": now + 500000},
    })
    data = meter.inspect(log, now=now + 60)
    assert data["status"] == "fresh"
    assert data["observed_at"] == now
    assert data["windows"]["primary"]["used_percent"] == 96
    assert data["windows"]["secondary"]["used_percent"] == 80
    assert data["policy"]["secondary_policy"] == "spend"
    assert data["relative_wallclock_rate_multiplier"] is None
    assert data["within_reserve"] is False


def test_empty_latest_cannot_hide_expired_primary_reset(tmp_path):
    now = 1767225600
    log = tmp_path / "empty-after-expiry.jsonl"
    write_windows(log, "2026-01-01T00:00:00Z",
                  {"used_percent": 96, "resets_at": now - 1},
                  {"used_percent": 46, "resets_at": now + 500000})
    write_limits(log, "2026-01-01T00:04:59Z", {"primary": None})
    data = meter.inspect(log, now=now + 1)
    assert data["status"] == "stale"
    assert data["windows"]["primary"]["status"] == "expired_snapshot"
    assert data["within_reserve"] is False


def test_unknown_rate_uses_configured_small_probe_fraction(tmp_path):
    log = tmp_path / "log.jsonl"
    write(log, 10)
    data = meter.inspect(log, now=1767225601)
    window = data["windows"]["primary"]
    assert data["relative_wallclock_rate_multiplier"] is None
    assert data["configured_unknown_rate_probe_fraction"] == .05
    assert window["relative_wallclock_rate_multiplier"] is None
    assert window["unknown_rate_probe_fraction"] == .05
    assert window["rate_status"] == "unknown"
    assert window["recent_percent_per_minute"] is None
    assert window["planning_rate_upper_bound_percent_per_minute"] is None
    assert "collect another fresh snapshot" in window["sampling_advice"]
    custom = meter.inspect(log, now=1767225601, unknown_rate_probe_fraction=.02)
    assert custom["relative_wallclock_rate_multiplier"] is None
    assert custom["configured_unknown_rate_probe_fraction"] == .02


def test_rounded_zero_rate_is_not_treated_as_proof_of_zero_consumption(tmp_path):
    log = tmp_path / "log.jsonl"
    write(log, 10, "2026-01-01T00:00:00Z")
    write(log, 10, "2026-01-01T00:10:00Z")
    window = meter.inspect(log, now=1767226200)["windows"]["primary"]
    assert window["recent_percent_per_minute"] == 0
    assert window["planning_rate_upper_bound_percent_per_minute"] == .1
    assert window["rate_sample_seconds"] == 600
    assert window["rate_status"] == "rounded_upper_bound"
    assert "not a measured continuous rate" in window["sampling_advice"]


def test_wallclock_multiplier_is_not_an_absolute_future_duty_fraction(tmp_path):
    log = tmp_path / "log.jsonl"
    now = 1767225600
    reset = now + 3700 + 24300
    write(log, 0, "2026-01-01T00:00:00Z", reset)
    write(log, 14, "2026-01-01T01:00:00Z", reset)
    data = meter.inspect(log, now=now + 3700)
    window = data["windows"]["primary"]
    assert window["permitted_percent_per_minute"] == .2
    assert abs(window["recent_percent_per_minute"] - (14 * 60 / 3600)) < 1e-12
    assert window["planning_rate_upper_bound_percent_per_minute"] == .25
    # With prior duty=25% and active use=1 pp/min, .25 pp/min is a
    # wall-clock observation; .9 * .2 / .25 = .72 is only a relative
    # rate multiplier, not a recommendation to run at 72% future duty.
    assert abs(window["relative_wallclock_rate_multiplier"] - .72) < 1e-12
    assert window["unknown_rate_probe_fraction"] is None
    assert "not an absolute future active-duty fraction" in window["sampling_advice"]
