"""Read only this chat's local usage metadata; never invoke a model or resume a chat."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import time

ROOT = Path(__file__).resolve().parents[1]


def inspect(path: Path, *, now: float | None = None, reserve_percent: float = 5.,
            secondary_policy: str = "spend", secondary_reserve_percent: float = 20.,
            pace_safety_fraction: float = .9,
            unknown_rate_probe_fraction: float = .05) -> dict:
    now = time.time() if now is None else now
    if secondary_policy not in {"spend", "paced"}:
        raise ValueError("secondary_policy must be 'spend' or 'paced'")
    for name, value in (("reserve_percent", reserve_percent),
                        ("secondary_reserve_percent", secondary_reserve_percent),
                        ("pace_safety_fraction", pace_safety_fraction),
                        ("unknown_rate_probe_fraction", unknown_rate_probe_fraction)):
        if not 0. <= value <= 100.:
            raise ValueError(f"{name} must be between 0 and 100")
    if pace_safety_fraction > 1. or unknown_rate_probe_fraction > 1.:
        raise ValueError("pace_safety_fraction and unknown_rate_probe_fraction must be fractions from 0 to 1")
    with path.open("rb") as stream:
        stream.seek(0, 2)
        length = stream.tell()
        stream.seek(max(0, length - 2_000_000))
        lines = stream.read().splitlines()
    rows = []
    for line in lines:
        try:
            record = json.loads(line)
            payload = record.get("payload") or {}
            limits = payload.get("rate_limits")
            if record.get("type") != "event_msg" or payload.get("type") != "token_count" or not limits:
                continue
            timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00")).timestamp()
            # Retain only complete numeric window observations. An empty or
            # partial later token_count must not erase a previously usable
            # window or make that older observation look newly sampled.
            valid_limits = {}
            for key in ("primary", "secondary"):
                window = limits.get(key)
                if not isinstance(window, dict):
                    continue
                used, reset = window.get("used_percent"), window.get("resets_at")
                if isinstance(used, (int, float)) and isinstance(reset, (int, float)):
                    valid_limits[key] = window
            if valid_limits:
                rows.append((timestamp, valid_limits))
        except (ValueError, KeyError, TypeError):
            continue
    if not rows:
        return {"status": "unknown", "reason": "no_local_snapshot", "within_reserve": False,
                "relative_wallclock_rate_multiplier": None,
                "configured_unknown_rate_probe_fraction": None}
    latest = {}
    for timestamp, limits in rows:
        for key, window in limits.items():
            latest[key] = (timestamp, window)
    # Primary freshness controls overall advice. In report-only weekly mode,
    # secondary freshness does not throttle daily work.
    stamp = latest.get("primary", (rows[-1][0], None))[0]
    status = "fresh" if 0 <= now - stamp <= 300 else "stale"
    windows = {}
    primary_rate_multiplier = None
    paced_secondary_rate_multiplier = None
    for key, (window_stamp, window) in latest.items():
        used, reset = window["used_percent"], window["resets_at"]
        if key == "secondary" and secondary_policy == "paced" and not 0 <= now - window_stamp <= 300:
            status = "stale"
        remaining = reset - now
        if remaining <= 0:
            windows[key] = {"status": "expired_snapshot", "used_percent": used,
                            "observed_at": window_stamp, "resets_at": reset}
            # A weekly report-only window cannot invalidate fresh primary
            # pacing advice merely because its displayed reset has passed.
            if key != "secondary" or secondary_policy == "paced":
                status = "stale"
            continue
        is_secondary = key == "secondary"
        policy = secondary_policy if is_secondary else "paced"
        reserve = secondary_reserve_percent if is_secondary and policy == "paced" else (0. if is_secondary else reserve_percent)
        budget = max(0., 100. - reserve - used)
        # The default weekly policy reports remaining capacity but does not let
        # a long reset horizon throttle the user's chosen daily work rate.
        permitted = budget * 60 / remaining
        prior = [(t, w[key]) for t, w in rows if key in w and t < window_stamp - 60
                 and w[key].get("resets_at") == reset and t >= window_stamp - 3600]
        observed_rate = None
        planning_rate = None
        rate_status = "unknown"
        sample_seconds = None
        if prior:
            earlier, value = prior[0]
            old_used = value.get("used_percent")
            if isinstance(old_used, (int, float)) and old_used <= used:
                sample_seconds = window_stamp - earlier
                observed_delta = used - old_used
                observed_rate = observed_delta * 60 / sample_seconds
                # Display percentages are rounded. A displayed zero delta does
                # not establish zero consumption, so include one point of
                # quantization uncertainty over the actual sample interval.
                planning_rate = (observed_delta + 1.) * 60 / sample_seconds
                rate_status = "rounded_upper_bound"
        if planning_rate is None:
            rate_multiplier = None
            probe_fraction = unknown_rate_probe_fraction
            sampling_advice = ("No usable same-reset rate sample; collect another fresh snapshot. "
                               "The configured probe is a bounded sampling choice, not a measured "
                               "absolute active-duty recommendation.")
        else:
            rate_multiplier = min(1., pace_safety_fraction * permitted / planning_rate)
            probe_fraction = None
            sampling_advice = ("The planning rate is an upper bound including up to 1 percentage point "
                               "of display quantization, not a measured continuous rate. The multiplier "
                               "compares allowed percent/minute with a recent wall-clock planning rate, "
                               "which already includes prior rests; it is a relative-rate diagnostic, "
                               "not an absolute future active-duty fraction.")
        if budget <= 0 and (not is_secondary or policy == "paced"):
            if planning_rate is not None:
                rate_multiplier = 0.
            probe_fraction = 0.
        if not is_secondary:
            primary_rate_multiplier = rate_multiplier
        elif policy == "paced":
            paced_secondary_rate_multiplier = rate_multiplier
        windows[key] = {"used_percent": used, "reserve_percent": reserve,
                        "resets_at": reset, "seconds_to_reset": remaining,
                        "planning_horizon_seconds": remaining,
                        "budget_before_reserve_percent": budget,
                        "policy": policy,
                        "hard_exhausted": used >= 100.,
                        "permitted_percent_per_minute": permitted if policy == "paced" else None,
                        "rate_status": rate_status,
                        "recent_percent_per_minute": observed_rate,
                        "planning_rate_upper_bound_percent_per_minute": planning_rate,
                        "rate_sample_seconds": sample_seconds,
                        "sampling_advice": sampling_advice,
                        "relative_wallclock_rate_multiplier": rate_multiplier if policy == "paced" else None,
                        "unknown_rate_probe_fraction": probe_fraction if policy == "paced" and planning_rate is None else None}
    if not windows:
        status = "unknown"
    multipliers = [value for value in (primary_rate_multiplier, paced_secondary_rate_multiplier)
                   if value is not None]
    reserve_advice = []
    for row in windows.values():
        if not isinstance(row, dict) or row.get("policy") != "paced":
            continue
        multiplier = row.get("relative_wallclock_rate_multiplier")
        probe = row.get("unknown_rate_probe_fraction")
        if multiplier is not None:
            reserve_advice.append(multiplier)
        elif probe is not None:
            reserve_advice.append(probe)
    probes = [row.get("unknown_rate_probe_fraction") for row in windows.values()
              if isinstance(row, dict) and row.get("unknown_rate_probe_fraction") is not None]
    secondary = windows.get("secondary", {})
    secondary_exhausted = bool(secondary.get("hard_exhausted"))
    return {"status": status, "observed_at": stamp, "read_at": now,
            "windows": windows,
            "relative_wallclock_rate_multiplier": min(multipliers) if multipliers else None,
            "configured_unknown_rate_probe_fraction": min(probes) if probes else None,
            "within_reserve": status == "fresh" and bool(reserve_advice) and min(reserve_advice) > 0,
            "secondary_hard_exhausted": secondary_exhausted,
            "continuity_warning": ("secondary allowance is exhausted; primary-window advice cannot guarantee future turns"
                                   if secondary_exhausted else None),
            "policy": {"primary_reserve_percent": reserve_percent,
                       "secondary_policy": secondary_policy,
                       "secondary_reserve_percent_when_paced": secondary_reserve_percent,
                       "pace_safety_fraction": pace_safety_fraction,
                       "unknown_rate_probe_fraction": unknown_rate_probe_fraction},
            "scope": "local snapshot and pacing advice only; cannot control or auto-resume desktop turns"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=dev_state_root(ROOT) / "development-allowance")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--reserve-percent", type=float, default=5.,
                        help="primary-window reserve percent (default: 5)")
    parser.add_argument("--primary-window-seconds", type=float, default=18000.,
                        help="configured five-hour pacing policy, not an inferred server duration (default: 18000)")
    parser.add_argument("--secondary-policy", choices=("spend", "paced"), default="spend",
                        help="weekly window is report-only by default; use paced to retain a separate weekly throttle")
    parser.add_argument("--secondary-reserve-percent", type=float, default=20.,
                        help="weekly reserve when --secondary-policy=paced (default: 20)")
    parser.add_argument("--pace-safety-fraction", type=float, default=.9,
                        help="fraction of measured allowed rate when pacing (default: 0.9)")
    parser.add_argument("--unknown-rate-probe-fraction", type=float, default=.05,
                        help="configured bounded probe fraction while rate is unknown; not a measured duty recommendation (default: 0.05)")
    args = parser.parse_args()
    output = args.output.resolve()
    if dev_state_root(ROOT).resolve() not in output.parents:
        raise ValueError("Output must stay beneath the development state root")
    output.mkdir(parents=True, exist_ok=True)
    from dev_allowance_cadence import decide

    while not (output / "stop").exists():
        try:
            result = inspect(args.log, reserve_percent=args.reserve_percent,
                             secondary_policy=args.secondary_policy,
                             secondary_reserve_percent=args.secondary_reserve_percent,
                             pace_safety_fraction=args.pace_safety_fraction,
                             unknown_rate_probe_fraction=args.unknown_rate_probe_fraction)
        except OSError as exc:
            result = {"status": "unknown", "reason": type(exc).__name__, "within_reserve": False}
        prior_metadata = []
        history_path = output / "history.jsonl"
        if history_path.is_file():
            with history_path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 262144))
                for line in stream.read().splitlines():
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            prior_metadata.append(row)
                    except (ValueError, TypeError):
                        continue
        result["cadence"] = decide(result, prior_metadata, base_reserve_percent=args.reserve_percent,
                                   primary_window_seconds=args.primary_window_seconds)
        result["cadence"]["configured_primary_window_seconds"] = args.primary_window_seconds
        # Own development metadata; no source conversation text is retained.
        temporary = output / "latest.tmp"
        temporary.write_text(json.dumps(result, indent=2), encoding="utf8")
        temporary.replace(output / "latest.json")
        with (output / "history.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(result) + "\n")
        if not args.watch:
            print(json.dumps(result))
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
