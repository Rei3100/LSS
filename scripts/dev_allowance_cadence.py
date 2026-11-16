"""Pure advisory gate for quota-consuming development work.

Inputs are quota-only metadata, typically the result of ``dev_allowance_meter.inspect``
and a list of its prior snapshots. This module does not read logs, schedule work,
or control application requests. A HOLD_UNTIL_RESET decision describes one
deadline wakeup; callers must not turn it into periodic agent/model polling.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


WORK = "WORK"
HOLD_FOR_FRESH_PRIMARY = "HOLD_FOR_FRESH_PRIMARY"
HOLD_UNTIL_RESET = "HOLD_UNTIL_RESET"
PACED_QUIET = "PACED_QUIET"
EXHAUSTED = "EXHAUSTED"


def _number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _primary(metadata: Mapping):
    windows = metadata.get("windows")
    primary = windows.get("primary") if isinstance(windows, Mapping) else None
    if not isinstance(primary, Mapping):
        return None
    used, reset = primary.get("used_percent"), primary.get("resets_at")
    if not _number(used) or not _number(reset):
        return None
    return used, reset


def _secondary_exhausted(metadata: Mapping) -> bool:
    windows = metadata.get("windows")
    secondary = windows.get("secondary") if isinstance(windows, Mapping) else None
    used = secondary.get("used_percent") if isinstance(secondary, Mapping) else None
    return _number(used) and used >= 100.


def decide(snapshot: Mapping, history_metadata: Iterable[Mapping] = (), *,
           now: float | None = None, base_reserve_percent: float = 5.,
           rounding_margin_percent: float = 1.,
           closeout_margin_percent: float = 2.,
           primary_window_seconds: float | None = None,
           burst_allowance_percent: float = 2.,
           max_measurement_interval_seconds: float = 300.) -> dict:
    """Return a quota-only advisory decision.

    ``history_metadata`` contains prior quota-only snapshot dictionaries with
    ``observed_at`` and ``windows.primary.{used_percent,resets_at}``. The
    Only nonnegative increases across same-reset observations no more than
    ``max_measurement_interval_seconds`` apart, in the prior hour, contribute to
    the closeout reserve. Longer gaps are retained as uncertainty/rate
    diagnostics because their cumulative increase is not a bounded closeout
    cost. Even short interval jumps are observations, not guaranteed per-request
    consumption bounds. The dynamic reserve is ``max(base, short_jump +
    rounding + closeout)``. Usage is compared with that reserve directly, so the
    rounding margin is not added to usage a second time.

    Optional ``primary_window_seconds`` is an explicit pacing policy, not an
    inferred server window. It supplies one quiet deadline if usage is ahead
    of a linear envelope (with a small burst allowance); it never compounds a
    relative wall-clock rate multiplier or proves future capacity.

    The weekly secondary window is intentionally ignored, including its reset
    horizon and remaining capacity. Only an explicitly exhausted secondary
    window is outside this gate's scope. No decision guarantees that a request
    will fit or prevents the host application from consuming quota.
    """
    for name, value in (("base_reserve_percent", base_reserve_percent),
                        ("rounding_margin_percent", rounding_margin_percent),
                        ("closeout_margin_percent", closeout_margin_percent)):
        if not _number(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if primary_window_seconds is not None and (
            not _number(primary_window_seconds) or primary_window_seconds <= 0):
        raise ValueError("primary_window_seconds must be a finite positive number or None")
    if not _number(burst_allowance_percent) or burst_allowance_percent < 0:
        raise ValueError("burst_allowance_percent must be a finite nonnegative number")
    if (not _number(max_measurement_interval_seconds)
            or max_measurement_interval_seconds <= 0):
        raise ValueError("max_measurement_interval_seconds must be a finite positive number")
    if not isinstance(snapshot, Mapping):
        snapshot = {}
    now = snapshot.get("read_at") if now is None else now
    if not _number(now):
        now = None
    primary = _primary(snapshot)
    observed_at = snapshot.get("observed_at")
    if (primary is None or not _number(observed_at) or now is None
            or snapshot.get("status") != "fresh"):
        return {"action": HOLD_FOR_FRESH_PRIMARY,
                "reason": "A fresh primary-window observation with a known reset is required.",
                "dynamic_reserve_percent": None,
                "measured_interval_jump_percent": None,
                "reset_deadline": None,
                "wake_at": None,
                "weekly_policy": "report_only"}

    used, reset = primary
    # Do not derive fresh capacity from an expired snapshot, even if its old
    # status field claims freshness.
    if not (0 <= now - observed_at <= 300) or reset <= now:
        return {"action": HOLD_FOR_FRESH_PRIMARY,
                "reason": "The primary observation is stale or its reset has passed; confirm the new window.",
                "observed_at": observed_at,
                "known_reset_deadline": reset if reset > now else None,
                "reset_deadline": None,
                "wake_at": None,
                "dynamic_reserve_percent": None,
                "measured_interval_jump_percent": None,
                "weekly_policy": "report_only"}

    prior = []
    if isinstance(history_metadata, Iterable):
        for item in history_metadata:
            if not isinstance(item, Mapping):
                continue
            item_primary = _primary(item)
            stamp = item.get("observed_at")
            if (item_primary is None or not _number(stamp)
                    or item_primary[1] != reset or stamp >= observed_at
                    or stamp < observed_at - 3600):
                continue
            prior.append((stamp, item_primary[0]))
    prior.sort()
    by_timestamp = {}
    for stamp, value in prior:
        by_timestamp.setdefault(stamp, set()).add(value)
    duplicate_timestamp_count = sum(len(values) > 1 for values in by_timestamp.values())
    measurements = [(stamp, next(iter(values)))
                    for stamp, values in sorted(by_timestamp.items()) if len(values) == 1]
    measurements.append((observed_at, used))
    short_jumps = []
    ignored_gaps = []
    for (earlier_at, earlier_used), (later_at, later_used) in zip(
            measurements, measurements[1:]):
        spacing = later_at - earlier_at
        jump = max(0., later_used - earlier_used)
        if spacing <= max_measurement_interval_seconds:
            short_jumps.append(jump)
        else:
            ignored_gaps.append({
                "elapsed_seconds": spacing,
                "cumulative_increase_percent": jump,
                "rate_percent_per_hour": jump * 3600. / spacing,
                "reason": "gap exceeds closeout measurement interval; cumulative increase is not a bounded closeout cost",
            })
    max_jump = max(short_jumps, default=0.)
    reserve = max(base_reserve_percent,
                  max_jump + rounding_margin_percent + closeout_margin_percent)
    result = {"observed_at": observed_at,
              "used_percent": used,
              "known_reset_deadline": reset,
              "dynamic_reserve_percent": reserve,
              "measured_interval_jump_percent": max_jump,
              "qualifying_short_interval_count": len(short_jumps),
              "measurement_interval_limit_seconds": max_measurement_interval_seconds,
              "duplicate_timestamp_count": duplicate_timestamp_count,
              "ignored_gap_diagnostics": ignored_gaps,
              "largest_ignored_gap_increase_percent": max(
                  (gap["cumulative_increase_percent"] for gap in ignored_gaps),
                  default=0.),
              "measurement_uncertainty": (
                  "short interval observations do not bound per-request consumption"
                  if short_jumps else
                  "no qualifying short interval; base reserve only and closeout cost remains unknown"),
              "reserve_basis": "max qualifying short same-reset interval jump plus rounding and closeout margins, or base reserve; observations are not guaranteed per-request bounds",
              "secondary_hard_exhausted": _secondary_exhausted(snapshot),
              "weekly_policy": "report_only",
              "wake_at": None}
    if used >= 100.:
        result.update(action=EXHAUSTED, reason="The fresh primary window is exhausted.",
                      reset_deadline=reset, wake_at=reset)
    elif used + reserve >= 100.:
        result.update(action=HOLD_UNTIL_RESET,
                      reason="Remaining primary allowance is within the dynamic closeout reserve.",
                      reset_deadline=reset, wake_at=reset)
    elif (primary_window_seconds is not None and 0 <= used < 100.
          and used > burst_allowance_percent):
        target = 100. - reserve
        window_start = reset - primary_window_seconds
        fraction = min(1., max(0., (now - window_start) / primary_window_seconds))
        envelope = target * fraction + burst_allowance_percent
        if used > envelope and target > 0:
            wake_at = min(reset, max(now, window_start +
                                     max(0., used - burst_allowance_percent) /
                                     target * primary_window_seconds))
            result.update(action=PACED_QUIET,
                          reason="Usage is ahead of the configured primary-window linear pacing envelope.",
                          pacing_window_start=window_start,
                          pacing_target_percent=target,
                          pacing_burst_allowance_percent=burst_allowance_percent,
                          pacing_envelope_percent=envelope,
                          reset_deadline=reset, wake_at=wake_at)
        else:
            result.update(action=WORK, reason="Fresh primary allowance is within the configured pacing envelope.",
                          reset_deadline=None,
                          pacing_window_start=window_start,
                          pacing_target_percent=target,
                          pacing_burst_allowance_percent=burst_allowance_percent,
                          pacing_envelope_percent=envelope)
    else:
        result.update(action=WORK, reason="Fresh primary allowance exceeds the dynamic closeout reserve.",
                      reset_deadline=None)
    result["scope"] = "advisory only; cannot control host application requests or guarantee future continuity"
    return result
