"""Same recording heard at another tempo: verify a time-stretched passage between two separated BGM cues.

TV editors often fit a score recording to a scene by changing its tempo while keeping its pitch (grain-based
time stretch). Two episodes then play the same recording with a lag that grows linearly through the passage
(real corpus-a pairs: +1.8 % to +5.9 % and +1.3 %). The landmark matcher (`identity.match`) needs a
constant offset (window drift <= one frame), so it never reaches waveform verification for these pairs, although
each short window is still the same waveform: grains are copied, not resynthesised. Resampling one side to the
other tempo destroys the match (the pitch would change), which is how a stretch differs from a speed change.

`stretched_matches` follows the lag window by window (1 s windows: a stretch moves the lag inside a longer window; measured on
real pairs, 1 s windows explain more than 2 s or 0.5 s) from a seed alignment, measures in every window how much of
the query the other cue explains through a short FIR at the tracked lag (as `identity.waveform_blocks` does at a
fixed lag), and keeps runs whose lags lie on one straight line (one tempo ratio) within ``LINE_TOLERANCE``. The same
tracker started from alignments that are known to be wrong gives the chance level; a run must clear it by
``MIN_MARGIN``. A different recording does not keep a straight lag path with explained waveform, so the line and the
null together decide; a bare fingerprint vote decides nothing.

Only for comparing two separated estimates of published cues (the easy direction). It is not a Reference EXACT
gate and never compares a cue with an original mixture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METHOD = "stretch-lag-path-v1"
SAMPLE_RATE = 11025
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
# Lag may move this much between neighbouring windows while tracking (6 % tempo over 0.5 s is 30 ms).
STEP_SEARCH_SECONDS = 0.06
SEED_SEARCH_SECONDS = 0.4
FIR_TAPS = 16
# A window counts as explained (same as `identity.MIN_BLOCK_EXPLAINED`).
MIN_WINDOW_EXPLAINED = 0.15
# Tempo ratios beyond this are not treated as the same recording.
MAX_RATE_DEVIATION = 0.1
LINE_TOLERANCE_SECONDS = 0.02
MIN_RUN_WINDOWS = 8
ALLOWED_MISSES = 2
MIN_MEAN_EXPLAINED = 0.25
MIN_MARGIN = 0.1
NULL_SEEDS = 6
NULL_MIN_DISTANCE_SECONDS = 5.0


@dataclass(frozen=True)
class StretchedMatch:
    query_start_seconds: float
    query_end_seconds: float
    other_start_seconds: float
    other_end_seconds: float
    rate: float            # other time advances ``rate`` seconds per query second
    lag_at_start_seconds: float
    windows: int
    mean_explained: float
    null_max: float
    evidence_kind: str = "stretched_lag_path"

    def to_dict(self) -> dict:
        return {"query_start_seconds": self.query_start_seconds, "query_end_seconds": self.query_end_seconds,
                "other_start_seconds": self.other_start_seconds, "other_end_seconds": self.other_end_seconds,
                "rate": self.rate, "lag_at_start_seconds": self.lag_at_start_seconds, "windows": self.windows,
                "mean_explained": self.mean_explained, "null_max": self.null_max,
                "evidence_kind": self.evidence_kind, "method": METHOD}


def _explained(target: np.ndarray, source: np.ndarray) -> float:
    from numpy.lib.stride_tricks import sliding_window_view

    energy = float(target @ target)
    if energy < 1e-6 * target.size or not np.any(source):
        return 0.0
    half = FIR_TAPS // 2
    design = sliding_window_view(np.pad(source, (half, FIR_TAPS - half - 1)), FIR_TAPS)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return max(0.0, 1.0 - float(residual @ residual) / energy)


def _best_lag(query: np.ndarray, other: np.ndarray, start: int, width: int, centre: int, search: int):
    """Lag (samples, other = query + lag) near ``centre`` with the highest normalised correlation, or None."""
    from scipy.signal import correlate

    segment = query[start:start + width]
    if segment.size < width or float(segment @ segment) < 1e-6 * width:
        return None
    low = max(0, start + centre - search)
    high = min(other.size - width, start + centre + search)
    if high <= low:
        return None
    region = other[low:high + width]
    values = correlate(region, segment, mode="valid", method="fft")
    cumulative = np.concatenate([[0.0], np.cumsum(region * region)])
    norm = np.sqrt(np.maximum(cumulative[width:] - cumulative[:-width], 0.0)) * np.sqrt(segment @ segment) + 1e-9
    index = int(np.argmax(values / norm))
    return low + index - start


def track(query: np.ndarray, other: np.ndarray, seed_seconds: float, seed_lag_seconds: float,
          rate: int = SAMPLE_RATE) -> list[tuple[int, int, float]]:
    """(window start, lag, explained) for every window reachable from the seed by small lag steps."""
    width, hop = int(WINDOW_SECONDS * rate), int(HOP_SECONDS * rate)
    step, seed_search = int(STEP_SEARCH_SECONDS * rate), int(SEED_SEARCH_SECONDS * rate)
    if query.size < width or other.size < width:
        return []
    seed = min(max(0, int(round(seed_seconds * rate)) // hop * hop), (query.size - width) // hop * hop)
    first = _best_lag(query, other, seed, width, int(round(seed_lag_seconds * rate)), seed_search)
    if first is None:
        return []
    rows = {}

    def visit(start: int, lag: int) -> int | None:
        found = _best_lag(query, other, start, width, lag, step)
        if found is None:
            return None
        rows[start] = (found, _explained(query[start:start + width], other[start + found:start + found + width]))
        return found

    visit(seed, first)
    for direction in (1, -1):
        lag = rows.get(seed, (first,))[0]
        start = seed + direction * hop
        while 0 <= start <= query.size - width:
            found = visit(start, lag)
            if found is None:
                break
            lag = found
            start += direction * hop
    return [(start, lag, value) for start, (lag, value) in sorted(rows.items())]


def _line_runs(rows: list[tuple[int, int, float]], rate: int, trace: dict | None = None) -> list[tuple[int, int, float, float]]:
    """Runs of explained windows on one lag line: (first index, last index, slope, intercept[samples])."""
    if len(rows) < MIN_RUN_WINDOWS:
        if trace is not None:
            trace["decision"] = "too_few_tracked_windows"
            trace["tracked_window_count"] = len(rows)
        return []
    starts = np.array([row[0] for row in rows], np.float64)
    lags = np.array([row[1] for row in rows], np.float64)
    values = np.array([row[2] for row in rows], np.float64)
    good = values >= MIN_WINDOW_EXPLAINED
    if good.sum() < MIN_RUN_WINDOWS:
        if trace is not None:
            trace["decision"] = "too_few_explained_windows"
            trace["tracked_window_count"] = len(rows)
            trace["explained_window_count"] = int(good.sum())
        return []
    # Robust line through the explained windows: least squares, then refit without outliers.
    inliers = good.copy()
    slope = intercept = 0.0
    iterations = [] if trace is not None else None
    for _ in range(3):
        if inliers.sum() < 2:
            if trace is not None:
                trace["decision"] = "fit_lost_minimum_inliers"
                trace["fit_iterations"] = iterations
            return []
        slope, intercept = np.polyfit(starts[inliers], lags[inliers], 1)
        residual = np.abs(lags - (slope * starts + intercept))
        next_inliers = good & (residual <= LINE_TOLERANCE_SECONDS * rate)
        if trace is not None:
            iterations.append({"slope_lag_samples_per_query_sample": float(slope),
                               "intercept_lag_samples": float(intercept),
                               "inlier_indexes": np.flatnonzero(next_inliers).astype(int).tolist()})
        inliers = next_inliers
    if abs(slope) > MAX_RATE_DEVIATION:
        if trace is not None:
            trace["decision"] = "rate_outside_limit"
            trace["fit_iterations"] = iterations
            trace["slope_lag_samples_per_query_sample"] = float(slope)
        return []
    residual = np.abs(lags - (slope * starts + intercept))
    on_line = good & (residual <= LINE_TOLERANCE_SECONDS * rate)
    runs, current, misses = [], None, 0
    for index, ok in enumerate(on_line):
        if ok:
            current = [index, index] if current is None else [current[0], index]
            misses = 0
        elif current is not None and misses < ALLOWED_MISSES:
            misses += 1
        else:
            if current is not None:
                runs.append(tuple(current))
            current, misses = None, 0
    if current is not None:
        runs.append(tuple(current))
    accepted = [(first, last, float(slope), float(intercept)) for first, last in runs
                if int(on_line[first:last + 1].sum()) >= MIN_RUN_WINDOWS]
    if trace is not None:
        trace.update({"decision": "line_runs_found" if accepted else "no_contiguous_line_run",
                      "fit_iterations": iterations,
                      "slope_lag_samples_per_query_sample": float(slope),
                      "intercept_lag_samples": float(intercept),
                      "line_tolerance_samples": float(LINE_TOLERANCE_SECONDS * rate),
                      "max_rate_deviation": MAX_RATE_DEVIATION,
                      "window_fit": [{"index": index, "query_start_sample": int(starts[index]),
                                      "lag_samples": int(lags[index]), "explained": float(values[index]),
                                      "explained_threshold_pass": bool(good[index]),
                                      "residual_samples": float(residual[index]),
                                      "line_inlier": bool(on_line[index])}
                                     for index in range(len(rows))],
                      "line_runs": [{"first_index": first, "last_index": last,
                                     "line_inlier_count": int(on_line[first:last + 1].sum()),
                                     "accepted_minimum_run": int(on_line[first:last + 1].sum()) >= MIN_RUN_WINDOWS}
                                    for first, last in runs]})
    return accepted


def _null_level(query: np.ndarray, other: np.ndarray, seed_seconds: float, true_lag: float, rate: int,
                rng: np.random.Generator, diagnostics: list[dict] | None = None) -> float:
    """Best mean explained the same tracker reaches from alignments at least 5 s away from the true one."""
    span = (other.size - query.size) / rate
    best = 0.0
    tried = 0
    candidates = rng.uniform(-query.size / rate, query.size / rate + max(span, 0.0), size=NULL_SEEDS * 4)
    for draw_index, candidate in enumerate(candidates):
        if abs(candidate - true_lag) < NULL_MIN_DISTANCE_SECONDS:
            if diagnostics is not None:
                diagnostics.append({"draw_index": draw_index, "candidate_lag_seconds": float(candidate),
                                    "seed_lag_samples_used": int(round(float(candidate) * rate)),
                                    "status": "excluded_near_true_lag"})
            continue
        rows = track(query, other, seed_seconds, float(candidate), rate)
        values = [value for _, _, value in rows]
        trial = None
        if len(values) >= MIN_RUN_WINDOWS:
            top_count = max(MIN_RUN_WINDOWS, len(values) // 2)
            ordered = np.sort(values)[::-1][:top_count]
            best = max(best, float(np.mean(ordered)))
            if diagnostics is not None:
                indexes = np.argsort(np.asarray(values))[::-1][:top_count]
                trial = {"draw_index": draw_index, "candidate_lag_seconds": float(candidate),
                         "seed_lag_samples_used": int(round(float(candidate) * rate)),
                         "status": "scored", "top_window_indexes": indexes.astype(int).tolist(),
                         "score": float(np.mean(ordered)),
                         "rows": [[int(start), int(start + lag), int(lag), float(value)]
                                  for start, lag, value in rows]}
        elif diagnostics is not None:
            trial = {"draw_index": draw_index, "candidate_lag_seconds": float(candidate),
                     "seed_lag_samples_used": int(round(float(candidate) * rate)),
                     "status": "too_few_tracked_windows", "score": None,
                     "rows": [[int(start), int(start + lag), int(lag), float(value)]
                              for start, lag, value in rows]}
        if trial is not None and diagnostics is not None:
            diagnostics.append(trial)
        tried += 1
        if tried >= NULL_SEEDS:
            break
    return best


def stretched_matches(query: np.ndarray, other: np.ndarray, seeds: list[tuple[float, float]], *,
                      rate: int = SAMPLE_RATE, rng: np.random.Generator | None = None,
                      diagnostics: list[dict] | None = None) -> list[StretchedMatch]:
    """Verified stretched passages from fingerprint seeds ``(query seconds, lag seconds)``.

    Seeds whose tracked windows are already covered by an accepted run are skipped. Each accepted run carries
    its tempo ratio, its range on both clocks, the mean explained share and the chance level it cleared.
    """
    rng = rng or np.random.default_rng(0)
    query = np.asarray(query, np.float64)
    other = np.asarray(other, np.float64)
    width = int(WINDOW_SECONDS * rate)
    found: list[StretchedMatch] = []
    covered: list[tuple[float, float]] = []
    call_trace = ({"schema": "stretch-lag-trace-r1", "method": METHOD, "sample_rate": int(rate),
                   "query_samples": int(query.size), "reference_samples": int(other.size),
                   "authority": {"publication": False, "exact": False, "donor": False,
                                 "reconstruction": False}, "attempts": []}
                  if diagnostics is not None else None)
    for seed_index, (seed_seconds, seed_lag) in enumerate(seeds):
        if any(low <= seed_seconds <= high for low, high in covered):
            if call_trace is not None:
                call_trace["attempts"].append({"seed_index": seed_index,
                    "seed_query_seconds": float(seed_seconds), "seed_lag_seconds": float(seed_lag),
                    "decision": "seed_inside_previously_covered_run", "tracked_rows": [], "runs": []})
            continue
        rows = track(query, other, seed_seconds, seed_lag, rate)
        attempt_trace = None
        if call_trace is not None:
            attempt_trace = {"seed_index": seed_index, "seed_query_seconds": float(seed_seconds),
                             "seed_query_sample_requested": int(round(seed_seconds * rate)),
                             "seed_lag_seconds": float(seed_lag),
                             "seed_lag_samples_used": int(round(seed_lag * rate)),
                             "tracked_rows": [[int(start), int(start + lag), int(lag), float(value)]
                                              for start, lag, value in rows],
                             "runs": []}
        fit_trace = {} if attempt_trace is not None else None
        line_runs = _line_runs(rows, rate, trace=fit_trace)
        if attempt_trace is not None:
            attempt_trace["line_fit"] = fit_trace
            attempt_trace["decision"] = "line_runs_found" if line_runs else fit_trace.get("decision", "no_line_run")
        for run_index, (first, last, slope, intercept) in enumerate(line_runs):
            part = rows[first:last + 1]
            values = [value for _, _, value in part if value >= MIN_WINDOW_EXPLAINED]
            mean = float(np.mean(values))
            q_start = part[0][0] / rate
            q_end = (part[-1][0] + width) / rate
            lag_start = (slope * part[0][0] + intercept) / rate
            lag_end = (slope * (part[-1][0] + width) + intercept) / rate
            run_trace = None
            if attempt_trace is not None:
                fitted_rows = fit_trace.get("window_fit") or []
                fit_by_index = {row["index"]: row for row in fitted_rows}
                run_trace = {"run_index": run_index, "first_tracked_row_index": int(first),
                             "last_tracked_row_index": int(last),
                             "query_sample_bounds": [int(part[0][0]), int(part[-1][0] + width)],
                             "reference_sample_bounds": [float(part[0][0] + lag_start * rate),
                                                          float(part[-1][0] + width + lag_end * rate)],
                             "lag_fit": {"slope_lag_samples_per_query_sample": float(slope),
                                         "intercept_lag_samples": float(intercept),
                                         "rate": float(1.0 + slope),
                                         "lag_at_start_samples": float(lag_start * rate),
                                         "lag_at_end_samples": float(lag_end * rate),
                                         "line_tolerance_samples": float(LINE_TOLERANCE_SECONDS * rate)},
                             "windows": []}
                for index in range(first, last + 1):
                    start, lag, explained = rows[index]
                    fitted = fit_by_index.get(index, {})
                    run_trace["windows"].append({"query_start_sample": int(start),
                        "reference_start_sample": int(start + lag), "lag_samples": int(lag),
                        "explained": float(explained),
                        "explained_threshold_pass": bool(explained >= MIN_WINDOW_EXPLAINED),
                        "line_residual_samples": fitted.get("residual_samples"),
                        "line_inlier": fitted.get("line_inlier", False), "inside_run": True,
                        # Existing aggregate intentionally includes any explained window in this run slice,
                        # even when that row is an off-line miss tolerated by _line_runs.
                        "mean_included": bool(explained >= MIN_WINDOW_EXPLAINED)})
                run_trace["mean_explained"] = mean
                run_trace["mean_included_window_indexes"] = [int(index) for index in range(first, last + 1)
                    if rows[index][2] >= MIN_WINDOW_EXPLAINED]
            if mean < MIN_MEAN_EXPLAINED:
                if run_trace is not None:
                    run_trace.update({"decision": "mean_below_minimum",
                                      "mean_gate": {"passed": False, "observed": mean,
                                                     "minimum": MIN_MEAN_EXPLAINED},
                                      "null_trials": []})
                    attempt_trace["runs"].append(run_trace)
                continue
            coverage_overlaps = [max(0.0, min(high, q_end) - max(low, q_start)) for low, high in covered]
            duplicate_coverage = any(overlap > 0.5 * (q_end - q_start) for overlap in coverage_overlaps)
            if duplicate_coverage:
                if run_trace is not None:
                    run_trace.update({"decision": "overlaps_previously_accepted_run",
                                      "mean_gate": {"passed": True, "observed": mean,
                                                     "minimum": MIN_MEAN_EXPLAINED},
                                      "duplicate_coverage_gate": {"passed": False,
                                          "overlap_seconds": max(coverage_overlaps),
                                          "maximum_fraction": 0.5}, "null_trials": []})
                    attempt_trace["runs"].append(run_trace)
                continue
            centre_lag = (slope * (part[len(part) // 2][0]) + intercept) / rate
            null_trials = [] if run_trace is not None else None
            null = _null_level(query, other, (q_start + q_end) / 2, centre_lag, rate, rng,
                               diagnostics=null_trials)
            if mean < null + MIN_MARGIN:
                if run_trace is not None:
                    run_trace.update({"decision": "null_margin_not_cleared",
                                      "mean_gate": {"passed": True, "observed": mean,
                                                     "minimum": MIN_MEAN_EXPLAINED},
                                      "duplicate_coverage_gate": {"passed": True,
                                          "overlap_seconds": max(coverage_overlaps, default=0.0),
                                          "maximum_fraction": 0.5},
                                      "null_max": float(null), "null_trials": null_trials,
                                      "margin_gate": {"passed": False, "mean": mean, "null_max": float(null),
                                                      "minimum_margin": MIN_MARGIN,
                                                      "required_mean": float(null + MIN_MARGIN)}})
                    attempt_trace["runs"].append(run_trace)
                continue
            if run_trace is not None:
                run_trace.update({"decision": "accepted",
                                  "mean_gate": {"passed": True, "observed": mean,
                                                 "minimum": MIN_MEAN_EXPLAINED},
                                  "duplicate_coverage_gate": {"passed": True,
                                      "overlap_seconds": max(coverage_overlaps, default=0.0),
                                      "maximum_fraction": 0.5},
                                  "null_max": float(null), "null_trials": null_trials,
                                  "margin_gate": {"passed": True, "mean": mean, "null_max": float(null),
                                                  "minimum_margin": MIN_MARGIN,
                                                  "required_mean": float(null + MIN_MARGIN)}})
                attempt_trace["runs"].append(run_trace)
            found.append(StretchedMatch(
                round(q_start, 3), round(q_end, 3), round(q_start + lag_start, 3), round(q_end + lag_end, 3),
                round(1.0 + slope, 5), round(lag_start, 4), len(part), round(mean, 4), round(null, 4)))
            covered.append((q_start, q_end))
        if call_trace is not None:
            call_trace["attempts"].append(attempt_trace)
    if diagnostics is not None:
        diagnostics.append(call_trace)
    return found
