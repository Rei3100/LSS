"""Decide `<PART>_VER` version tags (`Only` / `Cut`) from per-part activity envelopes.

Pure logic, independent of any separation model. Contract: docs/engine/LISTENING_AND_PRESENTATION.md
("Versions are open-ended"). Presence is the default and is never written; anything not clearly decided
writes nothing (UNKNOWN is not ABSENT).

Input envelopes are frame RMS in dB relative to the cue's own mix RMS (0 dB = the whole mix), fixed hop,
one value per frame; silence may be -inf or any very low number.

- ACTIVE: level >= ACTIVE_DB (-20 dB) in at least ACTIVE_SHARE (50 %) of the frames. A part within 20 dB
  of the mix for half of the span is playing.
- SILENT: level < SILENT_DB (-35 dB) in at least SILENT_SHARE (90 %) of the frames. Separators leak (a
  stem rarely drops below -45 dB even when the part is absent), so the floor sits at the leakage level,
  not at digital silence; the 90 % share tolerates short bleed bursts.
- The band between -35 and -20 dB (and shares between the two rules) decides nothing.
- Only(P): P active and every other supplied part silent over the whole cue. Needs at least two parts
  supplied (otherwise "others silent" cannot be observed).
- Cut(P): over the time-aligned span P is silent in this cue and active in the compared version, and at
  least one other part is active in this cue there (guards against aligning onto silence). Needs an
  alignment span of at least MIN_OVERLAP_SECONDS and MIN_OVERLAP_SHARE of the cue.

Measured on 23 real TV cues and 7 official OST tracks (development Evidence, not in this repository): the
thresholds above are kept. On cues whose official track has no vocals the VOCALS stem sits at a median of
about -45 dB (p90 -30, p95 -15 dB: TV dialogue/SFX bleed), so SILENT_DB -35 with a 90 % share is already
conservative; looser ACTIVE (-25 dB / 40 %) writes a false Cut(VOCALS) on a cue whose vocals are audibly
present in the release (the separator assigns the same sound to another stem in the TV mix), so ACTIVE stays
-20 dB / 50 %. Stem envelopes of two different mixes are not directly comparable; a Cut needs more than this.

Positive controls (development Evidence): re-mixed stems of 7 OST tracks, re-separated;
DRUMS Cut 3/3 and Only 3/3, 0 false tags on 10 negatives (re-mix, -6 dB). BASS/VOCALS recall unmeasured (never ACTIVE in
the OST tracks). OTHER must stay a supplied part: without it an untouched re-mix yields a false Only and Cut is missed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

ACTIVE_DB = -20.0
ACTIVE_SHARE = 0.50
SILENT_DB = -35.0
SILENT_SHARE = 0.90
MIN_OVERLAP_SECONDS = 5.0
MIN_OVERLAP_SHARE = 0.5
ONLY = "Only"
CUT = "Cut"
SUFFIX = "_VER"
_KEY = re.compile(r"^[A-Z0-9][A-Z0-9_ ]{0,63}$")


@dataclass(frozen=True)
class Alignment:
    """Cue frame i is the same music as compared-version frame i + offset_frames, for i in [start, end)."""
    offset_frames: int
    start: int
    end: int


def tag_name(part: str) -> str | None:
    """`chorus` -> `CHORUS_VER`; None when the part has no valid ASCII name (stays internal)."""
    base = re.sub(r"[^A-Za-z0-9]+", "_", str(part)).strip("_").upper()
    name = base + SUFFIX
    return name if base and _KEY.match(name) else None


def _stats(env: np.ndarray) -> dict:
    n = int(env.size)
    if n == 0:
        return {"frames": 0, "active_share": 0.0, "silent_share": 0.0, "median_db": None}
    finite = np.nan_to_num(env.astype(np.float64), nan=-200.0, neginf=-200.0)
    return {"frames": n, "active_share": float(np.mean(finite >= ACTIVE_DB)),
            "silent_share": float(np.mean(finite < SILENT_DB)), "median_db": float(np.median(finite))}


def _active(s: dict) -> bool:
    return s["frames"] > 0 and s["active_share"] >= ACTIVE_SHARE


def _silent(s: dict) -> bool:
    return s["frames"] > 0 and s["silent_share"] >= SILENT_SHARE


def _span(a: Alignment, n: int, other_size: int) -> tuple[int, int] | None:
    lo = max(a.start, 0, -a.offset_frames)
    hi = min(a.end, n, other_size - a.offset_frames)
    return (lo, hi) if hi > lo else None


def decide(cue: dict[str, np.ndarray], other: dict[str, np.ndarray] | None = None,
           alignment: Alignment | None = None, *, hop_seconds: float = 0.25) -> tuple[dict, dict]:
    """Return ({tag: value}, evidence). Evidence always carries thresholds and per-part statistics."""
    cue = {p: np.asarray(v, dtype=np.float64).ravel() for p, v in cue.items()}
    evidence: dict = {"thresholds": {"active_db": ACTIVE_DB, "active_share": ACTIVE_SHARE,
                                     "silent_db": SILENT_DB, "silent_share": SILENT_SHARE,
                                     "min_overlap_seconds": MIN_OVERLAP_SECONDS,
                                     "min_overlap_share": MIN_OVERLAP_SHARE},
                      "hop_seconds": hop_seconds, "parts": {}, "only": None, "cut": {}}
    tags: dict[str, str] = {}
    lengths = {v.size for v in cue.values()}
    if not cue or len(lengths) != 1 or 0 in lengths:
        evidence["note"] = "no usable cue envelopes"
        return tags, evidence
    n = next(iter(lengths))
    stats = {p: _stats(v) for p, v in cue.items()}
    evidence["parts"] = {p: dict(s) for p, s in stats.items()}

    if len(cue) >= 2:
        actives = [p for p, s in stats.items() if _active(s)]
        if len(actives) == 1 and all(_silent(s) for p, s in stats.items() if p != actives[0]):
            name = tag_name(actives[0])
            evidence["only"] = {"part": actives[0], "tag": name}
            if name:
                tags[name] = ONLY

    if other and alignment is not None:
        other = {p: np.asarray(v, dtype=np.float64).ravel() for p, v in other.items()}
        common = [p for p in cue if p in other]
        span = _span(alignment, n, min((other[p].size for p in common), default=0))
        evidence["alignment"] = {"offset_frames": alignment.offset_frames, "start": alignment.start,
                                 "end": alignment.end, "used": list(span) if span else None}
        if span and (span[1] - span[0]) * hop_seconds >= MIN_OVERLAP_SECONDS \
                and (span[1] - span[0]) >= MIN_OVERLAP_SHARE * n:
            lo, hi = span
            off = alignment.offset_frames
            here = {p: _stats(cue[p][lo:hi]) for p in cue}
            for part in common:
                theirs = _stats(other[part][lo + off:hi + off])
                mine = here[part]
                others_active = any(_active(s) for p, s in here.items() if p != part)
                cut = bool(_silent(mine) and _active(theirs) and others_active)
                evidence["cut"][part] = {"here": mine, "compared": theirs,
                                         "others_active_here": others_active, "cut": cut}
                name = tag_name(part)
                if cut and name and name not in tags:
                    tags[name] = CUT
    return tags, evidence
