"""Bounded positive Music evidence from agreeing source and named subtypes.

AudioSet's independently predicted parent can be weaker than its musical child.
Only add evidence inside an existing materialized semantic body. This module
cannot extend an edge, merge cues, authorize raw audio or assert an identity.

Two witnesses exist. The strong one needs a child >= .5 plus a source-Music
seed and may bridge a single weak second. The quiet one (v2) is for music-box /
mallet cues whose parent Music hovers at .35-.45 while a musical child stays
.25-.6: a second counts only when it belongs to a long run (at least
``minimum_quiet_run_seconds``) in which one musical child is dominant at every
second and neighbouring embeddings stay similar. An environmental bell,
doorbell or SFX chime has no separated-stem Music, so it never qualifies.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Policy:
    revision: str = "source-corroborated-musical-subtypes-v2"
    child_probability: float = .5
    source_music_probability: float = .5
    weak_probability: float = .25
    minimum_seed_seconds: int = 3
    minimum_body_seconds: int = 8
    maximum_internal_hole_seconds: int = 1
    minimum_embedding_similarity: float = .92
    quiet_subtype_probability: float = .25
    minimum_quiet_run_seconds: int = 20


POLICY = Policy()
# Pitched/instrumental subclasses and musical styles/roles. The broad instrument
# parent and environmental bells/chimes are deliberately not positive witnesses.
# Exact ordering: upstream audioset_tagging_cnn metadata, pinned in Evidence.
CHILD_CLASSES = tuple(range(139, 200)) + tuple(range(216, 283))
MUSIC_CLASS = 137


def revision():
    return {**asdict(POLICY), "child_classes": list(CHILD_CLASSES)}


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _quiet_runs(mask, dominant, embeddings):
    """Maximal runs of qualifying seconds with one dominant child and stable embeddings."""
    start = None
    for t in range(len(mask)+1):
        keep = t < len(mask) and bool(mask[t])
        if keep and start is not None:
            lhs, rhs = embeddings[t-1], embeddings[t]
            norm = float(np.linalg.norm(lhs)*np.linalg.norm(rhs))
            similarity = float(lhs@rhs/norm) if norm else 0.
            if dominant[t] != dominant[t-1] or similarity < POLICY.minimum_embedding_similarity:
                yield start, t
                start = t
        elif keep:
            start = t
        elif start is not None:
            yield start, t
            start = None


def observe(segments, evidence, probabilities, mixture, recipe):
    """Return one positive boolean per source second and an inspectable receipt.

    A one-second internal dropout needs the SAME musical child on both flanks,
    weak evidence in the actual hole, and stable neighboring embeddings. The
    seeds are frozen before bridging, so a bridge cannot bootstrap another.
    """
    p, source = (np.asarray(a, dtype=float) for a in (probabilities, mixture))
    n = evidence.count
    if (p.shape != (n, 527) or source.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(source).all()
            or np.any((p < 0) | (p > 1)) or np.any((source < 0) | (source > 1))):
        raise ValueError("Musical corroboration needs aligned finite AudioSet observations")
    children = p[:, CHILD_CLASSES]
    audible = np.asarray(evidence.music_db) >= recipe.floor_music_db
    weak = ((p[:, MUSIC_CLASS] >= POLICY.weak_probability)
            & (source[:, MUSIC_CLASS] >= POLICY.weak_probability) & audible)
    seeds = ((children.max(axis=1) >= POLICY.child_probability)
             & (p[:, MUSIC_CLASS] >= POLICY.weak_probability)
             & (source[:, MUSIC_CLASS] >= POLICY.source_music_probability) & audible)
    quiet = ((children.max(axis=1) >= POLICY.quiet_subtype_probability)
             & (p[:, MUSIC_CLASS] >= POLICY.quiet_subtype_probability)
             & (source[:, MUSIC_CLASS] >= POLICY.quiet_subtype_probability) & audible)
    dominant = children.argmax(axis=1)
    positive = np.zeros(n, dtype=bool)
    records = []
    for part in segments:
        a, b = map(float, part.get("core_seconds", [part["start_seconds"], part["end_seconds"]]))
        if not np.isfinite([a, b]).all() or not 0 <= a < b <= n:
            raise ValueError("Musical corroboration body leaves the observed source grid")
        if b-a < POLICY.minimum_body_seconds:
            continue
        lo, hi = int(np.floor(a)), int(np.ceil(b))
        # Do not turn rounded semantic coverage into a materialized audio claim.
        cursor = a
        for first, last in sorted(part.get("candidates", [])):
            if last <= cursor:
                continue
            if first > cursor + 1e-6:
                break
            cursor = max(cursor, last)
            if cursor >= b:
                break
        if cursor < b - 1e-6:
            continue
        local = seeds[lo:hi].copy()
        bridges = []
        for first, last in _runs(~local):
            if (first == 0 or last == len(local)
                    or last-first > POLICY.maximum_internal_hole_seconds):
                continue
            t, stop = lo+first, lo+last
            common = ((children[t-1] >= POLICY.child_probability)
                      & (children[stop] >= POLICY.child_probability))
            if not common.any() or not np.all(children[t:stop, common].max(axis=1) >= POLICY.weak_probability):
                continue
            lhs, rhs = evidence.embeddings[t-1], evidence.embeddings[stop]
            norm = float(np.linalg.norm(lhs)*np.linalg.norm(rhs))
            similarity = float(lhs@rhs/norm) if norm else 0.
            if np.all(weak[t:stop]) and similarity >= POLICY.minimum_embedding_similarity:
                local[first:last] = True
                bridges.extend(range(t, stop))
        accepted = np.zeros(hi-lo, dtype=bool)
        for first, last in _runs(local):
            if int(seeds[lo+first:lo+last].sum()) >= POLICY.minimum_seed_seconds:
                accepted[first:last] = True
        sustained = np.zeros(hi-lo, dtype=bool)
        for first, last in _quiet_runs(quiet[lo:hi], dominant[lo:hi], evidence.embeddings[lo:hi]):
            if last-first >= POLICY.minimum_quiet_run_seconds:
                sustained[first:last] = True
        added = (accepted | sustained) & (p[lo:hi, MUSIC_CLASS] < recipe.music_probability)
        if added.any():
            positive[lo:hi] |= added
            records.append({"body": [a, b], "positive_music_seconds": (np.flatnonzero(added)+lo).tolist(),
                            "seed_seconds": (np.flatnonzero(seeds[lo:hi] & accepted)+lo).tolist(),
                            "bridged_seconds": [t for t in bridges if accepted[t-lo]],
                            "quiet_subtype_seconds": (np.flatnonzero(added & sustained & ~accepted)+lo).tolist()})
    return positive, {"policy": revision(), "records": records,
                      "positive_music_seconds": np.flatnonzero(positive).tolist(),
                      "boundary_authority": "none", "original_mixture_authority": "none",
                      "identity_authority": "none", "absence_authority": "none"}
