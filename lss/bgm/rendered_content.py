"""Recheck source-heavy scenes against the exact tentative saved PCM.

Source event density nominates difficult work; it cannot describe an already
processed result. Only a sole source-event hold can request this optional check.
The source scope stays HELD until the rendered sound passes. Missing/failed
evidence keeps it HELD; no ordinary published cue is newly subjected to this gate.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import soundfile as sf

from lss.bgm import cue_scope, cue_semantics, cue_publication, music_corroboration
from lss.core import audio_identity

METHOD = "exact-rendered-pcm-source-heavy-recovery-v3-chime-gap-continuity"


def revision():
    return {"method": METHOD, "nomination": "sole_source_event_hold_and_first_pass_event_clear",
            "publication_handoff": "receipt_bound_before_acceptance; any_nonpass_restores_source_hold",
            "rendered_event_thresholds": "unchanged_cue_scope_safety",
            "minimum_music_fraction": cue_semantics.default_recipe().minimum_music_fraction,
            "music_continuity": {"short_interior_gaps": "counted_as_music_like_cue_semantics",
                                 "raw_floor": cue_semantics.default_recipe().continuity_raw_floor},
            "event_windows": {"chime_music_discount": cue_scope.SAFETY_METHOD,
                              "first_pass": "source_mixture_and_first_pass_Music",
                              "rendered": "source_mixture_and_rendered_Music"},
            "music_corroboration": music_corroboration.revision(),
            "serialization": "same_PCM24_headroom_policy_and_exact_saved_PCM_identity",
            "applies_to": "source_heavy_nominees_only; existing_publishable_routes_unchanged"}


def nominate(segments, source, first_probabilities, mixture_probabilities=None):
    """``mixture_probabilities`` (source tags) lets first-pass chime-only Music windows
    be discounted for the nomination only (the decision hold is never discounted); without it every event window counts."""
    first_tags = np.asarray(first_probabilities)
    music = None
    if mixture_probabilities is not None:
        music = (np.asarray(mixture_probabilities)[:, cue_semantics.MUSIC_CLASS],
                 first_tags[:, cue_semantics.MUSIC_CLASS])
    for item in segments:
        if cue_scope.publishable(item) or not cue_scope.publication_candidate(item):
            continue
        if not (item.get("safety") or {}).get("heavy_nonmusic_event_scene"):
            continue
        integrity = (item.get("scope") or {}).get("integrity")
        if integrity is not None and integrity.get("integrity_passed") != 1.0:
            continue
        trial = deepcopy(item)
        trial["safety"]["heavy_nonmusic_event_scene"] = False
        # Scope checks safety before other holds. Reclassify, never infer that a
        # leading safety_hold reason establishes the absence of other blockers.
        quality = {0: {"integrity_passed": integrity["integrity_passed"]}} if integrity else None
        cue_scope.decide([trial], source, quality=quality)
        if not cue_scope.publishable(trial):
            continue
        first = cue_scope.safety(item, first_tags, music=music)
        heavy = first.get("chime_discounted_heavy_nonmusic_event_scene", first["heavy_nonmusic_event_scene"])
        if heavy:
            continue
        item["rendered_content_check"] = {
            "method": METHOD, "state": "required", "source_safety": deepcopy(item["safety"]),
            "first_pass_safety": first,
            "decision_event_window_fraction": item["safety"]["event_window_fraction"],
            "nomination_event_window_fraction": first.get("chime_discounted_event_window_fraction",
                                                          first["event_window_fraction"]),
            "chime_music_windows_not_counted": first.get("chime_music_windows_not_counted", 0)}


def nominated(item):
    check = item.get("rendered_content_check") or {}
    return check.get("method") == METHOD and check.get("state") == "required"


def prepare(audio, rate, start_frame, directory):
    """Serialize privately, then put exact PCM on the original one-second grid."""
    x = np.asarray(audio)
    if (rate != 44100 or x.ndim != 2 or x.shape[1] != 2 or not len(x)
            or not np.isfinite(x).all() or type(start_frame) is not int or start_frame < 0):
        raise ValueError("Rendered content needs finite native analysis-grid stereo")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    probe, signal = directory / "tentative.flac", directory / "aligned.wav"
    scaled, shift, gain = cue_publication.serialization_audio(x)
    sf.write(probe, scaled, rate, subtype="PCM_24")
    exact, _ = sf.read(probe, dtype="float32", always_2d=True)
    prefix = start_frame % rate
    count = (prefix + len(exact) + rate-1)//rate
    aligned = np.zeros((count*rate, 2), dtype=np.float32)
    aligned[prefix:prefix+len(exact)] = exact
    sf.write(signal, aligned, rate, subtype="FLOAT")
    levels = 10*np.log10(np.maximum(np.mean(aligned.reshape(count,rate,2).astype(float)**2, axis=(1,2)), 1e-20))
    return {"probe": str(probe), "signal": str(signal), "checked_audio": audio_identity.audio_identity(probe),
            "signal_sha256": audio_identity.file_digest(signal), "headroom_shift": shift, "gain": gain,
            "source_origin_seconds": start_frame//rate, "prefix_frames": prefix, "frames": len(exact),
            "rate": rate, "count": count, "levels": levels.tolist()}


def evaluate(item, preparation, probabilities, embeddings, source_probabilities):
    """Accept only measured event-clear, Music-corroborated actual cue samples."""
    p, emb, source = np.asarray(probabilities), np.asarray(embeddings), np.asarray(source_probabilities)
    n, origin = preparation["count"], preparation["source_origin_seconds"]
    if (p.shape != (n,527) or emb.ndim != 2 or len(emb) != n or not emb.shape[1]
            or source.ndim != 2 or source.shape[1] != 527 or origin+n > len(source)
            or not np.isfinite(p).all() or not np.isfinite(emb).all()
            or np.any((p<0)|(p>1))):
        raise ValueError("Rendered check has incomplete or invalid event observations")
    a = preparation["prefix_frames"] / preparation["rate"]
    b = a + preparation["frames"] / preparation["rate"]
    local = {"start_seconds": a, "end_seconds": b, "core_seconds": [a,b], "candidates": [[a,b]]}
    semantic_recipe = cue_semantics.default_recipe()
    window = cue_semantics.WindowEvidence.from_arrays(preparation["levels"], p, emb)
    extra, music_receipt = music_corroboration.observe([local], window, p, source[origin:origin+n], semantic_recipe)
    music = cue_semantics.music_windows(window, semantic_recipe) | extra
    indexes = np.arange(n)
    weights = np.maximum(0., np.minimum(indexes+1,b)-np.maximum(indexes,a))
    event, _ = cue_scope.event_windows(p, source[origin:origin+n, cue_semantics.MUSIC_CLASS],
                                       p[:, cue_semantics.MUSIC_CLASS])
    fraction = float(np.sum(weights*event)/weights.sum())
    music_fraction, raw_music_fraction = cue_semantics.continuity_fraction(music, weights, semantic_recipe)
    passed = fraction < cue_scope.EVENT_SCENE_FRACTION and music_fraction >= semantic_recipe.minimum_music_fraction
    result = {**item["rendered_content_check"], "state": "passed" if passed else "held",
              "event_window_fraction": fraction, "music_fraction": music_fraction,
              "raw_music_fraction": raw_music_fraction,
              "checked_audio": preparation["checked_audio"],
              "headroom_shift": preparation["headroom_shift"], "gain": preparation["gain"],
              "source_origin_seconds": origin, "local_body_seconds": [a,b],
              "prefix_frames_excluded_from_denominator": preparation["prefix_frames"],
              "music_corroboration": music_receipt,
              "scope": "event/Music observations, not a clean-master or wanted-component guarantee"}
    item["rendered_content_check"] = result
    if passed:
        item["safety"] = {**item["safety"], "source_observation": deepcopy(item["safety"]),
                          "method": METHOD, "event_window_fraction": round(fraction,4),
                          "heavy_nonmusic_event_scene": False, "checked_audio": preparation["checked_audio"]}
    return result
