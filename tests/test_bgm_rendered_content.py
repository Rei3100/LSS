"""Guard rails for rechecking only the exact PCM of source-heavy held cues."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss.bgm import cue_scope, cue_semantics, cue_publication, rendered_content

SOURCE = {"path": "episode.mkv", "sha256": "a" * 64}
RATE = 44100


def segment(**extra):
    quality = extra.pop("quality", None)
    item = {"start_seconds": 20.0, "end_seconds": 40.0, "core_seconds": [20.0, 40.0],
            "start_edge": {"kind": "music_gap"}, "end_edge": {"kind": "music_gap"},
            "semantic_class": cue_semantics.WANTED, "reasons": [], "features": {}, **extra}
    mixture = np.zeros((80, 527), np.float32)
    mixture[:, 137] = .9
    mixture[:, 300] = .3
    cue_scope.decide([item], SOURCE, mixture=mixture, quality={0: quality} if quality else None)
    if quality and quality.get("integrity_passed") == 0.0:
        cue_scope.final(item, 0, SOURCE, {"integrity_passed": 0.0})
    if extra_context := item.pop("_test_context", None):
        item["context"] = extra_context
    return item


def clear_probabilities(count, music=True):
    p = np.zeros((count, 527), np.float32)
    if music:
        p[:, 137] = .9
    emb = np.ones((count, 8), np.float32)
    return p, emb


def prepared(tmp_path, start_frame=20 * RATE, frames=10 * RATE, peak=.2):
    t = np.arange(frames, dtype=np.float32) / RATE
    audio = np.column_stack((peak * np.sin(2*np.pi*440*t), peak * np.sin(2*np.pi*660*t)))
    return audio, rendered_content.prepare(audio, RATE, start_frame, tmp_path)


@pytest.mark.parametrize("extra", [
    {"_test_context": {"insert_section": True}},
    {"semantic_class": cue_semantics.UNCERTAIN, "reasons": ["partly_registered_reference"]},
    {"quality": {"integrity_passed": 0.0}},
    {"end_edge": {"kind": "candidate_edge"}},
])
def test_nomination_keeps_other_scope_holds(tmp_path, extra):
    item = segment(**extra)
    assert item["scope"]["disposition"] == "HELD"
    mixture = np.zeros((80, 527), np.float32)
    mixture[:, 137] = .9
    mixture[:, 300] = .3
    rendered_content.nominate([item], SOURCE, mixture)
    assert not rendered_content.nominated(item)
    assert item["scope"]["disposition"] == "HELD"


def nominated_item(tmp_path):
    item = segment()
    # Source-heavy hold is the sole blocker; source Music and event evidence nominate it.
    mixture = np.zeros((80, 527), np.float32)
    mixture[:, 137] = .9
    rendered_content.nominate([item], SOURCE, mixture)
    assert rendered_content.nominated(item)
    return item


def test_sole_source_heavy_hold_is_positive_nomination_control():
    item = segment()
    first = np.zeros((80, 527), np.float32)
    first[:, 137] = .9
    rendered_content.nominate([item], SOURCE, first)
    assert item["scope"]["disposition"] == "HELD"
    assert item["scope"]["reason"] == "safety_hold"
    assert rendered_content.nominated(item)


def test_rendered_event_scene_vetoes_and_keeps_source_held(tmp_path):
    item = nominated_item(tmp_path)
    _, prep = prepared(tmp_path)
    p, emb = clear_probabilities(prep["count"])
    p[:, 300] = .9
    source = np.zeros((80, 527), np.float32)
    source[:, 137] = .9
    result = rendered_content.evaluate(item, prep, p, emb, source)
    assert result["state"] == "held"
    assert result["event_window_fraction"] >= cue_scope.EVENT_SCENE_FRACTION
    assert item["safety"]["heavy_nonmusic_event_scene"] is True


def test_music_absent_in_saved_pcm_vetoes_even_when_events_are_clear(tmp_path):
    item = nominated_item(tmp_path)
    _, prep = prepared(tmp_path)
    p, emb = clear_probabilities(prep["count"], music=False)
    source = np.zeros((80, 527), np.float32)
    source[:, 137] = .9
    result = rendered_content.evaluate(item, prep, p, emb, source)
    assert result["state"] == "held"
    assert result["music_fraction"] < cue_semantics.default_recipe().minimum_music_fraction
    assert item["safety"]["heavy_nonmusic_event_scene"] is True


@pytest.mark.parametrize("p,emb,source", [
    (np.zeros((1, 527)), np.zeros((1, 3)), np.zeros((1, 527))),
    (np.full((10, 527), np.nan), np.zeros((10, 3)), np.zeros((80, 527))),
    (np.full((10, 527), 1.1), np.zeros((10, 3)), np.zeros((80, 527))),
    (np.zeros((10, 527)), np.zeros((10, 3)), np.zeros((25, 527))),
    (np.zeros((10, 527)), np.zeros((9, 3)), np.zeros((80, 527))),
])
def test_invalid_or_incomplete_observations_fail_closed(tmp_path, p, emb, source):
    item = nominated_item(tmp_path)
    _, prep = prepared(tmp_path)
    with pytest.raises(ValueError):
        rendered_content.evaluate(item, prep, p, emb, source)
    assert rendered_content.nominated(item)


def test_fractional_grid_prefix_is_excluded_from_weighted_decision(tmp_path):
    # An audio beginning 10.499s into the source occupies .499s of its first grid window.
    _, prep = prepared(tmp_path, start_frame=10*RATE + round(.499*RATE), frames=9*RATE)
    assert prep["prefix_frames"] == round(.499*RATE)
    p, emb = clear_probabilities(prep["count"])
    p[0, 300] = .99  # first-window evidence is weighted only over its .501s target overlap
    source = np.zeros((20, 527), np.float32)
    source[:, 137] = .9
    item = nominated_item(tmp_path)
    result = rendered_content.evaluate(item, prep, p, emb, source)
    assert result["event_window_fraction"] == pytest.approx(.501 / 9.0, abs=1e-6)
    assert result["music_fraction"] == pytest.approx(1.0)
    assert result["prefix_frames_excluded_from_denominator"] == round(.499*RATE)


def test_pcm24_headroom_serialization_matches_publication_identity(tmp_path):
    audio, prep = prepared(tmp_path, frames=RATE, peak=1.4)
    assert prep["headroom_shift"] > 0
    published = cue_publication.publish(audio, RATE, tmp_path / "published.flac", key="k", title="t", album="a",
                                        provenance={}, input_sha256="a"*64, recipe_id="r", version="v")
    assert prep["checked_audio"] == published["audio"]
    exact, _ = sf.read(prep["probe"], dtype="float32", always_2d=True)
    assert np.max(np.abs(exact)) < 1.0
    assert published["headroom_shift"] == prep["headroom_shift"]


def test_sung_music_remains_music_evidence(tmp_path):
    item = nominated_item(tmp_path)
    _, prep = prepared(tmp_path)
    p, emb = clear_probabilities(prep["count"], music=False)
    p[:, 254] = .9  # Vocal music class; the wanted-music contract includes musical vocals.
    p[:, 137] = .3
    source = np.zeros((80, 527), np.float32)
    source[:, 137] = .9
    result = rendered_content.evaluate(item, prep, p, emb, source)
    assert result["state"] == "passed"
    assert result["music_fraction"] == pytest.approx(1.0)


def test_rendered_check_uses_gap_aware_continuity_and_the_chime_discount(tmp_path):
    # Same rules as the source decision: short interior Music dips are continuous music, and
    # chime-only windows where both the source and the rendered PCM are Music are not events.
    item = nominated_item(tmp_path)
    _, prep = prepared(tmp_path)
    p, emb = clear_probabilities(prep["count"])
    p[3, 137], p[6, 137] = .45, .35  # raw 8/10 windows of Music
    p[4:7, 495] = .5                # "Jingle, tinkle" only: 3/10 would hold
    source = np.zeros((80, 527), np.float32)
    source[:, 137] = .9
    result = rendered_content.evaluate(item, prep, p, emb, source)
    assert result["raw_music_fraction"] == pytest.approx(.8)
    assert result["music_fraction"] == pytest.approx(1.0)
    assert result["event_window_fraction"] == pytest.approx(.1)   # window 6 is not stem Music >= .4
    assert result["state"] == "passed"
    source[24:27, 137] = .3         # source not Music there: the chimes count again
    item = nominated_item(tmp_path)
    assert rendered_content.evaluate(item, prep, p, emb, source)["state"] == "held"


def test_rendered_check_publishes_a_chime_heavy_held_cue_only_when_the_rendered_pcm_passes(tmp_path):
    # Source events: 4 of 10 windows (0.4) held at decision level; rendered Music is high and only
    # chime events remain, so the discounted rendered check passes. Non-chime events still hold.
    item = nominated_item(tmp_path)
    _, prep = prepared(tmp_path)
    p, emb = clear_probabilities(prep["count"])
    p[2:6, 495] = .5                 # "Jingle, tinkle" in 4/10 windows, rendered Music .9
    source = np.zeros((80, 527), np.float32)
    source[:, 137] = .9
    result = rendered_content.evaluate(item, prep, p, emb, source)
    assert result["state"] == "passed"
    assert result["event_window_fraction"] == pytest.approx(0.0)
    p[2:6, 300] = .5                 # a non-chime event class alongside: the windows count again
    item = nominated_item(tmp_path)
    held = rendered_content.evaluate(item, prep, p, emb, source)
    assert held["state"] == "held" and held["event_window_fraction"] == pytest.approx(.4)
