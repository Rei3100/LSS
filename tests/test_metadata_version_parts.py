import re

import numpy as np

from lss.metadata import version_parts as vp

HOP = 0.25
N = 200  # 50 s


def env(db, n=N):
    return np.full(n, float(db))


def test_only_detected():
    tags, ev = vp.decide({"chorus": env(-6), "vocals": env(-50), "bass": env(-60), "drums": env(-70)})
    assert tags == {"CHORUS_VER": "Only"}
    assert ev["only"]["part"] == "chorus"


def test_only_needs_other_parts_silent_and_observed():
    assert vp.decide({"bass": env(-6)})[0] == {}
    assert vp.decide({"bass": env(-6), "drums": env(-30)})[0] == {}  # gap band: nothing
    assert vp.decide({"bass": env(-6), "drums": env(-10)})[0] == {}  # two active


def test_ambiguous_levels_write_nothing():
    assert vp.decide({"bass": env(-27), "drums": env(-60)})[0] == {}
    mixed = np.where(np.arange(N) % 10 < 3, -6.0, -60.0)
    assert vp.decide({"bass": mixed, "drums": env(-60)})[0] == {}


def test_cut_vs_version_with_part_active():
    cue = {"drums": env(-60), "other": env(-5), "vocals": env(-9)}
    other = {"drums": env(-8), "other": env(-5), "vocals": env(-9)}
    tags, ev = vp.decide(cue, other, vp.Alignment(0, 0, N), hop_seconds=HOP)
    assert tags == {"DRUMS_VER": "Cut"}
    assert ev["cut"]["drums"]["cut"] and not ev["cut"]["other"]["cut"]


def test_no_compared_version_no_cut():
    cue = {"drums": env(-60), "other": env(-5), "vocals": env(-9)}
    assert vp.decide(cue)[0] == {}
    assert vp.decide(cue, None, vp.Alignment(0, 0, N))[0] == {}


def test_compared_part_not_clearly_active_no_cut():
    cue = {"drums": env(-60), "other": env(-5), "vocals": env(-9)}
    other = {"drums": env(-28), "other": env(-5), "vocals": env(-9)}
    assert vp.decide(cue, other, vp.Alignment(0, 0, N), hop_seconds=HOP)[0] == {}


def test_short_or_misaligned_overlap_nothing():
    cue = {"drums": env(-60), "other": env(-5), "vocals": env(-9)}
    other = {"drums": env(-8), "other": env(-5), "vocals": env(-9)}
    assert vp.decide(cue, other, vp.Alignment(0, 0, 10), hop_seconds=HOP)[0] == {}      # 2.5 s
    assert vp.decide(cue, other, vp.Alignment(0, 0, 60), hop_seconds=HOP)[0] == {}      # <50 % of cue
    assert vp.decide(cue, other, vp.Alignment(500, 0, N), hop_seconds=HOP)[0] == {}     # off the end
    assert vp.decide(cue, other, vp.Alignment(0, 150, 100), hop_seconds=HOP)[0] == {}   # empty


def test_offset_alignment_uses_matching_frames():
    cue = {"drums": env(-60, 100), "other": env(-5, 100), "vocals": env(-9, 100)}
    d = np.concatenate([env(-60, 70), env(-8, 80)])  # drums only active after frame 70
    other = {"drums": d, "other": env(-5, 150), "vocals": env(-9, 150)}
    assert vp.decide(cue, other, vp.Alignment(70, 0, 80), hop_seconds=HOP)[0] == {"DRUMS_VER": "Cut"}
    assert vp.decide(cue, other, vp.Alignment(0, 0, 80), hop_seconds=HOP)[0] == {}


def test_multiple_parts_cut_one_tag_each():
    cue = {"drums": env(-60), "bass": env(-55), "other": env(-5), "vocals": env(-9)}
    other = {"drums": env(-8), "bass": env(-12), "other": env(-5), "vocals": env(-9)}
    tags, _ = vp.decide(cue, other, vp.Alignment(0, 0, N), hop_seconds=HOP)
    assert tags == {"DRUMS_VER": "Cut", "BASS_VER": "Cut"}


def test_cut_not_claimed_when_cue_is_silent_everywhere():
    cue = {"drums": env(-80), "other": env(-80)}
    other = {"drums": env(-8), "other": env(-5), "vocals": env(-9)}
    assert vp.decide(cue, other, vp.Alignment(0, 0, N), hop_seconds=HOP)[0] == {}


def test_tag_names_valid_for_finalizer():
    key = re.compile(r"^[A-Z0-9][A-Z0-9_ ]{0,63}$")
    for part in ("chorus", "lead vocals", "Lead-Vocals", "bass", "other", "male voice"):
        name = vp.tag_name(part)
        assert name.endswith("_VER") and key.match(name)
    assert vp.tag_name("コーラス") is None
    assert vp.tag_name("") is None
