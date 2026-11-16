"""Independent evidence contracts for within-body musical subtype support."""
from dataclasses import replace

import numpy as np
import pytest

from lss.bgm import cue_semantics as cs, music_corroboration as mc


def case(count=16):
    p = np.zeros((count, 527)); source = p.copy()
    p[:, 137] = .35
    p[:, 181] = .7  # Glockenspiel, pinned upstream index (not environmental bell).
    source[:, 137] = .8
    e = cs.WindowEvidence(np.full(count, -25.), p[:,137], np.zeros(count), np.tile([1., 0.], (count, 1)))
    part = dict(start_seconds=0., end_seconds=float(count), core_seconds=[0., float(count)],
                candidates=[[0., float(count)]])
    return [part], e, p, source


def observe(values):
    return mc.observe(*values, cs.default_recipe())


def test_continuing_glockenspiel_adds_positive_observation_without_boundary_authority():
    values = case()
    mask, record = observe(values)
    assert mask.all()
    assert record['boundary_authority'] == record['original_mixture_authority'] == 'none'
    assert values[0][0]['core_seconds'] == [0., 16.]


@pytest.mark.parametrize('column', [200, 201, 203, 205, 206, 356, 481, 482, 483, 495, 501, 502, 504])
def test_bells_doorbells_tonal_effects_and_ambiguous_harmonics_are_not_musical_witnesses(column):
    parts,e,p,source = case()
    p[:,181] = 0.; p[:,column] = 1.
    assert not observe((parts,e,p,source))[0].any()


def test_source_music_absence_short_seed_and_unmaterialized_hole_cannot_authorize():
    parts,e,p,source = case()
    source[:,137] = .2
    assert not observe((parts,e,p,source))[0].any()
    source[:,137] = .8; p[:,181] = 0.; p[5:7,181] = .8
    assert not observe((parts,e,p,source))[0].any()
    p[:,181] = .8; parts[0]['candidates'] = [[0.,6.], [6.1,16.]]
    assert not observe((parts,e,p,source))[0].any()


def test_single_weak_second_requires_same_child_on_flanks_and_stable_embeddings():
    parts,e,p,source = case()
    p[7,181] = .3; source[7,137] = .3
    mask, report = observe((parts,e,p,source))
    assert mask[7] and report['records'][0]['bridged_seconds'] == [7]
    e = replace(e, embeddings=e.embeddings.copy()); e.embeddings[8] = [0.,1.]
    assert not observe((parts,e,p,source))[0][7]
    e.embeddings[8] = [1.,0.]; p[8:,181] = 0.; p[8:,153] = .8
    assert not observe((parts,e,p,source))[0][7]


def test_two_second_hole_and_digital_silence_stay_unconfirmed():
    parts,e,p,source = case()
    p[7:9,181] = .3; source[7:9,137] = .3
    assert not observe((parts,e,p,source))[0][7:9].any()
    p[8,181] = .8; source[8,137] = .8
    e = replace(e, music_db=e.music_db.copy()); e.music_db[7] = -200.
    assert not observe((parts,e,p,source))[0][7]


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.1, 1.1])
def test_invalid_observations_are_rejected(bad):
    values = case(); values[2][0,137] = bad
    with pytest.raises(ValueError):
        observe(values)


def quiet_case(count=30, music=.4, child=.35):
    """Music-box / glockenspiel cue: parent Music hovers below the strict .5 mask."""
    p = np.zeros((count, 527)); source = p.copy()
    p[:, 137] = music; p[:, 181] = child; source[:, 137] = music
    e = cs.WindowEvidence(np.full(count, -25.), p[:, 137], np.zeros(count), np.tile([1., 0.], (count, 1)))
    part = dict(start_seconds=0., end_seconds=float(count), core_seconds=[0., float(count)],
                candidates=[[0., float(count)]])
    return [part], e, p, source


def test_sustained_quiet_glockenspiel_run_becomes_music_without_authority_beyond_the_body():
    parts, e, p, source = quiet_case()
    mask, record = observe((parts, e, p, source))
    assert mask.all() and record['records'][0]['quiet_subtype_seconds'] == list(range(30))
    assert record['policy']['revision'].endswith('-v2')
    assert record['boundary_authority'] == record['identity_authority'] == 'none'
    assert parts[0]['core_seconds'] == [0., 30.]
    parts[0].update(start_seconds=5., end_seconds=25., core_seconds=[5., 25.], candidates=[[5., 25.]])
    assert np.flatnonzero(observe((parts, e, p, source))[0]).tolist() == list(range(5, 25))


def test_doorbell_chime_and_dialogue_segment_stays_non_music():
    parts, e, p, source = quiet_case()
    p[:, 181] = 0.; p[:, 137] = .1; source[:, 137] = .2
    p[:, [481, 482, 483, 495, 200, 201]] = .8  # Doorbell/Ding-dong/Bell-like effects + speech-ish mixture
    p[:, 0] = .9; source[:, 0] = .9
    assert not observe((parts, e, p, source))[0].any()
    p[:, 137] = .4; source[:, 137] = .4  # even with a quiet-Music reading, bells are not child classes
    assert not observe((parts, e, p, source))[0].any()


def test_pure_sfx_chime_sequence_with_no_stem_music_is_not_lifted():
    parts, e, p, source = quiet_case()
    p[:, 137] = .05; source[:, 137] = .5   # a mixture-only reading; the separated stem has no Music
    assert not observe((parts, e, p, source))[0].any()
    p[:, 137] = .4; source[:, 137] = .1    # stem-only reading: the mixture does not corroborate
    assert not observe((parts, e, p, source))[0].any()
    e2 = replace(e, music_db=np.full(30, -90.))  # below the audibility floor
    p[:, 137] = .4; source[:, 137] = .4
    assert not observe((parts, e2, p, source))[0].any()


def test_single_second_subtype_spike_and_short_runs_are_not_sustained():
    parts, e, p, source = quiet_case()
    p[:, 181] = 0.; p[12, 181] = .9
    assert not observe((parts, e, p, source))[0].any()
    p[:, 181] = .35; p[10, 181] = 0.   # splits the cue into runs of 10 and 19 seconds
    assert not observe((parts, e, p, source))[0].any()


def test_changing_dominant_child_or_unstable_embeddings_break_the_quiet_run():
    parts, e, p, source = quiet_case(38)
    p[19:, 181] = 0.; p[19:, 153] = .35
    assert not observe((parts, e, p, source))[0].any()
    parts, e, p, source = quiet_case(38)
    e = replace(e, embeddings=e.embeddings.copy()); e.embeddings[19:] = [0., 1.]
    assert not observe((parts, e, p, source))[0].any()
