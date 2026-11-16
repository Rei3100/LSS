"""Incoming near-threshold heads, centered nuisance and layered-score falsifiers."""
from copy import deepcopy
import numpy as np
from lss.bgm import cue_formation as cf, cue_semantics as cs, recording_support as rs
from tests.test_recording_support import parts, component


def head_fixture():
    e = cs.WindowEvidence(np.full(40,-30.),np.r_[np.ones(15),np.full(7,.46),np.ones(18)],
                         np.zeros(40),np.r_[np.tile([1.,0.],(14,1)),np.tile([0.,1.],(26,1))])
    before = [parts(5.,15.),parts(22.,35.)]
    usage = {"onsets":[{"predicted_seconds":14.,"track":"incoming","role":"score"}]}
    return e,before,usage,np.ones(40)


def recover(e,before,usage,mix,*,candidates=((0.,40.),),matches=(),components=None):
    seen=[]
    result,witness = cf._recover_nominated_heads(before,candidates,e,usage,matches,components,mix,seen)
    return result,witness,seen


def test_incoming_head_recovers_music_without_swallowing_outgoing_track():
    e,before,usage,mix=head_fixture()
    old=deepcopy(before)
    found,witness,seen=recover(e,before,usage,mix)
    assert before==old
    assert len(found)==2 and found[0]['start_seconds']==5.
    assert found[0]['end_seconds']==found[1]['start_seconds']==14.
    assert found[1]['end_seconds']==35.
    assert witness==list(range(14,22))
    assert seen[0]['state']=='nominated_head_recovered_without_outgoing_join'
    assert found[1]['start_edge']['recording_identity']=='unresolved'


def test_weak_mixture_seconds_remain_unconfirmed_and_missing_tags_keep_local_path():
    e,before,usage,mix=head_fixture()
    mix[17]=.2
    found,witness,seen=recover(e,before,usage,mix)
    assert witness==[14,15,16,18,19,20,21]
    assert found[1]['nominated_head']['unconfirmed_seconds']==[17]
    assert recover(e,before,usage,None)==(before,[],[])


def test_uncertain_intro_cannot_downgrade_an_existing_useful_body():
    e,before,usage,mix=head_fixture()
    mix[17:20]=.2
    found,witness,seen=recover(e,before,usage,mix)
    assert found==before and witness==[]
    assert seen[0]['state']=='nominated_head_kept_without_downgrading_body'


def test_nuisance_silence_or_unmaterialized_hole_cannot_create_a_head():
    e,before,usage,mix=head_fixture()
    p=e.music_probability.copy();p[18]=.05
    bad=cs.WindowEvidence(e.music_db,p,e.singing_probability,e.embeddings)
    assert recover(bad,before,usage,mix)[0]==before
    levels=e.music_db.copy();levels[18]=-100.
    bad=cs.WindowEvidence(levels,e.music_probability,e.singing_probability,e.embeddings)
    assert recover(bad,before,usage,mix)[0]==before
    assert recover(e,before,usage,mix,candidates=((0.,18.),(18.5,40.)))[0]==before


def test_label_without_transition_competing_onset_or_overlay_cannot_force_a_switch():
    e,before,usage,mix=head_fixture()
    flat=cs.WindowEvidence(e.music_db,e.music_probability,e.singing_probability,np.ones((40,2)))
    assert recover(flat,before,usage,mix)[0]==before
    extra={'onsets':usage['onsets']+[{'predicted_seconds':18.,'track':'another','role':'score'}]}
    assert recover(e,before,extra,mix)[0]==before
    records={'records':[component(5.,30.,track='outgoing')]}
    found,witness,seen=recover(e,before,usage,mix,components=records)
    assert found==before and witness==[] and seen[0]['state']=='nominated_head_has_overlapping_component'


def test_stereo_side_finds_recorded_music_under_centered_speech_without_raw_permission():
    rate=1000;rng=np.random.default_rng(984)
    ref=rng.normal(0,.1,(rate*12,2))
    source=np.zeros((rate*20,2));source[4*rate:16*rate]=ref
    speech=rng.normal(0,2.,(rate*20,1))
    mixed=source+speech
    assert not rs.observe(mixed.mean(1),ref.mean(1),rate)['spans']
    seen=rs.observe_stereo(mixed,source,ref,rate)
    assert seen and all(r['projection']=='stereo_difference' for r in seen)
    assert seen[0]['spans'][0]['start_seconds'] < 4.5
    assert seen[0]['identity_authority']==seen[0]['absence_authority']=='none'
    assert not rs.observe_stereo(mixed,speech.repeat(2,axis=1),ref,rate)


def test_stereo_source_only_intro_cannot_be_recovered_from_later_rendered_music():
    rate=1000;rng=np.random.default_rng(722)
    ref=rng.normal(0,.1,(rate*12,2))
    source=np.zeros((rate*20,2));source[4*rate:16*rate]=ref
    rendered=source.copy();rendered[:9*rate]=0.
    seen=rs.observe_stereo(source,rendered,ref,rate)
    assert seen and all(s['start_seconds']>=8.5 for r in seen for s in r['spans'])


def test_mono_render_tail_veto_survives_clean_side_under_centered_nuisance():
    rate=1000;rng=np.random.default_rng(802)
    ref=rng.normal(0,.1,(rate*12,2))
    source=np.zeros((rate*20,2));source[4*rate:16*rate]=ref
    rendered=source.copy();rendered[14*rate:16*rate]+=rng.normal(0,4.,(2*rate,1))
    seen=rs.observe_stereo(source,rendered,ref,rate)
    assert seen
    support={**seen[0]['spans'][0],'render_trace':seen[0]['render_trace']}
    assert not rs.tail_supported(support,14.,16.)
