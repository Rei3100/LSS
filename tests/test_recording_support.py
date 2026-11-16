"""Independent component truth and negative clocks, never whole-span EXACT."""
import numpy as np
from lss.bgm import recording_support as rs, cue_formation as cf, cue_semantics as cs


def test_aligned_component_has_two_ends_and_shift_controls_reject_other_sound():
    rate=1000
    rng=np.random.default_rng(18)
    reference=rng.normal(0,.1,rate*12)
    source=rng.normal(0,.04,rate*20)
    source[4*rate:16*rate]+=reference
    seen=rs.observe(source,reference,rate)
    assert seen['state']=='component_observed'
    [span]=seen['spans']
    assert abs(span['start_seconds']-4)<.5 and abs(span['end_seconds']-16)<.5
    assert seen['identity_authority']==seen['absence_authority']=='none'
    assert rs.observe(rng.normal(0,.1,rate*20),reference,rate)['spans']==[]


def test_constant_tonal_similarity_is_not_a_recording_clock():
    rate=1000
    reference=np.sin(2*np.pi*100*np.arange(rate*12)/rate)
    assert rs.observe(reference,reference,rate)['spans']==[]


def test_two_overlapping_recordings_keep_both_positive_envelopes():
    rate=1000
    rng=np.random.default_rng(12)
    a,b=rng.normal(0,.1,(2,rate*12))
    mixture=np.zeros(rate*20)
    mixture[2*rate:14*rate]+=a
    mixture[6*rate:18*rate]+=b
    [first]=rs.observe(mixture,a,rate)['spans']
    [second]=rs.observe(mixture,b,rate)['spans']
    assert first['start_seconds']<second['start_seconds']<first['end_seconds']<second['end_seconds']


def parts(a,b):
    return dict(start_seconds=a,end_seconds=b,core_seconds=[a,b],stretch=[0.,40.],
                candidates=[[0.,40.]],start_edge={'kind':cs.EDGE_SILENCE},end_edge={'kind':cs.EDGE_SILENCE})


def component(a,b,*,sha='a',onset=5.,track='score'):
    return dict(state='component_observed',reference_sha256=sha,offset_seconds=-a,
                onset={'predicted_seconds':onset,'track':track},spans=[dict(start_seconds=a,end_seconds=b)],
                render_trace={'origin':0.,'times':np.arange(0,40,.1).tolist(),
                              'fraction':np.ones(400).tolist(),'control':np.zeros(400).tolist()})


def test_measured_component_recovers_only_materialized_classifier_hole():
    before=[parts(5.,14.),parts(19.,30.)]
    found=cf._recover_components(before,[(0.,40.)],{'records':[component(4.9,31.)]},[])
    assert len(found)==1 and found[0]['start_seconds']==4.9 and found[0]['end_seconds']==31.
    assert before[0]['end_seconds']==14.  # does not mutate the baseline


def test_competing_component_prevents_join_and_positive_matches_never_trim_tail():
    before=[parts(5.,14.),parts(19.,30.)]
    found=cf._recover_components(before,[(0.,40.)],{'records':[component(5.,28.),
          component(13.,20.,sha='b',onset=13.,track='different')]},[])
    assert len(found)==2 and found[-1]['end_seconds']>=30.


def test_measured_silent_pause_is_not_connected_from_flank_color_alone():
    found=cf._recover_components([parts(5.,14.),parts(19.,30.)],[(0.,40.)],
                                {'records':[component(5.,14.),component(19.,30.)]},[])
    assert len(found)==2


def test_an_unmatched_incoming_usage_row_blocks_a_same_recording_join():
    before=[parts(5.,14.),parts(19.,30.)]
    observations=[]
    found=cf._recover_components(before,[(0.,40.)],{'records':[component(5.,30.)],
          'onsets':[{'predicted_seconds':16.,'track':'unreleased incoming'}]},observations)
    assert len(found)==2
    assert not any(o['state']=='recording_component_recovers_classifier_hole' for o in observations)


def test_unresolved_recording_role_remains_unknown_when_the_music_classifier_is_low():
    observed=component(5.,9.);observed['music_corroborated']=False
    [part]=cf._recover_components([parts(7.,9.)],[(0.,40.)],{'records':[observed]},[])
    e=cs.WindowEvidence(np.full(40,-25.),np.zeros(40),np.zeros(40),np.ones((40,2)))
    [decided]=cs.decide([part],e,cs.default_recipe(),source_seconds=40.)
    assert decided['semantic_class']==cs.UNCERTAIN
    assert decided['reasons']==['observed_recording_role_unresolved']
    assert part['unresolved_recording_regions']==[[5.,9.]]


def test_late_edit_cannot_establish_the_missing_intro():
    observed=component(20.,30.,onset=5.)
    found=cf._recover_components([parts(22.,28.)],[(0.,40.)],{'records':[observed]},[])
    assert found[0]['start_seconds']==22. and found[0]['end_seconds']==28.


def test_source_component_does_not_bridge_missing_materialized_audio():
    before=[parts(5.,14.),parts(14.5,30.)]
    found=cf._recover_components(before,[(0.,14.),(14.5,40.)],{'records':[component(5.,30.)]},[])
    assert len(found)==2 and found[0]['end_seconds']==14.


def test_source_presence_does_not_extend_a_noise_dominated_separator_tail():
    r=component(5.,31.)
    r['render_trace']['fraction']=[.05]*400
    observations=[]
    found=cf._recover_components([parts(5.,30.)],[(0.,40.)],{'records':[r]},observations)
    assert found[0]['end_seconds']==30.
    assert observations[0]['state']=='rendered_tail_not_corroborated'


def test_positive_switch_and_layered_component_are_not_merged():
    a,b=parts(5.,15.),parts(15.,30.)
    edge=dict(kind=cs.EDGE_CANDIDATE,authority='audio_checked_positive_nomination',track_label='second')
    a['end_edge']=b['start_edge']=edge
    records=[component(5.,20.),component(15.,30.,sha='b',onset=15.,track='second')]
    observations=[]
    found=cf._recover_components([a,b],[(0.,40.)],{'records':records},observations)
    assert len(found)==2
    assert any(r['state']=='overlapping_recording_components' for r in observations)


def test_measured_music_presence_is_local_and_duplicate_votes_cannot_cover_holes():
    r={'state':'component_observed','trace_origin':0.,'times':np.arange(3.05,6.05,.1).tolist(),
       'fraction':[.5]*30,'control':[.001]*30}
    mask=rs.confirmed_music_windows({'records':[r,r]},10)
    assert mask.tolist()==[False,False,False,True,True,True,False,False,False,False]
    r['fraction']=[0.]*30
    assert not rs.confirmed_music_windows({'records':[r]},10).any()


def test_positive_presence_changes_only_music_evidence_not_unknown_edges_or_singing():
    e=cs.WindowEvidence(np.full(20,-25.),np.r_[np.ones(8),np.zeros(4),np.ones(8)],np.zeros(20),
                        np.tile([1.,0.],(20,1)))
    part=parts(0.,20.)
    before=cs.decide([part],e,cs.default_recipe(),source_seconds=20.)[0]
    mask=np.zeros(20,dtype=bool);mask[8:12]=True
    after=cs.decide([part],e,cs.default_recipe(),source_seconds=20.,additional_music_windows=mask)[0]
    assert 'music_not_continuous' in before['reasons']
    assert 'music_not_continuous' not in after['reasons']
    assert after['features']['known_fraction']==0.


def test_recovered_head_inside_materialization_keeps_the_cue_publishable():
    # A synthetic recovered head extends the cue while preserving its original support.
    found=cf._recover_components([parts(7.,30.)],[(0.,40.)],{'records':[component(6.6,30.)]},[])
    [part]=found
    assert part['start_seconds']==6.6 and part['start_edge']['kind']==cs.EDGE_SILENCE
    assert part['start_edge']['previous_seconds']==7.
    e=cs.WindowEvidence(np.full(40,-25.),np.r_[np.zeros(7),np.ones(23),np.zeros(10)],np.zeros(40),
                        np.tile([1.,0.],(40,1)))
    [decided]=cs.decide(found,e,cs.default_recipe(),source_seconds=40.)
    assert not any(r.endswith('_music_continues_outside_candidate') for r in decided['reasons'])


def test_head_clipped_at_the_materialized_start_may_continue_outside():
    part=parts(7.,30.);part['stretch']=[5.,40.]
    [found]=cf._recover_components([part],[(5.,40.)],{'records':[component(4.,30.)]},[])
    assert found['start_seconds']==5. and found['start_edge']['kind']==cs.EDGE_CANDIDATE


def test_tail_beyond_materialization_keeps_prior_useful_boundary():
    part=parts(5.,30.);part['stretch']=[0.,31.]
    observations=[]
    found=cf._recover_components([part],[(0.,31.)],{'records':[component(5.,32.)]},observations)
    assert found[0]['end_seconds']==30. and found[0]['end_edge']['kind']==cs.EDGE_SILENCE
    assert observations[0]['state']=='positive_tail_exceeds_materialized_support'
