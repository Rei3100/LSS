from dataclasses import replace

import numpy as np
import soundfile as sf

from lss.bgm import cue_formation as cf, cue_semantics as cs, episode_usage as eu


def evidence(vectors, probability=None):
    n=len(vectors)
    return cs.WindowEvidence(np.full(n,-25.),np.ones(n) if probability is None else probability,
                             np.zeros(n),np.asarray(vectors,dtype=float))


def match(a,b,reference_start=0.,path='a.flac',sha='a'):
    return dict(identity_state='known_exact',query_start_seconds=a,query_end_seconds=b,
                reference_start_seconds=reference_start,reference_path=path,reference_sha256=sha)


def test_nomination_needs_audio_and_never_changes_identity_or_absence():
    e=evidence(np.tile([1.,0.],(40,1)))
    usage={'onsets':[{'predicted_seconds':20.,'track':'unknown wanted'}]}
    found,receipt=cf.segment([(0,40)],e,cs.default_recipe(),usage=usage)
    assert [(r['start_seconds'],r['end_seconds']) for r in found]==[(0.,40.)]
    assert receipt['observations'][0]['state']=='nomination_without_acoustic_change'
    assert receipt['identity_authority']==receipt['absence_authority']=='none'


def test_short_different_cue_can_be_split_at_audio_checked_nominated_position():
    e=evidence(np.r_[np.tile([1.,0.],(20,1)),np.tile([0.,1.],(10,1))])
    usage={'onsets':[{'predicted_seconds':20.5,'track':'second'}]}
    # Old whole-interval splitter can be conservative; nominations use local
    # evidence rather than weakening its universal threshold.
    found,_=cf.segment([(0,30)],e,replace(cs.default_recipe(),split_distance=2.),usage=usage)
    assert len(found)==2
    assert found[0]['end_seconds']==found[1]['start_seconds']==20
    assert found[1]['end_seconds']==30


def test_fragment_continuity_does_not_veto_unknown_music_outside_coverage():
    assert cf._continuity([match(5,15)],10)
    assert not cf._continuity([match(5,15)],20)
    assert not cf._continuity([match(5,15),match(6,14,path='b',sha='b')],10)


def test_classifier_hole_joins_with_continuing_music_but_real_gap_stays():
    e=evidence(np.tile([1.,0.],(30,1)),np.r_[np.ones(10),np.full(4,.3),np.ones(16)])
    found,_=cf.segment([(0,30)],e,cs.default_recipe())
    assert len(found)==1 and found[0]['end_seconds']==30
    e=evidence(np.tile([1.,0.],(30,1)),np.r_[np.ones(10),np.zeros(4),np.ones(16)])
    found,_=cf.segment([(0,30)],e,cs.default_recipe())
    assert len(found)==2


def test_disagreeing_music_flanks_cannot_bridge_classifier_hole():
    e=evidence(np.r_[np.tile([1.,0.],(10,1)),np.tile([.5,.5],(4,1)),np.tile([.90,.43589],(16,1))],
               np.r_[np.ones(10),np.full(4,.3),np.ones(16)])
    found,record=cf.segment([(0,30)],e,replace(cs.default_recipe(),split_distance=2.))
    assert len(found)==2
    assert not any(r['state']=='continuation_hole_joined' for r in record['observations'])


def test_refused_hole_leaves_the_per_second_evidence_that_decided_it():
    e=evidence(np.tile([1.,0.],(30,1)),np.r_[np.ones(10),np.full(1,.1),np.full(3,.3),np.ones(16)])
    found,record=cf.segment([(0,30)],e,cs.default_recipe())
    assert len(found)==2
    refused=[r for r in record['observations'] if r['state']=='continuation_hole_refused']
    assert len(refused)==1
    row=refused[0]
    assert row['span']==[10,14] and row['audible'] is False and row['similar_enough'] is True
    assert row['min_music_probability']==0.1 and row['nominated_onset'] is False
    assert row['identity_authority']=='none'


def test_fine_edges_do_not_cross_original_candidate_support(tmp_path):
    rate=8000
    path=tmp_path/'music.wav'
    x=np.zeros((rate*8,2));x[rate*2:rate*5]=.1
    sf.write(path,x,rate,subtype='FLOAT')
    e=evidence(np.tile([1.,0.],(8,1)),np.r_[np.zeros(2),np.ones(3),np.zeros(3)])
    found,_=cf.segment([(2,5)],e,cs.default_recipe(),music=path)
    assert len(found)==1 and 2<=found[0]['start_seconds']<found[0]['end_seconds']<=5


def theme_matches():
    return [match(100.,188.,0.,'星をつなごう TV size ver..flac'),
            match(1300.,1388.,0.,'まるい雲 (TV size).flac')]


def test_no_shipped_usage_table_nominates_anything_whatever_the_file_name():
    # A development snapshot of per-episode tables is not shipped: no file name selects a work.
    for name in ('series-alpha-137.mkv', '星空冒険団 第137話.mkv', 'other137.mkv'):
        assert eu.nominate(name, theme_matches()) == {'state': 'no_usage_table', 'onsets': []}


def test_fractional_unknown_music_is_held_instead_of_false_absence():
    e=evidence(np.tile([1.,0.],(20,1)))
    item=dict(start_seconds=10.51,end_seconds=10.63,core_seconds=[10.51,10.63],
              start_edge={'kind':cs.EDGE_CANDIDATE},end_edge={'kind':cs.EDGE_SILENCE})
    [found]=cs.decide([item],e,cs.default_recipe(),source_seconds=20.)
    assert found['semantic_class']==cs.UNCERTAIN
    assert found['features']['seconds']==10.63-10.51
    assert found['features']['music_fraction']==1.
    assert 'music_not_heard' not in found['reasons']
