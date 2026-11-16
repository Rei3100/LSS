"""The mixture and the separated base have different processing authority."""
import numpy as np
from dataclasses import replace
from lss.bgm import cue_representation as cr, dialogue_residual as dr


def nomination():
    return dict(state=cr.STATE_RAW,representation=cr.RAW,reasons=['mixture_equals_music'],fallback=False)


def test_low_total_residual_cannot_restore_short_one_channel_effect():
    rate=8000
    t=np.arange(4*rate)/rate
    target=np.repeat((.2*np.sin(2*np.pi*220*t))[:,None],2,axis=1)
    source=target.copy()
    source[rate:rate+40,1]+=.08
    assert 10*np.log10(np.sum((source-target)**2)/np.sum(target**2)) < -20
    result=cr.certify_raw(nomination(),source,target,rate,allowed=np.ones(16,dtype=bool),native_grid=True)
    assert result['representation']==cr.PROCESSED
    assert result['original_check']['additional_processing_authority']=='none'


def test_raw_retains_every_sample_only_with_full_permission_and_native_witness():
    rate=8000
    x=np.random.default_rng(3).normal(0,.1,(rate*2,2))
    allowed=np.ones(8,dtype=bool)
    assert cr.certify_raw(nomination(),x,x,rate,allowed=allowed,native_grid=True)['representation']==cr.RAW
    assert cr.certify_raw(nomination(),x,x,rate,allowed=allowed,native_grid=False)['representation']==cr.PROCESSED
    allowed[3]=False
    assert cr.certify_raw(nomination(),x,x,rate,allowed=allowed,native_grid=True)['representation']==cr.PROCESSED


def test_skipping_additional_cleanup_retains_separated_music_and_protects_choir():
    p=np.zeros((8,527),dtype=np.float32)
    p[2:5,0]=.9
    p[3,28]=.9
    plan=dr.processing_plan(p,0,8000,1000,dr.default_recipe())
    assert plan['regions']==[[2000,3000],[4000,5000]]
    assert plan['musical_voice_regions']==[[3000,4000]]
    assert plan['retain_separated_regions']==[[0,2000],[3000,4000],[5000,8000]]
    assert plan['original_mixture_authority']==plan['absence_authority']=='none'


def test_dilated_speech_context_never_overwrites_protected_choir():
    p=np.zeros((8,527),dtype=np.float32);p[2,0]=.9;p[3,28]=.9
    plan=dr.processing_plan(p,0,8000,1000,replace(dr.default_recipe(),local_dilation_windows=2))
    assert plan['regions']==[[0,3000],[4000,5000]]
    assert [3000,4000] in plan['musical_voice_regions']
    assert all(not(a<4000 and b>3000) for a,b in plan['regions'])


def test_missing_classifier_tail_is_unknown_and_keeps_separated_base():
    p=np.zeros((3,527),dtype=np.float32)
    plan=dr.processing_plan(p,2000,5000,1000,dr.default_recipe())
    assert plan['unknown_regions']==[[1000,3000]]
    assert plan['retain_separated_regions']==[[0,3000]]
    assert plan['original_mixture_authority']=='none'
