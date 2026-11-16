import numpy as np
import pytest
import soundfile as sf
from dataclasses import replace

from lss.bgm import cue_semantics as cs, stereo_continuity as sc


def fixture():
    n=30
    e=cs.WindowEvidence(np.full(n,-25.),np.r_[np.ones(10),np.zeros(4),np.ones(16)],
                        np.zeros(n),np.tile([1.,0.],(n,1)))
    recipe=cs.default_recipe()
    return e,recipe


def test_plan_requires_exact_support_and_same_flanks():
    e,r=fixture()
    assert len(sc.plan_holes([(0,30)],e,r))==1
    assert not sc.plan_holes([(0,11.9),(12,30)],e,r)
    parts=cs.segment([(0,30)],e,r)
    parts[1]['stretch']=99
    assert not sc.nominate(parts,[(0,30)],e,r)
    parts=cs.segment([(0,30)],e,r)
    parts[1]['start_edge']['kind']=cs.EDGE_CHANGE
    assert not sc.nominate(parts,[(0,30)],e,r)
    parts=cs.segment([(0,30)],e,r)
    parts[0]['end_seconds']=9.9
    assert not sc.nominate(parts,[(0,30)],e,r)
    e.embeddings[14:]=[0.,1.]
    assert not sc.plan_holes([(0,30)],e,r)


@pytest.mark.parametrize('mode',['side','silence','center'])
def test_projection_and_positive_grid(tmp_path,mode):
    e,r=fixture()
    plan=sc.plan_holes([(0,30)],e,r)[0]
    x=np.full((30*44100,2),.1,dtype=np.float32)
    if mode=='side':x[:,1]*=-1
    if mode=='silence':x[:]=0
    source=tmp_path/'source.wav'
    sf.write(source,x,44100,subtype='FLOAT')
    receipt=sc.project(source,plan,tmp_path/'projection')
    y,rate=sf.read(receipt['signal_path'])
    assert rate==44100 and np.array_equal(y[:,0],y[:,1])
    assert np.max(np.abs(y)) <= .100001
    n=receipt['count']
    vectors=np.tile([1.,0.],(n,1))
    result=sc.evaluate(plan,receipt,np.ones(n),vectors,r)
    assert result['positive_music_seconds']==(list(range(10,14)) if mode=='side' else [])
    low=np.ones(n);low[11-receipt['origin_seconds']]=.1
    assert not sc.evaluate(plan,receipt,low,vectors,r)['positive_music_seconds']
    vectors[n//2:]=[0.,1.]
    assert not sc.evaluate(plan,receipt,np.ones(n),vectors,r)['positive_music_seconds']
    with pytest.raises(ValueError):sc.project(source,plan,tmp_path/'projection')
    for p in (np.ones(n-1),np.full(n,np.nan)):
        with pytest.raises(ValueError):sc.evaluate(plan,receipt,p,vectors,r)
    p=np.ones((n,527));p[0,0]=np.nan
    with pytest.raises(ValueError):sc.evaluate(plan,receipt,p,vectors,r)
    with pytest.raises(ValueError):
        sc.evaluate(plan,receipt,np.ones(n),vectors,replace(r,doubt_distance=.12))
    with pytest.raises(ValueError):sc.evaluate(plan,receipt,np.ones(n),vectors[:-1],r)
    assert not sc.evaluate(plan,receipt,np.ones(n),np.zeros_like(vectors),r)['positive_music_seconds']


def test_noise_without_music_cannot_supply_positive_seconds(tmp_path):
    e,r=fixture();plan=sc.plan_holes([(0,30)],e,r)[0]
    noise=np.random.default_rng(1).normal(0,.1,(30*44100,2)).astype('float32')
    source=tmp_path/'noise.wav';sf.write(source,noise,44100,subtype='FLOAT')
    receipt=sc.project(source,plan,tmp_path/'projection')
    n=receipt['count']
    result=sc.evaluate(plan,receipt,np.full(n,.1),np.tile([1.,0.],(n,1)),r)
    assert result['side_audible'] and not result['positive_music_seconds']
