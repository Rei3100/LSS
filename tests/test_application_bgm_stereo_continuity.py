"""The optional witness must preserve ordinary processing and owned state."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from lss import application
from lss.bgm import cue_formation as cf, cue_semantics as cs, event_tagging as et, stereo_continuity as sc


def evidence():
    return cs.WindowEvidence(np.full(30,-25.),np.r_[np.ones(10),np.zeros(4),np.ones(16)],
                             np.zeros(30),np.tile([1.,0.],(30,1)))


@pytest.mark.parametrize('failure',['projection','tagging'])
def test_optional_failure_keeps_ordinary_no_reference_route(tmp_path,monkeypatch,failure):
    e=evidence(); recipe=cs.default_recipe()
    scratch=tmp_path/'scratch';scratch.mkdir()
    if failure=='projection':
        def broken(*args):raise ValueError('projection unavailable')
        monkeypatch.setattr(sc,'project',broken)
    else:
        def broken(*args):raise application._BgmAnalysisFailed({'status':'error','message':'tagging unavailable'})
        monkeypatch.setattr(application,'_tag_job',broken)
    source=tmp_path/'music.wav'
    audio=np.full((30*44100,2),.1,dtype='float32');audio[:,1]*=-1
    sf.write(source,audio,44100,subtype='FLOAT')
    before=source.read_bytes()
    records,receipts,seconds=application._bgm_stereo_continuations(
        [(0,30)],e,recipe,source,scratch,tmp_path/'evidence',{},None)
    assert records[0]['state']=='stereo_continuation_not_checked'
    assert records[0]['span']==[10,14] and seconds==0
    baseline,_=cf.segment([(0,30)],e,recipe)
    fallback,_=cf.segment([(0,30)],e,recipe,stereo_continuations=records)
    assert fallback==baseline and source.read_bytes()==before
    owned,_=application._owned_analysis_files(scratch,'source',{},receipts=[(Path(p),r) for p,r in receipts])
    assert source.resolve() not in owned


def test_native_receipt_boundary_preserves_evidence_and_unknown_files(tmp_path,monkeypatch):
    e=evidence();recipe=cs.default_recipe();scratch=tmp_path/'scratch';scratch.mkdir()
    source=tmp_path/'music.wav'; audio=np.full((30*44100,2),.1,dtype='float32');audio[:,1]*=-1
    sf.write(source,audio,44100,subtype='FLOAT')
    def tags(signal,folder,*args):
        folder.mkdir(parents=True)
        n=sf.info(signal).frames//44100
        outputs={}
        for name,data in [('probabilities',np.full((n,527),.7,dtype='float16')),
                          ('embeddings',np.ones((n,2048),dtype='float16'))]:
            path=folder/f'{name}.npy';np.save(path,data)
            outputs[name]={'path':str(path),'sha256':application.file_sha256(path),'shape':list(data.shape)}
        receipt={'type':'bgm_event_tags_receipt','outputs':outputs}
        (folder/'receipt.json').write_text(json.dumps(receipt),encoding='utf8')
        return receipt
    monkeypatch.setattr(application,'_tag_job',tags)
    records,receipts,seconds=application._bgm_stereo_continuations(
        [(0,30)],e,recipe,source,scratch,tmp_path/'evidence',{},None)
    assert records[0]['state']=='stereo_music_continuation_observed'
    assert records[0]['positive_music_seconds']==list(range(10,14)) and seconds==20
    found,formation=cf.segment([(0,30)],e,recipe,stereo_continuations=records)
    assert len(found)==1 and formation['observations'][0]['state']=='stereo_continuation_hole_joined'
    vetoed,_=cf.segment([(0,30)],e,recipe,stereo_continuations=records,
        usage={'onsets':[{'predicted_seconds':12,'track':'incoming'}]})
    assert len(vetoed)==2
    sentinel=scratch/'side.wav';sentinel.write_bytes(b'user-owned lookalike')
    owned,documents=application._owned_analysis_files(scratch,'source',{},receipts=[(Path(p),r) for p,r in receipts])
    assert source.resolve() not in owned and sentinel.resolve() not in owned
    assert len(owned)==3 and len(documents)==2
    assert all(Path(item['path']).is_file() for item in records[0]['evidence']['outputs'].values())


def test_any_exception_in_one_hole_is_not_checked_and_the_job_continues(tmp_path,monkeypatch):
    e=evidence();recipe=cs.default_recipe();scratch=tmp_path/'scratch';scratch.mkdir()
    source=tmp_path/'music.wav'; audio=np.full((30*44100,2),.1,dtype='float32');audio[:,1]*=-1
    sf.write(source,audio,44100,subtype='FLOAT')
    def broken(*args):raise RuntimeError('unexpected')
    monkeypatch.setattr(sc,'project',broken)
    records,receipts,seconds=application._bgm_stereo_continuations(
        [(0,30)],e,recipe,source,scratch,tmp_path/'evidence',{},None)
    assert records[0]['state']=='stereo_continuation_not_checked' and records[0]['error']=='RuntimeError'
    assert seconds==0
    def interrupted(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(sc,'project',interrupted)
    with pytest.raises(KeyboardInterrupt):
        application._bgm_stereo_continuations([(0,30)],e,recipe,source,scratch,tmp_path/'evidence',{},None)
