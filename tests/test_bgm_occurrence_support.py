import hashlib
import json
import numpy as np
import soundfile as sf
from lss.bgm import occurrence_support as osup, cue_semantics as cs, identity
from tests.test_bgm_identity import _music


def fixture(tmp_path):
    rate=identity.SAMPLE_RATE
    signal=_music(173,12.)
    source=tmp_path/'prior.wav';sf.write(source,signal,rate,subtype='FLOAT')
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    evidence=cs.WindowEvidence(np.full(30,-25.),np.ones(30),np.zeros(30),np.ones((30,2)))
    usage={'state':'aligned_positive_nominations','episode':1,'onsets':[
        {'role':'score','track':'unreleased','track_id':[375,99],'predicted_seconds':1.}]}
    osup.remember(source,sha,usage,evidence,tmp_path)
    query=np.r_[np.zeros(5*rate),signal,np.zeros(4*rate)]
    current={**usage,'episode':2,'onsets':[{**usage['onsets'][0],'predicted_seconds':6.}]}
    return source,sha,evidence,usage,query,current


def test_prior_recording_presence_cannot_create_donor_or_release_authority(tmp_path):
    source,sha,e,usage,query,current=fixture(tmp_path)
    r=osup.nominate(query,query,current,e,'b'*64,tmp_path)
    assert r['records'] and r['used_sources']
    assert r['donor_authority']==r['identity_authority']=='none'
    assert all(x['reference_type']=='episode_occurrence' and x['availability_authority']=='none' for x in r['records'])
    assert osup.nominate(query,query,current,e,sha,tmp_path)['records']==[]


def test_missing_or_modified_occurrence_and_bad_clock_keep_local_route(tmp_path):
    source,sha,e,usage,query,current=fixture(tmp_path)
    source.write_bytes(b'changed source')
    r=osup.nominate(query,query,current,e,'b'*64,tmp_path)
    assert not r['records'] and r['skipped']
    assert not osup.nominate(query,query,{**current,'state':'theme_clocks_disagree'},e,'b'*64,tmp_path)['records']


def test_memory_does_not_overwrite_unknown_user_record(tmp_path):
    source,sha,e,usage,query,current=fixture(tmp_path)
    target=tmp_path/'bgm-occurrence-support'/f'{sha}.json'
    target.write_text('{"user_note":"keep"}',encoding='utf8')
    before=target.read_bytes()
    assert osup.remember(source,sha,usage,e,tmp_path)['state']=='existing_unowned_record_preserved'
    assert target.read_bytes()==before


def test_unrelated_recording_and_source_only_nuisance_do_not_recover_music(tmp_path):
    source,sha,e,usage,query,current=fixture(tmp_path)
    unrelated=np.random.default_rng(42).normal(0,.1,len(query)).astype(np.float32)
    assert not osup.nominate(unrelated,unrelated,current,e,'b'*64,tmp_path)['records']
    assert not osup.nominate(query,unrelated,current,e,'b'*64,tmp_path)['records']


def test_observed_recording_with_unresolved_role_is_kept_without_music_publication_authority(tmp_path):
    from lss.bgm import recording_support as rs
    source,sha,e,usage,query,current=fixture(tmp_path)
    e=cs.WindowEvidence(e.music_db,np.zeros(30),e.singing_probability,e.embeddings)
    path=tmp_path/'bgm-occurrence-support'/f'{sha}.json'
    record=json.loads(path.read_text('utf8'));record['music_positive_seconds']=[]
    path.write_text(json.dumps(record),encoding='utf8')
    found=osup.nominate(query,query,current,e,'b'*64,tmp_path)
    assert found['records'] and all(not r['music_corroborated'] for r in found['records'])
    assert not rs.confirmed_music_windows(found,30).any()
