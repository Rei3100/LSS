import numpy as np
from lss.bgm import cue_semantics as cs, reference_relations as rr, release_availability as ra
from tests.release_receipts import distribution_receipt


def test_exact_audio_with_title_only_cannot_suppress_an_unknown_release():
    sha='a'*64
    relation={'state':rr.EXACT,'reference':{'reference_sha256':sha,'display_title':'Released-looking title'}}
    item={'reference_relation':relation,'semantic_class':cs.WANTED,'reasons':[],'start_seconds':0.,'end_seconds':20.}
    [kept]=rr.apply([item],cs.KNOWN_RELEASED)
    assert kept['semantic_class']==cs.WANTED
    assert kept['independent_availability']=='unknown_preserve_extraction'
    assert rr.occurrences([kept])==[]
    relation['reference']['independent_distribution']=distribution_receipt(sha)
    [known]=rr.apply([item],cs.KNOWN_RELEASED)
    assert known['semantic_class']==cs.KNOWN_RELEASED


def test_distribution_receipt_cannot_transfer_to_different_audio_bytes():
    receipt=distribution_receipt('a'*64)
    assert ra.checked(receipt,'a'*64)
    assert not ra.checked(receipt,'b'*64)
    assert not ra.checked({**receipt,'recording_url':None},'a'*64)
    assert ra.lookup('b'*64,{'independent_distribution':receipt})['state']=='independent_distribution_unknown'


def _acquired(sha='c'*64, track='AbCdEfGhIjK', album='https://music.youtube.com/browse/MPREb_abc123'):
    return {'sha256':sha,'origin':'acquired',
            'provenance':{'authority':{'id':track},'acquired_at':'2026-10-19T00:00:00Z'},
            'discovery':{'track_id':track,'release':{'album_url':album,'album_title':'Series Alpha Original Soundtrack'}}}


def test_a_track_lss_acquired_from_a_verified_release_listing_carries_its_own_receipt():
    receipt=ra.from_record(_acquired())
    assert ra.checked(receipt,'c'*64) and receipt['release_id']=='MPREb_abc123'
    assert receipt['recording_url']=='https://music.youtube.com/watch?v=AbCdEfGhIjK'
    listed=_acquired(album='https://music.youtube.com/playlist?list=OLAK5uy_xyz')
    assert ra.from_record(listed)['release_id']=='OLAK5uy_xyz'


def test_no_receipt_without_a_release_listing_or_for_other_bytes_or_user_media():
    record=_acquired()
    assert ra.from_record({**record,'discovery':{'track_id':'AbCdEfGhIjK'}}) is None       # song search only
    assert ra.from_record({**record,'origin':'user_library'}) is None
    other=_acquired(); other['provenance']['authority']['id']='ZZZZZZZZZZZ'                 # a different video
    assert ra.from_record(other) is None
    assert ra.from_record(_acquired(album='https://www.youtube.com/playlist?list=PLuser')) is None
    assert ra.from_record(_acquired(album='https://music.youtube.com/playlist?list=PLuser')) is None


def test_the_users_own_copy_of_a_release_needs_no_extraction_but_lss_output_does_not_count(tmp_path):
    import soundfile as sf
    from mutagen.flac import FLAC
    sha='d'*64
    owned={'sha256':sha,'origin':'user_library','metadata':{'lss_output':False}}
    copy=ra.user_copy(owned)
    assert ra.available(copy,sha) and not ra.available(copy,'e'*64)
    assert ra.lookup(sha,{'independent_distribution':copy})['state']==ra.USER_COPY
    assert ra.user_copy({**owned,'metadata':{'lss_output':True}}) is None
    assert ra.user_copy({**owned,'origin':'acquired'}) is None
    # LSS's own published file placed among the references is recognised by its LSS_* tags.
    path=tmp_path/'episode - BGM 00m01s.flac'
    sf.write(str(path),np.zeros((4410,2),dtype=np.float32),44100)
    audio=FLAC(str(path)); audio['title']='x'; audio['LSS_BGM_PROVENANCE']='{}'; audio.save()
    assert rr.reference_metadata(path)['lss_output'] is True
    plain=tmp_path/'released.flac'
    sf.write(str(plain),np.zeros((4410,2),dtype=np.float32),44100)
    assert rr.reference_metadata(plain)['lss_output'] is False
