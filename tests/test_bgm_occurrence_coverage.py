"""A positive recording summary must not turn unobserved gaps into EXACT."""
from lss.bgm import reference_relations as rr
from tests.release_receipts import distribution_receipt

def exact(a,b,offset=5.):
    return {'start_seconds':a,'end_seconds':b,'reference_relation':{
        'state':rr.EXACT,'reference':{'reference_sha256':'a'*64,'offset_seconds':offset,
            'independent_distribution':distribution_receipt('a'*64)},
        'reference_range_seconds':[a+offset,b+offset],
        'starts_at_reference_start':False,'ends_at_reference_end':False}}

def test_half_second_unknown_gap_and_one_second_pause_keep_two_exact_scopes():
    for gap in (.5,1.):
        found=rr.occurrences([exact(10.,20.),exact(20.+gap,30.)])
        assert [(r['start_seconds'],r['end_seconds']) for r in found]==[(10.,20.),(20.+gap,30.)]
        assert sum(r['end_seconds']-r['start_seconds'] for r in found)==20.-gap

def test_tv_edit_offsets_and_duplicate_overlap_cannot_claim_one_continuous_recording():
    assert len(rr.occurrences([exact(10.,20.),exact(20.,30.,5.02)]))==2
    assert len(rr.occurrences([exact(10.,20.),exact(19.5,30.)]))==2

def test_verified_contiguous_scopes_still_merge_with_one_clock():
    [found]=rr.occurrences([exact(10.,20.),exact(20.,30.)])
    assert found['start_seconds']==10. and found['end_seconds']==30.
    assert found['reference_range_seconds']==[15.,35.]
    assert found['pieces']==2
