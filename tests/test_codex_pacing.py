import json
import pytest
from scripts import codex_pacing as pacing


def test_native_primary_uses_codex_bucket_without_weekly_pacing():
    response = {'ordinaryUsageAllowed': True, 'rateLimits': {'primary': None},
                'rateLimitsByLimitId': {'codex': {'primary': {
                    'usedPercent': 73, 'windowDurationMins': 300, 'resetsAt': 110000},
                    'secondary': {'usedPercent': 100}}}}
    samples = pacing.native_observations(response, now=100000, source='native')
    assert samples == [dict(timestamp=100000, used_percent=73, reset_epoch=110000, source='native')]


def test_native_permission_denial_and_malformed_or_expired_windows_are_not_spare_allowance():
    primary = dict(usedPercent=1, windowDurationMins=300, resetsAt=110000)
    for patch in [dict(usedPercent=True), dict(resetsAt=99999), dict(windowDurationMins=10080)]:
        with pytest.raises(ValueError):
            pacing.native_observations({'rateLimits': {'primary': {**primary, **patch}}}, now=100000, source='native')
    with pytest.raises(ValueError, match='denies'):
        pacing.native_observations({'ordinaryUsageAllowed': False, 'rateLimits': {'primary': primary}}, now=100000, source='native')
    for malformed in [None, {'rateLimitsByLimitId': [1]}, {'rateLimits': []}, {'rateLimits': {'primary': [1]}}]:
        with pytest.raises(ValueError):
            pacing.native_observations(malformed, now=100000, source='native')


def sample(used, now, *, age=0, elapsed=9000):
    return dict(timestamp=now-age, used_percent=used,
                reset_epoch=now+18000-elapsed, source='saved-log')


def test_shared_late_worker_burst_never_gets_delegated():
    now = 100000
    result = pacing.advise([sample(86, now), sample(100, now)], now=now, active_workers=1)
    assert not result['allow_delegation'] and not result['allow_parent_step']
    assert result['interrupt_workers'] and result['used_percent'] == 100


def test_concurrent_percentages_use_high_water_not_sum_or_latest_lower():
    now = 100000
    a = sample(40, now, age=10, elapsed=12000)
    b = sample(30, now, elapsed=12000)
    result = pacing.advise([a, b], now=now)
    assert result['used_percent'] == 40 and result['allow_parent_step']


def test_old_reset_deadline_does_not_establish_renewal():
    now = 100000
    result = pacing.advise([dict(timestamp=now-1, used_percent=100, reset_epoch=now-1, source='old')], now=now)
    assert result['action'] == 'mechanical_only' and not result['allow_delegation']
    assert pacing.advise([sample(1, now, age=301)], now=now)['action'] == 'mechanical_only'


def test_early_burst_quiets_before_late_limit_and_weekly_is_unused(tmp_path):
    now = 100000
    assert not pacing.advise([sample(40, now, elapsed=3600)], now=now)['allow_parent_step']
    path = tmp_path/'log.jsonl'
    path.write_text(json.dumps({'timestamp':pacing.utc(now), 'payload':{'rate_limits':{
        'primary':{'used_percent':1,'window_minutes':300,'resets_at':now+17900},
        'secondary':{'used_percent':100,'window_minutes':10080,'resets_at':now+100000}}}})+'\n')
    result = pacing.advise(list(pacing.observations(path)), now=now)
    assert result['allow_delegation'] and result['weekly_pacing'] is False


def test_only_one_worker_and_no_new_worker_near_reset():
    now = 100000
    assert not pacing.advise([sample(10, now, elapsed=9000)], now=now, active_workers=1)['allow_delegation']
    assert not pacing.advise([sample(54, now, elapsed=17000)], now=now)['allow_delegation']
    result = pacing.advise([sample(80, now, elapsed=17000)], now=now, active_workers=1)
    assert result['interrupt_workers'] and not result['allow_parent_step']


def test_partial_tail_and_invalid_values_fail_closed(tmp_path):
    path=tmp_path/'log.jsonl'
    path.write_text('broken\n'+json.dumps({'timestamp':pacing.utc(100000),'payload':{'rate_limits':{
        'primary':{'used_percent':True,'window_minutes':300,'resets_at':110000}}}})+'\n')
    assert list(pacing.observations(path)) == []
