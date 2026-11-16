"""The proposed OS quota must never round the user's quiet ceiling upward."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from quiet_cpu_rate_probe import rate_units
from quiet_cpu import capped_child_env, quiet_wall_timeout


@pytest.mark.parametrize('cores', [1, 2, 8, 16, 24, 32, 48, 50])
def test_representable_quota_rounds_down(cores):
    units = rate_units(.005, cores)
    assert units >= 1
    assert units * cores / 10000 <= .005
    assert (units+1) * cores / 10000 > .005


@pytest.mark.parametrize('duty,cores', [(.005, 51), (.005, 128), (0., 16),
    (-1., 16), (float('nan'), 16), (float('inf'), 16), (True, 16), (.005, True), (.005, 0)])
def test_unrepresentable_or_invalid_quota_refuses(duty, cores):
    with pytest.raises(ValueError):
        rate_units(duty, cores)


def test_quiet_wall_timeout_uses_only_valid_child_duty(monkeypatch):
    monkeypatch.delenv('LSS_QUIET_CPU_ONE_CORE_DUTY', raising=False)
    assert quiet_wall_timeout(15) == 15
    monkeypatch.setenv('LSS_QUIET_CPU_ONE_CORE_DUTY', '0.25')
    assert quiet_wall_timeout(15) == 60
    for invalid in ('0', '-1', '1.1', 'nan', 'inf', 'broken'):
        monkeypatch.setenv('LSS_QUIET_CPU_ONE_CORE_DUTY', invalid)
        assert quiet_wall_timeout(15) == 15


def test_actual_child_cap_overrides_inherited_duty_without_mutating_parent(monkeypatch):
    monkeypatch.setenv('LSS_QUIET_CPU_ONE_CORE_DUTY', '0.5')
    parent = {'LSS_QUIET_CPU_ONE_CORE_DUTY': '0.5', 'KEEP': 'yes'}
    child = capped_child_env(parent, units=2, cores=8)
    assert child['LSS_QUIET_CPU_ONE_CORE_DUTY'] == '0.0016'
    assert child['KEEP'] == 'yes'
    assert parent['LSS_QUIET_CPU_ONE_CORE_DUTY'] == '0.5'
    assert quiet_wall_timeout(10) == 20
