"""A faster setting cannot win after a quality or repeatability failure."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import importlib.util
from pathlib import Path


ARTIFACT = dev_evidence_root(Path(__file__).resolve().parents[1]) / 'release-evidence/strata-matrix-20261029/hardware_tune.py'
spec = importlib.util.spec_from_file_location('strata_hardware_tune', ARTIFACT)
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


def measured(settings, seconds, **changes):
    return dict(label='fixture', knobs=settings, status='measured', request_seconds=seconds,
                controls_passed=4, repairs_accepted=2, resident_confirmed=True, **changes)


def test_speed_does_not_override_quality_and_failed_repeat_disqualifies_setting():
    baseline = measured(tune.knobs(), 100)
    fast = measured(tune.knobs(reserve=512), 50)
    failed = dict(fast, controls_passed=3, request_seconds=20)
    assert tune.choose([baseline, failed])[0] == baseline
    assert tune.choose([baseline, fast])[0] == fast
    assert tune.choose([baseline, fast, failed])[0] == baseline
    assert tune.choose([failed]) is None


def test_residency_must_be_confirmed_and_only_measured_results_qualify():
    row = measured(tune.knobs(resident=8), 60)
    assert tune.qualified(row)
    assert not tune.qualified(dict(row, resident_confirmed=False))
    assert not tune.qualified(dict(row, status='execution-failed'))
    assert not tune.qualified(dict(row, repairs_accepted=1))


def test_final_selection_requires_two_qualified_measurements():
    baseline = measured(tune.knobs(), 100)
    fastest_once = measured(tune.knobs(reserve=512), 30)
    assert tune.choose([fastest_once], min_repeats=2) is None
    chosen = tune.choose([baseline, dict(baseline, request_seconds=90), fastest_once], min_repeats=2)
    assert chosen[0] == baseline
