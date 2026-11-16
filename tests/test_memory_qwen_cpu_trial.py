"""A broken or constant reranker never qualifies for the frozen population."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_ai_memory_qwen_cpu_trial as trial


@pytest.mark.parametrize('scores', [[], [[1., 0.]], [[1., 1.]] * 3,
                                  [[0., 1.]] * 3, [[1., 0.], [1., 0.], [0., 1.]]])
def test_incomplete_constant_reversed_or_japanese_failure_does_not_qualify(scores):
    assert not trial.calibration_passes(scores)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '1'])
def test_invalid_calibration_score_is_refused(bad):
    with pytest.raises(ValueError):
        trial.calibration_passes([[bad, 0.], [1., 0.], [1., 0.]])


def test_both_english_and_japanese_calibrations_are_required():
    assert trial.calibration_passes([[.7, -.2], [.3, .1], [.01, .009]])


def test_uncontrolled_worker_is_refused_before_runtime_access(monkeypatch):
    monkeypatch.delenv('LSS_QWEN_CPU_SUPERVISED', raising=False)
    monkeypatch.setattr(trial, 'verify_runtime', lambda: pytest.fail('Runtime touched'))
    with pytest.raises(RuntimeError, match='capped supervisor'):
        trial.worker()
