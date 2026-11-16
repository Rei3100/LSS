import importlib.util
import json
from pathlib import Path

import pytest
import numpy as np

from lss.validation.instrumental_quality import instrumental_quality
from lss.validation.stereo_quality import stereo_quality


@pytest.fixture
def repeat_script():
    path = Path(__file__).resolve().parents[1] / "scripts/probe_instrumental_repeat.py"
    spec = importlib.util.spec_from_file_location("instrumental_repeat_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_added_packages_remain_an_explicit_runtime_confound(repeat_script):
    before = {"runtime_id": "old", "torch_version": "fixed", "adapter_artifacts": [],
              "dependency_versions": [{"name": "torch", "version": "fixed"}]}
    after = {**before, "runtime_id": "new", "dependency_versions": before["dependency_versions"] + [
        {"name": "yt-dlp", "version": "extra"},
    ]}
    transition = repeat_script.runtime_transition(before, after)
    assert transition["identity_equal"] is False
    assert transition["equivalence"] == "not_asserted"
    assert transition["added_dependencies"] == [{"name": "yt-dlp", "version": "extra"}]


@pytest.mark.parametrize("change", ["version", "adapter", "missing"])
def test_renderer_or_existing_dependency_change_stops_comparison(repeat_script, change):
    before = {"runtime_id": "old", "adapter_artifacts": [],
              "dependency_versions": [{"name": "torch", "version": "fixed"}]}
    after = {**before, "runtime_id": "new"}
    if change == "adapter":
        after["adapter_artifacts"] = [{"path": "infer.py", "sha256": "changed"}]
    else:
        after["dependency_versions"] = [] if change == "missing" else [{"name": "torch", "version": "changed"}]
    with pytest.raises(ValueError, match="変更"):
        repeat_script.runtime_transition(before, after)


def test_absence_regression_is_not_hidden_by_active_scores(repeat_script):
    before = {"target_present": False, "target_absent_output_to_vocals_db": -50}
    after = {"target_present": False, "target_absent_output_to_vocals_db": -40}
    assert repeat_script.comparison(before, after)["regression"] is True


def test_better_snr_does_not_hide_extra_gain_damage(repeat_script):
    def quality(snr, gain):
        fidelity = stereo_quality(np.ones((2, 100)), np.ones((2, 100)))
        fidelity["scale_dependent_snr_db"] = snr
        fidelity["projection_gain_db"]["joint"] = gain
        return {"target_present": True, "fidelity": fidelity}
    assert repeat_script.comparison(quality(10, -0.1), quality(20, -1))["regression"] is True


def test_global_gain_and_snr_cannot_hide_continuous_channel_loss(repeat_script):
    time = np.arange(4096) / 4096
    target = np.array([np.sin(2 * np.pi * 41 * time), 0.1 * np.sin(2 * np.pi * 71 * time)])
    before = target.copy()
    before[0] += 0.6 * np.sin(2 * np.pi * 103 * time)
    after = target.copy()
    after[1] *= 0.01
    unwanted = np.zeros_like(target)
    result = repeat_script.comparison(instrumental_quality(before, target, unwanted),
                                     instrumental_quality(after, target, unwanted))
    assert result["uncorrected_snr_delta_db"] > 0
    assert result["absolute_gain_error_delta_db"] < 0.1
    assert result["regression"] is True
    assert result["spatial_regressions"]


def test_real_gain_regression_can_be_saved_to_json(repeat_script):
    target = np.random.default_rng(15).normal(0, 0.1, (2, 257))
    before = instrumental_quality(target * 0.99, target, np.zeros_like(target))
    after = instrumental_quality(target * 0.7, target, np.zeros_like(target))
    result = repeat_script.comparison(before, after)
    assert result["regression"] is True
    assert json.loads(json.dumps({"quality": after, "comparison": result}))["comparison"]["regression"] is True
