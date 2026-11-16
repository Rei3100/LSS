"""Unmeasured admission is distinct from quality failure and image memory fit."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import importlib.util
from pathlib import Path

ARTIFACT = dev_evidence_root(Path(__file__).resolve().parents[1]) / 'release-evidence/strata-matrix-20261029/hardware_confirm.py'
spec = importlib.util.spec_from_file_location('strata_hardware_confirm', ARTIFACT)
confirm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirm)


def row(settings, **changes):
    value = dict(label='fixture', knobs=settings, status='measured', request_seconds=40,
                 controls_passed=4, repairs_accepted=2, resident_confirmed=True,
                 minimum_ram_gb=5, minimum_commit_gb=7)
    return dict(value, **changes)


def test_admission_without_inference_does_not_erase_real_quality_failure():
    good = row(confirm.tune.knobs())
    refused = dict(good, status='admission-refused')
    failed = dict(good, controls_passed=3)
    rows = confirm.seed_rows({'candidates': [good, refused, failed]})
    assert len(rows) == 2
    assert confirm.tune.choose(rows) is None


def test_all_repeats_need_measured_encoder_memory_margin():
    text_only = row(confirm.tune.knobs(resident=12), minimum_ram_gb=2.7)
    safer = row(confirm.tune.knobs(resident=10), minimum_ram_gb=3.8)
    assert confirm.vision_groups([text_only, safer]) == [safer]
    assert confirm.vision_groups([safer, dict(safer, minimum_commit_gb=2)]) == []
