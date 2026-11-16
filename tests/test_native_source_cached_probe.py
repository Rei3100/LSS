"""Source-only cached trial rejects acquisition, bad receipts and GPU pauses."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import probe_native_source_cached as probe
import dev_quiet_mode


def valid():
    return dict(status='verified', source_commit=probe.COMMIT, ready=True,
                existing_fixture_files_unchanged=True, manifest_sha256='exact')


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('source_commit', 'other'),
    ('ready', False), ('existing_fixture_files_unchanged', False), ('manifest_sha256', 'changed')])
def test_cached_receipt_refuses_failure_or_changed_identity(field, value):
    record = valid()
    record[field] = value
    with pytest.raises(ValueError):
        probe.validate_receipt(record, {'status': 'passed'}, 'exact')


def test_cached_receipt_requires_supervisor():
    with pytest.raises(ValueError):
        probe.validate_receipt(valid(), {}, 'exact')
    probe.validate_receipt(valid(), {'status': 'passed'}, 'exact')


def test_acquisition_fails_before_any_side_effect():
    with pytest.raises(RuntimeError, match='Acquisition is disabled'):
        probe.deny_acquisition('https://example.invalid', Path('must-not-exist'))


def test_user_gpu_pause_precedes_source_import(monkeypatch):
    def pause():
        raise dev_quiet_mode.GPUDeferred('explicit user pause')
    monkeypatch.setattr(dev_quiet_mode, 'require_gpu_allowed', pause)
    with pytest.raises(dev_quiet_mode.GPUDeferred):
        probe.worker()
