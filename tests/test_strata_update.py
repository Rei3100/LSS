"""Pin/rollback logic of scripts/strata_update.py with fake operations (no network, GPU or Strata process)."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import strata_pin  # noqa: E402
import strata_update as up  # noqa: E402

SHA = 'ab' * 32


def report(offered='0.1.54', installed='0.1.52', status='available', trusted='built-in keys', sequence=117, ok=True):
    return {'ok': ok, 'error': {'message': 'boom'}, 'feed': {'trusted_by': trusted, 'sequence': sequence},
            'engine': {'status': status, 'installed': {'version': installed, 'signed': True},
                       'offered': {'version': offered, 'url': f'strata-engine-{offered}-windows-x86_64-cuda-cuda13.tar.gz',
                                   'sha256': SHA, 'size': 5}},
            'control': {'installed': '0.1.53', 'offered': '0.1.54'}}


def default_pin():
    return strata_pin.load(Path('does-not-exist.json'))


def test_default_pin_matches_first_install_and_validates():
    pin = default_pin()
    assert pin['engine'] == '0.1.52' and pin['release_tag'] == 'v0.2.7'
    strata_pin.validate(pin)


def test_advance_and_revert_round_trip():
    pin = default_pin()
    new = strata_pin.advance(pin, {**{k: pin[k] for k in strata_pin.REQUIRED}, 'engine': '0.1.54', 'release_tag': 'v0.2.8'})
    assert new['previous']['engine'] == '0.1.52' and new['engine'] == '0.1.54'
    back = strata_pin.revert(new)
    assert back['engine'] == '0.1.52' and back['previous'] is None and back['rejected'] == ['0.1.54']
    with pytest.raises(ValueError):
        strata_pin.revert(back)


def test_save_is_atomic_and_rejects_bad_pins(tmp_path):
    path = tmp_path / 'pin.json'
    pin = default_pin()
    strata_pin.save(pin, path)
    assert strata_pin.load(path) == pin
    bad = copy.deepcopy(pin)
    bad['engine_sha256'] = 'zz'
    with pytest.raises(ValueError):
        strata_pin.save(bad, path)


@pytest.mark.parametrize('kwargs,action', [
    ({}, 'update'),
    ({'offered': '0.1.52'}, 'current'),
    ({'status': 'up-to-date'}, 'current'),
    ({'trusted': 'unsigned'}, 'refuse'),
    ({'ok': False}, 'refuse'),
    ({'offered': '0.2.0'}, 'refuse'),
    ({'sequence': 100}, 'refuse'),
    ({'installed': '0.1.50'}, 'refuse'),
])
def test_decide(kwargs, action):
    pin = {**default_pin(), 'feed_sequence': 116}
    assert up.decide(report(**kwargs), pin)['action'] == action


def test_decide_minor_step_and_rejected_versions_need_flags():
    pin = default_pin()
    assert up.decide(report(offered='0.2.0'), pin, allow_minor=True)['action'] == 'update'
    pin['rejected'] = ['0.1.54']
    assert up.decide(report(), pin)['action'] == 'refuse'
    assert up.decide(report(), pin, force=True)['action'] == 'update'


def test_regression_rules():
    base = {'ids_run': ['a', 'b', 'c', 'd'], 'passed_ids': ['a', 'b', 'c'], 'decode_tok_s': 40}
    good = {'smoke_passed': True, 'ids_run': ['a', 'b', 'c', 'd'], 'passed_ids': ['a', 'b', 'd'], 'decode_tok_s': 42}
    assert up.regression(base, good) is None  # one loss is inside the tolerance
    assert 'regression' in up.regression(base, {**good, 'passed_ids': ['a']})
    assert 'smoke' in up.regression(base, {**good, 'smoke_passed': False})
    assert 'speed' in up.regression(base, {**good, 'decode_tok_s': 20})
    assert up.regression(None, good) is None
    # tasks not run now (smaller context) are not counted as losses
    assert up.regression(base, {**good, 'ids_run': ['a'], 'passed_ids': ['a']}) is None


class FakeOps:
    def __init__(self, gate=None, held=False, update_ok=True, archive=None):
        self.calls, self.held, self.update_ok = [], held, update_ok
        self._gate = gate or {'smoke_passed': True, 'ids_run': ['a', 'b'], 'passed_ids': ['a', 'b'], 'decode_tok_s': 40}
        self.archive = archive

    def latest_tag(self):
        return 'v0.2.8'

    def check(self, tag):
        self.calls.append('check')
        return report()

    def lease_held(self):
        return self.held

    def server_running(self):
        return False

    def run_update(self, tag, out):
        self.calls.append('update')
        return {'ok': self.update_ok, 'control': {'installed': '0.1.54'}, 'error': {'message': 'no'}}

    def archive_sha(self, name):
        return self.archive

    def gate(self, label, out):
        self.calls.append('gate')
        return self._gate

    def rollback(self):
        self.calls.append('rollback')
        return {'ok': True}


def run(tmp_path, ops, **kwargs):
    saved = []
    pin = {**default_pin(), 'bench_baseline': {'ids_run': ['a', 'b'], 'passed_ids': ['a', 'b'], 'decode_tok_s': 40}}
    code, receipt = up.do_apply(ops, pin, directory=tmp_path, save=saved.append, **kwargs)
    return code, receipt, saved, pin


def test_apply_success_moves_pin_one_step_and_keeps_previous(tmp_path):
    code, receipt, saved, pin = run(tmp_path, FakeOps())
    assert code == 0 and receipt['status'] == 'applied'
    final = saved[-1]
    assert final['engine'] == '0.1.54' and final['control'] == '0.1.54' and final['release_tag'] == 'v0.2.8'
    assert final['previous']['engine'] == '0.1.52' and final['bench_baseline']['ids_run'] == ['a', 'b']
    assert json.loads(next(tmp_path.glob('*-apply.json')).read_text())['status'] == 'applied'


def test_apply_reverts_on_quality_regression(tmp_path):
    ops = FakeOps(gate={'smoke_passed': True, 'ids_run': ['a', 'b'], 'passed_ids': [], 'decode_tok_s': 40})
    code, receipt, saved, pin = run(tmp_path, ops)
    assert code == 1 and receipt['status'] == 'reverted' and 'rollback' in ops.calls
    assert saved[-1]['engine'] == '0.1.52' and saved[-1]['rejected'] == ['0.1.54']


def test_apply_reverts_when_smoke_fails(tmp_path):
    code, receipt, saved, _ = run(tmp_path, FakeOps(gate={'smoke_passed': False}))
    assert code == 1 and receipt['reason'] == 'smoke failed' and saved[-1]['engine'] == '0.1.52'


def test_apply_skips_when_gpu_lease_is_held_and_changes_nothing(tmp_path):
    ops = FakeOps(held=True)
    code, receipt, saved, _ = run(tmp_path, ops)
    assert code == 3 and receipt['status'] == 'skipped' and not saved and 'update' not in ops.calls


def test_apply_failed_update_keeps_pin(tmp_path):
    code, receipt, saved, _ = run(tmp_path, FakeOps(update_ok=False))
    assert code == 1 and receipt['status'] == 'failed-update' and not saved


def test_apply_reverts_on_archive_hash_mismatch(tmp_path):
    ops = FakeOps(archive='cd' * 32)
    code, receipt, saved, _ = run(tmp_path, ops)
    assert code == 1 and 'hash' in receipt['reason'] and 'rollback' in ops.calls and 'gate' not in ops.calls


def test_apply_is_idempotent_when_current(tmp_path):
    ops = FakeOps()
    ops.check = lambda tag: report(offered='0.1.52')
    code, receipt, saved, _ = run(tmp_path, ops)
    assert code == 0 and receipt['status'] == 'current' and 'update' not in ops.calls and not saved


def test_apply_error_after_update_restores_previous_pin(tmp_path):
    ops = FakeOps()
    ops.gate = lambda label, out: (_ for _ in ()).throw(RuntimeError('crash'))
    code, receipt, saved, pin = run(tmp_path, ops)
    assert code == 1 and receipt['status'] == 'error' and receipt['reverted'] and saved[-1] == pin


def test_dry_run_changes_nothing(tmp_path):
    ops = FakeOps()
    code, receipt, saved, _ = run(tmp_path, ops, dry_run=True)
    assert code == 0 and receipt['status'] == 'dry-run' and ops.calls == ['check']


def test_gate_without_bench_result_reverts_and_empty_baseline_is_no_baseline():
    import strata_update
    assert 'no result' in strata_update.regression(None, {'smoke_passed': True, 'exit': 120})
    empty = {'label': None, 'ids_run': None, 'passed_ids': None, 'decode_tok_s': None}
    assert strata_update.regression(empty, {'smoke_passed': True, 'ids_run': ['a'], 'passed_ids': ['a']}) is None
