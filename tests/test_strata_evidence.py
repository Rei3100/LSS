try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import strata_evidence as evidence


def study(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, 'ROOT', tmp_path)
    monkeypatch.setattr(evidence, "dev_evidence_root", lambda checkout_root: checkout_root)
    source = tmp_path / '.cache/local-ai/strata-fixture'
    source.mkdir(parents=True)
    (source / 'result.json').write_text('{"status":"completed"}')
    (source / 'weights.gguf').write_bytes(b'not-evidence')
    return source, tmp_path / 'release-evidence/strata-fixture'


def test_export_excludes_weights_and_refuses_changed_existing_evidence(tmp_path, monkeypatch):
    source, target = study(tmp_path, monkeypatch)
    manifest = evidence.export(source, target)
    assert [row['path'] for row in manifest['files']] == ['result.json']
    assert not (target / 'weights.gguf').exists()
    assert evidence.export(source, target) == manifest
    (target / 'result.json').write_text('changed')
    with pytest.raises(ValueError, match='content changed'):
        evidence.export(source, target)
    assert (target / 'result.json').read_text() == 'changed'


def test_unfinished_and_escaped_destinations_cannot_be_certified(tmp_path, monkeypatch):
    source, target = study(tmp_path, monkeypatch)
    (source / 'result.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError, match='not completed'):
        evidence.export(source, target)
    with pytest.raises(ValueError, match='escapes'):
        evidence.export(source, tmp_path / 'other')


def test_protocol_gpu_completion_waits_for_successful_grading(tmp_path, monkeypatch):
    source, target = study(tmp_path, monkeypatch)
    renamed = source.with_name('strata-protocol-fixture')
    source.rename(renamed)
    (renamed / 'resources-summary.json').write_text('{"status":"completed"}')
    with pytest.raises(ValueError, match='grading is not completed'):
        evidence.export(renamed, target)
    (renamed / 'repair-summary.json').write_text('{"status":"completed"}')
    (renamed / 'score-cpu-receipt.json').write_text('{"returncode":0}')
    assert evidence.export(renamed, target)['terminal_receipt'] == 'resources-summary.json'


def test_actual_inline_catalog_failure_cannot_be_certified(tmp_path, monkeypatch):
    source, target = study(tmp_path, monkeypatch)
    actual = dev_evidence_root(Path(__file__).resolve().parents[1]) / 'release-evidence/strata-matrix-20261029/catalog-task/invalid-quiet-completion-r2.json'
    payload = json.loads(actual.read_text(encoding='utf-8'))
    assert payload['status'] == 'completed'
    assert payload['cpu_cap']['returncode'] == 3
    assert payload['reference']['counts']['collected'] == 0
    renamed = source.with_name('strata-catalog-quiet-reference-r2-20261030')
    source.rename(renamed)
    (renamed / 'result.json').write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='diagnostic completion'):
        evidence.export(renamed, target)
    # A separately written generic terminal receipt must not hide the inline failure.
    (renamed / 'resources-summary.json').write_text('{"status":"completed"}')
    with pytest.raises(ValueError, match='diagnostic completion'):
        evidence.export(renamed, target)


@pytest.mark.parametrize('collected,returncode,active,valid', [
    (18, 0, 0, True), (18, 1, 0, True), (0, 1, 0, False), (True, 0, 0, False),
    (18, 3, 0, False), (18, True, 0, False), (18, 1, None, False),
    (18, 1, [], False), (18, 1, False, False), (18, 1, 1, False),
])
def test_inline_reference_accounting(tmp_path, monkeypatch, collected, returncode, active, valid):
    source, target = study(tmp_path, monkeypatch)
    payload = {'status': 'completed', 'reference': {'counts': {'collected': collected}},
               'cpu_cap': {'returncode': returncode, 'active_descendants': active}}
    (source / 'result.json').write_text(json.dumps(payload))
    if valid:
        assert evidence.export(source, target)['terminal_receipt'] == 'result.json'
    else:
        with pytest.raises(ValueError, match='diagnostic completion'):
            evidence.export(source, target)
