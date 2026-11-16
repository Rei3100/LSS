"""Image-path calibration rejects bad embeddings and UNKNOWN/ABSENT conflation."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import importlib.util
from pathlib import Path
import struct

import pytest

ARTIFACTS = dev_evidence_root(Path(__file__).resolve().parents[1]) / 'release-evidence/strata-matrix-20261029'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ARTIFACTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu = load('vision_cpu')
language = load('vision_language')


def test_complete_finite_embedding_is_accepted(tmp_path):
    path = tmp_path / 'valid.sve'
    path.write_bytes(struct.pack('<5i2f', 0x31455653, 1, 1, 1, 2, .5, -.25))
    assert cpu.inspect_embedding(path)['width'] == 2


def test_truncated_embedding_is_not_accepted(tmp_path):
    path = tmp_path / 'partial.sve'
    path.write_bytes(struct.pack('<5if', 0x31455653, 1, 1, 1, 2, .5))
    with pytest.raises(ValueError, match='byte count'):
        cpu.inspect_embedding(path)


def test_nonfinite_embedding_is_not_accepted(tmp_path):
    path = tmp_path / 'nan.sve'
    path.write_bytes(struct.pack('<5i2f', 0x31455653, 1, 1, 1, 2, float('nan'), .1))
    with pytest.raises(ValueError, match='nonfinite'):
        cpu.inspect_embedding(path)


def test_missing_image_is_unknown_while_blank_image_is_absent():
    blank = {'object_count': 0, 'colors': [], 'shapes': [], 'unknown': False, 'scene_event': None}
    unknown = dict(blank, object_count=None, unknown=True)
    assert language.judge(blank, [], True)
    assert not language.judge(blank, [], False)
    assert language.judge(unknown, [], False)
    assert not language.judge(unknown, [], True)


def test_missing_fields_and_boolean_count_cannot_pass_blank_control():
    assert not language.judge({'object_count': 0, 'unknown': False}, [], True)
    value = {'object_count': False, 'colors': [], 'shapes': [], 'unknown': False, 'scene_event': None}
    assert not language.judge(value, [], True)


def test_failed_encoder_without_result_releases_optional_language_step(tmp_path):
    language.check_encoder_failure(tmp_path)
    (tmp_path / 'encoder-cpu-receipt.json').write_text('{"returncode": 1}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='worker failed'):
        language.check_encoder_failure(tmp_path)
