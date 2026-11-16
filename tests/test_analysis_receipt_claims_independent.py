"""Unknown receipt-shaped neighbors do not confer current-job retirement authority."""
import hashlib
import json

import pytest

from lss import application


def test_foreign_analysis_receipt_does_not_authorize_removing_its_files(tmp_path):
    scratch = tmp_path / 'analysis'
    folder = scratch / 'foreign-source'
    folder.mkdir(parents=True)
    own = scratch / 'analysis.wav'
    own.write_bytes(b'Current source analysis')
    (scratch / 'analysis-preparation.json').write_text(json.dumps({
        'source_sha256': '0' * 64, 'sha256': hashlib.sha256(own.read_bytes()).hexdigest()}), encoding='utf-8')
    audio = folder / 'user-retained.wav'
    audio.write_bytes(b'Foreign-source retained audio')
    receipt = folder / 'receipt.json'
    original = {'type': 'bgm_sfc_analysis_receipt', 'schema_version': 1,
                'input': {'sha256': 'f' * 64},
                'stems': {'music': {'path': str(audio), 'sha256': hashlib.sha256(audio.read_bytes()).hexdigest()}}}
    receipt.write_text(json.dumps(original), encoding='utf-8')
    expected = {p: p.read_bytes() for p in (audio, receipt)}
    application._retire_analysis_scratch(scratch, '0' * 64, {})
    assert all(p.is_file() and p.read_bytes() == content for p, content in expected.items())


@pytest.mark.parametrize('bad_stems', [[], 'unknown legacy shape'])
def test_unknown_receipt_shape_does_not_break_retirement(tmp_path, bad_stems):
    scratch = tmp_path / 'analysis'
    folder = scratch / 'unknown'
    folder.mkdir(parents=True)
    receipt = folder / 'receipt.json'
    receipt.write_text(json.dumps({'type': 'bgm_sfc_analysis_receipt', 'stems': bad_stems}), encoding='utf-8')
    original = receipt.read_bytes()
    application._retire_analysis_scratch(scratch, '0' * 64, {})
    assert receipt.is_file() and receipt.read_bytes() == original
