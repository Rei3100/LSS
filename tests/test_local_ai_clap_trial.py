import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_ai_clap_trial as trial


def test_import_does_not_load_inference_libraries():
    source = Path(trial.__file__).read_text(encoding='utf-8-sig')
    import ast
    top = ast.parse(source).body
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                   ('torch' in ast.unparse(n) or 'transformers' in ast.unparse(n)) for n in top)


def test_conversion_stereo_resampling_and_nonfinite():
    audio = np.array([[1, -1], [.5, .5], [0, 0], [0, 0]], dtype=np.float32)
    converted, meta = trial.convert_audio(audio, 24000, np, resample_poly)
    assert len(converted) == 8 and meta['up'] == 2 and meta['source_channels'] == 2
    assert np.isfinite(converted).all()
    with pytest.raises(ValueError): trial.convert_audio(audio[:, :1], 24000, np, resample_poly)
    audio[0, 0] = np.nan
    with pytest.raises(ValueError): trial.convert_audio(audio, 24000, np, resample_poly)


def test_fixture_guard_checks_hot_before_read_and_sha(tmp_path):
    fixture = tmp_path / 'fixture.wav'; fixture.write_bytes(b'hello')
    row = {'fixture_path': str(fixture), 'fixture_sha256': trial.sha256(fixture)}
    def refuse(*a): raise RuntimeError('cold path')
    with pytest.raises(RuntimeError, match='cold path'): trial.validate_fixtures([row], refuse)
    assert trial.validate_fixtures([row], lambda *a: None) == [fixture.resolve()]
    fixture.write_bytes(b'mutated')
    with pytest.raises(ValueError, match='SHA mismatch'): trial.validate_fixtures([row], lambda *a: None)


def test_ranking_uses_source_identity_and_ties_are_unresolved():
    ranking = trial.rank_scores(['c', 'a', 'b'], [0.2, 0.9, 0.5], 'a')
    assert ranking['top1_same_source'] is True and ranking['top_ids'] == ['a']
    assert [v['rank'] for v in ranking['cosines']] == [3, 1, 2]
    tie = trial.rank_scores(['c', 'a', 'b'], [0.9, 0.9, 0.5], 'a')
    assert tie['top1_same_source'] is None and tie['tie_unresolved']
    with pytest.raises(ValueError): trial.rank_scores(['a', 'a'], [1, 0], 'a')
    with pytest.raises(ValueError): trial.rank_scores(['a'], [float('nan')], 'a')


def test_model_provenance_fails_before_weight_read(tmp_path, monkeypatch):
    metadata = tmp_path / 'manifest.json'
    metadata.write_text(json.dumps({'repository': 'wrong', 'revision': trial.REVISION}))
    monkeypatch.setattr(trial, 'sha256', lambda *a: pytest.fail('weight read'))
    with pytest.raises(ValueError, match='provenance'): trial.verify_model(tmp_path, metadata)


def test_metadata_blob_and_weight_mutation_rejected(tmp_path, monkeypatch):
    import hashlib
    weight = tmp_path / 'pytorch_model.bin'; weight.write_bytes(b'weight')
    meta = tmp_path / 'config.json'; meta.write_bytes(b'config')
    blob = hashlib.sha1(b'blob 6\0config').hexdigest()
    monkeypatch.setattr(trial, 'WEIGHT_SIZE', 6)
    monkeypatch.setattr(trial, 'WEIGHT_SHA', hashlib.sha256(b'weight').hexdigest())
    monkeypatch.setattr(trial, 'METADATA', ('config.json',))
    receipt = tmp_path / 'hub.json'
    receipt.write_text(json.dumps({'repository':'laion/clap-htsat-unfused', 'revision':trial.REVISION,
        'files':[{'rfilename':'pytorch_model.bin','lfs':{'sha256':trial.WEIGHT_SHA}},
                 {'rfilename':'config.json','size':6,'blobId':blob}]}))
    assert trial.verify_model(tmp_path, receipt)['config.json']['git_blob_sha1'] == blob
    meta.write_bytes(b'Config')
    with pytest.raises(ValueError, match='metadata mismatch'): trial.verify_model(tmp_path, receipt)
    meta.write_bytes(b'config'); weight.write_bytes(b'Weight')
    with pytest.raises(ValueError, match='weight identity'): trial.verify_model(tmp_path, receipt)
