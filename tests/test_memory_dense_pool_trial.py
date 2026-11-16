import math
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_ai_memory_dense_pool_trial as trial


def test_stable_union_keeps_lexical_order_and_dense_novelty():
    assert trial.stable_union([3, 1, 3, 2], [2, 5, 1, 4, 5]) == [3, 1, 2, 5, 4]


@pytest.mark.parametrize('query', ['abcdef123456', 'f' * 64, 'docs/engine/WANTED_SOUND.md'])
def test_exact_control_never_embeds_or_searches_dense(monkeypatch, query):
    monkeypatch.setattr(trial.frozen, 'lexical', lambda *args, **kwargs: ([7, 8], []))
    monkeypatch.setattr(trial.devmem, 'dense_ids', Mock(side_effect=AssertionError('dense called')))
    embedder = Mock()
    embedder.embed_query.side_effect = AssertionError('model called')
    result = trial.pool(None, query, embedder)
    assert result['candidate_ids'] == [7, 8]
    assert result['exact_control_bypass'] is True
    embedder.embed_query.assert_not_called()
    trial.devmem.dense_ids.assert_not_called()


@pytest.mark.parametrize('values', [
    [1.0] * 1023,
    [0.0] * 1024,
    [True] + [1.0] * 1023,
    [math.nan] + [1.0] * 1023,
    [math.inf] + [1.0] * 1023,
    ['1'] + [1.0] * 1023,
])
def test_invalid_query_vectors_fail(values):
    with pytest.raises(ValueError):
        trial.validate_vector(values)


def test_valid_query_vector_is_normalized():
    vector = trial.validate_vector([2.0] * 1024)
    assert vector.shape == (1024,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_query_uses_shared_private_transport_and_pinned_instruction(monkeypatch):
    def response(url, body, *, timeout):
        assert url == 'http://127.0.0.1:12345/v1/embeddings'
        assert body['input'] == ['Instruct: ' + trial.devmem.QUERY_INSTRUCT + '\nQuery:history']
        assert timeout == 120
        return {'data': [{'index': 0, 'embedding': [2.] * 1024}]}
    monkeypatch.setattr(trial.local_llm, '_http_json', response)
    vector = trial.CheckedEmbedder('http://127.0.0.1:12345/v1').embed_query('history')
    assert np.linalg.norm(vector) == pytest.approx(1.)
