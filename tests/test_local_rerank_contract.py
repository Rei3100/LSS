"""Malformed model rankings never become silently plausible search scores."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_llm


def test_valid_ranking_preserves_input_identity(monkeypatch):
    def reply(url, body, timeout):
        assert body['documents'] == ['same text', 'same text']
        assert timeout == 17
        return {'results': [{'index': 1, 'relevance_score': -2.5},
                            {'index': 0, 'relevance_score': 0}]}
    monkeypatch.setattr(local_llm, '_http_json', reply)
    assert local_llm.rerank('http://127.0.0.1:1/v1', 'query',
                            ['same text', 'same text'], timeout=17) == [0, -2.5]


@pytest.mark.parametrize('results', [
    [], [{'index': 0, 'relevance_score': 1}],
    [{'index': 0, 'relevance_score': 1}] * 2,
    [{'index': -1, 'relevance_score': 1}, {'index': 0, 'relevance_score': 2}],
    [{'index': 2, 'relevance_score': 1}, {'index': 0, 'relevance_score': 2}],
    [{'index': True, 'relevance_score': 1}, {'index': 0, 'relevance_score': 2}],
    [{'index': 1.0, 'relevance_score': 1}, {'index': 0, 'relevance_score': 2}],
    *[[{'index': 0, 'relevance_score': score}, {'index': 1, 'relevance_score': 2}]
      for score in (True, '1', None, float('nan'), float('inf'), -float('inf'))],
    [None, {'index': 1, 'relevance_score': 2}],
])
def test_invalid_ranking_is_rejected(monkeypatch, results):
    monkeypatch.setattr(local_llm, '_http_json', lambda *args: {'results': results})
    with pytest.raises(ValueError):
        local_llm.rerank('http://127.0.0.1:1/v1', 'query', ['a', 'b'])


@pytest.mark.parametrize('raw', [None, [], {}, {'results': {}}])
def test_invalid_envelope_is_rejected(monkeypatch, raw):
    monkeypatch.setattr(local_llm, '_http_json', lambda *args: raw)
    with pytest.raises(ValueError):
        local_llm.rerank('http://127.0.0.1:1/v1', 'query', ['a'])


def test_empty_input_needs_no_server(monkeypatch):
    def unexpected(*args):
        pytest.fail('Empty ranking must not launch an HTTP request')
    monkeypatch.setattr(local_llm, '_http_json', unexpected)
    assert local_llm.rerank('http://127.0.0.1:1/v1', 'query', []) == []
