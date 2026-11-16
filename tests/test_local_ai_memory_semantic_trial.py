import json
from pathlib import Path
import sqlite3
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_ai_memory_semantic_trial as trial


def test_frozen_index_preflight_reproduces_original_22():
    assert trial.sha(trial.INDEX) == trial.INDEX_SHA
    assert trial.sha(trial.BENCH) == trial.BENCH_SHA
    data = json.loads(trial.BENCH.read_text(encoding='utf-8-sig'))
    con = sqlite3.connect(trial.INDEX.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        baseline = trial.preflight(con, data)
        assert len(baseline) == 22
        assert sum(trial.support_hit(hits, c['support_source'], c['support_lines'])
                   for c, (_, hits) in zip(data['cases'], baseline)) == 18
    finally:
        con.close()


def test_missing_embedding_refused_before_query():
    con = sqlite3.connect(':memory:')
    con.executescript('CREATE TABLE chunk(hash TEXT); CREATE TABLE emb(hash TEXT, vec BLOB);')
    data = {'cases': [{'id': str(i)} for i in range(22)]}
    with pytest.raises(ValueError, match='incomplete'): trial.preflight(con, data)


def test_exact_control_uses_identical_order_without_embedding(monkeypatch):
    monkeypatch.setattr(trial, 'lexical', lambda c, q: ([4, 2, 9], [{'path': 'x', 'lines': [1, 4], 'commit': ''}]))
    class Forbidden:
        def embed_query(self, q): pytest.fail('exact control must bypass dense')
    for q in ['430dab3a6f4a3a8dcb34940b80fd1e1e5f5770da', 'docs/product/BGM_AND_AVAILABILITY.md']:
        ids, _, note = trial.candidate(None, q, Forbidden())
        assert ids == [4, 2, 9] and note['exact_control_bypass']
    assert not trial.exact_control('What is the Product policy?')


def test_missing_query_embedding_is_error_not_lexical_result(monkeypatch):
    monkeypatch.setattr(trial.devmem, 'search', lambda *a, **kw: ([1], {'dense': False}))
    with pytest.raises(RuntimeError, match='no silent fallback'):
        trial.candidate(None, 'wanted sound question', object())


def test_support_measures_source_and_span_overlap():
    hits = [{'path': 'a', 'lines': [3, 7]}, {'path': 'b', 'lines': [1, 10]}]
    assert trial.support_hit(hits, 'a', [7, 9])
    assert not trial.support_hit(hits, 'a', [8, 9])
    assert not trial.support_hit(hits, 'c', [3, 7])
