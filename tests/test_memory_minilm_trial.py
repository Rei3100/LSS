"""Ranking cannot invent chunks or alter exact identifier search."""
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_ai_memory_minilm_trial as trial


def test_equal_scores_keep_lexical_order():
    assert trial.ranked_ids([7, 5, 11], [0., 0., 0.]) == [7, 5, 11]


@pytest.mark.parametrize('scores', [[1.], [1., 2., 3.], [1., float('nan')],
                                    [float('inf'), 2.], [True, 1.], ['1', 2.]])
def test_corrupt_scores_do_not_become_search_results(scores):
    with pytest.raises(ValueError):
        trial.ranked_ids([7, 11], scores)


def test_exact_hash_control_never_calls_model(monkeypatch):
    monkeypatch.setattr(trial.frozen, 'lexical', lambda con, query: ([9, 1], ['exact hits']))
    def forbidden(*args):
        pytest.fail('Model must not process an exact identifier')
    assert trial.candidate(None, 'ab12cd34', forbidden) == (
        [9, 1], ['exact hits'], {'exact_control_bypass': True})


def test_ranker_receives_only_lexical_candidates_and_reorders_their_ids(monkeypatch):
    con = sqlite3.connect(':memory:')
    con.execute('CREATE TABLE chunk(id INTEGER, text TEXT)')
    con.executemany('INSERT INTO chunk VALUES (?, ?)', [(4, 'same'), (8, 'same'), (13, 'different')])
    def search(connection, query, **kwargs):
        assert kwargs == {'k': 30, 'mode': 'auto', 'embedder': None}
        return [8, 4, 13], {}
    monkeypatch.setattr(trial.devmem, 'search', search)
    monkeypatch.setattr(trial.devmem, 'hits', lambda connection, ids, *args: list(ids))
    def ranker(query, texts):
        assert query == 'find previous decision'
        assert texts == ['same', 'same', 'different']
        return [-3., 7., 2.]
    try:
        ids, hits, note = trial.candidate(con, 'find previous decision', ranker)
        assert ids == hits == [4, 13, 8]
        assert note['input_ids'] == [8, 4, 13]
    finally:
        con.close()
