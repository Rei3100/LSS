"""Optional model search cannot alter default/exact paths or invent candidates."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import devmem_rerank as route


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    from dataclasses import replace
    model = route.local_llm.load_registry()[route.MODEL]
    monkeypatch.setattr(route.local_llm, 'get_model', lambda name: replace(model, path=tmp_path / 'model.gguf'))
    monkeypatch.setattr(route, 'require_hot', lambda *args: None)
    cfg = route.devmem.Config(repo=tmp_path, db_dir=tmp_path / 'index')
    con = route.devmem.connect(cfg)
    route.devmem._add_source(con, 'a', 'repo', 0, 0,
        [(1, 2, 'first', 'recovery algorithm evidence alpha')], 'docs/a.md', 'today')
    route.devmem._add_source(con, 'b', 'repo', 0, 0,
        [(3, 4, 'second', 'recovery algorithm evidence beta')], 'docs/b.md', 'today')
    con.commit()
    con.close()
    return cfg


def args(text='recovery algorithm'):
    return SimpleNamespace(text=text, k=2, refresh=False, json=True)


@pytest.mark.parametrize('query', ['abcdef12345', 'f' * 64, 'docs/a.md', 'zzzznomatchzzzz'])
def test_exact_or_empty_search_never_touches_gpu(cfg, monkeypatch, capsys, query):
    monkeypatch.setattr(route, 'require_gpu_allowed', lambda: pytest.fail('GPU checked for bypass'))
    monkeypatch.setattr(route.batch, 'run_command', lambda *a, **k: pytest.fail('Model launched for bypass'))
    before = route.local_llm._sha256(cfg.db)
    assert route.run(args(query), cfg) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['gpu_used'] is False and result['reranked'] is False
    assert route.local_llm._sha256(cfg.db) == before


@pytest.mark.parametrize('scores', [[1.], [1., 2., 3.], [float('nan'), 0.],
                                   [float('inf'), 0.], [True, 0.], ['1', 0.]])
def test_missing_extra_or_invalid_scores_never_change_search(scores):
    with pytest.raises(ValueError):
        route.ordered_ids([8, 4], scores)


def test_stable_ties_and_duplicate_candidate_refusal():
    assert route.ordered_ids([8, 4], [0., 0.]) == [8, 4]
    with pytest.raises(ValueError):
        route.ordered_ids([8, 8], [1., 0.])


def test_explicit_route_preserves_snapshot_locators_and_receipts(cfg, tmp_path, monkeypatch, capsys):
    original = route.snapshot(cfg, 'recovery algorithm', 2)
    before = route.local_llm._sha256(cfg.db)
    monkeypatch.setattr(route, 'CACHE', tmp_path / 'receipts')
    monkeypatch.setattr(route, 'require_gpu_allowed', lambda: None)
    def backend(command, cwd, log, duty, **kwargs):
        assert duty == .1
        request = Path(command[-1])
        data = json.loads(request.read_text())
        assert data['documents'] == original['documents']
        assert data['hits'] == original['hits']
        scores = list(range(len(data['ids'])))
        result = {'status': 'complete', 'ids': list(reversed(data['ids'])), 'scores': scores,
                  'request_sha256': route.local_llm._sha256(request)}
        (request.parent / 'result.json').write_text(json.dumps(result))
        return 0
    monkeypatch.setattr(route.batch, 'run_command', backend)
    assert route.run(args(), cfg) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['hits'] == list(reversed(original['hits']))
    assert result['reranked'] is True
    assert Path(result['receipt'], 'dispatch-result.json').is_file()
    assert route.local_llm._sha256(cfg.db) == before


def test_failed_worker_cannot_be_reported_as_success_or_silent_fallback(cfg, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(route, 'CACHE', tmp_path / 'receipts')
    monkeypatch.setattr(route, 'require_gpu_allowed', lambda: None)
    monkeypatch.setattr(route.batch, 'run_command', lambda *a, **k: 1)
    assert route.run(args(), cfg) == 3
    captured = capsys.readouterr()
    assert captured.out == '' and 'receipt:' in captured.err


def test_refresh_and_oversized_pool_are_refused_before_snapshot(cfg, monkeypatch):
    monkeypatch.setattr(route, 'snapshot', lambda *a: pytest.fail('Snapshot accessed'))
    request = args()
    request.refresh = True
    with pytest.raises(ValueError):
        route.run(request, cfg)
    request.refresh, request.k = False, 31
    with pytest.raises(ValueError):
        route.run(request, cfg)
