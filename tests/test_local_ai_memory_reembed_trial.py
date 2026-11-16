import sqlite3
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_ai_memory_reembed_trial as trial


def index(path):
    con = sqlite3.connect(path)
    con.executescript('CREATE TABLE chunk(id INTEGER, hash TEXT, path TEXT, heading TEXT, text TEXT); CREATE TABLE emb(hash TEXT PRIMARY KEY, vec BLOB);')
    v = np.zeros(1024, dtype=np.float16); v[0] = 1
    con.executemany('INSERT INTO chunk VALUES(?,?,?,?,?)', [(1,'a','file1','h','a'),(2,'b','file2','h','bb')])
    con.executemany('INSERT INTO emb VALUES(?,?)', [('a',v.tobytes()),('b',v.tobytes())]); con.commit()
    return con


def test_sqlite_backup_keeps_original_and_refuses_replacement(tmp_path):
    original = tmp_path/'source.sqlite'; copy = tmp_path/'copy.sqlite'
    source = index(original); source.close()
    before = original.read_bytes()
    trial.fresh_backup(original, copy)
    target = sqlite3.connect(copy)
    target.execute('UPDATE emb SET vec=? WHERE hash=?', (b'changed', 'a')); target.commit(); target.close()
    assert original.read_bytes() == before
    with pytest.raises(ValueError, match='already exists'): trial.fresh_backup(original, copy)


def test_reembed_updates_copy_only_and_reports_cosines(tmp_path):
    con = index(tmp_path/'copy.sqlite')
    identity = trial.row_identity(con)
    class Fake:
        def embed(self, texts):
            assert all('file' in text for text in texts)
            a = np.zeros((len(texts),1024), dtype=np.float32)
            for i, text in enumerate(texts): a[i, 0 if 'file1' in text else 1] = 1
            return a
    progress = {}
    report = trial.reembed(con, Fake(), progress, batch=1, expected=2)
    assert progress['completed_documents'] == 2
    assert report['float16_bytes_identical'] == 1
    assert report['cosine_min'] == pytest.approx(0)
    assert trial.row_identity(con) == identity
    con.close()


def test_invalid_batch_fails_without_changing_current_batch(tmp_path):
    con = index(tmp_path/'copy.sqlite')
    before = con.execute('SELECT hash, vec FROM emb ORDER BY hash').fetchall()
    class Invalid:
        def embed(self, texts): return np.full((len(texts),1024), np.nan)
    progress = {}
    with pytest.raises(ValueError, match='finite'): trial.reembed(con, Invalid(), progress, expected=2)
    assert con.execute('SELECT hash, vec FROM emb ORDER BY hash').fetchall() == before
    assert not progress.get('completed_documents')
    con.close()


def test_partial_batch_failure_records_completed_without_full_result(tmp_path):
    con = index(tmp_path/'copy.sqlite')
    class Partial:
        n = 0
        def embed(self, texts):
            self.n += 1
            if self.n == 2: raise RuntimeError('server unavailable')
            vec = np.zeros((len(texts),1024),dtype=np.float32); vec[:,0] = 1
            return vec
    progress = {}
    with pytest.raises(RuntimeError, match='server unavailable'):
        trial.reembed(con, Partial(), progress, batch=1, expected=2)
    assert progress['completed_documents'] == 1
    con.close()


def test_interrupted_run_writes_unmeasured_receipt_without_comparison(tmp_path, monkeypatch):
    import contextlib
    import json
    from types import SimpleNamespace
    import dev_hot_paths
    source = tmp_path / 'source.sqlite'; con = index(source); con.close()
    bench = tmp_path / 'benchmark.json'; bench.write_text('{}')
    derived = tmp_path / 'derived.sqlite'
    evidence = tmp_path / 'out'
    model = tmp_path / 'model.gguf'; model.write_bytes(b'model')
    runtime = tmp_path / 'runtime'; runtime.mkdir()
    (runtime / 'llama-server.exe').write_bytes(b'runtime')
    monkeypatch.setattr(trial, 'COPY', derived)
    monkeypatch.setattr(trial, 'EVIDENCE', evidence)
    monkeypatch.setattr(trial.paired, 'INDEX', source)
    monkeypatch.setattr(trial.paired, 'BENCH', bench)
    monkeypatch.setattr(trial.paired, 'INDEX_SHA', trial.paired.sha(source))
    monkeypatch.setattr(trial.paired, 'BENCH_SHA', trial.paired.sha(bench))
    monkeypatch.setattr(trial.paired, 'preflight', lambda c, d: [])
    monkeypatch.setattr(dev_hot_paths, 'require_hot', lambda *a: None)
    monkeypatch.setattr(trial.local_llm, 'get_model', lambda name: SimpleNamespace(
        path=model, sha256=trial.paired.sha(model), name=name, ctx=8192, args=()))
    monkeypatch.setattr(trial.local_llm, 'llama_dir', lambda: runtime)
    @contextlib.contextmanager
    def serve(*args, **kwargs): yield 'http://127.0.0.1:1/v1'
    def broken(con, embedder, progress, **kwargs):
        progress['completed_documents'] = 1
        raise RuntimeError('simulated interruption')
    monkeypatch.setattr(trial, 'reembed', broken)
    result = trial.run(serving=serve)
    receipt = json.loads((evidence/'report.json').read_text())
    assert result['status'] == receipt['status'] == 'unmeasured'
    assert receipt['completed_documents'] == 1 and 'comparison' not in receipt
    assert trial.paired.sha(source) == trial.paired.INDEX_SHA
