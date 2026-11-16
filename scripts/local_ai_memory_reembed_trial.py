"""Re-embed the immutable devmem snapshot into a fresh owned copy, then rerun its frozen evaluation.

Only the copy's emb table is modified. No source refresh, Product mutation, or hidden fallback.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import devmem
import local_llm
import local_ai_memory_semantic_trial as paired

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / '.cache/local-ai/memory-reembed-r1.sqlite'
EVIDENCE = dev_evidence_root(ROOT) / 'release-evidence/local-ai-memory-semantic-r1/reembed-r1'
BATCH = 16
COUNT = 4379
DIM = 1024
PURPOSE = 'frozen devmem document reembedding r1'


def row_identity(con):
    h = hashlib.sha256()
    for row_id, digest in con.execute('SELECT id, hash FROM chunk ORDER BY id'):
        h.update(f'{row_id}:{digest}\n'.encode('utf-8'))
    return h.hexdigest()


def fresh_backup(source, target):
    require_evidence_writable(target)
    if target.exists(): raise ValueError('derived index already exists')
    # Exclusive creation claims the identity before SQLite opens the destination.
    with target.open('xb'): pass
    source_conn = sqlite3.connect(source.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    target_conn = sqlite3.connect(str(target))
    try:
        source_conn.backup(target_conn)
        target_conn.commit()
    finally:
        source_conn.close(); target_conn.close()


def reembed(con, embedder, progress, *, batch=BATCH, expected=COUNT, on_progress=None):
    if not 1 <= batch <= 16: raise ValueError('batch must be 1..16')
    rows = con.execute('SELECT c.hash, c.path, c.heading, c.text, e.vec FROM chunk c '
                       'JOIN emb e ON c.hash=e.hash').fetchall()
    if len(rows) != expected or len({r[0] for r in rows}) != expected:
        raise ValueError('embedding population mismatch')
    rows.sort(key=lambda row: len(row[3]))  # Same ordering as devmem.embed_missing.
    cosines = []
    identical = 0
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        vectors = np.asarray(embedder.embed([devmem.embed_text(p, heading, text) for _, p, heading, text, _ in part]), dtype=np.float32)
        if vectors.shape != (len(part), DIM) or not np.isfinite(vectors).all():
            raise ValueError('embedding shape or finite check failed')
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms < 1e-9):
            raise ValueError('zero or nonfinite embedding norm')
        updates, batch_cosines, batch_identical = [], [], 0
        for row, vec in zip(part, vectors):
            old = np.frombuffer(row[4], dtype=np.float16).astype(np.float32)
            new_bytes = vec.astype(np.float16).tobytes()
            new = np.frombuffer(new_bytes, dtype=np.float16).astype(np.float32)
            if old.shape != (DIM,) or not np.isfinite(old).all() or not np.isfinite(new).all():
                raise ValueError('invalid stored vector')
            denominator = np.linalg.norm(old) * np.linalg.norm(new)
            if not np.isfinite(denominator) or denominator < 1e-9: raise ValueError('invalid cosine norm')
            batch_cosines.append(float(np.dot(old, new) / denominator))
            batch_identical += int(row[4] == new_bytes)
            updates.append((new_bytes, row[0]))
        with con:
            con.executemany('UPDATE emb SET vec=? WHERE hash=?', updates)
        cosines.extend(batch_cosines)
        identical += batch_identical
        progress['completed_documents'] = len(cosines)
        if on_progress is not None: on_progress()
    return {'count': len(cosines), 'float16_bytes_identical': identical,
            'float16_bytes_identical_fraction': identical / len(cosines),
            'cosine_min': float(np.min(cosines)), 'cosine_mean': float(np.mean(cosines)),
            'cosine_median': float(np.median(cosines)), 'cosine_p01': float(np.quantile(cosines, .01)),
            'cosines_by_hash': [{'hash': row[0], 'cosine': cosine} for row, cosine in zip(rows, cosines)]}


def run(*, serving=local_llm.serving):
    from dev_hot_paths import require_hot
    if COPY.exists() or EVIDENCE.exists(): raise ValueError('reembed-r1 output already exists')
    model = local_llm.get_model(paired.MODEL)
    for label, path in [('original index', paired.INDEX), ('benchmark', paired.BENCH), ('copy', COPY),
                        ('evidence', EVIDENCE), ('model', model.path), ('runtime', local_llm.llama_dir()),
                        ('environment', sys.prefix)]: require_hot(path, label)
    require_evidence_writable(EVIDENCE)
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    report = {'status': 'unmeasured', 'stage': 'preflight', 'completed_documents': 0,
              'scope': 'same frozen index and questions; replace document embeddings in an owned copy only'}
    started = time.monotonic()
    def checkpoint():
        temp = EVIDENCE / 'report.partial'
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temp, EVIDENCE / 'report.json')
    checkpoint()
    con = None
    try:
        if paired.sha(paired.INDEX) != paired.INDEX_SHA or paired.sha(paired.BENCH) != paired.BENCH_SHA:
            raise ValueError('frozen source identity mismatch')
        doc = json.loads(paired.BENCH.read_text(encoding='utf-8-sig'))
        source = sqlite3.connect(paired.INDEX.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            baseline = paired.preflight(source, doc)
            ids_before = row_identity(source)
        finally:
            source.close()
        model_sha = paired.sha(model.path)
        if not model.sha256 or model_sha.lower() != model.sha256.lower(): raise ValueError('model SHA mismatch')
        runtime = local_llm.llama_dir()
        report['identity'] = {'source_sha256': paired.INDEX_SHA, 'benchmark_sha256': paired.BENCH_SHA,
            'model_sha256': model_sha, 'model': model.name, 'model_ctx': model.ctx, 'model_args': list(model.args),
            'runtime_build': local_llm.BUILD, 'runtime_sha256': paired.sha(runtime / 'llama-server.exe'),
            'runtime_dlls_sha256': {p.name: paired.sha(p) for p in sorted(runtime.glob('*.dll'))},
            'devmem_sha256': paired.sha(Path(devmem.__file__)), 'evaluator_sha256': paired.sha(Path(paired.__file__)),
            'trial_sha256': paired.sha(Path(__file__)), 'document_template': 'devmem.embed_text',
            'query_instruction': devmem.QUERY_INSTRUCT, 'chunk_identity_sha256': ids_before,
            'lifecycle_receipts': str(local_llm.runs_file()), 'lifecycle_purpose': PURPOSE}
        report['stage'] = 'backup'
        fresh_backup(paired.INDEX, COPY)
        con = sqlite3.connect(str(COPY))
        if row_identity(con) != ids_before: raise ValueError('copied chunk identity differs')
        # Recheck fixed lexical baseline on the copy prior to model launch.
        paired.preflight(con, doc)
        report['stage'] = 'reembedding'
        checkpoint()
        with serving(paired.MODEL, purpose=PURPOSE) as base:
            embedder = devmem.Embedder(base)
            report['embedding_comparison'] = reembed(con, embedder, report, on_progress=checkpoint)
            if report['completed_documents'] != COUNT: raise RuntimeError('incomplete reembedding')
            if row_identity(con) != ids_before: raise RuntimeError('chunk identity changed')
            devmem._MATRIX.clear()
            report['stage'] = 'paired_queries'
            result = paired.evaluate(con, doc, baseline, embedder)
            if len(result['cases']) != 22 or any(not x['same_ordered_chunk_ids'] for x in result['exact_controls']):
                raise RuntimeError('comparison or exact controls incomplete')
            report['comparison'] = result
        if paired.sha(paired.INDEX) != paired.INDEX_SHA: raise RuntimeError('original index changed')
        report['stage'] = 'complete'; report['status'] = 'complete'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report.pop('comparison', None)
    finally:
        if con is not None: con.close()
        try:
            if COPY.exists(): report['derived_index_sha256'] = paired.sha(COPY)
            report['source_sha256_after'] = paired.sha(paired.INDEX)
            if report['source_sha256_after'] != paired.INDEX_SHA:
                raise ValueError('original index changed')
        except Exception as exc:
            report['status'] = 'unmeasured'; report['error'] = f'index verification: {type(exc).__name__}: {exc}'
            report.pop('comparison', None)
        report['wall_s'] = round(time.monotonic() - started, 3)
        checkpoint()
    return report


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = run()
    print(json.dumps({'status': report['status'], 'stage': report['stage'],
                      'completed_documents': report['completed_documents']}))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__': raise SystemExit(main())
