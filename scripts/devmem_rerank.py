"""Explicit, receipted Qwen reranking of an existing SSD developer-memory index.

The default memory command stays model-free. Exact identifiers and empty pools
bypass the model. Other queries use the measured lexical-auto top30 pool, full
documents, context8192 and 10% tree duty. No index refresh or silent fallback.
"""
from dataclasses import replace
from contextlib import closing
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
from uuid import uuid4

import batch
from dev_hot_paths import require_hot
from dev_quiet_mode import require_gpu_allowed
import devmem
import local_llm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/local-ai/memory-rerank'
MODEL = 'qwen3-reranker-0.6b'
MODEL_SHA = '22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48'


def exact_identifier(text):
    return bool(re.fullmatch(r'[0-9a-fA-F]{7,64}', text)
                or re.fullmatch(r'[^\s/\\]+(?:[/\\][^\s/\\]+)+', text))


def ordered_ids(ids, scores):
    if (len(ids) != len(scores) or len(set(ids)) != len(ids)
            or any(type(s) not in (int, float) or not math.isfinite(s) for s in scores)):
        raise ValueError('Reranking must preserve candidates with finite scores')
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])]


def snapshot(cfg, text, k):
    require_hot(cfg.db, 'developer memory index')
    with closing(sqlite3.connect(cfg.db.resolve().as_uri() + '?mode=ro', uri=True)) as con:
        con.execute('BEGIN')
        bypass = exact_identifier(text)
        ids, _ = devmem.search(con, text, k=k if bypass else 30,
                               mode='bm25' if bypass else 'auto', embedder=None)
        ids = ids[:k] if bypass else ids[:30]
        return {'query': text, 'k': k, 'ids': ids, 'bypass': bypass or not ids,
                'documents': [] if bypass else [con.execute('SELECT text FROM chunk WHERE id=?', (cid,)).fetchone()[0] for cid in ids],
                'hits': devmem.hits(con, ids, text, len(ids), cfg)}


def worker(request):
    request = request.resolve()
    require_hot(request, 'rerank request')
    if (request.name != 'request.json' or not request.is_relative_to(CACHE.resolve())
            or os.environ.get('LSS_DEVMEM_RERANK_SUPERVISED') != str(request.parent)):
        raise ValueError('Use the bounded developer-memory command')
    require_gpu_allowed()
    data = json.loads(request.read_text(encoding='utf-8'))
    model = local_llm.get_model(MODEL)
    require_hot(model.path, 'reranker model')
    if not 0 < len(data['ids']) <= 30 or len(data['documents']) != len(data['ids']):
        raise ValueError('Invalid candidate population')
    if local_llm._sha256(model.path) != MODEL_SHA or model.sha256 != MODEL_SHA:
        raise ValueError('Reranker model identity mismatch')
    experiment = replace(model, args=('-b', '512', '-ub', '512', '-t', '1', '-tb', '1'), ram_gb=2.)
    original_get, original_append = local_llm.get_model, local_llm._append_row
    local_llm.get_model = lambda name: experiment if name == MODEL else original_get(name)
    record = {'status': 'running', 'model': MODEL, 'model_sha256': MODEL_SHA,
              'request_sha256': local_llm._sha256(request), 'duty': .1,
              'runtime_server_sha256': local_llm._sha256(local_llm.llama_dir() / 'llama-server.exe')}
    def telemetry(row):
        record['serving'] = dict(row)
        original_append(row)
    local_llm._append_row = telemetry
    try:
        with local_llm.serving(MODEL, ctx=8192, purpose='explicit developer memory rerank') as base:
            scores = [local_llm.rerank(base, data['query'], [text], timeout=900)[0]
                      for text in data['documents']]
            record.update(ids=ordered_ids(data['ids'], scores), scores=scores, status='complete')
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__)
    with (request.parent / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    return 0 if record['status'] == 'complete' else 1


def run(args, cfg):
    if not 1 <= args.k <= 30:
        raise ValueError('Reranked memory supports k=1..30')
    if args.refresh:
        raise ValueError('Reranking reads an existing index; refresh it separately')
    require_hot(cfg.db, 'developer memory index')
    if not cfg.db.is_file():
        print('devmem: no index; run scripts/devmem.py build --no-embed', file=sys.stderr)
        return 2
    data = snapshot(cfg, args.text, args.k)
    note = {'mode': 'lexical_auto_then_rerank', 'reranked': False, 'gpu_used': False}
    rows = data['hits'][:args.k]
    if not data['bypass']:
        require_gpu_allowed()
        require_hot(CACHE, 'rerank receipts')
        output = CACHE / uuid4().hex
        output.mkdir(parents=True)
        request = output / 'request.json'
        with request.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False)
        previous = os.environ.get('LSS_DEVMEM_RERANK_SUPERVISED')
        os.environ['LSS_DEVMEM_RERANK_SUPERVISED'] = str(output)
        try:
            code = batch.run_command([sys.executable, str(Path(__file__).resolve()), '--worker', str(request)],
                ROOT, output / 'worker.log', .1, storage_gb=1,
                storage_paths=[str(cfg.db), str(local_llm.get_model(MODEL).path), str(local_llm.llama_dir())])
        finally:
            if previous is None:
                os.environ.pop('LSS_DEVMEM_RERANK_SUPERVISED', None)
            else:
                os.environ['LSS_DEVMEM_RERANK_SUPERVISED'] = previous
        with (output / 'dispatch-result.json').open('x', encoding='utf-8') as stream:
            json.dump({'returncode': code, 'request_sha256': local_llm._sha256(request)}, stream)
        try:
            result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            print('devmem: rerank has no valid result; receipt: ' + str(output), file=sys.stderr)
            return 3
        if code or result.get('status') != 'complete':
            print('devmem: rerank failed; receipt: ' + str(output), file=sys.stderr)
            return 3
        expected = ordered_ids(data['ids'], result['scores'])
        if result['ids'] != expected or result['request_sha256'] != local_llm._sha256(request):
            raise ValueError('Rerank result identity mismatch')
        by_id = dict(zip(data['ids'], data['hits']))
        rows = [by_id[cid] for cid in expected[:args.k]]
        note.update(reranked=True, gpu_used=True, model=MODEL, duty=.1, receipt=str(output))
    else:
        note['mode'] = 'exact_bypass' if exact_identifier(args.text) else 'empty_lexical_pool'
    if args.json:
        print(json.dumps({'hits': rows, **note}, ensure_ascii=False))
    else:
        print(devmem.format_hits(rows, {}))
        if note['reranked']:
            print('(Qwen reranked at10% duty; receipt: ' + note['receipt'] + ')')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('Use scripts/dev.py memory --rerank')
    raise SystemExit(worker(Path(sys.argv[2])))
