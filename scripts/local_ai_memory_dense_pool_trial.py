"""Frozen candidate-recall ceiling: lexical 30 union dense 20, never a retrieval-quality claim."""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import batch
import devmem
import local_llm
import local_ai_memory_semantic_trial as frozen
from acquire_audio_probe import save
from dev_hot_paths import require_hot
from dev_quiet_mode import require_gpu_allowed
from devmem_rerank import exact_identifier

ROOT = Path(__file__).resolve().parents[1]
OUT = dev_evidence_root(ROOT) / 'release-evidence/local-ai-memory-dense-pool-r1'
PROVENANCE = dev_evidence_root(ROOT) / 'release-evidence/local-ai-memory-semantic-r1/reembed-r1/report.json'
MODEL_SHA = '06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439'
PURPOSE = 'quiet10pct frozen devmem dense candidate pool'
ENV = 'LSS_DENSE_POOL_SUPERVISED'


def validate_vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 1024:
        raise ValueError('query vector must have 1024 elements')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real) or not math.isfinite(x) for x in values):
        raise ValueError('query vector must contain finite non-boolean numbers')
    vec = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if not np.isfinite(vec).all() or not np.isfinite(norm) or norm < 1e-9:
        raise ValueError('query vector has invalid norm')
    return vec / norm


class CheckedEmbedder(devmem.Embedder):
    # Keep inherited embed_query and its exact instruction; inspect JSON before float conversion.
    def embed(self, texts):
        data = local_llm._http_json(self.url + '/v1/embeddings',
            {'input': texts, 'model': devmem.EMBED_MODEL}, timeout=self.timeout)['data']
        if len(texts) != 1 or len(data) != 1 or data[0]['index'] != 0:
            raise ValueError('unexpected embedding response')
        return np.asarray([validate_vector(data[0]['embedding'])], dtype=np.float32)


def stable_union(lexical, dense):
    return list(dict.fromkeys([*lexical, *dense]))


def pool(con, query, embedder):
    if exact_identifier(query):
        lexical, _ = frozen.lexical(con, query)
        return {'lexical_ids': lexical, 'dense_ids': [], 'candidate_ids': lexical,
                'exact_control_bypass': True, 'query_vector_sha256': None}
    lexical, _ = devmem.search(con, query, k=30, mode='auto', embedder=None)
    lexical = lexical[:30]
    vec = embedder.embed_query(query)
    # Validate again at this boundary, including injected test clients.
    vec = validate_vector(list(vec))
    dense = devmem.dense_ids(con, vec, n=20)
    if len(dense) != 20 or len(set(dense)) != 20:
        raise ValueError('dense candidate population incomplete')
    return {'lexical_ids': lexical, 'dense_ids': dense, 'candidate_ids': stable_union(lexical, dense),
            'exact_control_bypass': False,
            'query_vector_sha256': hashlib.sha256(vec.astype('<f4').tobytes()).hexdigest()}


def evaluate(con, document, embedder, progress=None):
    cases = []
    for case in document['cases']:
        started = time.monotonic()
        item = pool(con, case['query'], embedder)
        def supported(ids):
            hits = devmem.hits(con, ids, case['query'], len(ids), None)
            if len(hits) != len(ids): raise ValueError('candidate identity missing')
            return frozen.support_hit(hits, case['support_source'], case['support_lines'])
        item.update(id=case['id'], query=case['query'], support_source=case['support_source'],
                    support_lines=case['support_lines'], lexical_support=supported(item['lexical_ids']),
                    candidate_support=supported(item['candidate_ids']))
        cases.append(item)
        item['wall_s'] = round(time.monotonic() - started, 3)
        if progress:
            progress(cases)
    controls = []
    for control in document['exact_identifier_controls']:
        query = control['query']
        if not frozen.exact_control(query): raise ValueError('invalid exact control')
        item = pool(con, query, embedder)
        expected, _ = frozen.lexical(con, query)
        if not item['exact_control_bypass'] or item['candidate_ids'] != expected:
            raise ValueError('exact control failed')
        controls.append({'query': query, **item})
    return {'cases': cases, 'exact_controls': controls,
            'counts': {'n': len(cases), 'lexical_pool_support': sum(x['lexical_support'] for x in cases),
                       'union_pool_support': sum(x['candidate_support'] for x in cases),
                       'missing_union_support': [x['id'] for x in cases if not x['candidate_support']]}}


def worker():
    started = time.monotonic()
    record = {'status': 'running', 'stage': 'preflight', 'scope': 'candidate-recall ceiling only; not top8 retrieval quality or adoption',
              'duty': .1, 'gpu_lease_owner': 'local_llm.serving'}
    con = None
    try:
        require_gpu_allowed()
        model = local_llm.get_model(frozen.MODEL)
        runtime = local_llm.llama_dir()
        for path in (model.path, runtime, frozen.INDEX, frozen.BENCH, PROVENANCE, OUT, Path(sys.executable)):
            require_hot(path, 'dense pool experiment')
        if frozen.sha(frozen.INDEX) != frozen.INDEX_SHA or frozen.sha(frozen.BENCH) != frozen.BENCH_SHA:
            raise ValueError('frozen source identity mismatch')
        document = json.loads(frozen.BENCH.read_text(encoding='utf-8-sig'))
        con = sqlite3.connect(frozen.INDEX.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        frozen.preflight(con, document)  # Must reproduce before serving takes its lease.
        provenance = json.loads(PROVENANCE.read_text(encoding='utf-8-sig'))
        old = provenance['identity']
        dlls = {p.name: frozen.sha(p) for p in sorted(runtime.glob('*.dll'))}
        model_sha = frozen.sha(model.path)
        if (model_sha != MODEL_SHA or model.sha256.lower() != MODEL_SHA or
            provenance['status'] != 'complete' or provenance['embedding_comparison']['float16_bytes_identical'] != 4379 or
            provenance['embedding_comparison']['count'] != 4379 or old['source_sha256'] != frozen.INDEX_SHA or
            old['benchmark_sha256'] != frozen.BENCH_SHA or old['model_sha256'] != MODEL_SHA or
            old['model'] != model.name or old['model_ctx'] != model.ctx or old['model_args'] != list(model.args) or
            old['query_instruction'] != devmem.QUERY_INSTRUCT or old['runtime_build'] != local_llm.BUILD or
            old['runtime_sha256'] != frozen.sha(runtime / 'llama-server.exe') or old['runtime_dlls_sha256'] != dlls):
            raise ValueError('embedding provenance or runtime mismatch')
        record['identity'] = {'index_path': str(frozen.INDEX), 'index_sha256': frozen.INDEX_SHA,
            'benchmark_path': str(frozen.BENCH), 'benchmark_sha256': frozen.BENCH_SHA,
            'provenance_path': str(PROVENANCE), 'provenance_sha256': frozen.sha(PROVENANCE),
            'model_path': str(model.path), 'model_sha256': model_sha, 'model': model.name,
            'model_ctx': model.ctx, 'model_args': list(model.args), 'runtime_path': str(runtime),
            'runtime_build': local_llm.BUILD, 'runtime_sha256': old['runtime_sha256'], 'runtime_dlls_sha256': dlls,
            'devmem_sha256': frozen.sha(Path(devmem.__file__)), 'trial_sha256': frozen.sha(Path(__file__)),
            'query_instruction': devmem.QUERY_INSTRUCT, 'query_vector_encoding': 'normalized little-endian float32 SHA256'}
        record['stage'] = 'serving'
        save(OUT / 'progress.json', record)
        original_append = local_llm._append_row
        def telemetry(row):
            record['serving'] = dict(row)
            original_append(row)
        local_llm._append_row = telemetry
        with local_llm.serving(frozen.MODEL, purpose=PURPOSE) as base:
            record['evaluation'] = evaluate(con, document, CheckedEmbedder(base),
                progress=lambda cases: save(OUT / 'ranking-progress.json', {'cases': cases}))
        if frozen.sha(frozen.INDEX) != frozen.INDEX_SHA or frozen.sha(frozen.BENCH) != frozen.BENCH_SHA:
            raise ValueError('frozen input changed')
        record.update(status='complete', stage='complete')
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        record.pop('evaluation', None)
    finally:
        if con is not None: con.close()
        record['wall_s'] = round(time.monotonic() - started, 3)
        save(OUT / 'result.json', record)
    return 0 if record['status'] == 'complete' else 1


def main():
    if sys.argv[1:] == ['--worker']:
        if os.environ.get(ENV) != str(OUT): raise RuntimeError('Use the duty-capped supervisor')
        return worker()
    if sys.argv[1:]: raise ValueError('No arbitrary model or command')
    require_gpu_allowed()
    for path in (ROOT, OUT, Path(sys.executable), local_llm.llama_dir()): require_hot(path, 'dense pool supervisor')
    require_evidence_writable(OUT)
    OUT.mkdir(exist_ok=False)
    save(OUT / 'started.json', {'status': 'queued', 'duty': .1, 'parent_pid': os.getpid()})
    record = {'status': 'running', 'duty': .1, 'gpu_lease_owner': 'local_llm.serving'}
    try:
        os.environ[ENV] = str(OUT)
        from quiet_cpu import run_capped
        from quiet_dev_checks import environment
        temp = ROOT / '.cache/local-ai/dense-pool-contracts-r1'
        require_hot(temp, 'dense pool contracts')
        temp.mkdir(parents=True, exist_ok=False)
        with (OUT / 'contracts.log').open('xb') as log:
            checks = run_capped([sys.executable, '-m', 'pytest', '-v',
                'tests/test_memory_dense_pool_trial.py', 'tests/test_local_llm_transport.py',
                '--basetemp=' + str(temp / 'pytest'), '--junitxml=' + str(OUT / 'tests.xml')],
                cwd=ROOT, env=environment(temp), stdout=log, duty=.05, timeout=1800)
        record['contract_cpu_accounting'] = checks
        if checks['returncode']:
            raise ValueError('Dense pool contracts failed; model is not loaded')
        code = batch.run_command([sys.executable, str(Path(__file__).resolve()), '--worker'], ROOT,
            OUT / 'worker.log', .1, storage_gb=1,
            storage_paths=[str(local_llm.get_model(frozen.MODEL).path), str(local_llm.llama_dir())])
        record.update(status='passed' if code == 0 else 'failed', returncode=code)
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        save(OUT / 'supervisor-result.json', record)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
