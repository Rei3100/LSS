"""CPU-only FP32 cross-encoder replay of the frozen 22-query memory population.

Acquires two pinned public files at <=0.25 MiB/s on SSD, uses the already present
ONNX Runtime and isolated audio environment's tokenizer, and never installs
dependencies. Run through quiet_dev_checks --minilm-memory. No GPU, HTTP model
service, source scan, index refresh, Product change or retrieval-default change.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

from acquire_audio_probe import acquire, save
from dev_hot_paths import require_hot
import devmem
import local_ai_memory_semantic_trial as frozen

ROOT = Path(__file__).resolve().parents[1]
OUT = dev_evidence_root(ROOT) / 'release-evidence/local-ai-memory-minilm-r1'
DEST = Path('C:/LSS-Work/local-ai-models/ms-marco-MiniLM-L6-v2')
TOKENIZER_SITE = ROOT / '.cache/local-ai/omni-env/Lib/site-packages'
REPO = 'cross-encoder/ms-marco-MiniLM-L6-v2'
REV = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
FILES = (
    ('onnx/model.onnx', 91011230, '5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a'),
    ('tokenizer.json', 711396, 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'),
)


def ranked_ids(ids, scores):
    if len(ids) != len(scores) or len(set(ids)) != len(ids):
        raise ValueError('Ranking must preserve the candidate population')
    if any(type(s) not in (int, float) or not math.isfinite(s) for s in scores):
        raise ValueError('Ranking requires finite numeric scores')
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])]


def candidate(con, query, ranker):
    if frozen.exact_control(query):
        ids, hits = frozen.lexical(con, query)
        return ids, hits, {'exact_control_bypass': True}
    ids, _ = devmem.search(con, query, k=30, mode='auto', embedder=None)
    documents = [con.execute('SELECT text FROM chunk WHERE id=?', (cid,)).fetchone()[0] for cid in ids]
    scores = ranker(query, documents)
    ranked = ranked_ids(ids, scores)
    return ranked[:8], devmem.hits(con, ranked, query, 8, None), {
        'candidate_pool': len(ids), 'input_ids': ids, 'scores': scores,
        'truncation': 'longest_first, 512 tokens per query/document pair'}


def evaluate(con, document, baseline, ranker, progress=None):
    cases = []
    for case, (base_ids, base_hits) in zip(document['cases'], baseline):
        start = time.monotonic()
        ids, hits, note = candidate(con, case['query'], ranker)
        cases.append({'id': case['id'], 'query': case['query'], 'set': case['set'],
            'support_source': case['support_source'], 'support_lines': case['support_lines'],
            'baseline_ids': base_ids, 'candidate_ids': ids, 'candidate_hits': hits,
            'baseline_support': frozen.support_hit(base_hits, case['support_source'], case['support_lines']),
            'candidate_support': frozen.support_hit(hits, case['support_source'], case['support_lines']),
            'wall_s': round(time.monotonic() - start, 3), 'note': note})
        if progress:
            progress(cases)
    controls = []
    for row in document['exact_identifier_controls']:
        if not frozen.exact_control(row['query']):
            raise ValueError('Invalid frozen exact control')
        expected, _ = frozen.lexical(con, row['query'])
        actual, _, note = candidate(con, row['query'], ranker)
        controls.append({'query': row['query'], 'same_ordered_chunk_ids': actual == expected, 'note': note})
    if any(not c['same_ordered_chunk_ids'] for c in controls):
        raise ValueError('Exact identifier control changed')
    return {'cases': cases, 'exact_controls': controls, 'counts': {
        'n': len(cases), 'baseline_support_top8': sum(c['baseline_support'] for c in cases),
        'candidate_support_top8': sum(c['candidate_support'] for c in cases),
        'lost_support': [c['id'] for c in cases if c['baseline_support'] and not c['candidate_support']],
        'gained_support': [c['id'] for c in cases if c['candidate_support'] and not c['baseline_support']]}}


def main():
    from gpu_run import memory_headroom, MIN_COMMIT_GB
    from local_llm import peak_working_set_gb
    for path in (OUT, DEST, TOKENIZER_SITE, frozen.INDEX, frozen.BENCH, Path(sys.executable)):
        require_hot(path, 'quiet cross-encoder replay')
    require_evidence_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / 'result.json'
    if report.exists():
        raise ValueError('Existing trial result is never overwritten')
    with (OUT / 'started.json').open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, stream)
    if frozen.sha(frozen.INDEX) != frozen.INDEX_SHA or frozen.sha(frozen.BENCH) != frozen.BENCH_SHA:
        raise ValueError('Frozen index or benchmark identity mismatch')
    document = json.loads(frozen.BENCH.read_text(encoding='utf-8-sig'))
    con = sqlite3.connect(frozen.INDEX.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    record = {'status': 'running', 'model': REPO, 'revision': REV, 'precision': 'FP32',
        'device': 'cpu', 'gpu_used': False, 'default_changed': False, 'files': []}
    save(OUT / 'progress.json', record)
    started = time.monotonic()
    cpu_started = time.process_time()
    available_min = float('inf')

    def memory_guard():
        nonlocal available_min
        commit, available = memory_headroom()
        available_min = min(available_min, available)
        if available < 5 or commit < MIN_COMMIT_GB:
            raise RuntimeError('CPU replay memory headroom below its admission floor')

    try:
        record['stage'] = 'frozen_baseline'
        baseline = frozen.preflight(con, document)
        memory_guard()
        record['stage'] = 'public_acquisition'
        for name, size, digest in FILES:
            (DEST / name).parent.mkdir(parents=True, exist_ok=True)
            row = acquire(name, size, digest, destination=DEST, repository=REPO, revision=REV,
                          rate=0.25 * 1024**2,
                          progress=lambda p: save(OUT / 'acquisition-progress.json', p))
            record['files'].append(row)
            save(OUT / 'progress.json', record)
        # Load CPU libraries before adding the isolated tokenizer package search path.
        record['stage'] = 'cpu_library_import'
        import numpy as np
        import onnxruntime as ort
        sys.path.insert(0, str(TOKENIZER_SITE))
        from tokenizers import Tokenizer
        import tokenizers
        record['stage'] = 'tokenizer_load'
        tokenizer = Tokenizer.from_file(str(DEST / 'tokenizer.json'))
        tokenizer.enable_truncation(max_length=512, strategy='longest_first')
        tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        record['stage'] = 'cpu_session_load'
        session = ort.InferenceSession(str(DEST / 'onnx/model.onnx'), options,
                                       providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Unexpected execution provider')
        names = {i.name for i in session.get_inputs()}
        if names != {'input_ids', 'attention_mask', 'token_type_ids'}:
            raise ValueError('Unexpected ONNX input contract')

        def ranker(query, documents):
            scores = []
            for text in documents:
                memory_guard()
                encoded = tokenizer.encode(query, text)
                batch = {name: np.asarray([value], dtype=np.int64) for name, value in {
                    'input_ids': encoded.ids, 'attention_mask': encoded.attention_mask,
                    'token_type_ids': encoded.type_ids}.items()}
                logits = session.run(None, batch)[0]
                if logits.shape != (1, 1):
                    raise ValueError('Unexpected ONNX output contract')
                scores.append(float(logits[0, 0]))
            return scores

        record['identity'] = {'index_sha256': frozen.INDEX_SHA, 'benchmark_sha256': frozen.BENCH_SHA,
            'trial_sha256': frozen.sha(Path(__file__)), 'devmem_sha256': frozen.sha(Path(devmem.__file__)),
            'onnxruntime_version': ort.__version__, 'tokenizers_version': tokenizers.__version__,
            'onnx_native_sha256': {p.name: frozen.sha(p) for p in
                sorted((Path(ort.__file__).parent / 'capi').glob('*.pyd'))},
            'tokenizer_native_sha256': {p.name: frozen.sha(p) for p in
                sorted(Path(tokenizers.__file__).parent.glob('*.pyd'))},
            'providers': session.get_providers(), 'candidate_pool_max': 30, 'top_k': 8, 'max_tokens': 512}
        record['stage'] = 'frozen_ranking'
        record.update(evaluate(con, document, baseline, ranker,
            progress=lambda cases: save(OUT / 'ranking-progress.json', {'cases': cases})))
        if frozen.sha(frozen.INDEX) != frozen.INDEX_SHA:
            raise ValueError('Frozen index changed during replay')
        record.update(status='complete', stage='complete')
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__)
    finally:
        con.close()
        record.update(wall_s=round(time.monotonic()-started, 3),
            process_cpu_s=round(time.process_time()-cpu_started, 3),
            available_ram_min_gb=None if available_min == float('inf') else available_min,
            process_peak_working_set_gb=peak_working_set_gb(os.getpid()))
        save(report, record)
    return 0 if record['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
