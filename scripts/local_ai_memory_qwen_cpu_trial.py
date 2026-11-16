"""Pinned pure-CPU Qwen reranker calibration followed by the frozen 22 queries.

Default entry owns a hidden, one-processor Windows Job hard-capped at .005 duty.
No acquisition, GPU, Product execution, source scan or retrieval default change.
The worker is internal; launching it outside the supervisor is refused.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from acquire_audio_probe import save
from dev_hot_paths import require_hot
import local_ai_memory_semantic_trial as frozen
import local_ai_memory_minilm_trial as comparison
import local_llm
from quiet_cpu import run_capped
from quiet_dev_checks import environment
import storage_budget

ROOT = Path(__file__).resolve().parents[1]
OUT = dev_evidence_root(ROOT) / 'release-evidence/local-ai-memory-qwen-cpu-r1'
RUNTIME = Path('C:/LSS-Work/local-ai-models/llama-b11878-cpu-x64/bin')
PREPARED = dev_evidence_root(ROOT) / 'release-evidence/pure-cpu-runtime-20261027/result.json'
MODEL = 'qwen3-reranker-0.6b'
MODEL_SHA = '22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48'
CALIBRATION = (
    ('What is the capital of France?', ['Paris is the capital of France.', 'Bananas are yellow fruits.']),
    ('Which animal barks?', ['A domestic dog barks.', 'A triangle has three sides.']),
    ('日本の首都はどこですか？', ['日本の首都は東京です。', 'バナナは黄色い果物です。']),
)


def calibration_passes(scores):
    return len(scores) == len(CALIBRATION) and all(
        len(row) == 2 and comparison.ranked_ids([0, 1], row)[0] == 0
        and row[0] > row[1] for row in scores)


def verify_runtime():
    prepared = json.loads(PREPARED.read_text(encoding='utf-8-sig'))
    if prepared['status'] != 'prepared' or prepared['tag'] != 'b11878':
        raise ValueError('Unverified CPU runtime receipt')
    for row in prepared['files']:
        path = (RUNTIME / row['name']).resolve()
        if path.parent != RUNTIME.resolve():
            raise ValueError('Invalid runtime file path')
        if path.stat().st_size != row['bytes'] or frozen.sha(path) != row['sha256']:
            raise ValueError('CPU runtime identity mismatch')
    return prepared['files']


def worker():
    if os.environ.get('LSS_QWEN_CPU_SUPERVISED') != str(OUT):
        raise RuntimeError('Run the capped supervisor entry')
    started = time.monotonic()
    record = {'status': 'running', 'stage': 'identity', 'device': 'cpu',
              'gpu_used': False, 'default_changed': False, 'model': MODEL,
              'model_sha256': MODEL_SHA, 'runtime_dir': str(RUNTIME)}
    con = None
    try:
        runtime_files = verify_runtime()
        model = local_llm.get_model(MODEL)
        require_hot(model.path, 'Qwen reranker weights')
        if frozen.sha(model.path) != MODEL_SHA or model.sha256 != MODEL_SHA:
            raise ValueError('Reranker identity mismatch')
        if frozen.sha(frozen.INDEX) != frozen.INDEX_SHA or frozen.sha(frozen.BENCH) != frozen.BENCH_SHA:
            raise ValueError('Frozen retrieval population mismatch')
        document = json.loads(frozen.BENCH.read_text(encoding='utf-8-sig'))
        con = sqlite3.connect(frozen.INDEX.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        baseline = frozen.preflight(con, document)
        # An explicit experiment alias preserves the registry and default batch.
        # Single-document requests keep full source text; context overflow fails.
        experiment = replace(model, args=('-b', '512', '-ub', '512'), ram_gb=2.0)
        original_get = local_llm.get_model
        local_llm.get_model = lambda name: experiment if name == MODEL else original_get(name)
        record['identity'] = {'index_sha256': frozen.INDEX_SHA, 'benchmark_sha256': frozen.BENCH_SHA,
            'trial_sha256': frozen.sha(Path(__file__)), 'runtime_files': runtime_files,
            'model_args': list(experiment.args), 'ctx': 8192,
            'documents_per_request': 1, 'candidate_pool_max': 30, 'top_k': 8,
            'truncation': 'none by harness; runtime context overflow is a failure'}
        record['stage'] = 'cpu_server_load'
        save(OUT / 'progress.json', record)
        with local_llm.serving(MODEL, cpu=True, runtime_dir=RUNTIME, ctx=8192,
                health_timeout=18000, purpose='quiet frozen Qwen reranker calibration and comparison') as base:
            def ranker(query, documents):
                values = []
                for text in documents:
                    values.extend(local_llm.rerank(base, query, [text], timeout=7200))
                comparison.ranked_ids(list(range(len(documents))), values)
                return values

            record['stage'] = 'calibration'
            record['calibration'] = []
            save(OUT / 'progress.json', record)
            for query, documents in CALIBRATION:
                scores = ranker(query, documents)
                record['calibration'].append({'query': query, 'documents': documents, 'scores': scores})
                save(OUT / 'progress.json', record)
            if not calibration_passes([row['scores'] for row in record['calibration']]):
                record.update(status='mechanism_unqualified', stage='calibration_failed')
            else:
                record['stage'] = 'frozen_ranking'
                save(OUT / 'progress.json', record)
                def progress(cases):
                    for case in cases:
                        if not case['note'].get('exact_control_bypass'):
                            case['note']['truncation'] = record['identity']['truncation']
                    save(OUT / 'ranking-progress.json', {'cases': cases})
                result = comparison.evaluate(con, document, baseline, ranker,
                    progress=progress)
                # The reused comparator's tokenizer description is MiniLM-specific.
                for case in result['cases']:
                    if not case['note'].get('exact_control_bypass'):
                        case['note']['truncation'] = record['identity']['truncation']
                record.update(result)
                record.update(status='complete', stage='complete')
        if frozen.sha(frozen.INDEX) != frozen.INDEX_SHA:
            raise ValueError('Frozen index changed')
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__)
    finally:
        if con is not None:
            con.close()
        record['wall_s'] = round(time.monotonic() - started, 3)
        save(OUT / 'result.json', record)
    return 0 if record['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker()
    for path in (ROOT, OUT, RUNTIME, PREPARED, frozen.INDEX, frozen.BENCH, Path(sys.executable)):
        require_hot(path, 'quiet CPU reranker')
    require_evidence_writable(OUT)
    OUT.mkdir(exist_ok=False)
    save(OUT / 'started.json', {'pid': os.getpid(), 'gpu_used': False, 'duty': .005})
    record = {'status': 'running', 'gpu_used': False, 'maintenance_nudges': False}
    try:
        with storage_budget.reserve(1, [str(ROOT), str(OUT), str(RUNTIME)],
                'quiet CPU reranker comparison', wait=False, _allow_nudge=False):
            temp = ROOT / '.cache/quiet-qwen-rerank-r1'
            temp.mkdir(exist_ok=False)
            env = environment(temp)
            for key in tuple(env):
                if key.startswith('LLAMA_ARG_') or key in ('GGML_RPC', 'RPC_SERVERS'):
                    env.pop(key)
            env.update(GGML_BACKEND_PATH=str(RUNTIME), LSS_QWEN_CPU_SUPERVISED=str(OUT))
            with (OUT / 'contracts.log').open('xb') as log:
                checks = run_capped([sys.executable, '-m', 'pytest', '-v',
                    'tests/test_memory_qwen_cpu_trial.py', 'tests/test_memory_minilm_trial.py',
                    '--basetemp=' + str(temp / 'pytest'), '--junitxml=' + str(OUT / 'tests.xml')],
                    cwd=ROOT, env=env, stdout=log, duty=.005, timeout=3600)
            record['contract_cpu_accounting'] = checks
            if checks['returncode']:
                raise RuntimeError('Comparison contracts failed; inference skipped')
            with (OUT / 'worker.log').open('xb') as log:
                metrics = run_capped([sys.executable, str(Path(__file__).resolve()), '--worker'],
                    cwd=ROOT, env=env, stdout=log, duty=.005, timeout=172800)
            record.update(cpu_accounting=metrics,
                          status='passed' if metrics['returncode'] == 0 else 'failed')
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__)
    finally:
        save(OUT / 'supervisor-result.json', record)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
