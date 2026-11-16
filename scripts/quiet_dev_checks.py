"""SSD-only, CPU-only overnight checks at 0.5% duty on one logical processor.

No EXE inference, GPU work, archive traversal or storage cleanup occurs. Explicit
reranker-acquire downloads one pinned file at <=0.25 MiB/s. Explicit minilm-memory
acquires two pinned files at that rate and runs a CPU-only FP32 frozen replay.
The original resource floors remain; insufficient capacity fails without nudging
the HDD maintenance task. Only the fixed mechanical commands below can run.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from quiet_cpu import pin_one_processor, run_capped

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path('C:/LSS-Work/wt/codex-scene-bootstrap-build-r2-20261027')
SOURCE = Path('C:/LSS-Work/wt/codex-scene-bootstrap-20261027')
DUTY = 0.005
CPU_TESTS = ('tests/test_codex_pacing.py', 'tests/test_dev_quiet_mode.py', 'tests/test_verified_model_retirement.py', 'tests/test_native_probe_gpu_pause.py',
             'tests/test_githooks.py', 'tests/test_strata_queue.py', 'tests/test_strata_evidence.py',
             'tests/test_check_packaged_scene_bridge.py', 'tests/test_local_ai_queue.py',
             'tests/test_local_rerank_contract.py', 'tests/test_memory_minilm_trial.py',
             'tests/test_quiet_storage.py', 'tests/test_storage_tiers.py', 'tests/test_storage_budget.py',
             'tests/test_quiet_cpu_rate_probe.py', 'tests/test_quiet_dev_receipts.py',
             'tests/test_local_llm_transport.py', 'tests/test_local_cpu_runtime.py')


def environment(temp):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'NVIDIA_VISIBLE_DEVICES': 'none',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1', 'TEMP': str(temp), 'TMP': str(temp),
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}


def run(output, **options):
    """Claim an owned directory and publish failure even before the first step.

    Claim happens outside the error handler: a second launcher cannot rewrite
    somebody else's receipt. A failed dependency/admission/native cap becomes an
    explicit failed result, so downstream jobs do not wait for absent output.
    """
    from dev_hot_paths import require_hot
    require_evidence_writable(output)
    output = output.resolve()
    if not output.is_relative_to((dev_evidence_root(ROOT) / 'release-evidence').resolve()) or output.exists():
        raise ValueError('Output must be a new owned evidence directory')
    require_hot(output, 'quiet result directory')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'started.json').write_text(json.dumps({'status': 'initializing', 'pid': os.getpid()}) + '\n', encoding='utf-8')
    try:
        return _run_claimed(output, **options)
    except Exception as exc:
        result = output / 'result.json'
        try:
            record = json.loads(result.read_text(encoding='utf-8'))
            if not isinstance(record, dict):
                record = {}
        except (OSError, ValueError):
            record = {}
        record.update(status='failed', error_type=type(exc).__name__,
                      gpu_inference=False, maintenance_nudges=False)
        # Error text can contain a signed acquisition URL; retain type only.
        result.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        raise


def _run_claimed(output, *, finalize=False, prepare=False, tests_only=False, readiness_tests=False, reranker_acquire=False, minilm_memory=False, storage_quiet_tests=False, source_loopback_tests=False, pure_cpu_runtime=False, after=None, test_files=None):
    from dev_hot_paths import require_hot
    import storage_budget
    require_evidence_writable(output)
    output = output.resolve()
    tests_here = tests_only and not any((finalize, prepare, readiness_tests, reranker_acquire,
                                       minilm_memory, storage_quiet_tests, source_loopback_tests, pure_cpu_runtime))
    work_paths = [ROOT, Path(sys.executable), output] if tests_here else [ROOT, SOURCE, BUILD, Path(sys.executable), output]
    temp = (ROOT / '.cache/local-ai/quiet-development-tmp').resolve()
    if not temp.is_relative_to((ROOT / '.cache').resolve()):
        raise ValueError('Temporary directory escapes the owned cache')
    if tests_here:
        work_paths.append(temp)
    for path in work_paths:
        require_hot(path, 'quiet development')
    if after:
        after = after.resolve()
        if not after.is_relative_to((dev_evidence_root(ROOT) / 'release-evidence').resolve()):
            raise ValueError('Dependency must be owned evidence')
        deadline = time.monotonic() + 4 * 60 * 60
        while True:
            try:
                dependency = json.loads(after.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                dependency = {}
            if dependency.get('status') == 'failed':
                raise RuntimeError('Required quiet check failed')
            if dependency.get('status') == 'passed':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Quiet dependency deadline')
            time.sleep(60)  # OS-owned wait; no agent calls or hardware polling.
    with storage_budget.reserve(1, [str(path) for path in work_paths],
                                'quiet CPU development checks', wait=False, _allow_nudge=False):
        if not temp.is_relative_to((ROOT / '.cache').resolve()):
            raise ValueError('Temporary directory escapes the owned cache')
        temp.mkdir(parents=True, exist_ok=True)
        require_hot(temp, 'quiet temporary directory')
        for name in ('pytest', 'readiness-pytest', 'storage-pytest', 'source-loopback-pytest'):
            base = (temp / name).resolve()
            if not base.is_relative_to(temp):
                raise ValueError('Pytest temporary path escapes the owned cache')
            require_hot(base, 'quiet pytest temporary directory')
        tasks = [
            ('pacing-tests', [sys.executable, '-m', 'pytest', *(test_files or CPU_TESTS[:3]),
                '-v', '--basetemp=' + str(temp / 'pytest'), '--junitxml=' + str(output / 'tests.xml')]),
            ('packaged-source-identity', [sys.executable, str(BUILD / 'scripts/check_runtime_sources.py'), str(BUILD / 'dist/LSS')]),
            ('package-exe-hash', [sys.executable, '-c',
                "import hashlib,json,sys; print(json.dumps({'exe_sha256':hashlib.file_digest(open(sys.argv[1],'rb'),'sha256').hexdigest()}))",
                str(BUILD / 'dist/LSS/LSS.exe')]),
        ]
        if finalize:
            tasks = [tasks[0], ('bridge-and-freeze', [sys.executable,
                str(ROOT / 'scripts/check_packaged_scene_bridge.py'), '--source', str(BUILD),
                '--commit', 'bac1653dae92d00d3e56ab57f758ce2e2c657174',
                '--freeze-out', str(dev_evidence_root(ROOT) / 'release-evidence/native-bootstrap-frozen-20261027')])]
        if tests_only:
            tasks = tasks[:1]
        if prepare:
            probe = dev_evidence_root(SOURCE) / 'release-evidence/scene-bootstrap-20261027/tools/packaged_scene_probe.py'
            require_hot(probe, 'synthetic fixture preparation')
            tasks = [('prepare-synthetic-only', [sys.executable, str(probe)])]
        if readiness_tests:
            tasks = [('source-readiness-tests', [sys.executable, '-m', 'pytest',
                str(SOURCE / 'tests/test_scene_readiness_status.py'), '-v',
                str(dev_evidence_root(SOURCE) / 'release-evidence/scene-bootstrap-20261027/tools/test_packaged_scene_probe.py'),
                '--basetemp=' + str(temp / 'readiness-pytest'),
                '--junitxml=' + str(output / 'tests.xml')])]
        if reranker_acquire:
            require_hot(Path('C:/LSS-Work/local-ai-models/qwen3-reranker-0.6b'), 'reranker acquisition')
            tasks = [('acquire-reranker-only', [sys.executable, str(ROOT / 'scripts/acquire_memory_reranker.py')])]
        if minilm_memory:
            require_hot(Path('C:/LSS-Work/local-ai-models/ms-marco-MiniLM-L6-v2'), 'CPU cross-encoder model')
            tasks = [('minilm-frozen-replay', [sys.executable, str(ROOT / 'scripts/local_ai_memory_minilm_trial.py')])]
        if storage_quiet_tests:
            # Wall-clock-sensitive child/crash tests remain pending at ordinary
            # duty; extreme suspension must not force relaxed timing assertions.
            tasks = [('storage-quiet-contracts', [sys.executable, '-m', 'pytest', '-v',
                'tests/test_quiet_storage.py', 'tests/test_storage_tiers.py', 'tests/test_storage_budget.py',
                '-k', 'not cross_process_lease_capacity_and_crash_recovery and not cli_nested_policy_exit_and_receipt and not cli_owns_descendants_on_exit_or_runner_death',
                '--basetemp=' + str(temp / 'storage-pytest'), '--junitxml=' + str(output / 'tests.xml')])]
        if source_loopback_tests:
            tasks = [('source-native-network', [sys.executable, '-m', 'pytest', '-v',
                str(SOURCE / 'tests/test_scene_native_loopback.py'),
                str(SOURCE / 'tests/test_scene_native_tls.py'),
                '--basetemp=' + str(temp / 'source-loopback-pytest'), '--junitxml=' + str(output / 'tests.xml')])]
        if pure_cpu_runtime:
            require_hot(Path('C:/LSS-Work/local-ai-models/llama-b11878-cpu-x64'), 'isolated CPU runtime')
            tasks = [('source-public-download-cpu-runtime', [sys.executable, str(ROOT / 'scripts/acquire_pure_cpu_runtime.py')])]
        record = {'status': 'running', 'duty': DUTY, 'cpu_processors': 1, 'gpu_inference': False,
                  'cpu_cap': 'windows_job_cpu_hard_cap',
                  'maintenance_nudges': False, 'weekly_pacing': False, 'steps': []}
        for name, command in tasks:
            started = time.monotonic()
            with (output / (name + '.log')).open('xb') as log:
                metrics = run_capped(command, cwd=SOURCE if readiness_tests or source_loopback_tests else ROOT,
                                     env=environment(temp), stdout=log, duty=DUTY)
                code = metrics['returncode']
            record['steps'].append({'name': name, 'returncode': code,
                                    'wall_seconds': round(time.monotonic() - started, 3), 'cpu_accounting': metrics})
            (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
            if code:
                record['skipped_steps'] = [task[0] for task in tasks[len(record['steps']):]]
                break
        record['status'] = 'passed' if all(s['returncode'] == 0 for s in record['steps']) else 'failed'
        (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--prepare', action='store_true', help='prepare own synthetic data; never run the EXE')
    parser.add_argument('--tests-only', action='store_true')
    parser.add_argument('--readiness-tests', action='store_true')
    parser.add_argument('--reranker-acquire', action='store_true')
    parser.add_argument('--minilm-memory', action='store_true')
    parser.add_argument('--storage-quiet-tests', action='store_true')
    parser.add_argument('--source-loopback-tests', '--source-network-tests', action='store_true')
    parser.add_argument('--pure-cpu-runtime', action='store_true')
    parser.add_argument('--test-file', action='append', choices=CPU_TESTS)
    parser.add_argument('--after', type=Path)
    args = parser.parse_args()
    if sum((args.finalize, args.prepare, args.tests_only, args.readiness_tests, args.reranker_acquire, args.minilm_memory, args.storage_quiet_tests, args.source_loopback_tests, args.pure_cpu_runtime)) > 1:
        parser.error('choose one check mode')
    if args.test_file and not args.tests_only:
        parser.error('explicit test files require tests-only mode')
    report = run(args.out, finalize=args.finalize, prepare=args.prepare, tests_only=args.tests_only,
                 readiness_tests=args.readiness_tests, reranker_acquire=args.reranker_acquire,
                 minilm_memory=args.minilm_memory,
                 storage_quiet_tests=args.storage_quiet_tests,
                 source_loopback_tests=args.source_loopback_tests,
                 pure_cpu_runtime=args.pure_cpu_runtime,
                 after=args.after, test_files=args.test_file)
    print(json.dumps(report))
    raise SystemExit(0 if report['status'] == 'passed' else 1)
