"""Source-only cached naming with acquisition explicitly disabled in this process.

Does not launch the frozen LSS EXE. Calls the checked LSS naming entry with an
explicit native model object and existing verified SSD paths. Install, download
and bootstrap functions raise before acquisition; native engine children receive
local model paths. A new synthetic fixture protects all prior probe states.
"""
try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parents[1]
COMMIT = '89e826f998ac49db89d9a191aa5caf1d8b65104f'
OUT = dev_evidence_root(ROOT) / 'release-evidence/scene-source-cached-20261028'
CASE = ROOT / '.cache/local-ai/scene-source-cached-20261028'
NATIVE = Path('C:/LSS-Work/local-ai-models/scene-native-probe-r3-20261027/native')
PROOF = dev_evidence_root(ROOT) / 'release-evidence/native-cached-manifest-verification-20261028-r2'


def save(path, value):
    require_evidence_writable(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def deny_acquisition(*args, **kwargs):
    raise RuntimeError('Acquisition is disabled for Source cached inference')


def validate_receipt(record, supervisor, observed):
    if (record.get('status') != 'verified' or record.get('source_commit') != COMMIT
            or record.get('ready') is not True or record.get('existing_fixture_files_unchanged') is not True
            or supervisor.get('status') != 'passed' or record.get('manifest_sha256') != observed):
        raise ValueError('Cached install proof mismatch')


def worker():
    from dev_quiet_mode import require_gpu_allowed
    require_gpu_allowed()
    if os.environ.get('LSS_SOURCE_CACHED_SUPERVISED') != str(OUT):
        raise ValueError('Use the duty supervisor')
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'scripts')]
    from lss.bgm import scene_native as native, scene_runtime, scene_job, scene_names
    native._download = deny_acquisition
    native.install = deny_acquisition
    scene_runtime.ensure_bootstrapped = deny_acquisition
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() != COMMIT:
        raise ValueError('Source identity changed')
    if subprocess.check_output(['git', 'diff', '--name-only'], cwd=SOURCE).strip():
        raise ValueError('Source tracked changes')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    validate_receipt(read(PROOF / 'verification.json'), read(PROOF / 'result.json'),
                     digest(NATIVE / native.MANIFEST))
    if not native.ready(NATIVE):
        raise ValueError('Cache no longer ready; no acquisition is permitted')
    spec = importlib.util.spec_from_file_location('synthetic_source_fixture',
        dev_evidence_root(SOURCE) / 'release-evidence/scene-bootstrap-20261027/tools/packaged_scene_probe.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    info = helper.prepare(CASE)
    from mutagen.flac import FLAC
    from lss.core.audio_identity import audio_identity
    def protected():
        return {'source': digest(Path(info['source'])), 'cue_pcm': audio_identity(Path(info['cue']))['pcm_sha256'],
                'current_best': digest(Path(info['current_best'])), 'accepted': digest(Path(info['guard'])),
                'genre': list(FLAC(info['cue']).get('GENRE', []))}
    before = protected()
    record = {'status': 'running', 'source_commit': COMMIT, 'frozen_exe_run': False,
              'scope': 'Source synthetic mechanism, not frozen acceptance or title quality',
              'acquisition_disabled': True, 'before': before, 'stages': []}
    started = time.monotonic()
    try:
        for label in ('first', 'repeat'):
            require_gpu_allowed()
            models = native.NativeSceneModels(root=NATIVE)
            stage = time.monotonic()
            try:
                result = scene_job.name_source_checked(Path(info['state']), Path(info['source']),
                                                       info['source_sha256'], models=models)
            finally:
                models.release()
            record['stages'].append({'label': label, 'seconds': time.monotonic() - stage, 'result': result})
            record[label + '_names'] = scene_names.read(Path(info['state']))
            save(OUT / 'progress.json', record)
        record['after'] = protected()
        if record['after'] != before:
            raise ValueError('Protected synthetic values changed')
        record['names_repeat_equal'] = record['first_names'] == record['repeat_names']
        record['gpu_lease_after'] = helper.observe_gpu_lease()
        record['status'] = 'observed'
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, detail=str(exc)[:300])
    finally:
        record['wall_seconds'] = time.monotonic() - started
        save(OUT / 'result.json', record)
    return 0 if record['status'] == 'observed' else 1


def main():
    if sys.argv[1:] == ['--worker']:
        return worker()
    if sys.argv[1:]:
        raise ValueError('Fixed synthetic Source trial only')
    from dev_quiet_mode import require_gpu_allowed
    from dev_hot_paths import require_hot
    import batch
    require_gpu_allowed()
    for p in (ROOT, SOURCE, OUT, CASE, NATIVE):
        require_hot(p, 'Source-only cached trial')
    require_evidence_writable(OUT)
    OUT.mkdir(exist_ok=False)
    from quiet_cpu import run_capped
    from quiet_dev_checks import environment
    temp = ROOT / '.cache/local-ai/scene-source-cached-tests'
    temp.mkdir(parents=True, exist_ok=True)
    with (OUT / 'tests.log').open('xb') as log:
        checked = run_capped([sys.executable, '-m', 'pytest', '-q',
            'tests/test_native_source_cached_probe.py', '--basetemp=' + str(temp / 'pytest'),
            '--junitxml=' + str(OUT / 'tests.xml')], cwd=ROOT, env=environment(temp),
            stdout=log, duty=.05, timeout=1200)
    save(OUT / 'test-result.json', checked)
    if checked['returncode']:
        return 1
    save(OUT / 'started.json', {'status': 'queued', 'incremental_storage_gb': .5,
                              'reason': 'No acquisition entry is enabled; only new10s synthetic fixture and bounded native outputs',
                              'frozen_exe_run': False, 'source_commit': COMMIT})
    os.environ['LSS_SOURCE_CACHED_SUPERVISED'] = str(OUT)
    code = batch.run_command([sys.executable, str(Path(__file__).resolve()), '--worker'], ROOT,
        OUT / 'worker.log', .1, storage_gb=.5, storage_paths=[str(ROOT), str(CASE), str(NATIVE)])
    save(OUT / 'supervisor-result.json', {'status': 'passed' if code == 0 else 'failed', 'returncode': code})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
