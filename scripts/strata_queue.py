"""Private Strata drafting queue. Replies are unverified; never execute or apply them."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import urllib.error
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import local_ai_queue as queue
import strata_dev as runtime
import strata_trials

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {'temperature': 0, 'chat_template_kwargs': {'reasoning_effort': 'low'}}
HASH_CACHE = ROOT / '.cache/local-ai/strata-byte-hashes.json'


def stat_key(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'device': stat.st_dev, 'inode': stat.st_ino,
            'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def file_hash(path, cache):
    path = Path(path)
    before = stat_key(path)
    entry = cache.get(before['path'])
    if entry and entry.get('stat') == before and isinstance(entry.get('sha256'), str):
        return entry['sha256']
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if stat_key(path) != before:
        raise RuntimeError(f'File changed during SHA256 calculation: {path}')
    cache[before['path']] = {'stat': before, 'sha256': sha}
    return sha


def identity(_model):
    profile = runtime.active_profile()
    home, model = runtime.paths(profile)
    runtime.select_variant(profile)
    if runtime.prepared_profile(home, model, profile) is None:
        raise ValueError('Selected private Strata profile is missing or does not match its configuration')
    config = runtime.state_root(home) / 'user-state/Strata/launch' / f'{runtime.profile_tag(profile)}.json'
    engine = home / 'components/engine' / f"{runtime.PIN['engine']}-windows-x86_64-cuda-cuda13"
    terminal = home / 'terminal/current'
    try:
        cache = json.loads(HASH_CACHE.read_text(encoding='utf-8'))
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    weights = {}
    for name, expected, size in runtime.PROFILES[profile]['files']:
        path = model.parent / name
        if path.stat().st_size != size:
            raise ValueError(f'Strata model shard size mismatch: {name}')
        sha = file_hash(path, cache)
        if sha.lower() != expected.lower():
            raise ValueError(f'Strata model shard SHA256 mismatch: {name}')
        weights[name] = sha
    paths = [engine / 'strata.exe', *sorted(engine.glob('*.dll')),
             terminal / 'strata.py', terminal / 'setup.py', config,
             Path(runtime.__file__), Path(queue.__file__), Path(__file__),
             ROOT / 'scripts/gpu_run.py', ROOT / 'scripts/strata_process_job.py']
    paths.extend(sorted((terminal / 'strata_control').rglob('*.py')))
    paths.extend(sorted((terminal / 'strata_cli').rglob('*.py')))
    paths.extend(sorted(p for p in (terminal / 'serve').rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.jinja') and not p.name.startswith('test_')))
    config_args = json.loads(config.read_text(encoding='utf-8'))['args']
    if '--expert-profile' in config_args:
        paths.append(Path(config_args[config_args.index('--expert-profile') + 1]))
    pack = home / 'Strata-data/packs' / runtime.PACKS[profile]
    if not pack.is_dir():
        raise ValueError('Strata model pack is missing')
    paths.extend(sorted(p for p in pack.rglob('*') if p.is_file()
                        and not p.name.endswith(('.tmp', '.partial', '.log', '.tmp.json'))))
    hashes = {str(p.resolve()): file_hash(p, cache) for p in paths}
    HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    queue.atomic_json(HASH_CACHE, cache)
    return {'model': f'strata-{profile}', 'shard_sha256': weights,
            'model_sha256': queue.digest(weights),
            'model_identity_kind': 'composite SHA256 of verified per-shard byte SHA256 values',
            'files_sha256': hashes,
            'variant': runtime.ACTIVE_VARIANT, 'request_defaults': DEFAULTS,
            'identity_method': 'SHA256 bytes with device/inode/size/mtime/ctime keyed cache'}


def schema_validator(schema):
    """Select an explicit, locally supported dialect; default to draft 2020-12."""
    from jsonschema import Draft7Validator, Draft201909Validator, Draft202012Validator
    dialects = {'http://json-schema.org/draft-07/schema#': Draft7Validator,
                'https://json-schema.org/draft/2019-09/schema': Draft201909Validator,
                'https://json-schema.org/draft/2020-12/schema': Draft202012Validator}
    declared = schema.get('$schema')
    if declared is not None and declared not in dialects:
        raise ValueError(f'Unsupported JSON Schema dialect: {declared}')
    return dialects.get(declared, Draft202012Validator)


def check_schemas(requests):
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('$ref', '$dynamicRef', '$id') and (not isinstance(item, str) or not item.startswith('#')):
                    raise ValueError('External schema references are not supported')
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    for request in requests:
        schema = request.get('schema')
        if schema is not None:
            inspect(schema)
            try:
                schema_validator(schema).check_schema(schema)
            except Exception as exc:
                raise ValueError(f'Invalid JSON schema: {exc}') from exc


def validate_receipt(request, receipt):
    if request.get('schema') is not None:
        schema_validator(request['schema'])(request['schema']).validate(json.loads(receipt['text']))


def verify_owned_worker():
    """Refuse model startup unless this process is in the controller's named Job."""
    if os.name != 'nt':
        raise RuntimeError('Strata worker ownership requires Windows Job Objects')
    import ctypes
    from ctypes import wintypes
    from strata_process_job import JOB_ENV
    name = os.environ.get(JOB_ENV)
    if not name:
        raise RuntimeError('Owned Job identity is missing')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    job = kernel.OpenJobObjectW(0x0004, False, name)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        contained = wintypes.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), job, ctypes.byref(contained)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not contained.value:
            raise RuntimeError('Strata worker is not a member of the owned Job')
    finally:
        kernel.CloseHandle(job)


def verify_owned_listener(port):
    """Verify the TCP listener and worker belong to the controller's exact Job."""
    if os.name != 'nt':
        raise RuntimeError('Strata listener ownership requires Windows Job Objects')
    import ctypes
    from ctypes import wintypes
    class TcpRow(ctypes.Structure):
        _fields_ = [('state', wintypes.DWORD), ('local_addr', wintypes.DWORD),
                    ('local_port', wintypes.DWORD), ('remote_addr', wintypes.DWORD),
                    ('remote_port', wintypes.DWORD), ('pid', wintypes.DWORD)]
    ip = ctypes.WinDLL('iphlpapi', use_last_error=True)
    ip.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                                     wintypes.BOOL, wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
    ip.GetExtendedTcpTable.restype = wintypes.ULONG
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k.OpenJobObjectW.restype = wintypes.HANDLE
    k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    k.IsProcessInJob.restype = wintypes.BOOL
    k.GetCurrentProcess.argtypes = []
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CloseHandle.restype = wintypes.BOOL
    size = wintypes.DWORD()
    if ip.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 3, 0) != 122:
        raise RuntimeError('Cannot query TCP listener ownership')
    buffer = ctypes.create_string_buffer(size.value)
    if ip.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 3, 0) != 0:
        raise RuntimeError('Cannot query TCP listener ownership')
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD)).contents.value
    rows = ctypes.cast(ctypes.addressof(buffer) + 4, ctypes.POINTER(TcpRow))
    pids = {int(rows[i].pid) for i in range(count) if socket.ntohs(rows[i].local_port & 0xffff) == port
            and rows[i].local_addr in (0, int.from_bytes(socket.inet_aton('127.0.0.1'), 'little'))}
    if len(pids) != 1:
        raise RuntimeError('Expected one local Strata listener')
    pid = pids.pop()
    from strata_process_job import JOB_ENV
    name = os.environ.get(JOB_ENV)
    if not name:
        raise RuntimeError('Owned Job identity is missing')
    job = k.OpenJobObjectW(0x0004, False, name)  # JOB_OBJECT_QUERY only
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        handle = k.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            for process in (k.GetCurrentProcess(), handle):
                contained = wintypes.BOOL()
                if not k.IsProcessInJob(process, job, ctypes.byref(contained)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not contained.value:
                    raise RuntimeError('Strata listener is not a live member of this worker Job')
            if k.WaitForSingleObject(handle, 0) != 258:
                raise RuntimeError('Strata listener is not a live member of this worker Job')
            return pid
        finally:
            k.CloseHandle(handle)
    finally:
        k.CloseHandle(job)


def complete(_url, messages, schema, *, max_tokens, extra):
    owner = verify_owned_listener(runtime.PORT)
    body = {'model': runtime.request('/v1/models')['data'][0]['id'], 'messages': messages,
            'max_tokens': max_tokens, **extra}
    if schema is not None:
        body['response_format'] = {'type': 'json_schema', 'json_schema':
                                   {'name': 'draft', 'strict': True, 'schema': schema}}
    if schema is not None and 'reasoning_budget_tokens' not in body:
        # Strict structured output is refused (HTTP 502 structured_output_failed) when the length limit cuts the JSON.
        # Strata's default thinking cap is 3/4 of max_tokens, which left too little room for a 100-line function.
        body['reasoning_budget_tokens'] = max_tokens // 2
    try:
        raw = runtime.request('/v1/chat/completions', body, timeout=3600)
    except urllib.error.HTTPError as exc:
        # A generation failure is one attempt, never permission to change settings
        # and silently try the same job again. Preserve the failure for its receipt.
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Strata generation HTTP {exc.code}: {detail}') from exc
    if verify_owned_listener(runtime.PORT) != owner:
        raise RuntimeError('Strata listener ownership changed during request')
    try:
        text = raw['choices'][0]['message'].get('content') or ''
    except (KeyError, IndexError, TypeError):
        text = ''
    return SimpleNamespace(text=text, raw=raw, usage=raw.get('usage', {}))


@contextlib.contextmanager
def serving(_name, **_kwargs):
    verify_owned_worker()
    profile = runtime.active_profile()
    home, model = runtime.paths(profile)
    runtime.select_variant(profile)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', runtime.PORT))
    runtime.start_ready(home, model, profile)
    verify_owned_listener(runtime.PORT)
    # The enclosing gpu_run Job owns cleanup, including failed starts.
    yield runtime.URL + '/v1'


def model():
    return SimpleNamespace(name='strata', request=DEFAULTS)


def run_worker(requests, out, fingerprint=None):
    check_schemas(requests)
    return queue.run_queue(requests, model(), out, serving=serving, complete=complete,
                           identify=(lambda _: fingerprint) if fingerprint is not None else identity,
                           admit=lambda _: None, validate_receipt=validate_receipt, preserve_attempts=True)


def all_cached(requests, out, fingerprint):
    ident = {**fingerprint, 'device': 'gpu'}
    for request in requests:
        key = queue.digest({'identity': ident, 'request': request})
        try:
            receipt = json.loads((out / (key + '.json')).read_text(encoding='utf-8'))
            if not queue.reusable(receipt, ident, request, key):
                return False
            validate_receipt(request, receipt)
        except Exception:
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('requests')
    parser.add_argument('--out', required=True)
    parser.add_argument('--profile', default='default')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cpu:
        parser.error('Strata draft does not support --cpu')
    if args.profile != 'default':
        parser.error('Strata draft supports only --profile default; test-vectors is Ornith-only')
    try:
        requests = queue.read_requests(args.requests)
        check_schemas(requests)
        out = strata_trials.output_directory(Path(args.out))
        runtime.ACTIVE_VARIANT = runtime.preferred_variant()
        if args.worker:
            result = run_worker(requests, out)
            print(json.dumps({'status': result['status'], 'count': len(result['entries']),
                              'cached': result['cached'], 'seconds': result['seconds'],
                              'summary': str(out / 'summary.json')}, ensure_ascii=False))
            return 0 if result['status'] == 'drafts_ready' else 1
        # Quiet mode must be checked before Strata identity walks and hashes
        # model and runtime files, which can be a large HDD workload.
        from dev_quiet_mode import GPUDeferred, require_gpu_allowed
        try:
            require_gpu_allowed()
        except GPUDeferred as exc:
            print(str(exc), file=sys.stderr)
            return 3
        fingerprint = identity(model())
        if all_cached(requests, out, fingerprint):
            # The parent may inspect cache only: any race to a miss must not
            # start the server outside the owned GPU Job.
            @contextlib.contextmanager
            def no_load(*_args, **_kwargs):
                raise RuntimeError('Cache changed during preflight')
                yield
            result = queue.run_queue(requests, model(), out, serving=no_load, complete=complete,
                                     identify=lambda _: fingerprint, admit=lambda _: None,
                                     validate_receipt=validate_receipt, preserve_attempts=True)
            if result['status'] == 'drafts_ready':
                print(json.dumps({'status': result['status'], 'count': len(result['entries']),
                              'cached': result['cached'], 'seconds': result['seconds'],
                              'summary': str(out / 'summary.json')}, ensure_ascii=False))
                return 0
        home, model_path = runtime.paths('coder')
        command = [sys.executable, str(ROOT / 'scripts/gpu_run.py'), '--purpose', 'Strata drafting queue',
                   '--owned-job', '--cap-receipt', str(out / 'cap.json'), '--duty', str(runtime.study_duty()),
                   '--protect', str(home), '--protect', str(model_path.parent), '--protect', str(out), '--',
                   sys.executable, str(Path(__file__).resolve()), str(Path(args.requests).resolve()), '--out', str(out), '--worker']
        command[4:4] = runtime.performance_args()
        return runtime.supervise_trial(command, out, admission=runtime.admission_requirements('coder'))
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
