"""CPU-only historical audio frontend repair oracle; no inference or Product imports.

--check writes controls and an answer-free request. Candidate source is validated
before execution in a timed, 1 GiB child. Only a generated FLOAT WAV is readable.
The original historical test executes unchanged; the other stereo tests need
Torch or other functions and are explicitly outside this target's coverage.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import argparse
import ast
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "f567482a9d2366ee4c84c93ab7e21e4d34d1d5e7"
PRODUCT = "lss/bgm/event_tagging.py"
TEST = "tests/test_event_tagging_stereo.py"
TEST_NAME = "test_legacy_input_and_stereo_timeline"
TARGET_ID = "fix-f567482a-model_input"
MEMORY_BYTES = 1024 ** 3
CONSTANTS = {'INPUT_RATE': 44100, 'STEREO_POWER': 'mean_channel_spectral_power_v1'}


def git_show(spec):
    return subprocess.check_output(['git', 'show', spec], cwd=ROOT).decode('utf-8')


def definition(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]) + '\n'


def sha(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def load_target():
    old, new = (git_show(f'{rev}:{PRODUCT}') for rev in (COMMIT + '^', COMMIT))
    test_module = git_show(f'{COMMIT}:{TEST}')
    # Frozen immutable dependencies are verified against the pinned source, not HEAD.
    assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(new).body
                   if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in CONSTANTS}
    if assignments != CONSTANTS:
        raise ValueError('historical dependency mismatch')
    exception = definition(new, 'EventTaggingError')
    if exception != definition(old, 'EventTaggingError'):
        raise ValueError('exception dependency changed')
    return dict(before=definition(old, 'model_input'), after=definition(new, 'model_input'),
                test=definition(test_module, TEST_NAME), exception=exception,
                provenance={'commit': COMMIT, 'product_path': PRODUCT, 'test_path': TEST,
                            'test_name': TEST_NAME, 'test_module_sha256': sha(test_module),
                            'dependency_constants': CONSTANTS, 'exception_sha256': sha(exception)})


NODES = {ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.Name,
         ast.Load, ast.Store, ast.Constant, ast.Call, ast.Attribute, ast.keyword,
         ast.Tuple, ast.Return, ast.If, ast.Compare, ast.NotEq, ast.Eq, ast.BoolOp,
         ast.Or, ast.UnaryOp, ast.Not, ast.Raise, ast.JoinedStr, ast.FormattedValue,
         ast.Subscript, ast.Expr}
IMPORTS = {'import numpy as np', 'import soundfile as sf', 'from scipy.signal import resample_poly'}


def validate_source(source):
    """Closed syntax/call surface, no reflection, mutation, file APIs or imports.

    The three historical imports are declarations replaced with trusted bindings.
    All assignments bind locals only; globals and parameter names cannot be rebound.
    """
    if not isinstance(source, str) or len(source) > 12000:
        raise ValueError('invalid source size')
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('exactly one function required')
    fn = tree.body[0]
    if fn.name != 'model_input' or fn.decorator_list or fn.args.defaults or fn.args.vararg or fn.args.kwarg:
        raise ValueError('unexpected function declaration')
    if [a.arg for a in fn.args.args] != ['signal_path'] or fn.args.posonlyargs:
        raise ValueError('unexpected positional signature')
    if [a.arg for a in fn.args.kwonlyargs] not in ([], ['channel_policy']):
        raise ValueError('unexpected keyword signature')
    if any(not isinstance(n, ast.Constant) or n.value != 'mean_of_channels' for n in fn.args.kw_defaults):
        raise ValueError('unexpected default')
    fn.returns = None
    for arg in ast.walk(fn.args):
        if isinstance(arg, ast.arg):
            arg.annotation = None
    body = []
    for n in fn.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if ast.unparse(n) not in IMPORTS:
                raise ValueError('unsupported import')
        else:
            body.append(n)
    fn.body = body
    reserved = {'np', 'sf', 'resample_poly', 'EventTaggingError', 'INPUT_RATE', 'STEREO_POWER', 'int', 'str'}
    for n in ast.walk(tree):
        if type(n) not in NODES:
            raise ValueError('unsupported syntax: ' + type(n).__name__)
        if isinstance(n, (ast.Name, ast.arg)) and '__' in (n.id if isinstance(n, ast.Name) else n.arg):
            raise ValueError('reflection name')
        if isinstance(n, ast.Constant) and (isinstance(n.value, (str, bytes)) and len(n.value) > 1000
                                           or isinstance(n.value, (int, float)) and abs(n.value) > 1000000):
            raise ValueError('oversized constant')
        if isinstance(n, ast.Attribute) and n.attr not in {'read', 'isfinite', 'float32', 'shape', 'all', 'mean', 'astype', 'T'}:
            raise ValueError('unsupported attribute')
        if isinstance(n, ast.Assign):
            targets = [e for t in n.targets for e in (t.elts if isinstance(t, ast.Tuple) else [t])]
            if any(not isinstance(t, ast.Name) or t.id in reserved for t in targets):
                raise ValueError('only local assignment')
        if isinstance(n, ast.Call):
            f = n.func
            direct = isinstance(f, ast.Name) and f.id in {'int', 'str', 'resample_poly', 'EventTaggingError'}
            bound = isinstance(f, ast.Attribute) and f.attr in {'all', 'mean', 'astype'}
            api = isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and (f.value.id, f.attr) in {('sf', 'read'), ('np', 'isfinite')}
            if not (direct or bound or api):
                raise ValueError('unsupported call')
    return tree


def _memory_limit():
    if os.name == 'nt':
        import ctypes
        from local_llm import Job, _ExtendedLimits
        job = Job()
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100
        limits.ProcessMemoryLimit = MEMORY_BYTES
        if not job._k.SetInformationJobObject(job._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise OSError('memory limit failed')
        job.assign(os.getpid())
        return job  # Keep the handle alive until child exits.
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))


def _child(source, target, sender):
    try:
        job = _memory_limit()
        for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[key] = '1'
        import numpy as np
        import soundfile as sf
        import scipy
        from scipy.signal import resample_poly
        import pytest
        tree = validate_source(source)
        with tempfile.TemporaryDirectory(prefix='audio-oracle-', dir=ROOT / '.cache') as folder:
            fixture = Path(folder) / 'input.wav'
            def read(path, **kwargs):
                if not isinstance(path, (str, Path)) or str(path) != str(fixture) or kwargs != {'dtype': 'float32', 'always_2d': True}:
                    raise ValueError('only exact oracle fixture read is allowed')
                return sf.read(fixture, **kwargs)
            def numpy_internal_import(name, globals=None, locals=None, fromlist=(), level=0):
                # NumPy's ndarray methods use the caller's import hook on some
                # versions. No imports are permitted in candidate bytecode.
                if name not in {'numpy._core._methods', 'numpy.core._methods'} or level:
                    raise ValueError('unsupported internal import: ' + name)
                import builtins
                return builtins.__import__(name, globals, locals, fromlist, level)
            scope = {'__builtins__': {'int': int, 'str': str, '__import__': numpy_internal_import}, **CONSTANTS,
                     'np': SimpleNamespace(isfinite=np.isfinite, float32=np.float32),
                     'sf': SimpleNamespace(read=read), 'resample_poly': resample_poly}
            trusted = {'RuntimeError': RuntimeError, '__name__': __name__}
            exec(target['exception'], trusted)
            scope['EventTaggingError'] = trusted['EventTaggingError']
            exec(compile(tree, '<bounded-audio-candidate>', 'exec'), scope)
            tests = {'np': np, 'sf': sf, 'pytest': pytest,
                     'et': SimpleNamespace(model_input=scope['model_input'], **CONSTANTS,
                                           EventTaggingError=trusted['EventTaggingError'])}
            exec(compile(target['test'], '<unchanged-historical-test>', 'exec'), tests)
            tests[TEST_NAME](Path(folder))
        sender.send({'pass': True, 'test': TEST_NAME,
                     'dependencies': {'numpy': np.__version__, 'soundfile': sf.__version__, 'scipy': scipy.__version__},
                     'torch_imported': 'torch' in sys.modules})
    except BaseException as exc:
        import traceback
        sender.send({'pass': False, 'error': type(exc).__name__ + ': ' + str(exc)[:1500],
                     'traceback': traceback.format_exc(limit=8)})
    finally:
        sender.close()


def evaluate_code(source, timeout=20):
    try:
        validate_source(source)
    except (ValueError, SyntaxError) as exc:
        return {'pass': False, 'error': str(exc), 'stage': 'validation'}
    # This utility reads Windows memory counters only; it does not acquire a GPU
    # lease, import Torch or start a GPU job. Reserve 4 GB after the full child cap.
    from gpu_run import memory_headroom
    commit_gb, ram_gb = memory_headroom()
    required_gb = 4 + MEMORY_BYTES / 1e9
    if min(commit_gb, ram_gb) < required_gb:
        return {'pass': False, 'error': 'insufficient memory headroom', 'stage': 'admission',
                'commit_gb': commit_gb, 'ram_gb': ram_gb, 'required_gb': required_gb}
    (ROOT / '.cache').mkdir(exist_ok=True)
    ctx = multiprocessing.get_context('spawn')
    receiver, sender = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_child, args=(source, load_target(), sender), daemon=True)
    child.start()
    sender.close()
    try:
        if receiver.poll(timeout):
            try:
                return receiver.recv()
            except EOFError:
                return {'pass': False, 'error': 'child exited without result'}
        return {'pass': False, 'error': 'timeout'}
    finally:
        if child.is_alive():
            child.terminate()
        child.join(3)
        receiver.close()


def generate_requests():
    target = load_target()
    return [{'id': TARGET_ID, 'provenance': {'commit': COMMIT, 'path': PRODUCT},
             'prompt': 'Repair only model_input. Return a JSON object with one code string. '
                       'The frontend needs an optional channel_policy: mean_of_channels retains historical '
                       'mono behavior; the STEREO_POWER policy preserves both channels at the model rate. '
                       'Reject unsupported policies. Retain validation and original-frame count. '
                       'Trusted constants: ' + json.dumps(CONSTANTS) + '. Only numpy, soundfile read '
                       'and scipy.signal.resample_poly imports are supported; files, network, reflection, '
                       'processes and other functions are unavailable. Hidden historical tests are withheld.\n'
                       + target['before']}]


def check(out):
    target = load_target()
    before, after = evaluate_code(target['before']), evaluate_code(target['after'])
    report = {'schema': 1, 'id': TARGET_ID, 'provenance': target['provenance'],
              'executor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python': {'executable': sys.executable, 'version': sys.version},
              'before_sha256': sha(target['before']), 'after_sha256': sha(target['after']),
              'test_function_sha256': sha(target['test']), 'before': before, 'after': after,
              'eligible_functional_targets_added': int(not before['pass'] and after['pass']),
              'limits': {'memory_bytes': MEMORY_BYTES, 'timeout_seconds': 20, 'threads': 1},
              'scope': 'one exact unchanged historical frontend test; excludes CNN, guarded dispatch and learned output',
              'model_runs': 0, 'generated_at_epoch': time.time()}
    require_evidence_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'requests.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in generate_requests()), encoding='utf-8')
    (out / 'original-test.py').write_text(target['test'], encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.add_argument('--out', type=Path, default=dev_evidence_root(ROOT) / 'release-evidence/local-ai-challengers-r1/audio-repair')
    args = parser.parse_args()
    result = check(args.out)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['eligible_functional_targets_added'] == 1 else 1)
