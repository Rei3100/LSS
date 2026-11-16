"""Pinned, isolated Strata for LSS development (Windows/NVIDIA).

    python scripts/strata_dev.py install
    python scripts/strata_dev.py plan
    python scripts/strata_dev.py smoke --out .cache/strata-smoke.json
    python scripts/strata_dev.py run

Run install through hidden_run.py for unattended acquisition. run/smoke hold the
LSS GPU lease and the recorded process duty (10% default) until the owned server stops. The server
is local-only, text-only, one slot, 8K context. No Product/provider configuration
or user PATH changes. Public downloads are pinned and SHA256 verified; curl is
used because the upstream Python acquisition stalled on this host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root, dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root, dev_evidence_root, require_evidence_writable

import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from storage_tiers import load_policy
import strata_pin

PIN = strata_pin.load()  # .state/strata-pin.json; strata_update.py moves it one signed step at a time
RELEASE = f"https://github.com/Niko1221/Strata/releases/download/{PIN['release_tag']}"
REVISION = '5bc3e238d916f48a861bac2f8a1990a0e9b7e98d'
MODEL_NAME = 'Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
MODEL_HASH = '55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1'
PROFILES = {
    'qwen36': {'family': 'qwen36', 'quant': 'UD-Q4_K_XL', 'repo': 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF',
               'revision': REVISION, 'free_gb': 24, 'files': [(MODEL_NAME, MODEL_HASH, 22853663008)],
               # 22.9 GB file on a 16 GB card: about 8 GB of it sits in VRAM (llama.cpp ncmoe 24 measured 2026-11-05),
               # the rest in RAM; commit covers the mapped file.
               'ram_gb': 14, 'commit_gb': 26},
    'coder': {'family': 'coder', 'quant': 'IQ1_M',
              'repo': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF',
              'revision': '5348543e0147355ac9cbcb031184a3546350988e', 'free_gb': 16,
              'files': [('Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf',
                         'e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad', 29608446496),
                        ('Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf',
                         '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113', 28800138432)]},
    # The general Qwen3.8-Flash-Next (not the Coder fine-tune) for development work that is not code: reading and
    # summarising papers, guides, Evidence and logs, Japanese reports, design and diff review, dsh research jobs.
    # Shard 2 (the n-gram table) is byte-identical to the Coder's; install hard-links it instead of downloading.
    'general': {'family': 'qwen', 'quant': 'IQ2_XS', 'subdir': 'IQ2_XS/',
                'repo': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                'revision': 'ed59f92082b1e93c0e96d60a8b11aab089b52f09', 'free_gb': 16,
                'files': [('Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf',
                           '92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7', 39225954592),
                          ('Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf',
                           '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113', 28800138432)]},
}
FLASH_NEXT = ('coder', 'general')   # the measured hardware-max residency (8 GiB resident, rest streamed) applies to both
ENGINE_NAME = PIN['engine_archive']
ENGINE_HASH = PIN['engine_sha256']
INSTALL_HASH = PIN['install_sha256']
PORT = 18766
URL = f'http://127.0.0.1:{PORT}'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
ACTIVE_VARIANT = 'standard'
QUIET_ARGS = {'--prefill': '512', '--resident-budget-gib': '6', '--vram-reserve-mib': '2048'}
BOUNDED_ARGS = dict(QUIET_ARGS, **{'--resident-budget-gib': '4', '--conversation-cache-mib': '0',
                                 '--conversation-cache-slots': '1', '--prompt-cache': '2'})


def development_policy():
    """Scoped user load authorization; absent policy retains quiet defaults.

    Explicit development state owns the machine policy. Without it, the main
    checkout wins over a worktree's copy, preserving the legacy fallback.
    """
    import strata_ledger
    if 'LSS_DEV_STATE' in os.environ:
        path = dev_state_root(ROOT) / 'strata-development.json'
    else:
        path = dev_state_root(strata_ledger._shared_root(ROOT)) / 'strata-development.json'
        if not path.exists():
            path = dev_state_root(ROOT) / 'strata-development.json'
    policy = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    duty = policy.get('process_duty', .1)
    tag = policy.get('run_tag', 'r1')
    if type(duty) not in (int, float) or not 0 < duty <= 1:
        raise ValueError('Strata process duty must be above zero and at most one')
    if duty > .1 and not str(policy.get('reason', '')).strip():
        raise ValueError('Above-quiet Strata duty requires a recorded user authorization reason')
    if not isinstance(tag, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,48}', tag):
        raise ValueError('Invalid Strata study run tag')
    return dict(policy, process_duty=float(duty), run_tag=tag)


def study_duty():
    return development_policy()['process_duty']


def performance_args():
    return ['--normal-priority'] if study_duty() == 1 else []


def vision_thread_count():
    count = development_policy().get('vision_threads', os.cpu_count() if study_duty() == 1 else 1)
    if type(count) is not int or not 1 <= count <= (os.cpu_count() or 1):
        raise ValueError('Vision CPU threads must fit the available logical processors')
    return count


def study_output(name):
    return (ROOT / f'.cache/local-ai/strata-{name}-20261030-{development_policy()["run_tag"]}').resolve()


LAUNCH_DEFAULT = {'context': '8k', 'kv': 'fp16', 'mtp': 'off'}
CONTEXT_TOKENS = {'8k': 8192, '16k': 16384, '32k': 32768, '48k': 49152, '64k': 65536, '128k': 131072}
PACKS = {'coder': 'coder-iq1_m', 'qwen36': 'qwen36-ud-q4_k_xl', 'general': 'iq2_xs'}


def active_profile():
    """The model Strata serves for drafting: LSS_STRATA_PROFILE, else the policy's `profile`, else the Coder."""
    name = os.environ.get('LSS_STRATA_PROFILE') or development_policy().get('profile') or 'coder'
    if name not in PROFILES:
        raise ValueError(f'Unknown Strata profile {name!r}; choose one of {sorted(PROFILES)}')
    return name


def select_variant(profile):
    """Coder keeps the measured hardware-max variant; other models start with Strata's own placement."""
    global ACTIVE_VARIANT
    ACTIVE_VARIANT = preferred_variant() if profile in FLASH_NEXT and development_policy().get('preferred_variant') else 'standard'
    return ACTIVE_VARIANT


def launch_settings(profile='coder'):
    """Context, KV format and MTP choice. Coder on hardware-max reads the policy `launch`; any profile may be
    overridden by the policy's `launch_profiles[profile]` (for example Qwen3.6 at 128k with its MTP layer)."""
    chosen = dict(LAUNCH_DEFAULT, **(development_policy().get('launch') or {})) if ACTIVE_VARIANT == 'hardware-max' else dict(LAUNCH_DEFAULT)
    if ACTIVE_VARIANT not in ('quiet-exact', 'quiet-bounded'):   # the quiet study variants stay at their fixed 8k
        chosen.update((development_policy().get('launch_profiles') or {}).get(profile) or {})
    if chosen['context'] not in CONTEXT_TOKENS or chosen['kv'] not in ('fp16', 'int8', 'q4_0', 'k8v4') or chosen['mtp'] not in ('on', 'off'):
        raise ValueError('Invalid Strata launch settings')
    return chosen


def profile_tag(profile='coder'):
    return f"{PROFILES[profile]['quant'].lower()}-{launch_settings(profile)['context']}"


def variant_args():
    if ACTIVE_VARIANT == 'hardware-max':
        supplied = development_policy().get('hardware_args', {})
        allowed = {'--prefill', '--resident-budget-gib', '--vram-reserve-mib', '--pool-workers'}
        if not supplied or set(supplied) - allowed:
            raise ValueError('Hardware tuning may change only measured residency, reserve, prefill and worker settings')
        for key, value in supplied.items():
            if key == '--prefill' and value == 'auto':
                continue
            if not isinstance(value, str) or not value.isdecimal() or int(value) > 32768:
                raise ValueError('Invalid measured hardware tuning value')
        return dict(BOUNDED_ARGS, **supplied)
    return {'quiet-exact': QUIET_ARGS, 'quiet-bounded': BOUNDED_ARGS}.get(ACTIVE_VARIANT, {})


def preferred_variant():
    variant = development_policy().get('preferred_variant', 'quiet-bounded')
    if variant not in ('quiet-bounded', 'quiet-exact', 'hardware-max'):
        raise ValueError('Invalid preferred Strata development variant')
    return variant


def study_admission(vision=False):
    policy = development_policy()
    field = 'hardware_vision_args' if vision else 'hardware_args'
    resident = int(policy.get(field, policy.get('hardware_args', {})).get('--resident-budget-gib', '4'))
    if study_duty() == 1:
        return resident * (2**30 / 1e9) + (6 if vision else 4), resident * (2**30 / 1e9) + (22 if vision else 20)
    return resident + (8 if vision else 6), resident + (24 if vision else 22)


def state_root(home):
    return home if ACTIVE_VARIANT == 'standard' else home / 'variants' / ACTIVE_VARIANT


def admission_requirements(profile):
    if profile in FLASH_NEXT and ACTIVE_VARIANT == 'hardware-max':
        return study_admission()
    if profile == 'coder' and ACTIVE_VARIANT == 'quiet-bounded':
        return 10, 26
    if profile == 'coder' and ACTIVE_VARIANT == 'quiet-exact':
        # Fixed 6 GiB CPU expert budget plus buffers; commit also covers WDDM.
        return 12, 28
    minimum = PROFILES[profile]['free_gb']
    return PROFILES[profile].get('ram_gb', minimum), PROFILES[profile].get('commit_gb', minimum)


def paths(profile='coder'):
    policy = load_policy()
    home = Path(policy['llama_runtime_root']).parent / 'strata'
    model = Path(policy['local_ai_hot_models']) / 'strata' / profile / PROFILES[profile]['files'][0][0]
    return home, model


def environment(home):
    env = os.environ.copy()
    state = state_root(home)
    env.update(STRATA_DATA_DIR=str(home), STRATA_INSTALL_DIR=str(home / 'install'),
               STRATA_COMPONENTS_DIR=str(home / 'components'),
               STRATA_PROFILE_DIR=str(state / 'profiles'), STRATA_RUNTIME_DIR=str(state / 'runtime'),
               LOCALAPPDATA=str(state / 'user-state'), STRATA_ROUTE_TAIL_SKIP='0',
               STRATA_UPDATE_FEED=RELEASE, STRATA_NO_PATH='1', PYTHONUTF8='1',
               PYTHONNOUSERSITE='1', PYTHONHOME='', PYTHONPATH='')
    return env


def cli(home, *args, check=True):
    terminal = home / 'terminal' / 'current'
    return subprocess.run([str(terminal / 'python' / 'python.exe'), str(terminal / 'strata.py'), *args],
                          env=environment(home), creationflags=FLAGS, check=check,
                          stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr)


RECOVER_STALE = ('import json\nfrom strata_control import runtime\n'
                 'stale = [i["profile"] for i in runtime.active_managed_instances() if i.get("state") == "stale"]\n'
                 'print(json.dumps({name: runtime._recover_stale(name, wait=20) for name in stale}))')


def recover_stale(home):
    """Clear owner records of launches whose host died: `strata update` counts a stale record as serving
    ("Strata's own launch is using the graphics card") and `launch --stop` leaves it (seen with Terminal 0.2.7/0.2.9)."""
    terminal = home / 'terminal' / 'current'
    return subprocess.run([str(terminal / 'python' / 'python.exe'), '-c', RECOVER_STALE], cwd=str(terminal),
                          env=environment(home), creationflags=FLAGS, check=False, stdin=subprocess.DEVNULL).returncode


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


HF_RESOLVE = re.compile(r'^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[0-9a-f]{40})/(?P<file>.+)$')


def hf_request(url):
    """(repo, revision, file) when ``url`` is a pinned Hugging Face file URL, else None."""
    found = HF_RESOLVE.match(url)
    return (found['repo'], found['rev'], found['file']) if found else None


def _download_with_hf(url, partial):
    """Fetch a pinned Hugging Face file with the ``hf`` CLI (parallel, resumable Xet transfer) into ``partial``.

    False when the URL is not a pinned HF file or ``hf`` is not installed: the caller falls back to curl.
    The caller verifies SHA-256, so a transport difference never changes what is accepted."""
    request = hf_request(url)
    hf = shutil.which('hf')
    if request is None or hf is None:
        return False
    repo, revision, name = request
    staging = partial.parent / (partial.name + '.hf')
    environment = dict(os.environ, HF_XET_HIGH_PERFORMANCE='1', HF_HUB_DISABLE_TELEMETRY='1')
    subprocess.run([hf, 'download', repo, name, '--revision', revision, '--local-dir', str(staging)],
                   env=environment, creationflags=FLAGS, check=True, stdin=subprocess.DEVNULL,
                   stdout=sys.stdout, stderr=sys.stderr)
    fetched = staging.joinpath(*name.split('/'))
    if not fetched.is_file():
        raise RuntimeError(f'hf download did not produce {fetched}')
    partial.unlink(missing_ok=True)
    fetched.replace(partial)
    shutil.rmtree(staging, ignore_errors=True)
    return True


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != expected:
            raise RuntimeError(f'Existing file has unexpected SHA256; retained: {path}')
        return
    partial = path.with_name(path.name + '.partial')
    if _download_with_hf(url, partial):
        actual = sha256(partial)
        if actual != expected:
            raise RuntimeError(f'SHA256 mismatch for {partial}: {actual}')
        partial.replace(path)
        print(f'Verified {path.name}: {expected}', flush=True)
        return
    subprocess.run(['curl.exe', '--fail', '--location', '--retry', '4', '--connect-timeout', '30',
                    '--speed-limit', '1024', '--speed-time', '120', '--continue-at', '-',
                    '--output', str(partial), url], creationflags=FLAGS, check=True,
                   stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr)
    actual = sha256(partial)
    if actual != expected:
        raise RuntimeError(f'SHA256 mismatch for {partial}: {actual}')
    partial.replace(path)
    print(f'Verified {path.name}: {expected}', flush=True)


def install(home, model, profile):
    from dev_hot_paths import require_hot
    require_hot(home, 'Strata runtime')
    require_hot(model.parent, 'Strata model')
    downloads = home / 'downloads'
    installer = downloads / 'install.ps1'
    download(f'{RELEASE}/install.ps1', installer, INSTALL_HASH)
    if not (home / 'terminal' / 'current' / 'strata.py').exists():
        subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(installer)],
                       env=environment(home), creationflags=FLAGS, check=True,
                       stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr)
    engine = downloads / ENGINE_NAME
    download(f'{RELEASE}/{ENGINE_NAME}', engine, ENGINE_HASH)
    cli(home, 'components', 'install-file', str(engine), '--feed', RELEASE, '--output', 'json')
    selected = PROFILES[profile]
    for filename, digest, _size in selected['files']:
        subdir = 'IQ1_M/' if profile == 'coder' else selected.get('subdir', '')
        target = model.parent / filename
        twin = paths('coder')[1].parent / filename.replace(selected['quant'], 'IQ1_M')
        if not target.exists() and filename.endswith('-00002-of-00002.gguf') and twin.exists()                 and twin.stat().st_size == _size:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(twin, target)   # same n-gram table bytes; the SHA256 check below still runs
        download(f"https://huggingface.co/{selected['repo']}/resolve/{selected['revision']}/{subdir}{filename}",
                 model.parent / filename, digest)
    receipt = {'release': PIN['release_tag'], 'control_build': PIN['control'], 'engine': ENGINE_NAME,
               'engine_sha256': ENGINE_HASH, 'model': str(model), 'profile': profile,
               'model_files': selected['files'], 'model_revision': selected['revision'],
               'scope': 'development only', 'installed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (home / f'lss-install-{profile}.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    cli(home, '--version')


def launch_args(model, profile='coder'):
    selected = PROFILES[profile]
    args = ['launch', '--family', selected['family'], '--model', selected['quant'], '--model-path', str(model),
            '--context', launch_settings(profile)['context'], '--kv', launch_settings(profile)['kv'], '--parallel', '1', '--vision', 'off',
            '--access', 'local', '--port', str(PORT), '--tune', 'skip', '--yes', '--no-tui', '--mtp', launch_settings(profile)['mtp']]
    for key, value in variant_args().items():
        args += ['--set', f'{key[2:]}={value}']
    for key, value in profile_engine_args(profile).items():
        args += ['--set', f'{key[2:]}={value}']
    return args


def profile_engine_args(profile):
    """Per-model engine settings from the policy's `engine_profiles[profile]` (e.g. Qwen3.6 `--prefill 512`, the chunk
    whose text equals the base engine's and needs the smallest host staging)."""
    chosen = (development_policy().get('engine_profiles') or {}).get(profile) or {}
    allowed = {'--prefill', '--resident-budget-gib', '--vram-reserve-mib', '--pool-workers'}
    if set(chosen) - allowed or not all(isinstance(v, str) and v.isdecimal() for v in chosen.values()):
        raise ValueError('engine_profiles may set only measured residency, reserve, prefill and worker settings')
    return chosen


def request(route, body=None, *, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + route, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def prepared_profile(home, model, profile):
    """Only reuse the fixed, private launch we generated for these exact files."""
    tag = profile_tag(profile)
    wanted = launch_settings(profile)
    name = f'launch-{tag}'
    state = state_root(home)
    source = state / 'user-state' / 'Strata' / 'launch' / f'{tag}.json'
    try:
        saved = json.loads((state / 'profiles' / f'{name}.json').read_text(encoding='utf-8'))
        cfg = json.loads(source.read_text(encoding='utf-8'))
        args = cfg['args']
        native = args[args.index('--native') + 1]
        overrides = saved['overrides']
        effective = {key: overrides.get(key, cfg.get(key)) for key in ('host', 'parallel', 'port')}
        matches = (Path(saved['source_config']).resolve() == source.resolve()
                   and not set(overrides) - {'host', 'parallel', 'port'}
                   and effective == {'host': '127.0.0.1', 'parallel': 1, 'port': PORT}
                   and Path(native).resolve() == model.resolve()
                   and args[args.index('--max-context') + 1] == str(CONTEXT_TOKENS[wanted['context']])
                   and cfg.get('mtp') is (wanted['mtp'] == 'on')
                   and (args[args.index('--kv') + 1] if '--kv' in args else 'fp16') == wanted['kv'])
        wanted_args = dict(variant_args(), **profile_engine_args(profile))
        if wanted_args:
            matches = matches and all(flag in args and args[args.index(flag) + 1] == value
                                      for flag, value in wanted_args.items())
        return name if matches else None
    except (OSError, ValueError, KeyError, IndexError):
        return None


def start_ready(home, model, profile):
    name = prepared_profile(home, model, profile)
    if name is None:
        # launch prepares/imports the model and writes the official private profile.
        # Its synchronous control RPC has a 5 s timeout, which can fail under the
        # process-tree duty cap. Reuse only a fully validated resulting profile.
        result = cli(home, *launch_args(model, profile), check=False)
        if result.returncode == 0:
            return
        name = prepared_profile(home, model, profile)
        if result.returncode != 6 or name is None:
            raise RuntimeError(f'Strata preparation failed (exit {result.returncode})')
    cli(home, 'runtime', 'start', '--profile', name, '--wait', '0')
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        try:
            request('/health')
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError('Strata did not become healthy within 1200 seconds; inspect the private runtime logs')


def worker(home, model, args):
    from gpu_run import memory_headroom
    commit, ram = memory_headroom()
    minimum_ram, minimum_commit = admission_requirements(args.profile)
    # Q36 retains every expert on CPU; Coder's low-RAM route splits CPU/GPU.
    if ram < minimum_ram or commit < minimum_commit:
        raise RuntimeError(f'Strata {args.profile} needs {minimum_ram} GB free RAM and {minimum_commit} GB commit headroom; available RAM={ram:.2f}, commit={commit:.2f} GB')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))  # refuse an occupied port before acquiring ownership
    started = False
    try:
        started = True  # failed launch may also have started a runtime host
        start_ready(home, model, args.profile)
        models = request('/v1/models')
        if args.command == 'trial':
            import strata_trials
            strata_trials.run(models['data'][0]['id'], args.out, variant=ACTIVE_VARIANT, duty=args.duty)
        elif args.command == 'smoke':
            results = []
            cases = [
                ('json', 'Return only this JSON object, with no explanation: {"status":"ok","value":42}',
                 lambda content: json.loads(content) == {'status': 'ok', 'value': 42}),
                ('identifier', 'Output exactly this identifier, unchanged, and nothing else: LSS_CurrentBest_cue_001',
                 lambda content: content == 'LSS_CurrentBest_cue_001'),
            ]
            for name, prompt, oracle in cases:
                then = time.monotonic()
                reply = request('/v1/chat/completions', {'model': models['data'][0]['id'],
                                'messages': [{'role': 'user', 'content': prompt}],
                                'max_tokens': 256, 'temperature': 0,
                                'chat_template_kwargs': {'enable_thinking': False}})
                content = reply['choices'][0]['message'].get('content', '').strip()
                try:
                    passed = oracle(content)
                except (ValueError, TypeError):
                    passed = False
                results.append({'case': name, 'passed': passed, 'seconds': time.monotonic() - then, 'reply': reply})
            report = {'health': request('/health'), 'models': models, 'cases': results,
                      'passed': all(row['passed'] for row in results), 'scope': 'API smoke only; no coding-quality acceptance'}
            require_evidence_writable(args.out)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            if not report['passed']:
                raise RuntimeError(f'Smoke failed: {args.out}')
            if args.bench:
                bench = dev_evidence_root(ROOT) / 'release-evidence/strata-capability-20261031/bench'
                log = state_root(home) / 'user-state/Strata/launch' / f'{profile_tag(args.profile)}.engine.log'
                command = [sys.executable, str(bench / 'run_bench.py'), '--label', args.bench,
                           '--ctx', str(CONTEXT_TOKENS[launch_settings(args.profile)['context']]), '--engine-log', str(log),
                           '--out', str(args.out.parent / 'bench')]
                if args.bench_only:
                    command += ['--only', args.bench_only]
                if subprocess.run(command, creationflags=FLAGS, stdin=subprocess.DEVNULL).returncode:
                    raise RuntimeError('Bench run failed')
        else:
            print(f'Strata ready at {URL}/v1; Ctrl+C stops this owned instance.', flush=True)
            while True:
                time.sleep(10)
                request('/health')
    finally:
        if started:
            cli(home, 'launch', '--stop', check=True)


def supervise_trial(command, output, *, admission=None):
    """Monitor outside the GPU duty cap; abort only this new tree on pressure."""
    import strata_trials
    from gpu_run import memory_headroom
    out = strata_trials.output_directory(output)
    floor = development_policy().get('memory_floor_gb', 2.5)
    if type(floor) not in (int, float) or not .5 <= floor <= 4:
        raise ValueError('Development memory floor must be between 0.5 and 4 GB')
    result = {'status': 'running', 'minimum_ram_gb': None, 'minimum_commit_gb': None,
              'memory_floor_gb': floor, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    process = None
    try:
        with (out / 'resources.jsonl').open('a', encoding='utf-8') as stream:
            minimum_ram, minimum_commit = admission or admission_requirements('coder')
            # Wait outside gpu_run: no GPU lease or model exists during admission.
            while True:
                commit, ram = memory_headroom()
                if ram >= minimum_ram and commit >= minimum_commit:
                    break
                result.update(status='waiting-for-admission', required_ram_gb=minimum_ram,
                              required_commit_gb=minimum_commit, available_ram_gb=ram, commit_headroom_gb=commit)
                strata_trials.atomic(out / 'resources-summary.json', result)
                time.sleep(15)
            result['status'] = 'running'
            strata_trials.atomic(out / 'resources-summary.json', result)
            process = subprocess.Popen(command, creationflags=FLAGS, stdin=subprocess.DEVNULL,
                                       stdout=sys.stdout, stderr=sys.stderr)
            while process.poll() is None:
                commit, ram = memory_headroom()
                sample = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'available_ram_gb': ram, 'commit_headroom_gb': commit}
                stream.write(json.dumps(sample) + '\n')
                stream.flush()
                result['minimum_ram_gb'] = min(ram, result['minimum_ram_gb'] or ram)
                result['minimum_commit_gb'] = min(commit, result['minimum_commit_gb'] or commit)
                if min(commit, ram) < result['memory_floor_gb']:
                    result.update(status='aborted-memory-pressure', last_sample=sample)
                    # process is our newly created gpu_run tree, never an attached/user process.
                    subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                                   creationflags=FLAGS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False)
                    process.wait(timeout=30)
                    return 3
                time.sleep(5)
        result.update(status='completed' if process.returncode == 0 else 'failed', returncode=process.returncode)
        return process.returncode
    finally:
        if process is not None and process.poll() is None:
            if result['status'] == 'running':
                result['status'] = 'interrupted'
            subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'], creationflags=FLAGS,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            process.wait(timeout=30)
        result['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        strata_trials.atomic(out / 'resources-summary.json', result)


def main():
    global ACTIVE_VARIANT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['install', 'plan', 'status', 'stop', 'run', 'smoke', 'trial'])
    parser.add_argument('--out', type=Path, default=ROOT / '.cache' / 'strata-smoke.json')
    parser.add_argument('--profile', choices=PROFILES, default='coder')
    parser.add_argument('--bench', help='after the smoke cases, run the generic coding bench under this label (results beside --out)')
    parser.add_argument('--bench-only', help='comma-separated bench task ids')
    parser.add_argument('--duty', type=float, help='process duty within the recorded development load authorization')
    parser.add_argument('--variant', choices=['standard', 'quiet-exact', 'quiet-bounded', 'hardware-max'])
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    ACTIVE_VARIANT = args.variant or (preferred_variant() if args.profile in FLASH_NEXT and development_policy().get('preferred_variant')
                                     else ('quiet-exact' if args.command == 'trial' else 'standard'))
    allowed_duty = study_duty()
    args.duty = allowed_duty if args.duty is None else args.duty
    if not 0 < args.duty <= allowed_duty:
        parser.error('--duty exceeds the recorded development load authorization')
    if args.command == 'trial' and args.profile != 'coder':
        parser.error('the frozen trial matrix currently uses the Coder profile')
    home, model = paths(args.profile)
    if args.command == 'install':
        import storage_budget
        if not os.environ.get(storage_budget.TOKEN_ENV):
            managed = storage_budget.command([sys.executable, str(Path(__file__).resolve()), 'install',
                                              '--profile', args.profile], required_gb=90 if args.profile == 'coder' else 27,
                                              paths=[str(home), str(model.parent)], purpose='Strata installation')
            return subprocess.run(managed, creationflags=FLAGS, stdin=subprocess.DEVNULL,
                                  stdout=sys.stdout, stderr=sys.stderr).returncode
        install(home, model, args.profile)
    elif args.command == 'stop':
        cli(home, 'launch', '--stop')
        recover_stale(home)
    elif args.command == 'status':
        from gpu_run import memory_headroom
        commit, ram = memory_headroom()
        files = []
        for name, _digest, size in PROFILES[args.profile]['files']:
            path = model.parent / name
            partial = path.with_name(path.name + '.partial')
            present = path if path.exists() else partial
            files.append({'path': str(path), 'complete': path.exists(), 'expected_bytes': size,
                          'present_bytes': present.stat().st_size if present.exists() else 0})
        pack_progress = None
        if args.profile == 'coder':
            pack = home / 'Strata-data' / 'packs' / 'coder-iq1_m'
            note = pack / 'experts.bin.tmp.json'
            if note.exists():
                try:
                    progress = json.loads(note.read_text(encoding='utf-8'))
                    pack_progress = {'completed_layers': progress['layers'],
                                     'expected_bytes': progress['want']['bytes'],
                                     'complete': (pack / 'experts.bin').exists()}
                except (OSError, ValueError, KeyError):
                    pack_progress = {'state': 'updating'}
        print(json.dumps({'profile': args.profile, 'runtime': str(home), 'model_files': files,
                          'pack': pack_progress, 'available_ram_gb': ram, 'commit_headroom_gb': commit}, indent=2), flush=True)
        cli(home, 'launch', '--status', '--output', 'json')
    elif args.command == 'plan':
        cli(home, *launch_args(model, args.profile), '--plan', '--output', 'json')
    elif args.worker:
        worker(home, model, args)
    else:
        for filename, _digest, size in PROFILES[args.profile]['files']:
            path = model.parent / filename
            if not path.exists() or path.stat().st_size != size:
                raise RuntimeError('Install the pinned Strata model first: python scripts/strata_dev.py install')
        from dev_hot_paths import require_hot
        require_hot(home, 'Strata runtime')
        require_hot(model, 'Strata model')
        command = [sys.executable, str(ROOT / 'scripts' / 'gpu_run.py'), '--purpose', 'Strata development', '--owned-job',
                                '--duty', str(args.duty), '--protect', str(home), '--protect', str(model.parent), '--',
                                sys.executable, str(Path(__file__).resolve()), args.command, '--worker',
                                '--profile', args.profile, '--out', str(args.out), '--duty', str(args.duty),
                                '--variant', ACTIVE_VARIANT]
        if args.bench:
            command += ['--bench', args.bench] + (['--bench-only', args.bench_only] if args.bench_only else [])
        command[4:4] = performance_args()
        if args.command == 'trial':
            command[4:4] = ['--cap-receipt', str(args.out / 'cap.json')]
            return supervise_trial(command, args.out)
        return subprocess.call(command,
                               creationflags=FLAGS, stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr)
    return 0


if __name__ == '__main__':
    # Imported trial helpers must share this process's selected variant.
    sys.modules['strata_dev'] = sys.modules[__name__]
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'strata_dev: {exc}', file=sys.stderr)
        raise SystemExit(1)
