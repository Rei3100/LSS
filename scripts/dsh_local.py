r"""Run DeepSeek Harness (dsh) headless against the local Strata server, judged by a test command.

    python scripts/dsh_local.py setup
    python scripts/dsh_local.py run --workspace C:/Users/<you>/<dir> --judge "<cmd that exits 0 when right>" "<task text>"

dsh is the agent loop (reads and edits files, runs commands); Strata (``strata_dev.py run``, 32k context, port
18766) is the model. This is a second kind of delegated work, ``harness``, next to ``strata_task.py``'s one-function
drafts: the ledger (``strata_ledger.py gate harness``) decides whether it pays. The judge, not dsh's own report,
decides PASS. dsh files live in ``<LSS_WORK_ROOT>/local-ai-runtimes/dsh`` (own ``DSH_HOME``, telemetry and the
DeepSeek account/search plugins disabled); nothing of the image-generation setup is touched.

The workspace must sit under the user profile: dsh's Windows sandbox cannot grant itself write access to a folder
outside it (``SetNamedSecurityInfoW failed (Win32 5)``) and then refuses every shell command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get('LSS_WORK_ROOT') or Path.home() / '.lss-work') / 'local-ai-runtimes' / 'dsh'
BIN = HOME / 'node_modules' / '@deepseek-ai' / 'dsh' / 'lib' / 'bin.js'
VERSION = '0.2.1-rc.5'
PORT = 18766
DISABLED = ('account-controller', 'session-telemetry-otel', 'desktop-product-telemetry', 'deepseek-account',
            'llm-deepseek', 'llm-deepseek-account', 'web-search-deepseek', 'web-deepseek-api')


def patch(model_id, workspace, context=32768, max_tokens=8192):
    lines = []
    for ident in DISABLED:
        lines += [f'- id: {ident}', '  disabled: true']
    lines += ['- id: llm-pi-ai', '  config:', '    providers:', '      lss-strata:',
              '        displayName: Strata (LSS development)', '        api: openai-completions',
              f'        baseURL: http://127.0.0.1:{PORT}/v1', '        apiKeyEnv: LSS_STRATA_API_KEY',
              '        models:', f'        - id: {model_id}', f'          name: Strata {model_id}',
              f'          contextWindow: {context}', f'          maxTokens: {max_tokens}', '          input:', '          - text',
              '- id: agent-default-model', '  config:', '    provider: lss-strata', f'    model: {model_id}',
              '- id: sandbox-policy', '  config:', '    mode: workspace-write',
              f'    workspaceRoot: {Path(workspace).resolve().as_posix()}']
    return '\n'.join(lines) + '\n'


def health():
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as response:
        return json.load(response)


def check_workspace(workspace):
    resolved = Path(workspace).resolve()
    if Path.home().resolve() not in resolved.parents:
        raise ValueError(f'workspace must be under {Path.home()} (dsh sandbox limit): {resolved}')
    return resolved


def setup():
    HOME.mkdir(parents=True, exist_ok=True)
    npm = 'npm.cmd' if os.name == 'nt' else 'npm'
    subprocess.run([npm, 'install', '--prefix', str(HOME), f'@deepseek-ai/dsh@{VERSION}', '--no-audit', '--no-fund'],
                   check=True)
    (HOME / 'home').mkdir(exist_ok=True)


def run(workspace, task, judge, timeout=1800, ledger=True):
    workspace = check_workspace(workspace)
    state = health()
    if state.get('max_context', 0) < 32768 or not state.get('loaded'):
        raise RuntimeError('Strata is not serving a loaded model with a 32k context (strata_dev.py run, launch 32k)')
    (HOME / 'home').mkdir(parents=True, exist_ok=True)
    # The served context, not a fixed 32k: a harness that runs out of context stops with "Output token limit reached".
    (HOME / 'home' / 'cordis.patch.yml').write_text(patch(state['model'], workspace, context=state['max_context']),
                                                    encoding='utf-8')
    env = dict(os.environ, DSH_HOME=str(HOME / 'home'), DSH_TELEMETRY_MODE='DISABLED', LSS_STRATA_API_KEY='local-only')
    start = time.monotonic()
    done = subprocess.run(['node', str(BIN), 'headless', task], cwd=workspace, env=env, capture_output=True,
                          text=True, encoding='utf-8', timeout=timeout)
    verdict = subprocess.run(judge, cwd=workspace, shell=True, capture_output=True, text=True, encoding='utf-8')
    seconds = round(time.monotonic() - start, 1)
    status = 'PASS' if verdict.returncode == 0 else 'FAIL'
    import strata_ledger
    out = workspace / '.dsh-run.json'
    if not ledger:
        print(json.dumps({'status': status, 'seconds': seconds, 'dsh_exit': done.returncode, 'model': state.get('model'),
                          'context': state.get('max_context'), 'answer': done.stdout.strip()[-600:],
                          'judge_tail': (verdict.stdout + verdict.stderr)[-400:]}, ensure_ascii=False))
        return 0 if status == 'PASS' else 1
    out.write_text(json.dumps({'identity': {'engine': state.get('model')}, 'tasks': [
        {'id': task[:60], 'kind': 'harness', 'status': status, 'attempts': 1, 'tokens': 0, 'seconds': seconds}]}),
        encoding='utf-8')
    strata_ledger.ingest(out)
    out.unlink()
    print(json.dumps({'status': status, 'seconds': seconds, 'dsh_exit': done.returncode,
                      'answer': done.stdout.strip()[-600:], 'judge_tail': (verdict.stdout + verdict.stderr)[-400:]},
                     ensure_ascii=False))
    return 0 if status == 'PASS' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup')
    go = sub.add_parser('run')
    go.add_argument('--workspace', required=True)
    go.add_argument('--judge', required=True)
    go.add_argument('--timeout', type=int, default=1800)
    go.add_argument('--no-ledger', action='store_true', help='comparisons: do not append to the shared ledger')
    go.add_argument('task')
    args = parser.parse_args(argv)
    if args.command == 'setup':
        setup()
        return 0
    return run(args.workspace, args.task, args.judge, args.timeout, ledger=not args.no_ledger)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
