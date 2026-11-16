"""Inspect native Codex hook discovery/trust without model calls or trust changes.

The runtime starts a separate read-only app-server connection. Discovery for a
workspace is not proof that an existing desktop chat loaded or fired its hooks.
The native response, runtime version and local firing receipts are kept apart.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


def native_call(runtime: str, cwd: Path, method: str, params) -> dict:
    process = subprocess.Popen([runtime, 'app-server'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, encoding='utf-8', cwd=cwd,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    replies = queue.Queue()
    def read():
        for line in process.stdout:
            try:
                replies.put(json.loads(line))
            except ValueError:
                pass
        replies.put({'error': 'app-server exited'})
    threading.Thread(target=read, daemon=True).start()
    def send(message):
        process.stdin.write(json.dumps(message) + '\n')
        process.stdin.flush()
    def response(identifier):
        deadline = time.monotonic() + 15
        while True:
            try:
                reply = replies.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as exc:
                raise TimeoutError(f'Codex app-server response timed out: {identifier}') from exc
            if reply.get('id') == identifier or 'error' in reply:
                if 'error' in reply:
                    raise RuntimeError(json.dumps(reply['error']))
                return reply['result']
    try:
        send({'id': 1, 'method': 'initialize', 'params': {'clientInfo': {
            'name': 'lss_hook_status', 'version': '1.0.0'},
            'capabilities': {'experimentalApi': True}}})
        initialized = response(1)
        send({'method': 'initialized'})
        send({'id': 2, 'method': method, 'params': params})
        discovered = response(2)
        return {'initialize': initialized, 'response': discovered}
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default=os.environ.get('CODEX_APP_RUNTIME_CLI_PATH'))
    parser.add_argument('--cwd', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.runtime:
        parser.error('--runtime or CODEX_APP_RUNTIME_CLI_PATH is required')
    data = native_call(args.runtime, args.cwd.resolve(), 'hooks/list', {'cwds': [str(args.cwd.resolve())]})
    data['discovery'] = data.pop('response')
    data['scope'] = 'Native workspace discovery only; not proof of firing in an existing chat'
    data['runtime'] = args.runtime
    data['version'] = subprocess.check_output([args.runtime, '--version'], text=True).strip()
    data['session_id'] = os.environ.get('CODEX_SESSION_ID')
    # The kit adapter (agent-kit/hooks/kitconfig.py state_dir) writes the log.
    state = os.environ.get('AGENT_KIT_STATE_DIR') or str(
        Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'agent-kit/hooks')
    log = Path(state) / 'fired.jsonl'
    data['firing_log'] = str(log)
    data['session_firing_receipts'] = []
    if log.is_file():
        # Existing log files are bounded by the agent-kit codex_adapter.record_fired.
        for line in log.read_text(encoding='utf-8').splitlines()[-2000:]:
            try:
                row = json.loads(line)
                if data['session_id'] and row.get('session_id') == data['session_id']:
                    data['session_firing_receipts'].append(row)
            except ValueError:
                pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    entries = data['discovery'].get('data', [])
    print(json.dumps({'version': data['version'], 'workspaces': [
        {'cwd': e['cwd'], 'errors': e['errors'], 'warnings': e['warnings'],
         'hooks': [{'event': h['eventName'], 'trust': h['trustStatus'], 'enabled': h['enabled'],
                    'source': h['sourcePath']} for h in e['hooks']]} for e in entries],
         'session_firing_receipts': len(data['session_firing_receipts']), 'receipt': str(args.out)}))


if __name__ == '__main__':
    main()
