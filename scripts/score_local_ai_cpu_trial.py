"""Score saved CPU replies against existing historical repair oracles, without inference.

No model, Product state, source media or network is used. Generated functions
enter only the existing AST allowlist and bounded child evaluator. Synthetic
smoke controls, behavioral repair and supplied-fact reasoning stay separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import local_ai_bench as bench
import local_ai_product_replay as product
import local_ai_replay_r2 as interval
import local_ai_audio_repair as audio_repair
from local_ai_output import decode_json


def require_complete(row):
    if row.get('finish_reason') == 'length':
        raise ValueError('output_truncated: token budget exhausted')
    if row.get('status') == 'error':
        raise ValueError('inference_runtime_error: ' + row.get('error', 'unknown'))


def score(path, adapt_fences=False):
    queue_mode = path.is_dir()
    source = path / 'summary.json' if queue_mode else path
    raw = source.read_bytes()
    trial = json.loads(raw)
    receipt_hashes = {}
    if queue_mode:
        summary = trial
        if (summary.get('status') == 'failed' and all(
                entry.get('status') == 'refused' and not entry.get('receipt')
                for entry in summary.get('entries', []))):
            return {'source':str(source.resolve()), 'source_sha256':hashlib.sha256(raw).hexdigest(),
                    'status':'unmeasured_admission_refused', 'model':summary.get('identity', {}).get('model'),
                    'refusals':summary.get('entries', []), 'repairs':[], 'reasoning':[],
                    'attempted_repair_ids':[], 'product_mutated':False,
                    'scope':'No model request was made; this is resource availability, not a repair-quality score.'}
        replies_from_queue = []
        for entry in summary['entries']:
            receipt = Path(entry['receipt']).resolve()
            if not receipt.is_relative_to(path.resolve()):
                raise ValueError('draft receipt escapes its queue output directory')
            data = receipt.read_bytes()
            saved = json.loads(data)
            receipt_hashes[entry['id']] = hashlib.sha256(data).hexdigest()
            replies_from_queue.append({'id':entry['id'], 'response':saved.get('text', ''),
                'status':saved.get('status'),
                'finish_reason':saved.get('response', {}).get('choices', [{}])[0].get('finish_reason')})
        trial = {'model_sha256':summary['identity']['model_sha256'], 'replays':replies_from_queue}
    replies = {row['id']: row for row in trial.get('replays', [])}
    adapters = {}
    if adapt_fences:
        for identifier, row in list(replies.items()):
            if not identifier.startswith(('fix-', 'currentbest-')):
                continue
            try:
                value, adapter = decode_json(row.get('response'))
                if adapter != 'none':
                    replies[identifier] = dict(row, response=json.dumps(value, ensure_ascii=False))
                    adapters[identifier] = adapter
            except ValueError:
                pass
    tasks = [t for t in bench.load_replays() if t['cases']]
    records = []
    # The saved normalise result already exposed the original executor's
    # missing ListComp support. Use the reviewed r2 evaluator for these three.
    for task in tasks:
        row = replies.get(task['id'], {})
        record = {'id': task['id'], 'commit': task['commit'], 'case_count': len(task['cases'])}
        evaluate = interval.evaluate_code if task['name'] == '_normalise' else bench.evaluate_code
        record['baseline'] = evaluate(task['before'], task['name'], task['constants'], task['cases'])
        record['positive_control'] = evaluate(task['after'], task['name'], task['constants'], task['cases'])
        if record['baseline']['pass'] or not record['positive_control']['pass']:
            raise RuntimeError(f"invalid historical controls: {task['id']}: {record}")
        try:
            require_complete(row)
            code = json.loads(row['response'])['code']
            record['validation'] = evaluate(code, task['name'], task['constants'], task['cases'])
        except Exception as exc:
            record['validation'] = {'pass': False, 'error': f'{type(exc).__name__}: {exc}'}
        records.append(record)
    task = product.load_task()
    identifier = 'currentbest-' + product.COMMIT
    row = replies.get(identifier, {})
    record = {'id': identifier, 'commit': task['commit'], 'case_count': len(task['cases'])}
    try:
        require_complete(row)
        code = json.loads(row['response'])['code']
        record['baseline'] = product.evaluate(task['before'], task)
        record['positive_control'] = product.evaluate(task['after'], task)
        record['validation'] = product.evaluate(code, task)
    except Exception as exc:
        record['validation'] = {'pass': False, 'error': f'{type(exc).__name__}: {exc}'}
    records.append(record)
    for interval_task in interval.load_replays():
        row = replies.get(interval_task['id'], {})
        try:
            require_complete(row)
            records.extend(interval.validate_responses(
                [dict(row, raw_output=row.get('response', ''))], [interval_task]))
        except ValueError as exc:
            records.append({'id':interval_task['id'], 'status':'rejected',
                            'candidate':{'pass':False, 'error':str(exc)},
                            'failure_kind':'truncated_or_inference_runtime',
                            'raw_output':row.get('response')})
    audio_task = audio_repair.generate_requests()[0]
    if audio_task['id'] in replies:
        row = replies[audio_task['id']]
        result = {'id': audio_task['id'], 'scope': 'one unchanged historical stereo frontend test'}
        try:
            require_complete(row)
            result['validation'] = audio_repair.evaluate_code(json.loads(row['response'])['code'])
        except Exception as exc:
            result['validation'] = {'pass': False, 'error': f'{type(exc).__name__}: {exc}'}
        records.append(result)
    reasoning = []
    for identifier, kind, prompt, expected in bench.REASONING:
        row = replies.get(identifier, {})
        # Only the entire explicit choice counts; a correct word embedded in
        # contradictory prose is not accepted as a correct experiment decision.
        answer = row.get('response', '').strip().strip('.').lower()
        reasoning.append({'id': identifier, 'kind': kind, 'expected': expected,
                          'answer': answer, 'passed': answer == expected})
    if queue_mode:
        records = [record for record in records if record['id'] in replies]
        reasoning = [record for record in reasoning if record['id'] in replies]
    return {'source': str(source.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'queue_receipts_sha256':receipt_hashes,
            'fence_adaptation_enabled':adapt_fences, 'output_adapters':adapters,
            'scope': ('requested subset of pinned historical repairs; never autonomous repository repair' if queue_mode
                      else 'five common historical targets, plus stereo frontend when attempted; never autonomous repository repair'),
            'model_sha256': trial['model_sha256'], 'repairs': records, 'reasoning': reasoning,
            'attempted_repair_ids':[r['id'] for r in records if r['id'] in replies],
            'smoke_controls': trial.get('tasks', []), 'product_mutated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--adapt-fences', action='store_true')
    args = parser.parse_args()
    report = score(args.trial, args.adapt_fences)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'repair_targets': len(report['repairs'])}))


if __name__ == '__main__':
    main()
