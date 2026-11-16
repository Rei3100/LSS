"""The saved-answer scorer must pass the historical fixes and retain failures."""
import json
from scripts import score_local_ai_cpu_trial as scorer


def test_all_five_original_fixes_pass_same_frozen_oracles(tmp_path):
    replies = [{'id': t['id'], 'response': json.dumps({'code': t['after']})}
               for t in scorer.bench.load_replays() if t['cases']]
    replies.append({'id': 'currentbest-' + scorer.product.COMMIT,
                    'response': json.dumps({'code': scorer.product.load_task()['after']})})
    replies.extend({'id': t['id'], 'response': json.dumps({'code': t['after']})}
                   for t in scorer.interval.load_replays())
    for identifier, kind, prompt, expected in scorer.bench.REASONING:
        replies.append({'id': identifier, 'response': expected})
    path = tmp_path / 'saved.json'
    path.write_text(json.dumps({'model_sha256': 'fixture', 'replays': replies}), encoding='utf-8')
    report = scorer.score(path)
    assert len(report['repairs']) == 5
    assert all(row.get('validation', {}).get('pass') or row.get('status') == 'accepted'
               for row in report['repairs'])
    assert all(row['passed'] for row in report['reasoning'])


def test_missing_responses_remain_failures(tmp_path):
    path = tmp_path / 'missing.json'
    path.write_text(json.dumps({'model_sha256': 'fixture', 'replays': []}), encoding='utf-8')
    report = scorer.score(path)
    assert len(report['repairs']) == 5
    assert all(not row.get('validation', {}).get('pass') and row.get('status') != 'accepted'
               for row in report['repairs'])


def test_additional_stereo_target_is_counted_only_when_attempted(tmp_path):
    target = scorer.audio_repair.load_target()
    request = scorer.audio_repair.generate_requests()[0]
    path = tmp_path / 'audio.json'
    path.write_text(json.dumps({'model_sha256': 'fixture', 'replays': [
        {'id': request['id'], 'response': json.dumps({'code': target['after']})}]}), encoding='utf-8')
    report = scorer.score(path)
    assert len(report['repairs']) == 6
    assert report['repairs'][-1]['validation']['pass']


def test_single_json_fence_is_optional_and_never_changes_candidate_code(tmp_path):
    task = next(t for t in scorer.bench.load_replays() if t['name'] == '_normalise')
    response = '```json\n' + json.dumps({'code': task['after']}) + '\n```'
    path = tmp_path / 'fenced.json'
    path.write_text(json.dumps({'model_sha256': 'fixture', 'replays': [
        {'id': task['id'], 'response': response}]}), encoding='utf-8')
    strict = scorer.score(path)
    adapted = scorer.score(path, adapt_fences=True)
    assert not strict['repairs'][1]['validation']['pass']
    assert adapted['repairs'][1]['validation']['pass']
    assert adapted['output_adapters'] == {task['id']: 'single_json_fence'}
    assert json.loads(path.read_text())['replays'][0]['response'] == response


def test_valid_but_truncated_code_never_passes_any_repair_family(tmp_path):
    replies = [{'id': t['id'], 'response': json.dumps({'code': t['after']}), 'finish_reason':'length'}
               for t in scorer.bench.load_replays() if t['cases']]
    replies.append({'id':'currentbest-' + scorer.product.COMMIT, 'finish_reason':'length',
                    'response':json.dumps({'code':scorer.product.load_task()['after']})})
    replies.extend({'id':t['id'], 'response':json.dumps({'code':t['after']}), 'finish_reason':'length'}
                   for t in scorer.interval.load_replays())
    replies.append({'id':scorer.audio_repair.generate_requests()[0]['id'], 'finish_reason':'length',
                    'response':json.dumps({'code':scorer.audio_repair.load_target()['after']})})
    path = tmp_path / 'truncated.json'
    path.write_text(json.dumps({'model_sha256':'fixture', 'replays':replies}),encoding='utf-8')
    report = scorer.score(path)
    assert len(report['repairs']) == 6
    for row in report['repairs']:
        validation = row.get('validation', row.get('candidate'))
        assert not validation['pass']
        assert 'truncated' in validation['error']


def test_existing_draft_queue_scores_only_requested_pinned_function(tmp_path):
    task = next(t for t in scorer.bench.load_replays() if t['name'] == '_normalise')
    receipt = tmp_path / 'saved.json'
    receipt.write_text(json.dumps({'status':'draft', 'text':json.dumps({'code':task['after']}),
        'response':{'choices':[{'finish_reason':'stop'}]}}),encoding='utf-8')
    (tmp_path/'summary.json').write_text(json.dumps({'identity':{'model_sha256':'fixture','device':'cpu'},
        'entries':[{'id':task['id'], 'receipt':str(receipt)}]}),encoding='utf-8')
    result = scorer.score(tmp_path)
    assert len(result['repairs']) == 1
    assert result['repairs'][0]['validation']['pass']
    assert set(result['queue_receipts_sha256']) == {task['id']}


def test_draft_summary_cannot_redirect_receipt_read_outside_queue(tmp_path):
    import pytest
    folder = tmp_path/'queue'; folder.mkdir()
    outside = tmp_path/'private.json'; outside.write_text('not a receipt',encoding='utf-8')
    (folder/'summary.json').write_text(json.dumps({'identity':{'model_sha256':'fixture'},
        'entries':[{'id':'bad', 'receipt':str(outside)}]}),encoding='utf-8')
    with pytest.raises(ValueError, match='escapes its queue'):
        scorer.score(folder)


def test_admission_refusal_is_unmeasured_not_six_incorrect_repairs(tmp_path):
    (tmp_path/'summary.json').write_text(json.dumps({'identity':{'model':'large'}, 'status':'failed',
        'entries':[{'id':str(i), 'status':'refused', 'error':'RAM admission'} for i in range(6)]}),encoding='utf-8')
    report = scorer.score(tmp_path)
    assert report['status'] == 'unmeasured_admission_refused'
    assert report['repairs'] == []
    assert report['attempted_repair_ids'] == []
