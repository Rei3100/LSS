"""Technology lookup must preserve scoped evidence and never re-intake installed assets."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import technology as tech


def run_git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


@pytest.fixture
def corpus(tmp_path):
    run_git(tmp_path, 'init', '-b', 'main')
    run_git(tmp_path, 'config', 'user.email', 'fixture@example.invalid')
    run_git(tmp_path, 'config', 'user.name', 'Fixture')
    report = tmp_path / 'trial.md'
    report.write_text('# Research\nContext, not adoption.\n\n## SAM-Audio: music role\n'
                      'Population: 13 rows. Named-sound use remains open.\n'
                      '| input | result |\n|---|---|\n| original | no gain |\n'
                      '\n## Another method\nNot yet measured here.\n', encoding='utf-8')
    run_git(tmp_path, 'add', 'trial.md')
    run_git(tmp_path, 'commit', '-m', 'Initial fixture')
    commit = run_git(tmp_path, 'rev-parse', 'HEAD')
    data = {'schema_version': 1, 'sources': [{'id': 'trial', 'path': 'trial.md',
            'commit': commit, 'follow_ref': 'main'}], 'technologies': [], 'coverage_boundary': 'Fixture only'}
    return tmp_path, report, data


@pytest.mark.parametrize('query', ['SAM Audio', 'SAM-Audio', 'sam_audio'])
def test_search_keeps_population_table_and_role_qualification(corpus, query):
    root, _, data = corpus
    result = tech.search(data, query, root=root)
    assert len(result) == 1
    assert result[0]['kind'] == 'original'
    assert '13 rows' in result[0]['text']
    assert 'Named-sound use remains open' in result[0]['text']
    assert '| original | no gain |' in result[0]['text']
    assert result[0]['line'] == 4
    assert 'Another method' not in result[0]['text']


def test_pinned_source_survives_changed_working_copy_and_follows_blob(corpus):
    root, report, data = corpus
    report.write_text('# Replaced\nChanged experiment\n', encoding='utf-8')
    assert tech.search(data, 'SAM Audio', root=root)[0]['kind'] == 'original'
    # Working prose is not the committed result; a new recorded source is stale.
    run_git(root, 'add', 'trial.md')
    run_git(root, 'commit', '-m', 'New experiment')
    result = tech.check(data, freshness=True, root=root)
    assert result['status'] == 'STALE'
    assert result['stale'][0]['status'] == 'NEW_SOURCE_CONTENT'


def test_unrelated_commit_does_not_make_source_stale(corpus):
    root, _, data = corpus
    (root/'other.md').write_text('Unrelated', encoding='utf-8')
    run_git(root, 'add', 'other.md')
    run_git(root, 'commit', '-m', 'Unrelated commit')
    assert tech.check(data, freshness=True, root=root)['status'] == 'VALID'


def test_missing_follow_ref_keeps_pinned_record(corpus):
    root, _, data = corpus
    data['sources'][0]['follow_ref'] = 'removed-lane'
    checked = tech.check(data, freshness=True, root=root)
    assert checked['status'] == 'STALE'
    assert checked['stale'][0]['status'] == 'FOLLOW_REF_UNAVAILABLE_PIN_RETAINED'
    assert tech.search(data, 'sam_audio', root=root)[0]['kind'] == 'original'


def test_snapshot_tampering_is_not_accepted_as_evidence(tmp_path):
    path = tmp_path/'snapshot.md'
    path.write_bytes(b'# Original\n')
    data = {'sources': [{'id': 'snapshot', 'path': 'snapshot.md',
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}],
            'technologies': [], 'coverage_boundary': 'Fixture'}
    assert tech.check(data, root=tmp_path)['status'] == 'VALID'
    path.write_bytes(b'# Different\n')
    assert tech.check(data, root=tmp_path)['status'] == 'INVALID'


def test_missing_source_is_reported_instead_of_silent_no_match(corpus):
    root, _, data = corpus
    data['sources'][0]['path'] = 'absent.md'
    result = tech.search(data, 'anything', root=root)
    assert result[0]['kind'] == 'unavailable_source'
    assert tech.check(data, root=root)['status'] == 'INVALID'


def test_static_asset_read_does_not_execute_provider(tmp_path):
    provider = tmp_path/'provider.py'
    provider.write_text("raise RuntimeError('must never execute')\n"
                        "WEIGHTS = {'cache_name':'model', 'file_name':'weights.bin', "
                        "'size_bytes':3, 'sha256':'abc', 'url': f'{unknown}/weights'}\n", encoding='utf-8')
    spec = tech.static_spec(provider, 'WEIGHTS')
    assert spec == {'cache_name': 'model', 'file_name': 'weights.bin', 'size_bytes': 3, 'sha256': 'abc'}


def test_asset_missing_size_and_hash_are_distinct(tmp_path):
    asset = tmp_path/'weights.bin'
    spec = {'size_bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
    assert tech.observe_asset(asset, spec)['status'] == 'MISSING_AT_THIS_PATH'
    asset.write_bytes(b'ab')
    assert tech.observe_asset(asset, spec, hash_file=True)['status'] == 'SIZE_MISMATCH'
    asset.write_bytes(b'xyz')
    assert tech.observe_asset(asset, spec)['status'] == 'PRESENT_SIZE_MATCH_HASH_UNCHECKED'
    assert tech.observe_asset(asset, spec, hash_file=True)['status'] == 'HASH_MISMATCH'
    asset.write_bytes(b'abc')
    assert tech.observe_asset(asset, spec, hash_file=True)['status'] == 'HASH_MATCH'


def test_existing_other_store_is_reused_without_model_import_or_login(tmp_path, monkeypatch):
    monkeypatch.delenv('LSS_MODEL_DIR', raising=False)
    provider = tmp_path/'provider.py'
    provider.write_text("raise RuntimeError('no imports')\nWEIGHTS = " + repr({
        'cache_name': 'sam', 'file_name': 'weights.bin', 'size_bytes': 3,
        'sha256': hashlib.sha256(b'abc').hexdigest()}), encoding='utf-8')
    installed = tmp_path/'existing-installed'
    (installed/'sam').mkdir(parents=True)
    (installed/'sam/weights.bin').write_bytes(b'abc')
    data = {'installed_model_root': str(installed), 'technologies': [
        {'id': 'sam-audio', 'assets': [{'module':'provider.py', 'symbol':'WEIGHTS'}]}]}
    observations = tech.assets(data, 'sam-audio', root=tmp_path)
    assert observations[0]['status'] == 'MISSING_AT_THIS_PATH'
    locations = tech.reusable_locations(observations)
    assert len(locations) == 1 and locations[0]['model_root'] == str(installed)
    assert locations[0]['identity'] == 'SIZE_MATCH_HASH_UNCHECKED'
    assert 'LSS_MODEL_DIR' in locations[0]['next_action']
    assert not (tmp_path/'.models').exists()  # no cache initialization or downloads


def test_incomplete_store_is_not_suggested_as_complete():
    rows = [{'store':'installed', 'model_root':'known', 'status':'HASH_MATCH'},
            {'store':'installed', 'model_root':'known', 'status':'MISSING_AT_THIS_PATH'}]
    assert tech.reusable_locations(rows) == []


def test_research_files_without_expected_identity_are_not_hash_verified(tmp_path, monkeypatch):
    monkeypatch.delenv('LSS_MODEL_DIR', raising=False)
    asset = tmp_path/'existing.ckpt'
    asset.write_bytes(b'existing research bytes')
    data = {'installed_model_root': str(tmp_path/'product'), 'technologies': [
        {'id':'research', 'assets':[{'path':str(asset), 'kind':'checkpoint'}],
         'reuse':{'development_use':'Use corrected research loader and these paths'}}]}
    observations = tech.assets(data, 'research', hash_file=True, root=tmp_path)
    assert observations[0]['status'] == 'HASH_OBSERVED_NO_EXPECTATION'
    reuse = tech.reusable_locations(observations)[0]
    assert reuse['identity'] == 'PRESENT_FILES_IDENTITY_UNCHECKED'
    assert 'corrected research loader' in reuse['next_action']


def test_uncatalogued_assets_do_not_mean_absent(monkeypatch, capsys):
    monkeypatch.setattr(tech, 'read_ledger', lambda: {
        'installed_model_root':'unused', 'technologies':[{'id':'known-executed', 'assets':[]}]})
    assert tech.main(['--json', 'assets', 'known-executed']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['inventory_status'] == 'UNKNOWN_NOT_CATALOGUED'
    assert result['observations'] == []


def test_legacy_model_state_is_preserved_not_promoted(corpus):
    root, _, data = corpus
    (root/'models.tsv').write_text('model_id\tstate\tnote\nspecialist\texcluded\tInstrumental slot only\n', encoding='utf-8')
    run_git(root, 'add', 'models.tsv')
    run_git(root, 'commit', '-m', 'Legacy model ledger')
    source = {'id':'models', 'path':'models.tsv', 'commit':run_git(root,'rev-parse','HEAD')}
    record = tech.source_records(source, root=root)[0]
    row = json.loads(record['text'])
    assert row['state'] == 'excluded' and row['note'] == 'Instrumental slot only'
    assert 'adoption' not in row
