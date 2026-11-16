"""Select existing fixed research cases; replay one cached Instrumental comparison.

    python scripts/research_tests.py list
    python scripts/research_tests.py plan bgm-separation --pattern order
    python scripts/research_tests.py check [--bank condition-6s] [--hash-assets]
    python scripts/research_tests.py compare-instrumental protocol.json --out result.json

No inference, model imports during lookup, acquisition, fixture regeneration or
promotion. Native corpus/evaluator formats remain owned by their existing tools.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = dev_evidence_root(ROOT) / 'ledger/research_tests.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f'Duplicate JSON key: {key}')
        out[key] = value
    return out


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def local_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and (path.parts[0].endswith('-evidence') or path.parts[0] == 'ep001-truth'
                       or (path.parts[0] == 'ledger' and path.as_posix() not in {
                           'ledger/models.tsv', 'ledger/model_stem_semantics.tsv', 'ledger/test_set.tsv.example'})):
        return dev_evidence_root(root) / path
    return root / path


def catalog(path: Path = CATALOG) -> dict:
    data = read(path)
    if data['schema_version'] != 1:
        raise ValueError('Unsupported research test index')
    for collection in ('banks', 'profiles', 'patterns'):
        ids = [row['id'] for row in data[collection]]
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate {collection} identity')
    banks = {bank['id']: bank for bank in data['banks']}
    for bank in banks.values():
        ids = [case['id'] for case in bank['cases']]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'Empty/duplicate cases: {bank["id"]}')
        for case in bank['cases']:
            if not case['truth'] or not case['conditions']:
                raise ValueError('Case truth and conditions must be explicit')
    for profile in data['profiles']:
        for selection in profile['select']:
            if selection['bank'] not in banks:
                raise ValueError(f'Unknown bank: {selection["bank"]}')
            if not selected_cases(banks[selection['bank']], selection):
                raise ValueError(f'Empty selection: {profile["id"]}')
    return data


def selected_cases(bank: dict, selection: dict) -> list[dict]:
    return [c for c in bank['cases'] if not selection.get('truth') or c['truth'] == selection['truth']]


def plan(data: dict, profile_id: str, patterns: list[str]) -> dict:
    profile = next((p for p in data['profiles'] if p['id'] == profile_id), None)
    if profile is None:
        raise ValueError(f'Unknown question: {profile_id}')
    variation = {p['id']: p for p in data['patterns']}
    if set(patterns) - variation.keys():
        raise ValueError(f'Unknown patterns: {sorted(set(patterns) - variation.keys())}')
    banks = {b['id']: b for b in data['banks']}
    groups, coverage = [], set()
    for selection in profile['select']:
        bank = banks[selection['bank']]
        cases = selected_cases(bank, selection)
        coverage.update(t for c in cases for t in c['conditions'])
        groups.append({'bank': bank['id'], 'bank_sha256': identity(bank),
                       'definition': bank['definition'], 'cases': cases,
                       'evaluation': bank['evaluation'], 'limits': bank['limits'],
                       'truth_counts': dict(Counter(c['truth'] for c in cases))})
    return {'schema_version': 1, 'catalog_revision': data['revision'],
            'question': profile['question'], 'profile': profile_id,
            'status': 'SELECTED_NOT_EXECUTED' if groups else 'NO_REGISTERED_AUDIO_CASES',
            'promotion_allowed': False, 'groups': groups,
            'patterns': [variation[p] for p in dict.fromkeys(patterns)],
            'missing_conditions': sorted(set(profile['required_conditions']) - coverage),
            'uncovered': profile['uncovered'], 'unit_tests': profile['tests'],
            'comparison_contract': ['Fix intended target/output stem, current baseline and candidate identity before scoring.',
                                    'Keep case IDs, input/context, channel/rate, alignment and measurement identical.',
                                    'Report paired changes, worst cases and wanted deletion separately; no pooled truth denominator.',
                                    'Missing cache or unknown semantic truth stays unresolved; never generate inference here.']}


def check(data: dict, *, bank_id: str | None = None, hash_assets: bool = False,
          root: Path = ROOT) -> dict:
    banks = [b for b in data['banks'] if bank_id is None or b['id'] == bank_id]
    if not banks:
        raise ValueError(f'Unknown bank: {bank_id}')
    errors, rows, seen = [], [], {}
    for bank in banks:
        definition = local_path(bank['definition']['path'], root)
        definition_state = 'MISSING'
        if definition.is_file():
            definition_state = 'MATCH' if digest(definition) == bank['definition']['sha256'] else 'CHANGED'
            native = read(definition)
            native_rows = native.get('cases', native.get('rows', []))
            native_ids = [c.get('case_id', c.get('id')) for c in native_rows]
            if sorted(native_ids) != sorted(c['id'] for c in bank['cases']):
                errors.append(f'{bank["id"]}: indexed/native case sets differ')
        if definition_state != 'MATCH':
            errors.append(f'{bank["id"]}: definition {definition_state}')
        states = Counter()
        for case in bank['cases']:
            for artifact in case['artifacts'].values():
                path = local_path(artifact['path'], root)
                key = (str(path), artifact['sha256'])
                if key not in seen:
                    state = 'MISSING'
                    if path.is_file():
                        state = ('HASH_MATCH' if digest(path) == artifact['sha256'] else 'HASH_MISMATCH') if hash_assets else 'PRESENT_HASH_UNCHECKED'
                    seen[key] = state
                states[seen[key]] += 1
                if seen[key] in {'MISSING', 'HASH_MISMATCH'}:
                    errors.append(f'{bank["id"]}/{case["id"]}: {path} {seen[key]}')
        rows.append({'bank': bank['id'], 'cases': len(bank['cases']),
                     'definition': definition_state, 'artifact_observations': dict(states),
                     'conditions': dict(Counter(t for c in bank['cases'] for t in c['conditions'])),
                     'truth_counts': dict(Counter(c['truth'] for c in bank['cases'])),
                     'limits': bank['limits'], 'audio_status': 'NOT_INDEXED' if not states else 'OBSERVED'})
    for profile in data['profiles']:
        for test in profile['tests']:
            if not (root / test).is_file():
                errors.append(f'Missing unit-test locator: {test}')
    return {'status': 'INVALID' if errors else 'INDEX_VALID', 'banks': rows, 'errors': errors,
            'boundary': 'Index/asset observations only, not test execution, semantic coverage, untouched holdout or Product acceptance.'}


def compare_instrumental(data: dict, protocol: dict, *, root: Path = ROOT) -> dict:
    """One existing bank/measurement path, with exact full-set output correspondence."""
    bank = next(b for b in data['banks'] if b['id'] == 'instrumental-controls')
    if protocol.get('schema_version') != 1 or protocol.get('bank_sha256') != identity(bank):
        raise ValueError('Protocol must fix the exact Instrumental bank before comparison')
    ids = [c['id'] for c in bank['cases']]
    for name in ('baseline', 'candidate'):
        method = protocol[name]
        if not method.get('identity') or method.get('target') != 'Instrumental':
            raise ValueError('Explicit method identity and Instrumental target required')
        if set(method['outputs']) != set(ids):
            raise ValueError('Output case set must exactly match the fixed bank')
        for case in bank['cases']:
            row = method['outputs'][case['id']]
            if row['input_sha256'] != case['artifacts']['input']['sha256']:
                raise ValueError('Output belongs to another input')
            if row['encoding'] not in {'float-wav', 'lss-flac'}:
                raise ValueError('Explicit supported serialization required')
    implementations = {p: digest(root / p) for p in [
        'scripts/research_tests.py', 'lss/validation/instrumental_quality.py',
        'lss/validation/stereo_quality.py', 'lss/validation/stereo_comparison.py',
        'scripts/probe_instrumental_repeat.py']}
    consumed = {}
    # Lazy imports: lookup never imports audio, Product workers or model packages.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import numpy as np
    import soundfile as sf
    from mutagen.flac import FLAC
    from lss.validation.instrumental_quality import instrumental_quality
    from scripts.probe_instrumental_repeat import comparison, DECISION_RULE

    def verify_artifact(artifact):
        path = local_path(artifact['path'], root)
        if digest(path) != artifact['sha256']:
            raise ValueError(f'Changed audio: {path}')
        consumed[str(path)] = artifact['sha256']
        return path

    def audio(artifact, case, encoding='float-wav'):
        path = verify_artifact(artifact)
        values, rate = sf.read(path, always_2d=True, dtype='float64')
        if rate != case['sample_rate'] or values.shape != (case['frames'], 2) or not np.isfinite(values).all():
            raise ValueError(f'Finite native stereo/timeline required: {path}')
        if encoding == 'lss-flac':
            values *= 2.0 ** int(FLAC(path)['LSS_OUTPUT_HEADROOM_SHIFT'][0])
        elif sf.info(path).subtype not in {'FLOAT', 'DOUBLE'}:
            raise ValueError('float-wav must be unscaled floating-point PCM')
        return values.T

    rows = []
    for case in bank['cases']:
        input_artifact = case['artifacts']['input']
        verify_artifact(input_artifact)
        target = audio(case['artifacts']['target'], case)
        vocals = audio(case['artifacts']['vocals'], case)
        measured = {}
        for name in ('baseline', 'candidate'):
            row = protocol[name]['outputs'][case['id']]
            measured[name] = instrumental_quality(audio(row, case, row['encoding']), target, vocals)
        rows.append({'case_id': case['id'], 'conditions': case['conditions'], **measured,
                     'comparison': comparison(measured['baseline'], measured['candidate'])})
    for path, expected in consumed.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Audio changed during comparison: {path}')
    if any(digest(root / p) != expected for p, expected in implementations.items()):
        raise ValueError('Measurement implementation changed during comparison')
    return {'status': 'DEVELOPMENT_DIAGNOSTIC', 'promotion_allowed': False,
            'bank_sha256': identity(bank), 'protocol_sha256': identity(protocol),
            'methods': {name: protocol[name]['identity'] for name in ('baseline', 'candidate')},
            'measurement': {'quality': 'lss.validation.instrumental_quality.instrumental_quality',
                            'comparison': 'scripts.probe_instrumental_repeat.comparison',
                            'decision_rule': DECISION_RULE,
                            'source_sha256': implementations,
                            'runtime': {'numpy': np.__version__, 'soundfile': sf.__version__,
                                        'libsndfile': sf.__libsndfile_version__},
                            'consumed_artifacts': [{'path': p, 'sha256': sha} for p, sha in consumed.items()],
                            'source_identities_rechecked': True,
                            'preprocessing': 'Native rate/stereo/full timeline; decode recorded FLAC headroom only; no fitting/alignment/resampling/normalisation.'},
            'regression_cases': [r['case_id'] for r in rows if r['comparison']['regression']], 'cases': rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--json', action='store_true', help='Full machine-readable records; default output is compact')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    select = commands.add_parser('plan')
    select.add_argument('profile')
    select.add_argument('--pattern', action='append', default=[])
    select.add_argument('--out', type=Path)
    validate = commands.add_parser('check')
    validate.add_argument('--bank')
    validate.add_argument('--hash-assets', action='store_true')
    compare = commands.add_parser('compare-instrumental')
    compare.add_argument('protocol', type=Path)
    compare.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = catalog()
        if args.command == 'list':
            result = {'profiles': [{k: p[k] for k in ('id', 'question')} for p in data['profiles']],
                      'banks': [{'id': b['id'], 'cases': len(b['cases']), 'limits': b['limits']} for b in data['banks']],
                      'patterns': data['patterns']}
        elif args.command == 'plan':
            result = plan(data, args.profile, args.pattern)
        elif args.command == 'check':
            result = check(data, bank_id=args.bank, hash_assets=args.hash_assets)
        else:
            result = compare_instrumental(data, read(args.protocol))
        rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if getattr(args, 'out', None):
            require_evidence_writable(args.out)
            with args.out.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(rendered)
            print(f'{result["status"]}: {args.out}')
        elif args.json:
            print(rendered, end='')
        elif args.command == 'list':
            for profile in result['profiles']:
                print(f'{profile["id"]}: {profile["question"]}')
            for bank in result['banks']:
                print(f'bank {bank["id"]}: {bank["cases"]} indexed rows')
            print('Patterns: ' + ', '.join(p['id'] for p in result['patterns']))
        elif args.command == 'plan':
            print(f'{result["status"]}: {result["question"]}')
            for group in result['groups']:
                print(f'{group["bank"]} @{group["bank_sha256"][:12]}: ' + ', '.join(c['id'] for c in group['cases']))
                print('  Truth groups: ' + json.dumps(group['truth_counts']))
            for pattern in result['patterns']:
                print(f'Vary {pattern["id"]}: {pattern["vary"]}; fix {pattern["keep_fixed"]}')
            print('Missing tags: ' + ', '.join(result['missing_conditions']))
            for gap in result['uncovered']:
                print('UNRESOLVED: ' + gap)
            print('Tests: ' + ' '.join(result['unit_tests']))
            print('Use --json before the command or --out <new-file> for full cases, constraints and native evaluator locators.')
        else:
            print(result['status'])
            for bank in result['banks']:
                print(f'{bank["bank"]}: {bank["cases"]} rows; definition {bank["definition"]}; {bank["artifact_observations"]}; audio {bank["audio_status"]}')
            for error in result['errors'][:10]:
                print('ERROR: ' + error)
            if len(result['errors']) > 10:
                print(f'{len(result["errors"])} errors total; --json includes every path.')
        return 1 if result.get('errors') else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
