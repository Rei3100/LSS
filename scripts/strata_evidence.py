"""Freeze a completed local Strata study with verified portable file identities.

Only derived JSON/JSONL/log/PNG receipts are copied; weights and source media
are excluded. Existing evidence is immutable. A failed/partial study requires
an explicitly separate investigation export, not a completion certificate.
"""
try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, target, *, top_level_only=False):
    requested_target = Path(target)
    source, target = Path(source).resolve(), requested_target.resolve()
    cache, evidence = (ROOT / '.cache/local-ai').resolve(), (dev_evidence_root(ROOT) / 'release-evidence').resolve()
    if source == cache or not source.is_relative_to(cache) or not source.name.startswith('strata-'):
        raise ValueError('Source must be one derived Strata study under the local-AI cache')
    if target == evidence or not target.is_relative_to(evidence):
        raise ValueError('Evidence destination escapes the shared Evidence root')
    terminal = source / 'resources-summary.json'
    if terminal.exists():
        state = json.loads(terminal.read_text(encoding='utf-8'))
    else:
        terminal = source / 'result.json'
        state = json.loads(terminal.read_text(encoding='utf-8')) if terminal.exists() else {}
    if state.get('status') != 'completed':
        raise ValueError('Study is not completed; do not export it as completion evidence')
    # Inline diagnostic accounting can mark the study completed even when
    # the supervised test process never actually collected any tests.
    result_path = source / 'result.json'
    try:
        result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
    except (OSError, ValueError):
        result = {}
    if isinstance(result, dict) and 'reference' in result and 'cpu_cap' in result:
        reference, cap = result['reference'], result['cpu_cap']
        counts = reference.get('counts') if isinstance(reference, dict) else None
        collected = counts.get('collected') if isinstance(counts, dict) else None
        returncode = cap.get('returncode') if isinstance(cap, dict) else None
        active = cap.get('active_descendants') if isinstance(cap, dict) else None
        if (type(collected) is not int or collected <= 0 or type(returncode) is not int
                or returncode not in (0, 1) or type(active) is not int or active != 0):
            raise ValueError('Reference diagnostic completion is not validated')
    if source.name.startswith('strata-protocol-'):
        grading, accounting = source / 'repair-summary.json', source / 'score-cpu-receipt.json'
        if (not grading.exists() or not accounting.exists()
                or json.loads(grading.read_text(encoding='utf-8')).get('status') != 'completed'
                or json.loads(accounting.read_text(encoding='utf-8')).get('returncode') != 0):
            raise ValueError('Protocol grading is not completed; GPU shutdown alone is insufficient')
    rows = []
    candidates = source.iterdir() if top_level_only else source.rglob('*')
    for path in sorted(candidates):
        if path.is_file() and path.suffix.lower() in ('.json', '.jsonl', '.log', '.png'):
            if not path.resolve().is_relative_to(source):
                raise ValueError('Receipt symlink/junction escapes the study')
            rows.append({'path': path.relative_to(source).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
    manifest = {'schema': 1, 'source': str(source), 'terminal_receipt': terminal.name,
                'terminal_sha256': sha(terminal), 'files': rows,
                'scope': 'Frozen development observations, not Product deployment or quality promotion'}
    if top_level_only:
        manifest['selection'] = 'top-level receipts only; temporary fixture trees excluded'
    if target.exists():
        existing = target / 'EXPORT_MANIFEST.json'
        if not existing.exists() or json.loads(existing.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing evidence differs; use a distinct destination')
        if any(not (target / row['path']).is_file() or sha(target / row['path']) != row['sha256'] for row in rows):
            raise ValueError('Existing evidence content changed')
        return manifest
    require_evidence_writable(requested_target)
    target.mkdir(parents=True)
    for row in rows:
        destination = target / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row['path'], destination)
        if sha(destination) != row['sha256']:
            raise ValueError('Copied evidence integrity mismatch')
    (target / 'EXPORT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--top-level-only', action='store_true',
                        help='freeze root receipts without nested temporary fixture trees')
    args = parser.parse_args()
    result = export(args.source, args.target, top_level_only=args.top_level_only)
    print(json.dumps({'files': len(result['files']), 'terminal_sha256': result['terminal_sha256']}))
