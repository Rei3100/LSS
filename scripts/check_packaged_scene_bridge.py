"""Compare five frozen native scene modules with source without running the EXE.

Optionally freeze only an owned SSD build, using the existing manifest mechanism.
This establishes bridge bytecode identity, not inference or naming quality.
"""
try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

MODULES = ('scene_job', 'scene_names', 'scene_models', 'scene_native', 'scene_runtime')


def matches(source, code):
    if not isinstance(code, types.CodeType):
        return False
    expected = compile(source, code.co_filename, 'exec', dont_inherit=True, optimize=0)
    return expected == code


def check(source_root, package, archive):
    rows = []
    for name in MODULES:
        relative = Path('lss/bgm') / (name + '.py')
        source = (source_root / relative).read_bytes()
        physical = package / '_internal' / relative
        code = archive.extract('lss.bgm.' + name)
        rows.append({'module': 'lss.bgm.' + name,
                     'physical_source_matches': physical.read_bytes() == source,
                     'embedded_bytecode_matches': matches(source, code)})
    return {'scope': 'Native bridge identity only; no EXE execution, GPU or model loading.',
            'modules': rows, 'passed': all(r['physical_source_matches'] and r['embedded_bytecode_matches'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--freeze-out', type=Path)
    parser.add_argument('--branch', default='codex/scene-native-bootstrap-20261027')
    args = parser.parse_args()
    source = args.source.resolve()
    package = source / 'dist/LSS'
    from dev_hot_paths import require_hot
    for path in (source, package, Path(sys.executable)):
        require_hot(path, 'packaged bridge identity')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
        text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip()
    if commit != args.commit:
        raise ValueError('Build checkout commit mismatch')
    from PyInstaller.archive.readers import CArchiveReader
    archive = CArchiveReader(str(package / 'LSS.exe')).open_embedded_archive('PYZ.pyz')
    report = check(source, package, archive)
    report['source_commit'] = commit
    if not report['passed']:
        print(json.dumps(report))
        return 1
    if args.freeze_out:
        owned = Path('C:/LSS-Work/wt').resolve()
        if not source.is_relative_to(owned) or not source.name.startswith('codex-scene-bootstrap-build'):
            raise ValueError('Freeze target is not an owned build')
        require_evidence_writable(args.freeze_out)
        out = args.freeze_out.resolve()
        evidence = dev_evidence_root(Path(__file__).resolve().parents[1]) / 'release-evidence'
        if not out.is_relative_to(evidence.resolve()) or out.exists():
            raise ValueError('Freeze output must be new owned evidence')
        require_hot(out, 'frozen manifest')
        for path in package.rglob('*'):
            status = path.lstat()
            if getattr(status, 'st_file_attributes', 0) & 0x400 or path.is_symlink():
                raise ValueError('Package reparse point refused')
            if path.is_file() and (status.st_nlink != 1 or not path.resolve().is_relative_to(package)):
                raise ValueError('Linked or external package file refused')
        import freeze_candidate
        original = sys.argv
        try:
            sys.argv = ['freeze_candidate.py', commit, args.branch, str(package), str(out)]
            freeze_candidate.main()
        finally:
            sys.argv = original
        report['manifest'] = str(out / 'package-manifest.json')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
