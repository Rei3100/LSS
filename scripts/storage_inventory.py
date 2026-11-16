"""Read-only inventory of named heavy dependencies. Equal names/sizes are only
hash candidates; equal file IDs are already shared. No deletion or hardlinking
across independently updated applications is authorized by this report.
The OS task periodically refreshes this inventory; unknown paths stay untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

NAMES = {'torch_cuda.dll', 'cublaslt64_12.dll', 'bandit-split.ckpt',
         'cudnn_engines_precompiled64_9.dll', 'cublaslt64_13.dll', 'becruily_deux.ckpt',
         'cudnn_adv64_9.dll', 'cufft64_11.dll', 'torch_cpu.dll', 'cusolver64_11.dll',
         'ffmpeg.exe', 'cusolvermg64_11.dll', 'cudnn_ops64_9.dll', 'llvmlite.dll',
         'node.exe', 'nvrtc64_120_0.alt.dll', 'nvrtc64_120_0.dll',
         'nvjitlink_120_0.dll', 'cudnn_heuristic64_9.dll'}


def inventory(roots, *, hash_budget_gb=16, prior=None):
    from storage_tiers import is_link
    rows, errors, seen = [], [], set()
    for root in map(Path, roots):
        if not root.is_dir() or is_link(root):
            continue
        def failed(exc):
            errors.append(str(exc))
        for here, dirs, files in os.walk(root, onerror=failed):
            dirs[:] = [n for n in dirs if n not in {'.git', 'node_modules'} and not is_link(Path(here) / n)]
            for name in files:
                if name.lower() not in NAMES:
                    continue
                path = Path(here) / name
                key = str(path).lower()
                if key in seen or is_link(path):
                    continue
                seen.add(key)
                try:
                    st = path.stat()
                    rows.append({'path': str(path), 'name': name.lower(), 'bytes': st.st_size,
                                 'mtime_ns': st.st_mtime_ns, 'device': st.st_dev,
                                 'file_id': st.st_ino, 'links': st.st_nlink})
                except OSError as exc:
                    errors.append(str(exc))
    groups = {}
    for row in rows:
        groups.setdefault((row['name'], row['bytes']), []).append(row)
    remaining = int(hash_budget_gb * 1e9)
    hashes = {}
    previous = {r['path']: r for r in (prior or {}).get('files', []) if 'sha256' in r}
    for row in rows:
        old = previous.get(row['path'])
        if old and all(old.get(k) == row[k] for k in ('bytes', 'mtime_ns', 'device', 'file_id')):
            row['sha256'] = old['sha256']
            row['verification'] = 'cached_hash_metadata_unchanged'
            hashes[(row['device'], row['file_id'])] = old['sha256']
    for same_size in groups.values():
        if len(same_size) < 2:
            continue
        for row in same_size:
            identity = (row['device'], row['file_id'])
            if identity in hashes:
                row['sha256'] = hashes[identity]
                continue
            if row['bytes'] > remaining:
                row['verification'] = 'deferred_hash_budget'
                continue
            try:
                digest = hashlib.sha256()
                with Path(row['path']).open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                after = Path(row['path']).stat()
                remaining -= row['bytes']
                if (after.st_size, after.st_mtime_ns, after.st_ino) != (row['bytes'], row['mtime_ns'], row['file_id']):
                    row['verification'] = 'changed_during_hash'
                    continue
                row['sha256'] = hashes[identity] = digest.hexdigest()
            except OSError as exc:
                errors.append(str(exc))
    exact = {}
    for row in rows:
        if 'sha256' in row:
            exact.setdefault((row['bytes'], row['sha256']), []).append(row)
    duplicates = []
    for (size, digest), members in exact.items():
        unique = {(r['device'], r['file_id']) for r in members}
        if len(members) > 1:
            duplicates.append({'bytes_each': size, 'sha256': digest, 'paths': [r['path'] for r in members],
                               'distinct_physical_files': len(unique),
                               'duplicate_bytes': size * (len(unique) - 1),
                               'decision': 'retain_managed_dependencies; prune_only_owned_obsolete_packages'})
    return {'schema': 1, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'read_only': True, 'names': sorted(NAMES), 'files': rows, 'verified_groups': duplicates,
            'hash_bytes': int(hash_budget_gb * 1e9) - remaining, 'errors': errors[:100]}


def refresh(policy):
    from storage_tiers import local_state
    rule = policy.get('dependency_inventory')
    if not rule:
        return
    state = local_state(policy)
    output = state / 'dependency-inventory.json'
    try:
        prior = json.loads(output.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        prior = None
    if (prior and prior.get('names') == sorted(NAMES)
            and not prior.get('errors')
            and time.time() - output.stat().st_mtime < rule.get('every_hours', 168) * 3600):
        return
    report = inventory([os.path.expandvars(r) for r in rule['roots']],
                       hash_budget_gb=rule.get('hash_budget_gb', 16), prior=prior)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    os.replace(temporary, output)
    if report.get('errors'):
        raise RuntimeError('; '.join(report['errors']))
