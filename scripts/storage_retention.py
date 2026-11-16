"""Scheduled retention: retire unneeded validation stems, archive old legacy
Product payloads with verified junctions. JSON, accepted/current candidates and
unknown files remain. This module never runs maintenance on import.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import time
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root, require_owned_for_deletion
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root, require_owned_for_deletion


import storage_tiers as tiers


def local_path(path, base):
    if '..' in path.parts or not path.is_relative_to(base) or tiers.is_link(base):
        return False
    return not any(tiers.is_link(p) for p in (path, *path.parents) if p == base or p.is_relative_to(base))


def validation_gc(policy, root, *, apply=False, log=print):
    if apply:
        tiers.require_machine_overlay(policy)
    import disk_cleanup as disk
    if root.resolve() != disk.ROOT.resolve():
        raise ValueError('Validation classifier must belong to this checkout')
    if apply:
        require_owned_for_deletion(root)
    rows, unreadable = disk.validation_media_candidates()
    if unreadable:
        raise OSError(f'Cannot verify validation dependencies: {unreadable[:3]}')
    processes = tiers.process_texts()
    if '<process list unavailable>' in processes:
        raise OSError('Cannot verify running consumers')
    cwds, _ = tiers.process_cwds()
    base = dev_state_root(root) / 'validation'
    cutoff = time.time() - policy['retention']['validation_idle_hours'] * 3600
    deleted, kept = [], []
    for row in rows:
        path = root / row['path']
        if row['class'] != 'RETIRE' or not local_path(path, base):
            kept.append(row)
            continue
        try:
            st = path.stat()
            if st.st_mtime > cutoff or tiers.worktree_users(path.parent, processes, cwds):
                kept.append({**row, 'reason': 'recent or running consumer'})
                continue
            if path.name.lower() not in disk.TRISTEM_NAMES or st.st_size != row['bytes']:
                raise OSError('Candidate changed after classification')
            item = {**row, 'sha256': tiers._hash(path)}
            if apply:
                with tiers.storage_mutation(policy, [path]):
                    after = path.stat()
                    if (st.st_size, st.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise OSError('Candidate changed during verification')
                    path.unlink()
            deleted.append(item)
            log(f"{'retired' if apply else 'would retire'} validation stem {path} ({st.st_size / 1e9:.3f} GB)")
        except OSError as exc:
            kept.append({**row, 'reason': str(exc)})
    return {'action': 'validation_stems', 'applied': apply, 'deleted': deleted,
            'kept': kept, 'bytes': sum(x['bytes'] for x in deleted)}


def legacy_payloads(policy):
    rule = policy['retention']
    if not rule.get('legacy_candidates_root'):
        return []
    base = Path(rule['legacy_candidates_root'])
    if not base.is_dir():
        return []
    if tiers.is_link(base):
        if not tiers.on_hot_volume(base, policy):
            return []
        base = base.resolve()
    cutoff = time.time() - rule['legacy_idle_hours'] * 3600
    found = []
    for case in base.iterdir():
        if not re.fullmatch('[0-9a-f]{64}', case.name) or tiers.is_link(case) or not case.is_dir():
            continue
        # Current/held/failed contemporary runs stay under their Product lifecycle.
        if ((case / 'run/run.json').exists() or (case / 'recipe.json').exists()
                or not (case / '.owner.lock').is_file()):
            continue
        try:
            recipe = json.loads((case / 'activity-recipe.json').read_text(encoding='utf-8'))
            intent = json.loads((case / 'run/intent.json').read_text(encoding='utf-8'))
            separation = json.loads((case / 'run/separation.json').read_text(encoding='utf-8'))
            if (recipe.get('method') != 'windowed_rms_relative_hysteresis'
                    or intent.get('corpus_role') != 'product'
                    or not re.fullmatch('[0-9a-f]{64}', intent.get('source_sha256', ''))
                    or not isinstance(separation.get('artifacts'), list)):
                continue
            # Junction creation updates directory mtimes. Eligibility follows
            # real payload/record writes, so our first move cannot defer its
            # sibling payloads for another 14 days.
            files, _ = tiers._scan_tree(case, links=[])
            newest = max((p.stat().st_mtime for p in files if p.name != '.owner.lock'), default=time.time())
            if newest > cutoff:
                continue
            for name in ('artifacts', 'candidates', 'decode-work'):
                payload = case / 'run' / name
                if local_path(payload, base) and payload.is_dir():
                    tree = tiers.survey(payload)
                    if tree.bytes and not tree.links:
                        found.append(payload)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return found


def archive_legacy(policy, *, apply=False, log=print):
    if apply:
        tiers.require_machine_overlay(policy)
    if not policy.get('move_to_hdd', True):
        # Archiving moves Product state to the HDD; with HDD moves disabled it never runs.
        return {'action': 'legacy_payload_archive', 'applied': apply, 'moved': [], 'bytes': 0,
                'skipped': 'move_to_hdd is false'}
    processes = tiers.process_texts()
    if '<process list unavailable>' in processes:
        raise OSError('Cannot verify running consumers')
    cwds, _ = tiers.process_cwds()
    rows = []
    for payload in legacy_payloads(policy):
        case = payload.parent.parent
        if tiers.worktree_users(case, processes, cwds):
            continue
        if not apply:
            rows.append({'source': str(payload), 'bytes': tiers.survey(payload).bytes})
            continue
        # The Product uses exactly this lock; its legacy payload cannot be
        # regenerated while it is being moved. The lock itself stays on SSD.
        with tiers.Lock(case / '.owner.lock'):
            if payload not in legacy_payloads(policy):
                continue
            with tiers.Lock(tiers.local_state(policy) / '.mover.lock'):
                journal = tiers.local_state(policy) / 'journal.json'
                if journal.exists():
                    tiers.recover(policy, log)
                    journal.unlink()
                journal.write_text(json.dumps({'source': str(payload), 'rule': 'legacy_payload'}), encoding='utf-8')
                record = tiers.demote_one(policy, payload)
                journal.unlink()
            rows.append(record)
            log(f"archived legacy payload {payload}: {record['bytes'] / 1e9:.3f} GB, verified junction")
    return {'action': 'legacy_payload_archive', 'applied': apply, 'moved': rows,
            'bytes': sum(x['bytes'] for x in rows)}


def maintain(policy, root, *, log=print):
    tiers.require_machine_overlay(policy)
    rule = policy.get('retention')
    if not rule:
        return
    state = tiers.local_state(policy)
    stamp = state / 'last-retention.json'
    signature = hashlib.sha256(json.dumps(rule, sort_keys=True).encode() + Path(__file__).read_bytes()).hexdigest()
    try:
        prior = json.loads(stamp.read_text(encoding='utf-8'))
        if prior.get('policy_signature') == signature and time.time() - stamp.stat().st_mtime < rule['every_hours'] * 3600:
            return
    except (OSError, ValueError):
        pass
    report = {'schema': 1, 'policy_signature': signature,
              'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    report['errors'] = []
    for name, action in [('validation', lambda: validation_gc(policy, root, apply=True, log=log)),
                         ('legacy', lambda: archive_legacy(policy, apply=True, log=log))]:
        try:
            report[name] = action()
        except Exception as exc:
            report['errors'].append({'step': name, 'error': str(exc)})
            log(f'retention {name} failed: {exc}')
    attempt = state / 'last-retention-attempt.json'
    attempt.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['errors']:
        raise RuntimeError(f"Retention incomplete: {report['errors']}")
    stamp.write_text(json.dumps(report, indent=2), encoding='utf-8')
