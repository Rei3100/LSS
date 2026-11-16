"""OS-owned C/D/E placement. Copy + SHA256 verification precedes an atomic
legacy-path junction switch. Source/worktree foundations stay C:, hot data goes
to the explicitly owned D: root, cold references remain E:. Existing destinations
are never overwritten. Journals resume interrupted switches; unknown D: data stays.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import ctypes
import time
from pathlib import Path

import storage_tiers as tiers


def _available_gb(policy, volume):
    """Called under storage_mutation's admission mutex: existing jobs can still
    grow while we copy, so their outstanding estimates must remain reserved.
    """
    import storage_budget as budget
    records = budget._active(budget._root(policy))
    reserved = sum(r['required_gb'] for r in records if 'volumes' not in r or volume in r['volumes'])
    return tiers.free_gb(volume) - reserved

MANIFEST = '.lss-layout-manifest.json'
OWNER = '.lss-layout-copy.json'


def _clone_directory_link(source, target):
    """Copy the reparse record itself, including a broken junction. Resolving a
    missing target to recreate it would incorrectly prevent cache relocation.
    """
    if os.name != 'nt':
        os.symlink(os.readlink(source), target, target_is_directory=True)
        return
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                   wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                       ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    def open_link(path, access):
        handle = kernel.CreateFileW(str(path), access, 7, None, 3, 0x02200000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle
    source_handle = open_link(source, 0)
    data, length = ctypes.create_string_buffer(16384), wintypes.DWORD()
    try:
        if not kernel.DeviceIoControl(source_handle, 0x900A8, None, 0, data, len(data), ctypes.byref(length), None):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(source_handle)
    target.mkdir()
    target_handle = open_link(target, 0x40000000)
    try:
        got = wintypes.DWORD()
        if not kernel.DeviceIoControl(target_handle, 0x900A4, data, length.value, None, 0, ctypes.byref(got), None):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(target_handle)


def pairs(policy):
    rule = policy.get('layout') or {}
    rows = [(Path(r['source']), Path(r['target'])) for r in rule.get('moves', [])]
    for scope in rule.get('scopes', []):
        base = Path(scope['source'])
        if base.is_dir() and not tiers.is_link(base):
            rows += [(p, Path(scope['target']) / p.name) for p in base.iterdir()
                     if p.is_dir() and p.name not in scope.get('exclude', []) and not tiers.is_link(p)
                     and not p.name.startswith(tuple(scope.get('exclude_prefixes', [])))
                     and '.lss-layout-' not in p.name]
    if rule.get('worktree_outputs'):
        from storage_paths import build_directory
        for entry in tiers.worktree_entries(tiers.main_root()):
            repo = Path(entry['worktree'])
            if not tiers.on_volume(repo, policy['hot_volume']):
                continue
            for name in ('build', 'dist'):
                if (repo / name).is_dir() and not tiers.is_link(repo / name):
                    rows.append((repo / name, build_directory(repo, policy) / name))
    return list(dict.fromkeys(rows))


def _write(path, value):
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(pending, path)


def _scan(root):
    links = []
    files, dirs = tiers._scan_tree(root, links=links)
    # A file symlink has different semantics and is not silently converted.
    for link in links:
        if not (getattr(link.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_DIRECTORY
                or link.is_dir()):
            raise OSError(f'Cannot relocate a file link: {link}')
    scan = {'files': files, 'dirs': dirs, 'links': links}
    snapshot = tiers._snapshot(root, scan)
    snapshot['link_targets'] = {str(p.relative_to(root)): os.readlink(p) for p in links}
    return scan, snapshot


def _discard_owned(root):
    """Unhook nested junctions before deleting our verified old/staging copy."""
    links = []
    tiers._scan_tree(root, links=links)
    for link in links:
        os.rmdir(link)
    tiers._remove_tree(root)


def _verify(root, manifest):
    scan, _ = _scan(root)
    files = {str(p.relative_to(root)): p for p in scan['files'] if p.name not in (MANIFEST, OWNER)}
    if files.keys() != manifest['files'].keys():
        raise OSError('Relocation file set changed; original retained')
    for rel, path in files.items():
        item = manifest['files'][rel]
        if path.stat().st_size != item['bytes'] or tiers._hash(path) != item['sha256']:
            raise OSError(f'Relocation bytes differ: {path}')
    actual = {str(p.relative_to(root)): os.path.realpath(p) for p in scan['links']}
    expected = {rel: os.path.realpath(destination) for rel, destination in manifest['links'].items()}
    if actual != expected:
        raise OSError('Relocation links differ; original retained')


def migrate(policy, source, target, *, apply=False, log=print):
    if apply:
        tiers.require_machine_overlay(policy)
    source, target = Path(source), Path(target)
    rule = policy['layout']
    owned = Path(rule['destination_root'])
    if (not source.is_absolute() or not target.is_absolute() or '..' in source.parts or '..' in target.parts
            or not target.is_relative_to(owned) or target == owned or target.is_relative_to(source)):
        raise ValueError('Migration leaves its explicit owned destination')
    if tiers.is_link(owned) or any(tiers.is_link(p) for p in target.parents if p != owned and p.is_relative_to(owned)):
        raise ValueError('Destination ancestor is redirected')
    ident = hashlib.sha256(f'{source}|{target}'.encode()).hexdigest()[:16]
    state = tiers.local_state(policy)
    journal = state / f'layout-{ident}.json'
    old = source.with_name(source.name + '.lss-layout-old-' + ident)
    staging = target.with_name(target.name + '.lss-layout-copy-' + ident)
    record = {'source': str(source), 'target': str(target), 'old': str(old), 'staging': str(staging), 'schema': 1}
    if tiers.is_link(source):
        if source.resolve() != target.resolve():
            return dict(record, status='existing_link_preserved')
        if not old.exists():
            return dict(record, status='already_placed')
    if not source.exists() and not old.exists():
        return dict(record, status='absent')
    if not apply:
        return dict(record, status='would_migrate', bytes=tiers.survey(source).bytes)
    if not owned.anchor or not Path(owned.anchor).is_dir():
        raise OSError('Destination volume is missing')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tiers.storage_mutation(policy, [source, target, old]):
        # A journal plus an owned manifest is required to resume a switch.
        if old.exists():
            if not journal.is_file() or not (target / MANIFEST).is_file():
                raise OSError('Unproven interrupted migration; both copies retained')
            manifest = json.loads((target / MANIFEST).read_text(encoding='utf-8'))
            if manifest.get('source') != str(source) or manifest.get('target') != str(target):
                raise OSError('Interrupted migration identity differs')
            _verify(old, manifest)
            _verify(target, manifest)
            if not source.exists():
                tiers.make_junction(source, target)
            elif not tiers.is_link(source) or source.resolve() != target.resolve():
                raise OSError('Legacy path was recreated; copies retained')
            _discard_owned(old)
            record.update(status='migrated_recovered', bytes=manifest['bytes'])
            _write(journal, record)
            return record
        if target.exists() and not journal.exists():
            raise FileExistsError(f'Existing destination is not owned by this migration: {target}')
        scan, before = _scan(source)
        size = sum(p.stat().st_size for p in scan['files'])
        if _available_gb(policy, str(target.anchor)) - size / 1e9 < rule['minimum_free_gb']:
            raise OSError('Destination capacity floor would be crossed')
        record.update(status='copying', bytes=size)
        _write(journal, record)
        if target.exists():
            manifest = json.loads((target / MANIFEST).read_text(encoding='utf-8'))
            if manifest.get('source') != str(source) or manifest.get('target') != str(target):
                raise OSError('Existing destination identity differs')
            _verify(target, manifest)
            try:
                _verify(source, manifest)
            except OSError:
                # Only our unchanged, never-switched copy can be refreshed.
                # Any independent D writes make _verify(target) refuse above.
                _discard_owned(target)
        if not target.exists():
            if staging.exists():
                marker_file = staging / OWNER if (staging / OWNER).exists() else staging / MANIFEST
                marker = json.loads(marker_file.read_text(encoding='utf-8'))
                if marker.get('source') != str(source) or marker.get('target') != str(target):
                    raise OSError('Unknown staging directory retained')
                _discard_owned(staging)
            staging.mkdir()
            _write(staging / OWNER, {'source': str(source), 'target': str(target)})
            links = []
            rows = tiers.copy_verified(source, staging, allow_existing=True, links=links)
            link_rows = {str(p.relative_to(source)): os.path.realpath(p) for p in links}
            for rel in link_rows:
                _clone_directory_link(source / rel, staging / rel)
            manifest = dict(record, files=rows, links=link_rows, bytes=size)
            _write(staging / MANIFEST, manifest)
            (staging / OWNER).unlink()
            tiers._replace_with_retry(staging, target)
        _, after = _scan(source)
        if before != after:
            raise OSError('Source changed while copying; source retained')
        record['status'] = 'switching'
        _write(journal, record)
        os.rename(source, old)
        try:
            tiers.make_junction(source, target)
        except BaseException:
            os.rename(old, source)
            raise
        _, final = _scan(old)
        if final != before:
            raise OSError('Original changed during switch; old copy retained')
        _discard_owned(old)
        record.update(status='migrated_verified', files=len(manifest['files']), links=len(manifest['links']))
        _write(journal, record)
        log(f'layout migrated {source} -> {target}: {size / 1e9:.3f} GB, verified junction')
        return record


def maintain(policy, root, *, log=print):
    tiers.require_machine_overlay(policy)
    if not policy.get('layout') or not policy['layout'].get('enabled', True):
        return
    processes = tiers.process_texts()
    if '<process list unavailable>' in processes:
        raise OSError('Cannot verify migration consumers')
    cwds, _ = tiers.process_cwds()
    report = {'schema': 1, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'moves': [], 'errors': []}
    for source, target in pairs(policy):
        if tiers.worktree_users(source, processes, cwds):
            report['moves'].append({'source': str(source), 'status': 'deferred_active_consumer'})
            continue
        try:
            with tiers.Lock(tiers.local_state(policy) / '.mover.lock'):
                report['moves'].append(migrate(policy, source, target, apply=True, log=log))
        except (OSError, RuntimeError, ValueError) as exc:
            report['errors'].append({'source': str(source), 'error': str(exc)})
            log(f'layout kept {source}: {exc}')
    for directory in policy['layout'].get('ensure', []):
        path = Path(directory)
        if not path.is_relative_to(Path(policy['layout']['destination_root'])):
            raise ValueError('Layout initialization outside owned root')
        path.mkdir(parents=True, exist_ok=True)
    for item in policy['layout'].get('prepare_files', []):
        source, target = Path(item['source']), Path(item['target'])
        try:
            if not target.is_relative_to(Path(policy['layout']['destination_root'])) or '..' in target.parts:
                raise ValueError('Prepared file outside owned root')
            owned = Path(policy['layout']['destination_root'])
            if any(tiers.is_link(p) for p in target.parents if p == owned or p.is_relative_to(owned)):
                raise ValueError('Prepared model ancestor is redirected')
            proof = tiers.local_state(policy) / ('layout-prepared-' + hashlib.sha256(str(target).encode()).hexdigest()[:16] + '.json')
            def stamp():
                st = target.stat()
                return [st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino]
            if target.exists():
                try:
                    prior = json.loads(proof.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    prior = {}
                if prior.get('sha256') == item['sha256'] and prior.get('stat') == stamp():
                    continue
                if target.stat().st_size != item['bytes'] or tiers._hash(target) != item['sha256']:
                    raise OSError('Prepared target differs; retained without overwrite')
                _write(proof, {'sha256': item['sha256'], 'stat': stamp(), 'target': str(target)})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.lss-prepare-partial')
            with tiers.storage_mutation(policy, [target, partial]):
                if _available_gb(policy, str(target.anchor)) - item['bytes'] / 1e9 < policy['layout']['minimum_free_gb']:
                    raise OSError('Preparation would cross destination floor')
                size, digest = tiers._copy_file(source, partial)
                if size != item['bytes'] or digest != item['sha256'] or tiers._hash(partial) != digest:
                    raise OSError('Prepared model verification failed; archive unchanged')
                os.replace(partial, target)
            _write(proof, {'sha256': item['sha256'], 'stat': stamp(), 'target': str(target)})
            report['moves'].append({'source': str(source), 'target': str(target), 'status': 'prepared_archive_copy', 'bytes': size})
            log(f'layout prepared archive model {target}: {size / 1e9:.3f} GB; archive unchanged')
        except (OSError, ValueError) as exc:
            report['errors'].append({'source': str(source), 'error': str(exc)})
    report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    _write(tiers.local_state(policy) / 'last-layout.json', report)
    if report['errors']:
        raise RuntimeError(f"Layout has {len(report['errors'])} deferred/error entries")
