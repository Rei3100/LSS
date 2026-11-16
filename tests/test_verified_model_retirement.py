"""Proof gate tests use synthetic metadata and never open the model files."""
import copy
import importlib.util
from contextlib import nullcontext
import ctypes
import json
import mmap
import os
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location(
    'retire_verified_gptoss_cache', Path(__file__).resolve().parents[1] / 'scripts/retire_verified_gptoss_cache.py')
retire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retire)


def fixture():
    hot = {'bytes': retire.SIZE, 'mtime_ns': 1, 'file_id': 2, 'volume_id': 3, 'hardlink_count': 1}
    cold = {'bytes': retire.SIZE, 'mtime_ns': 4, 'file_id': 5, 'volume_id': 6, 'hardlink_count': 1}
    digest = {'bytes': retire.SIZE, 'sha256': retire.SHA}
    result = {'status': 'verified', 'copies_identical': True, 'retirement_performed': False,
              'hot': str(retire.HOT), 'cold': str(retire.COLD), 'expected_sha256': retire.SHA,
              'hot_before': copy.deepcopy(hot), 'hot_after': copy.deepcopy(hot),
              'cold_before': copy.deepcopy(cold), 'cold_after': copy.deepcopy(cold),
              'hot_digest': digest.copy(), 'cold_digest': digest.copy()}
    supervisor = {'status': 'passed', 'cpu_accounting': {'returncode': 0, 'active_descendants': 0}}
    row = {'path': retire.HOT.as_posix(), 'archive_path': retire.COLD.as_posix(),
           'sha256': retire.SHA, 'size_bytes': retire.SIZE}
    return result, supervisor, row, hot, cold


def test_complete_proof_is_eligible(monkeypatch):
    assert retire.assess(*fixture()) == []
    monkeypatch.setattr(retire, 'apply', lambda: {'status': 'deferred'})
    assert retire.main(['--apply']) == 2


def test_supervisor_and_identity_fail_closed():
    result, supervisor, row, hot, cold = fixture()
    supervisor['cpu_accounting']['active_descendants'] = 1
    row['sha256'] = 'changed'
    assert len(retire.assess(result, supervisor, row, hot, cold)) == 2


def test_changed_stamp_and_multiple_links_refuse():
    result, supervisor, row, hot, cold = fixture()
    hot['hardlink_count'] = 2
    cold['mtime_ns'] += 1
    assert len(retire.assess(result, supervisor, row, hot, cold)) == 3


def test_missing_digest_and_redirect_refuse():
    result, supervisor, row, hot, cold = fixture()
    del result['cold_digest']
    assert len(retire.assess(result, supervisor, row, hot, cold, paths_direct=False)) == 2


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """Only owned tiny fixtures; no real models, registry, mutex or process probes."""
    hot, cold = tmp_path / 'hot.gguf', tmp_path / 'cold.gguf'
    hot.write_bytes(b'tiny verified archive fixture')
    cold.write_bytes(hot.read_bytes())
    monkeypatch.setattr(retire, 'HOT', hot)
    monkeypatch.setattr(retire, 'COLD', cold)
    monkeypatch.setattr(retire, 'ROOT', tmp_path)
    monkeypatch.setattr(retire, 'SIZE', hot.stat().st_size)
    monkeypatch.setattr(retire, 'REGISTRY', tmp_path / 'registry.json')
    (tmp_path / 'release-evidence').mkdir()
    monkeypatch.setattr(retire, 'lease', nullcontext)
    monkeypatch.setattr(retire, 'check_consumers', lambda: None)
    row = {'path': hot.as_posix(), 'archive_path': cold.as_posix(), 'sha256': retire.SHA,
           'size_bytes': retire.SIZE, 'license': 'unchanged'}
    raw = b'\xef\xbb\xbf' + json.dumps({'models': {'gpt-oss-20b': row, 'other': {'path': 'leave untouched'}}}, indent=2).replace('\n', '\r\n').encode()
    retire.REGISTRY.write_bytes(raw)
    h, c = retire.stamp(hot), retire.stamp(cold)
    result = {'status': 'verified', 'copies_identical': True, 'retirement_performed': False,
              'hot': str(hot), 'cold': str(cold), 'expected_sha256': retire.SHA,
              'hot_before': h, 'hot_after': h, 'cold_before': c, 'cold_after': c,
              'hot_digest': {'sha256': retire.SHA, 'bytes': retire.SIZE},
              'cold_digest': {'sha256': retire.SHA, 'bytes': retire.SIZE}}
    supervisor = {'status': 'passed', 'cpu_accounting': {'returncode': 0, 'active_descendants': 0}}
    monkeypatch.setattr(retire, 'load_state', lambda: (retire.REGISTRY.read_bytes(), result, supervisor, row))
    return hot, cold, raw, result


def test_raw_patch_changes_only_target_path(tiny):
    hot, cold, raw, _ = tiny
    changed = retire.archived_bytes(raw)
    assert changed == raw.replace(json.dumps(hot.as_posix()).encode(), json.dumps(cold.as_posix()).encode(), 1)
    row = json.loads(changed.decode('utf-8-sig'))['models']['gpt-oss-20b']
    assert 'archive_only' not in row and 'placement' not in row


def test_exclusive_handle_matches_python_identity(tiny):
    hot, _, _, _ = tiny
    held = retire.ExclusiveFile(hot)
    try:
        assert held.stamp() == retire.stamp(hot)
    finally:
        held.close()


@pytest.mark.parametrize('readonly', [False, True])
def test_tiny_success_and_restore_metadata(tiny, readonly):
    hot, cold, raw, result = tiny
    if readonly:
        os.chmod(hot, 0o444)
        result['hot_before'] = result['hot_after'] = retire.stamp(hot)
    outcome = retire.apply()
    assert outcome['status'] == 'retired', outcome
    assert not hot.exists() and cold.read_bytes() == b'tiny verified archive fixture'
    new = retire.REGISTRY.read_bytes()
    assert new == retire.archived_bytes(raw)
    journal = [json.loads(line) for line in Path(outcome['receipt']).read_text().splitlines()]
    assert journal[0]['original_row']['path'] == hot.as_posix()
    assert len(journal[0]['restoration']) == 2
    # Metadata restoration is the path update stage+pin performs; other bytes survive.
    retire.cas_registry(new, raw)
    assert retire.REGISTRY.read_bytes() == raw


def test_proof_refusal_never_opens_file(tiny, monkeypatch):
    hot, _, raw, result = tiny
    result['status'] = 'running'
    monkeypatch.setattr(retire, 'ExclusiveFile', lambda *_: pytest.fail('must not open'))
    assert retire.apply()['status'] == 'deferred'
    assert hot.exists() and retire.REGISTRY.read_bytes() == raw


def test_busy_handle_preserves_cache_and_registry(tiny):
    hot, _, raw, _ = tiny
    with hot.open('rb'):
        outcome = retire.apply()
    assert outcome['status'] == 'deferred', outcome
    assert hot.exists() and retire.REGISTRY.read_bytes() == raw


@pytest.mark.parametrize('close_original', [False, True])
def test_mapping_preserves_cache_even_after_original_handle_closes(tiny, close_original):
    hot, _, raw, _ = tiny
    stream = hot.open('rb')
    mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        if close_original:
            stream.close()
        outcome = retire.apply()
        assert outcome['status'] == 'deferred', outcome
        assert hot.exists() and retire.REGISTRY.read_bytes() == raw
        assert mapping[:] == b'tiny verified archive fixture'
    finally:
        mapping.close()
        stream.close()


def test_native_mapping_with_all_file_and_section_handles_closed(tiny, monkeypatch):
    """A view alone remains: unlike Python mmap, no duplicate file handle exists."""
    hot, _, raw, _ = tiny
    k = retire.kernel()
    k.CreateFileMappingW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                                     ctypes.c_ulong, ctypes.c_ulong, ctypes.c_wchar_p]
    k.CreateFileMappingW.restype = ctypes.c_void_p
    k.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                                ctypes.c_ulong, ctypes.c_size_t]
    k.MapViewOfFile.restype = ctypes.c_void_p
    k.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k.UnmapViewOfFile.restype = ctypes.c_int
    file_handle = k.CreateFileW(str(hot), 0x80000000, 7, None, 3, 0, None)
    assert file_handle != ctypes.c_void_p(-1).value
    section = k.CreateFileMappingW(file_handle, None, 2, 0, 0, None)
    assert section
    view = k.MapViewOfFile(section, 4, 0, 0, 0)
    assert view
    assert k.CloseHandle(file_handle)
    assert k.CloseHandle(section)
    calls = []
    dispose = retire.ExclusiveFile.dispose
    def track_dispose(held):
        calls.append('legacy_disposition')
        return dispose(held)
    monkeypatch.setattr(retire.ExclusiveFile, 'dispose', track_dispose)
    try:
        outcome = retire.apply()
        assert outcome['status'] == 'deferred', outcome
        assert calls == ['legacy_disposition']  # share-zero open succeeded; mapping stops disposition.
        assert hot.exists() and retire.REGISTRY.read_bytes() == raw
        assert ctypes.string_at(view, hot.stat().st_size) == b'tiny verified archive fixture'
    finally:
        assert k.UnmapViewOfFile(view)


def test_registry_failure_precedes_disposition(tiny, monkeypatch):
    hot, _, raw, _ = tiny
    def fail(*args):
        raise OSError('fixture registry failure')
    monkeypatch.setattr(retire, 'cas_registry', fail)
    monkeypatch.setattr(retire.ExclusiveFile, 'dispose', lambda _: pytest.fail('must not dispose'))
    assert retire.apply()['status'] == 'deferred'
    assert hot.exists() and retire.REGISTRY.read_bytes() == raw


def test_disposition_failure_rolls_back_readonly_and_registry(tiny, monkeypatch):
    hot, _, raw, result = tiny
    os.chmod(hot, 0o444)
    result['hot_before'] = result['hot_after'] = retire.stamp(hot)
    def fail(_):
        raise OSError('fixture disposition failure')
    monkeypatch.setattr(retire.ExclusiveFile, 'dispose', fail)
    try:
        outcome = retire.apply()
        assert outcome['status'] == 'deferred', outcome
        assert hot.exists() and retire.REGISTRY.read_bytes() == raw
        held = retire.ExclusiveFile(hot)
        try:
            assert held.attrs() & 1
        finally:
            held.close()
    finally:
        os.chmod(hot, 0o666)


def test_rollback_conflict_never_clobbers_other_edit(tiny, monkeypatch):
    hot, _, _, _ = tiny
    conflict = b'{"concurrent": "edit"}'
    def conflict_and_fail(_):
        retire.REGISTRY.write_bytes(conflict)
        raise OSError('fixture busy mapping')
    monkeypatch.setattr(retire.ExclusiveFile, 'dispose', conflict_and_fail)
    outcome = retire.apply()
    assert outcome['status'] == 'recovery_required', outcome
    assert hot.exists() and retire.REGISTRY.read_bytes() == conflict
    assert Path(outcome['receipt']).exists()


def test_retained_archive_allows_readers_but_blocks_writes_and_deletion(tiny):
    _, cold, _, result = tiny
    held = retire.ExclusiveFile(cold, retained=True)
    try:
        assert held.stamp(single_link=False) == result['cold_after']
        assert cold.read_bytes() == b'tiny verified archive fixture'
        with pytest.raises(OSError):
            cold.open('r+b')
        with pytest.raises(OSError):
            cold.unlink()
    finally:
        held.close()


def test_writing_archive_defers_before_registry_or_hot_cache_changes(tiny):
    hot, cold, raw, _ = tiny
    with cold.open('r+b'):
        outcome = retire.apply()
    assert outcome['status'] == 'deferred', outcome
    assert hot.exists() and retire.REGISTRY.read_bytes() == raw
    assert cold.read_bytes() == b'tiny verified archive fixture'
