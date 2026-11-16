"""復元前の配置・目録・内容検証と、検証したバイト列の保持を確認する。"""
import hashlib
import json
import zipfile

import pytest

from lss.assets import backup
from tests.test_asset_backup import app, _moment


def _archive(path, entries, *, files=None, schema=1):
    if files is None:
        files = [{"name": name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                 for name, content in entries]
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in entries:
            bundle.writestr(name, content)
        bundle.writestr(backup.MANIFEST, json.dumps({"schema": schema, "files": files}))
    return path


@pytest.mark.parametrize("name", [
    "state/../outside.json", "state/C:/outside.json", "state//queue.json", "state/./queue.json",
    "state/\\outside.json", "state/acquisitions/../../outside.json", "state/acquisitions/x:secret",
    "state/acquisitions/x.json.", "state/acquisitions/x.json ", "state/acquisitions/CON.json",
    "state/work/huge.bin", "state/login-profile/Cookies", "app/lss_gui.py", "app/providers/model.bin",
    "state/acquisitions/incomplete.tmp", "state/acquisitions",
])
def test_restore_rejects_noncanonical_or_unmanaged_paths_before_any_change(app, tmp_path, name):
    root, state = app
    archive = _archive(tmp_path / "unexpected.zip", [(name, b"invalid replacement")])
    original = (state / "queue.json").read_bytes()
    with pytest.raises(backup.BackupError, match="管理範囲外"):
        backup.restore_backup(archive, state_root=state, root=root, is_running=lambda: False, now=_moment(1))
    assert (state / "queue.json").read_bytes() == original
    assert not (root / "Backups").exists()


def test_duplicate_zip_entries_are_refused(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        archive = _archive(tmp_path / "duplicate.zip", [("state/queue.json", b"a"), ("state/queue.json", b"b")])
    with pytest.raises(backup.BackupError, match="重複"):
        backup.verify_backup(archive)


def test_duplicate_manifest_and_windows_case_aliases_are_refused(tmp_path):
    content = b"{}"
    item = {"name": "state/queue.json", "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    archive = _archive(tmp_path / "duplicate-manifest.zip", [(item["name"], content)], files=[item, item])
    with pytest.raises(backup.BackupError, match="重複"):
        backup.verify_backup(archive)
    archive = _archive(tmp_path / "case-alias.zip", [("state/acquisitions/a.json", content), ("state/acquisitions/A.json", content)])
    with pytest.raises(backup.BackupError, match="重複"):
        backup.verify_backup(archive)


@pytest.mark.parametrize("size", [False, -1, 20, "2"])
def test_declared_byte_count_is_verified(tmp_path, size):
    content = b"{}"
    item = {"name": "state/queue.json", "sha256": hashlib.sha256(content).hexdigest(), "bytes": size}
    archive = _archive(tmp_path / "size.zip", [(item["name"], content)], files=[item])
    with pytest.raises(backup.BackupError, match="壊れて"):
        backup.verify_backup(archive)


@pytest.mark.parametrize("files,schema", [(None, True), (None, "1"), ({}, 1), ([None], 1), ([{}], 1)])
def test_invalid_manifest_is_a_structured_backup_error(tmp_path, files, schema):
    archive = _archive(tmp_path / "schema.zip", [], files=files, schema=schema)
    with pytest.raises(backup.BackupError):
        backup.verify_backup(archive)


def test_restore_uses_verified_bytes_even_if_archive_is_replaced_after_validation(app, monkeypatch):
    root, state = app
    archive = backup.create_backup(state_root=state, root=root, now=_moment(0))
    old_queue = (state / "queue.json").read_bytes()
    (state / "queue.json").write_bytes(b"current content")
    verify = backup._read_verified_backup

    def replace_after_read(path):
        result = verify(path)
        _archive(path, [("state/queue.json", b"unverified substituted content")])
        return result

    monkeypatch.setattr(backup, "_read_verified_backup", replace_after_read)
    backup.restore_backup(archive, state_root=state, root=root, is_running=lambda: False, now=_moment(1))
    assert (state / "queue.json").read_bytes() == old_queue


def test_windows_target_resolving_outside_state_is_refused_before_backup(app, monkeypatch, tmp_path):
    root, state = app
    archive = backup.create_backup(state_root=state, root=root, now=_moment(0))
    original = (state / "queue.json").read_bytes()
    real_resolve = type(state).resolve

    def redirected(path, *args, **kwargs):
        if path == state / "queue.json":
            return tmp_path / "outside.json"
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(type(state), "resolve", redirected)
    with pytest.raises(backup.BackupError, match="管理範囲外"):
        backup.restore_backup(archive, state_root=state, root=root, is_running=lambda: False, now=_moment(1))
    assert (state / "queue.json").read_bytes() == original
    assert not list((root / "Backups").glob("before-restore-*"))


def test_legitimate_double_dot_filename_inside_reference_set_is_allowed(tmp_path):
    archive = _archive(tmp_path / "legitimate.zip", [("state/reference_sets/cue..v2.json", b"{}")])
    assert backup.verify_backup(archive)["files"][0]["name"] == "state/reference_sets/cue..v2.json"


def test_same_second_concurrent_backup_keeps_the_first_completed_archive(app, monkeypatch):
    root, state = app
    verify = backup.verify_backup
    nested = False
    completed = []

    def finish_other_backup(path):
        nonlocal nested
        if not nested:
            nested = True
            archive = backup.create_backup(state_root=state, root=root, now=_moment(0))
            completed.append((archive, archive.read_bytes()))
        return verify(path)

    monkeypatch.setattr(backup, "verify_backup", finish_other_backup)
    with pytest.raises(backup.BackupError, match="既にあります"):
        backup.create_backup(state_root=state, root=root, now=_moment(0))
    archive, content = completed[0]
    assert archive.read_bytes() == content
    assert not list((root / "Backups").glob("*.tmp"))
