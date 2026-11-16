"""保存状態のバックアップと復元。目録で確かめ、動作中は復元せず、失敗時は元へ戻す。"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone

import pytest

from lss.assets import backup


@pytest.fixture
def app(tmp_path):
    root = tmp_path / "LSS"
    state = root / ".state"
    (state / "publications").mkdir(parents=True)
    (state / "acquisitions").mkdir()
    (state / "work").mkdir()
    (state / "completed_recipes.tsv").write_text("input_sha256\trecipe_id\nA\tr1\n", encoding="utf-8")
    (state / "queue.json").write_text('{"items": [1]}', encoding="utf-8")
    (state / "acquisitions" / "x.json").write_text('{"revisions": []}', encoding="utf-8")
    (state / "work" / "huge.bin").write_bytes(b"0" * 1000)
    (root / "ledger").mkdir()
    (root / "ledger" / "models.tsv").write_text("model_id\nm\n", encoding="utf-8")
    return root, state


def _moment(second):
    return datetime(2026, 9, 13, 12, 0, second, tzinfo=timezone.utc)


def test_backup_contains_only_records_and_restores_them(app):
    root, state = app
    archive = backup.create_backup(state_root=state, root=root, now=_moment(0))
    assert archive.parent == root / "Backups"
    names = {item["name"] for item in backup.verify_backup(archive)["files"]}
    assert names == {"state/completed_recipes.tsv", "state/queue.json", "state/acquisitions/x.json", "app/ledger/models.tsv"}

    (state / "completed_recipes.tsv").write_text("broken", encoding="utf-8")
    (state / "acquisitions" / "new.json").write_text("{}", encoding="utf-8")
    saved = backup.restore_backup(archive, state_root=state, root=root, is_running=lambda: False, now=_moment(1))

    assert (state / "completed_recipes.tsv").read_text(encoding="utf-8").startswith("input_sha256")
    assert not (state / "acquisitions" / "new.json").exists()
    assert (saved / "state" / "completed_recipes.tsv").read_text(encoding="utf-8") == "broken"
    assert (saved / "state" / "acquisitions" / "new.json").exists()
    assert (state / "work" / "huge.bin").exists()


def test_restore_is_refused_while_running_or_for_a_damaged_backup(app, tmp_path):
    root, state = app
    archive = backup.create_backup(state_root=state, root=root, now=_moment(0))
    with pytest.raises(backup.BackupError, match="動いている"):
        backup.restore_backup(archive, state_root=state, root=root, is_running=lambda: True)

    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(damaged, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            target.writestr(item, data.replace(b"r1", b"r2") if item.filename.endswith(".tsv") else data)
    with pytest.raises(backup.BackupError, match="壊れて"):
        backup.restore_backup(damaged, state_root=state, root=root, is_running=lambda: False)
    assert "r1" in (state / "completed_recipes.tsv").read_text(encoding="utf-8")


def test_a_failed_restore_returns_to_the_previous_records(app, monkeypatch):
    root, state = app
    archive = backup.create_backup(state_root=state, root=root, now=_moment(0))
    (state / "queue.json").write_text('{"items": [2]}', encoding="utf-8")
    original_replace = type(state).replace
    calls = {"n": 0}

    def failing_replace(self, target):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk removed")
        return original_replace(self, target)

    monkeypatch.setattr(type(state), "replace", failing_replace)
    with pytest.raises(OSError):
        backup.restore_backup(archive, state_root=state, root=root, is_running=lambda: False, now=_moment(1))
    monkeypatch.undo()
    assert json.loads((state / "queue.json").read_text(encoding="utf-8")) == {"items": [2]}


def test_move_records_and_reference_sets_are_backed_up_but_the_login_profile_is_not(app):
    root, state = app
    (state / "relocations.json").write_text('{"moves": []}', encoding="utf-8")
    (state / "reference_sets").mkdir()
    (state / "reference_sets" / "series_e.json").write_text('{"entries": []}', encoding="utf-8")
    cookies = state / "login-profile" / "EBWebView" / "Default" / "Network" / "Cookies"
    cookies.parent.mkdir(parents=True)
    cookies.write_bytes(b"secret")
    archive = backup.create_backup(state_root=state, root=root, now=_moment(5))
    names = {item["name"] for item in backup.verify_backup(archive)["files"]}
    assert {"state/relocations.json", "state/reference_sets/series_e.json"} <= names
    assert not any("login-profile" in name for name in names)


def _at(day, hour=0):
    return datetime(2026, 9, day, hour, 0, 0, tzinfo=timezone.utc)


def test_the_first_start_takes_an_automatic_backup(app):
    root, state = app
    made = backup.ensure_recent_backup(state_root=state, root=root, now=_at(1))
    assert made is not None and made.name.startswith(backup.AUTOMATIC_PREFIX)
    names = {item["name"] for item in backup.verify_backup(made)["files"]}
    assert "state/completed_recipes.tsv" in names


def test_a_recent_automatic_backup_is_not_repeated(app):
    root, state = app
    backup.ensure_recent_backup(state_root=state, root=root, now=_at(1))
    (state / "queue.json").write_text('{"items": [1, 2]}', encoding="utf-8")
    assert backup.ensure_recent_backup(state_root=state, root=root, now=_at(1, 5)) is None


def test_unchanged_records_are_not_backed_up_again(app):
    root, state = app
    backup.ensure_recent_backup(state_root=state, root=root, now=_at(1))
    assert backup.ensure_recent_backup(state_root=state, root=root, now=_at(3)) is None
    (state / "queue.json").write_text('{"items": [1, 2]}', encoding="utf-8")
    assert backup.ensure_recent_backup(state_root=state, root=root, now=_at(3)) is not None


def test_only_automatic_backups_are_pruned(app):
    root, state = app
    manual = backup.create_backup(state_root=state, root=root, now=_at(1))
    saved = root / "Backups" / "before-restore-20260901T000000Z"
    saved.mkdir()
    for day in range(2, 8):
        (state / "queue.json").write_text(json.dumps({"items": [day]}), encoding="utf-8")
        backup.ensure_recent_backup(state_root=state, root=root, now=_at(day), keep=3)
    automatic = sorted(path.name for path in (root / "Backups").glob(f"{backup.AUTOMATIC_PREFIX}*.zip"))
    assert automatic == [f"{backup.AUTOMATIC_PREFIX}202609{day:02d}T000000Z.zip" for day in (5, 6, 7)]
    assert manual.is_file() and saved.is_dir()


def test_a_broken_latest_automatic_backup_is_replaced(app):
    root, state = app
    first = backup.ensure_recent_backup(state_root=state, root=root, now=_at(1))
    first.write_bytes(b"not a zip")
    made = backup.ensure_recent_backup(state_root=state, root=root, now=_at(1, 1))
    assert made is not None and backup.verify_backup(made)


def test_nothing_to_protect_takes_no_backup(tmp_path):
    root = tmp_path / "LSS"
    (root / ".state").mkdir(parents=True)
    assert backup.ensure_recent_backup(state_root=root / ".state", root=root, now=_at(1)) is None
    assert not (root / "Backups").exists()


def test_the_development_worker_does_not_write_backups(monkeypatch):
    from lss.core import worker

    called = []
    monkeypatch.setattr(backup, "ensure_recent_backup", lambda **_: called.append(1))
    monkeypatch.setattr("lss.paths.is_frozen", lambda: False)
    worker._keep_state_backup()
    assert called == []
    monkeypatch.setattr("lss.paths.is_frozen", lambda: True)
    worker._keep_state_backup()
    assert called == [1]


def test_a_failing_backup_does_not_stop_the_worker(monkeypatch):
    from lss.core import worker

    def fail(**_):
        raise backup.BackupError("disk full")

    monkeypatch.setattr(backup, "ensure_recent_backup", fail)
    monkeypatch.setattr("lss.paths.is_frozen", lambda: True)
    worker._keep_state_backup()
