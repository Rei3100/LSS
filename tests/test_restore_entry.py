"""配備版だけの環境でも、確認してから保存状態のバックアップを復元できる。"""
from __future__ import annotations

from pathlib import Path

import lss_gui
from lss.assets import backup, restore_entry


def _app(tmp_path):
    root = tmp_path / "LSS"
    state = root / ".state"
    state.mkdir(parents=True)
    (state / "completed_recipes.tsv").write_text("input_sha256\trecipe_id\nA\tr1\n", encoding="utf-8")
    (state / "queue.json").write_text('{"items": []}', encoding="utf-8")
    return root, state


def test_a_confirmed_restore_replaces_the_records(tmp_path):
    root, state = _app(tmp_path)
    archive = backup.create_backup(state_root=state, root=root)
    (state / "completed_recipes.tsv").write_text("broken", encoding="utf-8")
    messages = []
    code = restore_entry.run(
        archive, confirm=lambda text: True, report=lambda text, **kw: messages.append((text, kw)),
        restore=lambda path: backup.restore_backup(path, state_root=state, root=root,
                                                   is_running=lambda: False),
        worker_running=lambda: False)
    assert code == 0
    assert (state / "completed_recipes.tsv").read_text(encoding="utf-8").startswith("input_sha256")
    assert "復元しました" in messages[-1][0]


def test_declining_changes_nothing(tmp_path):
    root, state = _app(tmp_path)
    archive = backup.create_backup(state_root=state, root=root)
    (state / "completed_recipes.tsv").write_text("current", encoding="utf-8")
    called = []
    code = restore_entry.run(archive, confirm=lambda text: False, report=lambda *a, **k: called.append(a),
                             restore=lambda path: called.append("restore"))
    assert code == 1 and called == []
    assert (state / "completed_recipes.tsv").read_text(encoding="utf-8") == "current"


def test_a_running_lss_refuses_and_says_so(tmp_path):
    root, state = _app(tmp_path)
    archive = backup.create_backup(state_root=state, root=root)
    messages = []
    code = restore_entry.run(
        archive, confirm=lambda text: True, report=lambda text, **kw: messages.append((text, kw)),
        restore=lambda path: backup.restore_backup(path, state_root=state, root=root,
                                                   is_running=lambda: True),
        worker_running=lambda: False)
    assert code == 2 and messages[-1][1] == {"error": True}
    assert "そのまま" in messages[-1][0]


def test_a_file_that_is_not_a_backup_is_refused_before_asking(tmp_path):
    bogus = tmp_path / "lss-state-bogus.zip"
    bogus.write_bytes(b"not a zip")
    asked = []
    messages = []
    code = restore_entry.run(bogus, confirm=lambda text: asked.append(text) or True,
                             report=lambda text, **kw: messages.append(kw))
    assert code == 2 and asked == [] and messages == [{"error": True}]


def test_the_executable_routes_backups_to_the_restore_entry(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(restore_entry, "run", lambda path: seen.append(Path(path)) or 0)
    dragged = tmp_path / "lss-state-auto-20260919T000000Z.zip"
    assert lss_gui.main([str(dragged)]) == 0
    assert lss_gui.main(["--restore-backup", str(tmp_path / "other.zip")]) == 0
    assert seen == [dragged, tmp_path / "other.zip"]


def test_other_single_arguments_still_open_the_app(monkeypatch, tmp_path):
    import lss.ui.app as app

    opened = []
    monkeypatch.setattr(app, "main", lambda: opened.append(True) or 0)
    monkeypatch.setattr(restore_entry, "run", lambda path: (_ for _ in ()).throw(AssertionError))
    assert lss_gui.main([str(tmp_path / "song.flac")]) == 0
    assert opened == [True]


def test_the_restoring_process_itself_does_not_count_as_a_running_lss():
    listing = b'"LSS.exe","4242","Console","1","50,000 K"\r\n'
    assert backup._other_lss_processes(listing, own_pid=4242) == []
    both = listing + b'"LSS.exe","777","Console","1","60,000 K"\r\n'
    assert backup._other_lss_processes(both, own_pid=4242) == [777]
    assert backup._other_lss_processes(b"INFO: No tasks are running", own_pid=1) == []
    assert backup._other_lss_processes(b'"LSS.exe","?"\r\n', own_pid=1) == [-1]


def test_a_resident_worker_is_stopped_after_asking(tmp_path):
    root, state = _app(tmp_path)
    archive = backup.create_backup(state_root=state, root=root)
    running = [True]
    questions = []

    def stop():
        running[0] = False
        return True

    code = restore_entry.run(
        archive, confirm=lambda text: questions.append(text) or True, report=lambda *a, **k: None,
        restore=lambda path: backup.restore_backup(path, state_root=state, root=root,
                                                   is_running=lambda: running[0]),
        worker_running=lambda: running[0], stop_worker=stop)
    assert code == 0 and running == [False]
    assert any("常駐ワーカー" in text for text in questions)


def test_declining_to_stop_the_worker_changes_nothing(tmp_path):
    root, state = _app(tmp_path)
    archive = backup.create_backup(state_root=state, root=root)
    answers = iter([True, False])
    stopped = []
    code = restore_entry.run(
        archive, confirm=lambda text: next(answers), report=lambda *a, **k: None,
        restore=lambda path: stopped.append("restore"),
        worker_running=lambda: True, stop_worker=lambda: stopped.append("stop") or True)
    assert code == 1 and stopped == []


def test_a_worker_that_does_not_stop_is_reported(tmp_path):
    root, state = _app(tmp_path)
    archive = backup.create_backup(state_root=state, root=root)
    messages = []
    code = restore_entry.run(
        archive, confirm=lambda text: True, report=lambda text, **kw: messages.append(kw),
        restore=lambda path: None, worker_running=lambda: True, stop_worker=lambda: False)
    assert code == 2 and messages == [{"error": True}]
