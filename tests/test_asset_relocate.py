"""移動された原本・成果物を内容で見つけ、台帳を書き換えずに来歴をたどれるようにする。"""
from __future__ import annotations

import shutil

from lss.assets import history, relocate
from lss.core import session
from tests.test_reprocess_candidates import CURRENT, _output, _source


def test_moved_files_are_found_by_content_and_keep_the_current_best(tmp_path):
    state = tmp_path / "state"
    (tmp_path / "music").mkdir()
    source, sha = _source(tmp_path / "music" / "song.flac", b"song")
    output = _output(tmp_path / "music" / "song - Instrumental.flac", model="model_new", model_sha="n" * 64,
                     config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-new", "2026-09-01", source_path=source, outputs=[output],
                           state_root=state)
    ledger_before = session.completion_ledger_path(state).read_bytes()

    archive = tmp_path / "archive" / "renamed"
    archive.mkdir(parents=True)
    moved_output = archive / "別名 - 伴奏.flac"
    shutil.move(output, moved_output)
    shutil.move(source, archive / "song.flac")
    shutil.copyfile(archive / "song.flac", archive / "song (copy).flac")
    (archive / "unrelated.flac").write_bytes(b"other")

    assert history.history(sha, state_root=state, current=CURRENT)["current_best"] is None
    moves = relocate.find_moved([tmp_path / "archive"], state_root=state)
    by_kind = {move["kind"]: move for move in moves}
    assert by_kind["output"]["found_at"] == [str(moved_output.resolve())]
    assert len(by_kind["source"]["found_at"]) == 2  # 同じ内容が2つあれば両方を候補にする

    relocate.record_relocations(moves, state_root=state)
    relocate.record_relocations(moves, state_root=state)  # 重複して記録しない
    assert session.completion_ledger_path(state).read_bytes() == ledger_before
    record = history.history(sha, state_root=state, current=CURRENT)
    assert record["current_best"] == 0 and record["revisions"][0]["outputs_intact"] is True
    assert len(relocate.current_locations(sha, state_root=state)) == 2
