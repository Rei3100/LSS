"""過去の版のLSSが書いた完了台帳を、今の資産管理の読み取りと記録がそのまま扱える（保存状態の移行）。

完了台帳の形は次のように増えてきた（Git履歴）。
- 2026-08-30 `9b14365`: input_sha256 / recipe_id / completed_at の3列
- 2026-09-05 `be2e6f9`: source_path / outputs を追加（outputsは path と sha256）
- 2026-09-13 現行: outcome / zero_reason / validation_result を追加
移行は書き換えではなく、読める列だけで判断し、次の記録で見出しを増やして古い行を残す。
"""
from __future__ import annotations

import json

from lss.assets import backup, catalog, duplicates, history
from lss.assets.reprocess import ProcessingIdentity, reprocess_candidates
from lss.core import session
from lss.harness.tsv_io import load_tsv, save_tsv
from tests.test_reprocess_candidates import _output, _source

CURRENT = ProcessingIdentity("model_new", "n" * 64, "c" * 64)


def _legacy_ledger(tmp_path, state):
    first, first_sha = _source(tmp_path / "first.flac", b"first")
    second, second_sha = _source(tmp_path / "second.flac", b"second")
    old_output = _output(tmp_path / "second - Instrumental.flac", model="model_old", model_sha="o" * 64,
                         config_sha="c" * 64)
    rows = [
        # 3列の時代: 出力の記録が無く、recipe_idにはモデル識別子を記録していた
        {"input_sha256": first_sha, "recipe_id": "model_old", "completed_at": "2026-08-30"},
        # 5列の時代: 出力のパスとSHA-256はあるが、成果の種類（outcome）が無い
        {"input_sha256": second_sha, "recipe_id": "recipe-0905", "completed_at": "2026-09-05",
         "source_path": str(second.resolve()),
         "outputs": json.dumps([{"path": str(old_output.resolve()), "sha256": session.input_sha256_of(old_output)}])},
    ]
    path = session.completion_ledger_path(state)
    path.parent.mkdir(parents=True)
    save_tsv(path, ["input_sha256", "recipe_id", "completed_at", "source_path", "outputs"], rows)
    return (first, first_sha), (second, second_sha, old_output)


def test_legacy_ledger_rows_are_read_by_every_asset_reader_without_rewriting(tmp_path):
    state = tmp_path / "state"
    (first, first_sha), (second, second_sha, old_output) = _legacy_ledger(tmp_path, state)
    before = session.completion_ledger_path(state).read_bytes()

    assert history.history(first_sha, state_root=state, current=CURRENT)["current_best"] is None
    record = history.history(second_sha, state_root=state, current=CURRENT)
    assert record["revisions"][0]["outcome"] == "artifacts" and record["revisions"][0]["outputs_intact"]
    assert record["revisions"][0]["processing_state"] == "model_updated" and record["current_best"] == 0
    assert not session.is_completed(first_sha, "model_old", first, state_root=state)  # 出力の記録が無い行で飛ばさない
    assert session.is_completed(second_sha, "recipe-0905", second, state_root=state)

    candidates = {c["input_sha256"]: c for c in reprocess_candidates(state_root=state, current=CURRENT)["candidates"]}
    # 3列の行は、記録したモデル識別子から古いモデルの結果として候補になる。所在は記録が無いので不明。
    assert set(candidates) == {first_sha, second_sha}
    assert candidates[first_sha]["source_status"] == "missing" and candidates[first_sha]["outputs"] == []
    assert candidates[second_sha]["previous"]["model_id"] == "model_old"
    assert duplicates.same_file_groups(state) == []
    database = catalog.build(state_root=state, database=tmp_path / "catalog.sqlite")
    assert catalog.summary(database=database)["results"] >= 1
    assert session.completion_ledger_path(state).read_bytes() == before


def test_the_next_record_extends_the_header_and_keeps_legacy_rows(tmp_path):
    state = tmp_path / "state"
    (first, first_sha), (second, second_sha, old_output) = _legacy_ledger(tmp_path, state)
    new_output = _output(tmp_path / "second - Instrumental (r2).flac", model="model_new", model_sha="n" * 64,
                         config_sha="c" * 64)
    session.mark_completed(second_sha, "recipe-0913", "2026-09-13", source_path=second, outputs=[new_output],
                           state_root=state)

    header, rows = load_tsv(session.completion_ledger_path(state))
    assert header[:5] == ["input_sha256", "recipe_id", "completed_at", "source_path", "outputs"]
    assert {"outcome", "zero_reason", "validation_result"} <= set(header)
    assert [row["recipe_id"] for row in rows] == ["model_old", "recipe-0905", "recipe-0913"]
    record = history.history(second_sha, state_root=state, current=CURRENT)
    assert len(record["revisions"]) == 2 and record["revisions"][record["current_best"]]["recipe_id"] == "recipe-0913"
    assert old_output.is_file()


def test_backup_and_restore_carry_a_legacy_ledger_byte_for_byte(tmp_path):
    state = tmp_path / "state"
    _legacy_ledger(tmp_path, state)
    before = session.completion_ledger_path(state).read_bytes()
    root = tmp_path / "app"
    root.mkdir()
    archive = backup.create_backup(state_root=state, root=root)
    session.completion_ledger_path(state).write_bytes(b"input_sha256\trecipe_id\tcompleted_at\n")
    backup.restore_backup(archive, state_root=state, root=root, is_running=lambda: False)
    assert session.completion_ledger_path(state).read_bytes() == before
