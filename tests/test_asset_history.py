"""改訂履歴と現在の最良結果を、台帳と成果物から読み取る。"""
from __future__ import annotations

from pathlib import Path

from lss.assets import history
from lss.core import session
from tests.test_reprocess_candidates import CURRENT, _output, _source


def test_current_best_prefers_intact_results_from_the_current_processing(tmp_path):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "song.flac", b"song")
    old = _output(tmp_path / "old.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-old", "2026-08-01", source_path=source, outputs=[old], state_root=state)
    new = _output(tmp_path / "new.flac", model="model_new", model_sha="n" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-new", "2026-09-01", source_path=source, outputs=[new], state_root=state)

    record = history.history(sha, state_root=state, current=CURRENT)
    assert [r["processing_state"] for r in record["revisions"]] == ["model_updated", "current"]
    assert record["current_best"] == 1 and record["current_best_is_current_processing"]

    # 新しい結果が壊れたら、前の良い結果を現在の最良として示し、履歴は残す。
    new.write_bytes(new.read_bytes() + b"\0")
    record = history.history(sha, state_root=state, current=CURRENT)
    assert record["revisions"][1]["outputs_intact"] is False
    assert record["current_best"] == 0 and not record["current_best_is_current_processing"]

    old.unlink()
    record = history.history(sha, state_root=state, current=CURRENT)
    assert record["current_best"] is None and len(record["revisions"]) == 2
    assert history.history("f" * 64, state_root=state, current=CURRENT)["revisions"] == []


def test_reprocessing_with_a_new_config_keeps_the_previous_result_until_the_new_one_is_recorded(tmp_path, monkeypatch):
    """実際のworkerの保存経路で、設定の更新後の再処理が前の結果を上書きせず、履歴から最良を選べる。"""
    from lss.assets.reprocess import ProcessingIdentity
    from lss.core import models_store, worker
    from lss.harness import infer
    from tests.test_worker import _fake_msst_separate, _setup, _write_wav

    _setup(tmp_path, monkeypatch)
    state = tmp_path / "state"
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    monkeypatch.setattr(infer, "msst_separate", _fake_msst_separate)
    source = tmp_path / "mix.wav"
    _write_wav(source)
    job = {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}
    sha = session.input_sha256_of(source)

    first = worker.handle_job(job)
    [old_path] = [Path(p) for p in first["outputs"]]
    old_digest = session.input_sha256_of(old_path)
    config = tmp_path / "config"
    config.write_text(config.read_text(encoding="utf-8") + "# output-affecting revision\n", encoding="utf-8")

    # 新しい結果の完了記録の前に止まった再処理は、前の結果を最良のまま残す。
    original_mark = session.mark_completed

    def interrupted(*_args, **_kwargs):
        raise OSError("記録の途中で停止")

    monkeypatch.setattr(session, "mark_completed", interrupted)
    assert worker.handle_job(job)["status"] == "error"
    monkeypatch.setattr(session, "mark_completed", original_mark)
    unknown = ProcessingIdentity("other_model", "0" * 64, "0" * 64)
    record = history.history(sha, state_root=state, current=unknown)
    assert len(record["revisions"]) == 1 and record["current_best"] == 0
    assert session.input_sha256_of(old_path) == old_digest

    second = worker.handle_job(job)
    assert second["status"] in {"ok", "skipped"}
    [new_path] = [Path(p) for p in second["outputs"]]
    assert new_path != old_path and session.input_sha256_of(old_path) == old_digest
    revisions = history.history(sha, state_root=state, current=unknown)["revisions"]
    newest = revisions[-1]["processing"]
    current = ProcessingIdentity(newest["model_id"], newest["model_sha256"], newest["config_sha256"])
    record = history.history(sha, state_root=state, current=current)
    assert [r["processing_state"] for r in record["revisions"]][-1] == "current"
    assert record["revisions"][record["current_best"]]["outputs"] == [str(new_path)]

    new_path.write_bytes(new_path.read_bytes()[:-8])
    record = history.history(sha, state_root=state, current=current)
    assert record["revisions"][record["current_best"]]["outputs"] == [str(old_path)]
