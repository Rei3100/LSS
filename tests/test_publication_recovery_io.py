"""復旧後処理の一時的なI/O障害は、他の公開処理を妨げない。"""
from pathlib import Path

import pytest

from lss.core import session


@pytest.mark.parametrize("boundary", ["final", "staged", "quarantine", "corrupt_journal"])
def test_recovery_cleanup_io_failure_is_retryable(tmp_path, monkeypatch, boundary):
    state = tmp_path / "state"
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    staged = tmp_path / ".lss-first.tmp"
    final = tmp_path / "first.flac"
    missing = tmp_path / ".lss-missing.tmp"
    missing_final = tmp_path / "missing.flac"
    staged.write_bytes(b"candidate")
    missing.write_bytes(b"missing candidate")
    journal = session.prepare_publication(
        "a" * 64, "broken", source_path=source, outputs=[final, missing_final],
        staged=[(staged, final), (missing, missing_final)], state_root=state,
    )
    missing.unlink()
    if boundary == "staged":
        final.write_bytes(b"unrelated output")
    else:
        staged.rename(final)
    if boundary == "corrupt_journal":
        journal.write_text("invalid json", encoding="utf-8")
    # ファイル名順で失敗するjournalの後にも正常な公開を置く。
    first = journal.with_name("000.json")
    journal.rename(first)
    journal = first
    good_staged = tmp_path / ".lss-good.tmp"
    good_final = tmp_path / "good.flac"
    good_staged.write_bytes(b"good candidate")
    good_journal = session.prepare_publication(
        "a" * 64, "good", source_path=source, outputs=[good_final],
        staged=[(good_staged, good_final)], state_root=state,
    )
    good_journal.rename(good_journal.with_name("zzz.json"))

    with monkeypatch.context() as patch:
        unlink = Path.unlink
        replace = Path.replace

        def locked_unlink(path, *args, **kwargs):
            if path == (staged if boundary == "staged" else final):
                raise PermissionError("一時的に使用中")
            return unlink(path, *args, **kwargs)

        def locked_replace(path, *args, **kwargs):
            if path == journal:
                raise PermissionError("隔離先が一時的に使用中")
            return replace(path, *args, **kwargs)

        if boundary in {"final", "staged"}:
            patch.setattr(Path, "unlink", locked_unlink)
        else:
            patch.setattr(Path, "replace", locked_replace)
        results = session.reconcile_publications(state)
        assert [result["status"] for result in results] == ["pending", "completed"]
        assert journal.exists()
        assert good_final.read_bytes() == b"good candidate"
        assert session.reconcile_publications(state)[0]["status"] == "pending"

    results = session.reconcile_publications(state)
    assert results[0]["status"] == ("failed" if boundary == "corrupt_journal" else "rolled_back")
    assert not journal.exists()
    if boundary == "staged":
        assert final.read_bytes() == b"unrelated output"
        assert not staged.exists()
    elif boundary == "corrupt_journal":
        assert final.read_bytes() == b"candidate"
    else:
        assert not final.exists()
    assert not session.is_completed("a" * 64, "broken", source, state_root=state)
    assert session.is_completed("a" * 64, "good", source, state_root=state)
    assert session.reconcile_publications(state) == []
