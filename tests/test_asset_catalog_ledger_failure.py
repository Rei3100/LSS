"""完了台帳が読めない場合、空目録で完成済みの目録を置き換えない。"""
import pytest

from lss.assets import catalog
from lss.core import session
from tests.test_reprocess_candidates import _source, _output


@pytest.mark.parametrize("error", [PermissionError("ledger is locked"), ValueError("ledger cannot be parsed")])
def test_unreadable_completion_ledger_preserves_previous_catalog(tmp_path, monkeypatch, error):
    state = tmp_path / "state"
    source, sha = _source(tmp_path / "source.flac", b"source")
    output = _output(tmp_path / "output.flac", model="model_old", model_sha="o" * 64, config_sha="c" * 64)
    session.mark_completed(sha, "separation-v1-old", source_path=source, outputs=[output], state_root=state)
    path = catalog.build(state_root=state, database=tmp_path / "catalog.sqlite")
    before = path.read_bytes()
    assert catalog.summary(database=path)["results"] == 1

    def fail_read(_path):
        raise error

    monkeypatch.setattr(catalog, "load_tsv", fail_read)
    with pytest.raises(type(error), match="ledger"):
        catalog.build(state_root=state, database=path)
    assert path.read_bytes() == before
    assert catalog.summary(database=path)["results"] == 1
    assert not list(tmp_path.glob("*.building"))
