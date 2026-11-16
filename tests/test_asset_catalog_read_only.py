"""目録を読むだけで空DBを作らず、正常な目録も書き換えない。"""
import pytest

from lss.assets import catalog


@pytest.mark.parametrize("query", [lambda path: catalog.summary(database=path),
                                  lambda path: catalog.occurrences_of("unknown", database=path)])
def test_reading_missing_catalog_does_not_create_an_empty_database(tmp_path, query):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(catalog.sqlite3.OperationalError):
        query(path)
    assert not path.exists()


def test_read_only_uri_handles_spaces_hash_and_japanese_without_modifying_catalog(tmp_path):
    path = tmp_path / "目録 # 1.sqlite"
    catalog.build(state_root=tmp_path / "state", database=path)
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert catalog.summary(database=path)["sources"] == 0
    assert catalog.occurrences_of("unknown", database=path) == []
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    # 接続が閉じられ、Windowsでも読み取り直後に再構築できる。
    catalog.build(state_root=tmp_path / "state", database=path)
