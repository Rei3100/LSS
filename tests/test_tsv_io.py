from lss.harness.tsv_io import load_tsv, save_tsv
import pytest


def test_load_save_roundtrip(tmp_path):
    header = ["model_id", "状態"]
    rows = [{"model_id": "a", "状態": "候補"}, {"model_id": "b", "状態": "除外"}]
    path = tmp_path / "models.tsv"
    save_tsv(path, header, rows)

    loaded_header, loaded_rows = load_tsv(path)
    assert loaded_header == header
    assert loaded_rows == rows

    save_tsv(path, header, loaded_rows)
    reloaded_header, reloaded_rows = load_tsv(path)
    assert reloaded_header == header
    assert reloaded_rows == rows


def test_roundtrip_preserves_paths_quotes_and_multiline_values(tmp_path):
    rows = [{"path": 'C:\\音声\\曲.wav', "note": '引用"あり"\tタブ\n改行'}]
    path = tmp_path / "state.tsv"
    for _ in range(3):
        save_tsv(path, ["path", "note"], rows)
        _, actual = load_tsv(path)
        assert actual == rows


def test_failed_save_preserves_previous_ledger(tmp_path):
    path = tmp_path / "state.tsv"
    save_tsv(path, ["id"], [{"id": "original"}])
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_tsv(path, ["id"], [{"id": "replacement"}, {"unexpected": "fail"}])
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
