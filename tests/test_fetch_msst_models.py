"""The MSST model definitions are installed only from verified upstream bytes and never overwrite other content."""
import zipfile

import pytest

from scripts import fetch_msst_models as fetch

PREFIX = "upstream-abc/models/"


def _archive(path, files):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(PREFIX + name, data)
    return path


def _pins(files):
    return {"manifest": {name: fetch.blob_id(data) for name, data in files.items()}, "prefix": PREFIX}


def test_blob_id_is_gits():
    assert fetch.blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_verified_archive_installs_and_a_second_run_keeps_it(tmp_path):
    files = {"__init__.py": b"", "bs_roformer/attend.py": b"x = 1\n"}
    archive = _archive(tmp_path / "a.zip", {**files, "extra.py": b"not pinned\n"})
    target = tmp_path / "models"
    assert fetch.ensure(target, archive, **_pins(files)) == {"installed": 2, "present": 0}
    assert (target / "bs_roformer" / "attend.py").read_bytes() == b"x = 1\n"
    assert not (target / "extra.py").exists()
    assert fetch.ensure(target, archive, **_pins(files)) == {"installed": 0, "present": 2}


def test_a_mismatching_member_writes_nothing(tmp_path):
    pins = _pins({"a.py": b"good\n", "b.py": b"good\n"})
    archive = _archive(tmp_path / "a.zip", {"a.py": b"good\n", "b.py": b"tampered\n"})
    with pytest.raises(fetch.FetchError, match="b.py"):
        fetch.ensure(tmp_path / "models", archive, **pins)
    assert not (tmp_path / "models").exists()


def test_existing_other_content_stops_the_run(tmp_path):
    files = {"a.py": b"good\n"}
    archive = _archive(tmp_path / "a.zip", files)
    target = tmp_path / "models"
    target.mkdir()
    (target / "a.py").write_bytes(b"local edit\n")
    with pytest.raises(fetch.FetchError, match="other content"):
        fetch.ensure(target, archive, **_pins(files))
    assert (target / "a.py").read_bytes() == b"local edit\n"


def test_the_pinned_manifest_covers_the_upstream_models_tree():
    assert len(fetch.MANIFEST) == 76
    assert all(len(oid) == 40 for oid in fetch.MANIFEST.values())
