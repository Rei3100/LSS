"""scripts/fetch_ffmpeg.py: only an archive and executable matching both pins are installed; a foreign binary is never replaced."""
import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_ffmpeg  # noqa: E402

MEMBER = "build/bin/ffmpeg.exe"
EXE = b"MZ synthetic executable"


def _archive(tmp_path, payload=EXE):
    path = tmp_path / "build.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(MEMBER, payload)
        bundle.writestr("build/README.txt", "other member")
    return path


def _pins(archive, payload=EXE):
    return {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "member": MEMBER,
            "exe_sha256": hashlib.sha256(payload).hexdigest()}


def test_installs_the_pinned_member_only(tmp_path):
    archive = _archive(tmp_path)
    target = tmp_path / "bin" / "ffmpeg.exe"
    assert fetch_ffmpeg.ensure(target, archive, **_pins(archive)) == "installed"
    assert target.read_bytes() == EXE
    assert sorted(p.name for p in target.parent.iterdir()) == ["ffmpeg.exe"]
    assert fetch_ffmpeg.ensure(target, archive, **_pins(archive)) == "present"


def test_wrong_archive_or_member_hash_installs_nothing(tmp_path):
    archive = _archive(tmp_path)
    target = tmp_path / "bin" / "ffmpeg.exe"
    pins = _pins(archive)
    with pytest.raises(fetch_ffmpeg.FetchError):
        fetch_ffmpeg.ensure(target, archive, **{**pins, "archive_sha256": "0" * 64})
    with pytest.raises(fetch_ffmpeg.FetchError):
        fetch_ffmpeg.ensure(target, archive, **{**pins, "exe_sha256": "0" * 64})
    assert not target.exists() and not target.with_name("ffmpeg.exe.partial").exists()


def test_existing_binary_with_another_hash_is_kept(tmp_path):
    archive = _archive(tmp_path)
    target = tmp_path / "bin" / "ffmpeg.exe"
    target.parent.mkdir()
    target.write_bytes(b"MZ another build")
    with pytest.raises(fetch_ffmpeg.FetchError):
        fetch_ffmpeg.ensure(target, archive, **_pins(archive))
    assert target.read_bytes() == b"MZ another build"


def test_pins_are_full_sha256():
    for value in (fetch_ffmpeg.ARCHIVE_SHA256, fetch_ffmpeg.EXE_SHA256):
        assert len(value) == 64 and int(value, 16) >= 0
    assert fetch_ffmpeg.ARCHIVE_URL.startswith("https://") and fetch_ffmpeg.VERSION in fetch_ffmpeg.MEMBER
