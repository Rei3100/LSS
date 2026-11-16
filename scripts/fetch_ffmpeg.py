"""Put the pinned ffmpeg build at bin/ffmpeg.exe, verified by hash. The repository does not carry the binary.

The product resolves ffmpeg as `<resource root>/bin/ffmpeg.exe` and its exact-decode checks were measured with this
build, so a checkout gets this one file and nothing else: the release archive and the executable inside it must both
match their SHA-256. An existing bin/ffmpeg.exe with the pinned hash is kept; one with another hash stops the run.

    python scripts/fetch_ffmpeg.py                 # download, verify, install
    python scripts/fetch_ffmpeg.py --archive <zip> # use an archive already on disk (verified the same way)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VERSION = "9.0.1"
ARCHIVE_URL = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-full_build.zip"
ARCHIVE_SHA256 = "2e8e28af97c2ae338ccef92e36da9b2a4cd21d0cad9dde093545606cb07f5b00"
MEMBER = "ffmpeg-9.0.1-full_build/bin/ffmpeg.exe"
EXE_SHA256 = "57c56e369d5b4873b4d93fc1a1d833cb7cd8bc9325c14b05c34ce60b22842d8a"
ROOT = Path(__file__).resolve().parents[1]


def tls_context():
    """System trust plus certifi's roots when available (a venv's OpenSSL may lack the issuer)."""
    import ssl

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


class FetchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install(archive: Path, target: Path, archive_sha256: str = ARCHIVE_SHA256, member: str = MEMBER,
            exe_sha256: str = EXE_SHA256) -> str:
    """Verify `archive`, extract `member` to `target` atomically; return 'installed'."""
    if sha256_file(archive) != archive_sha256:
        raise FetchError("archive hash does not match the pinned build")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as source, partial.open("wb") as sink:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
            sink.write(block)
    if digest.hexdigest() != exe_sha256:
        partial.unlink()
        raise FetchError("executable hash does not match the pinned build")
    os.replace(partial, target)
    return "installed"


def ensure(target: Path, archive: Path | None = None, url: str = ARCHIVE_URL, **pins) -> str:
    exe_sha256 = pins.get("exe_sha256", EXE_SHA256)
    if target.exists():
        if sha256_file(target) == exe_sha256:
            return "present"
        raise FetchError(f"{target.name} exists with another hash; move it away to install the pinned build")
    if archive is not None:
        return install(archive, target, **pins)
    with tempfile.TemporaryDirectory(prefix="lss-ffmpeg-") as scratch:
        downloaded = Path(scratch) / "ffmpeg.zip"
        with urllib.request.urlopen(url, timeout=60, context=tls_context()) as response, downloaded.open("wb") as sink:
            for block in iter(lambda: response.read(1 << 20), b""):
                sink.write(block)
        return install(downloaded, target, **pins)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, help="release archive already downloaded")
    parser.add_argument("--target", type=Path, default=ROOT / "bin" / "ffmpeg.exe")
    args = parser.parse_args(argv)
    try:
        result = ensure(args.target, args.archive)
    except (FetchError, OSError, zipfile.BadZipFile, KeyError) as error:
        print(f"fetch_ffmpeg: FAILED: {error}", file=sys.stderr)
        return 1
    print(f"fetch_ffmpeg: {result} ffmpeg {VERSION} ({EXE_SHA256[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
