"""Read declared Git blobs in one process, preserving binary framing and missing objects.

Used by technology lookup and staged-document checking. No checkout, persistent
cache or index mutation. Callers choose the exact objects they are permitted to read.
``LSS_HISTORY_CHECKOUT`` names a frozen earlier checkout whose objects are lent read only,
so records pinned to commits before this tree's history still resolve.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess


def history_checkout() -> Path | None:
    """The frozen earlier checkout named by ``LSS_HISTORY_CHECKOUT``, if it exists (read only, never written)."""
    value = os.environ.get('LSS_HISTORY_CHECKOUT')
    return Path(value) if value and (Path(value) / '.git' / 'objects').is_dir() else None


def history_environment() -> dict[str, str] | None:
    """Lend the history checkout's objects to this one read (no alternates file is written)."""
    history = history_checkout()
    if history is None:
        return None
    lent = [item for item in os.environ.get('GIT_ALTERNATE_OBJECT_DIRECTORIES', '').split(os.pathsep) if item]
    return {**os.environ, 'GIT_ALTERNATE_OBJECT_DIRECTORIES': os.pathsep.join(
        [*lent, str(history / '.git' / 'objects')])}


def read_blobs(requests, *, root: Path) -> dict[str, bytes | None]:
    requests = list(dict.fromkeys(requests))
    if not requests:
        return {}
    if any(not request or any(c in request for c in '\r\n\0') for request in requests):
        raise ValueError('Invalid Git object request')
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=root,
                            input=('\n'.join(requests)+'\n').encode('utf-8'), env=history_environment(),
                            capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise ValueError('Cannot read declared Git objects')
    stream = io.BytesIO(result.stdout)
    blobs = {}
    for request in requests:
        header = stream.readline().rstrip(b'\n')
        if header.endswith(b' missing'):
            blobs[request] = None
            continue
        fields = header.split()
        if len(fields) != 3 or fields[1] != b'blob':
            raise ValueError(f'Expected a Git blob: {request}')
        try:
            size = int(fields[2])
        except ValueError as exc:
            raise ValueError('Invalid Git blob length') from exc
        raw = stream.read(size)
        if size < 0 or len(raw) != size or stream.read(1) != b'\n':
            raise ValueError('Truncated Git blob response')
        blobs[request] = raw
    if stream.read():
        raise ValueError('Unexpected data after Git blob response')
    return blobs
