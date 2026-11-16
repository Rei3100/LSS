"""Batch Git reads preserve original bytes, missing records and request boundaries."""
import subprocess

import pytest

from scripts.git_objects import read_blobs


def test_batch_keeps_binary_unicode_crlf_missing_and_duplicate_requests(tmp_path):
    subprocess.run(['git','init','-q'],cwd=tmp_path,check=True)
    raw = '日本語\r\n'.encode('utf-8') + b'\0\xff\n123 blob 2\nend'
    oid = subprocess.check_output(['git','hash-object','-w','--stdin'],input=raw,cwd=tmp_path).decode().strip()
    missing = '0'*40
    result = read_blobs([oid, missing, oid], root=tmp_path)
    assert result == {oid:raw, missing:None}


def test_batch_rejects_line_injection_and_non_blob_objects(tmp_path):
    subprocess.run(['git','init','-q'],cwd=tmp_path,check=True)
    with pytest.raises(ValueError):
        read_blobs(['HEAD\nHEAD'],root=tmp_path)
    tree = subprocess.check_output(['git','mktree'],input=b'',cwd=tmp_path).decode().strip()
    with pytest.raises(ValueError,match='blob'):
        read_blobs([tree],root=tmp_path)


def test_history_checkout_supplies_objects_only_when_named(tmp_path, monkeypatch):
    # A public tree has no prior history; a frozen earlier checkout named by LSS_HISTORY_CHECKOUT
    # lends its objects read only, so records pinned to earlier commits resolve instead of reading as absent.
    history, current = tmp_path / 'history', tmp_path / 'current'
    for repo in (history, current):
        repo.mkdir()
        subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    raw = b'prior trial record\n'
    oid = subprocess.check_output(['git', 'hash-object', '-w', '--stdin'], input=raw, cwd=history).decode().strip()
    monkeypatch.delenv('LSS_HISTORY_CHECKOUT', raising=False)
    assert read_blobs([oid], root=current) == {oid: None}
    monkeypatch.setenv('LSS_HISTORY_CHECKOUT', str(history))
    assert read_blobs([oid], root=current) == {oid: raw}
    assert not (current / '.git' / 'objects' / 'info' / 'alternates').exists()
