"""所有記録・process lock・中断回収と、既存Product状態の保護。GPU不要。"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pytest

from lss.core import scratch, session, worker, models_store
from lss.harness import infer
from tests.test_worker import _setup, _write_wav, _fake_msst_separate


def _crash(root):
    code = """
import json, os, sys
from pathlib import Path
from lss.core.scratch import ScratchDirectory
owner = ScratchDirectory(dir=Path(sys.argv[1]), purpose='input_decode', prefix='decode-')
path = Path(owner.__enter__())
(path / 'input.wav').write_bytes(b'regenerable decoded input' * 100)
print(json.dumps({'path': str(path), 'receipt': str(owner.receipt)}), flush=True)
os._exit(0)
"""
    run = subprocess.run([sys.executable, "-c", code, str(root)],
                         check=True, capture_output=True, text=True, timeout=30)
    return json.loads(run.stdout)


def _decisions(report):
    return [row["decision"] for row in report["artifacts"]]


def test_real_process_crash_has_grace_then_idempotent_recovery(tmp_path):
    root = tmp_path / "state" / "work"
    orphan = _crash(root)
    path = Path(orphan["path"])
    before = path.joinpath("input.wav").stat().st_size
    assert "keep_orphan_grace" in _decisions(scratch.reconcile(root))
    preview = scratch.reconcile(root, now=time.time() + 90000, dry_run=True)
    assert "would_remove" in _decisions(preview)
    assert path.exists() and preview["reclaimed_bytes"] == 0
    first = scratch.reconcile(root, now=time.time() + 90000)
    assert first["removed"] == [str(path)]
    assert first["reclaimed_bytes"] == before and first["reclaimed_files"] == 1
    assert not Path(orphan["receipt"]).exists()
    assert scratch.reconcile(root, now=time.time() + 90000)["removed"] == []


def test_another_process_using_scratch_is_protected_even_after_grace(tmp_path):
    root = tmp_path / "work"
    code = """
import sys
from pathlib import Path
from lss.core.scratch import ScratchDirectory
with ScratchDirectory(dir=Path(sys.argv[1]), purpose='separation_inference', prefix='infer-') as folder:
    (Path(folder) / 'stem.wav').write_bytes(b'active inference')
    print(folder, flush=True)
    sys.stdin.read(1)
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", code, str(root)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        path = Path(child.stdout.readline().strip())
        assert path.is_dir()
        report = scratch.reconcile(root, older_than_seconds=0, now=time.time() + 90000)
        assert report["removed"] == []
        assert "keep_active_or_io_blocked" in _decisions(report)
        assert (path / "stem.wav").read_bytes() == b"active inference"
    finally:
        child.kill()
        child.communicate(timeout=30)
    assert scratch.reconcile(root, older_than_seconds=0)["removed"] == [str(path)]


@pytest.mark.parametrize("fail_consumer", [False, True])
def test_scope_exit_cleans_success_and_failure(tmp_path, fail_consumer):
    root = tmp_path / "work"
    def run():
        with scratch.ScratchDirectory(dir=root, prefix="decode-", purpose="input_decode") as folder:
            (Path(folder) / "input.wav").write_bytes(b"temporary")
            if fail_consumer:
                raise RuntimeError("consumer failed")
    if fail_consumer:
        with pytest.raises(RuntimeError, match="consumer failed"):
            run()
    else:
        run()
    assert list(root.iterdir()) == []
    assert list(scratch._registry(root).iterdir()) == []


def test_older_r4_cleanup_cannot_claim_new_active_scratch(tmp_path):
    # 配備済みr4のsession cleanupが使う条件。旧版との同時使用・rollbackでも保護する。
    with scratch.ScratchDirectory(dir=tmp_path / "work", prefix="decode-", purpose="input_decode") as folder:
        name = Path(folder).name
        assert not name.startswith("decode-")
        assert not re.match(r"^[0-9a-f]{12}-", name)


def test_missing_payload_receipt_reconciles_idempotently(tmp_path):
    root = tmp_path / "work"
    orphan = _crash(root)
    path = Path(orphan["path"])
    assert path.resolve().parent == root.resolve()
    shutil.rmtree(path)
    report = scratch.reconcile(root)
    assert "missing_payload" in _decisions(report)
    assert not Path(orphan["receipt"]).exists()
    assert scratch.reconcile(root)["artifacts"] == []


def test_allocation_interrupted_before_identity_is_preserved(tmp_path):
    root = tmp_path / "work"
    orphan = _crash(root)
    receipt = Path(orphan["receipt"])
    record = json.loads(receipt.read_text())
    record["directory_identity"] = None
    receipt.write_text(json.dumps(record))
    assert "keep_unsafe_or_changed" in _decisions(scratch.reconcile(root, older_than_seconds=0))
    assert Path(orphan["path"]).exists()


def test_partial_cleanup_keeps_receipt_and_retries_without_affecting_siblings(tmp_path, monkeypatch):
    root = tmp_path / "work"
    orphan = _crash(root)
    path = Path(orphan["path"])
    (path / "second.wav").write_bytes(b"second")
    protected = root / "decode-user-source"
    protected.mkdir()
    (protected / "source.flac").write_bytes(b"user source")
    real_rmtree = shutil.rmtree
    def partial(target):
        assert target == path
        (path / "input.wav").unlink()
        raise PermissionError("interrupted cleanup")
    with monkeypatch.context() as patch:
        patch.setattr(scratch.shutil, "rmtree", partial)
        first = scratch.reconcile(root, older_than_seconds=0)
    assert "keep_active_or_io_blocked" in _decisions(first)
    assert Path(orphan["receipt"]).is_file()
    assert (path / "second.wav").exists()
    assert scratch.shutil.rmtree is real_rmtree
    second = scratch.reconcile(root, older_than_seconds=0)
    assert second["removed"] == [str(path)] and second["reclaimed_bytes"] == 6
    assert (protected / "source.flac").read_bytes() == b"user source"


def test_cleanup_failure_does_not_change_success_and_startup_retries_immediately(tmp_path, monkeypatch):
    root = tmp_path / "work"
    with monkeypatch.context() as patch:
        patch.setattr(scratch.shutil, "rmtree", lambda _: (_ for _ in ()).throw(PermissionError()))
        with scratch.ScratchDirectory(dir=root, prefix="decode-", purpose="input_decode") as folder:
            (Path(folder) / "input.wav").write_bytes(b"scratch")
    # 正常な解放後の残骸にはcrash graceを適用しない。
    assert scratch.reconcile(root)["removed"] == [folder]


@pytest.mark.parametrize("damage", ["malformed", "foreign_root", "traversal", "missing_lock", "replaced"])
def test_unknown_or_changed_ownership_is_never_deleted(tmp_path, damage):
    root = tmp_path / "work"
    orphan = _crash(root)
    receipt, path = Path(orphan["receipt"]), Path(orphan["path"])
    record = json.loads(receipt.read_text())
    if damage == "malformed":
        receipt.write_text("{partial")
    elif damage == "foreign_root":
        record["work_root"] = str(tmp_path)
        receipt.write_text(json.dumps(record))
    elif damage == "traversal":
        record["directory"] = "../source"
        receipt.write_text(json.dumps(record))
    elif damage == "missing_lock":
        receipt.with_suffix(".lock").unlink()
    else:
        # 別物への置換を、元directoryの所有権で削除しない。
        path.rename(root / "held-original")
        path.mkdir()
        (path / "user.flac").write_bytes(b"irreplaceable")
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert scratch.reconcile(root, older_than_seconds=0)["removed"] == []
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_hardlink_to_source_blocks_entire_scratch_cleanup(tmp_path):
    source = tmp_path / "source.flac"
    source.write_bytes(b"original")
    root = tmp_path / "work"
    orphan = _crash(root)
    path = Path(orphan["path"])
    os.link(source, path / "source.flac")
    assert "keep_unsafe_or_changed" in _decisions(scratch.reconcile(root, older_than_seconds=0))
    assert source.read_bytes() == b"original" and path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction安全性")
def test_junction_to_source_blocks_cleanup_without_traversal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "user.flac").write_bytes(b"original")
    root = tmp_path / "work"
    orphan = _crash(root)
    link = Path(orphan["path"]) / "nested"
    # 作成のみ。削除はProductコードに委ね、cmdでのfilesystem削除は使わない。
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(source)],
                   check=True, capture_output=True, timeout=30)
    try:
        assert "keep_unsafe_or_changed" in _decisions(scratch.reconcile(root, older_than_seconds=0))
        assert (source / "user.flac").read_bytes() == b"original"
    finally:
        link.rmdir()


def test_recovery_preserves_product_state_reuse_and_rejects_corrupt_results(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(models_store, "ensure_model_cached", lambda *a, **k: (tmp_path / "ckpt", tmp_path / "config"))
    calls = []
    def separate(*args, **kwargs):
        calls.append(1)
        # 正常Product call pathから作られた作業物もactive lockで守られる。
        inventory = scratch.reconcile(session.work_dir(), older_than_seconds=0, dry_run=True)
        assert "keep_active_or_io_blocked" in _decisions(inventory)
        assert session.cleanup_stale_work(older_than_seconds=0) == []
        return _fake_msst_separate(*args, **kwargs)
    monkeypatch.setattr(infer, "msst_separate", separate)
    source = tmp_path / "source.wav"
    _write_wav(source)
    source_bytes = source.read_bytes()
    job = {"mix_path": str(source), "model_id": "model_x", "stem_target": "Instrumental"}
    first = worker.handle_job(job)
    assert first["status"] == "ok", first
    output = Path(first["outputs"][0])
    durable = session.state_dir() / "bgm-current-best" / "history.json"
    durable.parent.mkdir(parents=True)
    durable.write_text('{"history": ["previous-known-good"], "current": "kept"}')
    cache = session.state_dir() / "runtime-cache" / "cached-intermediate"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"reusable state")
    rollback = session.state_dir() / "rollback" / "known-good"
    rollback.parent.mkdir(parents=True)
    rollback.write_bytes(b"rollback")
    evidence = session.state_dir() / "validation" / "accepted.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"accepted": true}')
    protected = [source, output, durable, cache, rollback, evidence, session.completion_ledger_path()]
    before = {str(p): p.read_bytes() for p in protected}
    orphan = _crash(session.work_dir())
    assert session.cleanup_stale_work(now=time.time() + 90000) == [Path(orphan["path"])]
    session.reconcile_publications()
    assert before == {str(p): p.read_bytes() for p in protected}
    assert worker.handle_job(job)["status"] == "skipped" and len(calls) == 1
    output.write_bytes(b"corrupt result")
    rebuilt = worker.handle_job(job)
    assert rebuilt["status"] == "ok" and len(calls) == 2
    assert worker.handle_job(job)["status"] == "skipped"
    assert source.read_bytes() == source_bytes
    assert list(session.work_dir().iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows junction")
def test_storage_layout_junction_above_work_root_is_used_and_cleaned(tmp_path):
    # The installed state has `.state\work` as a junction to another drive (storage tiers); BGM decoding failed
    # with "linkを含む作業物は回収しません" on every job.
    target = tmp_path / "d-drive" / "work"
    target.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    link = state / "work"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                   check=True, capture_output=True, timeout=30)
    try:
        root = link / "decode"
        with scratch.ScratchDirectory(dir=root, prefix="decode-", purpose="input_decode") as folder:
            (Path(folder) / "input.wav").write_bytes(b"temporary")
            assert (target / "decode" / Path(folder).name / "input.wav").is_file()
        assert list((target / "decode").iterdir()) == []
        assert scratch.reconcile(root, older_than_seconds=0)["artifacts"] == []
        assert target.is_dir() and link.is_dir()
    finally:
        link.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction")
def test_junction_as_the_scratch_directory_itself_is_still_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "user.flac").write_bytes(b"original")
    root = tmp_path / "work"
    root.mkdir()
    link = root / "scratch-decode-planted"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                   check=True, capture_output=True, timeout=30)
    try:
        with pytest.raises(ValueError, match="linkを含む作業物"):
            scratch._plain(link)
        scratch.reconcile(root, older_than_seconds=0)
        assert (outside / "user.flac").read_bytes() == b"original"
    finally:
        link.rmdir()
