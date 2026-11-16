from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import storage_tiers as tiers  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_sleep_policy(tmp_path, monkeypatch):
    import dev_quiet_mode
    monkeypatch.setattr(dev_quiet_mode, 'POLICY', tmp_path / 'absent-sleep-policy.json')


@pytest.mark.parametrize("free,expected", [(50.0, 87.5), (140.0, 0.0)])
def test_status_reports_unreachable_demotion_target(tmp_path, monkeypatch, free, expected):
    monkeypatch.setattr(tiers, "free_gb", lambda _: free)
    monkeypatch.setattr(tiers, "candidates", lambda *_: [
        tiers.Candidate("idle", "work", 12.5, 1, 5, True, "idle"),
        tiers.Candidate("busy", "work", 90, 1, 0, False, "in use"),
    ])
    report = tiers.status({"hot_volume": ('C' + chr(58) + '\\'), "start_free_gb": 100, "target_free_gb": 150}, tmp_path)
    assert report["gap_after_demote_gb"] == expected


def _policy(tmp_path, **overrides):
    policy = {**machine_policy(tmp_path), "schema_version": 1, "hot_volume": str(tmp_path.anchor), "archive_root": str(tmp_path / "archive"),
              "receipts_root": str(tmp_path / "receipts"), "local_state_root": str(tmp_path / "local"),
              "stage": {"root": str(tmp_path / "stage")},
              "disposable": {"names": ["__pycache__", ".pytest_cache"], "prefixes": ["pytest-of-", "pytest-"]},
              "start_free_gb": 1, "target_free_gb": 2,
              "min_idle_hours": 48, "min_candidate_gb": 0.0,
              "candidates": [{"glob": ".state/validation/*", "class": "evidence"},
                             {"glob": ".worktrees/*/.state", "class": "worktree_state", "requires_idle_worktree": True}],
              "never": [".state/releases"]}
    policy.update(overrides)
    return policy


def _tree(root: Path, files: dict[str, bytes], age_hours: float = 100.0):
    stamp = time.time() - age_hours * 3600
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, (stamp, stamp))
    for directory in sorted({p for p in root.rglob("*") if p.is_dir()} | {root}, key=lambda p: -len(p.parts)):
        os.utime(directory, (stamp, stamp))


def _digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != tiers.MANIFEST}


def test_demote_moves_verified_and_leaves_a_working_junction(tmp_path):
    source = tmp_path / "lss" / ".state" / "validation" / "old-case"
    _tree(source, {"a.flac": b"a" * 5000, "sub/b.json": b"{}", "sub/deep/c.bin": os.urandom(70000)})
    before = _digest(source)
    record = tiers.demote_one(_policy(tmp_path), source)
    assert tiers.is_link(source) and record["status"] == "DEMOTED_VERIFIED_JUNCTION"
    assert _digest(source) == before                      # readable through the original path
    archive = Path(record["archive"])
    assert archive.is_relative_to(tmp_path / "archive") and _digest(archive) == before
    receipt = json.loads(Path(record["receipt"]).read_text(encoding="utf-8"))
    assert receipt["file_count"] == 3 and set(receipt["files"]) == set(before)


def test_promote_brings_it_back_and_keeps_the_archive_copy(tmp_path):
    source = tmp_path / "lss" / ".state" / "validation" / "case"
    _tree(source, {"x.wav": os.urandom(9000), "y/z.txt": b"z"})
    before = _digest(source)
    record = tiers.demote_one(_policy(tmp_path), source)
    tiers.promote(_policy(tmp_path), source)
    assert not tiers.is_link(source) and source.is_dir() and _digest(source) == before
    assert _digest(Path(record["archive"])) == before


def test_a_failed_verification_keeps_the_original_untouched(tmp_path, monkeypatch):
    source = tmp_path / "lss" / ".state" / "validation" / "case"
    _tree(source, {"x.wav": os.urandom(9000)})
    before = _digest(source)
    monkeypatch.setattr(tiers, "_hash", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="verification failed"):
        tiers.demote_one(_policy(tmp_path), source)
    assert not tiers.is_link(source) and _digest(source) == before


def test_a_directory_holding_a_junction_is_never_moved_whole(tmp_path):
    root = tmp_path / "lss"
    elsewhere = tmp_path / "elsewhere"
    _tree(elsewhere, {"keep.bin": b"k"})
    case = root / ".state" / "validation" / "mixed"
    _tree(case, {"own/data.bin": os.urandom(4000)})
    tiers.make_junction(case / "linked", elsewhere)
    with pytest.raises(RuntimeError, match="link inside"):
        tiers.copy_verified(case, tmp_path / "out")
    found = {Path(c.path).name: c for c in tiers.candidates(_policy(tmp_path), root, processes=[])}
    assert "mixed" not in found and found["own"].eligible        # its own parts are judged one by one
    assert (elsewhere / "keep.bin").read_bytes() == b"k"


def test_a_junction_whose_target_was_reclaimed_does_not_stop_the_scan(tmp_path):
    root = tmp_path / "lss"
    target = tmp_path / "reclaimed"
    _tree(target, {"x.bin": b"x"})
    _tree(root / "work" / "valid" / "run", {"a.bin": b"a"})
    tiers.make_junction(root / "work" / "dangling", target)
    target.rename(tmp_path / "moved-away")                      # the junction now points nowhere
    for rule in ({"glob": "work/*/*"}, {"glob": str(root / "work" / "*" / "*"), "absolute": True}):
        policy = _policy(tmp_path, candidates=[{**rule, "class": "evidence"}])
        found = [Path(c.path) for c in tiers.candidates(policy, root, processes=[])]
        assert found == [root / "work" / "valid" / "run"]


def test_recent_writes_running_processes_and_active_worktrees_hold_a_candidate(tmp_path):
    root = tmp_path / "lss"
    _tree(root / ".state" / "validation" / "old", {"a": b"1"})
    _tree(root / ".state" / "validation" / "fresh", {"a": b"1"}, age_hours=1)
    _tree(root / ".state" / "validation" / "busy", {"a": b"1"})
    _tree(root / ".state" / "releases" / "x", {"a": b"1"})
    worktree = root / ".worktrees" / "lane"
    gitdir = tmp_path / "gitdir"
    _tree(gitdir, {"index": b"i", "HEAD": b"h"}, age_hours=2)
    _tree(worktree / ".state", {"run/out.flac": b"o"})
    (worktree / ".git").write_text(f"gitdir: {gitdir}", encoding="utf-8")
    busy = str(root / ".state" / "validation" / "busy").lower()
    found = {Path(c.path).name if Path(c.path).name != ".state" else "lane": c
             for c in tiers.candidates(_policy(tmp_path), root, processes=[f"python.exe\tpython run.py {busy}\\x"])}
    assert found["old"].eligible
    assert not found["fresh"].eligible and "written" in found["fresh"].reason
    assert not found["busy"].eligible and "process" in found["busy"].reason
    assert not found["lane"].eligible and "worktree active" in found["lane"].reason
    assert "x" not in found


def test_auto_does_nothing_above_the_start_watermark(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tiers, "load_policy", lambda: _policy(tmp_path, start_free_gb=0.0, target_free_gb=0.0))
    monkeypatch.setattr(tiers, "main_root", lambda: tmp_path)
    monkeypatch.setattr(tiers, "demote", lambda *a, **k: pytest.fail("mover must not start"))
    assert tiers.main(["auto"]) == 0


def test_an_interrupted_move_is_recovered_without_touching_sources(tmp_path):
    policy = _policy(tmp_path)
    finished = tmp_path / "lss" / ".state" / "validation" / "finished"
    _tree(finished, {"a.bin": os.urandom(3000)})
    record = tiers.demote_one(policy, finished)
    index = Path(policy["archive_root"]) / "index.jsonl"
    index.write_text("", encoding="utf-8")                     # killed before its receipt was recorded
    untouched = tmp_path / "lss" / ".state" / "validation" / "untouched"
    _tree(untouched, {"b.bin": b"b" * 100})
    partial = Path(policy["archive_root"]) / "x" / ("untouched" + tiers.PARTIAL_SUFFIX)
    _tree(partial, {"b.bin": b"b" * 10})
    (partial / tiers.PARTIAL).write_text(str(untouched), encoding="utf-8")
    foreign = Path(policy["archive_root"]) / "y" / ("someone" + tiers.PARTIAL_SUFFIX)
    _tree(foreign, {"keep.bin": b"k"})                          # no marker: not ours, never removed
    tiers.recover(policy, log=lambda *_: None)
    assert not partial.exists() and (foreign / "keep.bin").exists()
    assert (untouched / "b.bin").read_bytes() == b"b" * 100
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    assert [(r["archive"], r["status"]) for r in rows] == [(record["archive"], "DEMOTED_VERIFIED_JUNCTION_RECOVERED")]


def test_one_mover_at_a_time_and_a_dead_holder_frees_the_lock(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("must not signal a process"))
    lock = tmp_path / ".mover.lock"
    lock.write_text("4294967", encoding="utf-8")                 # leftover content decides nothing
    with tiers.Lock(lock):
        with pytest.raises(RuntimeError, match="another mover"):
            with tiers.Lock(lock):
                pass
    holder = subprocess.Popen([sys.executable, "-c",
                               "import sys,time; sys.path.insert(0, sys.argv[1]); import storage_tiers as t; "
                               "from pathlib import Path; l=t.Lock(Path(sys.argv[2])); l.__enter__(); print('held', flush=True); time.sleep(60)",
                               str(Path(tiers.__file__).parent), str(lock)], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    with pytest.raises(RuntimeError, match="another mover"):
        with tiers.Lock(lock):
            pass
    holder.kill()
    holder.wait()
    for _ in range(100):                                         # Windows releases a killed holder's lock shortly after
        try:
            with tiers.Lock(lock):
                break
        except RuntimeError:
            time.sleep(0.1)
    else:
        pytest.fail("the lock of a killed holder was never released")

def test_a_briefly_denied_rename_is_retried_and_the_marker_stays_until_it_succeeds(tmp_path, monkeypatch):
    source = tmp_path / "lss" / ".state" / "validation" / "case"
    _tree(source, {"a.bin": os.urandom(2000)})
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(Path(a))
        if len(calls) < 3:
            assert (Path(a) / tiers.PARTIAL).exists()
            raise PermissionError(5, "denied")
        return real(a, b)
    monkeypatch.setattr(tiers.os, "replace", flaky)
    monkeypatch.setattr(tiers.time, "sleep", lambda s: None)
    record = tiers.demote_one(_policy(tmp_path), source)
    assert len(calls) == 3 and tiers.is_link(source)
    assert not (Path(record["archive"]) / tiers.PARTIAL).exists()


def test_audit_accounts_for_every_moved_directory_and_flags_strays(tmp_path):
    policy = _policy(tmp_path)
    source = tmp_path / "lss" / ".state" / "validation" / "case"
    _tree(source, {"a.bin": os.urandom(500)})
    tiers.demote_one(policy, source)
    stray = Path(policy["archive_root"]) / "stray"
    _tree(stray, {"left.bin": b"x"})
    report = tiers.audit(policy, full_hash=True)
    assert [(m["state"], m["missing_or_different"]) for m in report["moved"]] == [("linked", 0)]
    assert not report["problems"] and [u["archive"] for u in report["unaccounted"]] == [str(stray)]
    (Path(report["moved"][0]["archive"]) / "a.bin").write_bytes(b"changed")
    assert tiers.audit(policy, full_hash=True)["problems"]


def test_a_run_with_nothing_to_move_never_touches_the_archive(tmp_path, monkeypatch):
    policy = _policy(tmp_path, archive_root=str(tmp_path / "hdd" / "archive"), receipts_root=str(tmp_path / "hdd" / "r"))
    root = tmp_path / "lss"
    _tree(root / ".state" / "validation" / "fresh", {"a": b"1"}, age_hours=1)
    monkeypatch.setattr(tiers, "recover", lambda *a, **k: pytest.fail("no recovery without an interrupted run"))
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    assert tiers.demote(policy, root, apply=True, until_free_gb=1e9, log=lambda *_: None) == []
    assert not (tmp_path / "hdd").exists()


def test_regenerable_dirs_are_deleted_on_the_ssd_and_the_rest_is_moved(tmp_path, monkeypatch):
    policy = _policy(tmp_path)
    root = tmp_path / "lss"
    case = root / ".state" / "validation" / "case"
    _tree(case, {"result.flac": os.urandom(3000), "pytest-b/test_x0/tmp.wav": os.urandom(5000),
                 "sub/__pycache__/m.pyc": b"c"})
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    done = tiers.demote(policy, root, apply=True, until_free_gb=1e9, log=lambda *_: None)
    assert [Path(r["source"]).name for r in done] == ["case"]
    archive = Path(done[0]["archive"])
    assert (archive / "result.flac").exists() and not (archive / "pytest-b").exists()
    assert not (archive / "sub" / "__pycache__").exists()       # junk never reaches the HDD
    receipts = list((tmp_path / "local" / "receipts").glob("tier-delete-*.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text(encoding="utf-8"))["bytes"] >= 5001


def test_a_tracked_lookalike_is_never_treated_as_regenerable(tmp_path):
    import subprocess

    root = tmp_path / "lss"
    case = root / ".state" / "validation" / "case"
    _tree(case, {"pytest-fixture/t.json": b"{}", "pytest-b/x.tmp": b"x"})
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", ".state/validation/case/pytest-fixture/t.json"], check=True)
    tiers._TRACKED.clear()
    assert {p.name for p in tiers.disposables(_policy(tmp_path), root, case)} == {"pytest-b"}

def test_an_interrupted_run_is_recovered_once_from_its_journal(tmp_path, monkeypatch):
    policy = _policy(tmp_path)
    root = tmp_path / "lss"
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "journal.json").write_text('{"source": "x"}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(tiers, "recover", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    tiers.demote(policy, root, apply=True, until_free_gb=1e9, log=lambda *_: None)
    tiers.demote(policy, root, apply=True, until_free_gb=1e9, log=lambda *_: None)
    assert calls == [1]

def _repo_with_worktree(tmp_path):
    import subprocess

    root = tmp_path / "lss"
    root.mkdir()
    run = lambda *a, cwd=root: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)  # noqa: E731
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (root / ".gitignore").write_text(".state/\n__pycache__/\n", encoding="utf-8")
    (root / "code.py").write_text("x = 1\n", encoding="utf-8")
    run("add", ".")
    run("commit", "-q", "-m", "base")
    worktree = root / ".worktrees" / "lane"
    run("worktree", "add", "-q", "-b", "lane", str(worktree))
    (worktree / "code.py").write_text("x = 2\n", encoding="utf-8")
    run("commit", "-q", "-am", "lane work", cwd=worktree)
    return root, worktree


def test_an_idle_clean_worktree_is_retired_with_its_branch_and_results_kept(tmp_path, monkeypatch):
    import subprocess

    root, worktree = _repo_with_worktree(tmp_path)
    _tree(worktree / ".state" / "run", {"result.flac": os.urandom(4000)})
    _tree(worktree / ".state" / "run" / "__pycache__", {"m.pyc": b"c"})
    shared = tmp_path / "shared-venv"
    _tree(shared, {"python.exe": b"py"})
    tiers.make_junction(worktree / ".venv", shared)
    (root / ".gitignore").write_text(".state/\n__pycache__/\n.venv\n", encoding="utf-8")
    (worktree / ".gitignore").write_text(".state/\n__pycache__/\n.venv\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-q", "-am", "ignore venv"], cwd=worktree, check=True, capture_output=True)
    policy = _policy(tmp_path, hot_volume=str(tmp_path.anchor), worktree_idle_hours=0)
    monkeypatch.setattr(tiers, "worktree_idle_hours", lambda path: 1000.0)
    monkeypatch.setattr(tiers, "survey", lambda path, _real=tiers.survey: _aged(_real(path)))
    found = [w for w in tiers.idle_worktrees(policy, root, processes=[]) if Path(w.path) == worktree]
    assert found and found[0].eligible, found
    record = tiers.retire_worktree(policy, root, found[0])
    assert not worktree.exists() and (shared / "python.exe").read_bytes() == b"py"      # link target untouched
    archive = Path(record["archive"])
    assert (archive / ".state" / "run" / "result.flac").exists() and record["regenerable_dropped"] == 1
    log = subprocess.run(["git", "log", "-1", "--format=%s", "lane"], cwd=root, capture_output=True, text=True).stdout
    assert log.strip() == "ignore venv"                                                  # the branch keeps everything


def _aged(tree):
    tree.newest = time.time() - 1000 * 3600
    return tree


def test_a_dirty_worktree_is_kept_until_it_has_been_idle_long_enough(tmp_path, monkeypatch):
    root, worktree = _repo_with_worktree(tmp_path)
    (worktree / "new.txt").write_text("not committed", encoding="utf-8")
    policy = _policy(tmp_path, hot_volume=str(tmp_path.anchor), worktree_idle_hours=0, dirty_worktree_idle_hours=48)
    monkeypatch.setattr(tiers, "worktree_idle_hours", lambda path: 10.0)
    found = {Path(w.path): w for w in tiers.idle_worktrees(policy, root, processes=[])}
    assert not found[worktree].eligible and "uncommitted" in found[worktree].reason
    monkeypatch.setattr(tiers, "worktree_idle_hours", lambda path: 1000.0)
    monkeypatch.setattr(tiers, "survey", lambda path, _real=tiers.survey: _aged(_real(path)))
    found = {Path(w.path): w for w in tiers.idle_worktrees(policy, root, processes=[])}
    assert found[worktree].eligible and found[worktree].reason == "idle; uncommitted work archived"


def test_an_idle_dirty_worktree_is_retired_with_its_uncommitted_work_archived(tmp_path, monkeypatch):
    import subprocess

    root, worktree = _repo_with_worktree(tmp_path)
    (worktree / "new.txt").write_text("not committed", encoding="utf-8")
    (worktree / "code.py").write_text("x = 3\n", encoding="utf-8")
    _tree(worktree / "pkg" / "__pycache__", {"m.pyc": b"c"})
    policy = _policy(tmp_path, hot_volume=str(tmp_path.anchor), worktree_idle_hours=0, dirty_worktree_idle_hours=48)
    monkeypatch.setattr(tiers, "worktree_idle_hours", lambda path: 1000.0)
    monkeypatch.setattr(tiers, "survey", lambda path, _real=tiers.survey: _aged(_real(path)))
    found = [w for w in tiers.idle_worktrees(policy, root, processes=[]) if Path(w.path) == worktree]
    assert found and found[0].eligible, found
    record = tiers.retire_worktree(policy, root, found[0])
    assert not worktree.exists()
    archive = Path(record["archive"])
    assert (archive / "new.txt").read_text(encoding="utf-8") == "not committed"
    assert (archive / "code.py").read_text(encoding="utf-8") == "x = 3\n"
    assert not (archive / "pkg").exists()
    assert sorted(record["uncommitted"]) == ["code.py", "new.txt"] and record["patch"] == "UNCOMMITTED.patch"
    assert tiers.PATCH in record["files"] and "git apply" in record["restore"]
    patch = archive / "UNCOMMITTED.patch"
    assert patch.read_bytes() and (archive / tiers.MANIFEST).exists()
    fresh = tmp_path / "fresh-checkout"
    subprocess.run(["git", "worktree", "add", "-q", str(fresh), "lane"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(patch)], cwd=fresh, check=True, capture_output=True)
    assert (fresh / "code.py").read_text(encoding="utf-8") == "x = 3\n"
    branches = subprocess.run(["git", "branch", "--list", "lane"], cwd=root, capture_output=True, text=True).stdout
    assert "lane" in branches


def test_a_dirty_worktree_whose_archive_fails_verification_is_kept(tmp_path, monkeypatch):
    root, worktree = _repo_with_worktree(tmp_path)
    (worktree / "new.txt").write_text("not committed", encoding="utf-8")
    policy = _policy(tmp_path, hot_volume=str(tmp_path.anchor), worktree_idle_hours=0, dirty_worktree_idle_hours=48)
    monkeypatch.setattr(tiers, "worktree_idle_hours", lambda path: 1000.0)
    monkeypatch.setattr(tiers, "survey", lambda path, _real=tiers.survey: _aged(_real(path)))
    found = [w for w in tiers.idle_worktrees(policy, root, processes=[]) if Path(w.path) == worktree]
    monkeypatch.setattr(tiers, "_hash", lambda path: "bad")
    with pytest.raises(RuntimeError):
        tiers.retire_worktree(policy, root, found[0])
    assert (worktree / "new.txt").exists()


def test_maintenance_runs_only_while_an_agent_is_developing(tmp_path):
    root, _ = _repo_with_worktree(tmp_path)
    policy = _policy(tmp_path, run_only_while_developing={"processes": ["claude.exe", "codex.exe"], "recent_git_hours": 3})
    assert tiers.developing(policy, root, names={"codex.exe"})            # git just changed (fresh repo)
    assert not tiers.developing(policy, root, names={"explorer.exe"})      # no agent running
    old = time.time() - 10 * 3600
    for path in (root / ".git").rglob("*"):
        if path.name in ("index", "HEAD"):
            os.utime(path, (old, old))
    assert not tiers.developing(policy, root, names={"claude.exe"})        # agent app open, but no LSS work


def test_auto_does_nothing_when_nobody_is_developing(tmp_path, monkeypatch):
    monkeypatch.setattr(tiers, "load_policy", lambda: _policy(tmp_path))
    monkeypatch.setattr(tiers, "main_root", lambda: tmp_path)
    monkeypatch.setattr(tiers, "developing", lambda *a, **k: False)
    monkeypatch.setattr(tiers, "candidates", lambda *a, **k: pytest.fail("no disk scan when idle"))
    monkeypatch.setattr(tiers, "free_gb", lambda *a, **k: pytest.fail("no disk access when idle"))
    assert tiers.main(["auto"]) == 0


def test_an_untracked_result_beside_tracked_files_is_still_a_candidate(tmp_path):
    import subprocess

    root = tmp_path / "lss"
    _tree(root / ".state" / "validation" / "case", {"tracked/receipt.json": b"{}", "results/out.wav": b"x" * 100})
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", ".state/validation/case/tracked/receipt.json"], check=True)
    tiers._TRACKED.clear()
    found = {Path(c.path).name: c for c in tiers.candidates(_policy(tmp_path), root, processes=[])}
    assert found["results"].eligible
    assert "tracked" not in found or not found["tracked"].eligible
    assert "case" not in found


def test_a_read_only_git_status_does_not_make_a_worktree_active(tmp_path):
    gitdir = tmp_path / "gitdir"
    gitdir.mkdir()
    old = time.time() - 50 * 3600
    for name in ("HEAD", "index"):
        (gitdir / name).write_text("x")
        os.utime(gitdir / name, (old, old))
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}")
    os.utime(gitdir / "index", None)  # `git status` refreshed the index just now
    assert tiers.worktree_idle_hours(worktree) > 49


def _prune_policy(tmp_path, **extra):
    policy = _policy(tmp_path, delete_package_copies=True, prune_roots=[str(tmp_path / "dev")],
                     model_stores=[str(tmp_path / "store")], frozen_roots=[str(tmp_path / "dev" / "release")],
                     min_idle_hours=1, dedupe_min_mb=0, never=[str(tmp_path / "protected")], **extra)
    policy["disposable"]["names"] = ["__pycache__", ".pytest_cache", "EBWebView"]
    tiers._PROTECTED.clear()
    return policy


def test_prune_deletes_package_copies_but_keeps_release_and_pending_packages(tmp_path):
    root = tmp_path / "lss"
    rel = root / ".state" / "releases" / "r1"
    rel.mkdir(parents=True)
    kept = tmp_path / "dev" / "release" / "build-aaa" / "dist" / "LSS"
    pending = tmp_path / "dev" / "release" / "build-bbb" / "dist" / "LSS"
    (rel / "product-release.json").write_text(json.dumps({"frozen_package": str(kept)}))
    copy = tmp_path / "dev" / "case" / "portable"
    for package in (kept, pending, copy):
        _tree(package, {"LSS.exe": b"exe", "_internal/a.dll": b"dll"})
    (tmp_path / "dev" / "release" / "FROZEN-bbb").mkdir(parents=True)        # no verdict yet
    _tree(tmp_path / "dev" / "case" / "tmp" / "EBWebView", {"cache.bin": b"c"})
    _tree(tmp_path / "dev" / "case" / "evidence", {"result.json": b"{}", "out.flac": b"f"})
    policy = _prune_policy(tmp_path)
    tiers.prune(policy, root, apply=True, log=lambda *_: None)
    assert (kept / "LSS.exe").exists() and (pending / "LSS.exe").exists()
    assert not copy.exists() and not (tmp_path / "dev" / "case" / "tmp" / "EBWebView").exists()
    assert (tmp_path / "dev" / "case" / "evidence" / "out.flac").read_bytes() == b"f"


def test_prune_deletes_only_model_copies_identical_to_a_stored_model(tmp_path):
    root = tmp_path / "lss"
    root.mkdir()
    _tree(tmp_path / "store", {"m/model.ckpt": b"w" * (1024 * 1024)})
    _tree(tmp_path / "dev", {"case/models/model.ckpt": b"w" * (1024 * 1024),
                             "probe/trained.ckpt": b"t" * (1024 * 1024)})
    tiers.prune(_prune_policy(tmp_path), root, apply=True, log=lambda *_: None)
    assert not (tmp_path / "dev" / "case" / "models" / "model.ckpt").exists()
    assert (tmp_path / "dev" / "probe" / "trained.ckpt").exists()
    assert (tmp_path / "store" / "m" / "model.ckpt").exists()


def test_identical_large_files_become_hard_links_with_their_paths_kept(tmp_path):
    root = tmp_path / "lss"
    root.mkdir()
    _tree(tmp_path / "dev", {"a/ep.mkv": b"v" * 4096, "b/ep.mkv": b"v" * 4096, "c/other.mkv": b"o" * 4096})
    report = tiers.dedupe(_prune_policy(tmp_path), [tmp_path / "dev"], apply=True, log=lambda *_: None)
    a, b = tmp_path / "dev" / "a" / "ep.mkv", tmp_path / "dev" / "b" / "ep.mkv"
    assert len(report["linked"]) == 1 and a.read_bytes() == b.read_bytes() == b"v" * 4096
    assert os.stat(a).st_ino == os.stat(b).st_ino
    assert os.stat(tmp_path / "dev" / "c" / "other.mkv").st_nlink == 1


def test_idle_threshold_relaxes_only_under_space_pressure():
    import storage_tiers as t
    policy = {"min_idle_hours": 6, "pressure_free_gb": 40, "pressure_min_idle_hours": 1, "hot_volume": ('C' + chr(58) + '\\')}
    assert t.idle_threshold_hours(policy, 39.9) == 1
    assert t.idle_threshold_hours(policy, 40.0) == 6
    assert t.idle_threshold_hours({"min_idle_hours": 6}, 5.0) == 6


def _git_run(cwd, *args, env=None):
    import subprocess
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout


def _old_commit(cwd, message, days=10):
    stamp = f"{int(time.time()) - days * 86400} +0000"
    env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    _git_run(cwd, "commit", "-q", "--allow-empty", "-m", message, env=env)


def _retire_fixture(tmp_path, monkeypatch):
    """A repo with worktrees: old-clean (retire), fresh-commit, dirty, locked, current (cwd), deployed build, open-frozen
    build, closed-frozen build, and a detached build kept by no branch."""
    root, old = _repo_with_worktree(tmp_path)
    _old_commit(old, "old lane work")
    gi = root / ".worktrees"
    for name in ("fresh", "dirty", "locked", "current"):
        _git_run(root, "worktree", "add", "-q", "-b", name, str(gi / name), "lane")
    _old_commit(gi / "fresh", "just now", days=0)
    _old_commit(gi / "dirty", "old")
    _old_commit(gi / "locked", "old")
    _old_commit(gi / "current", "old")
    (gi / "dirty" / "wip.txt").write_text("uncommitted", encoding="utf-8")
    _git_run(root, "worktree", "lock", str(gi / "locked"))
    builds = tmp_path / "builds"
    ids = {}
    for name in ("deployed", "open", "closed"):
        _git_run(old, "checkout", "-q", "lane")
        _old_commit(old, f"build {name}")
        sha = _git_run(old, "rev-parse", "HEAD").strip()
        ids[name] = sha[:7]
        _git_run(root, "branch", f"keep-{name}", sha)
        _git_run(root, "worktree", "add", "-q", "--detach", str(builds / f"build-{sha[:7]}"), sha)
    _git_run(old, "checkout", "-q", "lane")
    _git_run(root, "worktree", "add", "-q", "--detach", str(builds / "build-feedbee"), "lane~0")
    _old_commit(builds / "build-feedbee", "orphan commit")                       # kept by no branch
    release = tmp_path / "frozen"
    (release / f"FROZEN-{ids['open']}").mkdir(parents=True)                      # no verdict file yet
    (release / f"FROZEN-{ids['closed']}").mkdir()
    (release / f"FROZEN-{ids['closed']}" / "VERDICT-PASS.md").write_text("pass", encoding="utf-8")
    (root / ".state" / "releases").mkdir(parents=True)
    (root / ".state" / "releases" / "current-product.json").write_text(json.dumps(
        {"candidate_code_commit": _git_run(root, "rev-parse", f"keep-deployed").strip(), "release_id": "r-1",
         "user_change": f"mentions {ids['closed']} only in prose"}), encoding="utf-8")
    policy = _policy(tmp_path, frozen_roots=[str(release)], worktree_idle_hours=1)
    monkeypatch.setattr(tiers, "worktree_idle_hours", lambda path: 1000.0)
    monkeypatch.setattr(tiers, "survey", lambda path, _real=tiers.survey: _aged(_real(path)))
    tiers._PROTECTED.clear()
    return root, policy, ids


def test_retire_lists_only_clean_idle_unprotected_worktrees_on_every_volume(tmp_path, monkeypatch):
    root, policy, ids = _retire_fixture(tmp_path, monkeypatch)
    policy["hot_volume"] = ('Z' + chr(58) + '\\')                    # nothing is on the hot volume: only all_volumes sees them
    assert tiers.idle_worktrees(policy, root, processes=[], cwd=tmp_path) == []
    found = {Path(w.path).name: w for w in tiers.idle_worktrees(
        policy, root, processes=[], all_volumes=True, commit_hours=72, cwd=root / ".worktrees" / "current")}
    verdict = {name: (w.eligible, w.reason) for name, w in found.items()}
    assert verdict["lane"][0] and verdict[f"build-{ids['closed']}"][0], verdict
    assert not verdict["fresh"][0] and "commit" in verdict["fresh"][1]            # no commit for < 72 h
    assert not verdict["dirty"][0] and "uncommitted" in verdict["dirty"][1]
    assert not verdict["locked"][0] and verdict["locked"][1] == "locked"
    assert not verdict["current"][0] and "current session" in verdict["current"][1]
    assert not verdict[f"build-{ids['deployed']}"][0] and "deployed" in verdict[f"build-{ids['deployed']}"][1]
    assert not verdict[f"build-{ids['open']}"][0] and "frozen" in verdict[f"build-{ids['open']}"][1]
    assert not verdict["build-feedbee"][0] and "not kept by any branch" in verdict["build-feedbee"][1]
    assert root.name not in found                                                 # the main checkout is never listed


def test_retire_keeps_a_worktree_a_running_process_names(tmp_path, monkeypatch):
    root, policy, ids = _retire_fixture(tmp_path, monkeypatch)
    lane = root / ".worktrees" / "lane"
    found = {Path(w.path).name: w for w in tiers.idle_worktrees(
        policy, root, processes=[f"python.exe\tpython -c x --cwd {str(lane).lower()}"], all_volumes=True,
        commit_hours=72, cwd=tmp_path)}
    assert not found["lane"].eligible and "running process" in found["lane"].reason


def test_retire_cli_is_a_dry_run_until_apply_and_keeps_the_branch(tmp_path, monkeypatch, capsys):
    root, policy, ids = _retire_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(tiers, "load_policy", lambda: policy)
    monkeypatch.setattr(tiers, "main_root", lambda: root)
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    monkeypatch.chdir(root / ".worktrees" / "current")
    lane = root / ".worktrees" / "lane"
    _tree(lane / ".state" / "run", {"result.flac": b"r" * 3000})
    assert tiers.main(["worktrees", "--retire"]) == 0
    out = capsys.readouterr().out
    assert "RETIRE" in out and "dry run" in out and lane.exists()
    assert tiers.main(["worktrees", "--retire", "--apply"]) == 0
    assert not lane.exists() and (root / ".worktrees" / "dirty").exists() and (root / ".worktrees" / "locked").exists()
    assert "lane" in _git_run(root, "branch", "--list", "lane")
    journal = Path(policy["archive_root"]) / "index.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert any(r.get("action") == "retire_worktree" and r["branch"] == "lane" and r["source"] == str(lane) for r in rows)
    assert any((a.name == "result.flac") for a in Path(policy["archive_root"]).rglob("result.flac"))


def test_hygiene_scan_finds_dirty_skips_builds_and_caches_clean_results(tmp_path, monkeypatch):
    root, policy, ids = _retire_fixture(tmp_path, monkeypatch)
    def rows(**kw):
        return {Path(r["path"]).name: r for r in tiers.hygiene_scan(policy, root, **kw)}
    first = rows()
    assert first["dirty"]["dirty"] == 1 and first["lane"]["dirty"] == 0 and not first["lane"]["cached"]
    assert first[f"build-{ids['open']}"]["state"] == "skipped" and first["build-feedbee"]["state"] == "skipped"
    assert "keep-closed" in first["lane"]["contained"]
    second = rows()
    assert second["lane"]["cached"] and not second["dirty"]["cached"]            # clean cached, dirty re-checked
    (root / ".worktrees" / "lane" / "x.txt").write_text("new", encoding="utf-8")
    assert tiers.hygiene_scan(policy, root, ttl=0)                               # expired ttl rescans
    assert rows(include_builds=True)[f"build-{ids['open']}"]["state"] == "scanned"


def test_an_old_release_moves_but_the_deployed_release_and_its_rollback_target_stay(tmp_path):
    root = tmp_path / "lss"
    releases = root / ".state" / "releases"
    for name in ("release-new", "design-rollback", "integrated-old"):
        _tree(releases / name, {"rollback/LSS.exe": b"x" * 1000})
    (releases / "current-product.json").write_text(json.dumps(
        {"release_id": "release-new", "rollback": {"target_release_id": "design-rollback"}}), encoding="utf-8")
    rule = {"glob": ".state/releases/*", "class": "release", "keep_named_in": ".state/releases/current-product.json"}
    found = {Path(c.path).name: c for c in tiers.candidates(_policy(tmp_path, candidates=[rule], never=[]), root, processes=[])}
    assert found["integrated-old"].eligible
    assert not found["release-new"].eligible and not found["design-rollback"].eligible
    (releases / "current-product.json").unlink()            # unreadable record: nothing it guards moves
    found = tiers.candidates(_policy(tmp_path, candidates=[rule], never=[]), root, processes=[])
    assert not any(c.eligible for c in found)


def test_superseded_model_revisions_and_abandoned_downloads_are_deleted_the_live_one_kept(tmp_path, monkeypatch):
    store = tmp_path / "models"
    _tree(store / "net-r2", {"w.ckpt": b"a" * 2000, ".staging-r2/w.zip.part": b"p" * 500})
    _tree(store / "net-r3", {"w.ckpt": b"b" * 2000})
    _tree(store / "other-r1", {"w.ckpt": b"c" * 2000})
    _tree(store / "other-r2", {".staging-r2/w.zip.part": b"p" * 10}, age_hours=0.0)   # downloading now
    _tree(store / "big-r1", {"w.ckpt": b"d" * 2000, ".staging-r1/x.part": b"p"})
    policy = _policy(tmp_path, model_stores=[str(store)], superseded_models=True)
    gone = tiers.superseded_models(policy, [], time.time() - 3600)
    assert sorted(p.relative_to(store).as_posix() for p in gone) == ["big-r1/.staging-r1", "net-r2"]
    assert tiers.superseded_models(policy, [str(store / "net-r2").lower()], time.time() - 3600) == [store / "big-r1" / ".staging-r1"]


def test_idle_temp_extractions_are_deleted_and_recent_ones_kept(tmp_path, monkeypatch):
    _tree(tmp_path / "temp" / "_MEI111", {"a.pyd": b"x"}, age_hours=48)
    _tree(tmp_path / "temp" / "_MEI222", {"a.pyd": b"x"}, age_hours=1)
    _tree(tmp_path / "temp" / "other", {"a": b"x"}, age_hours=48)
    policy = _policy(tmp_path, temp_orphans={"root": str(tmp_path / "temp"), "prefixes": ["_MEI"], "min_idle_hours": 24})
    assert [p.name for p in tiers.temp_orphans(policy)] == ["_MEI111"]
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    report = tiers.clean_hot(policy, apply=True, log=lambda *_: None)
    assert [Path(r["path"]).name for r in report["deleted"]] == ["_MEI111.lss-delete"]
    assert sorted(p.name for p in (tmp_path / "temp").iterdir()) == ["_MEI222", "other"]


def test_any_stale_temp_entry_goes_and_a_young_one_or_a_junction_stays(tmp_path, monkeypatch):
    temp = tmp_path / "temp"
    _tree(temp / "old-dir", {"a": b"x", "sub/b": b"y"}, age_hours=200)
    _tree(temp / "young-dir", {"a": b"x"}, age_hours=10)
    _tree(temp / "mixed-dir", {"old": b"x"}, age_hours=200)
    (temp / "mixed-dir" / "new").write_bytes(b"n")
    for name, age in (("old.log", 200), ("young.log", 10)):
        (temp / name).write_bytes(b"x")
        os.utime(temp / name, (time.time() - age * 3600,) * 2)
    target = tmp_path / "elsewhere"
    _tree(target, {"keep": b"k"}, age_hours=500)
    tiers.make_junction(temp / "link", target)
    os.utime(temp / "link", (time.time() - 500 * 3600,) * 2)
    policy = _policy(tmp_path, temp_orphans=[
        {"root": str(temp), "prefixes": ["_MEI"], "min_idle_hours": 24},
        {"root": str(temp), "prefixes": [], "min_idle_hours": 168}])
    assert sorted(p.name for p in tiers.temp_orphans(policy)) == ["old-dir", "old.log"]
    monkeypatch.setattr(tiers, "process_texts", lambda: [])
    tiers.clean_hot(policy, apply=True, log=lambda *_: None)
    assert sorted(p.name for p in temp.iterdir()) == ["link", "mixed-dir", "young-dir", "young.log"]
    assert (target / "keep").exists()


def test_a_kept_release_keeps_its_children_and_a_generic_child_name_does_not_hold_an_old_one(tmp_path):
    root = tmp_path / "lss"
    releases = root / ".state" / "releases"
    _tree(releases / "release-new" / "rollback", {"LSS.exe": b"x" * 1000})
    _tree(releases / "integrated-old" / "rollback", {"LSS.exe": b"x" * 1000})
    tiers.make_junction(releases / "release-new" / "linked", tmp_path)          # a link makes children judged one by one
    tiers.make_junction(releases / "integrated-old" / "linked", tmp_path)
    (releases / "current-product.json").write_text('{"release_id": "release-new", "rollback": "x\\rollback\\y"}',
                                                   encoding="utf-8")
    rule = {"glob": ".state/releases/*", "class": "release", "keep_named_in": ".state/releases/current-product.json"}
    found = {str(Path(c.path).relative_to(releases)): c.eligible
             for c in tiers.candidates(_policy(tmp_path, candidates=[rule], never=[]), root, processes=[])}
    assert found == {str(Path("integrated-old", "rollback")): True, str(Path("release-new", "rollback")): False}



def test_public_policy_selects_no_user_cleanup():
    policy = json.loads((Path(__file__).resolve().parents[1] / "scripts" / "storage_policy.json").read_text(encoding="utf-8"))
    assert policy["user_cleanups"] == []
    assert "NuGet global-packages is never a cleanup target" in policy["user_cleanups_about"]


def test_user_cleanups_run_only_where_installed_and_at_most_once_per_interval(tmp_path, monkeypatch):
    rule = {"name": "tool", "exists": str(tmp_path / "tool" / "apps"), "every_hours": 24, "commands": ["echo cleanup"]}
    policy = _policy(tmp_path, user_cleanups=[rule])
    assert tiers.cleanups_due(policy) == []                  # tool not installed
    (tmp_path / "tool" / "apps").mkdir(parents=True)
    assert tiers.cleanups_due(policy) == [rule]
    calls = []
    monkeypatch.setattr(tiers.subprocess, "run", lambda command, **_: calls.append(command) or type("R", (), {"returncode": 0})())
    tiers.run_cleanup(policy, rule, log=lambda *_: None)
    assert calls and "echo cleanup" in calls[0][-1] and tiers.cleanups_due(policy) == []


def test_aged_cleanup_deletes_only_old_matching_entries_and_never_through_a_link(tmp_path):
    dumps = tmp_path / "dumps"
    _tree(dumps, {"old.dmp": b"x" * 100, "old.txt": b"keep"}, age_hours=24 * 30)
    _tree(dumps, {"new.dmp": b"x"}, age_hours=1)
    tiers.make_junction(dumps / "linked.dmp", tmp_path)
    rule = {"name": "dumps", "exists": str(dumps), "every_hours": 24, "older_than_days": 14, "glob": "*.dmp"}
    tiers.run_cleanup(_policy(tmp_path), rule, log=lambda *_: None)
    assert sorted(p.name for p in dumps.iterdir()) == ["linked.dmp", "new.dmp", "old.txt"]
