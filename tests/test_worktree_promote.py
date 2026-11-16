"""Worktree placement: `storage_tiers.py worktrees --report`, `promote <worktree>` (refusals, verified move, git
registration, legacy junction) and the dev_hygiene.ps1 detection of active lanes on the HDD (scripts/hdd_lanes.ps1).

Temp directories only. The "HDD" is the policy's hot volume pointing at a drive letter that does not exist, so every
temp path counts as off the SSD; C: free space is faked. Nothing is written to E:."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.machine_config_fixtures import machine_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import storage_tiers as tiers  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "nt", reason="junctions and directory-rename locking are Windows behaviour")
HYGIENE_LANES = Path(__file__).resolve().parents[1] / "scripts" / "hdd_lanes.ps1"


def git(cwd: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    if date:
        env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = date
    out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    main = tmp_path / "main"
    main.mkdir()
    git(main, "init", "-b", "main")
    (main / "tracked.txt").write_text("one\n", encoding="utf-8")
    (main / ".gitignore").write_text(".pytest_cache/\n__pycache__/\n*.log\n", encoding="utf-8")
    git(main, "add", "-A")
    git(main, "commit", "-m", "init")
    free = {"gb": 500.0}
    monkeypatch.setattr(tiers, "free_gb", lambda volume: free["gb"])
    policy = {**machine_policy(tmp_path), "schema_version": 1, "hot_volume": ('Z' + chr(58) + '\\'), "local_state_root": str(tmp_path / "local"),
              "archive_root": str(tmp_path / "archive"), "receipts_root": str(tmp_path / "receipts"),
              "disposable": {"names": ["__pycache__", ".pytest_cache"], "prefixes": ["pytest-of-"]},
              "worktree_promote": {"root": str(tmp_path / "wt"), "min_free_after_gb": 80, "active_commit_hours": 48}}
    (tmp_path / "archive").mkdir()
    return main, policy, free, tmp_path


def lane(main: Path, hdd: Path, name: str, **kw) -> Path:
    path = hdd / name
    git(main, "worktree", "add", "-b", f"claude/{name}", str(path), **kw)
    return path


def plain(main, policy, path, **kw):
    return tiers.promote_worktree(policy, main, path, processes=[], cwds=([], 0), log=lambda *_: None, **kw)


def test_dry_run_reports_the_plan_and_changes_nothing(repo):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "alpha")
    (wt / "scratch.txt").write_text("x" * 1000, encoding="utf-8")
    before = git(main, "worktree", "list", "--porcelain")
    record = plain(main, policy, wt)
    assert record["status"] == "DRY_RUN_WOULD_PROMOTE" and not record["refusals"]
    assert record["destination"] == str(tmp / "wt" / "alpha") and record["bytes"] >= 1000
    assert not (tmp / "wt").exists() and git(main, "worktree", "list", "--porcelain") == before


def test_refusals_leave_everything_untouched(repo, monkeypatch):
    main, policy, free, tmp = repo
    wt = lane(main, tmp / "hdd", "beta")
    refused = lambda path, **kw: plain(main, policy, path, **kw)["refusals"]   # noqa: E731
    assert "the main checkout never moves" in refused(main)
    assert refused(tmp / "nowhere") == ["not a registered worktree"]
    # a process uses it: its current directory inside, or its command line naming it
    inside = ([(4242, "python.exe", str(wt / "sub"))], 0)
    assert any("in use" in r and "4242" in r for r in tiers.promote_worktree(policy, main, wt, processes=[], cwds=inside)["refusals"])
    named = [f"node.exe\tnode {str(wt).lower()}\\x.js"]
    assert any("names it" in r for r in tiers.promote_worktree(policy, main, wt, processes=named, cwds=([], 0))["refusals"])
    mover = [f"python.exe\tpython storage_tiers.py promote {str(wt).lower()}"]
    assert tiers.promote_worktree(policy, main, wt, processes=mover, cwds=([], 0))["status"] == "DRY_RUN_WOULD_PROMOTE"
    # C: would end below the floor
    free["gb"] = 80.0001
    (wt / "big.bin").write_bytes(b"0" * 200_000_000)
    assert any("below the floor" in r for r in refused(wt))
    free["gb"] = 500.0
    # already on the SSD
    monkeypatch.setattr(tiers, "on_hot_volume", lambda path, policy: True)
    assert "already on the SSD" in refused(wt)
    monkeypatch.undo()
    monkeypatch.setattr(tiers, "free_gb", lambda volume: free["gb"])
    # locked, detached build checkout, frozen candidate, destination exists
    git(main, "worktree", "lock", str(wt))
    assert "locked" in refused(wt)
    git(main, "worktree", "unlock", str(wt))
    build = tmp / "claude-release" / "build-abc1234"
    git(main, "worktree", "add", "--detach", str(build))
    assert any("build checkout" in r for r in refused(build))
    frozen = tmp / "claude-release" / "FROZEN-abc1234" / "inbox" / "src"
    git(main, "worktree", "add", "--detach", str(frozen))
    assert "inside a frozen candidate" in refused(frozen)
    (tmp / "wt" / "beta").mkdir(parents=True)
    assert any("destination exists" in r for r in refused(wt))
    assert wt.is_dir() and (wt / "tracked.txt").is_file()


def test_apply_moves_a_verified_copy_keeps_git_consistent_and_leaves_the_old_path_working(repo):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "gamma")
    shared = tmp / "shared-venv"
    shared.mkdir()
    (shared / "python.txt").write_text("env", encoding="utf-8")
    tiers.make_junction(wt / ".venv", shared)                     # worktrees link the shared environment
    (wt / "tracked.txt").write_text("two\n", encoding="utf-8")   # uncommitted change
    (wt / "notes" / "deep").mkdir(parents=True)
    (wt / "notes" / "deep" / "new.txt").write_bytes(os.urandom(5000))   # untracked
    (wt / ".pytest_cache").mkdir()
    (wt / ".pytest_cache" / "v").write_text("regenerable", encoding="utf-8")
    status_before = git(wt, "status", "--porcelain=v1", "--untracked-files=all")
    record = plain(main, policy, wt, apply=True)
    destination = tmp / "wt" / "gamma"
    assert record["status"] == "WORKTREE_PROMOTED_VERIFIED_LEGACY_JUNCTION", record["refusals"]
    # the registered path is the real new location; the old path is a junction to it
    listed = [Path(line.split(" ", 1)[1]).resolve() for line in git(main, "worktree", "list", "--porcelain").splitlines()
              if line.startswith("worktree ")]
    assert destination.resolve() in listed and wt.resolve() == destination.resolve()
    assert tiers.is_link(wt) and not tiers.is_link(destination) and (wt / "tracked.txt").read_text(encoding="utf-8") == "two\n"
    assert git(destination, "status", "--porcelain=v1", "--untracked-files=all") == status_before
    assert git(wt, "rev-parse", "--abbrev-ref", "HEAD").strip() == "claude/gamma"
    assert (destination / "notes" / "deep" / "new.txt").stat().st_size == 5000
    assert not (destination / ".pytest_cache").exists()               # regenerable, left behind
    assert tiers.is_link(destination / ".venv") and (destination / ".venv" / "python.txt").read_text(encoding="utf-8") == "env"
    assert (shared / "python.txt").exists()                           # the link's target is never touched
    assert not list((tmp / "hdd").glob("*promoted*"))                 # the renamed original is deleted
    receipt = json.loads(Path(record["receipt"]).read_text(encoding="utf-8"))
    assert receipt["action"] == "promote_worktree" and receipt["file_count"] == len(receipt["files"]) and receipt["legacy_junction"] == str(wt)
    git(main, "worktree", "prune")
    assert destination.resolve() in [Path(line.split(" ", 1)[1]).resolve() for line in git(main, "worktree", "list", "--porcelain").splitlines()
                                     if line.startswith("worktree ")]
    assert plain(main, policy, wt)["status"] == "REFUSED"             # the old path names the SSD copy now: nothing to move


def test_a_file_held_open_refuses_the_move_at_apply_time(repo):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "delta")
    with open(wt / "tracked.txt", "rb"):                              # no psutil: Windows refuses to rename the directory
        record = plain(main, policy, wt, apply=True)
    assert record["status"] == "REFUSED" and any("holds files" in r for r in record["refusals"])
    assert wt.is_dir() and not tiers.is_link(wt) and not (tmp / "wt" / "delta").exists()
    assert not list((tmp / "hdd").glob("*probe*"))


def test_a_copy_that_fails_verification_keeps_the_original_and_its_registration(repo, monkeypatch):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "epsilon")
    (wt / "data.bin").write_bytes(os.urandom(4000))
    before = git(main, "worktree", "list", "--porcelain")
    monkeypatch.setattr(tiers, "_hash", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="verification failed"):
        plain(main, policy, wt, apply=True)
    assert not tiers.is_link(wt) and (wt / "data.bin").stat().st_size == 4000
    assert not (tmp / "wt" / "epsilon").exists() and not list((tmp / "wt").glob("*partial*"))
    assert git(main, "worktree", "list", "--porcelain") == before


def test_report_names_drive_commit_age_and_users(repo):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "zeta")
    rows = {Path(r["path"]).name: r for r in tiers.worktree_placement(
        policy, main, processes=[], cwds=([(7, "pwsh.exe", str(wt))], 3))}
    assert rows["zeta"]["hot"] is False and rows["zeta"]["commit_hours"] < 1 and rows["zeta"]["users"] == ["cwd pwsh.exe pid 7"]
    assert rows["main"]["main"] and rows["zeta"]["unreadable_processes"] == 3
    assert tiers.on_hot_volume(wt, {**policy, "hot_volume": os.path.splitdrive(str(wt))[0] + "\\"})
    cwds, unreadable = tiers.process_cwds()
    assert any(cwd.lower() == os.getcwd().lower().rstrip("\\") for pid, _, cwd in cwds if pid == os.getpid())


def lanes_via_powershell(main: Path, hot_root: str, hours: int = 48) -> list[dict]:
    command = (f". '{HYGIENE_LANES}'; $r = @(Get-HddActiveLanes -Root '{main}' -MainRoot '{main}' -HotRoot '{hot_root}' -Hours {hours}); "
               "ConvertTo-Json -InputObject $r -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout or "[]")


def test_hygiene_flags_recent_lanes_physically_off_the_hot_volume(repo):
    main, _, _, tmp = repo
    fresh = lane(main, tmp / "hdd", "recent-lane")
    old = tmp / "hdd" / "old-lane"
    git(main, "worktree", "add", "-b", "claude/old-lane", str(old))
    (old / "x.txt").write_text("x", encoding="utf-8")
    git(old, "add", "-A")
    git(old, "commit", "-m", "old", date=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 5 * 86400)) + "+0000")
    git(main, "branch", "claude/no-worktree")
    drive = os.path.splitdrive(str(tmp))[0] + "\\"
    flagged = lanes_via_powershell(main, ('Z' + chr(58) + '\\'))
    assert [Path(f["Path"]).name for f in flagged] == ["recent-lane"] and flagged[0]["Branch"] == "claude/recent-lane"
    assert 0 <= flagged[0]["AgeHours"] < 1
    assert sorted(Path(f["Path"]).name for f in lanes_via_powershell(main, ('Z' + chr(58) + '\\'), hours=24 * 30)) == ["old-lane", "recent-lane"]
    assert lanes_via_powershell(main, drive) == []                    # on the hot volume: nothing to say
    assert fresh.is_dir()


def test_only_regenerable_directories_are_left_behind_never_files_git_tracks_or_evidence_by_name(repo):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "eta")
    tracked_xml = wt / "release-evidence" / "opus" / "pytest-targeted-c92.xml"
    tracked_xml.parent.mkdir(parents=True)
    tracked_xml.write_text("<evidence/>", encoding="utf-8")
    tracked_cache = wt / "pkg" / "__pycache__" / "kept.txt"          # a regenerable NAME holding a tracked file
    tracked_cache.parent.mkdir(parents=True)
    tracked_cache.write_text("tracked", encoding="utf-8")
    git(wt, "add", "-f", str(tracked_xml), str(tracked_cache))
    git(wt, "commit", "-m", "evidence")
    (wt / "pytest-run.log").write_text("untracked ignored evidence log", encoding="utf-8")   # *.log is ignored
    (wt / "pytest-notes.txt").write_text("untracked evidence", encoding="utf-8")
    (wt / "pytest-of-user").mkdir()                                  # untracked, NOT ignored: copied, status must match
    (wt / "pytest-of-user" / "x").write_text("x", encoding="utf-8")
    (wt / ".pytest_cache").mkdir()                                   # ignored regenerable directory: left behind
    (wt / ".pytest_cache" / "v").write_text("v", encoding="utf-8")
    (wt / "__pycache__").mkdir()
    (wt / "__pycache__" / "m.pyc").write_bytes(b"pyc")
    scan = tiers._promotion_scan(policy, wt)
    assert sorted(scan["skipped"]) == [".pytest_cache", "__pycache__"]
    record = plain(main, policy, wt, apply=True)
    destination = tmp / "wt" / "eta"
    assert record["status"] == "WORKTREE_PROMOTED_VERIFIED_LEGACY_JUNCTION", record["refusals"]
    for kept in (tracked_xml, tracked_cache, wt / "pytest-run.log", wt / "pytest-notes.txt", wt / "pytest-of-user" / "x"):
        assert (destination / kept.relative_to(wt)).exists(), kept
    assert not (destination / ".pytest_cache").exists() and not (destination / "__pycache__").exists()


def test_an_original_that_changed_during_the_copy_is_kept(repo, monkeypatch):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "theta")
    (wt / "data.bin").write_bytes(b"a" * 3000)
    real_copy = tiers.copy_verified

    def copy_then_write(*args, **kwargs):
        rows = real_copy(*args, **kwargs)
        (wt / "late.txt").write_text("written while copying", encoding="utf-8")
        return rows

    monkeypatch.setattr(tiers, "copy_verified", copy_then_write)
    record = plain(main, policy, wt, apply=True)
    assert record["status"] == "WORKTREE_PROMOTED_ORIGINAL_CHANGED_KEPT" and "late.txt" in record["changed_during_copy"]
    kept = list((tmp / "hdd").glob("theta.lss-promoted-*"))
    assert len(kept) == 1 and (kept[0] / "late.txt").read_text(encoding="utf-8") == "written while copying"
    assert record["leftover_original"] and tiers.is_link(wt) and (tmp / "wt" / "theta" / "data.bin").exists()


def registered(main: Path) -> list[Path]:
    return [Path(line.split(" ", 1)[1]).resolve() for line in git(main, "worktree", "list", "--porcelain").splitlines()
            if line.startswith("worktree ")]


def test_a_failed_rename_or_junction_restores_the_original_and_its_registration(repo, monkeypatch):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "iota")
    (wt / "data.bin").write_bytes(b"d" * 2000)
    real_rename, real_junction = os.rename, tiers.make_junction

    def rename(src, dst):
        if ".lss-promoted-" in str(dst):
            raise OSError("simulated rename failure")
        return real_rename(src, dst)

    monkeypatch.setattr(tiers.os, "rename", rename)
    with pytest.raises(OSError, match="simulated rename"):
        plain(main, policy, wt, apply=True)
    assert wt.resolve() in registered(main) and not tiers.is_link(wt) and (wt / "data.bin").stat().st_size == 2000
    assert not (tmp / "wt" / "iota").exists()
    monkeypatch.setattr(tiers.os, "rename", real_rename)

    def junction(link, target):
        if Path(link) == wt:
            raise OSError("simulated junction failure")
        return real_junction(link, target)

    monkeypatch.setattr(tiers, "make_junction", junction)
    with pytest.raises(OSError, match="simulated junction"):
        plain(main, policy, wt, apply=True)
    assert wt.resolve() in registered(main) and not tiers.is_link(wt) and (wt / "data.bin").stat().st_size == 2000
    assert not (tmp / "wt" / "iota").exists() and not list((tmp / "hdd").glob("*promoted*"))
    assert git(wt, "status", "--porcelain=v1") == "?? data.bin\n"


def test_the_copy_is_not_deleted_when_the_registration_cannot_be_proven_back(repo, monkeypatch):
    main, policy, _, tmp = repo
    wt = lane(main, tmp / "hdd", "kappa")
    (wt / "data.bin").write_bytes(b"k" * 2000)
    real_entries, bad = tiers.worktree_entries, {"on": False}
    monkeypatch.setattr(tiers, "worktree_entries", lambda root: [] if bad["on"] else real_entries(root))

    def broken_hash(path):
        bad["on"] = True                                            # from here the registration cannot be verified
        return "0" * 64

    monkeypatch.setattr(tiers, "_hash", broken_hash)
    with pytest.raises(RuntimeError, match="registration was not restored"):
        plain(main, policy, wt, apply=True)
    assert list((tmp / "wt").glob("kappa*")), "the copy must be kept for a manual repair"
    assert (wt / "data.bin").stat().st_size == 2000
