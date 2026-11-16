"""scripts/state_status.py: the generated STATE.md block (fixture repo; no real .state needed)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import state_status as ss

NOW = 1_800_000_000.0


def _git(repo: Path, *args: str, when: float | None = None) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    if when is not None:
        env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = f"{int(when)} +0000"
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env).stdout.strip()


def _commit(repo: Path, name: str, subject: str, when: float) -> str:
    (repo / name).write_text(subject, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", subject, when=when)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "docs" / "current").mkdir(parents=True)
    (root / ".state" / "releases").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    deployed = _commit(root, "a.txt", "deployed code", NOW - 10 * 86400)
    exe = tmp_path / "LSS.exe"
    exe.write_bytes(b"installed-exe")
    rec = {"release_id": "release-x", "candidate_code_commit": deployed, "completed_at": "2026-10-18T00:37:01+00:00",
           "deployed_executable": str(exe), "exe_sha256": hashlib.sha256(b"installed-exe").hexdigest(),
           "rollback": {"target_release_id": "design-old"}}
    (root / ".state" / "releases" / "current-product.json").write_text(json.dumps(rec), encoding="utf-8")
    (root / "docs" / "current" / "STATE.md").write_text(
        f"# State\n\nhand written\n{ss.BEGIN}\nold\n{ss.END}\ntail\n", encoding="utf-8", newline="\n")
    # a stub Installed Product renderer so the fixture does not depend on the real repository
    monkeypatch.setattr(ss._installed, "render", lambda r: ss._installed.BEGIN + "\ninst\n" + ss._installed.END)
    return root, exe, deployed


def _branch(root: Path, name: str, when: float) -> str:
    _git(root, "checkout", "-q", "-b", name, "main")
    sha = _commit(root, name.replace("/", "_") + ".txt", f"work on {name}", when)
    _git(root, "checkout", "-q", "main")
    return sha


def test_block_contents_and_verified_exe(repo):
    root, _exe, deployed = repo
    _branch(root, "claude/lane-a", NOW - 3600 * 3)
    _branch(root, "codex/old-lane", NOW - 100 * 3600)
    block = ss.render(root, NOW, "")
    assert block.startswith(ss.BEGIN) and block.endswith(ss.END)
    assert "Generated 2027-01-15" in block
    assert f"`release-x` code `{deployed[:7]}`, 2026-10-18; rollback target `design-old`" in block
    assert "**verified** against the file" in block
    assert "claude/lane-a" in block and " 3h " in block
    assert "codex/old-lane" not in block  # older than 48 h
    assert "Undeployed release candidates: none." in block


def test_exe_mismatch_and_unreadable(repo):
    root, exe, _ = repo
    exe.write_bytes(b"tampered")
    assert "**MISMATCH**" in ss.render(root, NOW, "")
    exe.unlink()
    assert "**file unreadable**" in ss.render(root, NOW, "")


def test_candidates_markers_and_deployed_ancestor_skipped(repo):
    root, _exe, deployed = repo
    _git(root, "branch", "claude/release-candidate-merged", deployed)  # ancestor of the deployed code: not listed
    sha = _branch(root, "claude/release-candidate-new", NOW - 60)
    wt = root.parent / "wt-cand"
    _git(root, "worktree", "add", "-q", str(wt), "claude/release-candidate-new")
    block = ss.render(root, NOW, "")
    assert "release-candidate-merged" not in block
    assert f"`claude/release-candidate-new` `{sha[:7]}`: full-suite no, verdict no" in block
    (wt / "FULL-SUITE-DONE.txt").write_text("ok", encoding="utf-8")
    (wt / "FROZEN-abc" / "inbox-from-verifier").mkdir(parents=True)
    (wt / "FROZEN-abc" / "inbox-from-verifier" / "VERDICT-PASS.md").write_text("ok", encoding="utf-8")
    assert "full-suite yes, verdict yes" in ss.render(root, NOW, "")
    _git(root, "branch", "claude/release-candidate-nowt", sha)
    block = ss.render(root, NOW, "")
    assert "`claude/release-candidate-nowt`" in block and "no worktree" in block


def test_write_then_check_roundtrip_ignores_timestamp_and_lanes(repo):
    root, _exe, _ = repo
    _branch(root, "claude/lane-a", NOW - 3600)
    assert ss.process(root, check=True, write=False, now=NOW)[0] == 1  # placeholder block is stale
    code, msg = ss.process(root, check=False, write=True, now=NOW)
    assert code == 0 and "updated" in msg
    text = (root / "docs" / "current" / "STATE.md").read_text(encoding="utf-8")
    assert text.startswith("# State\n\nhand written\n") and text.endswith(f"{ss.END}\ntail\n")
    assert ss.process(root, check=True, write=False, now=NOW)[0] == 0
    # later time + a new lane: only the timestamp and lane rows move, so the block is still current
    _branch(root, "claude/lane-b", NOW + 1000)
    assert ss.process(root, check=True, write=False, now=NOW + 2000)[0] == 0
    # a durable fact changes: stale
    _branch(root, "claude/release-candidate-z", NOW + 10)
    code, msg = ss.process(root, check=True, write=False, now=NOW + 2000)
    assert code == 1 and "stale" in msg
    ss.process(root, check=False, write=True, now=NOW + 2000)
    assert ss.process(root, check=True, write=False, now=NOW + 2000)[0] == 0


def test_lanes_only_change_is_fresh_and_deployed_release_change_is_stale(repo):
    root, _exe, _ = repo
    ss.process(root, check=False, write=True, now=NOW)  # written while no lane exists: header says "none."
    assert ss.process(root, check=True, write=False, now=NOW)[0] == 0
    _branch(root, "claude/lane-new", NOW - 60)
    assert ss.process(root, check=True, write=False, now=NOW + 120)[0] == 0
    rf = root / ".state" / "releases" / "current-product.json"
    rec = json.loads(rf.read_text(encoding="utf-8"))
    rec["rollback"] = {"target_release_id": "design-other"}
    rf.write_text(json.dumps(rec), encoding="utf-8")
    assert ss.process(root, check=True, write=False, now=NOW + 120)[0] == 1
    ss.process(root, check=False, write=True, now=NOW + 120)
    assert ss.process(root, check=True, write=False, now=NOW + 120)[0] == 0
    rec["release_id"] = "release-y"
    rf.write_text(json.dumps(rec), encoding="utf-8")
    assert ss.process(root, check=True, write=False, now=NOW + 120)[0] == 1


def test_installed_block_staleness_is_reported(repo):
    root, _exe, _ = repo
    p = root / "docs" / "current" / "STATE.md"
    assert "Installed Product block: no installed block" in ss.render(root, NOW, p.read_text(encoding="utf-8"))
    wrong = f"{ss._installed.BEGIN}\nwrong\n{ss._installed.END}"
    assert "**STALE**" in ss.render(root, NOW, wrong)
    right = f"{ss._installed.BEGIN}\ninst\n{ss._installed.END}"
    assert "Installed Product block: current" in ss.render(root, NOW, right)


def test_crlf_preserved_and_missing_markers(repo):
    root, _exe, _ = repo
    p = root / "docs" / "current" / "STATE.md"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert ss.process(root, check=False, write=True, now=NOW)[0] == 0
    raw = p.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n")
    p.write_text("# no markers\n", encoding="utf-8")
    assert ss.process(root, check=True, write=False)[0] == 1
    assert ss.process(root, check=False, write=True)[0] == 1


def test_no_release_record_is_reported_not_fatal(repo):
    root, _exe, _ = repo
    (root / ".state" / "releases" / "current-product.json").unlink()
    assert "no `.state/releases/current-product.json`" in ss.render(root, NOW, "")


def test_main_cli_check_and_write(repo, capsys):
    root, _exe, _ = repo
    assert ss.main(["--root", str(root), "--check"]) == 1
    assert "stale" in capsys.readouterr().out
    assert ss.main(["--root", str(root), "--write"]) == 0
    assert ss.main(["--root", str(root), "--check"]) == 0
    capsys.readouterr()
    assert ss.main(["--root", str(root)]) == 0
    assert ss.BEGIN in capsys.readouterr().out
