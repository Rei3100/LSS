"""Offline entry-point tests: fake publication commands and injected deployment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "gates" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


gh = module("lss_gh")
deployment = module("deploy_gated")
HEAD = "a" * 40


@pytest.fixture
def publication(tmp_path, monkeypatch):
    deny = tmp_path / "synthetic-denylist.txt"
    deny.write_text("synthetic-" + "private-phrase\n", encoding="utf-8")
    monkeypatch.setenv("LSS_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("LSS_AGENT_BRANCH", "codex/test")
    monkeypatch.setenv("LSS_GITHUB_REPOSITORY", "example-owner/LSS")
    monkeypatch.setenv("LSS_PUBLISH_DENYLIST", str(deny))
    monkeypatch.setattr(gh.guard, "scan", lambda *a, **k: {"result": "PASS"})
    calls = []
    replies = {"branch": "codex/test", "run_head": HEAD,
               "push_urls": "https://github.com/example-owner/LSS.git\n"}

    def runner(argv, *, cwd, input=None):
        calls.append((argv, input))
        if argv[0] == "git":
            parts = argv[3:]
            if parts == ["remote", "get-url", "origin"]:
                return SimpleNamespace(stdout="https://github.com/example-owner/LSS.git\n")
            if parts == ["remote", "get-url", "--push", "--all", "origin"]:
                return SimpleNamespace(stdout=replies["push_urls"])
            if parts == ["symbolic-ref", "--short", "HEAD"]:
                return SimpleNamespace(stdout=replies["branch"])
            if parts in (["rev-parse", "HEAD"], ["rev-list", HEAD]):
                return SimpleNamespace(stdout=HEAD)
        if argv[:3] == ["gh", "run", "view"]:
            return SimpleNamespace(stdout=json.dumps({"headSha": replies["run_head"],
                                                      "headBranch": "codex/test", "event": "pull_request"}))
        if argv[:3] == ["gh", "pr", "view"]:
            return SimpleNamespace(stdout=json.dumps({"headRefOid": HEAD, "headRefName": "codex/test",
                                                      "isCrossRepository": False}))
        if argv[:3] == ["gh", "api", "graphql"] and any(x.startswith("query=query") for x in argv):
            return SimpleNamespace(stdout=json.dumps({"data": {"node": {"pullRequest": {
                "headRefOid": HEAD, "headRefName": "codex/test",
                "repository": {"nameWithOwner": "example-owner/LSS"}}}}}))
        raise AssertionError("Unexpected executable call: " + repr(argv))
    return runner, calls, replies


@pytest.mark.parametrize("argv", [["push", "--force"], ["push", "-f"], ["push", "--no-verify"],
                                  ["push", "origin", "main"], ["api"], ["issue-delete"]])
def test_unknown_and_unsafe_arguments_never_run(publication, argv):
    runner, calls, _ = publication
    with pytest.raises(SystemExit):
        gh.main(argv, runner=runner)
    assert calls == []


def test_main_branch_refused(publication, monkeypatch):
    runner, calls, replies = publication
    replies["branch"] = "main"
    monkeypatch.setenv("LSS_AGENT_BRANCH", "main")
    assert gh.main(["--dry-run", "push"], runner=runner) == 1
    assert not any("push" in argv for argv, _ in calls)


@pytest.mark.parametrize("body", ["synthetic-" + "private-phrase", "C:" + chr(92) + "Users" + chr(92) + "SyntheticUser"])
def test_body_guard_refuses_private_text(publication, tmp_path, body):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text(body, encoding="utf-8")
    assert gh.main(["--dry-run", "pr-create", "--title", "Safe title", "--body-file", str(path)], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


def test_exact_pr_dry_run_and_snapshot(publication, tmp_path, capsys):
    runner, calls, _ = publication
    body = tmp_path / "body.txt"
    body.write_text("Reviewed purpose\n", encoding="utf-8")
    assert gh.main(["--dry-run", "pr-create", "--title", "Safe title", "--body-file", str(body)], runner=runner) == 0
    assert json.loads(capsys.readouterr().out) == {"argv": ["gh", "pr", "create", "--repo", "example-owner/LSS",
        "--base", "main", "--head", "codex/test", "--title", "Safe title", "--body-file", "-"], "stdin": "Reviewed purpose\n"}
    assert all(argv[0] == "git" for argv, _ in calls)


def test_exact_push_dry_run(publication, tmp_path, capsys):
    runner, _, _ = publication
    assert gh.main(["--dry-run", "push"], runner=runner) == 0
    assert json.loads(capsys.readouterr().out) == {"argv": ["git", "-C", str(tmp_path.resolve()), "push", "origin",
        HEAD + ":refs/heads/codex/test"], "stdin": None}


@pytest.mark.parametrize("urls", ["", "https://example.invalid/other.git",
    "https://github.com/example-owner/LSS.git\nhttps://example.invalid/other.git"])
def test_every_push_url_must_be_fixed_repository(publication, urls):
    runner, calls, replies = publication
    replies["push_urls"] = urls
    assert gh.main(["--dry-run", "push"], runner=runner) == 1
    assert not any("push" in argv for argv, _ in calls)


@pytest.mark.parametrize("field,value", [("review_head_sha", "b" * 40), ("base_sha", "bad"),
    ("diff_digest", "bad"), ("engine", "human"), ("complete", False), ("summary", " "),
    ("findings_p0p1", -1), ("findings_p0p1", True), ("findings_p0p1", "0"),
    ("findings_p0p1", 0.0), ("findings_p0p1", None),
    ("unresolved_p0p1", False), ("unresolved_p0p1", "0"),
    ("unresolved_p0p1", 0.0), ("unresolved_p0p1", None)])
def test_review_record_rejects_malformed_or_stale(field, value):
    record = {"base_sha": "b" * 40, "review_head_sha": HEAD, "diff_digest": "d" * 64,
              "engine": "claude", "complete": True, "summary": "Independent review",
              "findings_p0p1": 0, "unresolved_p0p1": 0}
    record[field] = value
    with pytest.raises(ValueError):
        gh.format_review_record(record, HEAD)


@pytest.mark.parametrize("engine", ["claude", "codex"])
def test_review_record_posts_formatter_output(publication, tmp_path, capsys, engine):
    runner, _, _ = publication
    record = {"base_sha": "b" * 40, "review_head_sha": HEAD, "diff_digest": "d" * 64,
              "engine": engine, "complete": True, "summary": "Independent review",
              "findings_p0p1": 2, "unresolved_p0p1": 0}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    assert gh.main(["--dry-run", "review-record", "--pr", "1", "--reason-file", str(path)], runner=runner) == 0
    assert json.loads(capsys.readouterr().out)["stdin"] == gh.format_review_record(record, HEAD)


@pytest.mark.parametrize("field", ["findings_p0p1", "unresolved_p0p1"])
def test_review_record_requires_finding_counts(field):
    record = {"base_sha": "b" * 40, "review_head_sha": HEAD, "diff_digest": "d" * 64,
              "engine": "codex", "complete": True, "summary": "Independent review",
              "findings_p0p1": 0, "unresolved_p0p1": 0}
    del record[field]
    with pytest.raises(ValueError):
        gh.format_review_record(record, HEAD)


def test_review_record_transports_unresolved_findings():
    record = {"base_sha": "b" * 40, "review_head_sha": HEAD, "diff_digest": "d" * 64,
              "engine": "codex", "complete": True, "summary": "One finding remains",
              "findings_p0p1": 2, "unresolved_p0p1": 1}
    block = gh.format_review_record(record, HEAD)
    assert json.loads(block.splitlines()[1]) == record


def test_default_denylist_uses_private_root_and_missing_stays_closed(tmp_path, monkeypatch):
    import runpy
    monkeypatch.delenv("LSS_PUBLISH_DENYLIST", raising=False)
    monkeypatch.setenv("LSS_PRIVATE_ROOT", str(tmp_path))
    loaded = runpy.run_path(str(ROOT / "gates/public_guard.py"))
    assert Path(loaded["DEFAULT_DENYLIST"]) == tmp_path / "publish-denylist.txt"
    assert loaded["scan"](tmp_path)["result"] != "PASS"
    monkeypatch.delenv("LSS_PRIVATE_ROOT")
    loaded = runpy.run_path(str(ROOT / "gates/public_guard.py"))
    assert loaded["scan"](tmp_path)["result"] != "PASS"


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/codex/test", "refs/heads/claude/test-2/sub"])
def test_pre_push_accepts_agent_refs_with_full_scan(tmp_path, monkeypatch, ref):
    import io
    monkeypatch.setattr(gh.guard.sys, "stdin", io.StringIO(f"HEAD {HEAD} {ref} {'0' * 40}\n"))
    monkeypatch.setattr(gh.guard, "resolve_object", lambda *a: (HEAD, "commit"))
    monkeypatch.setattr(gh.guard, "git", lambda *a: (HEAD + "\n" + "b" * 40).encode())
    scanned = []
    def scan(repo, oid, **kwargs):
        scanned.append(oid)
        return {"result": "BLOCK" if oid == "b" * 40 else "PASS", "files_checked": 1,
                "findings": [], "object_oid": oid, "allowlist_sha256": "synthetic"}
    monkeypatch.setattr(gh.guard, "scan", scan)
    assert gh.guard.pre_push(tmp_path, None, None) == 1
    assert set(scanned) == {HEAD, "b" * 40}
    scanned.clear()
    monkeypatch.setattr(gh.guard.sys, "stdin", io.StringIO(f"HEAD {HEAD} {ref} {'0' * 40}\n"))
    monkeypatch.setattr(gh.guard, "git", lambda *a: HEAD.encode())
    assert gh.guard.pre_push(tmp_path, None, None) == 0
    assert scanned == [HEAD]


@pytest.mark.parametrize("ref,oid", [("refs/tags/v1", HEAD), ("refs/heads/other", HEAD),
    ("refs/heads/codex/", HEAD), ("refs/heads/codex/-bad", HEAD),
    ("refs/heads/claude/a!", HEAD), ("refs/heads/main", "0" * 40),
    ("refs/heads/codex/test", "0" * 40)])
def test_pre_push_refuses_other_refs_and_deletions(tmp_path, monkeypatch, ref, oid):
    import io
    monkeypatch.setattr(gh.guard.sys, "stdin", io.StringIO(f"HEAD {oid} {ref} {'0' * 40}\n"))
    monkeypatch.setattr(gh.guard, "scan", lambda *a, **k: pytest.fail("Forbidden update reached scan"))
    assert gh.guard.pre_push(tmp_path, None, None) != 0


def test_commit_guard_blocks_before_push(publication, monkeypatch):
    runner, calls, _ = publication
    monkeypatch.setattr(gh.guard, "scan", lambda *a, **k: {"result": "BLOCK"})
    assert gh.main(["--dry-run", "push"], runner=runner) == 1
    assert not any("push" in argv for argv, _ in calls)


def test_rerun_requires_current_head(publication, capsys):
    runner, _, replies = publication
    assert gh.main(["--dry-run", "ci-rerun", "--run-id", "12"], runner=runner) == 0
    assert json.loads(capsys.readouterr().out)["argv"] == ["gh", "run", "rerun", "12", "--repo", "example-owner/LSS"]
    replies["run_head"] = "b" * 40
    assert gh.main(["--dry-run", "ci-rerun", "--run-id", "12"], runner=runner) == 1


def test_resolve_records_reason_before_resolution(publication, tmp_path, capsys):
    runner, _, _ = publication
    reason = tmp_path / "reason.txt"
    reason.write_text("Addressed by current implementation", encoding="utf-8")
    assert gh.main(["--dry-run", "thread-resolve", "--thread", "PRRT_test", "--reason-file", str(reason)], runner=runner) == 0
    plans = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(plans) == 2
    assert "addPullRequestReviewThreadReply" in json.loads(plans[0]["stdin"])["query"]
    assert "resolveReviewThread" in plans[1]["argv"][4]


@pytest.fixture
def release(tmp_path):
    def write(name, data):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)
    package = tmp_path / "package"
    package.mkdir()
    (package / "LSS.exe").write_bytes(b"synthetic executable")
    exe_hash = deployment.sha(package / "LSS.exe")
    tree_hash = hashlib.sha256(("LSS.exe\t" + exe_hash + "\n").encode()).hexdigest()
    identity = {"source_commit": HEAD, "exe_sha256": exe_hash, "tree_sha256": tree_hash}
    manifest = {**identity, "type": "lss_frozen_candidate_package", "package_dir": str(package),
                "files_sha256": {"LSS.exe": {"bytes": (package / "LSS.exe").stat().st_size, "sha256": exe_hash}}}
    old = tmp_path / "previous.exe"
    old.write_bytes(b"previous executable")
    old_hash = deployment.sha(old)
    key = tmp_path / "answer-key.txt"
    key.write_text("synthetic acceptance scope", encoding="utf-8")
    state = {"repo": str(tmp_path), "manifest": write("manifest", manifest),
        "build_receipt": write("build", {"commit": HEAD, "exe_sha256": exe_hash,
            "status": "complete", "phase": "complete", "candidate": str(package)}),
        "packaged_receipt": write("packaged", {**identity, "status": "PASS", "builder": "builder-a", "scope": "real-machine-package"}),
        "verification_receipt": write("verifier", {**identity, "status": "PASS", "verifier": "verifier-b"}),
        "answer_key": str(key),
        "answer_key_receipt": write("answers", {**identity, "status": "PASS", "baseline_exe_sha256": old_hash,
            "answer_key_sha256": deployment.sha(key), "non_regression": True, "user_benefit": "Useful fix",
            "change_kind": "timing-only", "pcm_identical": True}),
        "current_product": write("current", {"deployed_executable": str(old), "exe_sha256": old_hash}),
        "rollback_receipt": write("rollback", {"status": "PASS", "exe": str(old), "exe_sha256": old_hash, "startup_pass": True}),
        "idle_receipt": write("idle", {**identity, "status": "PASS", "observed_at": 1000,
            "product_running": False, "other_gpu_jobs": 0, "user_heavy_use": False})}
    return state, write


@pytest.mark.parametrize("condition,key", list(zip(deployment.CONDITIONS, ["repo", "build_receipt", "packaged_receipt",
    "verification_receipt", "answer_key_receipt", "rollback_receipt", "idle_receipt"])))
def test_each_missing_condition_is_named(release, condition, key):
    state, _ = release
    del state[key]
    result = deployment.evaluate(state, git_reader=lambda *a: HEAD, now=1000)
    assert result["result"] == "BLOCK"
    assert condition in result["failures"]


@pytest.mark.parametrize("receipt,field,value,condition", [
    ("verification_receipt", "verifier", "builder-a", "independent_verification"),
    ("verification_receipt", "exe_sha256", "b" * 64, "independent_verification"),
    ("answer_key_receipt", "pcm_identical", False, "answer_key_non_regression"),
    ("answer_key_receipt", "baseline_exe_sha256", "b" * 64, "answer_key_non_regression"),
    ("rollback_receipt", "startup_pass", False, "rollback_ready"),
    ("idle_receipt", "observed_at", 900, "machine_idle"),
    ("idle_receipt", "product_running", True, "machine_idle"),
    ("idle_receipt", "other_gpu_jobs", 1, "machine_idle"),
    ("idle_receipt", "user_heavy_use", True, "machine_idle"),
])
def test_invalid_receipts_fail_closed(release, receipt, field, value, condition):
    state, write = release
    data = deployment.load(state[receipt])
    data[field] = value
    state[receipt] = write("changed", data)
    assert condition in deployment.evaluate(state, git_reader=lambda *a: HEAD, now=1000)["failures"]


def test_changed_package_fails_freeze(release):
    state, _ = release
    manifest = deployment.load(state["manifest"])
    (Path(manifest["package_dir"]) / "LSS.exe").write_bytes(b"changed")
    assert "frozen_candidate" in deployment.evaluate(state, git_reader=lambda *a: HEAD, now=1000)["failures"]


def test_full_pass_deploys_once_and_check_only_never_deploys(release, monkeypatch, capsys):
    state, write = release
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    path = write("state", state)
    calls = []
    def deployer(*args):
        calls.append(args)
    for options in (["--check-only"], []):
        assert deployment.main(["--state", path, *options], deployer=deployer, git_reader=lambda *a: HEAD) == 0
        assert json.loads(capsys.readouterr().out)["result"] == "PASS"
    assert len(calls) == 1
    del state["idle_receipt"]
    path = write("state", state)
    assert deployment.main(["--state", path], deployer=deployer, git_reader=lambda *a: HEAD) == 1
    assert len(calls) == 1


def test_deploy_command_only_uses_hidden_launcher(release, monkeypatch):
    state, _ = release
    state.update(deploy_log="synthetic.log", launch_receipt="launch.json", deploy_base="synthetic-deploy")
    calls = []
    monkeypatch.setattr(deployment, "git", lambda repo, *args: "" if args[0] == "status" else HEAD)
    monkeypatch.setattr(deployment.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    deployment.deploy(state, deployment.load(state["manifest"]))
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == [sys.executable, "-B", str(Path(state["repo"]) / "scripts" / "hidden_run.py")]
    assert argv[-1] == "-KeepPrevious"
    assert "-BuildOutput" in argv


@pytest.mark.parametrize("head,status", [("b" * 40, ""), (HEAD, " M scripts/hidden_run.py"),
                                         (HEAD, "?? untracked.py")])
def test_deploy_refuses_wrong_or_dirty_checkout_before_launcher(release, monkeypatch, head, status):
    state, _ = release
    calls = []
    reads = []
    def reader(repo, *args):
        reads.append(args)
        return status if args[0] == "status" else head
    monkeypatch.setattr(deployment, "git", reader)
    monkeypatch.setattr(deployment.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError):
        deployment.deploy(state, deployment.load(state["manifest"]))
    assert calls == []
    if head == HEAD:
        assert ("status", "--porcelain", "--untracked-files=all") in reads


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction")
def test_frozen_package_rejects_real_junction(release, tmp_path):
    state, write = release
    manifest = deployment.load(state["manifest"])
    link = tmp_path / "package-link"
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), manifest["package_dir"]],
                            capture_output=True, text=True)
    if result.returncode:
        pytest.skip("Cannot create junction: " + result.stderr)
    try:
        manifest["package_dir"] = str(link)
        state["manifest"] = write("linked-manifest", manifest)
        assert "frozen_candidate" in deployment.evaluate(state, git_reader=lambda *a: HEAD, now=1000)["failures"]
    finally:
        link.rmdir()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows installer")
def test_installer_plan_and_non_elevated_refusal(tmp_path):
    script = ROOT / "gates" / "install_gates.ps1"
    target = tmp_path / "private-bin"
    command = ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script), "-Target", str(target)]
    result = subprocess.run([*command, "-WhatIf"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["target"] == str(target) and plan["mutates"] is False and plan["elevation_required"] is True
    assert not target.exists()
    # Never call the real installer when this test process is elevated.
    import ctypes
    if ctypes.windll.shell32.IsUserAnAdmin():
        pytest.skip("Non-elevated refusal requires a normal-user process")
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "Elevation required" in result.stderr
    assert not target.exists()
