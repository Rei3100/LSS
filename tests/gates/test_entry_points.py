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
    replies = {"branch": "codex/test", "run_head": HEAD, "pr_branch": "codex/test",
               "cross_repository": False, "origin": "https://github.com/example-owner/LSS.git\n",
               "push_urls": "https://github.com/example-owner/LSS.git\n",
               "issue_url": "https://github.com/example-owner/LSS/issues/7",
               "write_stdout": "https://github.com/example-owner/LSS/issues/7\n"}

    def runner(argv, *, cwd, input=None):
        calls.append((argv, input))
        if argv[0] == "git":
            parts = argv[3:]
            if parts == ["remote", "get-url", "origin"]:
                return SimpleNamespace(stdout=replies["origin"])
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
            info = {"headRefOid": HEAD, "headRefName": replies["pr_branch"],
                    "isCrossRepository": replies["cross_repository"]}
            return SimpleNamespace(stdout=json.dumps({key: info[key] for key in argv[-1].split(",")}))
        if argv[:3] == ["gh", "issue", "view"]:
            return SimpleNamespace(stdout=json.dumps({"url": replies["issue_url"]}))
        if argv[:3] == ["gh", "api", "graphql"] and any(x.startswith("query=query") for x in argv):
            return SimpleNamespace(stdout=json.dumps({"data": {"node": {"pullRequest": {
                "headRefOid": HEAD, "headRefName": "codex/test",
                "repository": {"nameWithOwner": "example-owner/LSS"}}}}}))
        if (argv[:3] in (["gh", "issue", "create"], ["gh", "issue", "comment"],
                         ["gh", "issue", "edit"], ["gh", "issue", "close"],
                         ["gh", "pr", "create"], ["gh", "pr", "comment"],
                         ["gh", "pr", "ready"], ["gh", "pr", "close"])
                or argv[:4] == ["gh", "api", "--method", "DELETE"]):
            if argv[:3] == replies.get("fail_command"):
                raise subprocess.CalledProcessError(1, argv)
            return SimpleNamespace(stdout=replies["write_stdout"])
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
    monkeypatch.setattr(gh.guard, "published_main", lambda *a: None)
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


@pytest.mark.parametrize("arguments,expected", [
    (["issue-create", "--title", "Work item", "--body-file", "BODY", "--label", "work:queue",
      "--label", "agent_1-ready"],
     ["gh", "issue", "create", "--repo", "example-owner/LSS", "--title", "Work item",
      "--body-file", "-", "--label", "work:queue", "--label", "agent_1-ready"]),
    (["issue-comment", "--issue", "7", "--body-file", "BODY"],
     ["gh", "issue", "comment", "7", "--repo", "example-owner/LSS", "--body-file", "-"]),
    (["issue-edit", "--issue", "7", "--add-label", "queued", "--add-label", "agent:codex",
      "--remove-label", "held", "--remove-label", "old"],
     ["gh", "issue", "edit", "7", "--repo", "example-owner/LSS", "--add-label", "queued",
      "--add-label", "agent:codex", "--remove-label", "held", "--remove-label", "old"]),
    (["pr-comment", "--pr", "8", "--body-file", "BODY"],
     ["gh", "pr", "comment", "8", "--repo", "example-owner/LSS", "--body-file", "-"]),
    (["pr-create", "--title", "Interrupted work", "--body-file", "BODY", "--draft"],
     ["gh", "pr", "create", "--repo", "example-owner/LSS", "--base", "main", "--head",
      "codex/test", "--title", "Interrupted work", "--body-file", "-", "--draft"]),
    (["pr-ready", "--pr", "8"],
     ["gh", "pr", "ready", "8", "--repo", "example-owner/LSS"]),
    (["pr-draft", "--pr", "8"],
     ["gh", "pr", "ready", "8", "--repo", "example-owner/LSS", "--undo"]),
    (["comment-delete", "--comment-id", "123"],
     ["gh", "api", "--method", "DELETE", "repos/example-owner/LSS/issues/comments/123"]),
])
def test_new_operation_command_shapes(publication, tmp_path, capsys, arguments, expected):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text("Reviewed work\n", encoding="utf-8")
    args = [str(path) if arg == "BODY" else arg for arg in arguments]
    assert gh.main(["--dry-run", *args], runner=runner) == 0
    assert json.loads(capsys.readouterr().out) == {
        "argv": expected, "stdin": "Reviewed work\n" if "BODY" in arguments else None}
    assert not any(argv == expected for argv, _ in calls)


@pytest.mark.parametrize("kind", ["issue", "pr"])
@pytest.mark.parametrize("post_fails", [False, True])
def test_close_posts_reason_before_closing(publication, tmp_path, kind, post_fails):
    runner, calls, replies = publication
    path = tmp_path / "reason.txt"
    path.write_text("Work is superseded\n", encoding="utf-8")
    if post_fails:
        replies["fail_command"] = ["gh", kind, "comment"]
    assert gh.main([kind + "-close", "--" + kind, "7", "--reason-file", str(path)],
                   runner=runner) == (1 if post_fails else 0)
    writes = [(argv, text) for argv, text in calls if argv[:2] == ["gh", kind] and argv[2] != "view"]
    expected = [(["gh", kind, "comment", "7", "--repo", "example-owner/LSS", "--body-file", "-"],
                 "Work is superseded\n")]
    if not post_fails:
        expected.append((["gh", kind, "close", "7", "--repo", "example-owner/LSS"], None))
    assert writes == expected


@pytest.mark.parametrize("arguments", [
    ["issue-create", "--title", "Work", "--body-file"],
    ["issue-comment", "--issue", "7", "--body-file"],
    ["issue-close", "--issue", "7", "--reason-file"],
    ["issue-edit", "--issue", "7", "--add-label", "queued"],
    ["pr-comment", "--pr", "8", "--body-file"],
    ["comment-delete", "--comment-id", "123"],
])
def test_unbound_operations_work_without_agent_checkout(publication, tmp_path, monkeypatch, arguments):
    runner, calls, replies = publication
    replies["branch"] = "main"
    replies["pr_branch"] = "someone/other"
    monkeypatch.delenv("LSS_AGENT_BRANCH")
    path = tmp_path / "body.txt"
    path.write_text("Reviewed work", encoding="utf-8")
    args = arguments + ([str(path)] if arguments[-1].endswith("-file") else [])
    assert gh.main(args, runner=runner) == 0
    assert not any("symbolic-ref" in argv or "rev-parse" in argv for argv, _ in calls)


@pytest.mark.parametrize("operation", ["pr-ready", "pr-draft", "pr-close"])
@pytest.mark.parametrize("mismatch", ["remote_branch", "fork", "checkout", "missing_assignment"])
def test_pr_state_requires_assigned_branch(publication, tmp_path, monkeypatch, operation, mismatch):
    runner, calls, replies = publication
    if mismatch == "remote_branch":
        replies["pr_branch"] = "codex/other"
    elif mismatch == "fork":
        replies["cross_repository"] = True
    elif mismatch == "checkout":
        replies["branch"] = "main"
    else:
        monkeypatch.delenv("LSS_AGENT_BRANCH")
    path = tmp_path / "reason.txt"
    path.write_text("Reviewed reason", encoding="utf-8")
    args = [operation, "--pr", "8"]
    if operation == "pr-close":
        args += ["--reason-file", str(path)]
    assert gh.main(args, runner=runner) == 1
    assert not any(argv[:3] in (["gh", "pr", "ready"], ["gh", "pr", "comment"],
                               ["gh", "pr", "close"]) for argv, _ in calls)


@pytest.mark.parametrize("operation", ["issue-create", "issue-comment", "issue-close", "pr-comment", "pr-close"])
@pytest.mark.parametrize("unsafe", [
    "synthetic-" + "private-phrase", "gh" + "p_" + "x" * 40,
    "C:" + chr(92) + "Users" + chr(92) + "SyntheticUser", "LSS-" + "CANARY-12345678",
])
def test_all_new_body_paths_inspect_before_writing(publication, tmp_path, capsys, operation, unsafe):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text(unsafe, encoding="utf-8")
    args = [operation]
    if operation == "issue-create":
        args += ["--title", "Work"]
    else:
        args += ["--issue" if operation.startswith("issue-") else "--pr", "7"]
    args += ["--reason-file" if operation.endswith("-close") else "--body-file", str(path)]
    assert gh.main(args, runner=runner) == 1
    assert not any(argv[0] == "gh" and argv[2] != "view" for argv, _ in calls)
    assert unsafe not in capsys.readouterr().out


@pytest.mark.parametrize("suffix,accepted", [
    ("", True), ("\n", False), (" ", False), (" extra", False),
    ("\n" + "LSS-" + "CANARY-abcdefgh", False),
    ("\nsynthetic-" + "private-phrase", False),
    ("\n" + "gh" + "p_" + "x" * 40, False),
])
def test_canary_exception_requires_exactly_one_token(publication, tmp_path, suffix, accepted):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    text = "lss-" + "canary-1234abcd" + suffix
    path.write_text(text, encoding="utf-8")
    assert gh.main(["issue-comment", "--issue", "7", "--body-file", str(path),
                    "--allow-canary-test"], runner=runner) == (0 if accepted else 1)
    writes = [(argv, value) for argv, value in calls if argv[:3] == ["gh", "issue", "comment"]]
    assert len(writes) == int(accepted)
    if accepted:
        assert writes[0][1] == text


@pytest.mark.parametrize("text", ["Safe text", "", "LSS-" + "CANARY-short", " LSS-" + "CANARY-12345678"])
def test_canary_flag_rejects_non_canary_body(publication, tmp_path, text):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text(text, encoding="utf-8")
    assert gh.main(["issue-comment", "--issue", "7", "--body-file", str(path),
                    "--allow-canary-test"], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


def test_canary_exception_never_bypasses_private_literals(publication, tmp_path, monkeypatch, capsys):
    runner, calls, _ = publication
    text = "LSS-" + "CANARY-12345678"
    deny = tmp_path / "private-canary-list.txt"
    deny.write_text(text, encoding="utf-8")
    monkeypatch.setenv("LSS_PUBLISH_DENYLIST", str(deny))
    path = tmp_path / "body.txt"
    path.write_text(text, encoding="utf-8")
    assert gh.main(["issue-comment", "--issue", "7", "--body-file", str(path),
                    "--allow-canary-test"], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)
    assert text not in capsys.readouterr().out


@pytest.mark.parametrize("arguments", [
    ["issue-create", "--title", "Work", "--body-file", "unused", "--allow-canary-test"],
    ["pr-comment", "--pr", "7", "--body-file", "unused", "--allow-canary-test"],
    ["issue-close", "--issue", "7", "--reason-file", "unused", "--allow-canary-test"],
    ["issue-edit", "--issue", "7", "--title", "Changed"],
    ["issue-edit", "--issue", "7", "--body-file", "unused"],
    ["issue-comment", "--issue", "7", "--body-f", "unused"],
])
def test_new_operations_refuse_unlisted_options(publication, arguments):
    runner, calls, _ = publication
    with pytest.raises(SystemExit):
        gh.main(arguments, runner=runner)
    assert calls == []


@pytest.mark.parametrize("operation,selector", [
    ("issue-comment", "--issue"), ("issue-close", "--issue"), ("issue-edit", "--issue"),
    ("pr-comment", "--pr"), ("pr-ready", "--pr"), ("pr-draft", "--pr"), ("pr-close", "--pr"),
    ("comment-delete", "--comment-id"),
])
@pytest.mark.parametrize("value", ["abc", "0", "-1", "+1", "1.0", "1/2", "https://github.com/other/repo/issues/1", "\u0661"])
def test_new_ids_are_positive_ascii_digits(publication, operation, selector, value):
    runner, calls, _ = publication
    args = [operation, selector, value]
    if operation.endswith("-comment"):
        args += ["--body-file", "unused"]
    elif operation.endswith("-close"):
        args += ["--reason-file", "unused"]
    elif operation == "issue-edit":
        args += ["--add-label", "queued"]
    with pytest.raises(SystemExit):
        gh.main(args, runner=runner)
    assert calls == []


@pytest.mark.parametrize("option,operation", [
    ("--label", "issue-create"), ("--add-label", "issue-edit"), ("--remove-label", "issue-edit")])
@pytest.mark.parametrize("label", ["", "Upper", "has space", "a,b", "a" * 41, "a/b", "\u00e9"])
def test_invalid_labels_never_write(publication, tmp_path, option, operation, label):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text("Reviewed work", encoding="utf-8")
    args = ([operation, "--title", "Work", "--body-file", str(path)] if operation == "issue-create"
            else [operation, "--issue", "7"])
    assert gh.main([*args, option + "=" + label], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


@pytest.mark.parametrize("label", ["a", "a" * 40, "--repo"])
@pytest.mark.parametrize("option,operation", [
    ("--label", "issue-create"), ("--add-label", "issue-edit"), ("--remove-label", "issue-edit")])
def test_valid_label_boundaries_and_option_like_values(publication, tmp_path, label, option, operation):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text("Reviewed work", encoding="utf-8")
    args = ([operation, "--title", "Work", "--body-file", str(path)] if operation == "issue-create"
            else [operation, "--issue", "7"])
    assert gh.main([*args, option + "=" + label], runner=runner) == 0
    command = calls[-1][0]
    if label.startswith("-"):
        assert command[-1] == option + "=" + label
    else:
        assert command[-2:] == [option, label]
    assert command.count("--repo") == 1


@pytest.mark.parametrize("arguments", [
    ["issue-create", "--title", "synthetic-" + "private-phrase", "--body-file", "BODY"],
    ["issue-create", "--title", "Work", "--body-file", "BODY", "--label", "synthetic-private-phrase"],
    ["issue-edit", "--issue", "7", "--add-label", "synthetic-private-phrase"],
    ["issue-edit", "--issue", "7", "--remove-label", "synthetic-private-phrase"],
    ["issue-create", "--title", "LSS-" + "CANARY-12345678", "--body-file", "BODY"],
])
def test_titles_and_valid_labels_receive_public_inspection(publication, tmp_path, arguments):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text("Reviewed work", encoding="utf-8")
    assert gh.main([str(path) if arg == "BODY" else arg for arg in arguments], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


@pytest.mark.parametrize("operation", ["issue-create", "issue-comment", "issue-close", "issue-edit",
                                       "pr-comment", "pr-ready", "pr-draft", "pr-close", "comment-delete"])
@pytest.mark.parametrize("field", ["origin", "push_urls"])
def test_new_operations_refuse_wrong_repository(publication, operation, field):
    runner, calls, replies = publication
    replies[field] = "https://github.com/other-owner/other.git"
    args = [operation]
    if operation == "issue-create":
        args += ["--title", "Work", "--body-file", "unused"]
    elif operation == "comment-delete":
        args += ["--comment-id", "7"]
    else:
        args += ["--issue" if operation.startswith("issue-") else "--pr", "7"]
        if operation.endswith("-comment"):
            args += ["--body-file", "unused"]
        elif operation.endswith("-close"):
            args += ["--reason-file", "unused"]
        elif operation == "issue-edit":
            args += ["--add-label", "queued"]
    assert gh.main(args, runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


@pytest.mark.parametrize("data", [b"a" * (1024 * 1024 + 1), b"\xff", b" \n"])
def test_new_body_paths_fail_closed_on_invalid_files(publication, tmp_path, data):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_bytes(data)
    assert gh.main(["issue-comment", "--issue", "7", "--body-file", str(path)], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


def test_issue_create_prints_gh_result_and_reads_body_once(publication, tmp_path, monkeypatch, capsys):
    runner, calls, replies = publication
    path = tmp_path / "body.txt"
    path.write_bytes(b"Reviewed purpose\r\n")
    original_open = Path.open
    reads = []
    def tracked_open(self, *args, **kwargs):
        if self == path:
            reads.append((args, kwargs))
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", tracked_open)
    assert gh.main(["issue-create", "--title", "Work", "--body-file", str(path)], runner=runner) == 0
    assert len(reads) == 1
    assert calls[-1][1] == "Reviewed purpose\n"
    assert capsys.readouterr().out == replies["write_stdout"]


def test_issue_edit_requires_a_label_change(publication):
    runner, calls, _ = publication
    assert gh.main(["issue-edit", "--issue", "7"], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


@pytest.mark.parametrize("operation", ["issue-create", "pr-create"])
def test_title_inspection_uses_literal_text(publication, tmp_path, monkeypatch, operation):
    runner, calls, _ = publication
    text = 'synthetic-"private-phrase'
    deny = tmp_path / "quoted-denylist.txt"
    deny.write_text(text, encoding="utf-8")
    monkeypatch.setenv("LSS_PUBLISH_DENYLIST", str(deny))
    path = tmp_path / "body.txt"
    path.write_text("Reviewed work", encoding="utf-8")
    assert gh.main([operation, "--title", text, "--body-file", str(path)], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


@pytest.mark.parametrize("operation", ["issue-create", "pr-create"])
def test_title_size_limit_counts_utf8_bytes(publication, tmp_path, operation):
    runner, calls, _ = publication
    path = tmp_path / "body.txt"
    path.write_text("Reviewed work", encoding="utf-8")
    assert gh.main([operation, "--title", "\u00e9" * (512 * 1024 + 1),
                    "--body-file", str(path)], runner=runner) == 1
    assert not any(argv[0] == "gh" for argv, _ in calls)


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


@pytest.mark.parametrize("arguments", [
    ["issue-close", "--issue", "7", "--reason-file", "BODY"],
    ["issue-edit", "--issue", "7", "--add-label", "queued"],
])
@pytest.mark.parametrize("url", [
    "https://github.com/example-owner/LSS/pull/7",
    "https://github.com/example-owner/LSS/issues/8",
    "https://github.com/other-owner/LSS/issues/7",
])
def test_issue_state_operations_refuse_pull_requests(publication, tmp_path, arguments, url):
    """gh issue close/edit accept a pull request number; the entry must not."""
    runner, calls, replies = publication
    replies["issue_url"] = url
    path = tmp_path / "reason.txt"
    path.write_text("Work is superseded\n", encoding="utf-8")
    args = [str(path) if arg == "BODY" else arg for arg in arguments]
    assert gh.main(args, runner=runner) == 1
    assert not any(argv[:2] == ["gh", "issue"] and argv[2] != "view" for argv, _ in calls)


def test_run_carries_text_outside_the_console_code_page(tmp_path):
    """A body with an emoji reaches the child unchanged whatever the Windows code page is."""
    text = "Reviewed \U0001f916 日本語\n"
    echo = "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"
    assert gh.run([sys.executable, "-I", "-c", echo], cwd=tmp_path, input=text).stdout == text


def test_dry_run_output_is_ascii_json(publication, tmp_path, capsysbinary):
    runner, _, _ = publication
    path = tmp_path / "body.txt"
    path.write_text("Reviewed \U0001f916\n", encoding="utf-8")
    assert gh.main(["--dry-run", "issue-comment", "--issue", "7", "--body-file", str(path)], runner=runner) == 0
    out = capsysbinary.readouterr().out
    assert out.isascii() and json.loads(out)["stdin"] == "Reviewed \U0001f916\n"


def test_only_git_push_gets_the_longer_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(gh, "run_bounded", lambda argv, **k: seen.append(k["timeout"]))
    gh.run(["git", "-C", "repo", "push", "origin", "x:refs/heads/y"], cwd=".")
    gh.run(["git", "-C", "repo", "rev-parse", "HEAD"], cwd=".")
    gh.run(["gh", "pr", "create"], cwd=".")
    assert seen == [900, 120, 120]


def test_a_timeout_ends_children_that_hold_the_output_open(tmp_path):
    import time
    child = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        gh.run_bounded([sys.executable, "-c", child], cwd=tmp_path, timeout=2)
    assert time.monotonic() - started < 30


def test_bounded_run_reports_output_and_failure(tmp_path):
    done = gh.run_bounded([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], cwd=tmp_path,
                          input="body", timeout=30)
    assert done.stdout.strip() == "BODY"
    with pytest.raises(subprocess.CalledProcessError):
        gh.run_bounded([sys.executable, "-c", "raise SystemExit(3)"], cwd=tmp_path, timeout=30)


def test_a_failed_tree_kill_still_ends_the_command_and_reports_the_timeout(tmp_path, monkeypatch):
    import time
    real_run = subprocess.run
    def refuse_taskkill(argv, **k):
        if argv[:1] == ["taskkill"]:
            return SimpleNamespace(returncode=1)
        return real_run(argv, **k)
    monkeypatch.setattr(gh.subprocess, "run", refuse_taskkill)
    monkeypatch.setattr(gh.os, "killpg", lambda *a: (_ for _ in ()).throw(OSError("denied")), raising=False)
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        gh.run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path, timeout=2)
    assert time.monotonic() - started < 30


def test_a_new_branch_push_skips_commits_already_on_the_fetched_main(tmp_path, monkeypatch):
    import io
    main = "c" * 40
    monkeypatch.setattr(gh.guard.sys, "stdin", io.StringIO(f"HEAD {HEAD} refs/heads/claude/x {'0' * 40}\n"))
    monkeypatch.setattr(gh.guard, "resolve_object", lambda *a: (HEAD, "commit"))
    monkeypatch.setattr(gh.guard, "published_main", lambda repo, remote, url: main if remote == "origin" else None)
    asked = []
    monkeypatch.setattr(gh.guard, "git", lambda repo, *args: asked.append(args) or HEAD.encode())
    monkeypatch.setattr(gh.guard, "scan", lambda repo, oid, **k: {"result": "PASS", "files_checked": 1, "findings": [],
                                                                  "object_oid": oid, "allowlist_sha256": "synthetic"})
    assert gh.guard.pre_push(tmp_path, None, None, "origin", "https://example.invalid/repo.git") == 0
    assert asked == [("rev-list", HEAD, "^" + main)]


@pytest.mark.parametrize("reply,local,trusted", [
    ("c" * 40 + "\trefs/heads/main\n", True, True),
    ("c" * 40 + "\trefs/heads/main\n", False, False),     # the destination's main is not in this repository
    ("", True, False),                                    # no main at the destination
    (None, True, False),                                  # the destination cannot be asked
    ("c" * 40 + "\trefs/heads/main\n" + "d" * 40 + "\trefs/heads/main\n", True, False)])
def test_only_the_destinations_own_main_is_trusted(tmp_path, monkeypatch, reply, local, trusted):
    main = "c" * 40
    asked = []
    def git(repo, *args):
        asked.append(args)
        if reply is None:
            raise gh.guard.GuardError("network")
        return reply.encode()
    def resolve(repo, oid):
        if not local:
            raise gh.guard.GuardError("missing")
        return oid, "commit"
    monkeypatch.setattr(gh.guard, "git", git)
    monkeypatch.setattr(gh.guard, "resolve_object", resolve)
    monkeypatch.setattr(gh.guard, "url_rewrites", lambda repo: False)
    assert (gh.guard.published_main(tmp_path, "origin", "https://example.invalid/a.git") == main) == trusted
    assert asked == [("ls-remote", "--", "https://example.invalid/a.git", "refs/heads/main")]
    assert gh.guard.published_main(tmp_path, None, None) is None


def test_a_url_rewrite_rule_disables_the_trust(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert gh.guard.url_rewrites(tmp_path) is False
    subprocess.run(["git", "-C", str(tmp_path), "config", "url.https://example.invalid/b/.insteadOf",
                    "https://example.invalid/a/"], check=True)
    assert gh.guard.url_rewrites(tmp_path) is True
    assert gh.guard.published_main(tmp_path, "origin", "https://example.invalid/a/repo.git") is None
