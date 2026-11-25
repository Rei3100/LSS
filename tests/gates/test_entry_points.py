"""Offline entry-point tests: fake publication commands and injected deployment."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
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
    monkeypatch.setattr(gh.guard, "published_main", lambda *a: None)
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
            if parts in (["rev-parse", "HEAD"], ["rev-list", HEAD], ["--no-replace-objects", "rev-list", HEAD]):
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
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(deployment, "shutil", SimpleNamespace(which=lambda name: sys.executable))
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
    base = tmp_path / "destination" / "installed"
    old = base / "LSS" / "LSS.exe"
    old.parent.mkdir(parents=True)
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
            "product_running": False, "other_gpu_jobs": 0, "user_heavy_use": False}),
        "deploy_base": str(base), "deploy_log": str(tmp_path / "deploy.log"),
        "launch_receipt": str(tmp_path / "launch.json")}
    Path(state["deploy_log"]).touch()
    return state, write


def clean_release_git(repo, *args):
    return "" if args[0] == "status" else HEAD


@pytest.mark.parametrize("condition,key", [
    ("integrated_main", "repo"),
    ("frozen_candidate", "build_receipt"),
    ("packaged_acceptance", "packaged_receipt"),
    ("independent_verification", "verification_receipt"),
    ("answer_key_non_regression", "answer_key_receipt"),
    ("rollback_ready", "rollback_receipt"),
    ("machine_idle", "idle_receipt"),
    ("deploy_base", "deploy_base"),
    ("deploy_log", "deploy_log"),
    ("launch_receipt", "launch_receipt"),
])
def test_each_missing_condition_is_named(release, condition, key):
    state, _ = release
    del state[key]
    result = deployment.evaluate(state, git_reader=clean_release_git, now=1000)
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
    assert condition in deployment.evaluate(state, git_reader=clean_release_git, now=1000)["failures"]


@pytest.mark.parametrize("receipt,field,value,condition", [
    ("manifest", "package_dir", 42, "frozen_candidate"),
    ("packaged_receipt", "builder", 42, "independent_verification"),
    ("verification_receipt", "verifier", 42, "independent_verification"),
    ("answer_key_receipt", "user_benefit", 42, "answer_key_non_regression"),
    ("answer_key_receipt", "change_kind", 42, "answer_key_non_regression"),
    ("idle_receipt", "other_gpu_jobs", False, "machine_idle"),
    ("idle_receipt", "other_gpu_jobs", 0.0, "machine_idle"),
])
def test_wrong_receipt_field_type_fails_its_own_condition(release, receipt, field, value, condition):
    state, write = release
    data = deployment.load(state[receipt])
    data[field] = value
    state[receipt] = write("changed", data)

    result = deployment.evaluate(state, git_reader=clean_release_git, now=1000)
    assert result["result"] == "BLOCK"
    assert result["failures"] == [condition]
    assert result["failure_details"][condition] == f"{field} has the wrong type"


def test_frozen_candidate_rejects_non_integer_file_size(release):
    state, write = release
    manifest = deployment.load(state["manifest"])
    row = manifest["files_sha256"]["LSS.exe"]
    row["bytes"] = float(row["bytes"])
    state["manifest"] = write("changed", manifest)

    result = deployment.evaluate(state, git_reader=clean_release_git, now=1000)
    assert result["failures"] == ["frozen_candidate"]


@pytest.mark.parametrize("key,conditions", [
    ("repo", ("integrated_main", "checkout_head", "checkout_clean")),
    ("manifest", ("frozen_candidate",)),
    ("build_receipt", ("frozen_candidate",)),
    ("packaged_receipt", ("packaged_acceptance", "independent_verification")),
    ("verification_receipt", ("independent_verification",)),
    ("answer_key", ("answer_key_non_regression",)),
    ("answer_key_receipt", ("answer_key_non_regression",)),
    ("current_product", ("answer_key_non_regression", "rollback_ready")),
    ("rollback_receipt", ("rollback_ready",)),
    ("idle_receipt", ("machine_idle",)),
    ("deploy_base", ("deploy_base",)),
    ("deploy_log", ("deploy_log",)),
    ("launch_receipt", ("launch_receipt",)),
])
def test_relative_state_path_names_the_key(release, tmp_path, monkeypatch, key, conditions):
    state, _ = release
    monkeypatch.chdir(tmp_path)
    state[key] = str(Path(state[key]).relative_to(tmp_path))

    result = deployment.evaluate(state, git_reader=clean_release_git, now=1000)
    assert result["result"] == "BLOCK"
    for condition in conditions:
        assert condition in result["failures"]
        assert result["failure_details"][condition] == f"{key} must be an absolute path"


def test_changed_package_fails_freeze(release):
    state, _ = release
    manifest = deployment.load(state["manifest"])
    (Path(manifest["package_dir"]) / "LSS.exe").write_bytes(b"changed")
    assert "frozen_candidate" in deployment.evaluate(state, git_reader=clean_release_git, now=1000)["failures"]


def test_full_pass_deploys_once_and_check_only_never_deploys(release, monkeypatch, capsys):
    state, write = release
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    path = write("state", state)
    calls = []
    def deployer(*args):
        calls.append(args)
    for options in (["--check-only"], []):
        assert deployment.main(["--state", path, *options], deployer=deployer, git_reader=clean_release_git) == 0
        assert json.loads(capsys.readouterr().out)["result"] == "PASS"
    assert len(calls) == 1
    del state["idle_receipt"]
    path = write("state", state)
    assert deployment.main(["--state", path], deployer=deployer, git_reader=clean_release_git) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("head,status,condition", [
    ("b" * 40, "", "checkout_head"),
    (HEAD, " M scripts/hidden_run.py", "checkout_clean"),
    (HEAD, "?? untracked.py", "checkout_clean"),
])
def test_check_only_reports_wrong_or_dirty_checkout(release, monkeypatch, capsys, head, status, condition):
    state, write = release
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    def reader(repo, *args):
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return status
        return clean_release_git(repo, *args)

    path = write("state", state)
    assert deployment.main(["--state", path, "--check-only"], git_reader=reader) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["result"] == "BLOCK"
    assert result["failures"] == [condition]


def test_deploy_refusal_preserves_preflight_verdict(release, monkeypatch, capsys):
    state, write = release
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    def changed_checkout(repo, *args):
        return "?? untracked.py" if args[0] == "status" else HEAD

    monkeypatch.setattr(deployment, "git", changed_checkout)
    monkeypatch.setattr(deployment.subprocess, "run", lambda *a, **k: pytest.fail("Launcher called"))
    path = write("state", state)
    assert deployment.main(["--state", path], git_reader=clean_release_git) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["result"] == "BLOCK"
    assert all(result["conditions"].values())
    assert result["failures"] == []
    assert result["deployment"] == "REFUSED"
    assert result["deployment_error"] == "checkout_clean: deployment checkout has local changes"


@pytest.mark.parametrize("key,target", [
    ("deploy_base", "missing"),
    ("deploy_base", "previous.exe"),
    ("deploy_log", "missing/deploy.log"),
    ("deploy_log", "package"),
    ("launch_receipt", "missing/launch.json"),
    ("launch_receipt", "manifest.json"),
    ("launch_receipt", "deploy.log"),
])
def test_check_only_rejects_unusable_launch_paths(release, tmp_path, monkeypatch, capsys, key, target):
    state, write = release
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    state[key] = str(tmp_path / target)

    path = write("state", state)
    assert deployment.main(["--state", path, "--check-only"], git_reader=clean_release_git) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["result"] == "BLOCK"
    assert result["failures"] == (["rollback_ready", key] if key == "deploy_base" else [key])


def test_integrated_main_uses_verified_remote_tracking_commit(release):
    state, _ = release
    calls = []

    def reader(repo, *args):
        calls.append(args)
        if args == ("rev-parse", "origin/main"):
            return "b" * 40
        return clean_release_git(repo, *args)

    result = deployment.evaluate(state, git_reader=reader, now=1000)
    assert result["conditions"]["integrated_main"] is True
    assert ("rev-parse", "--verify", "refs/remotes/origin/main^{commit}") in calls


def test_deploy_command_only_uses_hidden_launcher(release, monkeypatch):
    state, _ = release
    calls = []
    monkeypatch.setattr(deployment, "git", clean_release_git)
    monkeypatch.setattr(deployment.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    _, plan = deployment.preflight(state, git_reader=clean_release_git, now=1000)
    deployment.deploy(state, plan)
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == [sys.executable, "-B", str(Path(state["repo"]) / "scripts" / "hidden_run.py")]
    assert argv[argv.index("--") + 1] == str(Path(sys.executable).resolve())
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
    _, plan = deployment.preflight(state, git_reader=clean_release_git, now=1000)
    with pytest.raises(ValueError):
        deployment.deploy(state, plan)
    assert calls == []
    if head == HEAD:
        assert ("status", "--porcelain", "--untracked-files=all") in reads


def test_deploy_uses_package_verified_relative_to_checkout(release, tmp_path, monkeypatch, capsys):
    state, write = release
    manifest = deployment.load(state["manifest"])
    package = manifest["package_dir"]
    manifest["package_dir"] = "package"
    state["manifest"] = write("manifest", manifest)
    build = deployment.load(state["build_receipt"])
    build["candidate"] = "package"
    state["build_receipt"] = write("build", build)
    caller = tmp_path / "caller"
    (caller / "package").mkdir(parents=True)
    (caller / "package" / "LSS.exe").write_bytes(b"different executable")
    monkeypatch.chdir(caller)
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    monkeypatch.setattr(deployment, "git", clean_release_git)
    calls = []
    monkeypatch.setattr(deployment.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    args = ["--state", write("state", state)]

    assert deployment.main([*args, "--check-only"], git_reader=clean_release_git) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "PASS"
    assert calls == []
    assert deployment.main(args, git_reader=clean_release_git) == 0
    assert json.loads(capsys.readouterr().out)["deployment"] == "LAUNCHED"
    argv, = calls
    assert argv[argv.index("-BuildOutput") + 1] == package
    assert Path(state["deploy_log"]).read_bytes() == b""


def test_deploy_refuses_manifest_replacement_after_preflight(release, monkeypatch, capsys):
    state, write = release
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    manifest = deployment.load(state["manifest"])
    manifest["package_dir"] = str(Path(state["repo"]) / "unverified-package")

    def replace_before_launch(state, plan):
        replacement = Path(write("replacement", manifest))
        replacement.replace(state["manifest"])
        deployment.deploy(state, plan)

    monkeypatch.setattr(deployment, "git", clean_release_git)
    monkeypatch.setattr(deployment.subprocess, "run", lambda *a, **k: pytest.fail("Launcher called"))
    assert deployment.main(["--state", write("state", state)], deployer=replace_before_launch,
                           git_reader=clean_release_git) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["deployment"] == "REFUSED"
    assert result["deployment_error"] == "frozen_candidate: manifest changed after preflight"


def test_check_only_requires_rollback_for_deployment_target(release, tmp_path, monkeypatch, capsys):
    state, write = release
    old = Path(deployment.load(state["current_product"])["deployed_executable"])
    other_base = tmp_path / "other-installation"
    other = other_base / "LSS" / "LSS.exe"
    other.parent.mkdir(parents=True)
    other.write_bytes(old.read_bytes())
    state["deploy_base"] = str(other_base)
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    assert deployment.main(["--state", write("state", state), "--check-only"],
                           git_reader=clean_release_git) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == ["rollback_ready"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction")
@pytest.mark.parametrize("link_is_base", [True, False], ids=["base", "parent"])
def test_check_only_refuses_deploy_base_junction(release, tmp_path, monkeypatch, capsys, link_is_base):
    state, write = release
    base = Path(state["deploy_base"])
    link = tmp_path / "deploy-link"
    target = base if link_is_base else base.parent
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                            capture_output=True, text=True)
    if result.returncode:
        pytest.skip("Cannot create junction: " + result.stderr)
    try:
        state["deploy_base"] = str(link if link_is_base else link / base.name)
        old = str(Path(state["deploy_base"]) / "LSS" / "LSS.exe")
        for key, field in [("current_product", "deployed_executable"), ("rollback_receipt", "exe")]:
            data = deployment.load(state[key])
            data[field] = old
            state[key] = write(key, data)
        monkeypatch.setattr(deployment.time, "time", lambda: 1000)

        assert deployment.main(["--state", write("state", state), "--check-only"],
                               git_reader=clean_release_git) == 1
        assert json.loads(capsys.readouterr().out)["failures"] == ["deploy_base"]
    finally:
        link.rmdir()


@pytest.mark.parametrize("condition", ["deploy_log", "launch_receipt"])
def test_check_only_refuses_blocked_launcher_output(release, monkeypatch, capsys, condition):
    state, write = release
    log = Path(state["deploy_log"])
    receipt = Path(state["launch_receipt"])
    if condition == "deploy_log":
        log.write_bytes(b"existing log")
        original_open = os.open

        def deny_log_append(path, flags, *args, **kwargs):
            if Path(path) == log and flags & os.O_APPEND:
                raise PermissionError("log append denied")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(deployment.os, "open", deny_log_append)
    else:
        receipt.with_name(receipt.name + ".tmp").mkdir()
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    assert deployment.main(["--state", write("state", state), "--check-only"],
                           git_reader=clean_release_git) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == [condition]
    assert not receipt.exists()
    if condition == "deploy_log":
        assert log.read_bytes() == b"existing log"


def test_check_only_refuses_missing_pwsh(release, tmp_path, monkeypatch, capsys):
    state, write = release
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(deployment.shutil, "which", lambda name: None)
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    assert deployment.main(["--state", write("state", state), "--check-only"],
                           git_reader=clean_release_git) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == ["pwsh"]


@pytest.mark.parametrize("phase, layout, refusal", [
    ("stage_ready", "staged", None),
    ("stage_ready", "cleaned-staging", None),
    ("stage_ready", "saved-previous", "phase 'stage_ready' has an existing .LSS.previous"),
    ("stage_ready", "other-target", "phase 'stage_ready' refers to different deployment paths"),
    ("previous_saved", "saved-previous", "phase 'previous_saved'"),
    ("target_promoted", "saved-previous", "phase 'target_promoted'"),
    ("mutable_preserved", "saved-previous", "phase 'mutable_preserved'"),
    ("awaiting_acceptance", "saved-previous", "phase 'awaiting_acceptance'"),
    ("unknown_phase", "staged", "phase 'unknown_phase'"),
    (None, "corrupt", "journal is corrupt or unreadable"),
    (None, "saved-previous", ".LSS.previous exists without a deployment journal"),
])
def test_check_only_distinguishes_pre_swap_recovery(release, monkeypatch, capsys, phase, layout, refusal):
    state, write = release
    base = Path(state["deploy_base"])
    journal = base / ".LSS-deploy.json"
    staging = base / ".LSS.staging"
    previous = base / ".LSS.previous"
    if layout == "staged":
        staging.mkdir()
        (staging / "LSS.exe").write_bytes(b"staged candidate")
    if layout == "saved-previous":
        previous.mkdir()
    if layout == "corrupt":
        journal.write_text("{", encoding="utf-8")
    elif phase is not None:
        target = base / ("other" if layout == "other-target" else "LSS")
        journal.write_text(json.dumps({"version": 1, "phase": phase,
            "target": str(target), "staging": str(staging),
            "previous": str(previous), "runtime_entries": ["LSS.exe"]}), encoding="utf-8")
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    assert deployment.main(["--state", write("state", state), "--check-only"],
                           git_reader=clean_release_git) == (0 if refusal is None else 1)
    result = json.loads(capsys.readouterr().out)
    if refusal is None:
        assert result["result"] == "PASS"
        assert journal.is_file()
        assert staging.exists() == (layout == "staged")
    else:
        assert result["failures"] == ["deploy_base"]
        assert refusal in result["failure_details"]["deploy_base"]


@pytest.mark.parametrize("location, overlaps", [
    ("Candidates/LSS", None),
    ("LSS/candidate", "LSS"),
    (".LSS.staging", ".LSS.staging"),
    (".LSS.previous/candidate", ".LSS.previous"),
    (".", "LSS"),
])
def test_check_only_matches_managed_deployment_path_overlap(release, monkeypatch, capsys, location, overlaps):
    state, write = release
    manifest = deployment.load(state["manifest"])
    base = Path(state["deploy_base"])
    source = Path(manifest["package_dir"])
    package = base / location
    if package == base:
        (source / "LSS.exe").rename(package / "LSS.exe")
    else:
        package.parent.mkdir(parents=True, exist_ok=True)
        source.rename(package)
    manifest["package_dir"] = str(package)
    state["manifest"] = write("manifest", manifest)
    build = deployment.load(state["build_receipt"])
    build["candidate"] = str(package)
    state["build_receipt"] = write("build", build)
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    assert deployment.main(["--state", write("state", state), "--check-only"],
                           git_reader=clean_release_git) == (0 if overlaps is None else 1)
    result = json.loads(capsys.readouterr().out)
    if overlaps is None:
        assert result["result"] == "PASS"
    else:
        assert result["failure_details"]["deploy_base"] == f"frozen package overlaps managed deployment path {overlaps!r}"


@pytest.mark.parametrize("temporary_entry", ["file", "directory", "configured-log"])
def test_check_only_requires_unused_receipt_temporary_path(release, monkeypatch, capsys, temporary_entry):
    state, write = release
    temporary = Path(state["launch_receipt"] + ".tmp")
    if temporary_entry == "file":
        temporary.write_text("existing receipt data", encoding="utf-8")
    elif temporary_entry == "directory":
        temporary.mkdir()
    else:
        state["deploy_log"] = str(temporary)
        temporary.touch()
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)

    assert deployment.main(["--state", write("state", state), "--check-only"],
                           git_reader=clean_release_git) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == ["launch_receipt"]


def test_check_only_ignores_pwsh_in_the_callers_directory(release, tmp_path, monkeypatch, capsys):
    state, write = release
    caller = tmp_path / "caller"
    caller.mkdir()
    on_path = tmp_path / "bin"
    on_path.mkdir()
    name = "pwsh.exe" if sys.platform == "win32" else "pwsh"
    for candidate in {caller / "pwsh", caller / name}:
        candidate.touch()
        candidate.chmod(0o755)
    monkeypatch.chdir(caller)
    monkeypatch.setenv("PATH", os.pathsep.join([".", str(on_path)]))
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setenv("PATHEXT", ".EXE")
    monkeypatch.setattr(deployment, "shutil", shutil)
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    args = ["--state", write("state", state), "--check-only"]

    assert deployment.main(args, git_reader=clean_release_git) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == ["pwsh"]

    valid = on_path / name
    valid.touch()
    valid.chmod(0o755)
    result, plan = deployment.preflight(state, git_reader=clean_release_git, now=1000)
    assert result["result"] == "PASS"
    assert plan.pwsh == valid.resolve()


@pytest.mark.parametrize("output", ["deploy_log", "launch_receipt"])
@pytest.mark.parametrize("root_key", ["package_dir", "deploy_base"])
def test_check_only_keeps_launcher_outputs_outside_protected_roots(release, monkeypatch, capsys, output, root_key):
    state, write = release
    paths = deployment.load(state["manifest"]) if root_key == "package_dir" else state
    root = Path(paths[root_key])
    name = Path(state[output]).name
    if output == "deploy_log":
        blocked_path = root / ("LSS.exe" if root_key == "package_dir" else "LSS/LSS.exe")
        prior_bytes = blocked_path.read_bytes()
    else:
        blocked_path = root / name
    state[output] = str(blocked_path)
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    args = ["--state", write("state", state), "--check-only"]

    assert deployment.main(args, git_reader=clean_release_git) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == [output]
    if output == "deploy_log":
        assert blocked_path.read_bytes() == prior_bytes
    else:
        assert not blocked_path.exists()

    sibling = root.with_name(root.name + "-outputs")
    sibling.mkdir()
    state[output] = str(sibling / name)
    if output == "deploy_log":
        Path(state[output]).touch()
    write("state", state)
    assert deployment.main(args, git_reader=clean_release_git) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "PASS"


def test_check_only_requires_precreated_appendable_log(release, tmp_path, monkeypatch, capsys):
    state, write = release
    log = tmp_path / "operator.log"
    state["deploy_log"] = str(log)
    monkeypatch.setattr(deployment.time, "time", lambda: 1000)
    args = ["--state", write("state", state), "--check-only"]

    assert deployment.main(args, git_reader=clean_release_git) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == ["deploy_log"]
    assert not log.exists()

    log.touch()
    assert deployment.main(args, git_reader=clean_release_git) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "PASS"
    assert log.read_bytes() == b""


def test_deployment_contract_documents_package_hash_window():
    guide = (ROOT / "gates" / "ENTRY_POINTS.md").read_text(encoding="utf-8")
    assert ("The gate hashes the package directory at preflight and cannot detect changes between preflight and the copy. "
            "The freeze protects the package during that interval.") in guide.splitlines()


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
        assert "frozen_candidate" in deployment.evaluate(state, git_reader=clean_release_git, now=1000)["failures"]
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


@pytest.mark.parametrize("published, expected", [("c" * 40, [HEAD, "b" * 40]), (None, [HEAD, "b" * 40, "c" * 40])])
def test_lss_gh_scans_only_commits_the_destinations_main_lacks(publication, monkeypatch, published, expected):
    runner, calls, _ = publication
    monkeypatch.setattr(gh.guard, "published_main", lambda repo, remote, url: published if remote == "origin" else None)
    scanned = []
    monkeypatch.setattr(gh.guard, "scan", lambda repo, oid, **k: scanned.append(oid) or {"result": "PASS"})
    def answer(argv, *, cwd, input=None):
        if argv[3:] == ["--no-replace-objects", "rev-list", HEAD, "^" + "c" * 40]:
            calls.append((argv, input))
            return SimpleNamespace(stdout=HEAD + "\n" + "b" * 40 + "\n")
        if argv[3:] == ["--no-replace-objects", "rev-list", HEAD]:
            calls.append((argv, input))
            return SimpleNamespace(stdout="\n".join([HEAD, "b" * 40, "c" * 40]) + "\n")
        return runner(argv, cwd=cwd, input=input)
    assert gh.main(["--dry-run", "push"], runner=answer) == 0
    assert scanned == expected


def test_lss_gh_scans_the_tip_even_when_main_already_holds_it(publication, monkeypatch):
    runner, calls, _ = publication
    monkeypatch.setattr(gh.guard, "published_main", lambda *a: HEAD)
    scanned = []
    monkeypatch.setattr(gh.guard, "scan", lambda repo, oid, **k: scanned.append(oid) or {"result": "PASS"})
    def answer(argv, *, cwd, input=None):
        if argv[3:] == ["--no-replace-objects", "rev-list", HEAD, "^" + HEAD]:
            return SimpleNamespace(stdout="")
        return runner(argv, cwd=cwd, input=input)
    assert gh.main(["--dry-run", "push"], runner=answer) == 0
    assert scanned == [HEAD]


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


def test_a_stalled_destination_query_ends_and_falls_back_to_a_full_scan(tmp_path, monkeypatch):
    import time
    real_popen = subprocess.Popen
    # git stands in for a stalled ls-remote whose helper (a grandchild) keeps git's output open.
    stall = ("import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
             "time.sleep(60)")
    monkeypatch.setattr(gh.guard.subprocess, "Popen", lambda argv, **k: real_popen(
        [sys.executable, "-c", stall] if argv[:1] == ["git"] else argv, **k))
    monkeypatch.setattr(gh.guard, "LS_REMOTE_TIMEOUT", 2)
    monkeypatch.setattr(gh.guard, "url_rewrites", lambda repo: False)
    started = time.monotonic()
    assert gh.guard.published_main(tmp_path, "origin", "https://example.invalid/repo.git") is None
    assert time.monotonic() - started < 30


def test_a_refused_tree_kill_still_ends_git_and_falls_back(tmp_path, monkeypatch):
    import time
    real_popen = subprocess.Popen
    def popen(argv, **k):
        if argv[:1] == ["taskkill"]:
            raise OSError("refused")
        return real_popen([sys.executable, "-c", "import time; time.sleep(60)"] if argv[:1] == ["git"] else argv, **k)
    monkeypatch.setattr(gh.guard.subprocess, "Popen", popen)
    monkeypatch.setattr(gh.guard, "LS_REMOTE_TIMEOUT", 2)
    monkeypatch.setattr(gh.guard, "url_rewrites", lambda repo: False)
    started = time.monotonic()
    assert gh.guard.published_main(tmp_path, "origin", "https://example.invalid/repo.git") is None
    assert time.monotonic() - started < 30


@pytest.mark.parametrize("reply,local,trusted", [
    ("c" * 40 + "\trefs/heads/main\n", True, True),
    ("c" * 40 + "\trefs/heads/main\n", False, False),     # the destination's main is not in this repository
    ("", True, False),                                    # no main at the destination
    (None, True, False),                                  # the destination cannot be asked
    ("c" * 40 + "\trefs/heads/main\n" + "d" * 40 + "\trefs/heads/main\n", True, False)])
def test_only_the_destinations_own_main_is_trusted(tmp_path, monkeypatch, reply, local, trusted):
    main = "c" * 40
    asked = []
    def git(repo, *args, **kwargs):
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


def test_a_url_rewrite_rule_disables_the_trust(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert gh.guard.url_rewrites(tmp_path) is False
    subprocess.run(["git", "-C", str(tmp_path), "config", "url.https://example.invalid/b/.insteadOf",
                    "https://example.invalid/a/"], check=True)
    assert gh.guard.url_rewrites(tmp_path) is True
    assert gh.guard.published_main(tmp_path, "origin", "https://example.invalid/a/repo.git") is None
    empty = tmp_path / "empty.config"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG", str(empty))   # narrows `git config` only; the rule still applies
    assert gh.guard.url_rewrites(tmp_path) is True
