"""Narrow, fail-closed publication entry point. No generic gh passthrough."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = sibling("public_guard")


COMMAND_TIMEOUT = 120
# The pre-push hook scans every commit the remote lacks (about 14 s each on this machine), so a new branch
# with ten commits outlasts the ordinary limit; a push still ends within this bound.
PUSH_TIMEOUT = 900


def run(argv, *, cwd, input=None):
    push = argv[:1] == ["git"] and argv[3:4] == ["push"]
    return run_bounded(argv, cwd=cwd, input=input, timeout=PUSH_TIMEOUT if push else COMMAND_TIMEOUT)


def run_bounded(argv, *, cwd, input=None, timeout):
    """subprocess.run(check=True) whose timeout also ends the command's children.

    A hook started by git can keep the output pipes open after git itself is killed, and reading them would then
    wait without limit; the whole process tree is ended first.
    """
    # UTF-8 explicitly: the Windows default code page cannot carry every character of a body.
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                               **options)
    try:
        stdout, stderr = process.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, timeout=30)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=30)
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, argv, stdout, stderr)
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def numeric_id(value):
    if not re.fullmatch(r"[0-9]+", value) or not value.strip("0"):
        raise argparse.ArgumentTypeError("Expected a positive numeric ID")
    return value


def format_review_record(record, head):
    """Validate design 7.3 identity/completion fields and emit the collector format."""
    if not isinstance(record, dict):
        raise ValueError("Invalid review record")
    for key, size in (("base_sha", 40), ("review_head_sha", 40), ("diff_digest", 64)):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{" + str(size) + r"}", record[key]):
            raise ValueError("Invalid review identity")
    if (record["review_head_sha"] != head or record.get("engine") not in ("claude", "codex")
            or record.get("complete") is not True or not isinstance(record.get("summary"), str)
            or not record["summary"].strip()):
        raise ValueError("Incomplete or stale independent review")
    if (type(record.get("findings_p0p1")) is not int or record["findings_p0p1"] < 0
            or type(record.get("unresolved_p0p1")) is not int):
        raise ValueError("Invalid independent review finding counts")
    return "```lss-review-record\n" + json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n```"


def main(argv=None, *, runner=run):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dry-run", action="store_true")
    sub = parser.add_subparsers(dest="operation", required=True)
    pr = sub.add_parser("pr-create", allow_abbrev=False)
    pr.add_argument("--title", required=True)
    pr.add_argument("--body-file", type=Path, required=True)
    pr.add_argument("--draft", action="store_true")
    p = sub.add_parser("issue-create", allow_abbrev=False)
    p.add_argument("--title", required=True)
    p.add_argument("--body-file", type=Path, required=True)
    p.add_argument("--label", action="append", default=[])
    for name in ("issue-comment", "issue-close", "issue-edit", "pr-comment",
                 "pr-ready", "pr-draft", "pr-close"):
        p = sub.add_parser(name, allow_abbrev=False)
        p.add_argument("--issue" if name.startswith("issue-") else "--pr",
                       type=numeric_id, required=True)
        if name in ("issue-comment", "pr-comment"):
            p.add_argument("--body-file", type=Path, required=True)
        if name == "issue-comment":
            p.add_argument("--allow-canary-test", action="store_true")
        if name in ("issue-close", "pr-close"):
            p.add_argument("--reason-file", type=Path, required=True)
        if name == "issue-edit":
            p.add_argument("--add-label", action="append", default=[])
            p.add_argument("--remove-label", action="append", default=[])
    p = sub.add_parser("comment-delete", allow_abbrev=False)
    p.add_argument("--comment-id", type=numeric_id, required=True)
    for name in ("thread-reply", "thread-resolve", "review-record"):
        p = sub.add_parser(name, allow_abbrev=False)
        p.add_argument("--reason-file", type=Path, required=True)
        if name == "review-record":
            p.add_argument("--pr", type=int, required=True)
        else:
            p.add_argument("--thread", required=True)
    sub.add_parser("push", allow_abbrev=False)
    p = sub.add_parser("ci-rerun", allow_abbrev=False)
    p.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        # owner/name comes from the machine, not the public tree (no account names in it).
        REPOSITORY = os.environ["LSS_GITHUB_REPOSITORY"]
        if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+", REPOSITORY):
            raise ValueError("Invalid repository name")
        repo = Path(os.environ["LSS_REPO_ROOT"]).resolve(strict=True)
        def git(*parts):
            return runner(["git", "-C", str(repo), *parts], cwd=repo).stdout.strip()
        origin = git("remote", "get-url", "origin")
        allowed = ("https://github.com/" + REPOSITORY + ".git", "git@github.com:" + REPOSITORY + ".git")
        push_urls = git("remote", "get-url", "--push", "--all", "origin").splitlines()
        if origin not in allowed or not push_urls or any(url not in allowed for url in push_urls):
            raise ValueError("Wrong repository")
        if args.operation not in ("issue-create", "issue-comment", "issue-close", "issue-edit",
                                  "pr-comment", "comment-delete"):
            branch = git("symbolic-ref", "--short", "HEAD")
            if (branch != os.environ["LSS_AGENT_BRANCH"] or
                    not re.fullmatch(r"(?:codex|claude)/[A-Za-z0-9][A-Za-z0-9._/-]*", branch)):
                raise ValueError("Not the assigned agent branch")
            head = git("rev-parse", "HEAD")
            if not re.fullmatch(r"[0-9a-f]{40,64}", head):
                raise ValueError("Invalid HEAD")
        reporter = guard.Reporter(guard.load_denylist(Path(os.environ.get(
            "LSS_PUBLISH_DENYLIST", guard.DEFAULT_DENYLIST))))
        commands = []
        def outgoing(parts, text=None, *, allow_canary=False):
            reporter.text("<outgoing>", json.dumps(parts, ensure_ascii=False))
            if text is not None:
                if len(text.encode("utf-8")) > 1024 * 1024:
                    raise ValueError("Text too large")
                if allow_canary:
                    if not guard.PATTERNS["CANARY"].fullmatch(text):
                        raise ValueError("Expected exactly one canary token")
                    canary_reporter = guard.Reporter(reporter.literals)
                    canary_reporter.text("<outgoing>", text)
                    if (len(canary_reporter.findings) != 1 or
                            canary_reporter.findings[0]["rule"] != "CANARY"):
                        raise ValueError("Outgoing canary rejected")
                else:
                    reporter.text("<outgoing>", text)
            commands.append((parts, text))
        def body(path):
            guard.require_plain_path(path)
            # Keep one bounded snapshot, including if the file grows after inspection.
            with path.open("rb") as stream:
                data = stream.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ValueError("Text too large")
            value = data.decode("utf-8").replace("\r\n", "\n")
            if not value.strip():
                raise ValueError("Empty text")
            return value
        def labels(parts, option, values):
            for value in values:
                if not re.fullmatch(r"[a-z0-9:_-]{1,40}", value):
                    raise ValueError("Invalid label")
                if value.startswith("-"):
                    parts.append(option + "=" + value)
                else:
                    parts.extend([option, value])
        issue_target = None
        def require_issue(number):
            # gh issue commands also accept pull request numbers; closing or relabelling a pull
            # request must stay behind the assigned-branch check of the pr-* operations.
            info = json.loads(runner(["gh", "issue", "view", number, "--repo", REPOSITORY,
                                      "--json", "url"], cwd=repo).stdout)
            if info != {"url": "https://github.com/" + REPOSITORY + "/issues/" + str(int(number))}:
                raise ValueError("Not an issue of the fixed repository")
        if hasattr(args, "title"):
            if len(args.title.encode("utf-8")) > 1024 * 1024:
                raise ValueError("Title too large")
            # Inspect the literal title too; JSON escaping must not hide private literals.
            reporter.text("<outgoing>", args.title)
        if args.operation in ("push", "pr-create"):
            # Scan every ancestor, including metadata. Do not assume remote history is safe.
            for oid in git("rev-list", head).splitlines():
                if guard.scan(repo, oid)["result"] != "PASS":
                    raise ValueError("Outgoing commit rejected")
        if args.operation == "push":
            commands.append((["git", "-C", str(repo), "push", "origin", head + ":refs/heads/" + branch], None))
            reporter.text("<outgoing>", branch)
        elif args.operation == "pr-create":
            parts = ["gh", "pr", "create", "--repo", REPOSITORY, "--base", "main",
                     "--head", branch, "--title", args.title, "--body-file", "-"]
            if args.draft:
                parts.append("--draft")
            outgoing(parts, body(args.body_file))
        elif args.operation == "issue-create":
            parts = ["gh", "issue", "create", "--repo", REPOSITORY,
                     "--title", args.title, "--body-file", "-"]
            labels(parts, "--label", args.label)
            outgoing(parts, body(args.body_file))
        elif args.operation in ("issue-comment", "pr-comment"):
            kind = "issue" if args.operation == "issue-comment" else "pr"
            number = args.issue if kind == "issue" else args.pr
            # A numeric selector and explicit --repo cannot select a PR/issue by foreign URL.
            outgoing(["gh", kind, "comment", number, "--repo", REPOSITORY, "--body-file", "-"],
                     body(args.body_file), allow_canary=getattr(args, "allow_canary_test", False))
        elif args.operation == "issue-edit":
            if not args.add_label and not args.remove_label:
                raise ValueError("No label changes")
            issue_target = args.issue
            parts = ["gh", "issue", "edit", args.issue, "--repo", REPOSITORY]
            labels(parts, "--add-label", args.add_label)
            labels(parts, "--remove-label", args.remove_label)
            outgoing(parts)
        elif args.operation in ("pr-ready", "pr-draft", "pr-close", "issue-close"):
            kind = "issue" if args.operation == "issue-close" else "pr"
            number = args.issue if kind == "issue" else args.pr
            if kind == "pr":
                info = json.loads(runner(["gh", "pr", "view", number, "--repo", REPOSITORY,
                                          "--json", "headRefName,isCrossRepository"], cwd=repo).stdout)
                if info != {"headRefName": branch, "isCrossRepository": False}:
                    raise ValueError("PR does not belong to assigned branch")
            else:
                issue_target = number
            if args.operation in ("pr-close", "issue-close"):
                outgoing(["gh", kind, "comment", number, "--repo", REPOSITORY,
                          "--body-file", "-"], body(args.reason_file))
                outgoing(["gh", kind, "close", number, "--repo", REPOSITORY])
            else:
                parts = ["gh", "pr", "ready", number, "--repo", REPOSITORY]
                if args.operation == "pr-draft":
                    parts.append("--undo")
                outgoing(parts)
        elif args.operation == "comment-delete":
            outgoing(["gh", "api", "--method", "DELETE",
                      "repos/" + REPOSITORY + "/issues/comments/" + args.comment_id])
        elif args.operation == "ci-rerun":
            if args.run_id <= 0:
                raise ValueError("Invalid run")
            info = json.loads(runner(["gh", "run", "view", str(args.run_id), "--repo", REPOSITORY,
                                      "--json", "headSha,headBranch,event"], cwd=repo).stdout)
            if info != {"headSha": head, "headBranch": branch, "event": "pull_request"}:
                raise ValueError("Run does not belong to current HEAD")
            outgoing(["gh", "run", "rerun", str(args.run_id), "--repo", REPOSITORY])
        else:
            text = body(args.reason_file)
            if args.operation == "review-record":
                record = json.loads(text)
                text = format_review_record(record, head)
                if args.pr <= 0:
                    raise ValueError("Stale review record")
                info = json.loads(runner(["gh", "pr", "view", str(args.pr), "--repo", REPOSITORY,
                                          "--json", "headRefOid,headRefName,isCrossRepository"], cwd=repo).stdout)
                if info != {"headRefOid": head, "headRefName": branch, "isCrossRepository": False}:
                    raise ValueError("PR does not belong to assigned branch")
                outgoing(["gh", "pr", "comment", str(args.pr), "--repo", REPOSITORY,
                          "--body-file", "-"], text)
            else:
                if not re.fullmatch(r"PRRT_[A-Za-z0-9_=-]+", args.thread):
                    raise ValueError("Invalid thread")
                query = ('query($id:ID!){node(id:$id){... on PullRequestReviewThread{'
                         'pullRequest{headRefOid headRefName repository{nameWithOwner}}}}}')
                info = json.loads(runner(["gh", "api", "graphql", "-f", "query=" + query,
                                          "-f", "id=" + args.thread], cwd=repo).stdout)
                pr_info = info["data"]["node"]["pullRequest"]
                if (pr_info["repository"]["nameWithOwner"] != REPOSITORY or
                        pr_info["headRefOid"] != head or pr_info["headRefName"] != branch):
                    raise ValueError("Thread belongs to another repository or HEAD")
                query = ('mutation($id:ID!,$body:String!){addPullRequestReviewThreadReply('
                         'input:{pullRequestReviewThreadId:$id,body:$body}){comment{id}}}')
                payload = json.dumps({"query": query, "variables": {"id": args.thread, "body": text}})
                outgoing(["gh", "api", "graphql", "--input", "-"], payload)
                if args.operation == "thread-resolve":
                    query = ('mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{id}}}')
                    outgoing(["gh", "api", "graphql", "-f", "query=" + query, "-f", "id=" + args.thread])
        if reporter.findings:
            raise ValueError("Outgoing text rejected")
        if issue_target is not None:
            # After inspection: a rejected request reaches GitHub with no call at all.
            require_issue(issue_target)
        for command, text in commands:
            if args.dry_run:
                print(json.dumps({"argv": command, "stdin": text}))
            else:
                result = runner(command, cwd=repo, input=text)
                if args.operation == "issue-create":
                    print(result.stdout, end="")
        return 0
    except (KeyError, TypeError, OSError, ValueError, guard.GuardError, subprocess.SubprocessError):
        print(json.dumps({"result": "BLOCK", "reason": "Repository, identity, guard or command check failed"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
