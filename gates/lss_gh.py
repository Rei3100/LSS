"""Narrow, fail-closed publication entry point. No generic gh passthrough."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = sibling("public_guard")


def run(argv, *, cwd, input=None):
    return subprocess.run(argv, cwd=cwd, input=input, text=True, capture_output=True,
                          check=True, timeout=120)


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
        def outgoing(parts, text=None):
            reporter.text("<outgoing>", json.dumps(parts, ensure_ascii=False))
            if text is not None:
                reporter.text("<outgoing>", text)
            commands.append((parts, text))
        def body(path):
            guard.require_plain_path(path)
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("Text too large")
            value = path.read_text(encoding="utf-8")
            if not value.strip():
                raise ValueError("Empty text")
            return value
        if args.operation in ("push", "pr-create"):
            # Scan every ancestor, including metadata. Do not assume remote history is safe.
            for oid in git("rev-list", head).splitlines():
                if guard.scan(repo, oid)["result"] != "PASS":
                    raise ValueError("Outgoing commit rejected")
        if args.operation == "push":
            commands.append((["git", "-C", str(repo), "push", "origin", head + ":refs/heads/" + branch], None))
            reporter.text("<outgoing>", branch)
        elif args.operation == "pr-create":
            outgoing(["gh", "pr", "create", "--repo", REPOSITORY, "--base", "main",
                      "--head", branch, "--title", args.title, "--body-file", "-"], body(args.body_file))
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
        for command, text in commands:
            if args.dry_run:
                print(json.dumps({"argv": command, "stdin": text}, ensure_ascii=False))
            else:
                runner(command, cwd=repo, input=text)
        return 0
    except (KeyError, TypeError, OSError, ValueError, guard.GuardError, subprocess.SubprocessError):
        print(json.dumps({"result": "BLOCK", "reason": "Repository, identity, guard or command check failed"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
