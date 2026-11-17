"""Read-only public surface scan. Finding entry_index values are zero-based."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlencode


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = sibling("public_guard")
KINDS = ("issue_title", "issue_body", "pull_title", "pull_body", "issue_comment",
         "review_comment", "commit_message", "branch_name", "secret_alert")


def run(argv):
    return subprocess.run(argv, text=True, encoding="utf-8", capture_output=True,
                          check=False, timeout=120)


class ScanError(Exception):
    """An incomplete scan; never expose upstream response or exception text."""

    def __init__(self, status=None):
        self.status = status


def instant(value):
    if not isinstance(value, str):
        raise ScanError()
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone required")
    return result.astimezone(timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


class API:
    def __init__(self, repo, runner):
        self.root = "repos/" + repo
        self.runner = runner

    def get(self, suffix="", **params):
        endpoint = self.root + suffix
        if params:
            endpoint += "?" + urlencode(params)
        response = self.runner(["gh", "api", "--hostname", "github.com", "--method", "GET",
                                "--include", "-H", "Accept: application/vnd.github+json",
                                endpoint])
        # HTTP status comes from the response headers, never from diagnostic prose.
        header, separator, body = response.stdout.replace("\r\n", "\n").partition("\n\n")
        match = re.match(r"HTTP/\S+ ([0-9]{3})(?:\s|$)", header)
        if not separator or not match:
            raise ScanError()
        status = int(match[1])
        if response.returncode or not 200 <= status < 300:
            raise ScanError(status)
        return json.loads(body)

    def pages(self, suffix, **params):
        page = 1
        while True:
            items = self.get(suffix, per_page=100, page=page, **params)
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise ScanError()
            yield items
            if len(items) < 100:
                break
            page += 1


class Checker:
    def __init__(self, literals, result):
        self.literals = literals
        self.result = result

    def matches(self, value):
        if not isinstance(value, str):
            raise ScanError()
        data = value.encode("utf-8")
        for index, literal in enumerate(self.literals):
            # Reuse both the word-boundary rules and the binary encoding helper
            # reached by Reporter.content; pattern findings are outside this scan.
            reporter = guard.Reporter([literal])
            reporter.content("<surface>", data)
            if any(item["rule"] == "DENYLIST" for item in reporter.findings):
                yield index

    def safe(self, value):
        if list(self.matches(value)) or list(self.matches(unquote(value))):
            return "<masked>"
        return value

    def text(self, kind, value, url):
        # GitHub bodies may be null, which means an empty body.
        if value is None and kind in ("issue_body", "pull_body", "issue_comment", "review_comment"):
            value = ""
        self.result["counts"][kind] += 1
        for index in self.matches(value):
            self.result["findings"].append({"kind": kind, "url": self.safe(url),
                                            "entry_index": index})
        # The canary is the one generic pattern reported here: it is the token the leak drill posts
        # through `lss-gh issue-comment --allow-canary-test`, and the drill passes only if this scan sees it.
        if guard.PATTERNS["CANARY"].search(value):
            self.result["findings"].append({"kind": kind, "url": self.safe(url),
                                            "entry_index": None, "rule": "CANARY"})


def positive_id(value):
    if type(value) is not int or value <= 0:
        raise ScanError()
    return str(value)


def collect(api, checker, repo, since, until):
    result = checker.result
    try:
        info = api.get()
    except ScanError as error:
        if error.status == 404:
            result["status"] = "not_published"
            return
        raise
    web = "https://github.com/" + repo
    default = info["default_branch"]
    checker.text("branch_name", default, web + "/branches")
    for items in api.pages("/issues", state="all", since=iso(since)):
        for item in items:
            if "pull_request" in item:
                continue  # Pulls are fetched separately, including their refs.
            url = web + "/issues/" + positive_id(item["number"])
            checker.text("issue_title", item["title"], url)
            checker.text("issue_body", item["body"], url)
    # The pulls endpoint has no since filter. Stop at the window boundary in
    # descending update order, rather than fetching the entire pull history.
    for items in api.pages("/pulls", state="all", sort="updated", direction="desc"):
        old = False
        for item in items:
            if instant(item["updated_at"]) < since:
                old = True
                break
            url = web + "/pull/" + positive_id(item["number"])
            checker.text("pull_title", item["title"], url)
            checker.text("pull_body", item["body"], url)
            for side in ("head", "base"):
                checker.text("branch_name", item[side]["ref"], url)
        if old:
            break
    for suffix, kind, anchor in (("/issues/comments", "issue_comment", "issuecomment-"),
                                 ("/pulls/comments", "review_comment", "discussion_r")):
        for items in api.pages(suffix, since=iso(since)):
            for item in items:
                # Construct a locator from numeric IDs; never persist remote URLs.
                parent = item["issue_url" if kind == "issue_comment" else "pull_request_url"]
                if not isinstance(parent, str):
                    raise ScanError()
                number = positive_id(int(parent.rsplit("/", 1)[-1]))
                url = web + ("/issues/" if kind == "issue_comment" else "/pull/") + number
                url += "#" + anchor + positive_id(item["id"])
                checker.text(kind, item["body"], url)
    # An empty repository answers 409 for commits (it still reports a default
    # branch name); its issues, pulls, comments and alerts are still scanned.
    try:
        for items in api.pages("/commits", sha=default, since=iso(since), until=iso(until)):
            for item in items:
                sha = item["sha"]
                if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40,64}", sha):
                    raise ScanError()
                checker.text("commit_message", item["commit"]["message"], web + "/commit/" + sha)
    except ScanError as error:
        if error.status != 409 or result["counts"]["commit_message"]:
            raise
    # Branches and outstanding alerts have no time window: old names and open
    # alerts remain exposed, so every run covers them too.
    for items in api.pages("/branches"):
        for item in items:
            checker.text("branch_name", item["name"], web + "/branches")
    try:
        for items in api.pages("/secret-scanning/alerts", state="open"):
            for item in items:
                if item["state"] != "open":
                    raise ScanError()
                url = web + "/security/secret-scanning/" + positive_id(item["number"])
                result["counts"]["secret_alert"] += 1
                result["findings"].append({"kind": "secret_alert", "url": checker.safe(url),
                                            "entry_index": None})
        result["alerts"] = "known"
    except (ScanError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        result["alerts"] = "unknown"
        return
    result["status"] = "findings" if result["findings"] else "clean"


def main(argv=None, *, runner=run):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", default=os.environ.get("LSS_GITHUB_REPOSITORY", ""),
                        help="owner/name; default: LSS_GITHUB_REPOSITORY, set on the machine (an unset value is a failed scan)")
    parser.add_argument("--since", help="ISO timestamp with timezone; default: three days ago")
    parser.add_argument("--denylist", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    result = {"repo": "<unverified>", "window": {"since": None, "until": iso(now)},
              "status": "unknown", "alerts": "unknown",
              "counts": dict.fromkeys(KINDS, 0), "findings": []}
    try:
        out = args.out_dir
        if out is None:
            out = Path(os.environ["LSS_PRIVATE_ROOT"]) / "run" / "scan"
        denylist = args.denylist or Path(os.environ.get("LSS_PUBLISH_DENYLIST", guard.DEFAULT_DENYLIST))
        checker = Checker(guard.load_denylist(denylist), result)
        if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+", args.repo):
            raise ScanError()
        result["repo"] = checker.safe(args.repo)
        since = instant(args.since) if args.since else now - timedelta(days=3)
        if since > now:
            raise ScanError()
        result["window"]["since"] = iso(since)
        collect(API(args.repo, runner), checker, args.repo, since, now)
    except (ScanError, guard.GuardError, OSError, ValueError, KeyError, TypeError,
            subprocess.SubprocessError):
        result["status"] = "unknown"
    try:
        if out is None:
            raise ScanError()
        guard.require_plain_path(out)
        out.mkdir(parents=True, exist_ok=True)
        # Exclusive creation and microseconds prevent silent replacement of runs.
        path = out / (now.strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
    except (ScanError, guard.GuardError, OSError):
        print('{"status": "unknown", "reason": "Result write failed"}')
        return 1 if result["findings"] else 2
    print(json.dumps({"status": result["status"], "alerts": result["alerts"],
                      "finding_count": len(result["findings"])}))
    if result["findings"]:
        return 1
    return 0 if result["status"] in ("clean", "not_published") else 2


if __name__ == "__main__":
    sys.exit(main())
