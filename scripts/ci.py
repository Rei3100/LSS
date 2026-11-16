"""CI glue: trusted-base classification, PR collection, forms and strict aggregation.

Network access exists only in the explicit fetch-review subcommand, on Actions.
API failures never print response bodies or authentication values.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.change_class import POLICY, diff_state, git, load_policy
from gates.review_gate import verify

ROOT = Path(os.environ.get("LSS_CI_REPO", Path(__file__).resolve().parents[1]))


def required_jobs(kind, event):
    if kind not in ("A", "B", "C") or event not in ("push", "pull_request", "pull_request_review"):
        raise ValueError("Unknown classification/event")
    jobs = {"classify", "public-guard", "docs-check"}
    if event != "push":
        jobs.add("pr-form")
    if kind in ("B", "C"):
        jobs.update(("test-linux", "test-windows"))
        if event != "push":
            jobs.add("review-gate")
    return jobs


def aggregate(needs, event):
    classification = needs.get("classify", {})
    if classification.get("result") != "success":
        return False
    try:
        required = required_jobs(classification.get("outputs", {}).get("class"), event)
    except ValueError:
        return False
    return all(needs.get(name, {}).get("result") == "success" for name in required)


def form_errors(body, kind):
    fields = ("Checked", "Not checked", "Rollback", "Implementation engines")
    if kind == "C":
        fields += ("Review",)
    headings = list(re.finditer(
        r"(?m)^ {0,3}(?P<atx>#{1,6})(?:[ \t]+(?P<title>[^\r\n]*)|[ \t]*$)\r?$"
        r"|^ {0,3}[^ \t\r\n][^\r\n]*\r?\n {0,3}(?:=+|-+)[ \t]*\r?$", body))
    populated = set()
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        if heading.group("atx") in ("##", "###") and body[heading.end():end].strip():
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group("title") or "").strip()
            populated.add(title)
    return ["Missing PR field: " + field for field in fields if field not in populated]


def event_revisions(event):
    if "pull_request" in event:
        return event["pull_request"]["base"]["sha"], event["pull_request"]["head"]["sha"]
    return event["before"], event["after"]


def classify_event(event):
    base, head = event_revisions(event)
    bootstrap = base == "0" * 40
    if bootstrap:
        # The empty tree is repository-format aware. No index mutation.
        base = subprocess.run(["git", "hash-object", "-t", "tree", "--stdin"], input=b"",
                              check=True, capture_output=True, cwd=ROOT).stdout.decode().strip()
        raw = git(ROOT, "show", f"{head}:{POLICY}")
    else:
        raw = git(ROOT, "show", f"{base}:{POLICY}")
    policy = load_policy(raw)
    result = diff_state(ROOT, base, head, policy)
    result.update(base_sha=base, head_sha=head, bootstrap=bootstrap)
    return result


def embedded_records(body, tag):
    records = []
    for raw in re.findall(r"```" + re.escape(tag) + r"\s*\n(.*?)\n```", body, re.S):
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError("Invalid receipt")
        records.append(row)
    return records


class GitHub:
    def __init__(self, repo, token):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Invalid repository")
        self.repo, self.token = repo, token

    def request(self, route, payload=None):
        url = "https://api.github.com/" + route
        request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None,
            headers={"Authorization": "Bearer " + self.token, "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if isinstance(result, dict) and result.get("errors"):
            raise ValueError("Incomplete GraphQL result")
        return result

    def pages(self, route):
        rows = []
        for page in range(1, 1001):
            values = self.request(f"repos/{self.repo}/{route}?per_page=100&page={page}")
            if not isinstance(values, list):
                raise ValueError("Incomplete REST collection")
            rows.extend(values)
            if len(values) < 100:
                return rows
        raise ValueError("REST pagination limit")

    def threads(self, number):
        owner, name = self.repo.split("/")
        cursor, rows = None, []
        query = """query($owner:String!,$name:String!,$number:Int!,$cursor:String) {
          repository(owner:$owner,name:$name) { pullRequest(number:$number) {
            reviewThreads(first:100,after:$cursor) {
              nodes { isResolved comments(first:100) { nodes { body } pageInfo { hasNextPage } } }
              pageInfo { hasNextPage endCursor }
            }
          } }
        }"""
        for _ in range(1000):
            result = self.request("graphql", {"query": query, "variables": {
                "owner": owner, "name": name, "number": number, "cursor": cursor}})
            connection = result["data"]["repository"]["pullRequest"]["reviewThreads"]
            for node in connection["nodes"]:
                comments = node["comments"]
                if comments["pageInfo"]["hasNextPage"]:
                    raise ValueError("Thread comments truncated; refusing partial review")
                text = "\n".join(c["body"] for c in comments["nodes"])
                match = re.search(r"\bP[01]\b", text)
                rows.append({"isResolved": node["isResolved"],
                             "priority": match[0] if match else None})
            if not connection["pageInfo"]["hasNextPage"]:
                return rows
            cursor = connection["pageInfo"]["endCursor"]
        raise ValueError("GraphQL pagination limit")


def collect_review(api, number, expected):
    current = api.request(f"repos/{api.repo}/pulls/{number}")
    if (current["head"]["sha"] != expected["head_sha"] or
            current["base"]["sha"] != expected["base_sha"]):
        raise ValueError("PR changed during run")
    body = current.get("body") or ""
    provenance = re.search(r"(?im)^#{2,3} Implementation engines\s*\n([^\n]+)", body)
    engines = [e.strip().lower() for e in provenance[1].split(",")] if provenance else []
    # Only the owner and collaborators can post records: anyone may comment on a public pull request.
    comments = [c for c in api.pages(f"issues/{number}/comments")
                if c.get("author_association") in ("OWNER", "COLLABORATOR", "MEMBER")]
    records = [r for c in comments for r in embedded_records(c.get("body", ""), "lss-review-record")]
    receipts = [r for c in comments for r in embedded_records(c.get("body", ""), "lss-review-receipt")]
    reviews = []
    for raw in api.pages(f"pulls/{number}/reviews"):
        matches = [r for r in receipts if r.get("review_id") == raw["id"]]
        # Conflicting receipts cannot silently select a favorable one.
        receipt = matches[0] if len(matches) == 1 else {}
        user = raw["user"]
        reviews.append({"author": {"login": user["login"], "id": user["node_id"], "type": user["type"]},
                        "app_id": (raw.get("performed_via_github_app") or {}).get("id"),
                        "state": raw["state"], "body": raw.get("body") or "",
                        "commit_id": raw["commit_id"], "receipt": receipt})
    ancestors, changes = {}, {}
    policy = load_policy(git(ROOT, "show", f"{expected['base_sha']}:{POLICY}"))
    for receipt in [r["receipt"] for r in reviews] + records:
        revision = receipt.get("review_head_sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            continue
        try:
            git(ROOT, "merge-base", "--is-ancestor", revision, expected["head_sha"])
            ancestors[revision] = True
            changes[revision] = diff_state(ROOT, revision, expected["head_sha"], policy)["class"]
        except subprocess.CalledProcessError:
            ancestors[revision] = False
    threads = api.threads(number)
    # Detect head/base races after pagination, as well as before it.
    latest = api.request(f"repos/{api.repo}/pulls/{number}")
    if latest["head"]["sha"] != expected["head_sha"] or latest["base"]["sha"] != expected["base_sha"]:
        raise ValueError("PR changed during collection")
    return {**expected, "draft": current["draft"], "complete": True,
            "implementation_engines": engines, "reviews": reviews, "review_threads": threads,
            "independent_records": records, "ancestors": ancestors, "changes_since_review": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("classify", "pr-form", "public-metadata", "fetch-review", "result"))
    args = parser.parse_args()
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        if args.command == "result":
            return int(not aggregate(json.loads(os.environ["CI_NEEDS"]), os.environ["GITHUB_EVENT_NAME"]))
        expected = classify_event(event)
        if args.command == "classify":
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
                out.write("class=" + expected["class"] + "\n")
            print(json.dumps(expected, sort_keys=True))
            return 0
        pr = event.get("pull_request", {})
        if args.command == "pr-form":
            errors = form_errors(pr.get("body") or "", expected["class"])
            print(json.dumps(errors))
            return int(bool(errors))
        if args.command == "public-metadata":
            from gates.public_guard import Reporter
            reporter = Reporter([])
            reporter.text("<pr-title>", pr.get("title") or "")
            reporter.text("<pr-body>", pr.get("body") or "")
            revisions = git(ROOT, "rev-list", expected["head_sha"] if expected["bootstrap"] else
                            expected["base_sha"] + ".." + expected["head_sha"]).decode().splitlines()
            for revision in revisions:
                reporter.text("<commit-metadata>", git(ROOT, "show", "-s", "--format=%B", revision).decode())
            print(json.dumps({"findings": reporter.findings}))
            return int(bool(reporter.findings))
        api = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
        data = collect_review(api, pr["number"], expected)
        policy = json.loads(git(ROOT, "show", f"{expected['base_sha']}:.github/review-policy.json"))
        # JSON is passed in memory, never uploaded or printed (PR text is untrusted).
        errors = verify(data, policy)
        print(json.dumps({"passed": not errors, "reasons": errors, "head_sha": expected["head_sha"]}))
        return int(bool(errors))
    except Exception:
        print("CI input, policy, Git or API operation failed; no input values logged", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
