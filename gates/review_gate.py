"""Fail-closed review receipt verification. Read JSON from stdin; emit reasons only.

Input identity and completion policy must come from the trusted base configuration,
never from PR prose. GitHub collection is owned by scripts/ci.py. See CI.md.
"""
from __future__ import annotations

import json
import re
import sys

SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def verify(data: dict, policy: dict) -> list[str]:
    try:
        return _verify(data, policy)
    except (ValueError, KeyError, TypeError, AttributeError, re.error):
        return ["Malformed review input"]


def _verify(data: dict, policy: dict) -> list[str]:
    errors = []
    mode = policy.get("mode")
    if mode not in ("bot", "local-records"):
        return ["Missing or invalid review policy mode"]
    if mode == "bot":
        identity = policy.get("bot", {})
        if (not identity.get("login") or not identity.get("id") or not identity.get("app_id")
                or not policy.get("completion_pattern")):
            return ["Bot identity/completion policy is not configured (S3)"]
    if data.get("draft") is not False:
        errors.append("Draft or unknown PR state")
    for key, pattern in (("head_sha", SHA), ("base_sha", SHA), ("diff_digest", DIGEST)):
        if not isinstance(data.get(key), str) or not pattern.fullmatch(data[key]):
            errors.append("Missing current revision identity")
    if data.get("complete") is not True:
        errors.append("Incomplete GitHub snapshot")
    engines = data.get("implementation_engines")
    if not isinstance(engines, list) or not engines or any(e not in ("claude", "codex", "human") for e in engines):
        errors.append("Missing or invalid implementation provenance")
        engines = []
    reviews = data.get("reviews", [])

    def current(receipt):
        if not isinstance(receipt, dict):
            return False
        r = receipt.get("review_head_sha")
        return (isinstance(r, str) and SHA.fullmatch(r) and
                receipt.get("base_sha") == data.get("base_sha") and
                receipt.get("diff_digest") == data.get("diff_digest") and
                (r == data.get("head_sha") or
                 (data.get("ancestors", {}).get(r) is True and
                  data.get("changes_since_review", {}).get(r) == "A")))

    if mode == "bot":
        accepted = False
        for review in reviews:
            author = review.get("author", {})
            if (author.get("login") != identity.get("login") or author.get("id") != identity.get("id")
                    or author.get("type") != "Bot" or review.get("app_id") != identity.get("app_id")):
                continue
            receipt = review.get("receipt", {})
            if (review.get("state") in ("APPROVED", "COMMENTED", "CHANGES_REQUESTED")
                    and review.get("commit_id") == receipt.get("review_head_sha")
                    and re.search(policy["completion_pattern"], review.get("body", ""))
                    and current(receipt)):
                accepted = True
        if not accepted:
            errors.append("No current completed bot review receipt")
    else:
        records = data.get("independent_records", [])
        if not isinstance(records, list):
            return ["Malformed independent review records"]
        reviewers = set()
        for record in records:
            if (isinstance(record, dict) and record.get("engine") in ("claude", "codex")
                    and record.get("complete") is True
                    and isinstance(record.get("summary"), str) and record["summary"].strip()
                    and type(record.get("findings_p0p1")) is int and record["findings_p0p1"] >= 0
                    and type(record.get("unresolved_p0p1")) is int and record["unresolved_p0p1"] == 0
                    and current(record)):
                reviewers.add(record["engine"])
        for engine in engines:
            required = {"claude": {"codex"}, "codex": {"claude"}, "human": {"claude", "codex"}}[engine]
            if not reviewers.intersection(required):
                errors.append("Missing current independent review record for " + engine + " implementation")
    threads = data.get("review_threads")
    if not isinstance(threads, list):
        errors.append("Missing review threads")
    else:
        for thread in threads:
            # An unknown unresolved thread is also blocking; never infer absence of priority.
            if thread.get("isResolved") is not True:
                priority = thread.get("priority")
                if priority in (None, "P0", "P1"):
                    errors.append("Unresolved P0/P1 or unclassified thread")
            if thread.get("disposition") == "rejected" and not thread.get("rejection_reason_recorded"):
                errors.append("Rejected thread lacks recorded reason")
    if mode == "bot" and "codex" in engines:
        if not any(r.get("engine") == "claude" and r.get("complete") is True
                   and isinstance(r.get("summary"), str) and r["summary"].strip()
                   and current(r) for r in data.get("independent_records", [])):
            errors.append("Missing current independent Claude record")
    return sorted(set(errors))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        errors = verify(payload["pr"], payload["policy"])
    except (ValueError, KeyError, TypeError, AttributeError, re.error):
        errors = ["Malformed review input"]
    print(json.dumps({"passed": not errors, "reasons": errors}, sort_keys=True))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
