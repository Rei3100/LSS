"""Classify base..head paths using the base's policy; no PR declarations are trusted.

The policy is JSON (a YAML subset), so this entry point needs only Python 3.11.
Unknown paths are C, the strictest diff class. D describes operations, not diffs.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess

POLICY = ".github/change-classes.yml"


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def load_policy(raw: bytes) -> dict:
    policy = json.loads(raw)
    if (policy.get("version") != 1 or policy.get("default") != "C"
            or set(policy.get("classes", {})) != {"A", "B", "C"}
            or any(not isinstance(v, list) or any(not isinstance(p, str) or not p for p in v)
                   for v in policy["classes"].values())):
        raise ValueError("Invalid change classification policy")
    return policy


def classify_path(path: str, policy: dict) -> str:
    path = path.replace("\\", "/")
    if path.startswith("/") or any(p in ("", ".", "..") for p in path.split("/")):
        return "C"
    if path == POLICY:
        return "C"
    # Highest class wins, independent of rule order; fnmatchcase '*' includes '/'.
    return next((c for c in ("C", "B", "A") if any(
        fnmatch.fnmatchcase(path, p) for p in policy["classes"][c])), policy["default"])


def classify(paths: list[str], policy: dict) -> dict:
    files = {p: classify_path(p, policy) for p in sorted(set(paths))}
    return {"class": max(files.values(), default="A"), "files": files}


def diff_state(root: Path, base: str, head: str, policy: dict) -> dict:
    names = git(root, "diff", "--no-ext-diff", "--name-only", "--no-renames", "-z", base, head, "--")
    result = classify([p.decode("utf-8") for p in names.split(b"\0") if p], policy)
    digest = hashlib.sha256()
    for path, kind in result["files"].items():
        if kind != "A":
            patch = git(root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "--full-index",
                        "--no-renames", "--no-color", "--src-prefix=a/", "--dst-prefix=b/",
                        base, head, "--", path)
            digest.update(path.encode() + b"\0" + len(patch).to_bytes(8, "big") + patch)
    result["diff_digest"] = digest.hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--bootstrap-policy", type=Path,
                        help="Only for an initial main import with no base commit")
    args = parser.parse_args()
    raw = (args.bootstrap_policy.read_bytes() if args.bootstrap_policy else
           git(args.repo, "show", f"{args.base}:{POLICY}"))
    print(json.dumps(diff_state(args.repo, args.base, args.head, load_policy(raw)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
