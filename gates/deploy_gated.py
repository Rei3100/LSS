"""Evaluate release evidence and launch prerequisites before deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import stat

CONDITIONS = ("integrated_main", "frozen_candidate", "packaged_acceptance",
              "independent_verification", "answer_key_non_regression",
              "rollback_ready", "machine_idle", "checkout_head", "checkout_clean",
              "deploy_base", "deploy_log", "launch_receipt")
INTEGRATED_MAIN_ARGS = ("rev-parse", "--verify", "refs/remotes/origin/main^{commit}")
CHECKOUT_STATUS_ARGS = ("status", "--porcelain", "--untracked-files=all")
CHECK_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError)


class InvalidField(ValueError):
    pass


def field(item, key, expected=(str,), *, absolute=False):
    value = item[key]
    if type(value) not in expected:
        raise InvalidField(f"{key} has the wrong type")
    if absolute and ("\0" in value or not Path(value).is_absolute()):
        raise InvalidField(f"{key} must be an absolute path")
    return value


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    item = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise ValueError("Expected a receipt object")
    return item


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


def link_like(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def evaluate(state, *, git_reader=git, now=None):
    """Missing, malformed, stale and contradictory evidence all fail closed."""
    now = time.time() if now is None else now
    checks = dict.fromkeys(CONDITIONS, False)
    details = {}

    def check(name, action):
        try:
            checks[name] = bool(action())
        except InvalidField as error:
            details[name] = str(error)
        except CHECK_ERRORS:
            checks[name] = False

    try:
        manifest = load(field(state, "manifest", absolute=True))
    except InvalidField as error:
        details["frozen_candidate"] = str(error)
        manifest = {}
    except CHECK_ERRORS:
        manifest = {}

    def receipt(key, *, candidate=True):
        item = load(field(state, key, absolute=True))
        if item.get("status") != "PASS":
            raise ValueError("Receipt did not pass")
        if candidate and any(item.get(k) != field(manifest, k)
                             for k in ("source_commit", "exe_sha256", "tree_sha256")):
            raise ValueError("Receipt identity mismatch")
        return item

    check("integrated_main", lambda: field(manifest, "source_commit") ==
          git_reader(field(state, "repo", absolute=True), *INTEGRATED_MAIN_ARGS))

    def frozen():
        root = Path(field(manifest, "package_dir"))
        commit = field(manifest, "source_commit")
        exe_hash = field(manifest, "exe_sha256")
        tree_hash = field(manifest, "tree_sha256")
        build = load(field(state, "build_receipt", absolute=True))
        if link_like(root):
            return False
        rows = field(manifest, "files_sha256", (dict,))
        actual = {}
        for path in root.rglob("*"):
            if link_like(path):
                return False
            if path.is_file():
                actual[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        tree = hashlib.sha256("".join(f"{k}\t{v['sha256']}\n" for k, v in sorted(actual.items())).encode()).hexdigest()
        return (manifest["type"] == "lss_frozen_candidate_package" and
                actual == rows and bool(actual) and
                all(type(row["bytes"]) is int for row in rows.values()) and tree == tree_hash and
                actual["LSS.exe"]["sha256"] == exe_hash and
                build["status"] == "complete" and build["phase"] == "complete" and
                build["commit"] == commit and build["exe_sha256"] == exe_hash and
                Path(field(build, "candidate")).resolve() == root.resolve())
    check("frozen_candidate", frozen)
    check("packaged_acceptance", lambda: receipt("packaged_receipt")["scope"] == "real-machine-package")

    def independent():
        item = receipt("verification_receipt")
        builder = field(receipt("packaged_receipt"), "builder")
        verifier = field(item, "verifier")
        return bool(builder) and bool(verifier) and verifier != builder
    check("independent_verification", independent)

    def answers():
        item = receipt("answer_key_receipt")
        current = load(field(state, "current_product", absolute=True))
        answer_key = Path(field(state, "answer_key", absolute=True))
        return (item["baseline_exe_sha256"] == field(current, "exe_sha256") and
                item["answer_key_sha256"] == sha(answer_key) and
                item["non_regression"] is True and bool(field(item, "user_benefit").strip()) and
                (field(item, "change_kind") != "timing-only" or item.get("pcm_identical") is True))
    check("answer_key_non_regression", answers)

    def rollback():
        item = receipt("rollback_receipt", candidate=False)
        current = load(field(state, "current_product", absolute=True))
        deployed = Path(field(current, "deployed_executable"))
        return (item["exe_sha256"] == field(current, "exe_sha256") == sha(deployed) and
                Path(field(item, "exe")).resolve() == deployed.resolve() and item["startup_pass"] is True)
    check("rollback_ready", rollback)

    def idle():
        item = receipt("idle_receipt")
        observed_at = field(item, "observed_at", (int, float))
        return (now - 60 <= observed_at <= now and item["product_running"] is False and
                field(item, "other_gpu_jobs", (int,)) == 0 and item["user_heavy_use"] is False)
    check("machine_idle", idle)

    check("checkout_head", lambda: field(manifest, "source_commit") ==
          git_reader(field(state, "repo", absolute=True), "rev-parse", "HEAD"))
    check("checkout_clean", lambda: not git_reader(field(state, "repo", absolute=True), *CHECKOUT_STATUS_ARGS))
    check("deploy_base", lambda: Path(field(state, "deploy_base", absolute=True)).is_dir())

    def deploy_log():
        path = Path(field(state, "deploy_log", absolute=True))
        return path.parent.is_dir() and (not path.exists() or path.is_file())
    check("deploy_log", deploy_log)

    def launch_receipt():
        path = Path(field(state, "launch_receipt", absolute=True))
        return (path.parent.is_dir() and not os.path.lexists(path) and
                (not checks["deploy_log"] or path.resolve() != Path(state["deploy_log"]).resolve()))
    check("launch_receipt", launch_receipt)
    return verdict(checks, details)


def verdict(checks, details=None):
    result = {"result": "PASS" if all(checks.values()) else "BLOCK", "conditions": checks,
              "failures": [name for name, passed in checks.items() if not passed]}
    if details:
        result["failure_details"] = details
    return result


def deploy(state, manifest):
    repo = Path(state["repo"]).resolve()
    # Deployment executes checkout code: bind those bytes to the integrated candidate.
    if git(repo, "rev-parse", "HEAD") != manifest["source_commit"]:
        raise ValueError("checkout_head: HEAD differs from the candidate source commit")
    if git(repo, *INTEGRATED_MAIN_ARGS) != manifest["source_commit"]:
        raise ValueError("integrated_main: origin/main differs from the candidate source commit")
    if git(repo, *CHECKOUT_STATUS_ARGS):
        raise ValueError("checkout_clean: deployment checkout has local changes")
    command = [sys.executable, "-B", str(repo / "scripts" / "hidden_run.py"),
               "--log", state["deploy_log"], "--receipt", state["launch_receipt"],
               "--cwd", str(repo), "--", "pwsh", "-NoProfile", "-NonInteractive", "-File",
               str(repo / "deploy-to-freesoft.ps1"), "-BuildOutput", manifest["package_dir"],
               "-DeployBase", state["deploy_base"], "-KeepPrevious"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)


def main(argv=None, *, deployer=deploy, git_reader=git):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--state", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        path = args.state or Path(os.environ["LSS_PRIVATE_ROOT"]) / "run" / "deploy-gate.json"
        state = load(path)
        result = evaluate(state, git_reader=git_reader)
    except CHECK_ERRORS:
        result = verdict(dict.fromkeys(CONDITIONS, False))
    if result["result"] == "PASS" and not args.check_only:
        try:
            deployer(state, load(state["manifest"]))
            result["deployment"] = "LAUNCHED"
        except CHECK_ERRORS as error:
            result["result"] = "BLOCK"
            result["deployment"] = "REFUSED"
            result["deployment_error"] = str(error)
    print(json.dumps(result))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
