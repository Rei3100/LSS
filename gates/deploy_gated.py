"""Evaluate release evidence and launch prerequisites before deployment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

CONDITIONS = ("integrated_main", "frozen_candidate", "packaged_acceptance",
              "independent_verification", "answer_key_non_regression",
              "rollback_ready", "machine_idle", "checkout_head", "checkout_clean",
              "deploy_base", "deploy_log", "launch_receipt", "pwsh")
INTEGRATED_MAIN_ARGS = ("rev-parse", "--verify", "refs/remotes/origin/main^{commit}")
CHECKOUT_STATUS_ARGS = ("status", "--porcelain", "--untracked-files=all")
CHECK_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError)


class InvalidField(ValueError):
    pass


@dataclass(frozen=True)
class LaunchPlan:
    repo: Path
    manifest_path: Path
    manifest_digest: str
    source_commit: str
    package_dir: Path
    deploy_base: Path
    pwsh: Path


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


def receipt_object(text):
    item = json.loads(text)
    if not isinstance(item, dict):
        raise ValueError("Expected a receipt object")
    return item


def load(path):
    return receipt_object(Path(path).read_text(encoding="utf-8"))


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, timeout=30).stdout.strip()


def link_like(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def has_reparse_ancestor(path):
    return any(link_like(entry) for entry in (path, *path.parents))


def path_entry_exists(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def path_contains(parent, child):
    # PowerShell compares entire path components without regard to case.
    parent_prefix = os.path.normpath(parent).rstrip(os.sep) + os.sep
    child_prefix = os.path.normpath(child).rstrip(os.sep) + os.sep
    return child_prefix[:len(parent_prefix)].casefold() == parent_prefix.casefold()


def resolve_pwsh():
    # Python 3.11 does not add PATHEXT when shutil.which receives an absolute path.
    executable = "pwsh.exe" if os.name == "nt" else "pwsh"
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        directory = Path(entry)
        if not directory.is_absolute():
            continue
        # An absolute command keeps shutil.which from adding the working directory on Windows.
        found = shutil.which(str(directory / executable))
        if found and Path(found).is_absolute():
            path = Path(found).resolve()
            if path.is_file():
                return path
    return None


def evaluate(state, *, git_reader=git, now=None):
    return preflight(state, git_reader=git_reader, now=now)[0]


def preflight(state, *, git_reader=git, now=None):
    """Missing, malformed, stale and contradictory evidence all fail closed."""
    now = time.time() if now is None else now
    checks = dict.fromkeys(CONDITIONS, False)
    details = {}

    def checked_value(name, action):
        try:
            return action()
        except InvalidField as error:
            details[name] = str(error)
        except CHECK_ERRORS:
            pass

    def check(name, action):
        value = checked_value(name, action)
        checks[name] = bool(value)
        return value

    @cache
    def checkout():
        return Path(field(state, "repo", absolute=True)).resolve()

    def checkout_path(value):
        path = Path(value)
        if not path.is_absolute():
            path = checkout() / path
        if not path.is_absolute():
            raise ValueError("Path cannot be made absolute against the deployment checkout")
        # Keep links visible to the reparse-point checks.
        return Path(os.path.normpath(path))

    try:
        manifest_path = checkout_path(field(state, "manifest", absolute=True))
        content = manifest_path.read_bytes()
        manifest = receipt_object(content.decode("utf-8"))
        manifest_digest = hashlib.sha256(content).hexdigest()
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
          git_reader(checkout(), *INTEGRATED_MAIN_ARGS))

    package_dir = checked_value("frozen_candidate", lambda: checkout_path(field(manifest, "package_dir")))

    def frozen(root):
        commit = field(manifest, "source_commit")
        exe_hash = field(manifest, "exe_sha256")
        tree_hash = field(manifest, "tree_sha256")
        build = load(field(state, "build_receipt", absolute=True))
        if has_reparse_ancestor(root):
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
                checkout_path(field(build, "candidate")) == root)
    check("frozen_candidate", lambda: package_dir is not None and frozen(package_dir))
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

    base = checked_value("deploy_base", lambda: checkout_path(field(state, "deploy_base", absolute=True)))

    def rollback():
        if base is None:
            return False
        item = receipt("rollback_receipt", candidate=False)
        current = load(field(state, "current_product", absolute=True))
        deployed = base / "LSS" / "LSS.exe"
        return (checkout_path(field(current, "deployed_executable")) == deployed and
                checkout_path(field(item, "exe")) == deployed and
                item["exe_sha256"] == field(current, "exe_sha256") == sha(deployed) and
                item["startup_pass"] is True)
    check("rollback_ready", rollback)

    def idle():
        item = receipt("idle_receipt")
        observed_at = field(item, "observed_at", (int, float))
        return (now - 60 <= observed_at <= now and item["product_running"] is False and
                field(item, "other_gpu_jobs", (int,)) == 0 and item["user_heavy_use"] is False)
    check("machine_idle", idle)

    check("checkout_head", lambda: field(manifest, "source_commit") ==
          git_reader(checkout(), "rev-parse", "HEAD"))
    check("checkout_clean", lambda: not git_reader(checkout(), *CHECKOUT_STATUS_ARGS))

    def deploy_base():
        return (base is not None and base.is_dir() and not has_reparse_ancestor(base) and
                not path_entry_exists(base / ".LSS-deploy.json") and
                not path_entry_exists(base / ".LSS.previous") and
                (package_dir is None or not (path_contains(base, package_dir) or path_contains(package_dir, base))))
    check("deploy_base", deploy_base)

    protected_roots = tuple(root for root in (package_dir, base) if root is not None)

    def outside_deploy_paths(path, resolved):
        return not any(path_contains(root, path) or path_contains(root.resolve(), resolved)
                       for root in protected_roots)

    def deploy_log():
        requested = Path(field(state, "deploy_log", absolute=True))
        path = requested.resolve()
        if not outside_deploy_paths(requested, path) or not path.parent.is_dir():
            return None
        try:
            info = path.stat()
        except FileNotFoundError:
            return path
        if not stat.S_ISREG(info.st_mode):
            return None
        os.close(os.open(path, os.O_WRONLY | os.O_APPEND))
        return path
    log = check("deploy_log", deploy_log)

    def launch_receipt():
        path = Path(field(state, "launch_receipt", absolute=True))
        temporary = path.with_suffix(path.suffix + ".tmp")
        resolved = path.resolve()
        return (outside_deploy_paths(path, resolved) and path.parent.is_dir() and
                not path_entry_exists(path) and not path_entry_exists(temporary) and
                log not in (resolved, temporary.resolve()))
    check("launch_receipt", launch_receipt)
    pwsh = check("pwsh", resolve_pwsh)
    result = verdict(checks, details)
    if result["result"] == "PASS":
        return result, LaunchPlan(checkout(), manifest_path, manifest_digest,
                                  manifest["source_commit"], package_dir, base, pwsh)
    return result, None


def verdict(checks, details=None):
    result = {"result": "PASS" if all(checks.values()) else "BLOCK", "conditions": checks,
              "failures": [name for name, passed in checks.items() if not passed]}
    if details:
        result["failure_details"] = details
    return result


def deploy(state, plan):
    repo = plan.repo
    # Deployment executes checkout code: bind those bytes to the integrated candidate.
    if git(repo, "rev-parse", "HEAD") != plan.source_commit:
        raise ValueError("checkout_head: HEAD differs from the candidate source commit")
    if git(repo, *INTEGRATED_MAIN_ARGS) != plan.source_commit:
        raise ValueError("integrated_main: origin/main differs from the candidate source commit")
    if git(repo, *CHECKOUT_STATUS_ARGS):
        raise ValueError("checkout_clean: deployment checkout has local changes")
    if sha(plan.manifest_path) != plan.manifest_digest:
        raise ValueError("frozen_candidate: manifest changed after preflight")
    command = [sys.executable, "-B", str(repo / "scripts" / "hidden_run.py"),
               "--log", state["deploy_log"], "--receipt", state["launch_receipt"],
               "--cwd", str(repo), "--", str(plan.pwsh), "-NoProfile", "-NonInteractive", "-File",
               str(repo / "deploy-to-freesoft.ps1"), "-BuildOutput", str(plan.package_dir),
               "-DeployBase", str(plan.deploy_base), "-KeepPrevious"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)


def main(argv=None, *, deployer=deploy, git_reader=git):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--state", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        path = args.state or Path(os.environ["LSS_PRIVATE_ROOT"]) / "run" / "deploy-gate.json"
        state = load(path)
        result, plan = preflight(state, git_reader=git_reader)
    except CHECK_ERRORS:
        result = verdict(dict.fromkeys(CONDITIONS, False))
        plan = None
    if plan is not None and not args.check_only:
        try:
            deployer(state, plan)
            result["deployment"] = "LAUNCHED"
        except CHECK_ERRORS as error:
            result["result"] = "BLOCK"
            result["deployment"] = "REFUSED"
            result["deployment_error"] = str(error)
    print(json.dumps(result))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
