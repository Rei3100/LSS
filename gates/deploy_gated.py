"""Evaluate seven release conditions from identity-bound receipts before deployment."""
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
              "rollback_ready", "machine_idle")


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
    try:
        manifest = load(state["manifest"])
        root = Path(manifest["package_dir"])
        commit = manifest["source_commit"]
        exe_hash = manifest["exe_sha256"]
        tree_hash = manifest["tree_sha256"]
        identity = {"source_commit": commit, "exe_sha256": exe_hash, "tree_sha256": tree_hash}
    except (OSError, ValueError, KeyError, TypeError):
        return verdict(checks)

    def receipt(key, *, candidate=True):
        item = load(state[key])
        if item.get("status") != "PASS":
            raise ValueError("Receipt did not pass")
        if candidate and any(item.get(k) != v for k, v in identity.items()):
            raise ValueError("Receipt identity mismatch")
        return item

    def check(name, action):
        try:
            checks[name] = bool(action())
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            checks[name] = False

    check("integrated_main", lambda: git_reader(state["repo"], "rev-parse", "origin/main") == commit)

    def frozen():
        if link_like(root):
            return False
        rows = manifest["files_sha256"]
        actual = {}
        for path in root.rglob("*"):
            if link_like(path):
                return False
            if path.is_file():
                actual[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        tree = hashlib.sha256("".join(f"{k}\t{v['sha256']}\n" for k, v in sorted(actual.items())).encode()).hexdigest()
        build = load(state["build_receipt"])
        return (manifest["type"] == "lss_frozen_candidate_package" and
                actual == rows and bool(actual) and tree == tree_hash and
                actual["LSS.exe"]["sha256"] == exe_hash and
                build["status"] == "complete" and build["phase"] == "complete" and
                build["commit"] == commit and build["exe_sha256"] == exe_hash and
                Path(build["candidate"]).resolve() == root.resolve())
    check("frozen_candidate", frozen)
    check("packaged_acceptance", lambda: receipt("packaged_receipt")["scope"] == "real-machine-package")

    def independent():
        item = receipt("verification_receipt")
        builder = receipt("packaged_receipt")["builder"]
        return bool(builder) and bool(item["verifier"]) and item["verifier"] != builder
    check("independent_verification", independent)

    def answers():
        item = receipt("answer_key_receipt")
        current = load(state["current_product"])
        return (item["baseline_exe_sha256"] == current["exe_sha256"] and
                item["answer_key_sha256"] == sha(Path(state["answer_key"])) and
                item["non_regression"] is True and bool(item["user_benefit"].strip()) and
                (item["change_kind"] != "timing-only" or item.get("pcm_identical") is True))
    check("answer_key_non_regression", answers)

    def rollback():
        item = receipt("rollback_receipt", candidate=False)
        current = load(state["current_product"])
        return (item["exe_sha256"] == current["exe_sha256"] == sha(Path(current["deployed_executable"])) and
                Path(item["exe"]).resolve() == Path(current["deployed_executable"]).resolve() and item["startup_pass"] is True)
    check("rollback_ready", rollback)

    def idle():
        item = receipt("idle_receipt")
        age = now - item["observed_at"]
        return (0 <= age <= 60 and item["product_running"] is False and
                item["other_gpu_jobs"] == 0 and item["user_heavy_use"] is False)
    check("machine_idle", idle)
    return verdict(checks)


def verdict(checks):
    return {"result": "PASS" if all(checks.values()) else "BLOCK", "conditions": checks,
            "failures": [name for name, passed in checks.items() if not passed]}


def deploy(state, manifest):
    repo = Path(state["repo"]).resolve()
    # Deployment executes checkout code: bind those bytes to the integrated candidate.
    if (git(repo, "rev-parse", "HEAD") != manifest["source_commit"]
            or git(repo, "rev-parse", "origin/main") != manifest["source_commit"]
            or git(repo, "status", "--porcelain", "--untracked-files=all")):
        raise ValueError("Deployment checkout must be integrated and clean")
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
        if result["result"] == "PASS" and not args.check_only:
            deployer(state, load(state["manifest"]))
            result["deployment"] = "LAUNCHED"
        print(json.dumps(result))
        return 0 if result["result"] == "PASS" else 1
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print(json.dumps(verdict(dict.fromkeys(CONDITIONS, False))))
        return 1


if __name__ == "__main__":
    sys.exit(main())
