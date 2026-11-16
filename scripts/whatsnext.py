"""Derive checkout/resume/environment state without changing it. Offline by default."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

try:
    from .env_check import check, ROOT
except ImportError:
    import sys
    # Run as a script (also under -I, which leaves the script folder off sys.path).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from env_check import check, ROOT


def command(root, args):
    try:
        proc = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=15)
        return proc.stdout if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def resume(private):
    if not private or not Path(private).is_absolute():
        return {"status": "UNKNOWN", "headings": []}
    try:
        # Only the leading resume block is read; prose never enters the output.
        with (Path(private) / "env-build/PROGRESS.md").open(encoding="utf-8") as stream:
            lines = []
            section_seen = False
            for _ in range(160):
                line = stream.readline()
                if not line:
                    break
                if line.startswith("## "):
                    if section_seen:
                        break
                    section_seen = True
                if line.startswith("#"):
                    lines.append(line.strip())
        return {"status": "PRESENT", "headings": lines}
    except FileNotFoundError:
        return {"status": "ABSENT", "headings": []}
    except (OSError, UnicodeError):
        return {"status": "UNKNOWN", "headings": []}


def gpu_state(root):
    path = root / "scripts/gpu_run.py"
    if not path.is_file():
        return {"status": "UNKNOWN"}
    try:
        spec = importlib.util.spec_from_file_location("lss_gpu_run_status", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # _holder() is descriptive metadata, not a lease probe. Never acquire a lease.
        return {"status": "UNKNOWN", "holder_metadata": module._holder(),
                "reason": "Metadata does not prove live lease ownership"}
    except (OSError, ImportError, AttributeError, ValueError):
        return {"status": "UNKNOWN"}


def snapshot(root=ROOT, environ=None, github=False):
    env = os.environ if environ is None else environ
    status = command(root, ["git", "status", "--porcelain=v1", "-z"])
    worktrees_raw = command(root, ["git", "worktree", "list", "--porcelain"])
    worktrees = []
    for block in (worktrees_raw or "").strip().split("\n\n"):
        rows = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        if "worktree" in rows:
            other = command(Path(rows["worktree"]), ["git", "status", "--porcelain=v1", "-z"])
            worktrees.append({"path": rows["worktree"], "head": rows.get("HEAD"),
                              "branch": rows.get("branch"),
                              "dirty": other != "" if other is not None else "UNKNOWN"})
    result = {"checkout": {"head": (command(root, ["git", "rev-parse", "HEAD"]) or "").strip() or None,
                           "branch": (command(root, ["git", "branch", "--show-current"]) or "").strip() or None,
                           "dirty": status != "" if status is not None else "UNKNOWN"},
              "worktrees": worktrees, "resume": resume(env.get("LSS_PRIVATE_ROOT")),
              "gpu": gpu_state(root), "environment": check(root, env),
              "github": {"status": "DISABLED"}, "deployed_version": "UNKNOWN",
              "product_usage": "UNKNOWN", "evidence_resolution": "UNKNOWN",
              "disk_free_bytes": shutil.disk_usage(root).free}
    if github:
        result["github"] = {"status": "UNKNOWN"}
        if shutil.which("gh"):
            for kind in ("pr", "issue"):
                raw = command(root, ["gh", kind, "list", "--state", "open", "--json", "number,title,url"]
                              + (["--assignee", "@me"] if kind == "issue" else []))
                if raw is not None:
                    result["github"][kind] = json.loads(raw)
            if all(k in result["github"] for k in ("pr", "issue")):
                result["github"]["status"] = "PASS"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--github", action="store_true", help="Explicitly allow read-only gh queries")
    args = parser.parse_args()
    print(json.dumps(snapshot(args.repo, github=args.github), sort_keys=True))


if __name__ == "__main__":
    main()
