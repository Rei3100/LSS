"""Which development lanes are moving right now, from git alone (no agent survey needed).

    python scripts/lanes.py [--hours 72] [--limit 12]

One line per claude/* and codex/* branch committed within the window, newest first: age, branch, last
commit subject, and the worktree that has it checked out (if any). Printed at session start by
scripts/dev_hygiene.ps1 so an agent knows the parallel lanes without spending effort to find them.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=cwd or Path(__file__).parent,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout


def lane_records(hours: float, limit: int, cwd: Path | None = None, now: float | None = None) -> list[dict]:
    """Structured lanes (newest first): branch, sha (7), commit unix time, age in hours, subject, worktree or None."""
    checked_out, current = {}, None
    for line in _git("worktree", "list", "--porcelain", cwd=cwd).splitlines():
        if line.startswith("worktree "):
            current = line[9:]
        elif line.startswith("branch refs/heads/"):
            checked_out[line[18:]] = current
    rows = []
    now = time.time() if now is None else now
    for line in _git("for-each-ref", "--sort=-committerdate",
                     "--format=%(committerdate:unix)	%(refname:short)	%(objectname:short=7)	%(subject)",
                     "refs/heads/claude", "refs/heads/codex", cwd=cwd).splitlines():
        stamp, branch, sha, subject = (line.split("	", 3) + ["", "", ""])[:4]
        age = (now - int(stamp)) / 3600
        if age > hours:
            break
        rows.append({"branch": branch, "sha": sha, "time": int(stamp), "age_h": age, "subject": subject,
                     "worktree": checked_out.get(branch)})
        if len(rows) >= limit:
            break
    return rows


def lanes(hours: float, limit: int) -> list[str]:
    return [f"{r['age_h']:5.1f} h  {r['branch']}  — {r['subject'][:90]}" + (f"  [{r['worktree']}]" if r["worktree"] else "")
            for r in lane_records(hours, limit)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=72)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)
    rows = lanes(args.hours, args.limit)
    print(f"Active lanes (last {args.hours:g} h): " + ("none" if not rows else ""))
    for row in rows:
        print("  " + row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
