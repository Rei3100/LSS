"""Generate the mechanical "Live status" block of docs/current/STATE.md from git and the release record.

    python scripts/state_status.py            # print the block
    python scripts/state_status.py --write    # rewrite the block in docs/current/STATE.md in place
    python scripts/state_status.py --check    # exit 1 when the block is stale (read-only)

The block sits between <!-- generated:state-status begin --> and <!-- generated:state-status end -->.
Facts it carries: deployed release (.state/releases/current-product.json, EXE hash verified against the file),
whether the "Installed Product" block (scripts/state_installed.py) is current, undeployed release-candidate
branches with their verdict markers, and the lanes of the last 48 h (scripts/lanes.py).
--check ignores the generation timestamp, the lanes header and the lane rows (they move with every commit of any agent); it
fails on the durable facts (release, EXE verification, installed block, candidates). Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root


try:  # run as a script (scripts/ on sys.path) or imported as scripts.state_status
    import lanes as _lanes
    import state_installed as _installed
except ImportError:  # pragma: no cover - import style depends on the caller
    from scripts import lanes as _lanes
    from scripts import state_installed as _installed

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- generated:state-status begin -->", "<!-- generated:state-status end -->"
LANE_PREFIX = "  · "
LANE_HEADER_PREFIX = "- Lanes, last "
STAMP_PREFIX = "Generated "
LANE_HOURS, LANE_LIMIT, CANDIDATE_LIMIT = 48.0, 8, 3
MARKER_PATTERNS = {
    "suite": ("FULL-SUITE-DONE*", "FROZEN-*/FULL-SUITE-DONE*"),
    "verdict": ("VERDICT*", "FROZEN-*/VERDICT*", "FROZEN-*/inbox-from-verifier/VERDICT*"),
}


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=30,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _is_ancestor(root: Path, sha: str, of: str) -> bool:
    if not sha or not of:
        return False
    try:
        return subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", sha, of], capture_output=True,
                              timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def release_file(root: Path) -> Path:
    """Use configured development state, retaining legacy checkout discovery."""
    here = dev_state_root(root) / "releases" / "current-product.json"
    if "LSS_DEV_STATE" in os.environ or here.is_file():
        return here
    common = _git(root, "rev-parse", "--git-common-dir").strip()
    if common:
        alt = dev_state_root((root / common).resolve().parent) / "releases" / "current-product.json"
        if alt.is_file():
            return alt
    return here


def exe_status(rec: dict) -> str:
    """verified / MISMATCH / unreadable: the recorded EXE hash against the file on disk."""
    path, want = str(rec.get("deployed_executable") or ""), str(rec.get("exe_sha256") or "").lower()
    if not path or not want:
        return "not recorded"
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return "**file unreadable**"
    got = digest.hexdigest()
    return "**verified** against the file" if got == want else f"**MISMATCH** (file `{got[:12]}…`)"


def installed_block_status(rec: dict, state_text: str) -> str:
    a, b = state_text.find(_installed.BEGIN), state_text.find(_installed.END)
    if a < 0 or b < a:
        return "no installed block"
    try:
        current = _installed.render(rec) == state_text[a:b + len(_installed.END)]
    except Exception:  # noqa: BLE001 - a git failure must not break generation
        return "unknown"
    return "current" if current else "**STALE** (`python scripts/state_installed.py`)"


def _markers(worktree: str | None) -> str:
    if not worktree:
        return "no worktree"
    wt = Path(worktree)
    if not wt.is_dir():
        return "worktree missing"
    found = {}
    for kind, patterns in MARKER_PATTERNS.items():
        found[kind] = any(next(wt.glob(p), None) is not None for p in patterns)
    return f"full-suite {'yes' if found['suite'] else 'no'}, verdict {'yes' if found['verdict'] else 'no'}"


def candidates(root: Path, deployed_sha: str) -> list[str]:
    checked_out, current = {}, None
    for line in _git(root, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            current = line[9:]
        elif line.startswith("branch refs/heads/"):
            checked_out[line[18:]] = current
    rows = []
    for line in _git(root, "for-each-ref", "--sort=-committerdate",
                     "--format=%(refname:short)\t%(objectname:short=7)\t%(objectname)",
                     "refs/heads/claude/release-candidate-*").splitlines():
        branch, short, full = (line.split("\t") + ["", ""])[:3]
        if deployed_sha and _is_ancestor(root, full, deployed_sha):
            continue  # already part of the deployed code
        rows.append(f"`{branch}` `{short}`: {_markers(checked_out.get(branch))}")
        if len(rows) >= CANDIDATE_LIMIT:
            break
    return rows


def _age(hours: float) -> str:
    return f"{hours:.0f}h" if hours >= 1.5 else f"{hours * 60:.0f}m"


def lane_rows(root: Path, now: float) -> list[str]:
    out = []
    for r in _lanes.lane_records(LANE_HOURS, LANE_LIMIT, cwd=root, now=now):
        out.append(f"{LANE_PREFIX}{r['branch']} `{r['sha']}` {_age(r['age_h'])} — {r['subject'][:56]}")
    return out


def render(root: Path = ROOT, now: float | None = None, state_text: str = "") -> str:
    now = time.time() if now is None else now
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    lines = [BEGIN,
             "<!-- generated by scripts/state_status.py (--write); do not edit by hand -->",
             f"{STAMP_PREFIX}{stamp}."]
    rf = release_file(root)
    rec = json.loads(rf.read_text(encoding="utf-8")) if rf.is_file() else None
    deployed_sha = ""
    if rec:
        deployed_sha = str(rec.get("candidate_code_commit", ""))
        rb = rec.get("rollback") or {}
        lines.append(f"- Deployed `{rec.get('release_id')}` code `{deployed_sha[:7]}`, {str(rec.get('completed_at', ''))[:10]}; "
                     f"rollback target `{rb.get('target_release_id')}`.")
        lines.append(f"- EXE `{rec.get('deployed_executable')}` `{str(rec.get('exe_sha256', ''))[:12]}…`: {exe_status(rec)}; "
                     f"Installed Product block: {installed_block_status(rec, state_text)}.")
    else:
        lines.append("- Deployed release: no `.state/releases/current-product.json` (identity unrecorded).")
    cands = candidates(root, deployed_sha)
    lines.append("- Undeployed release candidates:" + (" none." if not cands else ""))
    lines += [f"  - {c}" for c in cands]
    lanes = lane_rows(root, now)
    lines.append(f"{LANE_HEADER_PREFIX}{LANE_HOURS:g} h (live: `python scripts/lanes.py`):" + (" none." if not lanes else ""))
    lines += lanes
    lines.append(END)
    return "\n".join(lines)


def _normalise(block: str) -> str:
    """The comparable part of a block: without the timestamp line, the lanes header and the lane rows."""
    volatile = (STAMP_PREFIX, LANE_PREFIX, LANE_HEADER_PREFIX)
    return "\n".join(l for l in block.split("\n") if not l.startswith(volatile))


def process(root: Path, check: bool, write: bool, now: float | None = None) -> tuple[int, str]:
    target = root / "docs" / "current" / "STATE.md"
    raw = target.read_bytes()
    text = raw.decode("utf-8")
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    a, b = text.find(BEGIN), text.find(END)
    if a < 0 or b < a:
        return 1, "STATE.md has no generated:state-status block markers"
    old_block = text[a:b + len(END)]
    # The installed-block comparison needs the text outside our own block only.
    new_block = render(root, now, text)
    if check:
        if _normalise(new_block) == _normalise(old_block):
            return 0, ""
        return 1, "STATE.md generated status block is stale: run python scripts/state_status.py --write"
    if write:
        new = text[:a] + new_block + text[b + len(END):]
        if new != text:
            target.write_bytes((new.replace("\n", "\r\n") if crlf else new).encode("utf-8"))
            return 0, "STATE.md generated status block updated"
        return 0, "STATE.md generated status block unchanged"
    return 0, new_block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="rewrite the block in docs/current/STATE.md")
    parser.add_argument("--check", action="store_true", help="exit 1 when the block is stale (read-only)")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root (default: this checkout)")
    args = parser.parse_args(argv)
    if not (args.write or args.check):
        print(render(args.root.resolve(), state_text=(args.root / "docs" / "current" / "STATE.md").read_text(encoding="utf-8")
                     if (args.root / "docs" / "current" / "STATE.md").is_file() else ""))
        return 0
    code, message = process(args.root.resolve(), args.check, args.write)
    if message:
        print(message)  # stdout only: PowerShell 5.1 under ErrorActionPreference=Stop treats native stderr as an error
    return code


if __name__ == "__main__":
    raise SystemExit(main())
