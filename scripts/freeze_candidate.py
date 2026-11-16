"""Freeze a built candidate package: per-file SHA-256 manifest, tree hash, read-only files.

usage: python scripts/freeze_candidate.py <commit> <branch> <dist/LSS dir> <out dir> [--exclude BRANCH=REASON ...]
                                         [--exclude-file reasons.json]
Writes <out dir>/package-manifest.json. Never modifies file contents.

A candidate is the latest integrated Product: every local branch committed in the last RECENT_DAYS with Product
commits (lss/, LSS.spec, build.ps1) that the candidate lacks (patch-equivalent ones excluded) refuses the freeze unless
named with `--exclude BRANCH=REASON` (held, rejected, WIP with what is missing). The list and reasons go into the
manifest. Release 76b88f4 was frozen from one line while four finished Product lanes stayed outside it.
"""
from __future__ import annotations

try:
    from scripts.dev_state import require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import require_evidence_writable


import hashlib
import json
import argparse
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

RECENT_DAYS = 21
PRODUCT_PATHS = ("lss", "LSS.spec", "build.ps1")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def integration_gaps(commit: str, *, now: float | None = None) -> dict[str, list[str]]:
    """Recent local branches with Product commits the candidate lacks (by patch identity, `git cherry`)."""
    cutoff = (time.time() if now is None else now) - RECENT_DAYS * 86400
    gaps = {}
    for line in git("for-each-ref", "--format=%(committerdate:unix) %(refname:short)", "refs/heads").splitlines():
        stamp, _, name = line.partition(" ")
        if int(stamp) < cutoff:
            continue
        product = set(git("rev-list", f"{commit}..{name}", "--", *PRODUCT_PATHS).split())
        if not product:
            continue
        missing = [row[2:] for row in git("cherry", commit, name).splitlines() if row.startswith("+ ")]
        lacking = [c for c in missing if c in product]
        if lacking:
            gaps[name] = lacking
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commit")
    parser.add_argument("branch")
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--exclude", action="append", default=[], metavar="BRANCH=REASON")
    parser.add_argument("--exclude-file", type=Path, help="JSON {branch: reason}, e.g. the release Evidence's own list")
    args = parser.parse_args()
    commit, branch, root, out = args.commit, args.branch, args.root, args.out
    require_evidence_writable(out)
    require_evidence_writable(root)
    excluded = json.loads(args.exclude_file.read_text(encoding="utf-8")) if args.exclude_file else {}
    excluded.update(item.split("=", 1) for item in args.exclude)
    gaps = integration_gaps(commit)
    unexplained = {name: shas for name, shas in gaps.items() if not excluded.get(name, "").strip()}
    if unexplained:
        for name, shas in sorted(unexplained.items()):
            print(f"not integrated: {name} ({len(shas)} Product commits, e.g. {shas[0][:8]})", file=sys.stderr)
        sys.exit("refusing to freeze: integrate these lanes or pass --exclude BRANCH=REASON for each")
    rows = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows[p.relative_to(root).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    tree = hashlib.sha256("".join(f"{k}\t{v['sha256']}\n" for k, v in sorted(rows.items())).encode()).hexdigest()
    manifest = {"type": "lss_frozen_candidate_package", "candidate": f"lss-integrated-rc-{commit[:7]}",
                "source_commit": commit, "branch": branch, "package_dir": str(root),
                "exe_sha256": rows["LSS.exe"]["sha256"], "files": len(rows),
                "bytes": sum(v["bytes"] for v in rows.values()), "tree_sha256": tree,
                "integration": {"recent_days": RECENT_DAYS, "excluded": {name: {"reason": excluded[name], "commits": shas}
                                                                          for name, shas in gaps.items()}},
                "files_sha256": rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "package-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)
    print(manifest["exe_sha256"], manifest["files"], manifest["bytes"], tree)


if __name__ == "__main__":
    main()
