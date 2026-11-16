"""Copy admitted, inspected Git blobs from an exact source checkout revision."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

# Support isolated-mode execution without consulting the caller's import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_guard import (DEFAULT_DENYLIST, GuardError, OID_RE, POLICY_NAME, Policy,
                          Reporter, forbidden_type, git, link_like, load_denylist,
                          read_entry, require_plain_path, resolve_object, tree_entries)


def regenerable(path: str) -> bool:
    """Ignored output the destination checkout itself needs: its root .venv, its fetched bin/ tools, its .models link to the model store and bytecode caches."""
    parts = path.rstrip("/").split("/")
    return parts[0] in (".venv", "bin", ".models") or "__pycache__" in parts


def destination_clean(repo: Path) -> None:
    records = git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                  "--ignored=matching").split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        paths = [record[3:].decode("utf-8")]
        if b"R" in status or b"C" in status:
            if index >= len(records) or not records[index]:
                raise GuardError("Incomplete destination status")
            paths.append(records[index].decode("utf-8"))
            index += 1
        if status == b"!!" and all(regenerable(p) for p in paths):
            continue
        if any(not p.startswith("gates/") for p in paths):
            raise GuardError("Destination has uncommitted changes outside gates")


def ensure_destination(repo: Path, relative: str) -> Path:
    target = repo.joinpath(*relative.split("/"))
    require_plain_path(target)
    if not target.absolute().is_relative_to(repo.absolute()):
        raise GuardError("Destination escapes repository")
    return target


def migrate(source: Path, commit: str, destination: Path, allowlist: Path | None = None,
            denylist: Path | None = None) -> dict:
    require_plain_path(source)
    require_plain_path(destination)
    source, destination = source.absolute(), destination.absolute()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise GuardError("Source and destination must be separate checkouts")
    if not OID_RE.fullmatch(commit):
        raise GuardError("Source revision must be a full commit OID")
    oid, kind = resolve_object(source, commit)
    if oid != commit or kind != "commit" or git(source, "rev-parse", "HEAD").decode().strip() != oid:
        raise GuardError("Source HEAD does not equal pinned commit")
    if git(destination, "rev-parse", "--show-toplevel").decode().strip().replace("\\", "/").casefold() != destination.as_posix().casefold():
        raise GuardError("Destination must be the repository root")
    destination_clean(destination)
    policy_path = allowlist or destination / POLICY_NAME
    require_plain_path(policy_path)
    policy_raw = policy_path.read_bytes()
    policy = Policy.parse(policy_raw)
    literals = load_denylist(denylist or Path(os.environ.get("LSS_PUBLISH_DENYLIST", DEFAULT_DENYLIST)))
    policy_report = Reporter(literals)
    policy_report.content(POLICY_NAME, policy_raw)
    if policy_report.findings:
        raise GuardError("Policy contains prohibited content")
    planned = []
    skipped = Counter()
    seen = set()
    for entry in tree_entries(source, oid):
        path = entry.path
        if path == POLICY_NAME:
            skipped["DESTINATION_POLICY_PRESERVED"] += 1
            continue
        if entry.mode not in ("100644", "100755"):
            skipped["LINK_OR_SUBMODULE"] += 1
            continue
        admitted, rule = policy.admission(path, entry.oid)
        if not admitted:
            skipped[rule] += 1
            continue
        if entry.size > policy.max_blob_bytes:
            skipped["OVERSIZE"] += 1
            continue
        data = read_entry(source, entry)
        # Migration never copies media/models/binary, even with a type approval.
        if forbidden_type(path, data):
            skipped["BINARY_OR_MEDIA"] += 1
            continue
        report = Reporter(literals)
        report.text(path, path, path_value=True)
        report.content(path, data)
        if report.findings:
            for finding in report.findings:
                skipped[finding["rule"]] += 1
            continue
        if path.casefold() in seen or path == "gates/migration-manifest.json":
            raise GuardError("Destination path collision")
        seen.add(path.casefold())
        source_path = source.joinpath(*path.split("/"))
        require_plain_path(source_path)
        if not source_path.is_file() or source_path.read_bytes() != data:
            raise GuardError("Source copied file differs from pinned blob")
        index_records = git(source, "ls-files", "--stage", "-z", "--", ":(literal)" + path).split(b"\0")
        actual = [r.split(b"\t", 1)[0].decode().split() for r in index_records if r]
        if actual != [[entry.mode, entry.oid, "0"]]:
            raise GuardError("Source index differs from pinned blob")
        target = ensure_destination(destination, path)
        if target.exists():
            raise GuardError("Refusing to overwrite a destination file")
        planned.append((entry, data, target))
    # Complete preflight before the first write, including source/destination rechecks.
    if git(source, "rev-parse", "HEAD").decode().strip() != oid:
        raise GuardError("Source HEAD changed during preflight")
    destination_clean(destination)
    manifest_path = ensure_destination(destination, "gates/migration-manifest.json")
    if manifest_path.exists():
        raise GuardError("Refusing to overwrite an existing migration manifest")
    manifest = {"schema_version": 1, "source_commit": oid, "allowlist_sha256": policy.digest,
                "files": [], "skipped_counts": dict(sorted(skipped.items()))}
    for entry, data, target in planned:
        require_plain_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(data)
        if entry.mode == "100755" and os.name != "nt":
            target.chmod(0o755)
        manifest["files"].append({"source_commit": oid, "blob_oid": entry.oid,
                                  "destination_path": entry.path,
                                  "sha256": hashlib.sha256(data).hexdigest()})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Destination repository root")
    parser.add_argument("--allowlist", type=Path)
    parser.add_argument("--denylist", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = migrate(args.source_repo, args.source_commit, args.repo, args.allowlist, args.denylist)
        print(json.dumps({"result": "COPIED", "source_commit": manifest["source_commit"],
                          "files_copied": len(manifest["files"]),
                          "skipped_counts": manifest["skipped_counts"]}, sort_keys=True))
        return 0
    except (GuardError, OSError, UnicodeError, ValueError, KeyError, TypeError):
        # Exception strings may contain private paths or content; never interpolate them.
        print('migrate_to_public: BLOCK; preflight or copy failed; no input values displayed', file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
