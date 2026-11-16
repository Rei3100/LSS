"""Summarize evaluator acceptance from saved local AI repair reports."""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dev_hot_paths import require_hot

COMMON = (
    "fix-c3049e9e-remaining_without_blocker", "fix-10e946b6-_normalise",
    "fix-371e311d-is_allowlisted", "currentbest-ddfd56d1", "fix-993a0fb6-_covering",
)
OPTIONAL = "fix-f567482a-model_input"
KNOWN = set(COMMON) | {OPTIONAL}


def summarize(path: Path) -> dict:
    path = path.resolve()
    require_hot(path, "local AI repair report")
    raw = path.read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: report must be an object")
    repairs = doc.get("repairs")
    attempted = doc.get("attempted_repair_ids")
    if not isinstance(repairs, list) or not isinstance(attempted, list):
        raise ValueError(f"{path}: repairs and attempted_repair_ids must be lists")
    ids = [r.get("id") for r in repairs if isinstance(r, dict)]
    if len(ids) != len(repairs) or any(not isinstance(i, str) for i in ids):
        raise ValueError(f"{path}: malformed repair row or ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: malformed rows or duplicate repair IDs")
    if any(not isinstance(i, str) for i in attempted):
        raise ValueError(f"{path}: malformed attempted repair ID")
    if len(attempted) != len(set(attempted)):
        raise ValueError(f"{path}: duplicate attempted IDs")
    unknown = (set(ids) | set(attempted)) - KNOWN
    if unknown:
        raise ValueError(f"{path}: unknown repair IDs: {sorted(unknown)}")
    if set(ids) != set(attempted):
        raise ValueError(f"{path}: repair rows and attempted IDs disagree")
    refused = doc.get("status") == "unmeasured_admission_refused"
    if refused and repairs:
        raise ValueError(f"{path}: admission-refused report contains repair rows")
    rows = []
    for row in repairs:
        outcomes = []
        for key in ("validation", "candidate"):
            if key not in row:
                continue
            if not isinstance(row[key], dict):
                raise ValueError(f"{path}: malformed {key} result for {row['id']}")
            if "pass" in row[key]:
                outcomes.append(row[key]["pass"])
        if any(type(v) is not bool for v in outcomes) or len(set(outcomes)) > 1:
            raise ValueError(f"{path}: malformed or inconsistent pass flags for {row['id']}")
        if not outcomes:
            raise ValueError(f"{path}: missing evaluator outcome for {row['id']}")
        rows.append({"id": row["id"], "accepted": outcomes[0] if outcomes else None,
                     "error": next((row[k].get("error") for k in ("validation", "candidate") if isinstance(row.get(k), dict) and row[k].get("error")), None)})
    common_rows = [r for r in rows if r["id"] in COMMON]
    optional_rows = [r for r in rows if r["id"] == OPTIONAL]
    return {"source": str(path), "report_sha256": hashlib.sha256(raw).hexdigest(),
            "fence_adaptation_enabled": doc.get("fence_adaptation_enabled"),
            "status": "unmeasured_admission_refused" if refused else "measured",
            "attempts": 0 if refused else len(rows),
            "common5": {"attempts": len(common_rows), "accepted": sum(r["accepted"] is True for r in common_rows), "targets": common_rows},
            "optional6": {"attempts": len(optional_rows), "accepted": sum(r["accepted"] is True for r in optional_rows), "targets": optional_rows},
            "interpretation": "Evaluator acceptance only; not intrinsic correctness or autonomous coding readiness."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = {"runs": [summarize(Path(p)) for p in args.reports]}
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out is None:
        print(encoded, end="")
        return 0
    require_evidence_writable(args.out)
    dest = args.out.resolve()
    allowed = ((ROOT / ".cache/local-ai").resolve(), (dev_evidence_root(ROOT) / "release-evidence").resolve())
    if not any(dest.is_relative_to(root) for root in allowed):
        raise ValueError("--out must be under .cache/local-ai or release-evidence")
    require_hot(dest, "local AI repair summary output")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temp, dest)
        except FileExistsError:
            raise FileExistsError(f"output already exists: {dest}")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
