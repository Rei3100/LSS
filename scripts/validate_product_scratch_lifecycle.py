"""配備r4の実状態のcopyとProduct scratch処理から、隔離したEvidenceを生成する。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root

import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lss.core import scratch
from lss.harness.tsv_io import load_tsv


def fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def git_snapshot(path: Path) -> dict:
    def read(*args):
        return subprocess.run(["git", *args], cwd=path, check=True, capture_output=True).stdout
    return {"path": str(path), "head": read("rev-parse", "HEAD").decode().strip(),
            "status": read("status", "--porcelain").decode("utf-8"),
            "diff_sha256": hashlib.sha256(read("diff", "HEAD", "--binary")).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployed-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--reference-worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    # このrunnerの書き込み先は自分の隔離candidate内だけ。
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    state = args.deployed_root / ".state"
    representative = output / "representative"
    work = representative / ".state" / "work"
    work.mkdir(parents=True)
    before_lanes = [git_snapshot(args.main_root), git_snapshot(args.reference_worktree)]
    live_manifest = args.deployed_root / ".lss-runtime-manifest.json"
    release = json.loads(live_manifest.read_text())
    actual = [("durable_product_state", state / "completed_recipes.tsv", ".state/completed_recipes.tsv"),
              ("durable_product_state", state / "queue.json", ".state/queue.json"),
              ("rollback_recovery", live_manifest, "runtime/.lss-runtime-manifest.json")]
    for cls, folder, pattern, destination in (
        ("durable_product_state", state / "recipes", "*.json", ".state/recipes"),
        ("durable_product_state", state / "bgm-candidates", "run.json", ".state/bgm-candidates"),
        ("reusable_derived_cache", state / "runtime-cache", "*", ".state/runtime-cache"),
        ("development_evidence", dev_state_root(args.main_root) / "validation/bgm-r4-promotion-20261005",
         "final-state.json", ".state/validation"),
    ):
        found = next((p for p in sorted(folder.rglob(pattern)) if p.is_file()), None)
        if found is not None:
            actual.append((cls, found, f"{destination}/{found.name}"))
    rollback = dev_state_root(args.main_root) / "releases/bgm-r4-20261005/rollback/v10/.lss-runtime-manifest.json"
    if rollback.is_file():
        actual.append(("rollback_recovery", rollback, "rollback/.lss-runtime-manifest.json"))
    _, completed = load_tsv(state / "completed_recipes.tsv")
    for row in completed:
        source = Path(row["source_path"])
        assets = json.loads(row.get("outputs") or "[]")
        public = next((Path(item["path"]) for item in assets if Path(item["path"]).is_file()), None)
        if source.is_file() and public is not None and source.stat().st_size + public.stat().st_size < 128 * 2**20:
            actual += [("user_source_media", source, f"media/source{source.suffix}"),
                       ("published_listening_asset", public, f"media/LSS/public{public.suffix}")]
            break
    copies = []
    for cls, origin, relative in actual:
        target = representative / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        origin_before = fingerprint(origin)
        shutil.copy2(origin, target)
        copied = fingerprint(target)
        assert origin_before["sha256"] == copied["sha256"]
        assert origin.stat().st_nlink == 1 and target.stat().st_nlink == 1
        copies.append({"class": cls, "live": origin_before, "copy": copied})
    # 旧名に見える未登録folderも作り、age/nameが削除権限にならないことを確認する。
    unknown = work / "decode-unowned-source"
    unknown.mkdir()
    (unknown / "source.flac").write_bytes(b"unknown ownership must survive")
    os.utime(unknown, (1000, 1000))
    protected_before = [fingerprint(p) for p in sorted(representative.rglob("*")) if p.is_file()]
    code = """
import os, sys
from pathlib import Path
from lss.core.scratch import ScratchDirectory
owner = ScratchDirectory(dir=Path(sys.argv[1]), purpose='input_decode', prefix='decode-')
folder = Path(owner.__enter__())
(folder / 'input.wav').write_bytes(b'\\0' * (8 * 1024 * 1024))
(folder / 'conversion.log').write_bytes(b'interrupted local decode')
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", code, str(work)], cwd=ROOT,
                   check=True, timeout=30)
    before = scratch.reconcile(work, now=time.time() + 90000, dry_run=True)
    first = scratch.reconcile(work, now=time.time() + 90000)
    second = scratch.reconcile(work, now=time.time() + 90000)
    restart_code = """
import json, sys, time
from pathlib import Path
from lss.core import scratch
print(json.dumps(scratch.reconcile(Path(sys.argv[1]), now=time.time()+90000)))
"""
    restarted = subprocess.run([sys.executable, "-c", restart_code, str(work)], cwd=ROOT,
                               check=True, capture_output=True, text=True, timeout=30)
    restart = json.loads(restarted.stdout)
    protected_after = [fingerprint(Path(item["path"])) for item in protected_before]
    live_after = [fingerprint(Path(item["live"]["path"])) for item in copies]
    after_lanes = [git_snapshot(args.main_root), git_snapshot(args.reference_worktree)]
    checks = {"protected_copies_unchanged": protected_before == protected_after,
              "copied_live_files_unchanged": [item["live"] for item in copies] == live_after,
              "unrelated_worktrees_unchanged_during_validation": before_lanes == after_lanes,
              "idempotent_cleanup": second["removed"] == [],
              "fresh_process_restart_coherent": restart == second,
              "only_owned_scratch_reclaimed": len(first["removed"]) == 1 and unknown.exists(),
              "real_copies_no_hardlinks": all(Path(item["copy"]["path"]).stat().st_nlink == 1 for item in copies),
              "source_and_public_asset_represented": {"user_source_media", "published_listening_asset"}
                   <= {item["class"] for item in copies}}
    result = {"schema_version": 1, "outcome": "PASSED" if all(checks.values()) else "FAILED",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "candidate": git_snapshot(ROOT), "deployed_release": release["product_release"],
              "topology": "PARALLEL_SAFE", "live_deployment_performed": False,
              "reference_aware_lane_modified": False, "gpu_inference_performed": False,
              "copies": copies, "protected_before": protected_before, "protected_after": protected_after,
              "inventory_before": before, "cleanup": first, "second_cleanup": second,
              "fresh_process_restart": restart, "checks": checks,
              "unrelated_worktrees_before": before_lanes, "unrelated_worktrees_after": after_lanes,
              "live_work_inventory_read_only": scratch.reconcile(state / "work", dry_run=True),
              "limits": ["source-level candidate only; no packaged/UI/GPU acceptance",
                         "legacy unowned BGM/acquisition scratch is preserved; no retrospective deletion authority",
                         "model/cache/CurrentBest/reference lifecycle policy is unchanged",
                         "unclassified registry residue and incomplete directory identity are retained"]}
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outcome": result["outcome"], "checks": checks,
                      "reclaimed_bytes": first["reclaimed_bytes"], "reclaimed_files": first["reclaimed_files"],
                      "report": str(output / "report.json")}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
