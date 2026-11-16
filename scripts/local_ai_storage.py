"""Pin already verified SSD weights as permanent hot development assets.

    python scripts/local_ai_storage.py pin ornith-1.0-9b

This reads only the existing SSD stage and SSD model bytes; it never stages/copies
from the archive HDD. Hardlinks retain the verified bytes without a second large
copy and survive cache eviction. Archive originals stay unchanged. The registry
overlay records both locations and the actual SHA256; public metadata is never
rewritten with machine paths. A missing SSD stage refuses; prepare
assets on SSD before development, rather than silently running from the HDD.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import local_llm
import stage
import storage_tiers
from machine_config import load_config


def pin(name, *, registry=local_llm.REGISTRY, policy=None):
    policy = policy or storage_tiers.load_policy()
    storage_tiers.require_machine_overlay(policy)
    data = load_config(registry)
    if not data.get("_machine_overlay"):
        raise RuntimeError("Model pin refused: no machine model registry overlay was loaded")
    overlay_path = Path(data["_machine_overlay"])
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    if not isinstance(overlay, dict):
        raise ValueError("Model registry overlay must be an object")
    model = local_llm.parse_registry(data)[name]
    if model.path is None:
        raise ValueError(f"{name}: unavailable (no machine path configured)")
    hot = Path(policy["local_ai_hot_models"])
    from dev_hot_paths import require_hot
    fast = policy.get('fast_volumes', policy['hot_volume'])
    require_hot(hot, "permanent hot model root", fast)
    archive = model.path
    if archive.resolve().drive.lower() == hot.resolve().drive.lower():
        return {"model": name, "path": str(archive), "status": "already_hot"}
    chosen = None
    for entry in stage.entries(policy):
        metadata = json.loads((Path(entry["path"]) / stage.META).read_text(encoding="utf-8"))
        if os.path.normcase(metadata["source"]) != os.path.normcase(str(archive)):
            continue
        candidate = Path(entry["path"]) / archive.name
        record = metadata.get("files", {}).get(archive.name, {})
        if (candidate.resolve().drive.lower() == hot.resolve().drive.lower()
                and candidate.stat().st_size == model.size_bytes and record.get("bytes") == model.size_bytes):
            chosen = candidate, record["sha256"]
            break
    if chosen is None:
        raise ValueError(f"{name}: no verified SSD stage; refusing archive read")
    source, expected = chosen
    require_hot(source, "verified SSD stage", fast)
    actual = local_llm._sha256(source)
    if actual != expected or model.sha256 and actual != model.sha256:
        raise ValueError("SSD weights hash mismatch")
    target = hot / name / archive.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if local_llm._sha256(target) != actual:
            raise ValueError("existing hot model differs; refusing overwrite")
    else:
        os.link(source, target)
    overlay.setdefault("models", {}).setdefault(name, {}).update(
        path=target.as_posix(), archive_path=archive.as_posix(), sha256=actual)
    temporary = overlay_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, overlay_path)
    return {"model": name, "path": str(target), "archive": str(archive), "sha256": actual,
            "method": "same-volume hardlink from verified SSD stage", "bytes_copied_from_hdd": 0,
            "status": "pinned"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["pin"])
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    for name in args.models:
        print(json.dumps(pin(name), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
