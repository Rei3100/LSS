"""Prepare the pinned audio/text probe weights on SSD at bounded network/disk rate.

    python scripts/acquire_audio_probe.py --mib-per-second 4

Run via hidden_run.py --storage-gb 12. No GPU or source audio is used. Only the
two official shards containing the Thinker are acquired; Talker-only shard 3
is unused. Downloads resume with HTTP Range, validate length/SHA256, and keep
partial bytes on interruption. Public metadata/tokenizer files use the hf CLI.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from dev_hot_paths import require_hot

REPO = "Qwen/Qwen2.5-Omni-3B"
REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
DESTINATION = Path("C:/LSS-Work/local-ai-models/qwen2.5-omni-3b")
EVIDENCE = dev_evidence_root(Path(__file__).resolve().parents[1]) / "release-evidence/local-ai-audio-r1"
SHARDS = [
    ("model-00001-of-00003.safetensors", 4994850880,
     "349972cebff443030e9e96e1e930d7230979c6961cf046fa29fc580e4bcf49a6"),
    ("model-00002-of-00003.safetensors", 4999787448,
     "b29a76dfefb3aa33a5bd0faa19c681d039a44de57223653ea0e933df728c6d8c"),
]
METADATA = ["config.json", "model.safetensors.index.json", "generation_config.json",
            "added_tokens.json", "chat_template.json", "merges.txt", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    require_evidence_writable(path)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire(name, expected_size, expected_sha, *, rate, destination=DESTINATION, progress=None,
            repository=REPO, revision=REVISION):
    target = destination / name
    require_hot(target, "audio probe weight")
    if target.exists():
        if target.stat().st_size != expected_size or sha256(target) != expected_sha:
            raise ValueError("existing complete weight does not match pinned identity")
        return {"file": name, "bytes": expected_size, "sha256": expected_sha, "downloaded_bytes": 0}
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError("partial file exceeds expected size")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
    transferred = 0
    started = time.monotonic()
    if offset < expected_size:
        headers = {"User-Agent": "LSS-local-audio-research/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("server did not honor exact resume range")
            length = response.headers.get("Content-Length")
            if length and int(length) > expected_size - offset:
                raise ValueError("HTTP body exceeds expected remaining bytes")
            with partial.open("ab") as out:
                last_report = 0
                while True:
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    if offset + transferred + len(block) > expected_size:
                        raise ValueError("download exceeds pinned size")
                    out.write(block)
                    transferred += len(block)
                    # Bound the write rate as well as the download read rate.
                    delay = transferred / rate - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
                    if progress and transferred - last_report >= 32 * 1024 * 1024:
                        progress({"file": name, "complete_bytes": offset + transferred,
                                  "expected_bytes": expected_size, "elapsed_s": round(time.monotonic() - started, 2)})
                        last_report = transferred
    if partial.stat().st_size != expected_size or sha256(partial) != expected_sha:
        raise ValueError("download size/SHA256 verification failed; partial retained")
    os.replace(partial, target)
    return {"file": name, "bytes": expected_size, "sha256": expected_sha,
            "downloaded_bytes": transferred, "resumed_bytes": offset,
            "seconds": round(time.monotonic() - started, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mib-per-second", type=float, default=4)
    args = parser.parse_args()
    if not 0 < args.mib_per_second <= 8:
        parser.error("rate must be above zero and at most 8 MiB/s")
    require_hot(DESTINATION, "audio model preparation")
    require_hot(EVIDENCE, "audio probe evidence")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    require_evidence_writable(EVIDENCE)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    command = ["hf", "download", REPO, "--revision", REVISION, "--local-dir", str(DESTINATION),
               "--max-workers", "1", "--format", "json"]
    for name in METADATA:
        command.extend(["--include", name])
    subprocess.run(command, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    receipt = {"repository": REPO, "revision": REVISION, "root": str(DESTINATION),
               "rate_mib_s": args.mib_per_second, "status": "running", "files": [],
               "license": "qwen-research; development feasibility only; no Product adoption"}
    save(EVIDENCE / "acquisition.json", receipt)
    try:
        for name, size, checksum in SHARDS:
            record = acquire(name, size, checksum, rate=args.mib_per_second * 1024 * 1024,
                             progress=lambda row: save(EVIDENCE / "acquisition-progress.json", row))
            receipt["files"].append(record)
            save(EVIDENCE / "acquisition.json", receipt)
        receipt["status"] = "verified"
        return 0
    except Exception as exc:
        receipt["status"] = "failed"
        # Redirect URLs can contain temporary CDN signatures; never print them.
        receipt["error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
        return 1
    finally:
        save(EVIDENCE / "acquisition.json", receipt)


if __name__ == "__main__":
    sys.exit(main())
