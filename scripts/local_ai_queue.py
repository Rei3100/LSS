"""Resumable local drafting queue; no generated commands or code are executed.

    python scripts/local_ai_queue.py requests.jsonl --model ornith-1.0-9b --out .cache/local-ai/my-job

Each JSONL row has an id and messages, and optional schema, max_tokens (default 1024).
One lease/model load covers all uncached requests. Each atomic receipt includes the exact
request, response, model/runtime SHA256 and elapsed time. Repeating the queue skips successful
identical requests before starting a server. Errors stay errors and make the process exit 1;
a later run retries them. Responses are unverified drafts, never Product decisions or patches.
The output directory is restricted to this checkout's .cache/local-ai or the shared release-evidence.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import argparse
import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import local_llm

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_requests(path):
    requests, ids = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - {"id", "messages", "schema", "max_tokens", "python_def"}:
            raise ValueError(f"line {number}: invalid request fields")
        ident = row.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError(f"line {number}: nonempty unique id required")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"line {number}: messages required")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)):
                raise ValueError(f"line {number}: text messages only")
        tokens = row.get("max_tokens", 1024)
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= 8192:
            raise ValueError(f"line {number}: max_tokens must be 1..8192")
        if row.get("schema") is not None and not isinstance(row["schema"], dict):
            raise ValueError(f"line {number}: schema must be an object")
        if "python_def" in row and (not isinstance(row["python_def"], str) or not row["python_def"].isidentifier()):
            raise ValueError(f"line {number}: python_def must name one function")
        requests.append({**row, "max_tokens": tokens})
        ids.add(ident)
    if not requests:
        raise ValueError("queue is empty")
    return requests


def output_directory(path):
    require_evidence_writable(path)
    target = Path(path).resolve()
    from dev_hot_paths import require_hot
    require_hot(target, "local draft output")
    cache = (ROOT / ".cache/local-ai").resolve()
    allowed = [(dev_evidence_root(ROOT) / "release-evidence").resolve()]
    if cache.is_relative_to(ROOT.resolve()):
        allowed.append(cache)
    if not any(target.is_relative_to(base) and target != base for base in allowed):
        raise ValueError("output must be a subdirectory of the checkout cache or shared release-evidence")
    target.mkdir(parents=True, exist_ok=True)
    return target


def atomic_json(path, value):
    require_evidence_writable(path)
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def identity(model):
    """Hash real bytes, not a mutable model alias or a same-size file claim."""
    from dev_hot_paths import require_hot
    require_hot(model.path, "queue model weights")
    exe = local_llm.llama_dir() / "llama-server.exe"
    require_hot(exe, "queue inference runtime")
    weights = Path(local_llm.stage_path(model.path, purpose="local-ai-queue-identity"))
    digest_weights = local_llm._sha256(weights)
    if model.sha256 and digest_weights.lower() != model.sha256.lower():
        raise ValueError(f"model SHA256 mismatch: {model.name}")
    return {"model": model.name, "model_sha256": digest_weights,
            "runtime_sha256": local_llm._sha256(exe), "runtime_build": local_llm.BUILD,
            "runtime_dlls": {path.name: local_llm._sha256(path)
                             for path in sorted(exe.parent.glob("*.dll"))},
            "ctx": model.ctx, "args": list(model.args), "request_defaults": model.request,
            "client_sha256": local_llm._sha256(Path(local_llm.__file__)),
            "queue_sha256": local_llm._sha256(Path(__file__))}


def check_python_def(request, receipt):
    """A drop-in function request is a draft only when its JSON ``code`` parses to exactly that one ``def``.

    gpt-oss-20b answered such a request with a whole module (docstring, imports, a mid-file
    ``from __future__``): the logic was right and the insertion was a SyntaxError
    (release-evidence/strata-deploy-20261031/LOCAL-MODEL-TRIAL.md).
    """
    name = request.get("python_def")
    if name is None:
        return
    code = json.loads(receipt["text"]).get("code")
    if not isinstance(code, str):
        raise ValueError("python_def requires a JSON string field 'code'")
    body = ast.parse(code).body
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != name:
        raise ValueError(f"code must be exactly one 'def {name}', got {[type(node).__name__ for node in body]}")


def reusable(cached, fingerprint, request, key):
    try:
        return (cached["status"] == "draft" and cached["identity"] == fingerprint
                and cached["request"] == request and cached["key"] == key
                and isinstance(cached["text"], str) and bool(cached["text"])
                and cached["response_digest"] == digest(cached["response"])
                and cached["response"]["choices"][0]["finish_reason"] == "stop"
                and cached["text"] == cached["response"]["choices"][0]["message"]["content"])
    except (KeyError, TypeError, IndexError):
        return False


def run_queue(requests, model, out, *, serving=local_llm.serving, complete=local_llm.complete,
              identify=identity, admit=local_llm.admit, cpu=False, validate_receipt=None, preserve_attempts=False):
    require_evidence_writable(out)
    if not cpu:
        from dev_quiet_mode import require_gpu_allowed
        require_gpu_allowed()  # Defer before admission and gigabyte-scale identity reads.
    start = time.monotonic()
    # A fresh job must not copy/hash gigabytes only to discover that inference
    # is inadmissible. Existing drafts can still be read without loading a model.
    if not any(len(path.stem) == 64 for path in out.glob("*.json")):
        try:
            admit(model, cpu=True) if cpu else admit(model)
        except local_llm.AdmissionRefused as exc:
            summary = {"identity": {"model": model.name}, "entries": [
                {"id": r["id"], "status": "refused", "error": str(exc)} for r in requests],
                "cached": 0, "seconds": round(time.monotonic() - start, 3), "status": "failed"}
            atomic_json(out / "summary.json", summary)
            return summary
    fingerprint = {**identify(model), "device": "cpu" if cpu else "gpu"}
    entries, pending = [], []
    for request in requests:
        key = digest({"identity": fingerprint, "request": request})
        path = out / f"{key}.json"
        cached = None
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        # Content and identity are rechecked; interrupted/error/truncated replies never hit.
        valid_cache = reusable(cached, fingerprint, request, key)
        if valid_cache and validate_receipt:
            try:
                validate_receipt(request, cached)
            except Exception:
                valid_cache = False
        if valid_cache:
            entries.append({"id": request["id"], "status": "cached_draft", "receipt": str(path)})
        else:
            pending.append((request, key, path))
    if pending:
        try:
            with serving(model.name, purpose="resumable local drafting queue", **({"cpu": True} if cpu else {})) as url:
                for request, key, path in pending:
                    before = time.monotonic()
                    receipt = {"key": key, "identity": fingerprint, "request": request,
                               "status": "error", "unverified": True}
                    try:
                        reply = complete(url, request["messages"], request.get("schema"),
                                         max_tokens=request["max_tokens"], extra=model.request)
                        receipt.update(text=reply.text, response=reply.raw,
                                       response_digest=digest(reply.raw), usage=reply.usage)
                        finish = reply.raw["choices"][0].get("finish_reason")
                        receipt["status"] = "draft"
                        if not reusable(receipt, fingerprint, request, key):
                            receipt["status"] = "incomplete"
                    except Exception as exc:
                        receipt["error"] = f"{type(exc).__name__}: {exc}"
                    if receipt["status"] == "draft":
                        try:
                            if validate_receipt:
                                validate_receipt(request, receipt)
                            check_python_def(request, receipt)
                        except Exception as exc:
                            receipt.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
                    receipt["seconds"] = round(time.monotonic() - before, 3)
                    if preserve_attempts and path.exists():
                        attempts = out / "attempts"
                        attempts.mkdir(exist_ok=True)
                        previous = attempts / f"{key}-{uuid.uuid4().hex}.json"
                        os.replace(path, previous)
                    atomic_json(path, receipt)
                    entries.append({"id": request["id"], "status": receipt["status"], "receipt": str(path)})
        except Exception as exc:
            completed = {entry["id"] for entry in entries}
            for request, _, path in pending:
                if request["id"] not in completed:
                    entries.append({"id": request["id"], "status": "error",
                                    "error": f"{type(exc).__name__}: {exc}", "receipt": str(path)})
    summary = {"identity": fingerprint, "entries": entries, "cached": sum(
        entry["status"] == "cached_draft" for entry in entries), "seconds": round(time.monotonic() - start, 3),
        "status": "drafts_ready" if all(e["status"] in {"draft", "cached_draft"} for e in entries) else "failed"}
    atomic_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    from dev_quiet_mode import GPUDeferred
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", choices=["default", "test-vectors"], default="default")
    parser.add_argument("--cpu", action="store_true", help="run on CPU without a GPU lease")
    args = parser.parse_args(argv)
    try:
        requests = read_requests(args.requests)
        model = local_llm.get_model(args.model)
        if model.kind != "chat":
            raise ValueError("draft queue requires a chat model")
        if args.profile == "test-vectors":
            if not args.model.startswith("ornith-") or any(not r.get("schema") for r in requests):
                raise ValueError("test-vectors profile is measured only on Ornith with constrained JSON schemas")
            model = replace(model, request={**model.request, "chat_template_kwargs": {"enable_thinking": False}})
        result = run_queue(requests, model, output_directory(args.out), cpu=args.cpu)
    except (OSError, ValueError, GPUDeferred) as exc:
        print(str(exc), file=sys.stderr)
        return 3 if isinstance(exc, GPUDeferred) else 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "drafts_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
