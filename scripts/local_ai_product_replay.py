"""Supplemental real Product repair replay with frozen CurrentBest behavioral cases.

    python scripts/local_ai_product_replay.py --check
    python scripts/local_ai_product_replay.py --model ornith-1.0-9b --out report.json

The historical function itself runs with trusted pure dependencies and sanitized JSON
arguments. Tests never publish, load audio, modify CurrentBest, or change working files.
This measures one narrow Product function repair, not autonomous repository repair.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root


import local_ai_bench as bench

COMMIT = "ddfd56d1"
PATH = "lss/bgm/current_best.py"
NAME = "better"


def workflow_version(workflow):
    """The unchanged historical pure dependency; never imports the live Product."""
    found = re.fullmatch(r"episode-sfc-routed-cues-v(\d+)", str(workflow or ""))
    return int(found.group(1)) if found else None


def configure_subset():
    bench.SAFE_NODES.update({ast.Try, ast.ExceptHandler, ast.ListComp, ast.Dict,
                             ast.BinOp, ast.Add, ast.Sub})
    bench.SAFE_METHODS.add("get")
    bench.SAFE_BUILTINS.update({"float": float, "workflow_version": workflow_version,
                                "KeyError": KeyError, "TypeError": TypeError, "ValueError": ValueError})


def load_task():
    sha = bench.git("rev-parse", COMMIT).strip()
    old = bench.git("show", f"{sha}^:{PATH}")
    new = bench.git("show", f"{sha}:{PATH}")
    constants = {}
    for node in ast.parse(new).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"IMPROVEMENTS", "EXTENT_IMPROVEMENT", "COVER_TOLERANCE_SECONDS"}:
                constants[name] = ast.literal_eval(node.value)
    old_cue = {"extent": "unresolved", "start_seconds": 10, "end_seconds": 30,
               "pcm_sha256": "old", "workflow": "episode-sfc-routed-cues-v12"}
    new_cue = {"extent": "complete", "start_seconds": 9, "end_seconds": 40,
               "pcm_sha256": "new", "workflow": "episode-sfc-routed-cues-v13"}
    reason = constants["EXTENT_IMPROVEMENT"]
    cases = [
        ([new_cue, old_cue], [reason]),
        ([{**new_cue, "pcm_sha256": "old"}, old_cue], []),
        ([{**new_cue, "extent": "unresolved"}, old_cue], []),
        ([new_cue, {**old_cue, "extent": "complete"}], []),
        ([{**new_cue, "end_seconds": 29}, old_cue], []),
        ([{**new_cue, "end_seconds": "invalid"}, old_cue], []),
        ([{**new_cue, "extent": None}, old_cue], []),
        ([{**new_cue, "improvements": ["dialogue_residual_local_windows"]},
          {**old_cue, "workflow": "episode-sfc-routed-cues-v6", "extent": "complete"}],
         ["dialogue_residual_local_windows"]),
    ]
    return {"commit": sha, "before": bench.function_source(old, NAME),
            "after": bench.function_source(new, NAME), "constants": constants, "cases": cases}


def _worker(source, constants, cases, send):
    configure_subset()
    bench._child(source, NAME, constants, cases, send)


def evaluate(source, task, timeout=3):
    configure_subset()
    try:
        bench.validate_safe(source, NAME, task["constants"])
    except (ValueError, SyntaxError) as exc:
        return {"pass": False, "error": str(exc)}
    ctx = multiprocessing.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_worker, args=(source, task["constants"], task["cases"], send), daemon=True)
    child.start()
    send.close()
    try:
        if recv.poll(timeout):
            try:
                return recv.recv()
            except EOFError:
                return {"pass": False, "error": "worker exited"}
        return {"pass": False, "error": "timeout"}
    finally:
        if child.is_alive():
            child.terminate()
        child.join(3)
        recv.close()


def prompt_for(task):
    # No fixed function or case answers are exposed. The issue is the historical
    # Product intent; unchanged constants and helper signature are legitimate context.
    return ('Repair this actual LSS CurrentBest function. Return JSON only: {"code":"complete function source"}. '
            'An earlier unresolved cue fragment must be recognized as improved when a resolved, longer '
            'cue represents more complete wanted material. Preserve the existing same-audio guard and '
            'versioned improvement behavior. This function identifies improvement; coverage and identity '
            'eligibility are checked separately by its caller. Safely handle missing or malformed duration '
            'metadata. Use the existing extent labels and COVER_TOLERANCE_SECONDS. Keep this signature. '
            'No imports, file access or additional functions. Available pure helper: '
            'workflow_version(workflow) returns int version or None.\n'
            + json.dumps({"path": PATH, "constants": task["constants"], "buggy_source": task["before"]}, ensure_ascii=False))


def run(model, out, *, no_thinking=False, constrained=False):
    from local_llm import complete, get_model, serving
    task = load_task()
    baseline, positive = evaluate(task["before"], task), evaluate(task["after"], task)
    if baseline["pass"] or not positive["pass"]:
        raise RuntimeError(f"invalid controls: {baseline} {positive}")
    prompt = prompt_for(task)
    report = {"schema": 1, "model": model, "scope": "one historical Product function; no deployment/audio quality claim",
              "profile": {"no_thinking": no_thinking, "constrained_json": constrained},
              "commit": task["commit"], "path": PATH, "function": NAME, "case_count": len(task["cases"]),
              "prompt": prompt, "baseline": baseline, "positive_control": positive,
              "fixed_sha256": hashlib.sha256(task["after"].encode()).hexdigest(),
              "cloud_tokens_saved": None, "human_minutes_saved": None}
    started = time.monotonic()
    try:
        with serving(model, purpose="historical Product CurrentBest repair replay") as url:
            extra = dict(get_model(model).request)
            if no_thinking:
                extra["chat_template_kwargs"] = {"enable_thinking": False}
            schema = {"type": "object", "properties": {"code": {"type": "string"}},
                      "required": ["code"], "additionalProperties": False} if constrained else None
            reply = complete(url, [{"role": "user", "content": prompt}], max_tokens=2600,
                             extra=extra, schema=schema)
            report.update(output=reply.text, reasoning=reply.reasoning, usage=reply.usage, timings=reply.timings)
            answer = json.loads(reply.text)
            report["validation"] = evaluate(answer["code"], task)
    except Exception as exc:
        report["validation"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
    report["wall_seconds_including_load"] = time.monotonic() - started
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": model, "validation": report["validation"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ornith-1.0-9b")
    parser.add_argument("--out", default=str(dev_state_root() / "local-ai-bench/product.json"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--constrained", action="store_true")
    args = parser.parse_args()
    if args.check:
        task = load_task()
        old, fixed = evaluate(task["before"], task), evaluate(task["after"], task)
        print(json.dumps({"buggy": old, "fixed": fixed}))
        return 0 if not old["pass"] and fixed["pass"] else 1
    run(args.model, args.out, no_thinking=args.no_thinking, constrained=args.constrained)
    return 0


if __name__ == "__main__":
    sys.exit(main())
