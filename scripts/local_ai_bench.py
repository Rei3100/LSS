"""Bounded local-model development replay; never changes Product or executes shell output.

    python scripts/local_ai_bench.py --check
    python scripts/local_ai_bench.py --model ornith-1.0-9b --out .state/local-ai-bench/run.json
    python scripts/local_ai_bench.py --base-url http://127.0.0.1:8080/v1 --out run.json

Historical functions and fixed answers come from pinned Git objects, not the working tree.
Three replays have frozen behavioral cases; remaining replays measure conservative AST
reproduction only. That metric cannot establish the >=10 functional-fix adoption gate.
Model code runs only after a deliberately small AST allowlist, in a timed child process.
Music questions concern supplied observations and Product semantics, never listening quality.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from scripts.dev_state import dev_state_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_state_root, require_evidence_writable


ROOT = Path(__file__).resolve().parents[1]
# These are independent historical changes, not synthetic mutations. Large/IO functions
# remain static-only; generated code for them is never executed.
REPLAYS = [
    ("c3049e9e", "scripts/hooks/continue_guard.py", "remaining_without_blocker"),
    ("10e946b6", "scripts/state_status.py", "_normalise"),
    ("371e311d", "scripts/hooks/delegation_guard.py", "is_allowlisted"),
    ("ddfd56d1", "lss/bgm/current_best.py", "better"),
    ("acaaf5d2", "lss/bgm/memory_support.py", "_members"),
    ("f567482a", "lss/bgm/event_tagging.py", "model_input"),
    ("dc6e5ed7", "lss/bgm/cue_scope.py", "product_scope"),
    ("b1ec9a2e", "lss/bgm/scene_job.py", "_ensure_prefix"),
    ("993a0fb6", "lss/validation/recurrence_ground_truth.py", "_covering"),
    ("942da1a0", "lss/separation/scene_balance.py", "read_balance"),
]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8",
                                   stderr=subprocess.PIPE)


def function_source(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"


def normalized(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module)) and node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def regex_constants(source):
    """Read only literal re.compile declarations and string constants from trusted Git."""
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            result[name] = value.value
        elif isinstance(value, ast.Call) and ast.unparse(value.func) == "re.compile":
            pattern = ast.literal_eval(value.args[0])
            flags = 0
            if len(value.args) > 1:
                for part in ast.walk(value.args[1]):
                    if isinstance(part, ast.Attribute) and isinstance(part.value, ast.Name) and part.value.id == "re":
                        if part.attr not in {"I", "M", "S"}:
                            raise ValueError("unsupported trusted regex flag")
                        flags |= getattr(re, part.attr)
            result[name] = re.compile(pattern, flags)
    return result


def behavioral_cases(name):
    if name == "_normalise":
        return [(["Generated now\n- Lanes, last 48 h: none.\n  · lane\nrelease A"], "release A"),
                (["Generated then\n- Lanes, last 48 h:\nrelease B"], "release B"),
                (["durable\nfact"], "durable\nfact")]
    if name == "remaining_without_blocker":
        return [(["The report cited `next step` while explaining the earlier issue."], None),
                (["The report cited 「次の修正」 while explaining the earlier issue."], None),
                (["The next step is to repair the failure."], "The next step is to repair the failure."),
                (["The next step is blocked by an external capability."], None)]
    if name == "is_allowlisted":
        return [(["python scripts/hidden_run.py status"], True),
                (["python scripts/batch.py start queue.json"], True),
                (["python arbitrary.py"], False), (["git status"], True),
                (["python scripts/batch.py delete queue.json"], False)]
    return []


SAFE_BUILTINS = {"len": len, "bool": bool, "str": str, "isinstance": isinstance}
SAFE_METHODS = {"split", "splitlines", "startswith", "strip", "lower", "join", "search", "match", "end", "sub"}
SAFE_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.Expr,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.Call, ast.keyword, ast.Attribute,
    ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.GeneratorExp, ast.comprehension,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub,
}


def validate_safe(source, name, constants):
    if len(source) > 12000:
        raise ValueError("code too large")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
        raise ValueError("expected exactly the requested function")
    fn = tree.body[0]
    if fn.decorator_list or fn.args.defaults or fn.args.kw_defaults:
        raise ValueError("decorators/defaults are outside safe subset")
    # Annotations are not runtime code in this harness.
    fn.returns = None
    for arg in ast.walk(fn.args):
        if isinstance(arg, ast.arg):
            arg.annotation = None
    for node in ast.walk(tree):
        if type(node) not in SAFE_NODES:
            raise ValueError(f"unsupported AST: {type(node).__name__}")
        if isinstance(node, (ast.Name, ast.arg)) and "__" in (node.id if isinstance(node, ast.Name) else node.arg):
            raise ValueError("dunder name")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_METHODS | {"S", "I", "M"}:
            raise ValueError("unsupported attribute")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in SAFE_BUILTINS:
                raise ValueError("unsupported call")
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValueError("indirect call")
        if isinstance(node, ast.Assign) and any(not isinstance(t, ast.Name) for t in node.targets):
            raise ValueError("only local name assignment")
    return tree


def _child(source, name, constants, cases, connection):
    try:
        # Wall timeout also needs a memory bound: a pure string expression can
        # otherwise allocate faster than the parent can terminate the worker.
        if os.name == "nt":
            import ctypes
            from local_llm import Job, _ExtendedLimits
            memory_job = Job()
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100  # kill-on-close, process memory
            limits.ProcessMemoryLimit = 256 * 1024 * 1024
            if not memory_job._k.SetInformationJobObject(memory_job._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError("could not bound replay worker memory")
            memory_job.assign(os.getpid())
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        tree = validate_safe(source, name, constants)
        scope = {"__builtins__": SAFE_BUILTINS, "re": re, **constants}
        exec(compile(tree, "<bounded-replay>", "exec"), scope)
        results = [scope[name](*args) for args, _ in cases]
        connection.send({"pass": results == [expected for _, expected in cases], "results": results})
    except BaseException as exc:
        connection.send({"pass": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def evaluate_code(source, name, constants, cases, timeout=2):
    try:
        validate_safe(source, name, constants)
    except (ValueError, SyntaxError) as exc:
        return {"pass": False, "error": str(exc)}
    ctx = multiprocessing.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    child = ctx.Process(target=_child, args=(source, name, constants, cases, send), daemon=True)
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


def load_replays():
    tasks = []
    for short, path, name in REPLAYS:
        sha = git("rev-parse", short).strip()
        old = git("show", f"{sha}^:{path}")
        new = git("show", f"{sha}:{path}")
        before, after = function_source(old, name), function_source(new, name)
        if normalized(before) == normalized(after):
            raise ValueError(f"{sha}:{name} has no behavioral AST change")
        constants = regex_constants(new) if behavioral_cases(name) else {}
        tasks.append({"id": f"fix-{short}-{name}", "kind": "repair", "commit": sha,
                      "path": path, "name": name, "before": before, "after": after,
                      "issue": git("log", "-1", "--format=%s", sha).strip(),
                      "constants": constants, "cases": behavioral_cases(name)})
    return tasks


def prompt_for(task):
    constants = {k: ({"regex": v.pattern, "flags": v.flags} if isinstance(v, re.Pattern) else v)
                 for k, v in task["constants"].items()}
    return ("Repair this historical LSS function. Return JSON only: {\"code\": \"complete function source\"}. "
            "Keep its signature. Hidden tests are withheld. Do not import or access files/network/processes.\n"
            + json.dumps({"issue": task["issue"], "path": task["path"], "available_constants": constants,
                          "source": task["before"]}, ensure_ascii=False))


REASONING = [
    ("experiment-held", "reasoning", "A candidate recovers 13/18 usages versus deployed 9/18. A previously kept cue disappears. Independent certification is absent. Choose disposition: deploy, held, exact.", "held"),
    ("experiment-unknown", "reasoning", "A separator log contains no measurement of wanted choir preservation. Does this prove choir absent? Choose: absent, unknown, invalid.", "unknown"),
    ("music-choir", "music_semantics", "The wanted target is BGM. The finished stereo mix contains musical choir. Should wanted musical choir be kept or removed? Choose: keep, remove.", "keep"),
    ("music-partial", "music_semantics", "A fingerprint matches 12 seconds of a 60-second cue. The remaining 48 seconds are unverified wanted music. Can the whole cue be deleted as EXACT? Choose: yes, no.", "no"),
    ("music-scene", "music_semantics", "Observed scene: a person sits silently by a window. Observed sound: slow piano. Can a title claim that a family member died? Choose: yes, no.", "no"),
    ("music-currentbest", "music_semantics", "A newer candidate fails wanted preservation. An earlier accepted CurrentBest remains playable. Choose the active revision: newer, accepted.", "accepted"),
]


def run(base_url, out, model, max_tokens=1800, limit=None):
    from local_llm import complete, get_model
    tasks = load_replays()
    records = []
    extra = get_model(model).request if model else {}
    out = Path(out)
    require_evidence_writable(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "model": model, "head": git("rev-parse", "HEAD").strip(),
              "scope": "development-only; no audio quality or deployment claim; static matches are not functional repairs",
              "cloud_tokens_saved": None, "human_minutes_saved": None, "records": records}

    def request(identifier, kind, prompt, validator, baseline, provenance=None):
        started = time.monotonic()
        row = {"id": identifier, "kind": kind, "prompt": prompt, "baseline": baseline,
               "provenance": provenance}
        try:
            reply = complete(base_url, [{"role": "user", "content": prompt}], max_tokens=max_tokens, extra=extra)
            row.update(output=reply.text, reasoning=reply.reasoning, usage=reply.usage, timings=reply.timings)
            row["validation"] = validator(json.loads(reply.text))
        except Exception as exc:
            row["validation"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        row["wall_seconds"] = time.monotonic() - started
        records.append(row)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"id": identifier, "validation": row["validation"], "seconds": row["wall_seconds"]}), flush=True)

    for task in tasks[:limit]:
        cases = task["cases"]
        baseline = evaluate_code(task["before"], task["name"], task["constants"], cases) if cases else {"pass": False, "metric": "AST reproduction"}
        positive = evaluate_code(task["after"], task["name"], task["constants"], cases) if cases else {"pass": True}
        if baseline["pass"] or not positive["pass"]:
            raise RuntimeError(f"invalid control: {task['id']} {baseline} {positive}")

        def validate(answer, task=task):
            code = answer["code"]
            match = normalized(code) == normalized(task["after"])
            if task["cases"]:
                return {**evaluate_code(code, task["name"], task["constants"], task["cases"]), "metric": "frozen behavioral cases", "ast_match": match}
            return {"pass": match, "metric": "AST reproduction only"}
        request(task["id"], "repair_runtime" if cases else "repair_static", prompt_for(task), validate, baseline,
                {"commit": task["commit"], "path": task["path"], "function": task["name"], "fixed_sha256": hashlib.sha256(task["after"].encode()).hexdigest()})
    # Test generation is executable data, not model-generated Python. It must expose the
    # historical failure and also match the fixed implementation on every submitted case.
    task = next(t for t in tasks if t["name"] == "_normalise")
    def validate_tests(answer):
        cases = [(case["args"], case["expected"]) for case in answer["cases"]]
        if not 1 <= len(cases) <= 8 or any(len(a) != 1 or not isinstance(a[0], str) or len(a[0]) > 2000 or not isinstance(e, str) for a, e in cases):
            raise ValueError("invalid bounded test vectors")
        good = evaluate_code(task["after"], task["name"], task["constants"], cases)
        bad = evaluate_code(task["before"], task["name"], task["constants"], cases)
        return {"pass": good["pass"] and not bad["pass"], "fixed": good, "buggy": bad}
    request("test-generation-normalise", "test_generation", "Generate 1-8 black-box JSON test vectors for this bug: " + task["issue"] + ". Return {\"cases\":[{\"args\":[\"input string\"],\"expected\":\"output string\"}]}. Comparable output ignores generation timestamps (Generated ), lane rows (  · ), and lane headers (- Lanes, last ); retain durable facts.\n" + task["before"], validate_tests, {"pass": False, "metric": "no tests"})
    request("test-repair-normalise", "test_repair", 'Repair this stale test vector to the intended contract: ignore generation timestamps, lane rows and lane headers while retaining durable release facts. Return {"cases":[{"args":["input string"],"expected":"output string"}]}. Keep this input unchanged: ' + json.dumps("Generated yesterday\n- Lanes, last 48 h: none.\nrelease stable") + '. The stale expected output was "- Lanes, last 48 h: none.\\nrelease stable".',
            lambda answer: validate_tests(answer) if len(answer["cases"]) == 1 and answer["cases"][0]["args"] == ["Generated yesterday\n- Lanes, last 48 h: none.\nrelease stable"] else {"pass": False, "error": "test input changed"},
            {"pass": False, "metric": "stale expected result"})
    for identifier, kind, question, expected in REASONING:
        request(identifier, kind, question + ' Return JSON only: {"decision":"one choice","reason":"brief explanation"}.',
                lambda a, e=expected: {"pass": a.get("decision") == e, "metric": "frozen semantic choice; explanation unscored"},
                {"pass": False, "metric": "abstention"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model", default="ornith-1.0-9b")
    parser.add_argument("--base-url")
    parser.add_argument("--out", default=str(dev_state_root(ROOT) / "local-ai-bench/run.json"))
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.check:
        for task in load_replays():
            if task["cases"]:
                old = evaluate_code(task["before"], task["name"], task["constants"], task["cases"])
                new = evaluate_code(task["after"], task["name"], task["constants"], task["cases"])
                print(task["id"], json.dumps({"buggy": old, "fixed": new}))
                if old["pass"] or not new["pass"]:
                    return 1
        return 0
    from local_llm import serving
    manager = contextlib.nullcontext(args.base_url) if args.base_url else serving(args.model, purpose="bounded development task replay")
    with manager as url:
        run(url, args.out, args.model, args.max_tokens, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
