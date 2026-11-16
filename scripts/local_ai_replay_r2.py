"""Bounded follow-up to the local AI replay audit.

    python scripts/local_ai_replay_r2.py --check
    python scripts/local_ai_replay_r2.py --model ornith-1.0-9b

The default output is release-evidence/local-ai-repair-r2/run.json under the shared Evidence home.
An explicit --out is interpreted as a caller-supplied filesystem path.

This pass registers only historical Product function fixes that can run as written in
the AST sandbox and have black-box cases in their own commit's tests or contract. It
never runs model inference in --check and never executes model output outside the child
sandbox.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import argparse
import ast
import contextlib
import copy
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import local_ai_bench as bench

PROFILE = {"name": "ornith-constrained-json-no-thinking", "model": "ornith-1.0-9b",
           "constrained_json": True, "no_thinking": True}
ROOT = bench.ROOT
REPLAYS: list[dict] = []

# Keep the existing replay sandbox and negative cases intact. These additions are
# confined to this worker; candidates still cannot import, use reflection, or call
# arbitrary methods/functions.
bench.SAFE_NODES.update({ast.ListComp, ast.Dict, ast.BinOp, ast.Add, ast.Sub, ast.Starred})
bench.SAFE_METHODS.update({"get", "append"})
bench.SAFE_BUILTINS.update({"float": float, "int": int, "min": min, "max": max,
                            "abs": abs, "all": all, "any": any, "enumerate": enumerate,
                            "sorted": sorted, "zip": zip})
DATA_FIELDS = {"episode", "start_seconds", "end_seconds"}
PINNED_COMMIT = '993a0fb637311258f7c3491b0db512553f10fc70'
FROZEN_PLACEMENT_SOURCE = '@dataclass(frozen=True)\nclass Placement:\n    """話の中へ置かれた素材の一片。音楽に限らず台詞・効果音も含む。"""\n\n    episode: str\n    component_id: str\n    role: str\n    source_sha256: str\n    gain_db: float\n    start_seconds: float\n    end_seconds: float\n    source_start_seconds: float\n    source_end_seconds: float\n\n    def source_at(self, seconds: float) -> float:\n        """この話の中の時刻を、素材の中の時刻へ直す。"""\n        return self.source_start_seconds + (seconds - self.start_seconds)\n'
FROZEN_OVERLAP_SOURCE = 'def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:\n    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))\n'
# Only these reviewed, pinned pure definitions execute here, never the Product module.
exec(compile(FROZEN_PLACEMENT_SOURCE + FROZEN_OVERLAP_SOURCE, "<frozen-replay-dependencies>", "exec"), globals())
TRUSTED_OVERLAP = _overlap



def _local_target(node):
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_local_target(item) for item in node.elts)
    return isinstance(node, ast.Starred) and _local_target(node.value)


def validate_safe(source: str, name: str, constants: dict) -> ast.Module:
    """Apply the bench allowlist plus one explicit trusted pure global and data fields."""
    if len(source) > 12000:
        raise ValueError("code too large")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
        raise ValueError("expected exactly the requested function")
    fn = tree.body[0]
    if fn.decorator_list or fn.args.defaults or fn.args.kw_defaults:
        raise ValueError("decorators/defaults are outside safe subset")
    fn.returns = None
    for arg in ast.walk(fn.args):
        if isinstance(arg, ast.arg):
            arg.annotation = None
    for node in ast.walk(tree):
        if type(node) not in bench.SAFE_NODES:
            raise ValueError(f"unsupported AST: {type(node).__name__}")
        if isinstance(node, (ast.Name, ast.arg)) and "__" in (node.id if isinstance(node, ast.Name) else node.arg):
            raise ValueError("dunder name")
        if isinstance(node, ast.Attribute) and node.attr not in bench.SAFE_METHODS | DATA_FIELDS | {"S", "I", "M"}:
            raise ValueError("unsupported attribute")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in bench.SAFE_BUILTINS:
                if node.func.id != "_overlap" or constants.get("_overlap") is not TRUSTED_OVERLAP:
                    raise ValueError("unsupported call")
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                raise ValueError("indirect call")
        if isinstance(node, ast.Assign) and any(not _local_target(t) for t in node.targets):
            raise ValueError("only local name assignment or unpacking")
    return tree


def _child_r2(source, name, constants, cases, connection):
    try:
        if os.name == "nt":
            import ctypes
            from local_llm import Job, _ExtendedLimits
            memory_job = Job()
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x100
            limits.ProcessMemoryLimit = 256 * 1024 * 1024
            if not memory_job._k.SetInformationJobObject(memory_job._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError("could not bound replay worker memory")
            memory_job.assign(os.getpid())
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        tree = validate_safe(source, name, constants)
        scope = {"__builtins__": bench.SAFE_BUILTINS, **constants}
        exec(compile(tree, "<bounded-replay-r2>", "exec"), scope)
        results = [scope[name](*copy.deepcopy(args)) for args, _ in cases]
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
    child = ctx.Process(target=_child_r2, args=(source, name, constants, cases, send), daemon=True)
    child.start()
    send.close()
    try:
        if recv.poll(timeout):
            try:
                result = recv.recv()
                if name == "_covering" and "results" in result:
                    result["results"] = [[row.component_id for row in rows] for rows in result["results"]]
                return result
            except EOFError:
                return {"pass": False, "error": "worker exited"}
        return {"pass": False, "error": "timeout"}
    finally:
        if child.is_alive():
            child.terminate()
        child.join(3)
        recv.close()


def load_replays() -> list[dict]:
    """Load one real changed Product function with its source-level helper and vectors."""
    short, path, name = "993a0fb6", "lss/validation/recurrence_ground_truth.py", "_covering"
    sha = PINNED_COMMIT
    old = bench.git("show", f"{sha}^:{path}")
    new = bench.git("show", f"{sha}:{path}")
    before, after = bench.function_source(old, name), bench.function_source(new, name)
    dependency_hashes = {}
    for dependency, frozen in (("Placement", FROZEN_PLACEMENT_SOURCE), ("_overlap", FROZEN_OVERLAP_SOURCE)):
        node = next(n for n in ast.parse(new).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == dependency)
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        pinned = "\n".join(new.splitlines()[start - 1:node.end_lineno]) + "\n"
        digest = hashlib.sha256(frozen.encode()).hexdigest()
        if digest != hashlib.sha256(pinned.encode()).hexdigest():
            raise ValueError("frozen dependency differs from pinned Git: " + dependency)
        dependency_hashes[dependency] = digest
    if bench.function_source(old, "_overlap") != FROZEN_OVERLAP_SOURCE:
        raise ValueError("overlap helper changed across repair")
    # These cases come from the function's own stated interval-overlap contract: every
    # placement with positive overlap is returned, including off-center overlap; an exact
    # touching edge and a different episode are excluded. The historical pre-fix scalar
    # implementation raises on the interval tuple, while the fixed implementation passes.
    first = Placement("ep-a", "p1", "score", "sha-1", 0.0, 10.0, 20.0, 0.0, 10.0)
    second = Placement("ep-a", "p2", "dialogue", "sha-2", 0.0, 0.0, 10.5, 0.0, 10.5)
    other_episode = Placement("ep-b", "p3", "score", "sha-3", 0.0, 10.0, 20.0, 0.0, 10.0)
    cases = [
        ([(first, second, other_episode), "ep-a", (10.25, 11.0)], [first, second]),
        ([(first, second), "ep-a", (20.0, 21.0)], []),
        ([(first, second, other_episode), "ep-b", (19.0, 21.0)], [other_episode]),
    ]
    return [{"id": f"fix-{short}-{name}", "kind": "repair_runtime", "commit": sha,
             "path": path, "name": name, "before": before, "after": after,
             "issue": bench.git("log", "-1", "--format=%s", sha).strip(),
             "constants": {"_overlap": TRUSTED_OVERLAP}, "cases": cases,
             "dependency_sha256": dependency_hashes,
             "fixed_sha256": hashlib.sha256(after.encode()).hexdigest()}]


# The worker's public request/response API defaults to the pinned audited population.
REPLAYS = load_replays()


def revalidate_saved_normalise() -> dict:
    """Recheck the earlier gpt-oss function response after permitting its ListComp syntax."""
    task = next(t for t in bench.load_replays() if t["name"] == "_normalise")
    path = dev_evidence_root(ROOT) / "release-evidence/local-ai-practical-20261026/replay-gptoss.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    row = next(r for r in report["records"] if r.get("id") == task["id"])
    code = json.loads(row["output"])["code"]
    result = evaluate_code(code, task["name"], task["constants"], task["cases"])
    return {"id": task["id"], "model": report["model"], "source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prior_status": row["validation"], "revalidated": result}


def generate_requests(tasks: list[dict] | None = None) -> list[dict]:
    """Return model requests for runtime-eligible tasks; hidden behavioral cases stay private."""
    tasks = REPLAYS if tasks is None else tasks
    return [{"id": task["id"], "profile": PROFILE,
             "prompt": ('Repair this real historical LSS function. Return JSON only: '
                        '{"code":"complete function source"}. TASK CONTRACT: The caller changed from a scalar midpoint to the full report interval. '
                        'Keep the function name and three positional parameters; update the third parameter name, '
                        'annotation and docstring as needed. The third argument is a two-float tuple (start, end), '
                        'not a scalar timestamp. Intervals use half-open geometry: return all placements in input '
                        'order from the requested episode whose time interval has strictly positive duration '
                        'overlap with the report interval, including overlap away from the midpoint. Touching '
                        'endpoints alone have zero overlap. Multiple simultaneous placements all qualify. '
                        'Placement has episode: str, start_seconds: float and end_seconds: float. '
                        'The trusted pure helper '
                        '_overlap(first_interval, second_interval) accepts two (start, end) tuples and returns '
                        'nonnegative overlap duration in seconds; it is available; do not import or access files, '
                        'network, processes, or other helper functions. Hidden regression cases are withheld.\n'
                        + json.dumps({"issue": task["issue"], "path": task["path"],
                                      "source": task["before"]}, ensure_ascii=False))} for task in tasks]


def evidence_cases(task: dict) -> list[dict]:
    """Serialize trusted placement fixtures and expected IDs for an auditable report."""
    rows = []
    for (placements, episode, span), expected in task["cases"]:
        rows.append({"input": {"placements": [{"episode": p.episode, "component_id": p.component_id,
                                                 "start_seconds": p.start_seconds, "end_seconds": p.end_seconds}
                                                for p in placements],
                               "episode": episode, "span": list(span)},
                     "expected_component_ids": [p.component_id for p in expected]})
    return rows


def validate_responses(responses: list[dict], tasks: list[dict] | None = None) -> list[dict]:
    """Validate one JSON function response per eligible task, retaining raw status."""
    tasks = REPLAYS if tasks is None else tasks
    by_id = {row.get("id"): row for row in responses}
    results = []
    for task in tasks:
        row = by_id.get(task["id"], {})
        result = {"id": task["id"], "status": "rejected", "raw_output": row.get("raw_output")}
        try:
            answer = json.loads(row["raw_output"])
            code = answer["code"]
            before = evaluate_code(task["before"], task["name"], task["constants"], task["cases"])
            after = evaluate_code(task["after"], task["name"], task["constants"], task["cases"])
            candidate = evaluate_code(code, task["name"], task["constants"], task["cases"])
            result.update(before=before, fixed=after, candidate=candidate,
                          status="accepted" if not before["pass"] and after["pass"] and candidate["pass"] else "rejected")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def run(model: str, out: str, base_url: str | None = None) -> dict:
    require_evidence_writable(Path(out))
    from local_llm import complete, get_model, serving

    requests = generate_requests(load_replays())
    tasks = REPLAYS
    controls = []
    for task in tasks:
        controls.append({"id": task["id"], "commit": task["commit"], "path": task["path"],
                         "function": task["name"], "fixed_sha256": task["fixed_sha256"],
                         "cases": evidence_cases(task), "dependency_sha256": task["dependency_sha256"],
                         "before": evaluate_code(task["before"], task["name"], task["constants"], task["cases"]),
                         "fixed": evaluate_code(task["after"], task["name"], task["constants"], task["cases"])})
    report = {"schema": 1, "profile": {**PROFILE, "model": model},
              "scope": "bounded historical Product function repair with one additional runtime-eligible function",
              "head": bench.git("rev-parse", "HEAD").strip(), "controls": controls,
              "eligible_count": len(requests), "records": []}
    if not requests:
        report["status"] = "no_eligible_replays"
    else:
        manager = contextlib.nullcontext(base_url) if base_url else serving(model, purpose="bounded historical Product repair replay")
        with manager as url:
            extra = dict(get_model(model).request)
            extra["chat_template_kwargs"] = {**extra.get("chat_template_kwargs", {}), "enable_thinking": False}
            schema = {"type": "object", "properties": {"code": {"type": "string"}},
                      "required": ["code"], "additionalProperties": False}
            for request in requests:
                started = time.monotonic()
                record = {"id": request["id"], "prompt": request["prompt"], "profile": request["profile"]}
                task = next(item for item in tasks if item["id"] == request["id"])
                record["provenance"] = {"commit": task["commit"], "path": task["path"],
                                         "function": task["name"], "fixed_sha256": task["fixed_sha256"]}
                try:
                    reply = complete(url, [{"role": "user", "content": request["prompt"]}],
                                     max_tokens=1800, extra=extra, schema=schema)
                    record.update(raw_output=reply.text, reasoning=reply.reasoning,
                                  usage=reply.usage, timings=reply.timings)
                except Exception as exc:
                    record.update(status="request_failed", error=f"{type(exc).__name__}: {exc}")
                record["wall_seconds"] = time.monotonic() - started
                report["records"].append(record)
                destination = Path(out)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["validation"] = validate_responses(report["records"])
        report["status"] = "completed"
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model", default=PROFILE["model"])
    parser.add_argument("--base-url")
    parser.add_argument("--out", default=str(dev_evidence_root(ROOT) / "release-evidence/local-ai-repair-r2/run.json"))
    args = parser.parse_args()
    if args.check:
        tasks = load_replays()
        controls = []
        for task in tasks:
            before = evaluate_code(task["before"], task["name"], task["constants"], task["cases"])
            after = evaluate_code(task["after"], task["name"], task["constants"], task["cases"])
            controls.append({"id": task["id"], "commit": task["commit"], "buggy": before, "fixed": after})
            if before["pass"] or not after["pass"]:
                print(json.dumps(controls, ensure_ascii=False))
                return 1
        print(json.dumps({"controls": controls, "saved_normalise": revalidate_saved_normalise()}, ensure_ascii=False))
        return 0
    report = run(args.model, args.out, args.base_url)
    print(json.dumps({"status": report["status"], "eligible_count": report["eligible_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
