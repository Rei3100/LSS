"""Small, bounded CPU-only local language-model trial; saves raw drafts and deterministic scores.

    python scripts/local_ai_cpu_trial.py --model PATH --model-sha256 HEX --out EVIDENCE_HOME/release-evidence/cpu-trial/run

The model and runtime must already exist on the configured hot SSD. This tool never downloads,
executes, or applies generated code. Smoke scores are controls, not adoption evidence. Optional historical
responses are raw and unscored; use the existing repair validator separately.
"""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable


import argparse
import base64
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

import local_llm
from machine_config import load_config
from local_ai_output import decode_json

ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    {"id":"currentbest_safety", "prompt":"Return JSON only: {\"classification\":...}. Facts: cue A is HELD; cue B is UNKNOWN; cue C is latest but not accepted. Is any cue authorized as CurrentBest? Choose only YES or NO.", "expected":{"classification":"NO"}},
    {"id":"truncated_identifier", "prompt":"Extract only identifiers explicitly present. Text: accepted cue=Q-17; revision=R-4; the following source record was truncated. Return JSON {\"cue\":string,\"revision\":string,\"missing_source\":true}. Do not infer the missing source.", "expected":{"cue":"Q-17","revision":"R-4","missing_source":True}},
    {"id":"test_vector", "prompt":"Return JSON only with key cases, an array of objects {input,expected}. Function contract: normalize whitespace by splitting on any whitespace and joining with one ASCII space; preserve case. Include these two vectors: input ' a\\tb ' -> 'a b'; input 'X  Y' -> 'X Y'.", "expected":{"cases":[{"input":" a\tb ","expected":"a b"},{"input":"X  Y","expected":"X Y"}]}},
    {"id":"unicode_transfer", "prompt":"Copy the values exactly from source to JSON, with keys title and note. Source title: 『雨のあと』; note: café — 東京。 Do not translate or normalize.", "expected":{"title":"『雨のあと』","note":"café — 東京。"}},
    {"id":"simple_function_ast", "prompt":"Write only Python code defining normalize(text) that returns words joined by one ASCII space after splitting on whitespace. Do not add imports or other definitions.", "expected":None},
]


def server_command(model: Path, port: int, projector: Path | None = None) -> list[str]:
    command = local_llm._launcher() + ["-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--no-webui", "-np", "1", "-c", "4096", "-t", "2", "-tb", "2", "-ngl", "0", "--device", "none", "--fit", "off", "--jinja"]
    if projector is not None:
        command += ["--mmproj", str(projector), "--no-mmproj-offload"]
    return command


def request_body(prompt, max_tokens, thinking, constrained=False):
    body = {"model": "local", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if constrained:
        body['response_format'] = {'type': 'json_object'}
    return body


def admit_cpu(model, extra_bytes=0):
    commit, ram = local_llm.gpu_run.memory_headroom()  # CPU OS counters only
    needed = (model.stat().st_size + extra_bytes) / 1e9 + 4
    if ram < needed or commit < local_llm.gpu_run.MIN_COMMIT_GB:
        raise local_llm.AdmissionRefused(f"CPU model needs {needed:.2f} GB RAM and normal commit headroom")
    return {"available_ram_gb": ram, "commit_headroom_gb": commit, "required_ram_gb": needed}


def load_audio_rows(manifest: Path) -> list[dict]:
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("audio manifest must contain a rows array")
    prepared = [r for r in rows if isinstance(r, dict) and r.get("status") == "prepared"]
    if not 2 <= len(prepared) <= 8:
        raise ValueError("audio manifest must contain 2..8 prepared caption candidates")
    for r in prepared:
        if not isinstance(r.get("fixture_path"), str) or not isinstance(r.get("fixture_sha256"), str):
            raise ValueError("prepared audio row is missing fixture path or SHA256")
        if len(r["fixture_sha256"]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in r["fixture_sha256"]):
            raise ValueError("prepared fixture SHA256 is malformed")
        if not isinstance(r.get("caption"), str) or not r["caption"]:
            raise ValueError("prepared audio row is missing caption")
    return prepared


def audio_prompt(rows: list[dict], index: int) -> tuple[str, str]:
    count = len(rows)
    offset = (2 * index) % count
    labels = list('ABCDEFGH')[:count]
    choices = []
    for j in range(count):
        source = rows[(j + offset) % count]
        choices.append({"caption_id": labels[j], "caption": source["caption"]})
    prompt = ("Choose which caption best matches the audio. Include UNKNOWN if none fits. "
              "Return exactly one JSON object with one string field caption_id, whose value is one of the listed IDs or UNKNOWN.\n"
              + json.dumps({"candidates": choices + [{"caption_id":"UNKNOWN"}]}, ensure_ascii=False))
    return prompt, labels[(index - offset) % count]


def validate_audio_fixtures(rows: list[dict], require_hot=None, hot=None) -> list[Path]:
    paths = []
    for row in rows:
        fixture = Path(row["fixture_path"]).resolve()
        if require_hot is not None:
            require_hot(fixture, "CPU trial audio fixture", hot)
        if not fixture.is_file() or fixture.stat().st_size > 20_000_000:
            raise ValueError("audio fixture missing or exceeds 20 MB")
        if sha256_file(fixture).lower() != row["fixture_sha256"].lower():
            raise ValueError(f"audio fixture SHA256 mismatch: {fixture}")
        paths.append(fixture)
    return paths


def _wav_request_body(wav_bytes: bytes, prompt: str, max_tokens: int, thinking: bool) -> dict:
    return {"model":"local", "messages":[{"role":"user", "content":[
        {"type":"input_audio", "input_audio":{"data":base64.b64encode(wav_bytes).decode("ascii"), "format":"wav"}},
        {"type":"text", "text":prompt}]}], "max_tokens":max_tokens, "temperature":0,
        "stream":False, "chat_template_kwargs":{"enable_thinking":thinking}}


def _score_caption(response: str, allowed: set[str]) -> str | None:
    try:
        parsed, _ = decode_json(response)
    except ValueError:
        return None
    if isinstance(parsed, dict) and set(parsed) == {"caption_id"} and isinstance(parsed["caption_id"], str) and parsed["caption_id"] in allowed:
        return parsed["caption_id"]
    return None


def child_environment(parent=None) -> dict:
    env = dict(os.environ if parent is None else parent)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_requests(path: Path) -> list[dict]:
    requests = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or not isinstance(row.get("prompt"), str) or not row["prompt"]):
            raise ValueError(f"request line {number}: expected JSON object with nonempty string id and prompt")
        if "max_tokens" in row and (type(row["max_tokens"]) is not int or not 1 <= row["max_tokens"] <= 4096):
            raise ValueError(f"request line {number}: max_tokens must be an integer in 1..4096")
        requests.append(row)
    ids = [row["id"] for row in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("replay request ids must be unique")
    return requests


def _json_response(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def score(task: dict, text: str) -> dict:
    if task["id"] == "simple_function_ast":
        try:
            tree = ast.parse(text)
            fn = tree.body[0]
            valid = (len(tree.body) == 1 and isinstance(fn, ast.FunctionDef) and fn.name == "normalize"
                     and len(fn.args.args) == 1 and fn.args.args[0].arg == "text" and len(fn.body) == 1
                     and ast.dump(fn.body[0], include_attributes=False) == ast.dump(ast.parse(
                         "def normalize(text): return ' '.join(text.split())").body[0].body[0], include_attributes=False))
            return {"passed":valid,"parsed":"AST checked; code not executed", "unsupported":None if valid else "AST does not match the limited required function"}
        except (SyntaxError, IndexError, TypeError):
            return {"passed":False,"parsed":None,"unsupported":"response is not the required parseable Python function"}
    parsed = _json_response(text)
    # Coding task is scored solely by parsing a supplied function-shaped snippet; never execute it.
    if task["id"] == "test_vector":
        valid = (isinstance(parsed, dict) and parsed == task["expected"])
        return {"passed": valid, "parsed": parsed, "unsupported": None if valid else "test-vector JSON differs from fixed expected vectors"}
    valid = isinstance(parsed, dict) and parsed == task["expected"]
    return {"passed": valid, "parsed": parsed, "unsupported": None if valid else "expected structured fields were missing or different"}


def run(model: Path, expected_sha: str, out: Path, health_timeout: float = 120.0,
        requests_path: Path | None = None, max_tokens=512, thinking=False, constrained=False,
        projector: Path | None = None, projector_sha256: str | None = None,
        audio_manifest: Path | None = None, omit_audio=False) -> dict:
    from dev_hot_paths import require_hot
    policy = load_config(ROOT / "scripts/storage_policy.json")
    hot = policy.get("fast_volumes", policy["hot_volume"])
    require_evidence_writable(out)
    out = out.resolve()
    allowed = [dev_evidence_root(ROOT) / "release-evidence", ROOT / ".cache/local-ai"]
    if not any(out.is_relative_to(path.resolve()) for path in allowed):
        raise ValueError("output must be under shared release-evidence or the checkout .cache/local-ai")
    for label, path in (("repository", ROOT), ("model", model), ("output", out),
                        ("runtime", local_llm.llama_dir())):
        require_hot(path, f"CPU trial {label}", hot)
    if not model.is_file():
        raise FileNotFoundError(model)
    if (projector is None) != (projector_sha256 is None) or (projector is None) != (audio_manifest is None):
        raise ValueError("--projector, --projector-sha256, and --audio-manifest must be supplied together")
    if audio_manifest:
        require_hot(audio_manifest, "CPU trial audio manifest", hot)
    if requests_path:
        require_hot(requests_path, "CPU trial request input", hot)
    audio_rows = load_audio_rows(audio_manifest) if audio_manifest else []
    if projector and not projector.is_file(): raise FileNotFoundError(projector)
    admission = admit_cpu(model, projector.stat().st_size if projector else 0)
    digest = sha256_file(model)
    if digest.lower() != expected_sha.lower():
        raise ValueError(f"model SHA256 mismatch: got {digest}")
    projector_digest = None
    if projector:
        require_hot(projector, "CPU trial projector", hot)
        projector_digest = sha256_file(projector)
        if projector_digest.lower() != projector_sha256.lower():
            raise ValueError(f"projector SHA256 mismatch: got {projector_digest}")
        fixture_paths = validate_audio_fixtures(audio_rows, require_hot, hot)
    else:
        fixture_paths = []
    out.mkdir(parents=True, exist_ok=True)
    exe = local_llm.llama_dir() / "llama-server.exe"
    runtime_sha = sha256_file(exe)
    runtime_files = {p.name: sha256_file(p) for p in sorted(exe.parent.glob('*.dll'))}
    runtime_files[exe.name] = runtime_sha
    port = local_llm.free_port()
    base = f"http://127.0.0.1:{port}"
    env = child_environment()
    flags = local_llm.CREATE_NO_WINDOW | local_llm.BELOW_NORMAL_PRIORITY_CLASS
    job = local_llm.Job()
    proc = None
    record = {"model_path":str(model.resolve()), "model_sha256":digest, "runtime_path":str(exe.resolve()),
              "runtime_sha256":runtime_sha, "runtime_build":local_llm.BUILD, "device":"CPU only",
              "runtime_files_sha256":runtime_files, "harness_sha256":sha256_file(Path(__file__)),
              "command":server_command(model, port, projector), "environment":{"CUDA_VISIBLE_DEVICES":"-1"},
              "admission":admission, "max_tokens":max_tokens, "thinking":thinking,
              "requests_sha256":sha256_file(requests_path) if requests_path else None,
              "constrained_json":constrained, "tasks":[], "replays":[], "projector_path":str(projector.resolve()) if projector else None,
              "projector_sha256":projector_digest, "audio_manifest_sha256":sha256_file(audio_manifest) if audio_manifest else None,
              "audio_ablation":"omitted" if omit_audio else "none",
              "audio":[]}
    def checkpoint():
        temporary = out / 'trial.partial.json'
        temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(temporary, out / 'trial.json')
    record['status'] = 'starting'
    checkpoint()
    try:
        log_path = out / "server.log"
        with log_path.open("wb") as log:
            proc = __import__("subprocess").Popen(record["command"], stdin=__import__("subprocess").DEVNULL,
                stdout=log, stderr=__import__("subprocess").STDOUT, creationflags=flags, env=env)
        job.assign(proc.pid)
        local_llm._wait_health(proc, base, health_timeout, log_path)
        record['status'] = 'running'
        if audio_rows:
            record["stage"] = "audio_caption_selection_proxy"
            for i, row in enumerate(audio_rows):
                fixture = fixture_paths[i]
                prompt, expected_letter = audio_prompt(audio_rows, i)
                if omit_audio:
                    expected_letter = 'UNKNOWN'
                item = {"id":row["ytid"], "fixture_path":str(fixture), "fixture_sha256":row["fixture_sha256"],
                        "expected_letter":expected_letter, "prompt":prompt, "status":"starting"}
                record["audio"].append(item)
                checkpoint()
                try:
                    wav = fixture.read_bytes()
                    if hashlib.sha256(wav).hexdigest().lower() != row['fixture_sha256'].lower():
                        raise ValueError('audio fixture changed after initial verification')
                    started = time.monotonic()
                    body = _wav_request_body(wav, prompt, max_tokens, thinking)
                    if omit_audio:
                        body['messages'][0]['content'] = [{'type':'text', 'text':prompt}]
                    if constrained:
                        body['response_format'] = {'type':'json_object'}
                    result = local_llm._http_json(base + "/v1/chat/completions", body, timeout=1200)
                    choice = result["choices"][0]
                    response = choice["message"]["content"]
                    allowed = set('ABCDEFGH'[:len(audio_rows)]) | {"UNKNOWN"}
                    selected = _score_caption(response, allowed)
                    try:
                        _, adapter = decode_json(response)
                    except ValueError:
                        adapter = 'invalid'
                    correct = selected == expected_letter and choice.get('finish_reason') != 'length'
                    item.update({"response":response, "usage":result.get("usage"), "finish_reason":choice.get("finish_reason"),
                                 "elapsed_s":round(time.monotonic()-started,2), "selected_caption_id":selected,
                                 "selected_letter":selected, "passed":correct,
                                 "output_adapter":adapter,
                                 "correct_caption_id":row["ytid"] if correct else None,
                                 "status":"complete", "score_kind":"caption selection proxy; not full acoustic/naming/product quality"})
                except Exception as exc:
                    item.update({"status":"error", "stage_error":f"{type(exc).__name__}: {exc}"})
                checkpoint()
        else:
          for task in TASKS:
            started = time.monotonic()
            body = request_body(task["prompt"], max_tokens, thinking,
                                constrained and task['id'] != 'simple_function_ast')
            result = local_llm._http_json(base + "/v1/chat/completions", body, timeout=1200)
            response = result["choices"][0]["message"]["content"]
            record["tasks"].append({"id":task["id"], "prompt":task["prompt"], "response":response,
                                    "score":score(task,response), "usage":result.get("usage"),
                                    "finish_reason":result["choices"][0].get("finish_reason"),
                                    "reasoning":result["choices"][0]["message"].get("reasoning_content"),
                                    "elapsed_s":round(time.monotonic()-started,2)})
            checkpoint()
        if requests_path is not None and not audio_rows:
            for row in read_requests(requests_path):
                try:
                    body = request_body(row["prompt"], row.get("max_tokens", max_tokens), thinking,
                                        constrained and row['id'].startswith(('fix-', 'currentbest-')))
                    result = local_llm._http_json(base + "/v1/chat/completions", body, timeout=1200)
                    response = result["choices"][0]["message"]["content"]
                    record["replays"].append({"id":row["id"],"prompt":row["prompt"],"response":response,
                        "finish_reason":result["choices"][0].get("finish_reason"), "usage":result.get("usage"),
                        "status":"raw_unscored"})
                except Exception as exc:
                    record["replays"].append({"id":row["id"],"prompt":row["prompt"],"status":"error",
                                              "error":f"{type(exc).__name__}: {exc}"})
                checkpoint()
        record["status"] = ("error" if any(item.get("status") == "error" for item in record["audio"]) else "complete") if audio_rows else "complete"
        if record["status"] == "error":
            record["stage_error"] = "one or more audio requests failed; inspect per-row stage_error"
    except BaseException as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        job.close()
        if proc is not None:
            if proc.poll() is None:
                proc.kill()  # also covers failure before Job assignment
            proc.wait(timeout=30)
        checkpoint()
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--model-sha256",required=True)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--health-timeout",type=float,default=120)
    p.add_argument("--max-tokens",type=int,default=512)
    p.add_argument("--thinking",action="store_true")
    p.add_argument("--constrained",action="store_true")
    p.add_argument("--requests",type=Path,help="optional JSONL of historical prompts with id and prompt fields; responses are raw and unscored")
    p.add_argument("--projector",type=Path)
    p.add_argument("--projector-sha256")
    p.add_argument("--audio-manifest",type=Path)
    p.add_argument("--omit-audio",action="store_true",help="CPU-only no-audio control on the same caption prompts")
    a=p.parse_args(argv)
    if len(a.model_sha256)!=64 or any(c not in "0123456789abcdefABCDEF" for c in a.model_sha256):
        p.error("--model-sha256 must be 64 hexadecimal characters")
    if not 1 <= a.health_timeout <= 600: p.error("--health-timeout must be 1..600 seconds")
    if not 1 <= a.max_tokens <= 2048: p.error("--max-tokens must be 1..2048")
    if (a.projector is None) != (a.projector_sha256 is None) or (a.projector is None) != (a.audio_manifest is None):
        p.error("--projector, --projector-sha256, and --audio-manifest must be supplied together")
    if a.projector_sha256 and (len(a.projector_sha256)!=64 or any(c not in "0123456789abcdefABCDEF" for c in a.projector_sha256)):
        p.error("--projector-sha256 must be 64 hexadecimal characters")
    if a.omit_audio and not a.audio_manifest:
        p.error('--omit-audio requires --audio-manifest')
    run(a.model.resolve(),a.model_sha256,a.out.resolve(),a.health_timeout,a.requests.resolve() if a.requests else None,
        a.max_tokens,a.thinking,a.constrained,a.projector.resolve() if a.projector else None,
        a.projector_sha256,a.audio_manifest.resolve() if a.audio_manifest else None,a.omit_audio)
    return 0

if __name__ == "__main__": raise SystemExit(main())
