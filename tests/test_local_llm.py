"""scripts/local_llm.py: registry validation, RAM admission, GPU lease order, Job Object kill, runs.jsonl, `run`.

A tiny Python HTTP server stands in for llama-server (started through the same hidden, job-bound path); the GPU
lease, the staging cache, nvidia-smi and the memory figures are replaced, and everything is written under tmp_path."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_llm as llm  # noqa: E402

pytestmark = pytest.mark.skipif(os.name != "nt", reason="the runner uses Windows job objects")

FAKE_SERVER = r'''
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
port = int(argv[argv.index("--port") + 1])
with open(os.environ["FAKE_ARGV_OUT"], "w") as out:
    json.dump(argv, out)
with open(os.environ["FAKE_PID_OUT"], "w") as out:
    out.write(str(os.getpid()))
state = {"prompt": 0, "completion": 0}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/metrics":
            self._send(("llamacpp:prompt_tokens_total %d\nllamacpp:tokens_predicted_total %d\n"
                        % (state["prompt"], state["completion"])).encode(), "text/plain")
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(os.environ["FAKE_REQ_OUT"], "w") as out:
            json.dump(body, out)
        state["prompt"] += 7
        state["completion"] += 3
        self._send({"choices": [{"message": {"content": "pong", "reasoning_content": "hm"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "timings": {"predicted_per_second": 9.0}})

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


class FakeLease:
    events: list[str] = []

    def close(self) -> None:
        FakeLease.events.append("lease-closed")


@pytest.fixture
def env(tmp_path, monkeypatch):
    import dev_quiet_mode
    import dev_hot_paths
    policy = tmp_path / "hot-policy.json"
    policy.write_text(json.dumps({"hot_volume": ('C' + chr(58) + '/'),
                                  "fast_volumes": [('C' + chr(58) + '/'), ('D' + chr(58) + '/')]}), encoding="utf-8")
    monkeypatch.setattr(dev_hot_paths, "POLICY_PATH", policy)
    # These cases use a fake HTTP server/lease; ambient user sleep state is not their input.
    monkeypatch.setattr(dev_quiet_mode, 'POLICY', tmp_path / 'absent-sleep-policy.json')
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    weights = tmp_path / "fake.gguf"
    weights.write_bytes(b"x" * 64)
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"models": {
        "fake": {"path": str(weights), "size_bytes": 64, "kind": "chat", "ram_gb": 0.1, "args": ["--n-cpu-moe", "2"],
                 "request": {"chat_template_kwargs": {"x": 1}}},
        "fake-embed": {"path": str(weights), "size_bytes": 64, "kind": "embedding", "ram_gb": 0.1}}}), encoding="utf-8")
    monkeypatch.setattr(llm, "REGISTRY", registry)
    monkeypatch.setattr(llm, "HOME", tmp_path / "home")
    monkeypatch.setattr(llm.gpu_run, "HOLDER", tmp_path / "home" / "gpu-holder.json")
    monkeypatch.setattr(llm.gpu_run, "memory_headroom", lambda: (50.0, 50.0))
    monkeypatch.setattr(llm, "_vram_mib", lambda: None)
    # the base interpreter runs the fake directly (the venv launcher would spawn a child that outruns the job assignment)
    monkeypatch.setattr(llm, "_launcher", lambda: [sys._base_executable, str(script)])
    monkeypatch.setattr(llm, "stage_path", lambda path, purpose=None: Path(path))
    monkeypatch.setattr(llm, "_lease_holder_is_ancestor", lambda: False)
    FakeLease.events = []

    def acquire(*, on_wait=None):
        FakeLease.events.append("lease-taken")
        return FakeLease()
    monkeypatch.setattr(llm.gpu_lease, "acquire", acquire)
    for name in ("ARGV", "PID", "REQ"):
        monkeypatch.setenv(f"FAKE_{name}_OUT", str(tmp_path / f"fake_{name.lower()}.txt"))
    monkeypatch.setattr(llm, "_log", lambda message: None)
    monkeypatch.setattr(llm.time, "sleep", time.sleep)
    return tmp_path


def _rows(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "home" / "local-llm" / "runs.jsonl").read_text(encoding="utf-8").splitlines()]


def _pid(tmp_path) -> int:
    return int((tmp_path / "fake_pid.txt").read_text())


# --- registry ----------------------------------------------------------------------------------------------------------

def test_shipped_registry_is_valid_and_has_the_wanted_models():
    models = llm.load_registry(Path(llm.__file__).with_name("local_llm_models.json"))
    assert {"gpt-oss-20b", "qwen3.6-35b-a3b", "ornith-1.0-9b", "qwen3-embedding-0.6b", "qwen3-reranker-0.6b"} <= set(models)
    assert models["qwen3-embedding-0.6b"].kind == "embedding" and models["qwen3-reranker-0.6b"].kind == "rerank"
    assert all(len(m.sha256) == 64 for m in models.values() if m.sha256)
    assert "--n-cpu-moe" in models["gpt-oss-20b"].args


def test_hdd_weights_refuse_before_staging_or_gpu_launch(env, monkeypatch):
    from dataclasses import replace
    from dev_hot_paths import HotPathError
    model = replace(llm.get_model("fake"), path=Path(('E' + chr(58) + '/archive/not-read.gguf')))
    monkeypatch.setattr(llm, "get_model", lambda name: model)
    monkeypatch.setattr(llm, "stage_path", lambda *a, **kw: pytest.fail("HDD staging is forbidden at execution"))
    with pytest.raises(HotPathError):
        with llm.serving("fake"):
            pass
    assert FakeLease.events == []
    assert not (env / "fake_pid.txt").exists()


def test_registry_validation_lists_every_problem():
    bad = {"models": {
        "a": {"path": "relative.gguf", "size_bytes": 1, "ram_gb": 1},
        "b": {"path": ('C' + chr(58) + '/x.gguf'), "size_bytes": 0, "ram_gb": 1, "kind": "vision"},
        "c": {"path": ('C' + chr(58) + '/x.gguf'), "size_bytes": 1, "ram_gb": 1, "args": ["-c", "99"]},
        "d": {"path": ('C' + chr(58) + '/x.gguf'), "size_bytes": 1, "ram_gb": -1, "sha256": "abc"}}}
    with pytest.raises(llm.RegistryError) as raised:
        llm.parse_registry(bad)
    message = str(raised.value)
    for fragment in ("a: path", "b: kind", "b: size_bytes", "c: args", "d: ram_gb", "d: sha256"):
        assert fragment in message
    with pytest.raises(llm.RegistryError):
        llm.parse_registry({"models": {}})


def test_unknown_model_and_size_mismatch_are_refused(env):
    with pytest.raises(llm.RegistryError, match="unknown model"):
        with llm.serving("nope"):
            pass
    (env / "fake.gguf").write_bytes(b"short")
    with pytest.raises(llm.RegistryError, match="size"):
        with llm.serving("fake"):
            pass
    assert FakeLease.events == []


def test_server_args_are_f16_loopback_and_kind_flags():
    models = llm.load_registry(Path(llm.__file__).with_name("local_llm_models.json"))
    chat = llm.server_args(models["gpt-oss-20b"], 1234, Path("w.gguf"), None)
    assert chat[chat.index("-ctk") + 1] == "f16" and chat[chat.index("-ctv") + 1] == "f16"
    assert chat[chat.index("--host") + 1] == "127.0.0.1" and "--jinja" in chat and "--no-webui" in chat
    assert chat[chat.index("-np") + 1] == "1" and chat[-2:] == ["--n-cpu-moe", "6"]
    sized = llm.server_args(models["gpt-oss-20b"], 1, Path("w"), 4096)
    assert sized[sized.index("-c") + 1] == "4096"
    assert "--embeddings" in llm.server_args(models["qwen3-embedding-0.6b"], 1, Path("w"), None)
    assert "--reranking" in llm.server_args(models["qwen3-reranker-0.6b"], 1, Path("w"), None)


# --- admission ---------------------------------------------------------------------------------------------------------

def test_admission_refuses_low_ram_and_low_commit_before_anything_starts(env, monkeypatch):
    monkeypatch.setattr(llm.gpu_run, "memory_headroom", lambda: (50.0, 3.9))       # needs 0.1 + 4 GB RAM
    with pytest.raises(llm.AdmissionRefused, match="needs 4.1 GB"):
        with llm.serving("fake"):
            pass
    monkeypatch.setattr(llm.gpu_run, "memory_headroom", lambda: (3.0, 50.0))       # commit headroom below 4 GB
    with pytest.raises(llm.AdmissionRefused, match="commit"):
        with llm.serving("fake"):
            pass
    assert FakeLease.events == [] and not (env / "fake_pid.txt").exists()
    assert [r["status"].split(":")[0] for r in _rows(env)] == ["refused", "refused"]


def test_admission_wait_polls_until_headroom_returns(env, monkeypatch):
    seen = iter([(50.0, 1.0), (50.0, 2.0), (50.0, 9.0)])
    sleeps = []
    monkeypatch.setattr(llm.gpu_run, "memory_headroom", lambda: next(seen))
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    llm.admit(llm.get_model("fake"), wait=True, poll=7)
    assert sleeps == [7, 7]


def test_cli_exit_code_for_refusal(env, monkeypatch, tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("hi", encoding="utf-8")
    monkeypatch.setattr(llm.gpu_run, "memory_headroom", lambda: (50.0, 0.5))
    assert llm.main(["chat", "fake", "--prompt-file", str(prompt)]) == llm.EXIT_REFUSED


# --- lease, job object, rows -------------------------------------------------------------------------------------------

def test_lease_is_taken_before_the_server_and_released_after_it_died(env):
    with llm.serving("fake") as base:
        assert FakeLease.events == ["lease-taken"]
        pid = _pid(env)
        assert llm._alive(pid) and base.startswith("http://127.0.0.1:") and base.endswith("/v1")
        assert json.loads((llm.gpu_run.HOLDER).read_text(encoding="utf-8"))["pid"] == os.getpid()
    assert FakeLease.events == ["lease-taken", "lease-closed"]
    assert not llm._alive(pid)
    assert not llm.gpu_run.HOLDER.exists()
    argv = json.loads((env / "fake_argv.txt").read_text())
    assert argv[argv.index("--host") + 1] == "127.0.0.1" and "--n-cpu-moe" in argv and "--jinja" in argv


def test_job_object_kills_the_server_when_the_body_raises(env):
    with pytest.raises(RuntimeError, match="boom"):
        with llm.serving("fake"):
            pid = _pid(env)
            assert llm._alive(pid)
            raise RuntimeError("boom")
    assert not llm._alive(pid)
    assert FakeLease.events == ["lease-taken", "lease-closed"]
    row = _rows(env)[-1]
    assert row["status"].startswith("error: RuntimeError: boom")


def test_keyboard_interrupt_is_recorded_and_still_cleans_up(env):
    with pytest.raises(KeyboardInterrupt):
        with llm.serving("fake"):
            pid = _pid(env)
            raise KeyboardInterrupt
    assert not llm._alive(pid) and _rows(env)[-1]["status"] == "interrupted"


def test_server_that_never_becomes_healthy_fails_cleanly(env, monkeypatch):
    (env / "dead.py").write_text("import sys; sys.exit(5)", encoding="utf-8")
    monkeypatch.setattr(llm, "_launcher", lambda: [sys._base_executable, str(env / "dead.py")])
    with pytest.raises(llm.ServerFailed, match="exited with 5"):
        with llm.serving("fake"):
            pass
    assert FakeLease.events == ["lease-taken", "lease-closed"]
    assert _rows(env)[-1]["status"].startswith("error: ServerFailed")


def test_failed_job_assignment_kills_unowned_child(env, monkeypatch):
    pids = []

    def fail_assignment(self, pid):
        pids.append(pid)
        raise OSError("assignment refused")

    monkeypatch.setattr(llm.Job, "assign", fail_assignment)
    with pytest.raises(OSError, match="assignment refused"):
        with llm.serving("fake"):
            pass
    assert len(pids) == 1 and not llm._alive(pids[0])
    assert FakeLease.events == ["lease-taken", "lease-closed"]


def test_lease_held_by_an_ancestor_is_refused_not_waited_for(env, monkeypatch):
    monkeypatch.setattr(llm, "_lease_holder_is_ancestor", lambda: True)
    with pytest.raises(RuntimeError, match="gpu_run"):
        with llm.serving("fake"):
            pass
    assert FakeLease.events == []


def test_runs_jsonl_row_has_the_fields_and_the_server_token_counts(env):
    with llm.serving("fake", purpose="unit test") as base:
        reply = llm.complete(base, [{"role": "user", "content": "ping"}], {"type": "object"}, max_tokens=5,
                             extra={"chat_template_kwargs": {"x": 1}})
        assert reply.text == "pong" and reply.reasoning == "hm" and reply.usage["completion_tokens"] == 3
    request = json.loads((env / "fake_req.txt").read_text())
    assert request["response_format"]["json_schema"]["strict"] is True and request["max_tokens"] == 5
    assert request["temperature"] == 0 and request["chat_template_kwargs"] == {"x": 1}
    (row,) = _rows(env)
    assert row["model"] == "fake" and row["purpose"] == "unit test" and row["status"] == "ok"
    assert row["staged"] == str(env / "fake.gguf") and row["prompt_tokens"] == 7 and row["completion_tokens"] == 3
    assert row["load_s"] > 0 and row["wall_s"] >= row["load_s"] and row["ram_peak_gb"] > 0
    assert row["utc"].endswith("Z") and row["lease_wait_s"] is not None
    for key in ("vram_before_mib", "vram_peak_mib", "vram_after_mib", "ram_avail_min_gb", "stage_seconds"):
        assert key in row


# --- run / chat --------------------------------------------------------------------------------------------------------

def test_run_passes_the_exit_code_and_sets_the_endpoint_variables(env):
    probe = ("import os, sys, urllib.request; b = os.environ['LOCAL_LLM_BASE_URL']; "
             "ok = (b == os.environ['OPENAI_BASE_URL'] == os.environ['OPENAI_API_BASE'] and os.environ['OPENAI_API_KEY'] == 'local' "
             "and urllib.request.urlopen(b[:-3] + '/health').status == 200); sys.exit(7 if ok else 1)")
    assert llm.main(["run", "fake", "--", sys.executable, "-c", probe]) == 7
    assert not llm._alive(_pid(env))
    assert llm.main(["run", "fake", "--", sys.executable, "-c", "raise SystemExit(0)"]) == 0
    assert llm.main(["run", "fake", "--", "definitely-not-a-program-xyz"]) == 127
    assert [r["status"] for r in _rows(env)] == ["ok", "ok", "ok"]


def test_chat_prints_the_reply_and_stops_the_server(env, capsys):
    prompt = env / "prompt.txt"
    prompt.write_text("こんにちは", encoding="utf-8")
    assert llm.main(["chat", "fake", "--prompt-file", str(prompt), "--max-tokens", "9"]) == 0
    assert capsys.readouterr().out.strip() == "pong"
    request = json.loads((env / "fake_req.txt").read_text())
    assert request["messages"][-1]["content"] == "こんにちは" and request["chat_template_kwargs"] == {"x": 1}
    assert not llm._alive(_pid(env))


def test_install_is_idempotent_with_a_receipt(env, monkeypatch):
    monkeypatch.setattr(llm, "ASSETS", {})
    target = llm.llama_dir()
    target.mkdir(parents=True)
    (target / "llama-server.exe").write_bytes(b"")
    target.with_name(f"{llm.BUILD}.receipt.json").write_text(json.dumps({"assets": {}}), encoding="utf-8")
    assert llm.install() == target
    assert (target / "llama-server.exe").is_file()

def test_cpu_server_skips_gpu_and_forces_flags_environment(env, monkeypatch):
    monkeypatch.setattr(llm.gpu_lease, 'acquire', lambda **kw: pytest.fail('GPU lease'))
    monkeypatch.setattr(llm, '_lease_holder_is_ancestor', lambda: pytest.fail('GPU lease probe'))
    monkeypatch.setattr(llm, '_vram_mib', lambda: pytest.fail('VRAM probe'))
    monkeypatch.setattr(llm, '_vram_release', lambda *a: pytest.fail('VRAM probe'))
    monkeypatch.setattr(llm, '_wait_health', lambda *a: 0.1)
    monkeypatch.setattr(llm, '_metric', lambda *a: None)
    monkeypatch.setattr(llm, 'peak_working_set_gb', lambda pid: None)
    seen = {}
    class Process:
        pid = 123
        def poll(self): return None
        def wait(self, *a): pass
        def kill(self): seen['killed'] = True
    class Job:
        def assign(self, pid): seen['assigned'] = pid
        def close(self): seen['closed'] = True
    def launch(command, **kwargs):
        seen['command'], seen['env'] = command, kwargs['env']
        return Process()
    monkeypatch.setattr(llm, 'Job', Job)
    monkeypatch.setattr(llm.subprocess, 'Popen', launch)
    with llm.serving('fake', cpu=True):
        assert seen['command'][-9:] == ['-ngl', '0', '--device', 'none', '-t', '2', '-tb', '2', '--no-mmproj-offload']
        assert seen['env']['CUDA_VISIBLE_DEVICES'] == '-1'
    assert seen['assigned'] == 123 and seen['closed']
    assert _rows(env)[-1]['device'] == 'cpu'


def test_cpu_health_failure_cleans_child_without_gpu(env, monkeypatch):
    monkeypatch.setattr(llm.gpu_lease, 'acquire', lambda **kw: pytest.fail('GPU lease'))
    monkeypatch.setattr(llm, '_vram_mib', lambda: pytest.fail('VRAM probe'))
    monkeypatch.setattr(llm, 'peak_working_set_gb', lambda pid: None)
    seen = {}
    class Process:
        pid = 123
        def poll(self): return None
        def wait(self, *a): pass
        def kill(self): seen['killed'] = True
    class Job:
        def assign(self, pid): pass
        def close(self): seen['closed'] = True
    monkeypatch.setattr(llm, 'Job', Job)
    monkeypatch.setattr(llm.subprocess, 'Popen', lambda *a, **kw: Process())
    def failed(process, *args):
        raise llm.ServerFailed('fake health failure')
    monkeypatch.setattr(llm, '_wait_health', failed)
    with pytest.raises(llm.ServerFailed):
        with llm.serving('fake', cpu=True):
            pass
    assert seen['closed']
    assert _rows(env)[-1]['status'].startswith('error:')


def test_cpu_sampler_stops_only_its_server_on_ram_floor(monkeypatch):
    class Process:
        killed = False
        def poll(self): return None
        def kill(self): self.killed = True
    process = Process()
    monkeypatch.setattr(llm.gpu_run, 'memory_headroom', lambda: (50, 3.9))
    monkeypatch.setattr(llm, '_vram_mib', lambda: pytest.fail('VRAM probe'))
    sampler = llm.CpuSampler(process)
    sampler.sample()
    assert process.killed and sampler.tripped
