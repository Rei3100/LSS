"""Queue boundaries: re-use without inference, failure receipts, provenance and confinement."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_ai_queue as queue

GENERATED = json.loads((Path(__file__).parent / "data/local_ai_reader_cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture(autouse=True)
def isolated_fake_queue_policy(tmp_path, monkeypatch):
    import dev_quiet_mode
    # These cases inject fake serving/admission; user sleep state is tested separately.
    monkeypatch.setattr(dev_quiet_mode, 'POLICY', tmp_path / 'absent-sleep-policy.json')


def setup_queue(tmp_path):
    calls = []
    model = SimpleNamespace(name="test", request={})
    @contextlib.contextmanager
    def serve(*args, **kwargs):
        calls.append("load")
        yield "http://127.0.0.1:1/v1"
    def complete(*args, **kwargs):
        calls.append("request")
        return SimpleNamespace(text="draft", usage={}, raw={"choices": [
            {"finish_reason": "stop", "message": {"content": "draft"}}]})
    requests = [{"id": "one", "messages": [{"role": "user", "content": "Question"}], "max_tokens": 20},
                {"id": "two", "messages": [{"role": "user", "content": "Other question"}], "max_tokens": 20}]
    def run(**kwargs):
        return queue.run_queue(requests, model, tmp_path, serving=serve,
                               complete=kwargs.pop("complete", complete),
                               identify=kwargs.pop("identify", lambda m: {"sha256": "abc"}),
                               admit=lambda m: None, **kwargs)
    return calls, requests, run


def test_batch_loads_once_and_resume_does_not_load(tmp_path):
    calls, requests, run = setup_queue(tmp_path)
    result = run()
    assert result["status"] == "drafts_ready"
    assert calls == ["load", "request", "request"]
    calls.clear()
    result = run()
    assert calls == []
    assert result["cached"] == 2
    requests[0]["messages"][0]["content"] = "Changed question"
    assert run()["cached"] == 1
    assert calls == ["load", "request"]


def test_model_change_invalidates_and_corrupt_receipt_retries(tmp_path):
    calls, _, run = setup_queue(tmp_path)
    result = run()
    receipt = Path(result["entries"][0]["receipt"])
    data = json.loads(receipt.read_text())
    data["response"]["choices"] = []
    receipt.write_text(json.dumps(data))
    assert run()["cached"] == 1
    assert run(identify=lambda m: {"sha256": "changed"})["cached"] == 0


def test_failure_is_not_success_and_is_retried(tmp_path):
    calls, _, run = setup_queue(tmp_path)
    def broken(*args, **kwargs):
        raise TimeoutError("inference timeout")
    assert run(complete=broken)["status"] == "failed"
    assert run()["cached"] == 0
    assert run()["cached"] == 2


def test_token_limit_never_cached_as_complete(tmp_path):
    calls, _, run = setup_queue(tmp_path)
    def truncated(*args, **kwargs):
        return SimpleNamespace(text="part", usage={}, raw={"choices": [
            {"finish_reason": "length", "message": {"content": "part"}}]})
    assert run(complete=truncated)["status"] == "failed"
    assert run()["cached"] == 0


@pytest.mark.parametrize("value", [None, {}, [], {"status": "draft"}])
def test_malformed_cache_is_not_reused(value):
    assert not queue.reusable(value, {}, {}, "key")


def test_fresh_refusal_does_not_stage_or_hash_weights(tmp_path):
    def refused(model):
        raise queue.local_llm.AdmissionRefused("not enough memory")
    def forbidden(model):
        pytest.fail("must refuse before copying or hashing weights")
    summary = queue.run_queue([{"id": "one"}], SimpleNamespace(name="test"), tmp_path,
                              admit=refused, identify=forbidden)
    assert summary["status"] == "failed"
    assert summary["entries"][0]["status"] == "refused"


def test_output_rejects_product_and_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    for path in [tmp_path / ".state", tmp_path / "lss", tmp_path.parent / "escape", tmp_path / "release-evidence"]:
        with pytest.raises(ValueError):
            queue.output_directory(path)
    assert queue.output_directory(tmp_path / ".cache/local-ai/queue").is_dir()


@pytest.mark.parametrize("row", [{"id": "a", "messages": []},
    {"id": "a", "messages": [{"role": "user", "content": "x"}], "max_tokens": True},
    {"id": "a", "messages": [{"role": "tool", "content": "x"}]},
    {"id": "a", "messages": [{"role": "user", "content": "x"}], "command": "run"}])
def test_invalid_input_rejected_before_model_start(tmp_path, row):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        queue.read_requests(path)


@pytest.mark.parametrize("case", GENERATED, ids=[case["label"] for case in GENERATED])
def test_reviewed_local_model_regression_vectors(tmp_path, case):
    path = tmp_path / "generated-input.jsonl"
    path.write_text(case["rows"], encoding="utf-8")
    if case["expected"]:
        assert queue.read_requests(path)
    else:
        with pytest.raises(ValueError):
            queue.read_requests(path)

def test_cpu_and_gpu_have_distinct_resume_identities(tmp_path):
    import contextlib
    from types import SimpleNamespace
    calls = []
    @contextlib.contextmanager
    def serve(name, **kwargs):
        calls.append(kwargs.get('cpu', False))
        yield 'fake'
    def complete(*args, **kwargs):
        return SimpleNamespace(text='draft', usage={}, raw={'choices': [{'finish_reason': 'stop', 'message': {'content': 'draft'}}]})
    requests = [{'id': 'x', 'messages': [], 'max_tokens': 2}]
    model = SimpleNamespace(name='test', request={})
    def run(cpu):
        return queue.run_queue(requests, model, tmp_path, serving=serve, complete=complete,
                               identify=lambda m: {'sha256': 'same'}, admit=lambda m, **kw: None, cpu=cpu)
    assert run(False)['cached'] == 0
    assert run(True)['cached'] == 0
    assert run(False)['cached'] == 1
    assert run(True)['cached'] == 1
    assert calls == [False, True]


@pytest.mark.parametrize("text,content", [({"bad": 1}, {"bad": 1}), ("draft", "different")])
def test_fresh_draft_requires_full_reusable_invariant(tmp_path, text, content):
    calls, _, run = setup_queue(tmp_path)
    def malformed(*args, **kwargs):
        return SimpleNamespace(text=text, usage={}, raw={"choices": [
            {"finish_reason": "stop", "message": {"content": content}}]})
    result = run(complete=malformed)
    assert result["status"] == "failed"
    for entry in result['entries']:
        receipt = json.loads(Path(entry['receipt']).read_text(encoding='utf-8'))
        assert receipt['status'] == 'incomplete'
        assert receipt['response']['choices'][0]['message']['content'] == content
    assert run(complete=malformed)['cached'] == 0


def _real_fix_code(directory):
    evidence = dev_evidence_root(Path(__file__).resolve().parents[1]) / "release-evidence/strata-deploy-20261031" / directory
    for path in evidence.glob("*.json"):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("request", {}).get("id") == "native-asr-grounding-deployment-fix-20261031":
            return receipt["text"]
    raise AssertionError(directory)


def test_python_def_accepts_strata_drop_in_and_refuses_gpt_oss_module():
    request = {"id": "fix", "python_def": "_naming_transcription"}
    queue.check_python_def(request, {"text": _real_fix_code("strata-out")})
    with pytest.raises(ValueError, match="exactly one 'def _naming_transcription'"):
        queue.check_python_def(request, {"text": _real_fix_code("local-gpt-oss-20b")})
    with pytest.raises(SyntaxError):
        queue.check_python_def(request, {"text": json.dumps({"code": "def _naming_transcription(:\n"})})
    queue.check_python_def({"id": "plain"}, {"text": "not json"})  # no contract, nothing parsed


def test_python_def_request_field_names_one_identifier(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"id": "a", "messages": [{"role": "user", "content": "x"}],
                                "python_def": "not an identifier"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="python_def"):
        queue.read_requests(path)
