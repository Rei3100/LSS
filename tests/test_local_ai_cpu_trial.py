from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_ai_cpu_trial as trial


def test_cpu_command_disables_all_gpu_paths(monkeypatch):
    monkeypatch.setattr(trial.local_llm, "_launcher", lambda: ["llama-server.exe"])
    args = trial.server_command(Path("C:/weights.gguf"), 1234)
    assert "-ngl" in args and args[args.index("-ngl") + 1] == "0"
    assert "--device" in args and args[args.index("--device") + 1] == "none"
    assert "--fit" in args and args[args.index("--fit") + 1] == "off"
    assert "-t" in args and args[args.index("-t") + 1] == "2"
    assert "-c" in args and args[args.index("-c") + 1] == "4096"
    assert "--no-mmap" not in args


def test_audio_projector_command_forces_cpu_residency(monkeypatch):
    monkeypatch.setattr(trial.local_llm, "_launcher", lambda: ["llama-server.exe"])
    args = trial.server_command(Path("C:/weights.gguf"), 1234, Path("C:/mmproj.gguf"))
    assert args[args.index("--mmproj") + 1] == str(Path("C:/mmproj.gguf"))
    assert "--no-mmproj-offload" in args
    assert args[args.index("-ngl") + 1] == "0"
    assert args[args.index("--device") + 1] == "none"


def test_caption_order_rotates_correct_letter():
    rows = [{"ytid":f"id{i}", "caption":f"caption {i}"} for i in range(3)]
    letters = []
    for i in range(3):
        prompt, letter = trial.audio_prompt(rows, i)
        letters.append(letter)
        import json
        choices = json.loads(prompt.split('\n', 1)[1])['candidates']
        matching = next(c for c in choices if c.get('caption') == rows[i]['caption'])
        assert matching['caption_id'] == letter
        assert f'"id{i}"' not in prompt
    assert letters == ["A", "C", "B"]


def test_corrupt_fixture_rejected_before_request_encoding(tmp_path, monkeypatch):
    import hashlib
    import json
    import pytest
    fixture = tmp_path / "clip.wav"
    fixture.write_bytes(b"wav bytes")
    manifest = tmp_path / "acquisition.json"
    rows = [{"status":"prepared", "ytid":str(i), "caption":f"caption {i}",
             "fixture_path":str(fixture), "fixture_sha256":hashlib.sha256(b"other").hexdigest()} for i in range(3)]
    manifest.write_text(json.dumps({"rows":rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="fixture SHA256 mismatch"):
        prepared = trial.load_audio_rows(manifest)
        trial.validate_audio_fixtures(prepared)


def test_subprocess_environment_forces_cuda_hidden_in_runner(monkeypatch):
    env = trial.child_environment({"CUDA_VISIBLE_DEVICES":"0,1", "PATH":"tools"})
    assert env == {"CUDA_VISIBLE_DEVICES":"-1", "PATH":"tools"}


def test_request_profile_keeps_token_budget_and_thinking_explicit():
    body = trial.request_body("a real historical repair", 1024, False)
    assert body["max_tokens"] == 1024
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert trial.request_body("same repair", 1024, True)["chat_template_kwargs"]["enable_thinking"]
    assert trial.request_body("same repair", 1024, False, True)["response_format"] == {"type": "json_object"}


def test_safety_task_scores_unknown_and_held_as_not_currentbest():
    task = next(t for t in trial.TASKS if t["id"] == "currentbest_safety")
    assert trial.score(task, '{"classification":"NO"}') == {
        "passed": True, "parsed": {"classification":"NO"}, "unsupported": None}
    assert trial.score(task, '{"classification":"YES"}') ["passed"] is False


def test_identifier_task_does_not_accept_inferred_source():
    task = next(t for t in trial.TASKS if t["id"] == "truncated_identifier")
    assert trial.score(task, '{"cue":"Q-17","revision":"R-4","missing_source":true}') ["passed"]
    assert not trial.score(task, '{"cue":"Q-17","revision":"R-4","missing_source":false}') ["passed"]


def test_unicode_transfer_requires_exact_text():
    task = next(t for t in trial.TASKS if t["id"] == "unicode_transfer")
    exact = '{"title":"『雨のあと』","note":"café — 東京。"}'
    assert trial.score(task, exact)["passed"]
    assert not trial.score(task, exact.replace("café", "cafe"))["passed"]


def test_generated_code_checks_use_ast_only_and_never_execute():
    code = "def normalize(text):\n    return ' '.join(text.split())\n"
    tree = ast.parse(code)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == "normalize"
    assert ast.unparse(tree.body[0].body[0].value) == "' '.join(text.split())"


def test_optional_replay_requests_are_raw_prompt_records(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id":"repair-1","prompt":"Return code as text"}\n', encoding="utf-8")
    assert trial.read_requests(path) == [{"id":"repair-1","prompt":"Return code as text"}]


def test_optional_replay_requests_reject_duplicate_ids(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id":"x","prompt":"one"}\n{"id":"x","prompt":"two"}\n', encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="unique"):
        trial.read_requests(path)
