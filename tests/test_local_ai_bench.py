"""Controls and containment for local-model development replay."""
import sys
from pathlib import Path

import pytest

from tests.gates.gate_support import require_historical_commits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_ai_bench as bench


@pytest.mark.parametrize("source", [
    "def f(x):\n import os\n return os.system(x)",
    "def f(x):\n return open(x)",
    "def f(x):\n return x.__class__",
    "def f(x):\n return eval(x)",
    "def f(x):\n return f(x)",
    "def f(x):\n return x.read_text()",
    "def f(x):\n return [0] * 1000000000",
    "def f(x):\n global other\n return x",
    "@print\ndef f(x):\n return x",
    "def f(x=print('side effect')):\n return x",
    "def f(x):\n return x\nprint('extra code')",
])
def test_rejects_unsafe_generated_code(source):
    with pytest.raises((ValueError, SyntaxError)):
        bench.validate_safe(source, "f", {})


def test_runtime_child_executes_frozen_cases():
    assert bench.evaluate_code("def f(x):\n return x.strip()\n", "f", {}, [([" a "], "a")])["pass"]
    assert not bench.evaluate_code("def f(x):\n return x\n", "f", {}, [([" a "], "a")])["pass"]


def test_infinite_loop_is_killed():
    result = bench.evaluate_code("def f(x):\n while True:\n  pass\n", "f", {}, [(["x"], "x")], timeout=0.3)
    assert result == {"pass": False, "error": "timeout"}


def test_normalized_ast_ignores_docs_but_not_behavior():
    assert bench.normalized('def f(x):\n "docs"\n return x') == bench.normalized('def f(x):\n return x # comment')
    assert bench.normalized('def f(x):\n return x') != bench.normalized('def f(x):\n return None')


def test_regex_extraction_never_executes_other_assignments():
    constants = bench.regex_constants('import re\nA = re.compile("a", re.I)\nEVIL = open("do-not-open")')
    assert constants["A"].match("A")
    assert "EVIL" not in constants


def test_historical_replay_population_and_controls():
    for commit, _, _ in bench.REPLAYS:
        require_historical_commits(bench.ROOT, commit, f"{commit}^1")
    tasks = bench.load_replays()
    assert len(tasks) >= 10
    assert len({t["commit"] for t in tasks}) == len(tasks)
    runtime = [t for t in tasks if t["cases"]]
    assert len(runtime) == 3
    for task in runtime:
        assert not bench.evaluate_code(task["before"], task["name"], task["constants"], task["cases"])["pass"]
        assert bench.evaluate_code(task["after"], task["name"], task["constants"], task["cases"])["pass"]
    for task in tasks:
        prompt = bench.prompt_for(task)
        assert task["after"] not in prompt
        assert "expected" not in prompt


def test_music_semantics_is_not_audio_quality():
    assert sum(t[1] == "music_semantics" for t in bench.REASONING) == 4
    assert all(isinstance(t[3], str) for t in bench.REASONING)
