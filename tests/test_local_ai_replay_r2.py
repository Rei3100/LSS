"""Safety/profile and bounded-audit contract for local AI replay r2."""
try:
    from scripts.dev_state import dev_evidence_root
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import local_ai_bench as bench
import local_ai_replay_r2 as replay


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
def test_original_frozen_unsafe_cases_remain_rejected(source):
    with pytest.raises((ValueError, SyntaxError)):
        replay.validate_safe(source, "f", {})


def test_r2_profile_and_request_withhold_cases_and_fixed_answer():
    assert replay.PROFILE == {"name": "ornith-constrained-json-no-thinking", "model": "ornith-1.0-9b",
                              "constrained_json": True, "no_thinking": True}
    requests = replay.generate_requests()
    assert len(requests) == 1 and requests[0]["id"] == "fix-993a0fb6-_covering"
    assert "_covering" in requests[0]["prompt"] and "expected" not in requests[0]["prompt"]
    assert replay.REPLAYS[0]["after"] not in requests[0]["prompt"]
    assert replay.validate_responses([])[0]["status"] == "rejected"


def test_list_comprehension_and_pure_get_append_syntax_is_locally_permitted():
    source = "def f(rows):\n out = []\n for row in rows:\n  out.append(row.get('value'))\n return [v for v in out if v is not None]\n"
    assert bench.validate_safe(source, "f", {})


def test_real_buggy_and_fixed_historical_controls_pass_and_fail_as_expected():
    task, = replay.load_replays()
    assert len(task["cases"]) == 3
    assert not replay.evaluate_code(task["before"], task["name"], task["constants"], task["cases"])["pass"]
    assert replay.evaluate_code(task["after"], task["name"], task["constants"], task["cases"])["pass"]


def test_saved_gptoss_normalise_response_passes_after_listcomp_allowance():
    result = replay.revalidate_saved_normalise()
    assert result["model"] == "gpt-oss-20b"
    assert result["revalidated"]["pass"], result


def test_candidate_import_and_indirect_call_stay_rejected():
    for source in ("def f(x):\n import math\n return math.floor(x)",
                   "def f(x):\n return (lambda y: y)(x)"):
        with pytest.raises((ValueError, SyntaxError)):
            replay.validate_safe(source, "f", {})


def test_evidence_is_explicit_about_rejected_candidates_and_empty_control_count():
    evidence = json.loads((dev_evidence_root(ROOT) / "release-evidence/local-ai-repair-r2/audit.json").read_text(encoding="utf-8"))
    assert evidence["eligible_runtime_controls"] == 1
    assert len(evidence["rejected_candidates"]) >= 3
    assert all(row["commit"] and row["test_contract"] and row["reason"] for row in evidence["rejected_candidates"])


def test_corrected_contract_and_frozen_dependencies():
    task, = replay.load_replays()
    prompt = replay.generate_requests()[0]["prompt"]
    assert "not a scalar timestamp" in prompt
    assert "half-open" in prompt and "strictly positive" in prompt
    assert "update the third parameter name" in prompt
    assert set(task["dependency_sha256"]) == {"Placement", "_overlap"}
    assert replay.TRUSTED_OVERLAP.__module__ == replay.__name__
    assert replay.Placement.__module__ == replay.__name__
    assert "lss.validation" not in sys.modules or replay.Placement is not sys.modules["lss.validation"].__dict__.get("Placement")


def test_pure_unpacking_is_allowed_but_mutating_data_is_rejected():
    source = "def f(span):\n a, *tail = span\n return [a, tail[0]]\n"
    assert replay.evaluate_code(source, "f", {}, [([(2, 5)], [2, 5])])["pass"]
    for source in ("def f(row):\n row.episode, x = ('bad', 1)\n return x",
                   "def f(row):\n row[0], x = (2, 1)\n return x"):
        with pytest.raises(ValueError):
            replay.validate_safe(source, "f", {})
