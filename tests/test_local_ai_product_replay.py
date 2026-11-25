"""Frozen behavioral controls for real CurrentBest Product function repair."""
import sys
from pathlib import Path

import pytest

from tests.gates.gate_support import require_historical_commits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_ai_product_replay as replay


def test_real_product_bug_and_fixed_controls():
    require_historical_commits(replay.bench.ROOT, replay.COMMIT, f"{replay.COMMIT}^1")
    task = replay.load_task()
    assert len(task["cases"]) == 8
    assert not replay.evaluate(task["before"], task)["pass"]
    assert replay.evaluate(task["after"], task)["pass"]
    assert task["after"] not in replay.prompt_for(task)
    assert "cases" not in replay.prompt_for(task)


@pytest.mark.parametrize("source, error", [
    ('def better(new, old):\n return open("file")', "unsupported call"),
    ('def better(new, old):\n return new.__class__', "unsupported attribute"),
    ('def better(new, old):\n import os\n return []', "unsupported AST: Import"),
    ('def better(new, old):\n return new * 999999999', "unsupported AST: Mult"),
])
def test_supplement_still_rejects_unsafe_code(source, error):
    reason = "synthetic improvement"
    task = {
        "constants": {"EXTENT_IMPROVEMENT": reason},
        "cases": [([{"extent": "complete"}, {"extent": "unresolved"}], [reason])],
    }
    assert replay.evaluate(source, task) == {"pass": False, "error": error}
