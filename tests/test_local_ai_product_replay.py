"""Frozen behavioral controls for real CurrentBest Product function repair."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_ai_product_replay as replay


def test_real_product_bug_and_fixed_controls():
    task = replay.load_task()
    assert len(task["cases"]) == 8
    assert not replay.evaluate(task["before"], task)["pass"]
    assert replay.evaluate(task["after"], task)["pass"]
    assert task["after"] not in replay.prompt_for(task)
    assert "cases" not in replay.prompt_for(task)


@pytest.mark.parametrize("source", [
    'def better(new, old):\n return open("file")',
    'def better(new, old):\n return new.__class__',
    'def better(new, old):\n import os\n return []',
    'def better(new, old):\n return new * 999999999',
])
def test_supplement_still_rejects_unsafe_code(source):
    result = replay.evaluate(source, replay.load_task())
    assert not result["pass"]
    assert "error" in result
