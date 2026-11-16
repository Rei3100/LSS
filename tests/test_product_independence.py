"""LSS's results must come from LSS itself (AGENTS.md "Non-negotiable meaning": Product independence).

Nothing a developer or an agent looked up, judged, curated or wrote about specific works may ship as Product
knowledge: no per-work or per-episode tables, no hard-coded work names or file-name patterns in runtime code,
no answers in prompts or examples. These checks fail on any new instance. KNOWN_VIOLATIONS lists what is
already shipped and must be removed (replaced by what LSS acquires or measures itself); it may only shrink.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "lss"

# Private publication terms are supplied by the maintainer at runtime.
WORK_TERMS: tuple[str, ...] | None = None
GENERIC_EXCEPTIONS: set[str] = set()


def _work_terms() -> tuple[str, ...]:
    if WORK_TERMS is not None:
        return WORK_TERMS
    import os
    import pytest

    configured = os.environ.get("LSS_PUBLISH_DENYLIST")
    private_root = os.environ.get("LSS_PRIVATE_ROOT")
    path = Path(configured) if configured else (Path(private_root) / "publish-denylist.txt" if private_root else None)
    if path is None or not path.is_file():
        pytest.skip("Set LSS_PUBLISH_DENYLIST or LSS_PRIVATE_ROOT to supply the private publication term list")
    lines = (line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines())
    terms = tuple(line[len("word:"):].strip() if line.startswith("word:") else line
                  for line in lines if line and not line.startswith("#"))
    assert terms, "The configured publication term list must not be empty"
    return terms


KNOWN_VIOLATIONS: dict[str, str] = {}


def _runtime_strings(path: Path) -> list[tuple[int, str]]:
    """String constants that are not docstrings (docstrings/comments may cite Evidence)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                docstrings.add(id(first.value))
    return [(node.lineno, node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings]


def test_runtime_code_names_no_specific_work():
    terms = _work_terms()
    found = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in KNOWN_VIOLATIONS:
            continue
        for line, value in _runtime_strings(path):
            for number, term in enumerate(terms):
                if term in GENERIC_EXCEPTIONS:
                    continue
                # a Latin term is a word of its own (a short name must not match inside an API identifier);
                # a katakana term must not be part of a longer katakana word
                edge = "A-Za-z0-9" if term.isascii() else "ァ-ヴー"
                if re.search(rf"(?<![{edge}]){re.escape(term)}(?![{edge}])", value, re.I):
                    # the message names the list entry by number: private terms never reach a log
                    found.append(f"{rel}:{line}: private term #{number}")
    assert not found, "work-specific knowledge in Product code:\n" + "\n".join(found)


def test_shipped_data_holds_no_per_work_tables():
    terms = _work_terms()
    found = []
    for path in ROOT.rglob("*"):
        if path.suffix.lower() not in (".json", ".tsv", ".csv", ".txt") or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in KNOWN_VIOLATIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [f"#{number}" for number, term in enumerate(terms) if term not in GENERIC_EXCEPTIONS and term in text]
        if hits:
            found.append(f"{rel}: {hits[:5]}")
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if isinstance(data, dict) and isinstance(data.get("episodes"), dict):
                found.append(f"{rel}: per-episode table")
    assert not found, "per-work data shipped in the Product:\n" + "\n".join(found)


def test_the_checks_catch_a_planted_violation(tmp_path, monkeypatch):
    import sys
    import pytest

    module = sys.modules[__name__]
    (tmp_path / "bgm").mkdir()
    (tmp_path / "bgm" / "names.py").write_text(
        '"""Docstring may cite カナモリ Evidence."""\n'
        'EXAMPLES = ["カナモリ捜索作戦"]\nPATTERN = r"series-synthetic-(\\d+)"\n', encoding="utf-8")
    (tmp_path / "bgm" / "table.json").write_text(json.dumps({"episodes": {"1": []}}), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "WORK_TERMS", ("カナモリ", "series-synthetic"))
    monkeypatch.setattr(module, "KNOWN_VIOLATIONS", {})
    with pytest.raises(AssertionError) as code:
        test_runtime_code_names_no_specific_work()
    message = str(code.value)
    assert "names.py:2: private term #0" in message and "names.py:3: private term #1" in message
    assert "カナモリ" not in message and "Docstring" not in message
    with pytest.raises(AssertionError) as data:
        test_shipped_data_holds_no_per_work_tables()
    assert "per-episode table" in str(data.value)


def test_known_violations_only_shrink():
    # Every listed violation must still exist (remove it from the list when it is removed from the Product).
    missing = [rel for rel in KNOWN_VIOLATIONS if not (ROOT / rel).exists()]
    assert not missing, f"remove these from KNOWN_VIOLATIONS (they are gone): {missing}"
