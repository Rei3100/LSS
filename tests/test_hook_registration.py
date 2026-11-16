"""LSS registers one project hook, the cookie privacy guard, for Claude and Codex; every other hook entry point under
scripts/hooks would be dead code."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "scripts" / "hooks"
LIBRARY = {"shell_scan.py"}


def _names(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    commands = [hook.get("command", "") for groups in data["hooks"].values() for group in groups
                for hook in group.get("hooks", [])]
    return {name for command in commands for name in re.findall(r"scripts/hooks/([\w\-]+\.py)", command)}


def test_every_hook_entry_point_is_registered_for_claude_and_codex():
    entry_points = {p.name for p in HOOKS.glob("*.py") if p.name not in LIBRARY and "def main(" in p.read_text(encoding="utf-8")}
    assert entry_points == {"cookie_privacy_guard.py"}
    assert _names(ROOT / ".claude" / "settings.json") == entry_points
    assert _names(ROOT / ".codex" / "hooks.json") == entry_points
