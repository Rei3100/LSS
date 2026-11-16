"""User-layer drift checks: common instructions and the Codex permission profile."""
from pathlib import Path

import pytest

from scripts import env_check

PRINCIPLE = ("Irreversible or outward-facing actions (deleting data, publishing, pushing, deploying, "
             "sending) need the user's word, unless the user has delegated that class of action for a "
             "project. A delegation is recorded in that project's instructions and enforced by its gates "
             "and permissions; it never widens past them.")
PROFILE = """default_permissions = "work"
approval_policy = "never"

[windows]
sandbox = "elevated"

[permissions.work]
extends = ":workspace"

[permissions.work.network]
enabled = true
"""


def home(tmp_path: Path, *, claude="@~/.codex/AGENTS.md\n", agents=None, config=PROFILE) -> Path:
    if claude is not None:
        (tmp_path / ".claude").mkdir(exist_ok=True)
        (tmp_path / ".claude" / "CLAUDE.md").write_text(claude, encoding="utf-8")
    if agents is None:
        agents = "# Working rules\n\n- " + PRINCIPLE + "\n"
    if agents is not False:
        (tmp_path / ".codex").mkdir(exist_ok=True)
        (tmp_path / ".codex" / "AGENTS.md").write_text(agents, encoding="utf-8")
    if config is not None:
        (tmp_path / ".codex").mkdir(exist_ok=True)
        (tmp_path / ".codex" / "config.toml").write_text(config, encoding="utf-8")
    return tmp_path


def test_complete_user_layer_passes(tmp_path):
    assert env_check.user_layer_checks(home(tmp_path)) == {
        "global_instruction_injection": "PASS",
        "global_common_instructions": "PASS",
        "sandbox_profile": "PASS",
    }


def test_missing_home_or_files_are_unknown(tmp_path):
    assert set(env_check.user_layer_checks(None).values()) == {"UNKNOWN"}
    assert set(env_check.user_layer_checks(tmp_path).values()) == {"UNKNOWN"}


@pytest.mark.parametrize("claude", ["", "Some other text\n", "@~/.codex/OTHER.md\n"])
def test_claude_file_without_the_import_fails(tmp_path, claude):
    assert env_check.user_layer_checks(home(tmp_path, claude=claude))["global_instruction_injection"] == "FAIL"


def test_import_line_may_follow_blank_lines_and_a_bom(tmp_path):
    result = env_check.user_layer_checks(home(tmp_path, claude="﻿\n\n@~/.codex/AGENTS.md\r\n"))
    assert result["global_instruction_injection"] == "PASS"


def test_common_instructions_without_the_principle_fail(tmp_path):
    weakened = "- " + PRINCIPLE.replace("need the user's word", "may proceed") + "\n"
    assert env_check.user_layer_checks(home(tmp_path, agents=weakened))["global_common_instructions"] == "FAIL"


@pytest.mark.parametrize("config", [
    'sandbox_mode = "workspace-write"\n' + PROFILE,
    PROFILE + "\n[sandbox_workspace_write]\nnetwork_access = true\n",
    PROFILE.replace('default_permissions = "work"', 'default_permissions = "missing"'),
    PROFILE.replace('sandbox = "elevated"', 'sandbox = "unelevated"'),
    "default_permissions = [\n",
])
def test_mixed_legacy_settings_missing_profile_or_weak_sandbox_fail(tmp_path, config):
    assert env_check.user_layer_checks(home(tmp_path, config=config))["sandbox_profile"] == "FAIL"


@pytest.mark.parametrize("config", ['approval_policy = "never"\n', 'default_permissions = ":workspace"\n'])
def test_no_named_profile_is_unknown(tmp_path, config):
    assert env_check.user_layer_checks(home(tmp_path, config=config))["sandbox_profile"] == "UNKNOWN"


def test_values_never_appear_in_the_result(tmp_path):
    result = env_check.user_layer_checks(home(tmp_path, config=PROFILE.replace("work", "secret-name-x")))
    assert "secret-name-x" not in repr(result)
