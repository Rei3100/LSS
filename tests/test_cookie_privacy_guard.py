"""cookie_privacy_guard: cookie values under yt-dlpGUI\\Cookie are never read; listing and path-passing stay free."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "cookie_privacy_guard.py"
sys.path.insert(0, str(HOOK.parent))
_spec = importlib.util.spec_from_file_location("cookie_privacy_guard", HOOK)
guard = importlib.util.module_from_spec(_spec)
sys.modules["cookie_privacy_guard"] = guard
_spec.loader.exec_module(guard)
TOOL_ROOT = "Q" + chr(58) + chr(92) + "synthetic-tools"
DIR = chr(92).join((TOOL_ROOT, "yt-dlpGUI", "Cookie"))
COOKIE = DIR + chr(92) + "youtube.txt"
POSIX_DIR = "/" + DIR[0].lower() + DIR[2:].replace(chr(92), "/")


@pytest.fixture(autouse=True)
def synthetic_cookie_root(monkeypatch):
    monkeypatch.setenv("LSS_COOKIE_ROOT", DIR)


def _decision(tool, tool_input):
    return "deny" if guard.blocked({"tool_name": tool, "tool_input": tool_input}) else None


def _decision_subprocess(tool, tool_input):
    event = {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}
    proc = subprocess.run([sys.executable, "-I", str(HOOK)], input=json.dumps(event).encode(), capture_output=True)
    if proc.returncode == 2:
        assert proc.stderr.strip()
        return "deny"
    assert proc.returncode == 0
    out = proc.stdout.decode().strip()
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"] if out else None


@pytest.mark.parametrize("tool,tool_input", [
    ("Read", {"file_path": COOKIE}),
    ("Read", {"file_path": (DIR.replace(chr(92), "/") + '/niconico.txt')}),
    ("Read", {"file_path": (POSIX_DIR + '/cookies.txt')}),
    ("Grep", {"pattern": "SID", "path": DIR}),
    ("Grep", {"pattern": "SID", "path": COOKIE}),
    ("Grep", {"pattern": "SID", "path": TOOL_ROOT}),
])
def test_read_and_grep_tools_denied(tool, tool_input):
    assert _decision(tool, tool_input) == "deny"


@pytest.mark.parametrize("command", [
    f"cat {COOKIE}", f"type {COOKIE}", f"Get-Content {COOKIE}", f"gc '{DIR}\\niconico.txt'",
    f"head -n 3 {DIR}/youtube.txt", f"tail {DIR}/youtube.txt", f"Select-String SID {DIR}\\*", f"grep -r SID {DIR}",
    f"rg SID {DIR}", f"findstr SID {COOKIE}", f"Get-Content -Path {COOKIE} -TotalCount 2",
    f"cat {POSIX_DIR}/youtube.txt", f"sed -n 1,3p {COOKIE}",
    f"$c = '{COOKIE}'; Get-Content $c", f"c='{COOKIE}'; cat \"$c\"",
    f"Get-ChildItem {DIR} | Select-String SID", f"dir {DIR} | ForEach-Object {{ Get-Content $_.FullName }}",
    f"python -c \"print(open(r'{COOKIE}').read())\"",
    f"python - <<'EOF'\nprint(open(r'{COOKIE}').read())\nEOF",
    f"powershell -NoProfile -Command \"Get-Content '{COOKIE}'\"",
    f"echo $(cat {COOKIE})",
    f"[IO.File]::ReadAllText('{COOKIE}')",
    f"cat < {COOKIE}",
])
def test_shell_content_verbs_denied(command):
    assert _decision("Bash", {"command": command}) == "deny", command
    assert _decision("PowerShell", {"command": command}) == "deny", command


@pytest.mark.parametrize("command", [
    f"ls {DIR}", f"dir {DIR}", f"Get-ChildItem {DIR}", f"Test-Path {COOKIE}", f"Get-Item {COOKIE}",
    f"Get-ChildItem {DIR} | Select-Object Name, Length", f"ls -la {DIR} | wc -l",
    f"Get-FileHash {COOKIE}",
    f"yt-dlp --cookies {COOKIE} https://example.invalid/v",
    f"$env:LSS_COOKIES_FILE_YOUTUBE='{COOKIE}'; python scripts/probe.py",
    f"python -c \"import os; print(os.path.exists(r'{COOKIE}'))\"",
    f"python -c \"import yt_dlp; yt_dlp.YoutubeDL({{'cookiefile': r'{COOKIE}'}})\"",
    "cat docs/dev/AGENT_PLAYBOOK.md", "Get-Content README.md", "grep -rn cookie lss/",
    "git commit -m 'document the yt-dlpGUI Cookie rule'",
])
def test_listing_and_path_passing_allowed(command):
    assert _decision("Bash", {"command": command}) is None, command
    assert _decision("PowerShell", {"command": command}) is None, command


def test_other_reads_and_greps_allowed(tmp_path):
    assert _decision("Read", {"file_path": str(tmp_path / "source" / 'AGENTS.md')}) is None
    assert _decision("Grep", {"pattern": "cookie", "path": str(tmp_path / "source" / 'lss')}) is None
    assert _decision("Grep", {"pattern": "cookie"}) is None
    assert _decision("Glob", {"pattern": "**/*", "path": DIR}) == "deny"


def test_hook_process_end_to_end():
    assert _decision_subprocess("Read", {"file_path": COOKIE}) == "deny"
    assert _decision_subprocess("Bash", {"command": f"cat {COOKIE}"}) == "deny"
    assert _decision_subprocess("Bash", {"command": f"ls {DIR}"}) is None


def test_garbage_input_denies():
    proc = subprocess.run([sys.executable, "-I", str(HOOK)], input=b"nope", capture_output=True)
    assert proc.returncode == 2 and proc.stderr.strip()


def _run_hook(hook, event):
    return subprocess.run([sys.executable, "-I", "-B", str(hook)],
                          input=json.dumps(event).encode(), capture_output=True, timeout=15)


def _fault_copy(tmp_path, scanner):
    """Inject faults only into a disposable synthetic copy, never via production switches."""
    hook = tmp_path / HOOK.name
    hook.write_bytes(HOOK.read_bytes())
    (tmp_path / "RULES.json").write_bytes(HOOK.with_name("RULES.json").read_bytes())
    (tmp_path / "shell_scan.py").write_text(scanner, encoding="utf-8")
    return hook


@pytest.mark.parametrize("scanner", [
    "raise RuntimeError('synthetic startup failure')",
    "def norm_path(*args):\n    raise RuntimeError('synthetic matcher failure')\n",
    "import time\ndef norm_path(*args):\n    time.sleep(60)\n",
])
def test_worker_startup_exception_and_timeout_deny(tmp_path, scanner):
    hook = _fault_copy(tmp_path, scanner)
    proc = _run_hook(hook, {"tool_name": "Read", "tool_input": {"file_path": "README.md"}})
    assert proc.returncode == 2
    assert b"Access denied" in proc.stderr
    assert b"synthetic" not in proc.stderr


@pytest.mark.parametrize("raw", [b"", b"null", b"[]", b"{}", b"\xff", b'{"tool_name":"Read"}'])
def test_invalid_event_denies(raw):
    proc = subprocess.run([sys.executable, "-I", "-B", str(HOOK)], input=raw,
                          capture_output=True, timeout=15)
    assert proc.returncode == 2 and proc.stderr.strip()


def test_synthetic_files_and_codex_inputs(tmp_path, monkeypatch):
    public = tmp_path / "source" / "ordinary.py"
    public.parent.mkdir()
    public.write_text("# ordinary synthetic source\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    secret = private / "synthetic.txt"
    secret.write_text("synthetic fixture only\n", encoding="utf-8")
    monkeypatch.setenv("LSS_PRIVATE_ROOT", str(tmp_path))
    # The configured secret directory is synthetic and never refers to a real credential store.
    secrets = tmp_path / "secrets"
    private.rename(secrets)
    secret = secrets / secret.name
    assert _run_hook(HOOK, {"tool_name": "Read", "tool_input": {"file_path": str(public)}}).returncode == 0
    for tool, args in [
        ("Read", {"file_path": str(secret)}),
        ("Edit", {"file_path": str(secret)}),
        ("Grep", {"path": str(secrets)}),
        ("Glob", {"path": str(secrets), "pattern": "**/*"}),
        ("Bash", {"command": f'cat "{secret}"'}),
        ("PowerShell", {"command": f'Get-Content "{secret}"'}),
        ("exec_command", {"cmd": f'type "{secret}"'}),
        ("shell", {"command": ["cmd", "/c", "type", str(secret)]}),
    ]:
        assert _run_hook(HOOK, {"tool_name": tool, "tool_input": args}).returncode == 2


@pytest.mark.parametrize("target", ["cookies.txt", "browser/Network/Cookies", ".env", "auth.json",
                                    "credentials", "Login Data", "key4.db"])
def test_cookie_and_credential_names_denied(tmp_path, target):
    assert _decision("Read", {"file_path": str(tmp_path / target)}) == "deny"


def test_missing_policy_denies(tmp_path):
    hook = _fault_copy(tmp_path, "# scanner not used\n")
    (tmp_path / "RULES.json").unlink()
    assert _run_hook(hook, {"tool_name": "Read", "tool_input": {"file_path": "README.md"}}).returncode == 2


def test_windows_registration_startup_failure_denies(tmp_path):
    settings = json.loads((HOOK.parents[2] / ".codex" / "hooks.json").read_text())
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["commandWindows"]
    # No script exists in this synthetic cwd: this tests the registration's shell fallback.
    proc = subprocess.run(command, cwd=tmp_path, input=b"{}", capture_output=True, timeout=15)
    assert proc.returncode == 2
    assert b"Access denied" in proc.stderr


def test_windows_registration_allows_source_and_denies_cookie(tmp_path):
    settings = json.loads((HOOK.parents[2] / ".codex" / "hooks.json").read_text())
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["commandWindows"]
    for path, expected in [(str(tmp_path / "ordinary.py"), 0), (COOKIE, 2)]:
        event = {"tool_name": "Read", "tool_input": {"file_path": path}}
        proc = subprocess.run(command, cwd=HOOK.parents[2], input=json.dumps(event).encode(),
                              capture_output=True, timeout=15)
        assert proc.returncode == expected, proc.stderr.decode()
