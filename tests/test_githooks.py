"""Tracked git hooks (scripts/githooks): staged-diff secret scan and the English-subject rule."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "scripts" / "githooks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HOOKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pre_commit = _load("pre_commit")
commit_msg = _load("commit_msg")

TAB = "\t"
# Fixtures are assembled at run time so this file itself never trips the scan.
COOKIE_LINE = TAB.join([".example.com", "TRUE", "/", "TRUE", "1893456000", "SID", "abc123"])
HTTPONLY_LINE = TAB.join(["#HttpOnly_.example.com", "FALSE", "/", "FALSE", "0", "SID", "abc123"])
KEY_LINE = "-----BEGIN " + "RSA PRIVATE KEY-----"
SK_LINE = "key = " + "sk-" + "a1b2c3d4e5f6a1b2c3d4e5f6"
HF_LINE = "token=" + "hf_" + "A" * 34


@pytest.mark.parametrize("line", [
    COOKIE_LINE, HTTPONLY_LINE, KEY_LINE, SK_LINE, HF_LINE,
    "Authorization: " + "Bearer abcdefghijklmnop12345",
    ('' + 'Cookie' + chr(58) + ' ') + "SID=abcdef123456; HSID=zzz",
    ('  ' + 'Set-Cookie' + chr(58) + ' ') + "session=0123456789abcdef",
    "# Netscape HTTP " + "Cookie File",
    "curl -H x Bearer " + "A" * 30,
    "AKIA" + "ABCDEFGHIJKLMNOP",
], ids=[
    "netscape-cookie",
    "httponly-cookie",
    "private-key-header",
    "api-token-sk",
    "api-token-hf",
    "bearer-header",
    "cookie-header",
    "set-cookie-header",
    "netscape-cookie-header",
    "bearer-command",
    "api-key-aws",
])
def test_secret_lines_are_found(line):
    assert pre_commit.scan_lines([line])


@pytest.mark.parametrize("line", [
    chr(9).join(['tabs', 'but', 'not', 'a', 'cookie', 'line', 'here']),
    "Authorization: <token>",
    "Authorization: {token}",
    "the Authorization header carries the token",
    "key = os.environ['OPENAI_KEY']",
    "sk-short",
    "def hf_download(model): ...",
    ('' + 'Cookie' + chr(58) + ' ') + "placeholder value  # lss-secret-ok",
    "",
])
def test_ordinary_lines_pass(line):
    assert not pre_commit.scan_lines([line])


def test_findings_name_file_and_rule_but_not_value():
    diff = "\n".join(["+++ b/scripts/x.py", "@@ -0,0 +1 @@", "+" + SK_LINE])
    findings = pre_commit.scan_diff(diff)
    assert findings == ["  scripts/x.py: API token"]
    assert "a1b2c3" not in " ".join(findings)


def test_cookie_folder_and_names_are_refused(tmp_path):
    top = str(tmp_path)
    assert pre_commit.bad_path("cookies.txt", top)
    assert pre_commit.bad_path("sub/youtube.cookies.txt", top)
    assert pre_commit.bad_path("sub\\cookies.txt", top)
    assert pre_commit.bad_path("x/yt-dlpGUI/Cookie/youtube.txt", top)
    assert not pre_commit.bad_path("docs/cookie_policy.md", top)
    assert not pre_commit.bad_path("lss/cookie_jar.py", top)


def _repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for k, v in (("user.name", "t"), ("user.email", "t@example.com"), ("core.autocrlf", "false")):
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    return tmp_path


def _run_pre_commit(repo, codex=False):
    env = {k: v for k, v in os.environ.items() if k not in ("CODEX_SESSION_ID", "CODEX_THREAD_ID")}
    if codex:
        env["CODEX_THREAD_ID"] = "test-session"
    return subprocess.run([sys.executable, "-I", str(HOOKS / "pre_commit.py")], cwd=repo, capture_output=True,
                          text=True, encoding="utf-8", env=env)


def _docs_repo(tmp_path):
    repo = _repo(tmp_path)
    (repo / "docs/scripts").mkdir(parents=True)
    (repo / "scripts/hooks").mkdir(parents=True)
    shutil.copy(HOOKS.parents[1] / "docs/scripts/check_docs.ps1", repo / "docs/scripts/check_docs.ps1")
    agents = "2ch song-like Sound alone never invents a scene LSS-owned results\nselects a usable Product capability before an execution task\n"
    (repo / "AGENTS.md").write_text(agents, encoding="utf-8")
    manifest = {"schema_version": 2, "owner_count": 2, "owners": [
        {"path": "AGENTS.md", "class": "entrypoint", "sha256": hashlib.sha256((repo / "AGENTS.md").read_bytes()).hexdigest()}]}
    (repo / "docs/DOCS_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    rules = {"rules": [{"id": "duplicate", "mechanisms": [], "applies_to": ["codex"],
                        "enforced_for": ["codex"], "not_restated": ["duplicate rule phrase"]}]}
    (repo / "scripts/hooks/RULES.json").write_text(json.dumps(rules), encoding="utf-8")
    for name in ('lss-dev-loop', 'lss-local-draft'):
        path = Path('.agents/skills') / name / 'SKILL.md'
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(HOOKS.parents[1] / path, repo / path)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    return repo


@pytest.mark.skipif(not os.environ.get("LSS_GATE_PROBE"), reason="explicit bounded diagnostic only")
def test_codex_docs_gate_diagnostic(tmp_path):
    repo = _docs_repo(tmp_path)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    powershell = shutil.which("powershell")
    pwsh = shutil.which("pwsh")
    print("powershell_path:", powershell, "pwsh_path:", pwsh,
          "PSModulePath:", repr(os.environ.get("PSModulePath")), flush=True)
    commands = [("powershell_sentinel", [powershell, "-NoProfile", "-NonInteractive", "-Command",
                                          "Write-Output 'LSS_GATE_SENTINEL'"])]
    for label, engine in (("powershell_checker", powershell), ("pwsh_checker", pwsh)):
        if engine:
            commands.append((label, [engine, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                     "-File", str(repo / "docs/scripts/check_docs.ps1"), "-RepoRoot", str(repo)]))
    for label, command in commands:
        result = subprocess.run(command, cwd=repo, capture_output=True, encoding="utf-8", errors="replace",
                                creationflags=flags, timeout=900)
        print(label, "returncode:", result.returncode, "stdout:", repr(result.stdout),
              "stderr:", repr(result.stderr), flush=True)
    result = _run_pre_commit(repo, codex=True)
    print("actual_gate returncode:", result.returncode, "stdout:", repr(result.stdout),
          "stderr:", repr(result.stderr), flush=True)


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")), reason="requires PowerShell")
def test_codex_docs_gate_staged_and_non_codex(tmp_path):
    repo = _docs_repo(tmp_path)
    assert _run_pre_commit(repo, codex=True).returncode == 0
    agents = repo / "AGENTS.md"
    valid = agents.read_text(encoding="utf-8")
    agents.write_text(valid + "duplicate rule phrase\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "AGENTS.md"], check=True)
    assert _run_pre_commit(repo).returncode == 0
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode != 0
    assert "Rule restated" in result.stdout + result.stderr
    agents.write_text(valid, encoding="utf-8")
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode != 0
    assert "Rule restated" in result.stdout + result.stderr  # unstaged repair cannot hide staged violation


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")), reason="requires PowerShell")
def test_staged_docs_ignore_unrelated_working_changes_without_mutating_them(tmp_path):
    repo = _docs_repo(tmp_path)
    agents = repo / 'AGENTS.md'
    agents.write_text(agents.read_text(encoding='utf-8') + 'duplicate rule phrase\n', encoding='utf-8')
    before = agents.read_bytes()
    index_before = subprocess.check_output(['git', 'ls-files', '--stage'], cwd=repo)
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert agents.read_bytes() == before
    assert subprocess.check_output(['git', 'ls-files', '--stage'], cwd=repo) == index_before


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")), reason="requires PowerShell")
@pytest.mark.parametrize("name,anchor", [
    ('lss-dev-loop', 'Choose the usable result before the task'),
    ('lss-local-draft', 'delegation unit, not the Product objective'),
])
def test_skill_only_commit_checks_staged_capability_instructions(tmp_path, name, anchor):
    repo = _docs_repo(tmp_path)
    subprocess.run(['git', 'commit', '--no-verify', '-qm', 'baseline'], cwd=repo, check=True)
    relative = f'.agents/skills/{name}/SKILL.md'
    skill = repo / relative
    valid = skill.read_bytes()
    skill.write_bytes(valid.replace(anchor.encode(), b'Removed capability instruction'))
    subprocess.run(['git', 'add', relative], cwd=repo, check=True)
    # The repair is deliberately unstaged: only the actual commit content decides the gate.
    skill.write_bytes(valid + b'\n<!-- Instruction retained. -->\n')
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode != 0 and anchor in result.stdout + result.stderr
    subprocess.run(['git', 'add', relative], cwd=repo, check=True)
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run(['git', 'rm', '--cached', '-q', relative], cwd=repo, check=True)
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode != 0 and relative in result.stdout + result.stderr


@pytest.mark.skipif(not (shutil.which("pwsh") or shutil.which("powershell")), reason="requires PowerShell")
def test_staged_doc_reference_cannot_be_satisfied_by_untracked_file(tmp_path):
    repo = _docs_repo(tmp_path)
    agents = repo / 'AGENTS.md'
    agents.write_text(agents.read_text(encoding='utf-8') + '`scripts/untracked.py`\n', encoding='utf-8')
    missing = repo/'scripts/untracked.py'
    missing.write_text('# present only in worktree\n', encoding='utf-8')
    manifest_path = repo/'docs/DOCS_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['owners'][0]['sha256'] = hashlib.sha256(agents.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    subprocess.run(['git','add','AGENTS.md','docs/DOCS_MANIFEST.json'],cwd=repo,check=True)
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode != 0
    assert 'Dead path reference' in result.stdout + result.stderr
    subprocess.run(['git','add','scripts/untracked.py'],cwd=repo,check=True)
    result = _run_pre_commit(repo, codex=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pre_commit_blocks_staged_secret_and_passes_clean(tmp_path):
    repo = _repo(tmp_path)
    (repo / "ok.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "ok.txt"], check=True)
    assert _run_pre_commit(repo).returncode == 0
    (repo / "bad.txt").write_text(COOKIE_LINE + "\n", encoding="utf-8", newline="")
    subprocess.run(["git", "-C", str(repo), "add", "bad.txt"], check=True)
    result = _run_pre_commit(repo)
    assert result.returncode == 1
    assert "bad.txt: Netscape cookie line" in result.stderr
    assert "abc123" not in result.stderr


def test_pre_commit_ignores_removed_secret_lines(tmp_path):
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text(KEY_LINE + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "--no-verify", "-m", "seed"], check=True)
    (repo / "f.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "f.txt"], check=True)
    assert _run_pre_commit(repo).returncode == 0


JA_STOP = "ストップゲート"  # katakana
JA_BODY = "本文は日本語"
JA_TITLE = "日本語の件名"
JA_NAME = "ソラト"
JA_FIX = "修正"


@pytest.mark.parametrize("message,bad", [
    ("Add stop gate\n\nbody", False),
    (f"Add stop gate\n\n{JA_BODY}", False),
    (JA_STOP, True),
    (f"Add gate for {JA_STOP}", True),
    (f"Fix title match for `{JA_NAME}` titles", False),
    (f'Name the target "{JA_NAME}" in the fixture', False),
    (f"Name '{JA_NAME}' and then {JA_FIX}", True),
    ("# comment first\nFix thing", False),
    (f"# comment first\n{JA_FIX}", True),
    ("", False),
])
def test_commit_message_subject_rule(message, bad):
    assert bool(commit_msg.violation(message)) is bad


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh (Git for Windows)")
def test_shell_wrapper_runs_commit_msg(tmp_path):
    ok, bad = tmp_path / "ok.txt", tmp_path / "bad.txt"
    ok.write_text("Fine subject\n", encoding="utf-8")
    bad.write_text(JA_TITLE + "\n", encoding="utf-8")
    hook = HOOKS / "commit-msg"
    repo = HOOKS.parents[1]

    def run(f):
        return subprocess.run(["sh", str(hook), str(f)], cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace")

    assert run(ok).returncode == 0
    assert run(bad).returncode == 1
