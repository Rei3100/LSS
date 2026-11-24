"""Read-only drift detection. Output status codes, never configuration values.

Optional installations are UNKNOWN, malformed or contradictory installed settings
are FAIL. Of the user's home, only the common instruction files and the Codex
config.toml are read; no sandbox log, credential file or remote service is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path = ROOT, environ=None) -> dict:
    env = os.environ if environ is None else environ
    checks = {}
    settings = {}
    for name in (".claude/settings.json", ".codex/hooks.json", ".codex/config.toml"):
        path = root / name
        try:
            raw = path.read_text(encoding="utf-8-sig")
            value = tomllib.loads(raw) if name.endswith(".toml") else json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("Settings must be an object")
            settings[name] = value
            checks[name] = "PASS"
        except FileNotFoundError:
            checks[name] = "UNKNOWN"
        except (OSError, ValueError):
            checks[name] = "FAIL"
    for engine, name in (("claude", ".claude/settings.json"), ("codex", ".codex/hooks.json")):
        hooks = settings.get(name, {}).get("hooks", {})
        if not isinstance(hooks, dict) or not isinstance(hooks.get("PreToolUse", []), list):
            checks[engine + "_hook"] = "FAIL"
            continue
        entries = hooks.get("PreToolUse", [])
        commands = [h.get("command", "") for entry in entries if isinstance(entry, dict)
                    for h in entry.get("hooks", []) if isinstance(h, dict)]
        checks[engine + "_hook"] = ("UNKNOWN" if name not in settings else
            "PASS" if any("py -3 -I" in c and "scripts/hooks/" in c for c in commands) else "FAIL")
    if checks.get("codex_hook") == "PASS":
        # Codex runs every shell command through its code-mode host, where a PreToolUse hook cannot block
        # (measured with Codex CLI 0.172.0 on Windows). The file is checked, but Codex's secret protection
        # rests on the profile's deny-read and the project rules, not on this hook.
        checks["codex_hook"] = "NOT_RELIED"
    permissions = settings.get(".claude/settings.json", {}).get("permissions", {})
    if not isinstance(permissions, dict):
        permissions = {}
        checks[".claude/settings.json"] = "FAIL"
    denies = permissions.get("deny", [])
    if not isinstance(denies, list) or any(not isinstance(d, str) for d in denies):
        denies = []
        checks[".claude/settings.json"] = "FAIL"
    checks["project_allow_deny"] = ("PASS" if permissions.get("allow") and denies else
                                      "FAIL" if permissions else "UNKNOWN")
    checks["administrative_deny"] = ("PASS" if all(any(cmd in d for d in denies)
        for cmd in ("gh api", "gh repo edit", "gh secret", "gh ruleset", "gh repo delete")) else "UNKNOWN")
    for name in ("LSS_PRIVATE_ROOT", "LSS_DEV_STATE"):
        value = env.get(name)
        if not value:
            checks[name] = "UNKNOWN"
            continue
        try:
            path = Path(value).expanduser()
            checks[name] = ("FAIL" if not path.is_absolute() else
                            "PASS" if path.resolve(strict=True).is_dir() else "FAIL")
        except FileNotFoundError:
            checks[name] = "UNKNOWN"
        except (OSError, RuntimeError):
            checks[name] = "FAIL"
    checks["state_junction"] = "FAIL" if (root / ".state").is_symlink() or (
        (root / ".state").exists() and (root / ".state").resolve() != root.resolve() / ".state") else "PASS"
    private = env.get("LSS_PRIVATE_ROOT")
    folder = Path(private) / "bin" if private and Path(private).is_absolute() else None
    checks["gate_manifest"] = "UNKNOWN"
    if folder is not None and (folder / "gates-manifest.json").is_file():
        try:
            manifest = json.loads((folder / "gates-manifest.json").read_text(encoding="utf-8-sig"))
            files = manifest["files"]
            if manifest.get("schema_version") != 1 or not isinstance(files, list) or not files:
                raise ValueError("Empty gate manifest")
            ok = True
            seen = set()
            for item in files:
                name, digest = item["path"], item["sha256"]
                # No manifest may redirect this reader to unrelated private files.
                if (not isinstance(name, str) or not name or "\\" in name
                        or Path(name).is_absolute() or ":" in name
                        or any(p in ("", ".", "..") for p in name.split("/")) or name in seen):
                    raise ValueError("Invalid gate path")
                seen.add(name)
                ok = (hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest) and ok
                try:
                    upstream = subprocess.run(["git", "-C", str(root), "show", "origin/main:gates/" + name],
                        check=True, capture_output=True, timeout=10).stdout
                    if name in ("lss-gh.cmd", "deploy-gated.cmd"):
                        upstream = upstream.decode("utf-8-sig").replace(
                            "__LSS_GATE_PYTHON__", manifest["python"]).encode("utf-8")
                    checks["gate_main_hashes"] = ("PASS" if checks.get("gate_main_hashes", "PASS") == "PASS"
                        and hashlib.sha256(upstream).hexdigest() == digest else "FAIL")
                except (OSError, subprocess.SubprocessError):
                    checks.setdefault("gate_main_hashes", "UNKNOWN")
            checks["gate_manifest"] = "PASS" if ok else "FAIL"
        except (OSError, ValueError, KeyError, TypeError):
            checks["gate_manifest"] = "FAIL"
    checks["main_ci_jobs"] = "UNKNOWN"
    try:
        raw = subprocess.run(["git", "-C", str(root), "show", "origin/main:.github/workflows/ci.yml"],
                             capture_output=True, check=True, timeout=10).stdout.decode()
        checks["main_ci_jobs"] = "PASS" if all("  " + job + ":" in raw for job in
                                                      ("classify", "review-gate", "result")) else "FAIL"
    except (OSError, UnicodeError, subprocess.SubprocessError):
        pass
    home = env.get("USERPROFILE") or env.get("HOME")
    checks.update(user_layer_checks(Path(home) if home and Path(home).is_absolute() else None))
    for name in ("user_auto_mode", "gate_acl", "ruleset_expected_source", "repo_visibility",
                 "external_creation_restrictions", "automatic_reviews", "secret_read_denies",
                 "runtime_hook_execution", "state_guard_runtime"):
        checks[name] = "UNKNOWN"
    rules = root / ".codex/rules/lss.rules"
    checks["codex_send_rules"] = "UNKNOWN"
    if rules.is_file():
        checks["codex_send_rules"] = "PASS" if 'decision = "forbidden"' in rules.read_text(encoding="utf-8") else "FAIL"
    return {"status": "FAIL" if "FAIL" in checks.values() else "UNKNOWN" if "UNKNOWN" in checks.values() else "PASS",
            "checks": dict(sorted(checks.items()))}


PRINCIPLE = ("Irreversible or outward-facing actions (deleting data, publishing, pushing, deploying, "
             "sending) need the user's word, unless the user has delegated that class of action for a "
             "project. A delegation is recorded in that project's instructions and enforced by its gates "
             "and permissions; it never widens past them.")


def user_layer_checks(home) -> dict:
    """Common instructions and the Codex permission profile; status codes only, never values."""
    checks = dict.fromkeys(("global_instruction_injection", "global_common_instructions",
                            "sandbox_profile"), "UNKNOWN")
    if home is None:
        return checks

    def read(*parts):
        try:
            return Path(home, *parts).read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError):
            return ""

    claude = read(".claude", "CLAUDE.md")
    if claude is not None:
        first = next((line.strip() for line in claude.splitlines() if line.strip()), "")
        checks["global_instruction_injection"] = "PASS" if first == "@~/.codex/AGENTS.md" else "FAIL"
    agents = read(".codex", "AGENTS.md")
    if agents is not None:
        checks["global_common_instructions"] = "PASS" if PRINCIPLE in agents else "FAIL"
    config = read(".codex", "config.toml")
    if config is not None:
        try:
            data = tomllib.loads(config)
        except tomllib.TOMLDecodeError:
            data = None
        name = data.get("default_permissions") if isinstance(data, dict) else None
        if data is None:
            checks["sandbox_profile"] = "FAIL"
        elif isinstance(name, str) and not name.startswith(":"):
            profiles = data.get("permissions")
            windows = data.get("windows", {})
            sandbox = windows.get("sandbox") if isinstance(windows, dict) else "invalid"
            legacy = "sandbox_mode" in data or "sandbox_workspace_write" in data
            missing = not isinstance(profiles, dict) or not isinstance(profiles.get(name), dict)
            weak = sandbox is not None and sandbox not in ("elevated", "mxc")
            checks["sandbox_profile"] = "FAIL" if legacy or missing or weak else "PASS"
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    result = check(args.repo)
    print(json.dumps(result, sort_keys=True))
    return int(result["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
