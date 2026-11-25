"""Git helpers for history checks and disposable repositories used in tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "gates" / "public_guard.py"
MIGRATE = ROOT / "gates" / "migrate_to_public.py"


def require_historical_commits(repo: Path, *revisions: str) -> None:
    import pytest

    for revision in revisions:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=repo,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            pytest.skip(f"公開リポジトリの履歴にない旧コミット {revision}")


class RepoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".gate-tests-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.env = os.environ.copy()
        self.env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                         "GIT_TERMINAL_PROMPT": "0", "PYTHONDONTWRITEBYTECODE": "1",
                         "LSS_GUARD_PYTHON": sys.executable,
                         "GIT_AUTHOR_NAME": "Synthetic Test", "GIT_COMMITTER_NAME": "Synthetic Test",
                         "GIT_AUTHOR_EMAIL": "test@users.noreply.github.com",
                         "GIT_COMMITTER_EMAIL": "test@users.noreply.github.com"})
        self.deny = self.base / "fake-denylist.txt"
        self.literal = "synthetic-" + "private-phrase"
        self.deny.write_text(self.literal + "\n", encoding="utf-8")
        self.env["LSS_PUBLISH_DENYLIST"] = str(self.deny)
        self.init(self.repo)
        self.policy = {"schema_version": 1, "include": ["VERSION", "src/*.py", "tests/*.py"],
                       "deny": [], "max_blob_bytes": 8192,
                       "reviewed_code_paths": ["VERSION"], "approved_paths": {}}
        self.write_policy()
        self.write("VERSION", b"1\n")

    def run_cmd(self, args, cwd=None):
        return subprocess.run([str(x) for x in args], cwd=cwd, env=self.env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def git(self, *args, repo=None, check=True):
        proc = self.run_cmd(["git", "-c", "safe.directory=*", "-C", repo or self.repo, *args])
        if check and proc.returncode:
            self.fail("Fixture Git operation failed: " + str(args[0]) + "; " + proc.stderr.decode(errors="replace"))
        return proc

    def init(self, path, bare=False):
        path.mkdir()
        self.git("init", "-q", "--initial-branch=main", *(["--bare"] if bare else []), repo=path)
        if not bare:
            self.git("config", "core.autocrlf", "false", repo=path)
            self.git("config", "commit.gpgsign", "false", repo=path)

    def write(self, relative, data):
        target = self.repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def write_policy(self):
        self.write(".publish-allow.json", json.dumps(self.policy, indent=2).encode())

    def admit_code(self, path, data):
        self.policy["include"].append(path)
        self.policy["reviewed_code_paths"].append(path)
        self.write(path, data)
        self.write_policy()

    def pin(self, path, data):
        target = self.write(path, data)
        oid = self.git("hash-object", str(target)).stdout.decode().strip()
        self.policy["approved_paths"][path] = {"blob_oid": oid, "reason": "Synthetic reviewed fixture"}
        self.write_policy()
        return oid

    def commit(self, message="Synthetic clean commit", paths=None, repo=None):
        self.git("add", "--", *(paths or [".publish-allow.json", "VERSION"]), repo=repo)
        self.git("-c", "core.hooksPath=disabled-test-hooks", "commit", "-qm", message, repo=repo)
        return self.git("rev-parse", "HEAD", repo=repo).stdout.decode().strip()

    def guard(self, *args):
        proc = self.run_cmd([sys.executable, "-I", "-B", GUARD, "--repo", self.repo, *args])
        try:
            result = json.loads(proc.stdout)
        except ValueError:
            self.fail("Guard did not produce JSON: " + proc.stderr.decode(errors="replace"))
        return proc, result

    def blocked(self, rule):
        proc, result = self.guard()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(rule, result["counts"])
        return proc, result
