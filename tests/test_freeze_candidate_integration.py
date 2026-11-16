"""freeze_candidate refuses a candidate that lacks recent Product lanes unless each one is excluded with a reason."""
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_candidate.py"


def _git(repo, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _freeze(repo, commit, *extra):
    package = repo / "pkg"
    package.mkdir(exist_ok=True)
    (package / "LSS.exe").write_bytes(b"exe")
    return subprocess.run([sys.executable, str(SCRIPT), commit, "main", str(package), str(repo / f"out{len(extra)}"),
                           *extra], cwd=repo, capture_output=True, text=True)


def test_missing_product_lane_refuses_until_excluded_with_reason(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "lss").mkdir()
    (repo / "lss" / "a.py").write_text("a = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "lane")
    (repo / "lss" / "b.py").write_text("b = 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "lane work")
    _git(repo, "checkout", "-q", "-b", "docs-only", base)
    (repo / "notes.md").write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "docs")

    refused = _freeze(repo, base)
    assert refused.returncode != 0 and "not integrated: lane" in refused.stderr and "docs-only" not in refused.stderr
    assert _freeze(repo, base, "--exclude", "lane=").returncode != 0  # an empty reason is no reason
    reasons = repo / "reasons.json"
    reasons.write_text(json.dumps({"lane": "HELD: waits for the rendered recheck"}), encoding="utf-8")
    done = _freeze(repo, base, "--exclude-file", str(reasons))
    assert done.returncode == 0, done.stderr
    manifest = json.loads((repo / "out2" / "package-manifest.json").read_text(encoding="utf-8"))
    assert manifest["integration"]["excluded"]["lane"]["reason"].startswith("HELD")
