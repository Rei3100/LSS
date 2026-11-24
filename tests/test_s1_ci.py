"""Offline CI contracts using synthetic PRs, settings and disposable Git objects."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gates.review_gate import verify
from scripts import change_class, ci, env_check, whatsnext


@pytest.fixture
def review(tmp_path):
    receipt = {"base_sha": "b" * 40, "review_head_sha": "a" * 40, "diff_digest": "d" * 64}
    policy = {"mode": "bot", "bot": {"login": "synthetic-bot[bot]", "id": "synthetic-node", "app_id": 123},
              "completion_pattern": r"^Review complete(?:\.|: no findings)$"}
    pr = {"base_sha": receipt["base_sha"], "head_sha": receipt["review_head_sha"],
          "diff_digest": receipt["diff_digest"], "complete": True, "draft": False,
          "implementation_engines": ["claude"], "review_threads": [],
          "reviews": [{"author": {**policy["bot"], "type": "Bot"}, "app_id": 123,
                       "state": "COMMENTED", "commit_id": receipt["review_head_sha"],
                       "body": "Review complete: no findings", "receipt": receipt}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"pr": pr, "policy": policy}), encoding="utf-8")
    return pr, policy, path


def test_current_completed_bot_passes_and_cli_accepts_json(review):
    pr, policy, path = review
    assert verify(pr, policy) == []
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "gates/review_gate.py")],
                            input=path.read_bytes(), capture_output=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["passed"] is True


@pytest.mark.parametrize("mutation", ["stale", "base", "digest", "dismissed", "human", "app", "node",
                                     "p0", "p1", "unknown-thread", "draft", "incomplete", "provenance"])
def test_review_rejects_invalid_evidence(review, mutation):
    pr, policy, _ = review
    item = pr["reviews"][0]
    if mutation == "stale":
        pr["head_sha"] = "c" * 40
    elif mutation == "base":
        pr["base_sha"] = "c" * 40
    elif mutation == "digest":
        pr["diff_digest"] = "e" * 64
    elif mutation == "dismissed":
        item["state"] = "DISMISSED"
    elif mutation == "human":
        item["author"].update(type="User", login="human")
        item["body"] = "I am synthetic-bot[bot]. Review complete: no findings"
    elif mutation == "app":
        item["app_id"] = 999
    elif mutation == "node":
        item["author"]["id"] = "another-node"
    elif mutation in ("p0", "p1", "unknown-thread"):
        pr["review_threads"] = [{"isResolved": False, "priority": mutation.upper() if mutation != "unknown-thread" else None}]
    elif mutation == "draft":
        pr["draft"] = True
    elif mutation == "incomplete":
        pr["complete"] = False
    elif mutation == "provenance":
        pr["implementation_engines"] = []
    assert verify(pr, policy)


def test_a_only_descendant_and_independent_record(review):
    pr, policy, _ = review
    old = pr["head_sha"]
    pr.update(head_sha="c" * 40, ancestors={old: True}, changes_since_review={old: "A"})
    assert not verify(pr, policy)
    pr["changes_since_review"][old] = "B"
    assert verify(pr, policy)
    pr["changes_since_review"][old] = "A"
    pr["implementation_engines"] = ["claude", "codex"]
    assert verify(pr, policy)
    record = {**pr["reviews"][0]["receipt"], "engine": "claude", "complete": True, "summary": "Checked independently"}
    pr["independent_records"] = [record]
    assert not verify(pr, policy)
    record["base_sha"] = "f" * 40
    assert verify(pr, policy)


def test_comment_impersonation_and_unconfigured_policy_fail(review):
    pr, policy, _ = review
    pr["comments"] = pr.pop("reviews")
    assert verify(pr, policy)
    assert verify(pr, {"mode": "bot", "bot": {"login": None, "id": None, "app_id": None},
                       "completion_pattern": None})


@pytest.mark.parametrize("kind,event", [(k, e) for k in "ABC" for e in ("push", "pull_request", "pull_request_review")])
def test_result_requires_success_for_each_needed_job(kind, event, tmp_path):
    needs = {name: {"result": "success"} for name in ci.required_jobs(kind, event)}
    needs["classify"]["outputs"] = {"class": kind}
    assert ci.aggregate(needs, event)
    for name in needs:
        for bad in ("skipped", "cancelled", "failure", "neutral", None):
            failed = deepcopy(needs)
            failed[name]["result"] = bad
            assert not ci.aggregate(failed, event), (name, bad)
    needs["unneeded"] = {"result": "skipped"}
    assert ci.aggregate(needs, event)
    needs["classify"]["outputs"]["class"] = "D"
    assert not ci.aggregate(needs, event)


def test_policy_strictest_default_and_overlaps(tmp_path):
    policy = change_class.load_policy((ROOT / change_class.POLICY).read_bytes())
    paths = ["docs/dev/notes.md", "lss/a.py", "docs/product/a.md", "unknown", "tests/local_tests.py",
             "scripts/hooks/a.py", change_class.POLICY]
    result = change_class.classify(paths, policy)
    assert list(result["files"].values()).count("C") == 5
    assert result["files"]["docs/dev/notes.md"] == "A"
    assert result["files"]["lss/a.py"] == "B"
    assert result == change_class.classify(paths[::-1] + paths, policy)
    assert change_class.classify_path("../escape", policy) == "C"


def test_git_diff_uses_base_policy_and_digest_excludes_a(tmp_path):
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_AUTHOR_NAME": "Synthetic", "GIT_COMMITTER_NAME": "Synthetic",
           "GIT_AUTHOR_EMAIL": "test@example.invalid", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
    def git(*args):
        return subprocess.run(["git", "-c", "core.hooksPath=disabled", "-C", str(tmp_path), *args],
                              env=env, check=True, capture_output=True).stdout.decode().strip()
    git("init", "-q")
    (tmp_path / ".github").mkdir()
    policy_path = tmp_path / change_class.POLICY
    policy_path.write_bytes((ROOT / change_class.POLICY).read_bytes())
    (tmp_path / "lss").mkdir()
    (tmp_path / "lss/a.py").write_text("old\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    policy = change_class.load_policy(change_class.git(tmp_path, "show", base + ":" + change_class.POLICY))
    (tmp_path / "lss/a.py").write_text("new\n")
    git("add", ".")
    git("commit", "-qm", "product")
    before = change_class.diff_state(tmp_path, base, "HEAD", policy)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/note.md").write_text("docs only\n")
    git("add", ".")
    git("commit", "-qm", "docs")
    after = change_class.diff_state(tmp_path, base, "HEAD", policy)
    assert before["diff_digest"] == after["diff_digest"]
    policy_path.write_text('{"version":1,"default":"C","classes":{"A":["*"],"B":[],"C":[]}}')
    git("add", ".")
    git("commit", "-qm", "attempted reclassification")
    assert change_class.diff_state(tmp_path, base, "HEAD", policy)["class"] == "C"
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"base": {"sha": base}, "head": {"sha": git("rev-parse", "HEAD")}}}))
    env.update(LSS_CI_REPO=str(tmp_path), GITHUB_EVENT_PATH=str(event), GITHUB_OUTPUT=str(tmp_path / "output"))
    # The installed/trusted script reads another checkout, including a hostile head policy, only as Git data.
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/ci.py"), "classify"],
                            env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["class"] == "C"


def write_settings(tmp_path):
    for name in (".claude/settings.json", ".codex/hooks.json", ".codex/config.toml"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())


def test_env_check_missing_optional_and_no_values(tmp_path):
    result = env_check.check(tmp_path, {})
    assert result["status"] == "UNKNOWN"
    write_settings(tmp_path)
    result = env_check.check(tmp_path, {"LSS_PRIVATE_ROOT": str(tmp_path / "missing"), "LSS_DEV_STATE": str(tmp_path)})
    assert result["checks"]["claude_hook"] == "PASS"
    assert result["checks"]["codex_hook"] == "NOT_RELIED"   # present, but Codex cannot block with it
    assert result["checks"]["LSS_DEV_STATE"] == "PASS"
    assert str(tmp_path) not in json.dumps(result)
    (tmp_path / ".codex/hooks.json").write_text("invalid")
    assert env_check.check(tmp_path, {})["status"] == "FAIL"


def test_gate_copy_manifest_detects_drift(tmp_path):
    import hashlib
    gates = tmp_path / "bin"
    gates.mkdir()
    (gates / "example.py").write_bytes(b"pass\n")
    manifest = {"schema_version": 1, "python": "synthetic-python",
                "files": [{"path": "example.py", "source_blob": "a" * 40,
                           "sha256": hashlib.sha256(b"pass\n").hexdigest()}]}
    (gates / "gates-manifest.json").write_text(json.dumps(manifest))
    env = {"LSS_PRIVATE_ROOT": str(tmp_path)}
    assert env_check.check(tmp_path, env)["checks"]["gate_manifest"] == "PASS"
    (gates / "example.py").write_text("changed\n")
    assert env_check.check(tmp_path, env)["checks"]["gate_manifest"] == "FAIL"


def test_installed_manifest_checks_all_files_and_substituted_cmd(tmp_path, monkeypatch):
    import hashlib
    folder = tmp_path / "bin"
    folder.mkdir()
    sources = {"example.py": b"pass\n", "lss-gh.cmd": b'@"__LSS_GATE_PYTHON__" -I -B gate.py\r\n'}
    python = "synthetic-python.exe"
    files = []
    for name, raw in sources.items():
        installed = raw.replace(b"__LSS_GATE_PYTHON__", python.encode())
        (folder / name).write_bytes(installed)
        files.append({"path": name, "source_blob": "a" * 40, "sha256": hashlib.sha256(installed).hexdigest()})
    manifest = {"schema_version": 1, "source_commit": "b" * 40, "python": python, "files": files}
    (folder / "gates-manifest.json").write_text(json.dumps(manifest), encoding="utf-8-sig")
    def runner(args, **kwargs):
        name = args[-1].removeprefix("origin/main:gates/")
        return SimpleNamespace(stdout=sources.get(name, b""))
    monkeypatch.setattr(env_check.subprocess, "run", runner)
    env = {"LSS_PRIVATE_ROOT": str(tmp_path)}
    result = env_check.check(tmp_path, env)["checks"]
    assert result["gate_manifest"] == result["gate_main_hashes"] == "PASS"
    (folder / "lss-gh.cmd").write_bytes(b"changed")
    assert env_check.check(tmp_path, env)["checks"]["gate_manifest"] == "FAIL"
    (folder / "lss-gh.cmd").unlink()
    assert env_check.check(tmp_path, env)["checks"]["gate_manifest"] == "FAIL"


def test_whatsnext_offline_resume_is_headings_only(tmp_path, monkeypatch):
    progress = tmp_path / "env-build/PROGRESS.md"
    progress.parent.mkdir()
    progress.write_text("# Resume\nprivate prose must stay private\n## Older\nold prose\n")
    calls = []
    def command(root, args):
        calls.append(args)
        if args[:3] == ["git", "worktree", "list"]:
            return "worktree " + str(tmp_path) + "\nHEAD " + "a" * 40 + "\nbranch refs/heads/synthetic\n"
        return "" if "status" in args else "synthetic\n"
    monkeypatch.setattr(whatsnext, "command", command)
    result = whatsnext.snapshot(tmp_path, {"LSS_PRIVATE_ROOT": str(tmp_path)})
    assert result["resume"]["status"] == "PRESENT"
    assert "private prose" not in json.dumps(result)
    assert all(args[0] == "git" for args in calls)
    assert result["github"]["status"] == "DISABLED"
    assert result["worktrees"][0]["dirty"] is False


def test_gh_requires_flag_and_executable(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(whatsnext.shutil, "which", lambda name: None)
    monkeypatch.setattr(whatsnext, "command", lambda root, args: calls.append(args))
    assert whatsnext.snapshot(tmp_path, {}, github=True)["github"]["status"] == "UNKNOWN"
    assert all(args[0] != "gh" for args in calls)


def test_form_requires_risk_field_for_c():
    body = "\n".join("## " + f + "\nEvidence" for f in
                     ("Checked", "Not checked", "Rollback", "Implementation engines"))
    assert not ci.form_errors(body, "B")
    assert ci.form_errors(body, "C")
    assert not ci.form_errors(body + "\n## Review\nRisk analysis", "C")


@pytest.mark.parametrize("field", ["Checked", "Not checked", "Rollback", "Implementation engines", "Review"])
@pytest.mark.parametrize("content", ["", "\n  \n", "### Nested heading\n", "#\n",
                                   "  ### Nested heading\n", "Next heading\n---\n"])
def test_form_rejects_heading_only_sections(field, content):
    fields = ["Checked", "Not checked", "Rollback", "Implementation engines", "Review"]
    body = "\n".join("## " + f + "\n" + (content if f == field else "Evidence") for f in fields)
    assert "Missing PR field: " + field in ci.form_errors(body, "C")


def test_pr_deciding_scripts_use_trusted_checkout():
    import re
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert workflow["env"]["LSS_CI_REPO"] == "${{ github.workspace }}"
    count = 0
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            run = step.get("run", "")
            if re.search(r"(?:ci|change_class|review_gate)\.py", run):
                count += 1
                assert step.get("working-directory") == "_trusted"
                checkouts = [s for s in job["steps"] if s.get("with", {}).get("path") == "_trusted"]
                assert len(checkouts) == 1
                checkout = checkouts[0]
                assert checkout["uses"] == "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
                assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha || github.sha }}"
                assert checkout["with"]["persist-credentials"] is False
                assert checkout["with"]["fetch-depth"] == 0
                head_checkout = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")
                                     and s.get("with", {}).get("path") != "_trusted")
                assert job["steps"].index(head_checkout) < job["steps"].index(checkout)
    assert count >= 5


def test_actions_are_pinned_before_first_push():
    """Every action is pinned to a full commit; an all-zero placeholder fails."""
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    uses = __import__("re").findall(r"uses:\s*(\S+)", workflow)
    assert uses
    for action in uses:
        owner, sha = action.rsplit("@", 1)
        assert owner in ("actions/checkout", "actions/setup-python")
        assert __import__("re").fullmatch("[0-9a-f]{40}", sha)
        assert sha != "0" * 40, "ACTION_SHA_PLACEHOLDER: owner must pin verified action commits before first push"
    assert "ACTION_SHA_PLACEHOLDER" not in workflow


def test_workflow_contract():
    # PyYAML is available in the supplied venv; runtime tools do not require it.
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    assert jobs["result"]["if"] == "always()"
    assert set(jobs["result"]["needs"]) == set(jobs) - {"result"}
    runs = {job: [s.get("run") for s in jobs[job]["steps"]] for job in ("test-linux", "test-windows")}
    assert 'python scripts/dev.py test --ci -j 4 -m "not local"' in runs["test-windows"]
    assert {"python scripts/fetch_ffmpeg.py", "python scripts/fetch_msst_models.py"} <= set(runs["test-windows"])
    assert "python -B -m pytest -q -p no:cacheprovider --noconftest tests/gates" in runs["test-linux"]
    raw = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "secrets." not in raw and "pull_request_target" not in raw
    assert "persist-credentials: true" not in raw


def test_dev_ci_entry_reuses_runner_and_requires_nonlocal(monkeypatch, tmp_path):
    from scripts import dev
    captured = {}
    def runner(args, paths, selection, directory, reservation):
        captured.update(paths=paths, markers=args.markers, child_env=reservation.child_env())
        return 0
    monkeypatch.setattr(dev, "_run_tests_reserved", runner)
    assert dev.main(["test", "--ci", "-m", "not local"]) == 0
    assert "tests/gates/test_public_guard.py" in captured["paths"]
    assert captured["markers"] == "not local"
    assert captured["child_env"] == {}
    with pytest.raises(SystemExit):
        dev.main(["test", "--ci", "-m", "local"])


def test_rest_pagination_and_thread_truncation_fail_closed():
    api = ci.GitHub("synthetic/repo", "synthetic")
    calls = []
    def request(route, payload=None):
        calls.append(route)
        return [None] * 100 if route.endswith("&page=1") else [None]
    api.request = request
    assert len(api.pages("pulls/1/reviews")) == 101
    assert len(calls) == 2
    def truncated(route, payload):
        return {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": [{"isResolved": True, "comments": {"nodes": [], "pageInfo": {"hasNextPage": True}}}],
            "pageInfo": {"hasNextPage": False}}}}}}
    api.request = truncated
    with pytest.raises(ValueError, match="truncated"):
        api.threads(1)


def test_collector_transforms_raw_api_json(review, monkeypatch):
    pr, policy, _ = review
    receipt = {**pr["reviews"][0]["receipt"], "review_id": 123}
    current = {"head": {"sha": pr["head_sha"]}, "base": {"sha": pr["base_sha"]},
               "draft": False, "body": "## Implementation engines\nclaude\n"}
    raw_review = {"id": 123, "user": {"login": policy["bot"]["login"], "node_id": policy["bot"]["id"], "type": "Bot"},
                  "performed_via_github_app": {"id": 123}, "state": "COMMENTED",
                  "body": "Review complete: no findings", "commit_id": pr["head_sha"]}
    class FakeAPI:
        repo = "synthetic/repo"
        def request(self, route):
            return current
        def pages(self, route):
            return ([{"author_association": "OWNER", "body": "```lss-review-receipt\n" + json.dumps(receipt) + "\n```"}]
                    if "comments" in route else [raw_review])
        def threads(self, number):
            return []
    monkeypatch.setattr(ci, "git", lambda *args: (ROOT / change_class.POLICY).read_bytes())
    monkeypatch.setattr(ci, "diff_state", lambda *args: {"class": "A"})
    expected = {k: pr[k] for k in ("head_sha", "base_sha", "diff_digest")}
    actual = ci.collect_review(FakeAPI(), 1, expected)
    assert not verify(actual, policy)
    raw_review["performed_via_github_app"] = None
    assert verify(ci.collect_review(FakeAPI(), 1, expected), policy)
    current["base"]["sha"] = "f" * 40
    with pytest.raises(ValueError, match="changed"):
        ci.collect_review(FakeAPI(), 1, expected)


@pytest.mark.parametrize("mode,implementation,reviewer", [
    ("bot", "codex", "claude"),
    ("local-records", "codex", "claude"),
    ("local-records", "claude", "codex"),
])
def test_wrapper_record_collects_and_passes_review_gate(review, monkeypatch, mode, implementation, reviewer):
    from gates import lss_gh
    pr, policy, _ = review
    policy["mode"] = mode
    pr["implementation_engines"] = [implementation]
    receipt = {**pr["reviews"][0]["receipt"], "review_id": 123}
    record = {**pr["reviews"][0]["receipt"], "engine": reviewer, "complete": True,
              "summary": "Independent synthetic review", "findings_p0p1": 0, "unresolved_p0p1": 0}
    posted = lss_gh.format_review_record(record, pr["head_sha"])
    class API:
        repo = "synthetic/repo"
        association = "OWNER"
        def request(self, route):
            return {"head": {"sha": pr["head_sha"]}, "base": {"sha": pr["base_sha"]},
                    "draft": False, "body": "## Implementation engines\n" + implementation + "\n"}
        def pages(self, route):
            if "comments" in route:
                return [{"author_association": self.association, "body": posted},
                        {"author_association": self.association,
                         "body": "```lss-review-receipt\n" + json.dumps(receipt) + "\n```"}]
            if mode == "local-records":
                return []
            return [{"id": 123, "user": {"login": policy["bot"]["login"], "node_id": policy["bot"]["id"], "type": "Bot"},
                     "performed_via_github_app": {"id": 123}, "state": "COMMENTED",
                     "body": "Review complete: no findings", "commit_id": pr["head_sha"]}]
        def threads(self, number):
            return []
    monkeypatch.setattr(ci, "git", lambda *args: (ROOT / change_class.POLICY).read_bytes())
    monkeypatch.setattr(ci, "diff_state", lambda *args: {"class": "A"})
    expected = {k: pr[k] for k in ("head_sha", "base_sha", "diff_digest")}
    collected = ci.collect_review(API(), 1, expected)
    assert collected["independent_records"] == [record]
    assert verify(collected, policy) == []
    # Anyone may comment on a public pull request; only the owner and collaborators can post a record.
    API.association = "NONE"
    outsider = ci.collect_review(API(), 1, expected)
    assert outsider["independent_records"] == []
    if mode == "local-records":
        assert verify(outsider, policy)


def test_whatsnext_runs_as_an_isolated_script(tmp_path):
    import subprocess, sys
    proc = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/whatsnext.py")],
                          cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert proc.returncode in (0, 1), proc.stderr
    assert "Traceback" not in proc.stderr


# Hashes identify retired fixture strings without republishing their contents.
@pytest.mark.parametrize("filename,retired",
[('test_bgm_scene_name_spellings.py',
  [(4, 'd8996ca2aee833d98588c40d7dde25ff029ee70737e2cb514be4b291f0abe5fc'),
   (4, '7fccd1bd5f570b6082334129dd266c689af3ac14678ed4d98216f699017abf3e'),
   (2, 'fbc0af51e769bfc77a07069624a5f7296b3837e0aa67ca11310ed06d96fd5ea7'),
   (2, '484f767b0d09e8d0d07e45d55b984eaadcd2edb79ddb6997f1d6c6f1e107ef1d')]),
 ('test_bgm_scene_names.py',
  [(4, '6f4ef3da4ed14d7be53f76b4bad2d4b803a2a2e038535437670662b0713cbeae')]),
 ('test_bgm_work_credits.py',
  [(4, '83138968c1e265ca746a44393d86f9b584a29da40892d65407a98df32afad133'),
   (4, '066367da5daadf0a14b576ce56eb9d50822a8053dfbbd6e5f2d94325cd740f4c')]),
 ('test_metadata_musicbrainz.py',
  [(3, '8461cc8db59d671094ec02b30648f809c9d8702095220ffa9d497e6568420df6')]),
 ('test_metadata_resolve.py',
  [(5, '2518f2e59c90cb071b254e1e0b25c22fc8d35e57050c788d4e90468b26d6cc8b'),
   (21, '177fa5acd264631b10c22ae35ad0ca8e95d5871deddc7ac4feefe05795dc710e'),
   (13, 'de3e6dcc0462b2e3c49cea10cc6384a6236db83309d0641a6784f65ca77483c6'),
   (13, 'c36338afb2c0b6864f8f450882adae32d56a2a3334befca18586849f0434fb05')]),
 ('test_metadata_series_album.py',
  [(10, '2b50bde5d6bb9db2975e89bb8e70ba9884483c1253587d51cda25ec9eac52ddb'),
   (7, 'a99252febc5d1aeba6fdf27705416b2bbaee1f64a3036ac35cc10d4131a40e4f')])])
def test_public_fixtures_retire_reviewed_identifiers(filename, retired):
    import hashlib
    text = (ROOT / "tests" / filename).read_text(encoding="utf-8").casefold()
    for length, digest in retired:
        assert not any(hashlib.sha256(text[i:i + length].encode()).hexdigest() == digest
                       for i in range(len(text) - length + 1)), filename


def test_fingerprint_fixture_keeps_near_minimum_positive():
    import ast
    tree = ast.parse((ROOT / "tests/test_reference_discovery.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "test_fingerprint_only_verification_needs_votes_windows_and_a_steady_alignment")
    positives = [n.test.args[0].args[0].value for n in function.body
                 if isinstance(n, ast.Assert) and isinstance(n.test, ast.Call)
                 and isinstance(n.test.args[0], ast.Call)]
    assert any(40 <= votes <= 47 for votes in positives)
