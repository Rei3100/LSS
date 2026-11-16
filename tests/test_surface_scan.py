"""Synthetic public surfaces; no GitHub or private state is accessed."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest


spec = importlib.util.spec_from_file_location(
    "surface_scan", Path(__file__).resolve().parents[1] / "gates" / "surface_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

REPO = "example/synthetic"
ROOT = "repos/" + REPO
WORD = "synthetic-forbidden-marker"
SINCE = "2020-01-02T00:00:00Z"
UPDATED = "2020-01-03T00:00:00Z"


def surfaces():
    return {
        "": {"default_branch": "main"},
        "/issues": [{"number": 1, "title": "Issue", "body": "Details"},
                    {"number": 2, "pull_request": {}, "title": "Pull", "body": "Details"}],
        "/pulls": [{"number": 2, "title": "Pull", "body": "Details", "updated_at": UPDATED,
                    "head": {"ref": "topic"}, "base": {"ref": "main"}}],
        "/issues/comments": [{"id": 3, "issue_url": "https://api.github.com/" + ROOT + "/issues/1",
                              "body": "Comment"}],
        "/pulls/comments": [{"id": 4, "pull_request_url": "https://api.github.com/" + ROOT + "/pulls/2",
                             "body": "Review comment"}],
        "/commits": [{"sha": "a" * 40, "commit": {"message": "Commit"}}],
        "/branches": [{"name": "main"}],
        "/secret-scanning/alerts": [],
    }


class FakeRunner:
    def __init__(self, data=None, failures=None):
        self.data = deepcopy(surfaces() if data is None else data)
        self.failures = failures or {}
        self.calls = []

    def __call__(self, argv):
        # Any mutation, host change or non-gh command fails the test.
        assert argv[:8] == ["gh", "api", "--hostname", "github.com", "--method", "GET",
                            "--include", "-H"]
        assert argv[8] == "Accept: application/vnd.github+json"
        endpoint = urlsplit(argv[9])
        assert endpoint.path.startswith(ROOT)
        suffix = endpoint.path[len(ROOT):]
        query = parse_qs(endpoint.query)
        self.calls.append((suffix, query))
        if suffix in self.failures:
            failure = self.failures[suffix]
            if isinstance(failure, Exception):
                raise failure
            return SimpleNamespace(returncode=1, stdout=f"HTTP/2.0 {failure} Error\r\n\r\n"
                                   + json.dumps({"message": WORD}), stderr=WORD)
        rows = self.data[suffix]
        if suffix:
            assert query["per_page"] == ["100"]
            start = (int(query["page"][0]) - 1) * 100
            rows = rows[start:start + 100]
        return SimpleNamespace(returncode=0, stdout="HTTP/2.0 200 OK\r\nContent-Type: application/json"
                               + "\r\n\r\n" + json.dumps(rows), stderr="")


@pytest.fixture
def invoke(tmp_path, capsys):
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(WORD + "\n", encoding="utf-8")
    out = tmp_path / "results"

    def call(fake=None, *, extra=(), explicit_denylist=True):
        argv = ["--repo", REPO, "--since", SINCE, "--out-dir", str(out)]
        if explicit_denylist:
            argv += ["--denylist", str(denylist)]
        before = set(out.glob("*.json"))
        code = scan.main(argv + list(extra), runner=fake or FakeRunner())
        files = set(out.glob("*.json")) - before
        assert len(files) == 1
        path = files.pop()
        raw = path.read_text(encoding="utf-8")
        captured = capsys.readouterr()
        assert WORD.casefold() not in (raw + captured.out + captured.err).casefold()
        return code, json.loads(raw), path

    call.denylist = denylist
    call.out = out
    return call


def test_clean_run(invoke):
    fake = FakeRunner()
    code, result, path = invoke(fake)
    assert code == 0
    assert result["status"] == "clean"
    assert result["alerts"] == "known"
    assert result["repo"] == REPO
    assert result["window"]["since"] == SINCE
    assert scan.instant(result["window"]["until"]) <= datetime.now(timezone.utc)
    assert result["counts"] == {"issue_title": 1, "issue_body": 1, "pull_title": 1,
                                "pull_body": 1, "issue_comment": 1, "review_comment": 1,
                                "commit_message": 1, "branch_name": 4, "secret_alert": 0}
    assert result["findings"] == []
    assert path.name.endswith("Z.json")
    calls = dict(fake.calls)
    assert calls["/issues"]["state"] == ["all"]
    assert calls["/pulls"]["state"] == ["all"]
    for suffix in ("/issues", "/issues/comments", "/pulls/comments", "/commits"):
        assert calls[suffix]["since"] == [SINCE]
    assert calls["/commits"]["sha"] == ["main"]
    assert calls["/secret-scanning/alerts"]["state"] == ["open"]


@pytest.mark.parametrize("suffix,field,kind", [
    ("/issues", "title", "issue_title"), ("/issues", "body", "issue_body"),
    ("/pulls", "title", "pull_title"), ("/pulls", "body", "pull_body"),
    ("/issues/comments", "body", "issue_comment"),
    ("/pulls/comments", "body", "review_comment"),
    ("/commits", "message", "commit_message"), ("/branches", "name", "branch_name"),
    ("/pulls", "head", "branch_name"), ("/pulls", "base", "branch_name"),
    ("", "default_branch", "branch_name"),
])
def test_each_text_surface(invoke, suffix, field, kind):
    fake = FakeRunner()
    item = fake.data[suffix][0] if suffix else fake.data[suffix]
    value = "before " + WORD.upper() + " after"
    if field == "message":
        item["commit"][field] = value
    elif field in ("head", "base"):
        item[field]["ref"] = value
    else:
        item[field] = value
    code, result, _ = invoke(fake)
    assert code == 1
    assert result["status"] == "findings"
    assert result["findings"] == [{"kind": kind, "url": result["findings"][0]["url"],
                                    "entry_index": 0}]
    assert result["findings"][0]["url"].startswith("https://github.com/" + REPO)


@pytest.mark.parametrize("suffix", ["/issues", "/pulls", "/issues/comments", "/pulls/comments",
                                    "/commits", "/branches", "/secret-scanning/alerts"])
def test_pagination_reaches_later_findings(invoke, suffix):
    fake = FakeRunner()
    if suffix == "/secret-scanning/alerts":
        fake.data[suffix] = [{"state": "open", "number": n} for n in range(1, 102)]
    else:
        original = deepcopy(fake.data[suffix][0])
        fake.data[suffix] = [deepcopy(original) for _ in range(101)]
        last = fake.data[suffix][-1]
        if suffix == "/commits":
            last["commit"]["message"] = WORD
        else:
            last["name" if suffix == "/branches" else "body"] = WORD
    code, result, _ = invoke(fake)
    assert code == 1
    assert result["status"] == "findings"
    assert [query["page"] for path, query in fake.calls if path == suffix] == [["1"], ["2"]]
    assert len(result["findings"]) == (101 if suffix.endswith("alerts") else 1)


def test_old_pulls_stop_pagination(invoke):
    fake = FakeRunner()
    old = deepcopy(fake.data["/pulls"][0])
    old.update(updated_at="2020-01-01T00:00:00Z", body=WORD)
    fake.data["/pulls"] += [old] * 100
    code, result, _ = invoke(fake)
    assert code == 0
    assert result["counts"]["pull_body"] == 1
    assert len([path for path, _ in fake.calls if path == "/pulls"]) == 1


def test_not_published(invoke):
    fake = FakeRunner(failures={"": 404})
    code, result, _ = invoke(fake)
    assert code == 0
    assert result["status"] == "not_published"
    assert result["alerts"] == "unknown"
    assert len(fake.calls) == 1


def test_empty_repository_still_scans_the_other_surfaces(invoke):
    data = surfaces()
    data["/issues"][0]["body"] = WORD
    code, result, _ = invoke(FakeRunner(data, failures={"/commits": 409}))
    assert code == 1
    assert result["status"] == "findings"
    assert result["counts"]["commit_message"] == 0
    assert [item["kind"] for item in result["findings"]] == ["issue_body"]


def test_conflict_on_other_surfaces_is_unknown(invoke):
    code, result, _ = invoke(FakeRunner(failures={"/issues": 409}))
    assert code != 0
    assert result["status"] == "unknown"


@pytest.mark.parametrize("failure", [401, 403, 500, OSError(WORD),
                                     subprocess.TimeoutExpired("gh", 120, output=WORD),
                                     subprocess.CalledProcessError(1, "gh", output=WORD, stderr=WORD)])
def test_gh_failure_is_unknown(invoke, failure):
    code, result, _ = invoke(FakeRunner(failures={"": failure}))
    assert code != 0
    assert result["status"] == "unknown"


def test_later_failure_preserves_findings(invoke):
    fake = FakeRunner(failures={"/commits": 500})
    fake.data["/issues"][0]["body"] = WORD
    code, result, _ = invoke(fake)
    assert code == 1
    assert result["status"] == "unknown"
    assert result["findings"][0]["entry_index"] == 0


@pytest.mark.parametrize("failure", [403, 404, 500, OSError(WORD)])
def test_alerts_unavailable_are_unknown(invoke, failure):
    code, result, _ = invoke(FakeRunner(failures={"/secret-scanning/alerts": failure}))
    assert code != 0
    assert result["status"] == "unknown"
    assert result["alerts"] == "unknown"
    assert result["counts"]["commit_message"] == 1


def test_open_alert_is_a_finding_without_persisting_secret(invoke):
    fake = FakeRunner()
    fake.data["/secret-scanning/alerts"] = [{"number": 7, "state": "open", "secret": WORD,
                                             "html_url": "https://example.invalid/" + WORD}]
    code, result, _ = invoke(fake)
    assert code == 1
    assert result["alerts"] == "known"
    assert result["findings"] == [{"kind": "secret_alert", "entry_index": None,
                                    "url": "https://github.com/" + REPO + "/security/secret-scanning/7"}]


def test_missing_denylist_fails_before_network(invoke):
    fake = FakeRunner()
    invoke.denylist.unlink()
    code, result, _ = invoke(fake)
    assert code != 0
    assert result["status"] == "unknown"
    assert not fake.calls


@pytest.mark.parametrize("content", ["", "# only a comment\n", "word:\n", "\x00"])
def test_invalid_denylist_fails(invoke, content):
    invoke.denylist.write_text(content, encoding="utf-8")
    fake = FakeRunner()
    code, result, _ = invoke(fake)
    assert code != 0 and result["status"] == "unknown"
    assert not fake.calls


def test_unreadable_denylist_fails(invoke, monkeypatch):
    def denied(_):
        raise PermissionError(WORD)
    monkeypatch.setattr(scan.guard, "load_denylist", denied)
    code, result, _ = invoke()
    assert code != 0 and result["status"] == "unknown"


def test_denylist_environment_resolution_and_override(invoke, monkeypatch, tmp_path):
    monkeypatch.setenv("LSS_PUBLISH_DENYLIST", str(invoke.denylist))
    assert invoke(explicit_denylist=False)[0] == 0
    monkeypatch.delenv("LSS_PUBLISH_DENYLIST")
    # The guard's default is resolved at import time, just like public_guard.
    monkeypatch.setattr(scan.guard, "DEFAULT_DENYLIST", str(invoke.denylist))
    assert invoke(explicit_denylist=False)[0] == 0
    monkeypatch.setenv("LSS_PUBLISH_DENYLIST", str(tmp_path / "missing.txt"))
    assert invoke()[0] == 0  # Explicit path wins over the environment.


@pytest.mark.parametrize("value,indices", [("xxsyntheticunitxx", []), ("SYNTHETICUNIT!", [1]),
                                            ("prefix " + WORD.upper(), [0]),
                                            ("\x00架空照合語\x00", [2])])
def test_guard_content_matching_is_reused(invoke, value, indices):
    invoke.denylist.write_text(WORD + "\nword:syntheticunit\n架空照合語\n", encoding="utf-8")
    fake = FakeRunner()
    fake.data["/issues"][0]["body"] = value
    _, result, path = invoke(fake)
    assert [item["entry_index"] for item in result["findings"]] == indices
    raw = path.read_text(encoding="utf-8")
    assert "syntheticunit" not in raw.casefold()
    assert "架空照合語" not in raw


def test_untrusted_urls_are_never_copied(invoke):
    fake = FakeRunner()
    for rows in fake.data.values():
        if isinstance(rows, list):
            for row in rows:
                row["html_url"] = "https://example.invalid/" + WORD
    fake.data["/issues"][0]["body"] = WORD
    assert invoke(fake)[0] == 1


def test_repo_and_locators_are_masked_if_they_match(invoke):
    invoke.denylist.write_text("synthetic\n" + WORD, encoding="utf-8")
    fake = FakeRunner()
    fake.data["/issues"][0]["body"] = WORD
    _, result, _ = invoke(fake)
    assert result["repo"] == "<masked>"
    assert result["findings"][0]["url"] == "<masked>"


@pytest.mark.parametrize("suffix,value", [("/issues", {}), ("/commits", [{"sha": WORD}]),
                                          ("/pulls", [{"updated_at": WORD}]),
                                          ("/pulls", [{"updated_at": None}]),
                                          ("/issues/comments", [{"issue_url": None}]),
                                          ("/branches", [{"name": None}])])
def test_malformed_data_is_unknown(invoke, suffix, value):
    fake = FakeRunner()
    fake.data[suffix] = value
    code, result, _ = invoke(fake)
    assert code != 0 and result["status"] == "unknown"


def test_malformed_http_or_json_is_unknown(invoke):
    for stdout in (WORD, "HTTP/2.0 200 OK\n\n" + WORD):
        def runner(_):
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=WORD)
        code, result, _ = invoke(runner)
        assert code != 0 and result["status"] == "unknown"


def test_default_window_and_output_directory(tmp_path, monkeypatch, capsys):
    denylist = tmp_path / "denylist.txt"
    denylist.write_text(WORD, encoding="utf-8")
    monkeypatch.setenv("LSS_PRIVATE_ROOT", str(tmp_path))
    fake = FakeRunner()
    fake.data["/pulls"] = []
    before = datetime.now(timezone.utc)
    code = scan.main(["--repo", REPO, "--denylist", str(denylist)], runner=fake)
    result_file, = (tmp_path / "run" / "scan").glob("*.json")
    raw = result_file.read_text(encoding="utf-8")
    result = json.loads(raw)
    assert code == 0
    assert before - timedelta(days=3) <= scan.instant(result["window"]["since"])
    assert scan.instant(result["window"]["until"]) - scan.instant(result["window"]["since"]) == timedelta(days=3)
    captured = capsys.readouterr()
    assert WORD not in raw + captured.out + captured.err


def test_output_write_failure_is_nonzero(invoke):
    invoke.out.write_text("blocked", encoding="utf-8")
    assert scan.main(["--repo", REPO, "--denylist", str(invoke.denylist),
                      "--out-dir", str(invoke.out)], runner=FakeRunner()) != 0
