"""`dev.py test --changed` selection and the failure digest stay deterministic and conservative."""
from pathlib import Path

import pytest

from scripts import dev

SOURCES = {
    "lss/__init__.py": "",
    "lss/core.py": "def f():\n    return 1\n",
    "lss/pkg/__init__.py": "",
    "lss/pkg/a.py": "from lss.core import f\n",
    "lss/pkg/b.py": "from . import a\n",
    "lss/pkg/c.py": "from .a import f\n",
    "lss/lazy.py": "def g():\n    import lss.pkg.b\n",
    "lss/doc.py": '"""Mentions policy.json only in the docstring."""\n',
    "scripts/tool.py": "import lss.core\n",
    "scripts/hooks/rule.py": "x = 1\n",
    "tests/__init__.py": "",
    "tests/helper.py": "from lss.pkg import b\n",
    "tests/test_core.py": "from lss import core\n",
    "tests/test_deep.py": "from tests.helper import something\n",
    "tests/test_attr.py": "from lss.pkg.b import thing\n",
    "tests/test_script.py": "from scripts import tool\n",
    "tests/test_bare.py": "import tool\n",
    "tests/test_policy.py": "P = ROOT / 'scripts' / 'storage_policy.json'\n",
    "tests/test_ps1.py": "S = 'scripts\\\\dev_hygiene.ps1'\n",
    "tests/test_dotted.py": "import importlib\nm = importlib.import_module('scripts.hooks.rule')\n",
    "tests/test_patch.py": "def t(monkeypatch):\n    monkeypatch.setattr('lss.lazy.g', None)\n",
    "tests/test_unrelated.py": "import json\n",
    "tests/test_scan.py": "R = Path(__file__).parents[1] / 'lss'\nfiles = list(R.rglob('*.py'))\n",
    "tests/test_scan_tmp.py": "def t(tmp_path):\n    list(tmp_path.glob('*.flac'))\n",
    "tests/test_doc.py": "from lss import doc\n",
    "tests/test_broken.py": "def (:\n",
}
TESTS = sorted(p for p in SOURCES if p.startswith("tests/test_"))


def test_selection_skips_source_reads_only_when_the_graph_cannot_change_the_answer(monkeypatch):
    monkeypatch.setattr(dev, 'selected_tests', lambda *args: TESTS)
    monkeypatch.setattr(dev, 'read_sources', lambda *args: (_ for _ in ()).throw(AssertionError('unneeded source read')))
    monkeypatch.setattr(dev, 'changed_paths', lambda base: [])
    assert dev.select_changed('local', 'explicit-base')['selected'] == []
    monkeypatch.setattr(dev, 'changed_paths', lambda base: ['scripts/dev.py'])
    assert dev.select_changed('local', 'explicit-base')['selected'] == TESTS


def test_document_path_dependencies_still_read_the_source_graph(monkeypatch):
    monkeypatch.setattr(dev, 'selected_tests', lambda *args: ['tests/test_doc_link.py'])
    monkeypatch.setattr(dev, 'changed_paths', lambda base: ['docs/owner.md'])
    monkeypatch.setattr(dev, 'read_sources', lambda *args: {'tests/test_doc_link.py': "p = 'docs/owner.md'"})
    assert dev.select_changed('local', 'explicit-base')['selected'] == ['tests/test_doc_link.py']


def selected(*changed: str) -> set[str]:
    return set(dev.select_related(list(changed), SOURCES, TESTS))


def test_module_names():
    assert dev.module_name("lss/a/b.py") == "lss.a.b"
    assert dev.module_name("lss/a/__init__.py") == "lss.a"
    assert dev.package_of("lss/a/b.py") == "lss.a"


def test_transitive_imports_and_import_forms():
    assert selected("lss/core.py") == {
        "tests/test_core.py", "tests/test_deep.py", "tests/test_attr.py", "tests/test_script.py",
        "tests/test_bare.py", "tests/test_scan.py"}


def test_relative_and_function_level_imports_are_followed():
    # b imports a (relative), lazy imports b inside a function, helper imports b through the package.
    assert selected("lss/pkg/a.py") == {"tests/test_attr.py", "tests/test_deep.py", "tests/test_scan.py"}
    assert "lss/pkg/c.py" not in dev.select_related(["lss/pkg/b.py"], SOURCES, list(SOURCES))


def test_package_init_change_reaches_everything_below_it():
    assert {"tests/test_attr.py", "tests/test_deep.py"} <= selected("lss/pkg/__init__.py")
    assert "tests/test_unrelated.py" not in selected("lss/pkg/__init__.py")


def test_changed_test_file_selects_itself_and_unrelated_stays_out():
    assert selected("tests/test_unrelated.py") == {"tests/test_unrelated.py"}
    assert "tests/test_unrelated.py" not in selected("lss/core.py")


def test_non_python_file_is_found_by_basename_string():
    assert selected("scripts/storage_policy.json") == {"tests/test_policy.py"}
    assert selected("scripts/dev_hygiene.ps1") == {"tests/test_ps1.py"}
    assert selected("docs/unreferenced.md") == set()


def test_dotted_module_string_and_patch_target():
    assert selected("scripts/hooks/rule.py") == {"tests/test_dotted.py"}
    assert "tests/test_patch.py" in selected("lss/lazy.py")


def test_tree_scanning_test_depends_on_the_directory_it_names():
    assert "tests/test_scan.py" in selected("lss/brand_new.py")
    assert "tests/test_scan.py" not in selected("scripts/tool.py")
    assert "tests/test_scan_tmp.py" not in selected("lss/brand_new.py")
    assert dev.select_related(["lss/new.py"], SOURCES, TESTS)["tests/test_scan.py"] == "scans lss/"


def test_docstrings_do_not_count_as_references():
    assert selected("policy.json") == set()


def test_deleted_module_still_selects_its_importers():
    assert "tests/test_script.py" in dev.select_related(["scripts/gone.py"], {**SOURCES, "tests/test_script.py": "from scripts import gone\n"}, TESTS)


def test_unparseable_file_does_not_break_selection():
    assert selected("tests/test_broken.py") == {"tests/test_broken.py"}


@pytest.mark.parametrize("path", ["tests/conftest.py", "pytest.ini", "requirements.txt", "requirements-dev.txt", "constraints.txt",
                                  "scripts/dev.py"])
def test_environment_files_select_the_whole_suite(path):
    assert selected(path) == set(TESTS)


def test_selection_is_limited_to_the_given_suite_files():
    result = dev.select_related(["lss/core.py"], SOURCES, ["tests/test_core.py"])
    assert result == {"tests/test_core.py": "imports lss/core.py"}


def test_reasons_name_the_changed_file():
    reasons = dev.select_related(["lss/core.py"], SOURCES, TESTS)
    assert reasons["tests/test_deep.py"] == "imports lss/core.py"
    assert dev.select_related(["scripts/storage_policy.json"], SOURCES, TESTS)["tests/test_policy.py"] == \
        "references scripts/storage_policy.json"


def test_describe_selection_lists_top_causes():
    selection = {"base": "abcdef1234567", "changed": ["lss/core.py"], "suite_files": 20,
                 "selected": ["a", "b", "c"], "reasons": {"a": "imports lss/core.py", "b": "imports lss/core.py",
                                                         "c": "changed"}}
    lines = dev.describe_selection(selection)
    assert lines[0] == "changed since abcdef1234: 1 files; selected 3 of 20 test files"
    assert lines[1].strip() == "2 via lss/core.py"
    assert "(test file itself changed)" in lines[2]


def test_real_repository_selection_is_consistent():
    suite = dev.selected_tests("local", [])
    result = dev.select_related(["scripts/dev.py"], {}, suite)
    assert set(result) == set(suite)
    sources = dev.read_sources()
    assert "tests/test_dev.py" in sources
    related = dev.select_related(["lss/__init__.py"], sources, suite)
    assert "tests/test_worker.py" in related  # every lss module runs the package __init__


JUNIT = """<testsuites><testsuite tests="6">
<testcase classname="tests.test_dev" name="test_available_commit_is_measured"/>
<testcase classname="tests.test_dev" name="test_skip"><skipped message="x"/></testcase>
<testcase classname="tests.test_dev" name="test_assert[a-1]"><failure message="assert 1 == 2">tests/test_dev.py:5: in test_assert
    assert f() == 2
E   assert 1 == 2
E    +  where 1 = f()</failure></testcase>
<testcase classname="tests.test_dev.TestGroup" name="test_method"><failure message="AssertionError: boom">tests/test_dev.py:9: in test_method
E   AssertionError
E   boom is the real message</failure></testcase>
<testcase classname="tests.test_dev" name="test_teardown"><error message="failed on teardown with &quot;OSError: disk&quot;">no e lines here
OSError: disk</error></testcase>
<testcase classname="" name="tests.test_broken_collect"><error message="collection failure">E   ModuleNotFoundError: x
</error></testcase>
</testsuite></testsuites>"""


def test_failure_digest_names_each_failure_with_its_first_error_line(tmp_path):
    xml = tmp_path / "result.xml"
    xml.write_text(JUNIT, encoding="utf-8")
    digest = dev.failure_digest(xml, root=dev.ROOT)
    assert digest["total"] == 4 and digest["omitted"] == 0
    assert digest["lines"] == [
        "FAILURE tests/test_dev.py::test_assert[a-1] :: assert 1 == 2",
        "FAILURE tests/test_dev.py::TestGroup::test_method :: AssertionError: boom is the real message",
        "ERROR tests/test_dev.py::test_teardown :: failed on teardown with \"OSError: disk\"",
        "ERROR tests.test_broken_collect :: ModuleNotFoundError: x",
    ]


def test_failure_digest_caps_lines_and_truncates_long_errors(tmp_path):
    cases = "".join(f'<testcase classname="tests.test_dev" name="t{i}"><failure>E   {"x" * 500}</failure></testcase>'
                    for i in range(40))
    xml = tmp_path / "many.xml"
    xml.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>", encoding="utf-8")
    digest = dev.failure_digest(xml)
    assert len(digest["lines"]) == dev.DIGEST_LINES and digest["omitted"] == 10 and digest["total"] == 40
    assert all(len(line.split(" :: ", 1)[1]) <= dev.DIGEST_WIDTH for line in digest["lines"])
    assert digest["lines"][0].endswith("...")


def test_digest_text_has_verdict_counts_seconds_and_log(tmp_path):
    xml = tmp_path / "result.xml"
    xml.write_text(JUNIT, encoding="utf-8")
    digest = dev.digest_report("FAILED", dev.junit_counts(xml), 12.34, xml, tmp_path / "pytest.log", tmp_path / "report.json")
    text = dev.format_digest(digest)
    assert text.splitlines()[0] == "verdict: FAILED / tests 6 failed 2 errors 2 skipped 1 / 12.3s"
    assert f"log: {tmp_path / 'pytest.log'}" in text
    assert len(text.splitlines()) == 1 + 4 + 2


def test_digest_without_junit_shows_the_log_tail(tmp_path):
    log = tmp_path / "pytest.log"
    log.write_text("\n".join(f"line {i}" for i in range(30)), encoding="utf-8")
    digest = dev.digest_report("FAILED", {}, 1.0, tmp_path / "missing.xml", log, tmp_path / "report.json")
    assert digest["log_tail"][-1] == "line 29" and len(digest["log_tail"]) == 10


def test_run_pytest_keeps_everything_in_the_log_and_the_console_terse(tmp_path, capsys):
    script = tmp_path / "noisy.py"
    script.write_text("print('..F.  [ 50%]')\nprint('long traceback detail')\nprint('....  [100%]')\n", encoding="utf-8")
    log = tmp_path / "pytest.log"
    assert dev.run_pytest([dev.sys.executable, str(script)], {}, log, stream=False) == 0
    shown = capsys.readouterr().out
    assert "long traceback detail" not in shown and "50% (1 failed/error so far)" in shown
    assert "long traceback detail" in log.read_text(encoding="utf-8")
    assert dev.run_pytest([dev.sys.executable, str(script)], {}, log, stream=True) == 0
    assert "long traceback detail" in capsys.readouterr().out


def _no_match_repo(monkeypatch, changed):
    monkeypatch.setenv("LSS_STORAGE_RESERVATION", "supervised")  # selection is under test, not the storage supervisor re-exec
    monkeypatch.setattr(dev, "changed_paths", lambda base: changed)
    monkeypatch.setattr(dev, "read_sources", lambda root=dev.ROOT: SOURCES)
    monkeypatch.setattr(dev, "selected_tests", lambda suite, paths, root=dev.ROOT: TESTS)


@pytest.mark.parametrize('supervised',[True,False])
def test_changed_selection_is_computed_once_per_process_and_empty_run_never_reserves(monkeypatch,supervised):
    _no_match_repo(monkeypatch,['docs/unreferenced.md'])
    if not supervised:
        monkeypatch.delenv('LSS_STORAGE_RESERVATION',raising=False)
    calls=[];real=dev.select_changed
    def select(*args,**kwargs):
        calls.append(args);return real(*args,**kwargs)
    monkeypatch.setattr(dev,'select_changed',select)
    import storage_budget
    monkeypatch.setattr(storage_budget,'command',lambda *a,**k:(_ for _ in ()).throw(AssertionError('empty run must not reserve')))
    assert dev.main(['test','--changed'])==dev.NOT_RUN
    assert len(calls)==1


@pytest.mark.parametrize("changed", [[], ["docs/unreferenced.md"]])
def test_changed_with_no_affected_test_is_not_run_not_success(monkeypatch, capsys, changed):
    _no_match_repo(monkeypatch, changed)
    monkeypatch.setattr(dev, "default_base", lambda: "abc123def456")
    assert dev.main(["test", "--changed"]) == dev.NOT_RUN == 3
    out = capsys.readouterr().out
    assert "NOT_RUN" in out and "fell back" not in out
    assert dev.main(["test", "--changed-list"]) == 0


def test_head_fallback_base_is_stated(monkeypatch, capsys):
    _no_match_repo(monkeypatch, [])
    monkeypatch.setattr(dev, "default_base", lambda: "HEAD")
    assert dev.main(["test", "--changed"]) == dev.NOT_RUN
    assert "fell back to HEAD" in capsys.readouterr().out


def test_default_base_falls_back_to_head_when_no_main_resolves(monkeypatch):
    monkeypatch.setattr(dev.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 128, "stdout": ""})())
    assert dev.default_base() == "HEAD"


def test_shared_python_uses_real_git_common_root_and_keeps_local_priority(tmp_path):
    import subprocess
    main = tmp_path / "shared foundation"
    lane = tmp_path / "work lane"
    main.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(main), *args], check=True, capture_output=True)
    git("init")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "--allow-empty", "-m", "fixture")
    git("worktree", "add", "--detach", str(lane), "HEAD")
    assert dev.repository_python(lane) is None
    shared = main / ".venv/Scripts/python.exe"
    # Reuse the real foundation across drives, as actual linked worktrees do.
    import os
    if os.name == "nt":
        shared.parent.parent.mkdir(parents=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(shared.parent),
                        str(Path(dev.sys.executable).parent)], check=True, capture_output=True)
    else:
        shared.parent.mkdir(parents=True)
        shared.symlink_to(dev.sys.executable)
    assert dev.repository_python(lane).samefile(shared)
    scripts = lane / "scripts"
    scripts.mkdir()
    (scripts / "dev.py").write_bytes(Path(dev.__file__).read_bytes())
    result = subprocess.run([dev.sys.executable, str(scripts / "dev.py"), "doctor"],
                            cwd=lane, capture_output=True, text=True, encoding="utf-8")
    import json
    observed = json.loads(result.stdout)
    assert result.returncode == 0, result.stderr
    assert observed["repository_python"] is True
    assert observed["paths"][".venv/Scripts/python.exe"] is False
    assert Path(observed["expected_python"]).samefile(shared)
    assert observed["git"]["head"] == git("rev-parse", "HEAD").stdout.decode().strip()
    local = lane / ".venv/Scripts/python.exe"
    local.parent.mkdir(parents=True)
    local.touch()
    assert dev.repository_python(lane).samefile(local)


def test_doctor_accepts_shared_foundation_without_claiming_local_venv(monkeypatch, tmp_path, capsys):
    import json
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    monkeypatch.setattr(dev, "repository_python", lambda root: Path(dev.sys.executable))
    monkeypatch.setattr(dev, "identity", lambda: {})
    monkeypatch.setattr(dev, "storage", lambda: {})
    assert dev.doctor() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["repository_python"] is True
    assert result["paths"][".venv/Scripts/python.exe"] is False
    assert result["expected_python"] == dev.sys.executable


def test_missing_git_foundation_is_not_a_valid_python(tmp_path):
    assert dev.repository_python(tmp_path) is None
