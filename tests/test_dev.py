"""検証入口が明示実行境界と未検証の状態を取り違えないことを確認する。"""
from pathlib import Path

import pytest

from scripts import dev


def test_normal_suites_do_not_enable_real_acceptance(monkeypatch, tmp_path):
    monkeypatch.setenv("LSS_RUN_GPU_TEST", "1")
    monkeypatch.setenv("LSS_RUN_UNKNOWN_FUTURE_TEST", "1")
    monkeypatch.setenv("LSS_UI_ACCEPTANCE_DOWNLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PSModulePath", "C:/PowerShell7/Modules")
    environment = dev.test_environment("local", tmp_path)
    assert not any(key.startswith("LSS_RUN_") for key in environment)
    assert "LSS_UI_ACCEPTANCE_DOWNLOAD" not in environment
    assert "PYTEST_ADDOPTS" not in environment
    assert not any(key.upper() == "PSMODULEPATH" for key in environment)
    assert environment["LSS_STATE_DIR"] == str(tmp_path)
    for suite in ("local", "backend", "bgm"):
        selected = dev.selected_tests(suite, [])
        assert "tests/test_ui_lifecycle.py" not in selected
        assert not {f"tests/{value[0]}" for value in dev.ACCEPTANCE.values()} & set(selected)


@pytest.mark.parametrize("suite", dev.ACCEPTANCE)
def test_acceptance_selects_only_its_own_switch_and_file(suite, tmp_path):
    file, switch = dev.ACCEPTANCE[suite]
    assert dev.selected_tests(suite, []) == [f"tests/{file}"]
    assert (dev.ROOT / "tests" / file).is_file()
    assert switch in (dev.ROOT / "tests" / file).read_text(encoding="utf-8")
    environment = dev.test_environment(suite, tmp_path)
    assert {k: v for k, v in environment.items() if k.startswith("LSS_RUN_")} == {switch: "1"}


@pytest.mark.parametrize("path", ["tests/test_separation_gpu.py", "../other.py", "tests", "-s"])
def test_local_selection_cannot_escape_the_suite(path):
    with pytest.raises(ValueError, match="対象外"):
        dev.selected_tests("local", [path])


def test_specific_node_and_backend_boundary():
    node = "tests/test_worker.py::test_handle_job_rejects_multichannel_before_loading_model"
    assert dev.selected_tests("backend", [node]) == [node]
    assert all(not Path(p).name.startswith("test_ui_") for p in dev.selected_tests("backend", []))
    assert "tests/test_bgm_variant_receipt.py" in dev.selected_tests("bgm", [])


def test_real_login_window_requires_the_native_suite():
    node = dev.NATIVE["login-runtime"]
    assert dev.selected_tests("login-runtime", []) == [node]
    with pytest.raises(ValueError, match="対象外"):
        dev.selected_tests("local", [node])
    with pytest.raises(ValueError, match="対象外"):
        dev.selected_tests("login-runtime", ["tests/test_worker.py"])


def test_real_runtime_suites_never_run_in_parallel(monkeypatch):
    monkeypatch.setattr(dev.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(dev, "available_commit_bytes", lambda: 64 * 2 ** 30)
    files = ["tests/test_worker.py", "tests/test_session.py"]
    for suite in (*dev.NATIVE, *dev.ACCEPTANCE):
        assert dev.worker_count(suite, files, "auto", False) == 1
        assert dev.worker_count(suite, files, "6", False) == 1
    assert dev.worker_count("local", files, "auto", True) == 1
    assert dev.worker_count("local", files[:1], "auto", False) == 1
    assert dev.worker_count("local", files, "1", False) == 1
    assert dev.worker_count("local", files, "3", False) == 3


def test_parallel_workers_leave_cpu_for_other_work(monkeypatch):
    files = ["tests/test_worker.py", "tests/test_session.py"]
    monkeypatch.setattr(dev.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(dev, "available_commit_bytes", lambda: 64 * 2 ** 30)
    monkeypatch.setattr(dev.os, "cpu_count", lambda: 16)
    assert dev.worker_count("local", files, "auto", False) == 8
    monkeypatch.setattr(dev.os, "cpu_count", lambda: 64)
    assert dev.worker_count("local", files, "auto", False) == 8
    monkeypatch.setattr(dev.os, "cpu_count", lambda: 2)
    assert dev.worker_count("local", files, "auto", False) == 1
    monkeypatch.setattr(dev.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(dev, "available_commit_bytes", lambda: 10 * 2 ** 30)
    assert dev.worker_count("local", files, "auto", False) == 3
    monkeypatch.setattr(dev.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(dev, "available_commit_bytes", lambda: 64 * 2 ** 30)
    assert dev.worker_count("local", files, "auto", False) == 1


def test_available_commit_is_measured():
    assert 0 < dev.available_commit_bytes()


@pytest.mark.parametrize("code,counts,changed,collect,expected", [
    (0, {"tests": 3, "skipped": 0}, False, False, "PASSED"),
    (0, {"tests": 3, "skipped": 1}, False, False, "PASSED_WITH_SKIPS"),
    (0, {"tests": 3, "skipped": 3}, False, False, "NOT_VERIFIED"),
    (0, {}, False, False, "NOT_VERIFIED"),
    (0, {"tests": 3, "skipped": 0}, True, False, "WORKTREE_CHANGED"),
    (1, {"tests": 3, "skipped": 0}, False, False, "FAILED"),
    (0, {"tests": 3, "skipped": 0, "errors": 1}, False, False, "FAILED"),
    (0, {}, False, True, "COLLECTED_ONLY"),
    (2, {}, False, True, "FAILED"),
])
def test_exit_zero_is_not_by_itself_validation(code, counts, changed, collect, expected):
    assert dev.verdict(code, counts, changed, collect) == expected


def test_junit_counts_include_collection_and_teardown_errors(tmp_path):
    xml = tmp_path / "result.xml"
    xml.write_text('<testsuites><testsuite>'
                   '<testcase name="pass"/>'
                   '<testcase name="skip"><skipped/></testcase>'
                   '<testcase name="fail"><failure/></testcase>'
                   '<testcase name="teardown"><error/></testcase>'
                   '<error message="collection"/>'
                   '</testsuite></testsuites>', encoding="utf-8")
    assert dev.junit_counts(xml) == {"tests": 4, "skipped": 1, "failures": 1, "errors": 2}


def test_not_local_runs_do_not_name_files_that_cannot_be_imported_from_the_public_tree():
    listed = "tests/test_githooks.py"
    paths = [listed, listed + "::test_x", "tests/test_dev.py"]
    assert dev.collectable_paths(paths, "not local") == ["tests/test_dev.py"]
    assert dev.collectable_paths(paths, "not  local") == ["tests/test_dev.py"]
    assert dev.collectable_paths(paths, None) == paths
    assert dev.collectable_paths(paths, "local") == paths
